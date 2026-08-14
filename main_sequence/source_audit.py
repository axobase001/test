from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

KEYWORDS = ("btc", "bitcoin", "15m", "15min", "1h", "60m", "orderbook", "book", "snapshot", "trade", "fill")
TS_HINTS = ("ts", "time", "timestamp", "datetime", "date", "created", "received", "exchange")
CAT_HINTS = ("event_type", "side", "outcome", "asset", "timeframe", "duration", "market_type")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relevant(path: str) -> bool:
    p = path.lower()
    return any(k in p for k in KEYWORDS)


def choose_probe(files: list[dict], max_bytes: int) -> dict | None:
    usable = [x for x in files if 0 < (x.get("size") or 0) <= max_bytes]
    usable = [x for x in usable if x["path"].lower().endswith((".parquet", ".csv", ".csv.gz")) and relevant(x["path"])]
    if not usable:
        return None
    def score(x):
        p = x["path"].lower()
        return (
            0 if ("btc" in p or "bitcoin" in p) else 1,
            0 if ("15m" in p or "15min" in p) else 1,
            0 if ("snapshot" in p or "orderbook" in p or "book" in p) else 1,
            x.get("size") or 10**30,
        )
    return sorted(usable, key=score)[0]


def timestamp_summary(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0, "min": None, "max": None}
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return {"count": 0, "min": None, "max": None}
        med = float(vals.abs().median())
        unit = "s"
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        dt = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce").dropna()
        if len(dt):
            return {"count": int(len(vals)), "min": dt.min().isoformat(), "max": dt.max().isoformat(), "numeric_unit_guess": unit}
        return {"count": int(len(vals)), "min": float(vals.min()), "max": float(vals.max())}
    dt = pd.to_datetime(s, utc=True, errors="coerce").dropna()
    if len(dt):
        return {"count": int(len(s)), "min": dt.min().isoformat(), "max": dt.max().isoformat()}
    return {"count": int(len(s)), "min": str(s.min()), "max": str(s.max())}


def categorical_probe(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        lc = str(c).lower()
        if any(h == lc or h in lc for h in CAT_HINTS):
            try:
                vc = df[c].dropna().astype(str).value_counts().head(20)
                if len(vc):
                    out[str(c)] = {str(k): int(v) for k, v in vc.items()}
            except Exception:
                pass
    return out


def probe_file(repo: str, revision: str, fmeta: dict, cache: Path) -> dict:
    local = Path(hf_hub_download(repo_id=repo, filename=fmeta["path"], repo_type="dataset", revision=revision, cache_dir=cache))
    out = {"path": fmeta["path"], "bytes": local.stat().st_size, "sha256": sha256_file(local)}
    low = local.name.lower()
    if low.endswith(".parquet"):
        pf = pq.ParquetFile(local)
        cols = pf.schema_arrow.names
        out.update({"format": "parquet", "rows": int(pf.metadata.num_rows), "columns": cols})
        tscols = [c for c in cols if any(h in c.lower() for h in TS_HINTS)]
        if tscols:
            col = tscols[0]
            s = pd.read_parquet(local, columns=[col])[col]
            out["timestamp_probe"] = {"column": col, **timestamp_summary(s)}
        catcols = [c for c in cols if any(h == c.lower() or h in c.lower() for h in CAT_HINTS)]
        if catcols:
            sample = pf.read_row_group(0, columns=catcols).to_pandas().head(200_000)
            cats = categorical_probe(sample)
            if cats:
                out["categorical_probe_first_row_group"] = cats
    elif low.endswith((".csv", ".csv.gz")):
        out["format"] = "csv"
        first = pd.read_csv(local, nrows=200_000)
        out["columns"] = list(first.columns)
        cats = categorical_probe(first)
        if cats:
            out["categorical_probe_first_200k"] = cats
        tscols = [c for c in first.columns if any(h in str(c).lower() for h in TS_HINTS)]
        if tscols:
            col = tscols[0]
            count = 0; mins = []; maxs = []
            for chunk in pd.read_csv(local, usecols=[col], chunksize=500_000):
                count += len(chunk)
                s = chunk[col].dropna()
                if len(s):
                    if pd.api.types.is_numeric_dtype(s):
                        mins.append(pd.to_numeric(s, errors="coerce").min()); maxs.append(pd.to_numeric(s, errors="coerce").max())
                    else:
                        d = pd.to_datetime(s, utc=True, errors="coerce").dropna()
                        if len(d):
                            mins.append(d.min()); maxs.append(d.max())
            if mins:
                if isinstance(mins[0], pd.Timestamp):
                    tinfo = {"count": int(count), "min": min(mins).isoformat(), "max": max(maxs).isoformat()}
                else:
                    tinfo = timestamp_summary(pd.Series([min(mins), max(maxs)]))
                    tinfo["count"] = int(count)
                out["timestamp_probe"] = {"column": col, **tinfo, "rows_scanned": int(count)}
    return out


def inventory(api: HfApi, repo: str, revision: str | None) -> tuple[dict, list[dict]]:
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    rows = [{"path": s.rfilename, "size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None)} for s in info.siblings]
    rel = [x for x in rows if relevant(x["path"])]
    inv = {
        "repo_id": repo,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "revision_exact": bool(revision and info.sha == revision),
        "files": len(rows),
        "total_bytes": int(sum((x["size"] or 0) for x in rows)),
        "file_paths": [x["path"] for x in rows[:500]],
        "relevant_files": sorted(rel, key=lambda x: (x.get("size") or 0, x["path"]))[:200],
    }
    return inv, rows


def find_meta(files: list[dict], path: str) -> dict | None:
    return next((x for x in files if x["path"] == path), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--probe-max-mb", type=int, default=220)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    api = HfApi(); report = {"sources": {}, "promotion_rule": manifest["policy"]["strict_backtest"]}
    max_bytes = args.probe_max_mb * 1024 * 1024

    for name, src in manifest["sources"].items():
        repo = src.get("repo_id")
        if not repo:
            report["sources"][name] = {"status": "non_hf_source", "classification": src.get("classification"), "revision": src.get("revision")}
            continue
        revision = src.get("revision")
        if not revision and src.get("discover_revision_only"):
            try:
                inv, _ = inventory(api, repo, None)
                inv["status"] = "discovered_unpinned_do_not_use"
                inv["classification"] = src.get("classification")
                report["sources"][name] = inv
                print(f"DISCOVER {name} {repo} HEAD={inv['resolved_revision']} DO_NOT_USE_UNTIL_PINNED", flush=True)
            except Exception as exc:
                report["sources"][name] = {"status": "discovery_error", "repo_id": repo, "error": repr(exc)}
            continue
        if not revision:
            report["sources"][name] = {"status": "not_used_unpinned", "repo_id": repo, "classification": src.get("classification")}
            continue

        print(f"AUDIT {name} {repo}@{revision}", flush=True)
        try:
            inv, files = inventory(api, repo, revision)
            if not inv["revision_exact"]:
                raise RuntimeError(f"revision mismatch requested={revision} resolved={inv['resolved_revision']}")
            probes = []
            requested = src.get("probe_paths") or []
            if requested:
                for path in requested:
                    meta = find_meta(files, path)
                    if meta is None:
                        probes.append({"path": path, "status": "missing_at_revision"})
                        continue
                    if (meta.get("size") or 0) > max_bytes:
                        probes.append({"path": path, "status": "present_too_large_for_probe", "bytes": meta.get("size"), "blob_id": meta.get("blob_id")})
                        continue
                    print(f"PROBE {repo} {path} size={meta.get('size')}", flush=True)
                    probes.append(probe_file(repo, revision, meta, args.cache))
            else:
                meta = choose_probe(files, max_bytes)
                if meta:
                    probes.append(probe_file(repo, revision, meta, args.cache))
            inv["probes"] = probes
            inv["status"] = "pinned_audited"
            inv["classification"] = src.get("classification")
            forbidden = set(src.get("forbidden_feature_columns") or [])
            if forbidden:
                seen = set()
                for p in probes:
                    seen.update(p.get("columns") or [])
                inv["forbidden_feature_columns_present"] = sorted(forbidden & seen)
            report["sources"][name] = inv
        except Exception as exc:
            report["sources"][name] = {"status": "audit_error", "repo_id": repo, "requested_revision": revision, "error": repr(exc)}

    (args.out / "source_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
