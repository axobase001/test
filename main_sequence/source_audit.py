from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

KEYWORDS = ("btc", "bitcoin", "15m", "15min", "1h", "60m", "orderbook", "book", "snapshot", "trade")
TS_HINTS = ("ts", "time", "timestamp", "datetime", "date", "created", "received", "exchange")


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
    usable = [x for x in files if (x.get("size") or 0) > 0 and (x.get("size") or 0) <= max_bytes]
    usable = [x for x in usable if x["path"].lower().endswith((".parquet", ".csv", ".csv.gz")) and relevant(x["path"])]
    if not usable:
        return None
    # Prefer BTC 15m snapshots/orderbooks, then smallest useful structured file.
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
        if med > 1e17: unit = "ns"
        elif med > 1e14: unit = "us"
        elif med > 1e11: unit = "ms"
        try:
            dt = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce").dropna()
            return {"count": int(len(vals)), "min": dt.min().isoformat(), "max": dt.max().isoformat(), "numeric_unit_guess": unit}
        except Exception:
            return {"count": int(len(vals)), "min": float(vals.min()), "max": float(vals.max())}
    dt = pd.to_datetime(s, utc=True, errors="coerce").dropna()
    if len(dt):
        return {"count": int(len(s)), "min": dt.min().isoformat(), "max": dt.max().isoformat()}
    return {"count": int(len(s)), "min": str(s.min()), "max": str(s.max())}


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
    elif low.endswith((".csv", ".csv.gz")):
        out["format"] = "csv"
        first = pd.read_csv(local, nrows=20)
        out["columns"] = list(first.columns)
        tscols = [c for c in first.columns if any(h in str(c).lower() for h in TS_HINTS)]
        if tscols:
            col = tscols[0]
            mn = mx = None; count = 0; samples = []
            for chunk in pd.read_csv(local, usecols=[col], chunksize=500_000):
                count += len(chunk)
                samples.append(chunk[col])
                if len(samples) >= 4:
                    merged = pd.concat(samples, ignore_index=True)
                    samples = [merged]
            merged = pd.concat(samples, ignore_index=True) if samples else pd.Series(dtype=float)
            # Full-range min/max can be computed without retaining every other column.
            # If repeated concatenation above was collapsed, all timestamp values remain present.
            out["timestamp_probe"] = {"column": col, **timestamp_summary(merged), "rows_scanned": int(count)}
    return out


def inventory(api: HfApi, repo: str, revision: str | None) -> dict:
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    rows = [{"path": s.rfilename, "size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None)} for s in info.siblings]
    rel = [x for x in rows if relevant(x["path"])]
    return {
        "repo_id": repo,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "revision_exact": bool(revision and info.sha == revision),
        "files": len(rows),
        "total_bytes": int(sum((x["size"] or 0) for x in rows)),
        "relevant_files": sorted(rel, key=lambda x: (x.get("size") or 0, x["path"]))[:200],
        "_all": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--probe-max-mb", type=int, default=220)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    api = HfApi(); report = {"sources": {}, "promotion_rule": manifest["policy"]["strict_backtest"]}

    for name, src in manifest["sources"].items():
        repo = src.get("repo_id")
        if not repo or not src.get("revision"):
            report["sources"][name] = {"status": "not_used_unpinned", "repo_id": repo, "classification": src.get("classification")}
            continue
        print(f"AUDIT {name} {repo}@{src['revision']}", flush=True)
        try:
            inv = inventory(api, repo, src["revision"])
            all_files = inv.pop("_all")
            if not inv["revision_exact"]:
                raise RuntimeError(f"revision mismatch requested={src['revision']} resolved={inv['resolved_revision']}")
            probe = choose_probe(all_files, args.probe_max_mb * 1024 * 1024)
            if probe:
                print(f"PROBE {repo} {probe['path']} size={probe['size']}", flush=True)
                inv["probe"] = probe_file(repo, src["revision"], probe, args.cache)
            inv["status"] = "pinned_audited"
            inv["classification"] = src.get("classification")
            report["sources"][name] = inv
        except Exception as exc:
            report["sources"][name] = {"status": "audit_error", "repo_id": repo, "requested_revision": src.get("revision"), "error": repr(exc)}

    # OpenMarket sample is allowed only for schema discovery. Resolve HEAD -> SHA first,
    # then download by that exact SHA in the same run; it never contributes reported PnL.
    sample_repo = manifest["sources"]["openmarket_unified"]["sample_repo_id"]
    try:
        sample_inv = inventory(api, sample_repo, None); all_files = sample_inv.pop("_all")
        resolved = sample_inv["resolved_revision"]
        probe = choose_probe(all_files, args.probe_max_mb * 1024 * 1024)
        if probe:
            sample_inv["probe"] = probe_file(sample_repo, resolved, probe, args.cache)
        sample_inv["status"] = "schema_discovery_only_resolved_then_pinned_in_run"
        report["openmarket_sample"] = sample_inv
    except Exception as exc:
        report["openmarket_sample"] = {"status": "audit_error", "repo_id": sample_repo, "error": repr(exc)}

    (args.out / "source_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
