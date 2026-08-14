from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

OB_REPO = "obadiaha/polymarket-crypto-5m-15m"
OB_REV = "11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"
KR_REPO = "krish301/polymarket-crypto-trades-v1"
KR_REV = "f3bfb79f19611ea3115ad29c18720166ee6dd0b5"
WINDOW = "14-03-2026/00-00-00-UTC"
WINDOW_START = 1773446400
SLUG = f"btc-updown-15m-{WINDOW_START}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dl(repo: str, rev: str, name: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=rev, filename=name))


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "main-sequence-audit/1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def frame_probe(path: Path, name: str) -> dict:
    pf = pq.ParquetFile(path)
    df = pf.read_row_group(0).to_pandas().head(5000)
    out = {
        "name": name,
        "sha256": sha256(path),
        "rows": int(pf.metadata.num_rows),
        "columns": pf.schema_arrow.names,
        "head": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
    for col in ["asset", "outcome", "side", "market_id", "condition_id", "token_id", "question", "timeframe", "duration", "outcomes"]:
        if col in df.columns:
            vc = df[col].dropna().astype(str).value_counts().head(20)
            out[f"values_{col}"] = {str(k): int(v) for k, v in vc.items()}
    return out


def json_probe(path: Path, name: str) -> dict:
    obj = json.loads(path.read_text())
    return {
        "name": name,
        "sha256": sha256(path),
        "type": type(obj).__name__,
        "top_keys": list(obj.keys()) if isinstance(obj, dict) else None,
        "value": obj,
    }


def fills_probe(path: Path) -> dict:
    rows = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            rows.append(json.loads(line))
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    assets = {}
    for k in ("makerAssetId", "takerAssetId"):
        vals = pd.Series([str(r.get(k)) for r in rows if r.get(k) is not None]).value_counts().head(20)
        assets[k] = {str(a): int(n) for a, n in vals.items()}
    return {
        "name": "krish_fills",
        "sha256": sha256(path),
        "sample_rows": len(rows),
        "keys": keys,
        "asset_ids": assets,
        "head": rows[:5],
    }


def rows_for_slug(path: Path, slug: str) -> dict:
    df = pd.read_parquet(path)
    key = "market_id" if "market_id" in df.columns else ("slug" if "slug" in df.columns else None)
    if key is None:
        return {"status": "no_market_key", "columns": list(df.columns)}
    x = df[df[key].astype(str) == slug].copy()
    return {
        "key": key,
        "rows": int(len(x)),
        "columns": list(df.columns),
        "records": json.loads(x.head(100).to_json(orient="records", date_format="iso")),
    }


def main() -> None:
    out = Path("mapping_audit"); out.mkdir(exist_ok=True)
    report = {"slug": SLUG, "obadiaha": {}, "krish": {}}

    ob_paths = {}
    for name, rel in {
        "markets": "markets/all.parquet",
        "resolutions": "resolutions/all.parquet",
        "book": "orderbooks/2026-03-14.parquet",
    }.items():
        p = dl(OB_REPO, OB_REV, rel); ob_paths[name] = p
        report["obadiaha"][name] = frame_probe(p, name)
        report["obadiaha"][f"{name}_exact_slug"] = rows_for_slug(p, SLUG)

    exact_market = report["obadiaha"]["markets_exact_slug"].get("records", [])
    if exact_market:
        condition_id = str(exact_market[0]["condition_id"])
        report["official_clob"] = {
            "condition_id": condition_id,
            "endpoint": f"https://clob.polymarket.com/clob-markets/{condition_id}",
        }
        try:
            info = http_json(report["official_clob"]["endpoint"])
            report["official_clob"]["response"] = info
            report["official_clob"]["token_outcomes"] = {
                str(x.get("t")): str(x.get("o")) for x in info.get("t", []) if x.get("t") is not None
            }
        except Exception as exc:
            report["official_clob"]["error"] = repr(exc)

    api = HfApi()
    files = api.list_repo_files(repo_id=KR_REPO, repo_type="dataset", revision=KR_REV)
    special = [p for p in files if ("metadata" in p.lower() or "manifest" in p.lower() or "_index" in p.lower())]
    report["krish"]["file_count"] = len(files)
    report["krish"]["special_paths"] = special[:1000]
    report["krish"]["metadata_paths_present"] = any("metadata" in p.lower() for p in files)

    kr_fill_rel = f"data/raw/trades/btc/15m/{WINDOW}/fills.jsonl.gz"
    report["krish"]["fill_path_present"] = kr_fill_rel in files
    if kr_fill_rel in files:
        report["krish"]["fills"] = fills_probe(dl(KR_REPO, KR_REV, kr_fill_rel))

    json_candidates = [p for p in special if p.endswith(".json") and ("btc/15m" in p or "trades_by_window" in p)]
    for rel in json_candidates[:5]:
        try:
            report["krish"].setdefault("json_probes", {})[rel] = json_probe(dl(KR_REPO, KR_REV, rel), rel)
        except Exception as exc:
            report["krish"].setdefault("json_probe_errors", {})[rel] = repr(exc)

    book_exact = report["obadiaha"]["book_exact_slug"]["records"]
    book_tokens = sorted({str(r.get("token_id")) for r in book_exact if r.get("token_id") is not None})
    fill_assets = set()
    for side in ("makerAssetId", "takerAssetId"):
        fill_assets.update(report["krish"].get("fills", {}).get("asset_ids", {}).get(side, {}).keys())
    fill_assets.discard("0")
    report["cross_check"] = {
        "book_token_ids": book_tokens,
        "sample_fill_non_usdc_asset_ids": sorted(fill_assets),
        "book_tokens_seen_in_fill_sample": sorted(set(book_tokens) & fill_assets),
        "official_token_outcomes_for_book_tokens": {
            t: report.get("official_clob", {}).get("token_outcomes", {}).get(t) for t in book_tokens
        },
    }

    (out / "mapping_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
