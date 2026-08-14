from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OB_REPO = "obadiaha/polymarket-crypto-5m-15m"
OB_REV = "11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"
KR_REPO = "krish301/polymarket-crypto-trades-v1"
KR_REV = "f3bfb79f19611ea3115ad29c18720166ee6dd0b5"
WINDOW = "14-03-2026/00-00-00-UTC"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dl(repo: str, rev: str, name: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=rev, filename=name))


def frame_probe(path: Path, name: str) -> dict:
    pf = pq.ParquetFile(path)
    df = pf.read_row_group(0).to_pandas().head(5000)
    out = {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(pf.metadata.num_rows),
        "columns": pf.schema_arrow.names,
        "head": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
    }
    for col in ["asset", "outcome", "side", "market_id", "condition_id", "token_id", "question", "timeframe", "duration"]:
        if col in df.columns:
            vc = df[col].dropna().astype(str).value_counts().head(20)
            out[f"values_{col}"] = {str(k): int(v) for k, v in vc.items()}
    return out


def json_probe(path: Path, name: str) -> dict:
    obj = json.loads(path.read_text())
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "type": type(obj).__name__,
        "top_keys": list(obj.keys()) if isinstance(obj, dict) else None,
        "value": obj,
    }


def fills_probe(path: Path) -> dict:
    rows = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i >= 1000:
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


def main() -> None:
    out = Path("mapping_audit"); out.mkdir(exist_ok=True)
    report = {"obadiaha": {}, "krish": {}}
    for name, rel in {
        "markets": "markets/all.parquet",
        "resolutions": "resolutions/all.parquet",
        "book": "orderbooks/2026-03-14.parquet",
    }.items():
        p = dl(OB_REPO, OB_REV, rel)
        report["obadiaha"][name] = frame_probe(p, name)

    kr_meta_rel = f"data/raw/metadata/btc/15m/{WINDOW}/metadata.json"
    kr_fill_rel = f"data/raw/trades/btc/15m/{WINDOW}/fills.jsonl.gz"
    kr_manifest_rel = "data/raw/trades/btc/15m/_manifest.json"
    report["krish"]["metadata"] = json_probe(dl(KR_REPO, KR_REV, kr_meta_rel), "krish_metadata")
    report["krish"]["manifest"] = json_probe(dl(KR_REPO, KR_REV, kr_manifest_rel), "krish_manifest")
    report["krish"]["fills"] = fills_probe(dl(KR_REPO, KR_REV, kr_fill_rel))

    # Cross-map the audited midnight BTC15m window without using resolution outcome.
    meta = report["krish"]["metadata"]["value"]
    slug = meta.get("slug") or meta.get("market", {}).get("slug")
    markets_path = dl(OB_REPO, OB_REV, "markets/all.parquet")
    m = pd.read_parquet(markets_path)
    if slug and "market_id" in m.columns:
        rows = m[m["market_id"].astype(str) == str(slug)].copy()
        report["cross_mapping"] = json.loads(rows.head(20).to_json(orient="records", date_format="iso"))
    else:
        report["cross_mapping"] = {"status": "slug_or_market_id_missing", "slug": slug}

    (out / "mapping_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
