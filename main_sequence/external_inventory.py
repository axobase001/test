from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

SOURCES = [
    "Alezanello/polymarket-arena-capture",
    "kachoio/polymarket-5-minute-crypto-up-down-markets",
    "obadiaha/polymarket-crypto-5m-15m",
    "trentmkelly/polymarket_crypto_derivatives",
]


def main():
    out = Path("external_inventory")
    out.mkdir(exist_ok=True)
    api = HfApi()
    result = {}
    for repo in SOURCES:
        print(f"INVENTORY {repo}", flush=True)
        info = api.dataset_info(repo, files_metadata=True)
        rows = []
        for s in info.siblings:
            rows.append({"path": s.rfilename, "size": getattr(s, "size", None), "blob_id": getattr(s, "blob_id", None)})
        result[repo] = {"sha": info.sha, "files": rows}
        print(f"sha={info.sha} files={len(rows)} total_size={sum((x['size'] or 0) for x in rows)}", flush=True)
        for x in rows[:30]: print(x, flush=True)

    # Arena: materialize a recent small daily file from each table and print exact schema/sample.
    arena = result[SOURCES[0]]["files"]
    for key in ("cap_book", "cap_trades", "cap_prices"):
        cand = [x for x in arena if f"daily/{key}/" in x["path"] and x["path"].endswith(".parquet")]
        if not cand:
            cand = [x for x in arena if x["path"].endswith(f"{key}.parquet")]
        if not cand:
            continue
        cand = sorted(cand, key=lambda x: x["path"])
        x = cand[-1]
        p = hf_hub_download(SOURCES[0], x["path"], repo_type="dataset")
        df = pd.read_parquet(p)
        print(f"ARENA_SAMPLE {key} path={x['path']} rows={len(df)} cols={list(df.columns)}", flush=True)
        print(df.head(5).to_json(orient="records"), flush=True)
        if "slug" in df.columns:
            print("slug samples", df["slug"].dropna().astype(str).head(20).tolist(), flush=True)
        if "meta" in df.columns:
            vals = df["meta"].dropna().astype(str).head(50).tolist()
            keys = Counter()
            parsed=[]
            for v in vals:
                try:
                    obj=json.loads(v); parsed.append(obj)
                    if isinstance(obj,dict): keys.update(obj.keys())
                except Exception: pass
            print("meta keys", keys.most_common(), flush=True)
            print("meta parsed samples", json.dumps(parsed[:3], default=str), flush=True)

    (out/"inventory.json").write_text(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
