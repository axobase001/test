from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "trentmkelly/polymarket_crypto_derivatives"


def summarize(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    names = pf.schema.names
    out = {"path": str(path.name), "rows": int(pf.metadata.num_rows), "columns": names}
    wanted = [x for x in ["ts", "timestamp", "timestamp_ms", "progress", "chainlink_price", "binance_price",
                          "up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask",
                          "best_bid", "best_ask", "side", "price", "size", "token_id", "outcome"] if x in names]
    if wanted:
        df = pd.read_parquet(path, columns=wanted)
        sample = {}
        for c in wanted:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce").dropna()
                sample[c] = {
                    "min": float(x.min()) if len(x) else None,
                    "max": float(x.max()) if len(x) else None,
                    "first": float(x.iloc[0]) if len(x) else None,
                    "last": float(x.iloc[-1]) if len(x) else None,
                }
            else:
                sample[c] = {"examples": s.dropna().astype(str).head(5).tolist()}
        out["summary"] = sample
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--day", default="2026-03-14")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(REPO)
    sha = str(info.sha)
    files = api.list_repo_files(REPO, repo_type="dataset", revision=sha)
    # Directory names encode the window start date. Prefer BTC 15m episodes on requested day.
    pat = re.compile(r"(^|/)btc15m_.*" + re.escape(args.day) + r".*?/(steps|events|book_levels)\.parquet$", re.I)
    candidates = [p for p in files if pat.search(p)]
    if not candidates:
        # Preserve evidence if layout naming differs; list nearby btc15m files rather than guessing.
        candidates = [p for p in files if "btc15m" in p.lower() and args.day in p][:60]
    groups = {}
    for p in candidates:
        parent = str(Path(p).parent)
        groups.setdefault(parent, []).append(p)
    selected_parent = None
    for parent, ps in sorted(groups.items()):
        names = {Path(p).name for p in ps}
        if {"steps.parquet", "events.parquet", "book_levels.parquet"}.issubset(names):
            selected_parent = parent; break
    if selected_parent is None and groups:
        selected_parent = sorted(groups)[0]
    selected = sorted(groups.get(selected_parent, []))

    report = {
        "repo": REPO,
        "resolved_immutable_revision": sha,
        "repo_file_count": len(files),
        "requested_day": args.day,
        "candidate_count": len(candidates),
        "selected_episode": selected_parent,
        "selected_files": selected,
        "btc15m_first_files": [p for p in files if "btc15m" in p.lower()][:20],
        "btc15m_last_files": [p for p in files if "btc15m" in p.lower()][-20:],
        "tables": [],
    }
    for rel in selected:
        if Path(rel).name not in {"steps.parquet", "events.parquet", "book_levels.parquet"}:
            continue
        local = Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=sha, cache_dir=args.cache))
        report["tables"].append({"repo_path": rel, **summarize(local)})
    (args.out / "trent_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
