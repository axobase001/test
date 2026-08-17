from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from main_sequence import final_recent_replay as base
from main_sequence import v5_hourly_symmetric_stop as v5
from main_sequence import v5_tail_only_fast as fast


def score_fullanchor_shard(start: str, end: str, shard: int, nshards: int, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = v5.discover(start, end, workers=min(16, max(4, workers * 2)))
    markets = sorted(markets, key=lambda m: m.start)
    selected = [m for i, m in enumerate(markets) if i % nshards == shard]
    selected_ids = {m.condition_id for m in selected}
    inv = inv.copy()
    inv["selected_shard"] = inv["condition_id"].astype(str).isin(selected_ids)
    inv.to_csv(out / "inventory.csv", index=False)

    # Critical: every shard builds anchors and 1s spot over the SAME full [start,end) window.
    # Only the independent Polymarket market-tape work is partitioned.
    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")

    rows = []
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(fast.candidate_rows_for_market, m, spot, bn, der): m for m in selected}
        for i, f in enumerate(as_completed(fut), 1):
            m = fut[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                failed.append((m, repr(exc)))
            if i % 24 == 0 or i == len(selected):
                print("FULLANCHOR_SHARD", shard, i, "/", len(selected), "candidates", len(rows), "fail", len(failed), flush=True)

    # Preserve fail-closed semantics, but serially recover transient Data API failures.
    pending = failed
    retry_history = []
    for round_no, sleep_s in enumerate((5, 10, 20, 40), 1):
        if not pending:
            break
        time.sleep(sleep_s)
        nxt = []
        for m, err in pending:
            try:
                rows.extend(fast.candidate_rows_for_market(m, spot, bn, der))
                retry_history.append({"slug": m.slug, "round": round_no, "recovered": True, "prior_error": err})
            except Exception as exc:
                nxt.append((m, repr(exc)))
                retry_history.append({"slug": m.slug, "round": round_no, "recovered": False, "error": repr(exc)})
        pending = nxt

    if pending:
        (out / "failures.json").write_text(json.dumps([
            {"slug": m.slug, "condition_id": m.condition_id, "error": err} for m, err in pending
        ], indent=2), encoding="utf-8")
        raise RuntimeError(f"unresolved shard failures={len(pending)}")

    df = pd.DataFrame(rows, columns=fast.CANDIDATE_COLS)
    if len(df):
        df = df.drop_duplicates(["condition_id", "sec", "outcome"], keep="last").sort_values(["start", "sec", "outcome"], kind="mergesort")
    df.to_csv(out / "tail_candidates.csv", index=False)
    (out / "retry_history.json").write_text(json.dumps(retry_history, indent=2), encoding="utf-8")
    summary = {
        "period": [start, end], "shard": shard, "nshards": nshards,
        "full_window_markets_mapped": int(inv["mapped"].fillna(False).astype(bool).sum()),
        "selected_markets": len(selected), "candidate_rows": len(df),
        "anchor_meta": anchor_meta,
        "invariant": "anchors and Binance 1s are full-window; only Polymarket market tape is sharded",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--shard", type=int, required=True); ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    score_fullanchor_shard(a.start, a.end, a.shard, a.nshards, a.out, a.workers)

if __name__ == "__main__":
    main()
