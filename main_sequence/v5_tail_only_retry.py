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


def score_range(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = v5.discover(start, end, workers=min(12, max(4, workers)))
    inv.to_csv(out / "inventory.csv", index=False)
    if not markets:
        pd.DataFrame(columns=fast.CANDIDATE_COLS).to_csv(out / "tail_candidates.csv", index=False)
        summary = {"period": [start, end], "markets_expected": int(len(inv)), "markets_mapped": 0, "candidate_rows": 0, "unresolved_failures": 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")

    rows = []
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(fast.candidate_rows_for_market, m, spot, bn, der): m for m in markets}
        for i, f in enumerate(as_completed(fut), 1):
            m = fut[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                failed.append((m, repr(exc)))
            if i % 48 == 0:
                print("TAIL_RETRY_PAR", i, "/", len(markets), "candidates", len(rows), "fail", len(failed), flush=True)

    retry_history = []
    pending = failed
    for round_no, sleep_s in enumerate((5, 10, 20, 40), 1):
        if not pending:
            break
        print("TAIL_RETRY_ROUND", round_no, "pending", len(pending), "sleep", sleep_s, flush=True)
        time.sleep(sleep_s)
        next_pending = []
        for m, prior_error in pending:
            try:
                rr = fast.candidate_rows_for_market(m, spot, bn, der)
                rows.extend(rr)
                retry_history.append({"slug": m.slug, "round": round_no, "recovered": True, "prior_error": prior_error})
            except Exception as exc:
                next_pending.append((m, repr(exc)))
                retry_history.append({"slug": m.slug, "round": round_no, "recovered": False, "error": repr(exc)})
        pending = next_pending

    if pending:
        (out / "retry_failures.json").write_text(json.dumps([
            {"slug": m.slug, "condition_id": m.condition_id, "error": err} for m, err in pending
        ], indent=2), encoding="utf-8")
        raise RuntimeError(f"TAIL replay remains incomplete after serial retries: {len(pending)} market(s)")

    df = pd.DataFrame(rows, columns=fast.CANDIDATE_COLS)
    if len(df):
        df = df.drop_duplicates(["condition_id", "sec", "outcome"], keep="last").sort_values(["start", "sec", "outcome"], kind="mergesort")
    df.to_csv(out / "tail_candidates.csv", index=False)
    summary = {
        "period": [start, end],
        "markets_expected": int(len(inv)),
        "markets_mapped": int(inv["mapped"].fillna(False).astype(bool).sum()),
        "markets_with_tail_candidate": int(df["start"].nunique()) if len(df) else 0,
        "candidate_rows": int(len(df)),
        "initial_parallel_failures": int(len(failed)),
        "recovered_failures": int(len(failed)),
        "unresolved_failures": 0,
        "anchor_meta": anchor_meta,
        "entry_rule": "frozen V5 TAIL: fair>=0.99, exact-ticket post-fee settlement edge >0, same-second top-level size sufficient",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "retry_history.json").write_text(json.dumps(retry_history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("aggregate")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "score":
        score_range(args.start, args.end, args.out, args.workers)
    else:
        fast.aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
