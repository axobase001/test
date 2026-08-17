from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from main_sequence import final_recent_replay as base
from main_sequence import v4_hourly_core_tail as v4
from main_sequence import v5_hourly_symmetric_stop as v5
from main_sequence import v5_tail_only_fast as fast


def candidates_from_tape(m, g: pd.DataFrame, spot, bn, der):
    if g is None or g.empty:
        return []
    rows = []
    for sec in sorted(int(x) for x in g["timestamp"].dropna().unique().tolist() if m.start <= int(x) < m.close):
        fb = v4.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        q = g[g["timestamp"] == sec]
        for outcome in ("up", "down"):
            fair = float(fb[outcome])
            if fair < fast.TAIL_FLOOR:
                continue
            lv = v4.top_level(q, "BUY", outcome)
            if lv is None:
                continue
            ask, available = map(float, lv)
            if not (0.0 < ask < 1.0 and available > 0.0):
                continue
            rows.append({
                "start": m.start, "close": m.close, "slug": m.slug, "event_slug": m.event_slug,
                "condition_id": m.condition_id, "label_up": m.label_up,
                "fee_enabled": m.fee_enabled, "fee_type": m.fee_type,
                "fee_rate": m.fee_rate, "fee_exponent": m.fee_exponent, "fee_source": m.fee_source,
                "sec": sec, "outcome": outcome, "fair": fair, "ask": ask, "available": available,
                "p_rv": fb["p_rv"], "p_iv": fb["p_iv"], "rv": fb["rv"], "iv": fb["iv"],
            })
    return rows


def score_range(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = v5.discover(start, end, workers=min(12, max(4, workers)))
    inv.to_csv(out / "inventory.csv", index=False)
    if not markets:
        pd.DataFrame(columns=fast.CANDIDATE_COLS).to_csv(out / "tail_candidates.csv", index=False)
        summary = {"period": [start, end], "markets_expected": int(len(inv)), "markets_mapped": 0, "candidate_rows": 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")

    s0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    s1 = int(pd.Timestamp(end, tz="UTC").timestamp()) - 1
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-v5-tail-batched/1.0"})
    print("TAIL_BATCH_FETCH", start, end, "markets", len(markets), flush=True)
    raw = base.query_trade_rows(sess, markets, s0, s1)
    by_condition = base.normalize_trades(raw)
    print("TAIL_BATCH_FETCH_DONE", "raw", len(raw), "conditions", len(by_condition), flush=True)

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {
            ex.submit(candidates_from_tape, m, by_condition.get(m.condition_id, pd.DataFrame()), spot, bn, der): m
            for m in markets
        }
        for i, f in enumerate(as_completed(fut), 1):
            m = fut[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                failures.append({"slug": m.slug, "condition_id": m.condition_id, "error": repr(exc)})
            if i % 48 == 0:
                print("TAIL_BATCH_SCORE", i, "/", len(markets), "candidates", len(rows), "fail", len(failures), flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"batched score failures: {len(failures)}")

    df = pd.DataFrame(rows, columns=fast.CANDIDATE_COLS)
    if len(df):
        df = df.drop_duplicates(["condition_id", "sec", "outcome"], keep="last").sort_values(["start", "sec", "outcome"], kind="mergesort")
    df.to_csv(out / "tail_candidates.csv", index=False)

    mapped = int(inv["mapped"].fillna(False).astype(bool).sum())
    tape_conditions = len(set(by_condition).intersection({m.condition_id for m in markets}))
    summary = {
        "period": [start, end],
        "markets_expected": int(len(inv)),
        "markets_mapped": mapped,
        "mapped_markets_with_any_trade_tape": int(tape_conditions),
        "markets_with_tail_candidate": int(df["start"].nunique()) if len(df) else 0,
        "candidate_rows": int(len(df)),
        "raw_trade_rows": int(len(raw)),
        "anchor_meta": anchor_meta,
        "entry_rule": "frozen V5 TAIL; I/O batched only: fair>=0.99 and exact-ticket post-fee edge>0 with same-second top-level size sufficient",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=8)
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
