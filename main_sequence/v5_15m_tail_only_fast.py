from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from main_sequence import v4_15m_l1_immediate as v4i

# Frozen 15m policy inherited from V4/V3. This file changes I/O only:
# fetch several markets' public trade tape in one request tree, then score locally.
engine = v4i.engine
base = engine.base
TAIL_FAIR_FLOOR = float(engine.TAIL_FAIR_FLOOR)   # 0.95
CORE_NET_EDGE = float(engine.CORE_NET_EDGE)       # 0.03 upper bound for 15m TAIL
STAKE_TIERS = tuple(float(x) for x in v4i.STAKE_TIERS)
BATCH_MARKETS = 8


def tail_market_all_tiers(m, g: pd.DataFrame, spot, bn, der) -> list[dict]:
    if g is None or g.empty:
        return []
    pre = g[(g["timestamp"] >= m.close - base.MAX_S2C) &
            (g["timestamp"] <= m.close - base.MIN_S2C) &
            (g["side_u"] == "BUY")]
    if pre.empty:
        return []

    pending = set(STAKE_TIERS)
    out: list[dict] = []
    for sec in sorted(int(x) for x in pre["timestamp"].unique().tolist()):
        if not pending:
            break
        fb = base.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        qsec = pre[pre["timestamp"] == sec]
        budgets = tuple(sorted(pending))
        levels = {
            outcome: v4i.l1_levels_for_budgets(qsec[qsec["outcome_l"] == outcome], m, budgets)
            for outcome in ("up", "down")
        }
        for budget in budgets:
            obs = []
            for outcome in ("up", "down"):
                lv = levels[outcome].get(float(budget))
                if lv is None:
                    continue
                limit, qty = lv
                fair = float(fb[outcome])
                raw = fair - float(limit)
                fee_ps = engine.fee_ps_for_qty(m, float(limit), float(qty))
                edge = raw - fee_ps
                if fair >= TAIL_FAIR_FLOOR and 0.0 < edge < CORE_NET_EDGE:
                    obs.append((float(edge), outcome, float(limit), float(qty), float(raw), fair))
            if not obs:
                continue
            edge, outcome, limit, qty, raw, fair = max(obs, key=lambda x: (x[0], x[1] == "up"))
            out.append(v4i.execute_immediate(
                m, g, sec, outcome, budget, qty, limit, edge, raw, fair,
                "tail_favorite_sub3c", fb, spot, bn, der,
            ))
            pending.remove(float(budget))
    return out


def fetch_batch(markets, spot, bn, der):
    if not markets:
        return [], 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-v5-15m-tail-fast/1.0"})
    start = min(int(m.start) for m in markets) + 300
    end = max(int(m.close) for m in markets) - 1
    raw = base.query_trade_rows(sess, list(markets), start, end)
    tm = base.normalize_trades(raw)
    rows = []
    for m in markets:
        rows.extend(tail_market_all_tiers(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der))
    return rows, len(raw)


def score_range(start: str, end: str, shard: str, out: Path, workers: int = 6):
    out.mkdir(parents=True, exist_ok=True)
    hours, by_hour, inventory = base.discover(start, end, workers=min(16, max(4, workers)))
    inventory.to_csv(out / "market_inventory.csv", index=False)
    expected = int(len(inventory))
    mapped = int(inventory["mapped"].fillna(False).astype(bool).sum()) if expected else 0
    if mapped != expected:
        missing = inventory[~inventory["mapped"].fillna(False).astype(bool)] if expected else inventory
        missing.to_csv(out / "mapping_failures.csv", index=False)
        raise RuntimeError(f"15m TAIL fail-closed mapping coverage {mapped}/{expected}")

    markets = []
    for h in sorted(hours):
        markets.extend(sorted(by_hour.get(h, []), key=lambda m: int(m.start)))
    markets = sorted(markets, key=lambda m: int(m.start))
    if len(markets) != expected:
        raise RuntimeError(f"market object count mismatch {len(markets)} vs inventory {expected}")

    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")

    batches = [markets[i:i+BATCH_MARKETS] for i in range(0, len(markets), BATCH_MARKETS)]
    records: list[dict] = []
    raw_rows = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(fetch_batch, b, spot, bn, der): (i, b) for i, b in enumerate(batches)}
        for k, f in enumerate(as_completed(fut), 1):
            i, b = fut[f]
            try:
                rr, nr = f.result()
                records.extend(rr); raw_rows += int(nr)
            except Exception as exc:
                failures.append({
                    "batch": i,
                    "start": int(b[0].start) if b else None,
                    "end": int(b[-1].close) if b else None,
                    "error": repr(exc),
                })
            if k % 24 == 0 or k == len(batches):
                print("TAIL15_BATCH", shard, k, "/", len(batches), "records", len(records), "raw", raw_rows, "fail", len(failures), flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"15m TAIL batch failures {len(failures)}")

    df = pd.DataFrame(records, columns=engine.RECORD_COLUMNS)
    if len(df):
        df = df.drop_duplicates(["condition_id", "stake_budget", "signal_family"], keep="last")
        df = df.sort_values(["start", "stake_budget", "decision"], kind="mergesort")
    df.to_csv(out / "tail_tiered_records.csv", index=False)

    summary = {
        "shard": shard,
        "period": [start, end],
        "markets_expected": expected,
        "markets_mapped": mapped,
        "mapping_coverage": mapped / expected if expected else math.nan,
        "tail_records": int(len(df)),
        "tail_markets": int(df["start"].nunique()) if len(df) else 0,
        "raw_trade_rows": int(raw_rows),
        "anchor_meta": anchor_meta,
        "policy": {
            "fair_floor": TAIL_FAIR_FLOOR,
            "post_fee_edge": "0 < edge < 0.03",
            "execution": "same-second exact first-level full-size",
            "exit": "raw gap >=10c causal convergence L1 else settlement",
            "stake_tiers": list(STAKE_TIERS),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    score_range(args.start, args.end, args.shard, args.out, args.workers)


if __name__ == "__main__":
    main()
