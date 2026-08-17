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

# Frozen 15m policy inherited from V4/V3. This file changes I/O only.
engine = v4i.engine
base = engine.base
TAIL_FAIR_FLOOR = float(engine.TAIL_FAIR_FLOOR)   # 0.95
CORE_NET_EDGE = float(engine.CORE_NET_EDGE)       # 0.03 upper bound for 15m TAIL
STAKE_TIERS = tuple(float(x) for x in v4i.STAKE_TIERS)
BATCH_MARKETS = 8
TRADE_PAGE = 10_000  # current documented Data API maximum


def query_trade_rows_10k(sess: requests.Session, markets, start: int, end: int) -> list[dict]:
    """Complete public Data API retrieval using the current documented 10k page.

    A full page is treated as potentially truncated and recursively split by market,
    then time. A single-market/single-second overflow gets the only additional legal
    offset page (10k); >20k rows in one market-second fails closed.
    """
    if not markets or end < start:
        return []
    q = {
        "market": ",".join(m.condition_id for m in markets),
        "start": int(start), "end": int(end), "limit": TRADE_PAGE,
        "offset": 0, "takerOnly": "true",
    }
    rows = base.get_json(sess, base.DATA_API + "/trades", params=q, timeout=60)
    if len(rows) < TRADE_PAGE:
        return rows
    if len(markets) > 1:
        mid = len(markets) // 2
        return query_trade_rows_10k(sess, markets[:mid], start, end) + query_trade_rows_10k(sess, markets[mid:], start, end)
    if start < end:
        mid_t = (start + end) // 2
        return query_trade_rows_10k(sess, markets, start, mid_t) + query_trade_rows_10k(sess, markets, mid_t + 1, end)
    q2 = dict(q); q2["offset"] = TRADE_PAGE
    rr = base.get_json(sess, base.DATA_API + "/trades", params=q2, timeout=60)
    out = list(rows) + list(rr)
    if len(rr) < TRADE_PAGE:
        return out
    raise RuntimeError(f"Data API >20k trades in one market-second {markets[0].slug} {start}; refusing truncation")


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
    sess.headers.update({"User-Agent": "main-sequence-v5-15m-tail-fast/2.0"})
    start = min(int(m.start) for m in markets) + 300
    end = max(int(m.close) for m in markets) - 1
    raw = query_trade_rows_10k(sess, list(markets), start, end)
    tm = base.normalize_trades(raw)
    rows = []
    for m in markets:
        rows.extend(tail_market_all_tiers(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der))
    return rows, len(raw)


def score_range(start: str, end: str, shard: str, out: Path, workers: int = 3):
    out.mkdir(parents=True, exist_ok=True)
    hours, by_hour, inventory = base.discover(start, end, workers=min(12, max(4, workers * 2)))
    inventory.to_csv(out / "market_inventory.csv", index=False)
    theoretical = int(len(inventory))
    exists = inventory["exists_in_gamma"].fillna(False).astype(bool) if theoretical else pd.Series(dtype=bool)
    mapped_mask = inventory["mapped"].fillna(False).astype(bool) if theoretical else pd.Series(dtype=bool)
    tradable_expected = int(exists.sum()) if theoretical else 0
    mapped = int(mapped_mask.sum()) if theoretical else 0
    bad_existing = inventory[exists & ~mapped_mask] if theoretical else inventory
    if len(bad_existing):
        bad_existing.to_csv(out / "mapping_failures.csv", index=False)
        raise RuntimeError(f"15m TAIL fail-closed existing-market mapping {mapped}/{tradable_expected}; bad={len(bad_existing)}")

    markets = []
    for h in sorted(hours):
        markets.extend(sorted(by_hour.get(h, []), key=lambda m: int(m.start)))
    markets = sorted(markets, key=lambda m: int(m.start))
    if len(markets) != mapped:
        raise RuntimeError(f"market object count mismatch {len(markets)} vs mapped {mapped}")

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
        "theoretical_quarter_hours": theoretical,
        "gamma_contracts_existing": tradable_expected,
        "markets_mapped": mapped,
        "existing_market_mapping_coverage": mapped / tradable_expected if tradable_expected else math.nan,
        "nonexistent_quarter_hours": theoretical - tradable_expected,
        "tail_records": int(len(df)),
        "tail_markets": int(df["start"].nunique()) if len(df) else 0,
        "raw_trade_rows": int(raw_rows),
        "trade_page_limit": TRADE_PAGE,
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
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    score_range(args.start, args.end, args.shard, args.out, args.workers)


if __name__ == "__main__":
    main()
