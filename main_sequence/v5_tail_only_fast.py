from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from main_sequence import final_recent_replay as base
from main_sequence import v4_hourly_core_tail as v4
from main_sequence import v5_hourly_symmetric_stop as v5

TAIL_FLOOR = 0.99
INITIAL_EQUITY = 50.0
MAX_MARKET_OPEN_USD = 100.0

CANDIDATE_COLS = [
    "start", "close", "slug", "event_slug", "condition_id", "label_up",
    "fee_enabled", "fee_type", "fee_rate", "fee_exponent", "fee_source",
    "sec", "outcome", "fair", "ask", "available", "p_rv", "p_iv", "rv", "iv",
]


def candidate_rows_for_market(m, spot, bn, der):
    g = v4.market_tape(m)
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
            if fair < TAIL_FLOOR:
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
    markets, inv = v5.discover(start, end, workers=min(24, max(4, workers * 2)))
    inv.to_csv(out / "inventory.csv", index=False)
    if not markets:
        pd.DataFrame(columns=CANDIDATE_COLS).to_csv(out / "tail_candidates.csv", index=False)
        summary = {"period": [start, end], "markets_expected": int(len(inv)), "markets_mapped": 0, "candidate_rows": 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(candidate_rows_for_market, m, spot, bn, der): m for m in markets}
        for i, f in enumerate(as_completed(fut), 1):
            m = fut[f]
            try:
                rows.extend(f.result())
            except Exception as exc:
                failures.append({"slug": m.slug, "error": repr(exc)})
            if i % 48 == 0:
                print("TAIL_FAST", i, "/", len(markets), "candidates", len(rows), "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"tail-fast failures {failures[:10]} count={len(failures)}")

    df = pd.DataFrame(rows, columns=CANDIDATE_COLS)
    if len(df):
        df = df.sort_values(["start", "sec", "outcome"], kind="mergesort")
    df.to_csv(out / "tail_candidates.csv", index=False)
    summary = {
        "period": [start, end],
        "markets_expected": int(len(inv)),
        "markets_mapped": int(inv["mapped"].fillna(False).astype(bool).sum()),
        "markets_with_tail_candidate": int(df["start"].nunique()) if len(df) else 0,
        "candidate_rows": int(len(df)),
        "anchor_meta": anchor_meta,
        "entry_rule": "fair>=0.99 and exact current-ticket post-fee settlement edge > 0 with same-second top-level size sufficient",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def market_from_candidate(r):
    def fnum(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except Exception:
            return None
    def bval(x):
        if isinstance(x, bool):
            return x
        return str(x).strip().lower() in {"true", "1", "yes"}
    return v4.HourMarket(
        slug=str(r.slug), event_slug=str(r.event_slug), start=int(r.start),
        condition_id=str(r.condition_id), label_up=float(r.label_up), fee_enabled=bval(r.fee_enabled),
        fee_type="" if pd.isna(r.fee_type) else str(r.fee_type), fee_rate=fnum(r.fee_rate),
        fee_exponent=fnum(r.fee_exponent), fee_source="" if pd.isna(r.fee_source) else str(r.fee_source),
    )


def simulate_tail(candidates: pd.DataFrame, inv: pd.DataFrame, start: str, end: str):
    s0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    s1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    df = candidates[(candidates.start >= s0) & (candidates.start < s1)].sort_values(["start", "sec", "outcome"], kind="mergesort")
    iv = inv[(inv.start >= s0) & (inv.start < s1)].copy() if len(inv) else inv

    eq = INITIAL_EQUITY
    peak = eq
    maxdd = 0.0
    pnl_total = 0.0
    entry_capital = 0.0
    entries = wins = losses = 0
    max_ticket = 0.0
    events = []

    for market_start, g in df.groupby("start", sort=True):
        ticket = v5.ticket_for_equity(eq)
        if ticket is None:
            break
        g = g.sort_values(["sec", "outcome"], kind="mergesort")
        chosen = None
        m = market_from_candidate(g.iloc[0])
        for sec, sg in g.groupby("sec", sort=True):
            feasible = []
            for r in sg.itertuples(index=False):
                ask = float(r.ask); available = float(r.available); fair = float(r.fair)
                qty = v4.qty_for_budget(m, ask, ticket)
                if qty <= 0 or available + 1e-12 < qty:
                    continue
                cost = qty * ask + v4.fee_total(m, ask, qty)
                exp_profit = qty * fair - cost
                if fair >= TAIL_FLOOR and exp_profit > 0 and cost <= min(MAX_MARKET_OPEN_USD, eq) + 1e-9:
                    feasible.append((exp_profit, r, qty, cost))
            if feasible:
                exp_profit, r, qty, cost = max(feasible, key=lambda x: x[0])
                chosen = (r, qty, cost, exp_profit)
                break
        if chosen is None:
            continue

        r, qty, cost, exp_profit = chosen
        won_up = m.label_up >= 0.5
        won = won_up if str(r.outcome) == "up" else (not won_up)
        payout = qty if won else 0.0
        pnl = payout - cost
        eq += pnl
        pnl_total += pnl
        entry_capital += cost
        entries += 1; wins += int(won); losses += int(not won)
        max_ticket = max(max_ticket, ticket)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
        events.append({
            "market_start": int(market_start), "entry_sec": int(r.sec), "outcome": str(r.outcome),
            "fair": float(r.fair), "ask": float(r.ask), "available": float(r.available),
            "ticket": ticket, "qty": qty, "entry_cost": cost, "expected_profit": exp_profit,
            "won": bool(won), "pnl": pnl, "equity": eq,
        })

    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    cagr = (eq / INITIAL_EQUITY) ** (365.0 / days) - 1.0 if eq > 0 else math.nan
    expected = int(len(iv)) if len(iv) else 0
    mapped = int(iv.mapped.fillna(False).astype(bool).sum()) if len(iv) else 0
    return {
        "period": [start, end], "days": days, "initial_equity": INITIAL_EQUITY, "final_equity": eq,
        "total_return": eq / INITIAL_EQUITY - 1.0, "calendar_cagr": cagr, "realized_max_dd": maxdd,
        "total_pnl": pnl_total, "entry_capital": entry_capital,
        "pnl_over_entry_capital": pnl_total / entry_capital if entry_capital > 0 else math.nan,
        "tail_entries": entries, "tail_wins": wins, "tail_losses": losses,
        "win_rate": wins / entries if entries else math.nan, "max_ticket_used": max_ticket,
        "markets_expected": expected, "markets_mapped": mapped,
        "market_mapping_coverage": mapped / expected if expected else math.nan,
        "candidate_markets": int(df.start.nunique()) if len(df) else 0,
        "events": events,
    }


def aggregate(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cf = sorted(root.rglob("tail_candidates.csv"))
    inf = sorted(root.rglob("inventory.csv"))
    if not cf:
        raise RuntimeError("no tail_candidates.csv")
    candidates = pd.concat([pd.read_csv(p) for p in cf], ignore_index=True)
    if len(candidates):
        candidates = candidates.drop_duplicates(["condition_id", "sec", "outcome"], keep="last")
    inv = pd.concat([pd.read_csv(p) for p in inf], ignore_index=True).drop_duplicates(["start"], keep="last") if inf else pd.DataFrame()
    results = {name: simulate_tail(candidates, inv, a, b) for name, (a, b) in v5.WINDOWS.items()}
    (out / "TAIL_ONLY_3M_6M_9M_12M.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "events"} | {"window": name} for name, r in results.items()]).to_csv(out / "TAIL_ONLY_SUMMARY.csv", index=False)
    for name, r in results.items():
        pd.DataFrame(r["events"]).to_csv(out / f"TAIL_ONLY_EVENTS_{name}.csv", index=False)
    print(json.dumps({k: {x: v for x, v in r.items() if x != "events"} for k, r in results.items()}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--out", type=Path, required=True); s.add_argument("--workers", type=int, default=12)
    a = sub.add_parser("aggregate")
    a.add_argument("--root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "score": score_range(args.start, args.end, args.out, args.workers)
    else: aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
