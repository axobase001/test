from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from honest_backtest import Decision, grade_taker
from honest_backtest.adapters.parquet_pm import load_corpus


def stake_tier(realized_capital: float) -> float:
    """$5 at $50 bankroll; double stake at each bankroll doubling; cap $100."""
    if realized_capital < 100.0:
        return 5.0
    doublings = max(0, math.floor(math.log(realized_capital / 50.0, 2.0)))
    return min(100.0, 5.0 * (2.0 ** doublings))


def as_ms(ts: int) -> int:
    ts = int(ts)
    return ts if abs(ts) >= 10**12 else ts * 1000


def fee_per_share(px: float, fee_rate: float) -> float:
    return float(fee_rate) * float(px) * (1.0 - float(px))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-dir", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-cash", type=float, default=50.0)
    ap.add_argument("--latency-ms", type=int, default=1000)
    ap.add_argument("--tape-window-ms", type=int, default=1500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ctxs = list(load_corpus(str(args.pm_dir), coins=("btc",), durations=("15m",)))
    by_cid = {str(c.meta.condition_id): c for c in ctxs}
    recs = pd.read_csv(args.records).sort_values("ts_ms").reset_index(drop=True)

    cash = float(args.start_cash)
    open_pos: list[dict] = []
    trades: list[dict] = []
    attempts: list[dict] = []
    curve: list[dict] = [{"ts_ms": int(recs.ts_ms.min()), "equity": cash, "cash": cash, "event": "start"}]
    skipped = {"missing_ctx": 0, "cash": 0, "market_cap": 0, "no_tape": 0, "not_fillable": 0, "bad_px": 0}

    def conservative_equity() -> float:
        # Unsettled positions are carried only at cash cost. No future/resolution value
        # can affect sizing before settlement.
        return cash + sum(float(p["cost"]) for p in open_pos)

    def settle_until(ts_ms: int) -> None:
        nonlocal cash, open_pos
        due = [p for p in open_pos if int(p["close_ts_ms"]) <= int(ts_ms)]
        if not due:
            return
        due.sort(key=lambda p: (p["close_ts_ms"], p["ts_ms"]))
        for p in due:
            cash += float(p["shares"]) if bool(p["won"]) else 0.0
            p["settled_cash"] = cash
            p["pnl"] = (float(p["shares"]) if bool(p["won"]) else 0.0) - float(p["cost"])
            curve.append({"ts_ms": int(p["close_ts_ms"]), "equity": conservative_equity(), "cash": cash, "event": "settle"})
        ids = {id(p) for p in due}
        open_pos = [p for p in open_pos if id(p) not in ids]

    for r in recs.itertuples(index=False):
        ts_ms = int(r.ts_ms)
        settle_until(ts_ms)
        cid = str(r.cid)
        ctx = by_cid.get(cid)
        if ctx is None:
            skipped["missing_ctx"] += 1
            continue
        capital = conservative_equity()
        planned_cash = stake_tier(capital)
        if planned_cash > 100.0 + 1e-9:
            raise AssertionError("per-trade cap violated")
        if cash + 1e-9 < planned_cash:
            skipped["cash"] += 1
            continue
        market_exposure = sum(float(p["cost"]) for p in open_pos if p["cid"] == cid)
        if market_exposure + planned_cash > 200.0 + 1e-9:
            skipped["market_cap"] += 1
            continue

        yes = bool(r.yes)
        target_px = float(r.target_px if np.isfinite(r.target_px) else r.best_ask)
        fee_rate = float(r.fee_rate)
        target_cps = target_px + fee_per_share(target_px, fee_rate)
        if not (0.0 < target_px < 1.0 and target_cps > 0.0):
            skipped["bad_px"] += 1
            continue
        # User sizing is CASH, not shares. Convert cash budget into requested shares.
        shares = planned_cash / target_cps
        i = int(np.searchsorted(ctx.ts, ts_ms, side="right") - 1)
        if i < 0 or abs(int(ctx.ts[i]) - ts_ms) > 1000:
            skipped["missing_ctx"] += 1
            continue
        d = Decision(i=i, ts_ms=ts_ms, token_yes=yes, action="taker", target_px=target_px, size=shares)
        g = grade_taker(ctx, d, latency_ms=args.latency_ms, tape_window_ms=args.tape_window_ms)
        has_tape = bool(g.get("has_tape"))
        fillable = bool(g.get("fillable"))
        attempts.append({
            "ts_ms": ts_ms, "cid": cid, "capital": capital, "planned_cash": planned_cash,
            "shares_requested": shares, "target_px": target_px, "has_tape": has_tape,
            "fillable": fillable, "honest_sz": g.get("honest_sz"), "crossable": g.get("crossable"),
            "persisted": g.get("persisted"), "won": bool(r.won), "fee_rate": fee_rate,
        })
        if not has_tape:
            skipped["no_tape"] += 1
            continue
        if not fillable:
            skipped["not_fillable"] += 1
            continue
        fill_px = float(g.get("fill_px", target_px))
        actual_cps = fill_px + fee_per_share(fill_px, fee_rate)
        actual_cost = shares * actual_cps
        # A taker limit replay must never charge more than the cash budget implied by
        # the decision's target ask. Fail closed rather than silently borrowing cash.
        if actual_cost > planned_cash + 1e-6 or actual_cost > cash + 1e-6:
            raise RuntimeError(f"fill cost exceeds cash budget: cost={actual_cost} planned={planned_cash} cash={cash}")
        cash -= actual_cost
        close_ts_ms = as_ms(int(ctx.meta.close_ts))
        pos = {
            "ts_ms": ts_ms, "close_ts_ms": close_ts_ms, "cid": cid, "yes": yes,
            "won": bool(r.won), "capital_before": capital, "tier": planned_cash,
            "shares": shares, "target_px": target_px, "fill_px": fill_px,
            "fee_rate": fee_rate, "cost": actual_cost,
            "honest_sz": float(g.get("honest_sz") or 0.0),
        }
        open_pos.append(pos)
        trades.append(pos)
        curve.append({"ts_ms": ts_ms, "equity": conservative_equity(), "cash": cash, "event": "open"})

    settle_until(10**18)
    final_cash = cash
    cdf = pd.DataFrame(curve).sort_values(["ts_ms", "event"]).reset_index(drop=True)
    tdf = pd.DataFrame(trades)
    adf = pd.DataFrame(attempts)
    cdf.to_csv(args.out / "equity_curve.csv", index=False)
    tdf.to_csv(args.out / "filled_trades.csv", index=False)
    adf.to_csv(args.out / "attempts.csv", index=False)

    peak = cdf.equity.cummax()
    mdd = float((cdf.equity / peak - 1.0).min())
    first_ts = int(cdf.ts_ms.min())
    last_ts = int(cdf.ts_ms.max())
    days = max((last_ts - first_ts) / 86_400_000.0, 1e-9)
    cagr = (final_cash / args.start_cash) ** (365.0 / days) - 1.0 if final_cash > 0 else -1.0
    max_market_exposure = 0.0
    # One signal per market under the frozen structural policy, but report the audited cap.
    if not tdf.empty:
        max_market_exposure = float(tdf.groupby("cid").cost.sum().max())
    summary = {
        "name": "strict structural 3c dynamic-cash bankroll replay",
        "execution": {"latency_ms": args.latency_ms, "tape_window_ms": args.tape_window_ms, "requires_has_tape": True, "requires_fillable": True},
        "bankroll_rule": {"start_cash": args.start_cash, "initial_trade_cash": 5.0, "double_trade_cash_each_capital_doubling": True, "per_trade_cap": 100.0, "per_market_window_cap": 200.0, "no_leverage": True, "unsettled_mark_to_cost": True},
        "signals_considered": int(len(recs)), "attempts_graded": int(len(adf)), "fills": int(len(tdf)),
        "wins": int(tdf.won.sum()) if not tdf.empty else 0,
        "win_rate": float(tdf.won.mean()) if not tdf.empty else None,
        "start_cash": args.start_cash, "final_cash": final_cash, "total_return": final_cash / args.start_cash - 1.0,
        "elapsed_days": days, "raw_cagr": cagr, "max_drawdown": mdd, "max_market_exposure": max_market_exposure,
        "stake_counts": {str(k): int(v) for k, v in (tdf.tier.value_counts().sort_index().to_dict().items() if not tdf.empty else [])},
        "skipped": skipped,
        "causality": "Sizing uses only settled cash plus open positions at historical cost; execution grading uses only post-decision tape/persistence; terminal won label affects settlement only.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "SUMMARY.md").write_text(
        "# Strict dynamic-cash bankroll replay\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print((args.out / "SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
