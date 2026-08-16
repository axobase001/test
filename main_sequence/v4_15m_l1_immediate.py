from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence import final_v3_dual_bankroll_no_lookahead as engine

# Frozen V4: the signal rules are unchanged. Only historical execution admission
# is changed from a future +1..+5s print proxy to immediate same-second first-level
# evidence. The first level must contain the full dollar-ticket-derived quantity.
MAX_MARKET_CAP_USD = 200.0
STAKE_TIERS = (0.3125, 0.625, 1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 200.0)
engine.STAKE_TIERS = STAKE_TIERS

PROTOCOL = json.loads(json.dumps(engine.PROTOCOL))
PROTOCOL["name"] = "Main Sequence V4 BTC15m same-second L1 immediate CORE+TAIL / 2026-08-16 freeze"
PROTOCOL["entry"]["decision_liquidity"] = (
    "At decision second t, use only the best observed BUY level for the selected outcome. "
    "The full dollar-ticket-derived quantity must be visible at that exact first level; if so, "
    "an order sent at t is counted as filled at that level. No future print is required."
)
PROTOCOL["bankroll"]["max_single_market_open_capital_usd"] = MAX_MARKET_CAP_USD
PROTOCOL["bankroll"]["market_cap_scope"] = "combined open entry capital across CORE+TAIL in the same BTC15m market"
PROTOCOL["bankroll"]["execution"] = "same-second first-level immediate; future +1..+5s strict proxy retired"
PROTOCOL["bankroll"]["stake_tiers_usd"] = list(STAKE_TIERS)
PROTOCOL["anti_lookahead"] = [
    x for x in PROTOCOL.get("anti_lookahead", [])
    if "+1..+5" not in x and "Strict execution" not in x
] + [
    "Entry fill eligibility uses only the selected outcome's same-second best observed BUY level and exact-level size.",
    "A quote disappearing at t+1 cannot invalidate a t immediate order.",
    "Convergence exit uses only the same-second best observed SELL level and exact-level size at each causal exit second.",
    "Final outcome is settlement-only and never enters fair value, side selection or entry admission.",
]


def l1_levels_for_budgets(q: pd.DataFrame, m, budgets: tuple[float, ...]):
    if q is None or q.empty:
        return {}
    z = q[(q["side_u"] == "BUY") & (q["size"] > 0)].copy()
    if z.empty:
        return {}
    best = float(z["price"].min())
    at_best = z[np.isclose(z["price"].astype(float), best, rtol=0, atol=1e-12)]
    avail = float(at_best["size"].sum())
    out = {}
    for b in budgets:
        qty = engine.qty_for_budget(m, best, float(b))
        if qty > 0 and avail + 1e-10 >= qty:
            out[float(b)] = (best, float(qty))
    return out


def l1_sell_witness(qsec: pd.DataFrame, m, outcome: str, min_price: float, qty: float):
    z = qsec[(qsec["side_u"] == "SELL") & (qsec["outcome_l"] == outcome) &
             (qsec["price"] >= float(min_price)) & (qsec["size"] > 0)].copy()
    if z.empty:
        return None
    best = float(z["price"].max())
    at_best = z[np.isclose(z["price"].astype(float), best, rtol=0, atol=1e-12)]
    avail = float(at_best["size"].sum())
    if avail + 1e-10 < float(qty):
        return None
    gross = float(qty) * best
    fees = float(engine.base.fee_total(m, best, float(qty)))
    return {"proceeds": gross - fees, "avg": best}


def execute_immediate(m, g: pd.DataFrame, sec: int, outcome: str, budget: float, qty: float,
                      limit: float, edge: float, raw_gap: float, fair_selected: float,
                      family: str, fb: dict, spot, bn, der):
    won = (m.label_up >= 0.5) if outcome == "up" else (m.label_up < 0.5)
    payout = float(qty) if won else 0.0
    entry_cost = float(qty) * float(limit) + float(engine.base.fee_total(m, float(limit), float(qty)))

    reward = payout - entry_cost
    exit_time = int(m.close)
    exit_kind = "settlement" if raw_gap < engine.base.LARGE_RAW_GAP else "settlement_fallback"
    hold_s = int(m.close - sec)
    exit_px = 1.0 if won else 0.0

    if raw_gap >= engine.base.LARGE_RAW_GAP:
        sells = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] < m.close) &
                  (g["side_u"] == "SELL") & (g["outcome_l"] == outcome)]
        for sx in sorted(int(x) for x in sells["timestamp"].unique().tolist()):
            fbx = engine.base.fair_boundary(m, sx, spot, bn, der)
            if fbx is None:
                continue
            min_price = max(0.0, min(1.0, float(fbx[outcome]) - engine.base.EXIT_BAND))
            w = l1_sell_witness(sells[sells["timestamp"] == sx], m, outcome, min_price, float(qty))
            if w is not None:
                reward = float(w["proceeds"]) - entry_cost
                exit_time = int(sx)
                exit_kind = "convergence_l1"
                hold_s = int(sx - sec)
                exit_px = float(w["avg"])
                break

    return {
        "slug": m.slug, "condition_id": m.condition_id, "start": int(m.start), "close": int(m.close),
        "decision": int(sec), "s2c": int(m.close - sec), "signal_family": family,
        "side": "Up" if outcome == "up" else "Down", "won": bool(won),
        "stake_budget": float(budget), "qty_target": float(qty), "limit": float(limit),
        "signal_edge": float(edge), "initial_raw_gap": float(raw_gap),
        "selected_conservative_fair": float(fair_selected),
        "p_rv": float(fb["p_rv"]), "p_deribit": float(fb["p_iv"]), "rv": float(fb["rv"]),
        "deribit_iv": float(fb["div"]), "spot": float(fb["spot"]), "open_spot": float(fb["open_spot"]),
        "fee_enabled": m.fee_enabled, "fee_type": m.fee_type, "fee_rate": m.fee_rate,
        "fee_exponent": m.fee_exponent, "fee_source": m.fee_source,
        "signal_fee_per_share": engine.fee_ps_for_qty(m, float(limit), float(qty)),
        # Keep the inherited witness column names for shard compatibility. In V4 these ARE the immediate L1 execution layer.
        "witness_cost": float(entry_cost), "witness_reward": float(reward),
        "witness_exit_time": int(exit_time), "witness_exit_kind": exit_kind,
        "witness_hold_s": int(hold_s), "witness_exit_px": float(exit_px),
        "strict_cost": None, "strict_reward": None, "strict_entry_time": None,
        "strict_exit_time": None, "strict_exit_kind": "retired_v4", "strict_hold_s": None, "strict_exit_px": None,
    }


# score_market_all_tiers resolves these globals dynamically in engine's module.
engine.witness_levels_for_budgets = l1_levels_for_budgets
engine.execute_budget_signal = execute_immediate


def stake_for_equity(eq: float):
    if not (eq > 0 and math.isfinite(eq)):
        return None
    exp = int(math.floor(math.log(eq / engine.INITIAL_EQUITY, 2.0)))
    nominal = engine.BASE_STAKE * (2.0 ** exp)
    if nominal < min(STAKE_TIERS):
        return None
    return float(min(nominal, MAX_MARKET_CAP_USD))


def simulate_window(records: pd.DataFrame, start: str, end: str):
    s0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    s1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    x = records[(records["start"] >= s0) & (records["start"] < s1)].copy()
    x["stake_budget"] = pd.to_numeric(x["stake_budget"], errors="coerce")

    eq = float(engine.INITIAL_EQUITY)
    peak = eq
    max_dd = 0.0
    turnover = 0.0
    core_pnl = 0.0
    tail_pnl = 0.0
    trades = []
    path = [{"time": s0, "equity": eq, "event": "start"}]
    cap_rejects = 0
    both_markets = 0
    l1_selected = 0
    max_ticket = 0.0

    # BTC15m positions from a market exit no later than that market close; the next
    # market's decision window starts 5 minutes after that boundary. Thus market
    # starts are realized-equity sizing points and there is no cross-market overlap.
    for market_start, g in x.groupby("start", sort=True):
        ticket = stake_for_equity(eq)
        if ticket is None:
            break
        z = g[np.isclose(g["stake_budget"].astype(float), ticket, rtol=0, atol=1e-10)].copy()
        if z.empty:
            continue
        max_ticket = max(max_ticket, ticket)
        z = z.sort_values(["decision", "signal_family"], kind="mergesort")
        l1_selected += int(len(z))

        open_positions = []
        accepted = []
        for r in z.itertuples(index=False):
            t = int(r.decision)
            open_positions = [p for p in open_positions if int(p["exit_time"]) > t]
            occupied = sum(float(p["cost"]) for p in open_positions)
            cost = float(r.witness_cost)
            if occupied + cost > MAX_MARKET_CAP_USD + 1e-9 or cost > eq + 1e-9:
                cap_rejects += 1
                continue
            p = {"cost": cost, "exit_time": int(r.witness_exit_time), "family": str(r.signal_family)}
            open_positions.append(p)
            accepted.append(r)
        if len({str(r.signal_family) for r in accepted}) >= 2:
            both_markets += 1

        # Realize in actual exit-time order. Entry cost is already embedded in PnL;
        # no leverage is created because accepted entry costs are checked above.
        for r in sorted(accepted, key=lambda q: (int(q.witness_exit_time), int(q.decision))):
            pnl = float(r.witness_reward)
            turnover += float(r.witness_cost)
            fam = str(r.signal_family)
            if fam == "core_3c": core_pnl += pnl
            else: tail_pnl += pnl
            eq += pnl
            peak = max(peak, eq)
            max_dd = min(max_dd, eq / peak - 1.0 if peak > 0 else -1.0)
            path.append({"time": int(r.witness_exit_time), "equity": eq, "event": fam})
            trades.append({
                "market_start": int(r.start), "slug": r.slug, "decision": int(r.decision), "exit_time": int(r.witness_exit_time),
                "family": fam, "side": r.side, "ticket": ticket, "cost": float(r.witness_cost), "pnl": pnl,
                "exit_kind": r.witness_exit_kind, "limit": float(r.limit), "qty": float(r.qty_target),
            })

    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    ret = eq / engine.INITIAL_EQUITY - 1.0
    cagr = (eq / engine.INITIAL_EQUITY) ** (365.0 / days) - 1.0 if eq > 0 and days > 0 else math.nan
    return {
        "period": [start, end], "days": days, "initial_equity": engine.INITIAL_EQUITY, "final_equity": eq,
        "total_return": ret, "calendar_cagr": cagr, "max_dd": max_dd, "turnover": turnover,
        "core_pnl": core_pnl, "tail_pnl": tail_pnl, "trades": len(trades), "both_family_markets": both_markets,
        "cap_or_cash_rejects": cap_rejects, "signals_at_dynamic_tier": l1_selected, "max_ticket": max_ticket,
        "protocol": PROTOCOL,
    }, pd.DataFrame(trades), pd.DataFrame(path)


def aggregate(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("tiered_records.csv"))
    if not files:
        raise RuntimeError("no tiered_records.csv")
    recs = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    recs = recs.drop_duplicates(["slug", "stake_budget", "signal_family"], keep="last")
    recs = recs.sort_values(["start", "stake_budget", "signal_family", "decision"], kind="mergesort")
    recs.to_csv(out / "all_l1_records.csv", index=False)

    summaries = {}
    all_trades = []
    all_paths = []
    for name, (start, end) in engine.WINDOWS.items():
        s, t, p = simulate_window(recs, start, end)
        summaries[name] = s
        if len(t): t.insert(0, "window", name); all_trades.append(t)
        if len(p): p.insert(0, "window", name); all_paths.append(p)
    if all_trades: pd.concat(all_trades, ignore_index=True).to_csv(out / "bankroll_trades.csv", index=False)
    if all_paths: pd.concat(all_paths, ignore_index=True).to_csv(out / "equity_paths.csv", index=False)
    final = {"protocol": PROTOCOL, "windows": summaries}
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("protocol"); p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--shard", required=True); s.add_argument("--out", type=Path, required=True); s.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("aggregate"); a.add_argument("--root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "protocol":
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "FROZEN_V4_15M_L1_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))
        print(json.dumps(PROTOCOL, indent=2))
    elif args.cmd == "score":
        engine.score_shard(args.start, args.end, args.shard, args.out, args.workers)
    else:
        aggregate(args.root, args.out)

if __name__ == "__main__":
    main()
