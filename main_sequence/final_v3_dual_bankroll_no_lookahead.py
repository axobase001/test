from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from main_sequence import final_recent_no_lookahead as causal

# Importing final_recent_no_lookahead installs the v3 causal Deribit-universe
# builder and paced Data-API transport into the shared replay module.
base = causal.base

CORE_NET_EDGE = 0.03
TAIL_FAIR_FLOOR = 0.95
INITIAL_EQUITY = 50.0
BASE_STAKE = 5.0
# Precompute exact execution at every power-of-two stake the bankroll can select.
# Negative tiers make the path de-risk after drawdown; high tiers prevent a hidden
# capacity extrapolation if compounding is unusually strong.
TIER_EXP_MIN = -4
TIER_EXP_MAX = 12
STAKE_TIERS = tuple(BASE_STAKE * (2.0 ** k) for k in range(TIER_EXP_MIN, TIER_EXP_MAX + 1))
WINDOWS = {
    "3m": ["2026-05-15", "2026-08-15"],
    "6m": ["2026-02-15", "2026-08-15"],
    "12m": ["2025-08-15", "2026-08-15"],
}

PROTOCOL = json.loads(json.dumps(base.PROTOCOL))
PROTOCOL["name"] = "Main Sequence V3 dual-sleeve max-turnover bankroll / 2026-08-16 freeze"
PROTOCOL["windows"] = WINDOWS
PROTOCOL["entry"] = {
    **PROTOCOL["entry"],
    "core": "post-historical-fee conservative edge >=3c; one CORE entry per BTC15m market",
    "tail": "selected conservative fair >=95% AND 0<post-fee conservative edge<3c; one TAIL entry per BTC15m market",
    "dual_sleeve": "CORE and TAIL are independent; both may enter the same BTC15m market, including overlapping holding periods",
    "sizing": "$5 total-cost budget at $50 equity; stake changes only by powers of two with account equity; exact shares are derived from historical price+fee",
    "decision_liquidity": "same-second public taker BUY tape must support the full dollar-budget-derived share quantity at/below the frozen limit",
}
PROTOCOL["bankroll"] = {
    "initial_equity_usd": INITIAL_EQUITY,
    "base_trade_budget_usd": BASE_STAKE,
    "stake_rule": "stake=5*2^floor(log2(equity/50)); evaluated at each market start, with symmetric downshift after drawdown",
    "no_leverage": True,
    "max_turnover": "take every strict executable CORE and TAIL signal at the selected stake tier; released cash is immediately reusable in the next market",
    "same_market_concurrency": "CORE+TAIL may both be funded in one market; each budget is <=10% of equity at the lower boundary of its tier, so two sleeves require <=20%",
    "capacity": "no 5-share extrapolation: every stake tier recomputes decision witness quantity and requires fresh +1..+5s full-size public tape execution",
    "stake_tiers_usd": list(STAKE_TIERS),
}
PROTOCOL["anti_lookahead"].extend([
    "CORE and TAIL are found independently using only observations timestamped <= each decision second; one family cannot use the other family's later signal.",
    "Dollar sizing changes the required historical share quantity and therefore recomputes the causal witness limit independently at every stake tier.",
    "Strict execution can only use public prints at decision+1..+5s; those prints cannot change the already-frozen side, fair, family or limit.",
    "The account stake tier is determined from realized account equity at market start. No final outcome, later fill or future equity is used to choose stake.",
    "Final outcome is settlement-only. It is never read by the signal finder or stake selector.",
])

RECORD_COLUMNS = [
    "slug", "condition_id", "start", "close", "decision", "s2c", "signal_family", "side", "won",
    "stake_budget", "qty_target", "limit", "signal_edge", "initial_raw_gap", "selected_conservative_fair",
    "p_rv", "p_deribit", "rv", "deribit_iv", "spot", "open_spot",
    "fee_enabled", "fee_type", "fee_rate", "fee_exponent", "fee_source", "signal_fee_per_share",
    "witness_cost", "witness_reward", "witness_exit_time", "witness_exit_kind", "witness_hold_s", "witness_exit_px",
    "strict_cost", "strict_reward", "strict_entry_time", "strict_exit_time", "strict_exit_kind", "strict_hold_s", "strict_exit_px",
]


def fee_ps_for_qty(m, p: float, qty: float) -> float:
    if qty <= 0:
        return math.inf
    return float(base.fee_total(m, float(p), float(qty))) / float(qty)


def qty_for_budget(m, p: float, budget: float) -> float:
    """Conservative shares whose limit-price cost including fee is <= budget."""
    p = float(p); budget = float(budget)
    if not (0 < p < 1 and budget > 0):
        return 0.0
    # Fee is linear in quantity before deterministic USDC rounding. Iterate to
    # absorb that tiny rounding term without ever overspending the budget.
    fps = float(base.fee_total(m, p, 1000.0)) / 1000.0
    q = budget / max(p + fps, 1e-12)
    for _ in range(5):
        cost = q * p + float(base.fee_total(m, p, q))
        if cost <= 0:
            return 0.0
        q *= budget / cost
    cost = q * p + float(base.fee_total(m, p, q))
    if cost > budget:
        q *= budget / cost
    return max(float(q), 0.0)


def witness_levels_for_budgets(q: pd.DataFrame, m, budgets: tuple[float, ...]) -> dict[float, tuple[float, float]]:
    """Return budget -> (worst witness limit, conservative target shares)."""
    if q is None or q.empty:
        return {}
    z = q[(q["side_u"] == "BUY") & (q["size"] > 0)].copy()
    if z.empty:
        return {}
    z = z.sort_values(["price", "size"], kind="mergesort")
    prices = z["price"].to_numpy(float)
    cum_qty = np.cumsum(z["size"].to_numpy(float))
    # At each candidate worst price, calculate the largest total-cost budget that
    # the observed cumulative quantity can support. Make it monotone for search.
    capacity_budget = np.empty(len(prices), dtype=float)
    for i, (p, cq) in enumerate(zip(prices, cum_qty)):
        capacity_budget[i] = float(cq) * float(p) + float(base.fee_total(m, float(p), float(cq)))
    capacity_budget = np.maximum.accumulate(capacity_budget)
    out: dict[float, tuple[float, float]] = {}
    for b in budgets:
        if capacity_budget[-1] + 1e-12 < float(b):
            continue
        i = int(np.searchsorted(capacity_budget, float(b), side="left"))
        p = float(prices[i])
        qtarget = qty_for_budget(m, p, float(b))
        if qtarget > 0 and float(cum_qty[i]) + 1e-10 >= qtarget:
            out[float(b)] = (p, qtarget)
    return out


def execute_budget_signal(m, g: pd.DataFrame, sec: int, outcome: str, budget: float, qty: float,
                          limit: float, edge: float, raw_gap: float, fair_selected: float,
                          family: str, fb: dict, spot, bn, der) -> dict:
    # Outcome is read here for payout only, after family/side/time/limit are frozen.
    won = (m.label_up >= 0.5) if outcome == "up" else (m.label_up < 0.5)
    payout = float(qty) if won else 0.0
    paper_cost = float(qty) * float(limit) + float(base.fee_total(m, float(limit), float(qty)))
    post5 = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] <= sec + 5)]
    entry = base.buy_fill(post5, m, outcome, float(limit), float(qty))
    regime_large = raw_gap >= base.LARGE_RAW_GAP

    witness_reward = payout - paper_cost
    witness_exit_time = m.close
    witness_exit_kind = "settlement" if not regime_large else "settlement_fallback"
    witness_hold_s = m.close - sec
    witness_exit_px = 1.0 if won else 0.0

    strict_cost = strict_reward = strict_entry_time = strict_exit_time = strict_hold_s = strict_exit_px = None
    strict_exit_kind = "no_entry_fill" if entry is None else ("settlement" if not regime_large else "settlement_fallback")
    if entry is not None:
        strict_cost = float(entry["cost"])
        strict_entry_time = int(entry["done"])
        strict_reward = payout - strict_cost
        strict_exit_time = m.close
        strict_hold_s = m.close - strict_entry_time
        strict_exit_px = 1.0 if won else 0.0

    if regime_large:
        sells = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] < m.close) &
                  (g["side_u"] == "SELL") & (g["outcome_l"] == outcome)]
        witness_exit = None
        strict_exit = None
        for sx in sorted(int(x) for x in sells["timestamp"].unique().tolist()):
            fbx = base.fair_boundary(m, sx, spot, bn, der)
            if fbx is None:
                continue
            min_price = max(0.0, min(1.0, float(fbx[outcome]) - base.EXIT_BAND))
            w = base.sell_witness(sells[sells["timestamp"] == sx], m, outcome, min_price, float(qty))
            if w is None:
                continue
            if witness_exit is None:
                witness_exit = {**w, "sec": sx, "limit": min_price}
            if entry is not None and sx >= int(entry["done"]) + 1 and strict_exit is None:
                sf = base.strict_sell_fill(g, m, outcome, min_price, sx, m.close, float(qty))
                if sf is not None:
                    strict_exit = {**sf, "witness_sec": sx, "limit": min_price}
                    break
        if witness_exit is not None:
            witness_reward = float(witness_exit["proceeds"]) - paper_cost
            witness_exit_time = int(witness_exit["sec"])
            witness_exit_kind = "convergence"
            witness_hold_s = witness_exit_time - sec
            witness_exit_px = float(witness_exit["avg"])
        if entry is not None and strict_exit is not None:
            strict_reward = float(strict_exit["proceeds"]) - float(entry["cost"])
            strict_exit_time = int(strict_exit["done"])
            strict_exit_kind = "convergence"
            strict_hold_s = strict_exit_time - int(entry["done"])
            strict_exit_px = float(strict_exit["avg"])

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
        "signal_fee_per_share": fee_ps_for_qty(m, float(limit), float(qty)),
        "witness_cost": float(paper_cost), "witness_reward": float(witness_reward),
        "witness_exit_time": int(witness_exit_time), "witness_exit_kind": witness_exit_kind,
        "witness_hold_s": int(witness_hold_s), "witness_exit_px": float(witness_exit_px),
        "strict_cost": strict_cost, "strict_reward": strict_reward, "strict_entry_time": strict_entry_time,
        "strict_exit_time": strict_exit_time, "strict_exit_kind": strict_exit_kind,
        "strict_hold_s": strict_hold_s, "strict_exit_px": strict_exit_px,
    }


def score_market_all_tiers(m, g: pd.DataFrame, spot, bn, der) -> list[dict]:
    if g is None or g.empty:
        return []
    pre = g[(g["timestamp"] >= m.close - base.MAX_S2C) &
            (g["timestamp"] <= m.close - base.MIN_S2C) & (g["side_u"] == "BUY")]
    if pre.empty:
        return []

    pending_core = set(float(x) for x in STAKE_TIERS)
    pending_tail = set(float(x) for x in STAKE_TIERS)
    out: list[dict] = []

    for sec in sorted(int(x) for x in pre["timestamp"].unique().tolist()):
        if not pending_core and not pending_tail:
            break
        fb = base.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        qsec = pre[pre["timestamp"] == sec]
        budgets_needed = tuple(sorted(pending_core | pending_tail))
        levels = {
            outcome: witness_levels_for_budgets(qsec[qsec["outcome_l"] == outcome], m, budgets_needed)
            for outcome in ("up", "down")
        }

        for budget in budgets_needed:
            observations = []
            for outcome in ("up", "down"):
                lv = levels[outcome].get(float(budget))
                if lv is None:
                    continue
                limit, qty = lv
                fair = float(fb[outcome])
                raw = fair - float(limit)
                fee_ps = fee_ps_for_qty(m, float(limit), float(qty))
                edge = raw - fee_ps
                observations.append((float(edge), outcome, float(limit), float(qty), float(raw), fair))
            if not observations:
                continue

            if float(budget) in pending_core:
                cc = [x for x in observations if x[0] >= CORE_NET_EDGE]
                if cc:
                    edge, outcome, limit, qty, raw, fair = max(cc, key=lambda x: (x[0], x[1] == "up"))
                    out.append(execute_budget_signal(m, g, sec, outcome, budget, qty, limit, edge, raw,
                                                     fair, "core_3c", fb, spot, bn, der))
                    pending_core.remove(float(budget))

            if float(budget) in pending_tail:
                tc = [x for x in observations if x[5] >= TAIL_FAIR_FLOOR and 0.0 < x[0] < CORE_NET_EDGE]
                if tc:
                    edge, outcome, limit, qty, raw, fair = max(tc, key=lambda x: (x[0], x[1] == "up"))
                    out.append(execute_budget_signal(m, g, sec, outcome, budget, qty, limit, edge, raw,
                                                     fair, "tail_favorite_sub3c", fb, spot, bn, der))
                    pending_tail.remove(float(budget))
    return out


def score_hour(hour: int, markets: list, spot, bn, der):
    if not markets:
        return [], 0
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-v3-dual-bankroll/1.0"})
    raw = base.query_trade_rows(sess, markets, hour + 300, hour + 3599)
    tm = base.normalize_trades(raw)
    records: list[dict] = []
    for m in markets:
        records.extend(score_market_all_tiers(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der))
    return records, len(raw)


def score_shard(start: str, end: str, shard: str, out: Path, workers: int = 4):
    out.mkdir(parents=True, exist_ok=True)
    hours, by_hour, inventory = base.discover(start, end, workers)
    inventory.to_csv(out / "market_inventory.csv", index=False)
    mapped = int(inventory["mapped"].sum()) if len(inventory) else 0
    if mapped == 0:
        pd.DataFrame(columns=RECORD_COLUMNS).to_csv(out / "tiered_records.csv", index=False)
        summary = {"shard": shard, "period": [start, end], "mapped_markets": 0, "records": 0, "protocol": PROTOCOL}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("DUAL_BANKROLL_SHARD_NO_MARKETS", json.dumps(summary), flush=True)
        return

    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")
    active_hours = [h for h in hours if by_hour.get(h)]
    records: list[dict] = []
    raw_rows = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(score_hour, h, by_hour[h], spot, bn, der): h for h in active_hours}
        for k, f in enumerate(as_completed(fut), 1):
            h = fut[f]
            try:
                rr, nr = f.result(); records.extend(rr); raw_rows += int(nr)
            except Exception as exc:
                failures.append({"hour": h, "error": repr(exc)})
            if k % 48 == 0:
                print("DUAL_SCORE", shard, k, "/", len(active_hours), "records", len(records), "raw", raw_rows, "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"hour score failures {failures[:10]} count={len(failures)}")

    df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    if len(df):
        df = df.sort_values(["start", "stake_budget", "signal_family", "decision"], kind="mergesort")
    df.to_csv(out / "tiered_records.csv", index=False)
    summary = {
        "shard": shard, "period": [start, end], "expected_markets": int(len(inventory)), "mapped_markets": mapped,
        "records": int(len(df)), "strict_fills": int(df["strict_cost"].notna().sum()) if len(df) else 0,
        "raw_trade_rows": int(raw_rows), "anchor_meta": anchor_meta, "protocol": PROTOCOL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DUAL_BANKROLL_SHARD_DONE", json.dumps(summary), flush=True)


def stake_for_equity(equity: float) -> tuple[float | None, int | None]:
    if not (equity > 0 and math.isfinite(equity)):
        return None, None
    exp = int(math.floor(math.log(equity / INITIAL_EQUITY, 2.0)))
    if exp < TIER_EXP_MIN or exp > TIER_EXP_MAX:
        return None, exp
    return float(BASE_STAKE * (2.0 ** exp)), exp


def simulate_window(records: pd.DataFrame, start: str, end: str):
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    x = records[(records["start"] >= start_ts) & (records["start"] < end_ts)].copy()
    x["stake_budget"] = pd.to_numeric(x["stake_budget"], errors="coerce")
    equity = float(INITIAL_EQUITY)
    peak = equity
    max_dd = 0.0
    turnover = 0.0
    pnl = 0.0
    trades = []
    tier_hist: dict[str, int] = {}
    family_pnl = {"core_3c": 0.0, "tail_favorite_sub3c": 0.0}
    family_trades = {"core_3c": 0, "tail_favorite_sub3c": 0}
    both_market_count = 0
    selected_signal_count = 0
    selected_no_fill_count = 0
    coverage_error = None
    equity_path = [{"time": start_ts, "equity": equity, "event": "start"}]

    # Every BTC15m position settles/exits no later than its market close. The next
    # market's decision window begins five minutes after that boundary, so market
    # starts are safe realized-equity sizing points with no cross-market overlap.
    for market_start, g in x.groupby("start", sort=True):
        stake, exp = stake_for_equity(equity)
        if stake is None:
            coverage_error = {"market_start": int(market_start), "equity": equity, "required_exponent": exp}
            break
        tier_hist[str(stake)] = tier_hist.get(str(stake), 0) + 1
        z = g[np.isclose(g["stake_budget"].astype(float), stake, rtol=0, atol=1e-10)].copy()
        if z.empty:
            continue
        # One record max per family at a given tier/market.
        selected_signal_count += int(len(z))
        selected_no_fill_count += int(z["strict_cost"].isna().sum())
        fills = z[z["strict_cost"].notna()].copy()
        if fills.empty:
            continue
        if fills["signal_family"].nunique() >= 2:
            both_market_count += 1
        market_pnl = 0.0
        for r in fills.itertuples(index=False):
            cost = float(r.strict_cost); rew = float(r.strict_reward)
            turnover += cost; pnl += rew; market_pnl += rew
            fam = str(r.signal_family)
            family_pnl[fam] = family_pnl.get(fam, 0.0) + rew
            family_trades[fam] = family_trades.get(fam, 0) + 1
            trades.append({
                "window_start": start, "window_end": end, "market_start": int(r.start), "slug": r.slug,
                "family": fam, "side": r.side, "stake_budget": stake, "qty": float(r.qty_target),
                "entry_cost": cost, "pnl": rew, "won": bool(r.won), "entry_time": int(r.strict_entry_time),
                "exit_time": int(r.strict_exit_time), "exit_kind": r.strict_exit_kind,
                "equity_before_market": equity,
            })
        equity += market_pnl
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) if peak > 0 else -1.0
        max_dd = min(max_dd, dd)
        equity_path.append({"time": int(market_start) + 900, "equity": equity, "event": "market_close"})

    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    total_return = equity / INITIAL_EQUITY - 1.0
    cagr = (equity / INITIAL_EQUITY) ** (365.0 / days) - 1.0 if equity > 0 and days > 0 else math.nan
    tdf = pd.DataFrame(trades)
    wins = int((tdf["pnl"] > 0).sum()) if len(tdf) else 0
    losses = int((tdf["pnl"] < 0).sum()) if len(tdf) else 0
    max_stake = float(tdf["stake_budget"].max()) if len(tdf) else BASE_STAKE
    summary = {
        "period": [start, end], "calendar_days": days, "initial_equity": INITIAL_EQUITY, "final_equity": equity,
        "total_return": total_return, "calendar_cagr": cagr, "realized_market_close_max_dd": max_dd,
        "strict_trades": int(len(tdf)), "wins": wins, "losses": losses,
        "win_rate": float(wins / len(tdf)) if len(tdf) else math.nan,
        "strict_turnover": turnover, "turnover_multiple_initial_capital": turnover / INITIAL_EQUITY,
        "annualized_turnover_multiple_initial_capital": (turnover / INITIAL_EQUITY) * (365.0 / days),
        "net_pnl": pnl, "max_stake_budget": max_stake, "stake_tier_market_counts": tier_hist,
        "family_pnl": family_pnl, "family_trades": family_trades,
        "markets_with_both_core_and_tail_strict_fill": both_market_count,
        "selected_family_signals_at_dynamic_tier": selected_signal_count,
        "selected_signals_without_strict_fill": selected_no_fill_count,
        "tier_coverage_error": coverage_error,
    }
    return summary, tdf, pd.DataFrame(equity_path)


def aggregate(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rec_files = sorted(root.rglob("tiered_records.csv"))
    inv_files = sorted(root.rglob("market_inventory.csv"))
    if not rec_files:
        raise RuntimeError(f"no tiered records below {root}")
    recs = [pd.read_csv(p) for p in rec_files]
    records = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=RECORD_COLUMNS)
    if len(records):
        records = records.sort_values(["start", "stake_budget", "signal_family", "decision"], kind="mergesort")
    records.to_csv(out / "all_tiered_records.csv", index=False)
    if inv_files:
        inv = pd.concat([pd.read_csv(p) for p in inv_files], ignore_index=True).drop_duplicates(["slug", "start"])
        inv.to_csv(out / "all_market_inventory.csv", index=False)

    summaries = {}
    all_trades = []
    all_paths = []
    for name, (start, end) in WINDOWS.items():
        s, t, ep = simulate_window(records, start, end)
        summaries[name] = s
        if len(t):
            t.insert(0, "window", name); all_trades.append(t)
        if len(ep):
            ep.insert(0, "window", name); all_paths.append(ep)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    paths = pd.concat(all_paths, ignore_index=True) if all_paths else pd.DataFrame()
    trades.to_csv(out / "bankroll_trades.csv", index=False)
    paths.to_csv(out / "equity_paths.csv", index=False)

    final = {"protocol": PROTOCOL, "windows": summaries}
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    lines = ["# Main Sequence V3 Dual-Sleeve Dollar Bankroll", "", "No-lookahead, exact historical dollar-notional tape replay.", ""]
    for name in ("3m", "6m", "12m"):
        s = summaries[name]
        lines += [
            f"## {name}",
            f"- Equity: ${s['initial_equity']:.2f} -> ${s['final_equity']:.2f}",
            f"- Total return: {s['total_return']*100:.3f}%",
            f"- Calendar annualized/CAGR: {s['calendar_cagr']*100:.3f}%",
            f"- Realized market-close MaxDD: {s['realized_market_close_max_dd']*100:.3f}%",
            f"- Strict trades: {s['strict_trades']} (win rate {s['win_rate']*100:.2f}%)",
            f"- Turnover: ${s['strict_turnover']:.2f} = {s['turnover_multiple_initial_capital']:.2f}x initial capital",
            f"- Core PnL/trades: ${s['family_pnl'].get('core_3c',0):.2f} / {s['family_trades'].get('core_3c',0)}",
            f"- Tail PnL/trades: ${s['family_pnl'].get('tail_favorite_sub3c',0):.2f} / {s['family_trades'].get('tail_favorite_sub3c',0)}",
            f"- Markets with both strict CORE+TAIL: {s['markets_with_both_core_and_tail_strict_fill']}",
            f"- Max stake budget reached: ${s['max_stake_budget']:.2f}",
            f"- Tier coverage error: {s['tier_coverage_error']}",
            "",
        ]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print((out / "SUMMARY.md").read_text(), flush=True)
    print(json.dumps(final, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("protocol"); p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--shard", required=True); s.add_argument("--out", type=Path, required=True); s.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("aggregate"); a.add_argument("--root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "protocol":
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "FROZEN_DUAL_BANKROLL_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))
        print(json.dumps(PROTOCOL, indent=2), flush=True)
    elif args.cmd == "score":
        score_shard(args.start, args.end, args.shard, args.out, args.workers)
    else:
        aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
