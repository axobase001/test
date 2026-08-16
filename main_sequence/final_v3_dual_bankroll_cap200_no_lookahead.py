from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from main_sequence import final_v3_dual_bankroll_no_lookahead as engine

# User-added portfolio constraint, frozen before this replay's results are opened.
MAX_MARKET_CAP_USD = 200.0

# Once the nominal power-of-two ticket exceeds $200, the market-level cap makes
# larger execution tiers unnecessary. Keep all smaller original tiers plus the
# exact $200 cap tier so tape capacity is still checked at the real dollar size.
SCORED_BUDGETS = tuple(sorted(set(
    [float(x) for x in engine.STAKE_TIERS if float(x) < MAX_MARKET_CAP_USD]
    + [MAX_MARKET_CAP_USD]
)))
engine.STAKE_TIERS = SCORED_BUDGETS

engine.PROTOCOL["name"] = "Main Sequence V3 dual-sleeve max-turnover bankroll + $200 combined market cap / 2026-08-16 freeze"
engine.PROTOCOL["bankroll"]["max_single_market_open_capital_usd"] = MAX_MARKET_CAP_USD
engine.PROTOCOL["bankroll"]["market_cap_scope"] = (
    "Combined open entry capital across CORE+TAIL in the same BTC15m market. The cap is $200 total, not $200 per sleeve."
)
engine.PROTOCOL["bankroll"]["market_cap_execution"] = (
    "Causal first-fill-first-occupancy. At each strict fill timestamp, positions from the same market whose exits are already realized are released; "
    "the new fill may use only the remaining portion of the $200 cap. If a full-size candidate is tape-executable but only part of its capital fits, "
    "the fill is clipped to the remaining cap and its already-observed full-fill average prices are used as a conservative linear lower-bound for the smaller executable quantity."
)
engine.PROTOCOL["bankroll"]["nominal_ticket_rule"] = (
    "$5 at $50 equity; nominal ticket doubles/halves at each power-of-two equity boundary. Actual ticket is min(nominal ticket, $200 market cap availability)."
)
engine.PROTOCOL["bankroll"]["stake_tiers_scored_usd"] = list(SCORED_BUDGETS)
engine.PROTOCOL["anti_lookahead"].extend([
    "The $200 cap is enforced only from information known at each strict fill timestamp: already-realized exits and already-occurring fills in that same market.",
    "CORE and TAIL are never pre-reserved equal slices based on knowing that both will signal later; occupancy is first-fill-first-served, with deterministic tie-breaking.",
    "A partial cap-clipped fill is allowed only when the larger frozen order was itself strict-tape executable; therefore the smaller quantity is executable on the same tape and cannot create a fill from future knowledge.",
])


def nominal_stake_for_equity(equity: float) -> tuple[float | None, float | None, int | None]:
    if not (equity > 0 and math.isfinite(equity)):
        return None, None, None
    exp = int(math.floor(math.log(equity / engine.INITIAL_EQUITY, 2.0)))
    # Preserve the prior symmetric downside de-risk floor. Upside is unbounded in
    # nominal terms because the $200 market cap, not the score tier, becomes binding.
    if exp < engine.TIER_EXP_MIN:
        return None, None, exp
    nominal = float(engine.BASE_STAKE * (2.0 ** exp))
    executable_budget = float(min(nominal, MAX_MARKET_CAP_USD))
    return nominal, executable_budget, exp


def simulate_window_cap200(records: pd.DataFrame, start: str, end: str):
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    x = records[(records["start"] >= start_ts) & (records["start"] < end_ts)].copy()
    x["stake_budget"] = pd.to_numeric(x["stake_budget"], errors="coerce")

    equity = float(engine.INITIAL_EQUITY)
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
    cap_partial_trades = 0
    cap_blocked_trades = 0
    max_open_market_capital = 0.0
    coverage_error = None
    equity_path = [{"time": start_ts, "equity": equity, "event": "start"}]

    # Cross-market overlap is absent under the frozen 15m/60-600s protocol: all
    # positions in market M exit no later than M close, while the next market's
    # decision window starts five minutes after that close. Thus equity is fully
    # realized before sizing the next market.
    for market_start, g in x.groupby("start", sort=True):
        nominal_stake, order_budget, exp = nominal_stake_for_equity(equity)
        if order_budget is None:
            coverage_error = {"market_start": int(market_start), "equity": equity, "required_exponent": exp}
            break

        tier_hist[str(order_budget)] = tier_hist.get(str(order_budget), 0) + 1
        z = g[np.isclose(g["stake_budget"].astype(float), order_budget, rtol=0, atol=1e-10)].copy()
        if z.empty:
            continue

        # Exactly one pre-frozen signal per family at this executable dollar tier.
        selected_signal_count += int(len(z))
        selected_no_fill_count += int(z["strict_cost"].isna().sum())
        fills = z[z["strict_cost"].notna()].copy()
        if fills.empty:
            continue

        # Causal fill ordering. No family receives a reserved half-cap merely
        # because we know ex post that another family also has a signal.
        fills = fills.sort_values(["strict_entry_time", "decision", "signal_family"], kind="mergesort")
        active: list[dict] = []
        market_pnl = 0.0
        families_allocated: set[str] = set()

        for r in fills.itertuples(index=False):
            entry_time = int(r.strict_entry_time)
            exit_time = int(r.strict_exit_time)

            # Release only positions whose exit is already realized at this time.
            active = [a for a in active if int(a["exit_time"]) > entry_time]
            open_capital = float(sum(float(a["entry_capital"]) for a in active))
            available_cap = max(0.0, MAX_MARKET_CAP_USD - open_capital)
            full_cost = float(r.strict_cost)
            if full_cost <= 0 or available_cap <= 1e-12:
                cap_blocked_trades += 1
                continue

            # The full order is already proven strict executable. If the cap has
            # less room, a smaller quantity is necessarily executable at no worse
            # prices; scaling the full-fill economics is conservative.
            scale = min(1.0, available_cap / full_cost)
            if scale <= 1e-12:
                cap_blocked_trades += 1
                continue
            if scale < 1.0 - 1e-12:
                cap_partial_trades += 1

            actual_cost = full_cost * scale
            actual_pnl = float(r.strict_reward) * scale
            actual_qty = float(r.qty_target) * scale
            allocated_budget = float(order_budget) * scale

            active.append({"exit_time": exit_time, "entry_capital": actual_cost})
            open_after = float(sum(float(a["entry_capital"]) for a in active))
            if open_after > MAX_MARKET_CAP_USD + 1e-8:
                raise RuntimeError(f"market cap violation {open_after} > {MAX_MARKET_CAP_USD} at {r.slug}")
            max_open_market_capital = max(max_open_market_capital, open_after)

            turnover += actual_cost
            pnl += actual_pnl
            market_pnl += actual_pnl
            fam = str(r.signal_family)
            families_allocated.add(fam)
            family_pnl[fam] = family_pnl.get(fam, 0.0) + actual_pnl
            family_trades[fam] = family_trades.get(fam, 0) + 1
            trades.append({
                "window_start": start, "window_end": end, "market_start": int(r.start), "slug": r.slug,
                "family": fam, "side": r.side,
                "nominal_stake_before_market_cap": float(nominal_stake),
                "order_budget_before_concurrency_clip": float(order_budget),
                "allocated_budget": allocated_budget, "cap_scale": scale,
                "qty": actual_qty, "entry_cost": actual_cost, "pnl": actual_pnl,
                "won": bool(r.won), "entry_time": entry_time, "exit_time": exit_time,
                "exit_kind": r.strict_exit_kind, "equity_before_market": equity,
                "market_open_capital_after_fill": open_after,
            })

        if len(families_allocated) >= 2:
            both_market_count += 1

        equity += market_pnl
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) if peak > 0 else -1.0
        max_dd = min(max_dd, dd)
        equity_path.append({"time": int(market_start) + 900, "equity": equity, "event": "market_close"})

    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    total_return = equity / engine.INITIAL_EQUITY - 1.0
    cagr = (equity / engine.INITIAL_EQUITY) ** (365.0 / days) - 1.0 if equity > 0 and days > 0 else math.nan
    tdf = pd.DataFrame(trades)
    wins = int((tdf["pnl"] > 0).sum()) if len(tdf) else 0
    losses = int((tdf["pnl"] < 0).sum()) if len(tdf) else 0
    max_nominal_stake = float(tdf["nominal_stake_before_market_cap"].max()) if len(tdf) else engine.BASE_STAKE
    max_allocated_budget = float(tdf["allocated_budget"].max()) if len(tdf) else 0.0

    summary = {
        "period": [start, end], "calendar_days": days,
        "initial_equity": engine.INITIAL_EQUITY, "final_equity": equity,
        "total_return": total_return, "calendar_cagr": cagr,
        "realized_market_close_max_dd": max_dd,
        "strict_trades": int(len(tdf)), "wins": wins, "losses": losses,
        "win_rate": float(wins / len(tdf)) if len(tdf) else math.nan,
        "strict_turnover": turnover,
        "turnover_multiple_initial_capital": turnover / engine.INITIAL_EQUITY,
        "annualized_turnover_multiple_initial_capital": (turnover / engine.INITIAL_EQUITY) * (365.0 / days),
        "net_pnl": pnl,
        "max_stake_budget": max_allocated_budget,
        "max_nominal_stake_before_market_cap": max_nominal_stake,
        "max_single_market_open_capital_observed": max_open_market_capital,
        "max_single_market_open_capital_limit": MAX_MARKET_CAP_USD,
        "stake_tier_market_counts": tier_hist,
        "family_pnl": family_pnl, "family_trades": family_trades,
        "markets_with_both_core_and_tail_strict_fill": both_market_count,
        "cap_partial_trades": cap_partial_trades,
        "cap_blocked_trades": cap_blocked_trades,
        "selected_family_signals_at_dynamic_tier": selected_signal_count,
        "selected_signals_without_strict_fill": selected_no_fill_count,
        "tier_coverage_error": coverage_error,
    }
    return summary, tdf, pd.DataFrame(equity_path)


# Patch the imported engine so score/aggregate/main continue to use the exact
# same causal replay implementation, with only the user-added cap semantics.
engine.simulate_window = simulate_window_cap200


if __name__ == "__main__":
    engine.main()
