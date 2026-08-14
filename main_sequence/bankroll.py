from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_MS = 365.0 * 24 * 3600 * 1000


@dataclass
class Position:
    cid: str
    open_ts_ms: int
    close_ts_ms: int
    stake: float
    cost_per_share: float
    payout_per_share: float
    action: str
    source_row: int

    @property
    def shares(self) -> float:
        return self.stake / self.cost_per_share

    @property
    def payout(self) -> float:
        return self.shares * self.payout_per_share

    @property
    def pnl(self) -> float:
        return self.payout - self.stake


def target_stake(known_equity: float, initial_capital: float, base_trade: float, single_cap: float) -> float:
    """Start at base_trade; double size whenever settled accounting equity doubles."""
    if initial_capital <= 0 or base_trade <= 0 or single_cap <= 0:
        raise ValueError("capital/trade caps must be positive")
    ratio = max(float(known_equity) / float(initial_capital), 1.0)
    tier = max(int(math.floor(math.log2(ratio) + 1e-12)), 0)
    return min(float(base_trade) * (2.0 ** tier), float(single_cap))


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0, (arr - peak) / peak, 0.0)
    return float(-dd.min())


def load_fills(fills_path: Path, records_path: Path | None) -> pd.DataFrame:
    fills = pd.read_csv(fills_path).copy()
    required = {"ts_ms", "cid", "cost", "real_reward"}
    missing = required - set(fills.columns)
    if missing:
        raise ValueError(f"fills missing columns: {sorted(missing)}")
    fills["ts_ms"] = pd.to_numeric(fills["ts_ms"], errors="raise").astype(np.int64)
    fills["cid"] = fills["cid"].astype(str)
    fills["cost"] = pd.to_numeric(fills["cost"], errors="raise").astype(float)
    fills["real_reward"] = pd.to_numeric(fills["real_reward"], errors="raise").astype(float)

    if "close_ts_ms" in fills.columns:
        fills["close_ts_ms"] = pd.to_numeric(fills["close_ts_ms"], errors="raise").astype(np.int64)
    elif "close_ts" in fills.columns:
        vals = pd.to_numeric(fills["close_ts"], errors="raise").astype(np.int64)
        fills["close_ts_ms"] = np.where(vals < 10**12, vals * 1000, vals).astype(np.int64)
    else:
        if records_path is None:
            raise ValueError("fills do not contain close_ts[_ms]; --records is required")
        rec = pd.read_csv(records_path, usecols=["ts_ms", "cid", "s2c"]).copy()
        rec["ts_ms"] = pd.to_numeric(rec["ts_ms"], errors="raise").astype(np.int64)
        rec["cid"] = rec["cid"].astype(str)
        rec["s2c"] = pd.to_numeric(rec["s2c"], errors="raise").astype(float)
        rec = rec.sort_values(["ts_ms", "cid"]).drop_duplicates(["ts_ms", "cid"], keep="last")
        fills = fills.merge(rec, on=["ts_ms", "cid"], how="left", validate="many_to_one")
        if fills["s2c"].isna().any():
            bad = fills.loc[fills["s2c"].isna(), ["ts_ms", "cid"]].head(5).to_dict("records")
            raise ValueError(f"could not resolve close time for fills, examples={bad}")
        fills["close_ts_ms"] = fills["ts_ms"].astype(np.int64) + np.rint(fills["s2c"] * 1000.0).astype(np.int64)

    if (fills["close_ts_ms"] <= fills["ts_ms"]).any():
        raise ValueError("non-causal/invalid close time: close_ts_ms must be after decision ts_ms")
    if (~np.isfinite(fills["cost"])).any() or (fills["cost"] <= 0).any():
        raise ValueError("invalid non-positive/non-finite cost")

    fills["payout_per_share"] = fills["real_reward"] + fills["cost"]
    d0 = np.abs(fills["payout_per_share"])
    d1 = np.abs(fills["payout_per_share"] - 1.0)
    if np.minimum(d0, d1).max() > 1e-5:
        bad = fills.loc[np.minimum(d0, d1) > 1e-5, ["ts_ms", "cid", "payout_per_share"]].head(5)
        raise ValueError(f"non-binary realized payout found:\n{bad}")
    fills["payout_per_share"] = (fills["payout_per_share"] >= 0.5).astype(float)
    if "action" not in fills.columns:
        fills["action"] = "unknown"
    return fills.sort_values(["ts_ms", "close_ts_ms", "cid"]).reset_index(drop=True)


def replay(fills: pd.DataFrame, initial_capital: float = 50.0, base_trade: float = 5.0,
           single_cap: float = 100.0, market_cap: float = 200.0):
    if initial_capital <= 0 or market_cap <= 0:
        raise ValueError("capital and market cap must be positive")
    cash = float(initial_capital)
    active: list[Position] = []
    trade_rows, curve = [], []
    skipped_cash = skipped_market_cap = 0
    max_locked = max_market_seen = 0.0
    first_ts = int(fills["ts_ms"].min()) if len(fills) else None
    last_ts = first_ts

    def locked() -> float:
        return float(sum(p.stake for p in active))

    def known_equity() -> float:
        # Critical causality rule: unresolved contracts stay at cost basis. No future outcome or mark is peeked.
        return cash + locked()

    def market_exposure(cid: str) -> float:
        return float(sum(p.stake for p in active if p.cid == cid))

    def mark(ts: int, event: str) -> None:
        nonlocal max_locked, max_market_seen, last_ts
        lck = locked(); eq = cash + lck
        by_market = {}
        for p in active:
            by_market[p.cid] = by_market.get(p.cid, 0.0) + p.stake
        mm = max(by_market.values(), default=0.0)
        max_locked = max(max_locked, lck); max_market_seen = max(max_market_seen, mm)
        last_ts = max(int(ts), int(last_ts or ts))
        curve.append({"ts_ms": int(ts), "event": event, "cash": cash, "locked_cost": lck,
                      "known_equity": eq, "active_positions": len(active),
                      "max_active_market_exposure": mm,
                      "target_stake": target_stake(eq, initial_capital, base_trade, single_cap)})

    def settle_until(ts: int) -> None:
        nonlocal cash, active
        due = sorted((p for p in active if p.close_ts_ms <= ts),
                     key=lambda p: (p.close_ts_ms, p.open_ts_ms, p.source_row))
        for p in due:
            cash += p.payout
            active.remove(p)
            mark(p.close_ts_ms, "settle")

    if len(fills):
        mark(first_ts, "start")

    for idx, r in fills.iterrows():
        ts = int(r.ts_ms); settle_until(ts)
        eq = known_equity()
        stake = target_stake(eq, initial_capital, base_trade, single_cap)
        cid = str(r.cid); mexp = market_exposure(cid)
        reason = None
        if mexp + stake > market_cap + 1e-9:
            skipped_market_cap += 1; reason = "market_cap"
        elif cash + 1e-9 < stake:
            skipped_cash += 1; reason = "cash"
        if reason is not None:
            trade_rows.append({"source_row": int(idx), "ts_ms": ts, "close_ts_ms": int(r.close_ts_ms),
                               "cid": cid, "action": str(r.action), "status": f"skipped_{reason}",
                               "known_equity_before": eq, "cash_before": cash, "target_stake": stake,
                               "stake": 0.0, "cost_per_share": float(r.cost),
                               "payout_per_share": float(r.payout_per_share), "shares": 0.0,
                               "payout": 0.0, "pnl": 0.0})
            mark(ts, f"skip_{reason}"); continue

        p = Position(cid, ts, int(r.close_ts_ms), float(stake), float(r.cost),
                     float(r.payout_per_share), str(r.action), int(idx))
        cash_before = cash; cash -= p.stake; active.append(p)
        trade_rows.append({"source_row": int(idx), "ts_ms": ts, "close_ts_ms": p.close_ts_ms,
                           "cid": cid, "action": p.action, "status": "taken",
                           "known_equity_before": eq, "cash_before": cash_before, "target_stake": stake,
                           "stake": p.stake, "cost_per_share": p.cost_per_share,
                           "payout_per_share": p.payout_per_share, "shares": p.shares,
                           "payout": p.payout, "pnl": p.pnl})
        mark(ts, "open")

    if active:
        settle_until(max(p.close_ts_ms for p in active) + 1)
    if active:
        raise AssertionError("positions remained after final settlement")

    final_capital = float(cash)
    curve_df = pd.DataFrame(curve); trades_df = pd.DataFrame(trade_rows)
    taken = trades_df[trades_df["status"] == "taken"] if len(trades_df) else trades_df
    elapsed_ms = max((int(last_ts) - int(first_ts)), 0) if first_ts is not None else 0
    elapsed_days = elapsed_ms / (24 * 3600 * 1000); years = elapsed_ms / YEAR_MS
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if years > 0 and final_capital > 0 else None
    tier_counts = {f"{float(k):g}": int(v) for k, v in taken["stake"].round(10).value_counts().sort_index().items()} if len(taken) else {}
    summary = {
        "initial_capital": float(initial_capital), "final_capital": final_capital,
        "multiple": final_capital / initial_capital, "total_return": final_capital / initial_capital - 1.0,
        "elapsed_days": elapsed_days, "cagr": cagr,
        "cagr_note": "mechanical annualization of this historical interval; not a forecast",
        "max_drawdown_realized_cost_basis": max_drawdown(curve_df["known_equity"].tolist()) if len(curve_df) else 0.0,
        "candidate_fills": int(len(fills)), "trades_taken": int(len(taken)),
        "wins": int((taken["payout_per_share"] > 0.5).sum()) if len(taken) else 0,
        "losses": int((taken["payout_per_share"] < 0.5).sum()) if len(taken) else 0,
        "skipped_cash": int(skipped_cash), "skipped_market_cap": int(skipped_market_cap),
        "max_locked_cost": float(max_locked), "max_single_market_exposure": float(max_market_seen),
        "stake_counts": tier_counts,
        "rules": {"initial_capital": float(initial_capital), "base_trade": float(base_trade),
                  "double_on_each_equity_multiple_of_2": True, "single_trade_cap": float(single_cap),
                  "single_market_window_exposure_cap": float(market_cap), "leverage": False,
                  "partial_fill_for_cash_or_cap": False,
                  "equity_for_sizing": "cash + unsettled positions at cost basis; only settled PnL changes tier"},
    }
    return summary, trades_df, curve_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", type=Path, required=True); ap.add_argument("--records", type=Path)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--initial", type=float, default=50.0)
    ap.add_argument("--base-trade", type=float, default=5.0); ap.add_argument("--single-cap", type=float, default=100.0)
    ap.add_argument("--market-cap", type=float, default=200.0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    fills = load_fills(args.fills, args.records)
    summary, trades, curve = replay(fills, args.initial, args.base_trade, args.single_cap, args.market_cap)
    trades.to_csv(args.out / "bankroll_trades.csv", index=False)
    curve.to_csv(args.out / "bankroll_curve.csv", index=False)
    (args.out / "bankroll_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
