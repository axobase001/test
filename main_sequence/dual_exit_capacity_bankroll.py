from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence.bankroll import target_stake, max_drawdown

INITIAL_CAPITAL = 50.0
BASE_TRADE = 5.0
SINGLE_CAP = 100.0
MARKET_CAP = 200.0
SIZE_REF = 5.0
YEAR_S = 365.0 * 24.0 * 3600.0


@dataclass
class Position:
    cid: str
    slug: str
    entry_ts: int
    exit_ts: int
    stake: float
    cost_per_share: float
    proceeds_per_share: float
    regime: str

    @property
    def shares(self) -> float:
        return self.stake / self.cost_per_share

    @property
    def proceeds(self) -> float:
        return self.shares * self.proceeds_per_share

    @property
    def pnl(self) -> float:
        return self.proceeds - self.stake


def load_records(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/dual_exit_records.csv"))
    if not files:
        raise RuntimeError(f"no dual_exit_records.csv under {root}")
    frames = [pd.read_csv(p) for p in files]
    x = pd.concat(frames, ignore_index=True)
    if len(x) != 7661:
        raise RuntimeError(f"expected 7661 frozen signals, got {len(x)}")
    if x["slug"].duplicated().any():
        raise RuntimeError("duplicate slug in combined corrected replay")
    return x


def layer_candidates(records: pd.DataFrame, layer: str) -> pd.DataFrame:
    x = records.copy()
    if layer == "witness":
        need = ["witness_cost", "witness_reward", "witness_exit_time"]
        x = x.dropna(subset=need).copy()
        x["entry_ts"] = pd.to_numeric(x["decision"], errors="raise").astype(np.int64)
        x["exit_ts"] = pd.to_numeric(x["witness_exit_time"], errors="raise").astype(np.int64)
        x["cost_total_ref"] = pd.to_numeric(x["witness_cost"], errors="raise").astype(float)
        x["reward_total_ref"] = pd.to_numeric(x["witness_reward"], errors="raise").astype(float)
    elif layer == "strict5":
        need = ["strict_cost", "strict_reward", "strict_entry_time", "strict_exit_time"]
        x = x.dropna(subset=need).copy()
        x["entry_ts"] = pd.to_numeric(x["strict_entry_time"], errors="raise").astype(np.int64)
        x["exit_ts"] = pd.to_numeric(x["strict_exit_time"], errors="raise").astype(np.int64)
        x["cost_total_ref"] = pd.to_numeric(x["strict_cost"], errors="raise").astype(float)
        x["reward_total_ref"] = pd.to_numeric(x["strict_reward"], errors="raise").astype(float)
    else:
        raise ValueError(layer)

    if (x["exit_ts"] <= x["entry_ts"]).any():
        raise RuntimeError(f"noncausal {layer} exit")
    if (x["cost_total_ref"] <= 0).any():
        raise RuntimeError(f"nonpositive {layer} cost")
    x["cost_per_share"] = x["cost_total_ref"] / SIZE_REF
    x["proceeds_per_share"] = (x["cost_total_ref"] + x["reward_total_ref"]) / SIZE_REF
    if (~np.isfinite(x[["cost_per_share", "proceeds_per_share"]].to_numpy(float))).any():
        raise RuntimeError("nonfinite candidate economics")
    return x.sort_values(["entry_ts", "slug"], kind="mergesort").reset_index(drop=True)


def replay(candidates: pd.DataFrame, layer: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cash = INITIAL_CAPITAL
    active: list[Position] = []
    curve: list[dict] = []
    trades: list[dict] = []
    max_locked = 0.0
    max_market = 0.0
    skipped_cash = 0
    skipped_market_cap = 0
    first_ts = int(candidates["entry_ts"].min()) if len(candidates) else None
    last_ts = first_ts
    cap_reached_ts = None

    def locked() -> float:
        return float(sum(p.stake for p in active))

    def known_equity() -> float:
        # Preserve the original causal accounting convention: unresolved positions remain at cost basis.
        return float(cash + locked())

    def market_exposure(cid: str) -> float:
        return float(sum(p.stake for p in active if p.cid == cid))

    def mark(ts: int, event: str) -> None:
        nonlocal max_locked, max_market, last_ts, cap_reached_ts
        eq = known_equity()
        tgt = target_stake(eq, INITIAL_CAPITAL, BASE_TRADE, SINGLE_CAP)
        if tgt >= SINGLE_CAP - 1e-12 and cap_reached_ts is None:
            cap_reached_ts = int(ts)
        by_market: dict[str, float] = {}
        for p in active:
            by_market[p.cid] = by_market.get(p.cid, 0.0) + p.stake
        mm = max(by_market.values(), default=0.0)
        max_locked = max(max_locked, locked())
        max_market = max(max_market, mm)
        last_ts = max(int(ts), int(last_ts or ts))
        curve.append({
            "ts": int(ts), "event": event, "cash": float(cash), "locked_cost": locked(),
            "known_equity": eq, "active": len(active), "target_stake": tgt,
            "max_active_market_exposure": mm,
        })

    def settle_until(ts: int) -> None:
        nonlocal cash, active
        due = sorted((p for p in active if p.exit_ts <= ts), key=lambda p: (p.exit_ts, p.entry_ts, p.slug))
        for p in due:
            cash += p.proceeds
            active.remove(p)
            trades.append({
                "slug": p.slug, "cid": p.cid, "regime": p.regime, "event": "exit",
                "entry_ts": p.entry_ts, "exit_ts": p.exit_ts, "stake": p.stake,
                "shares": p.shares, "cost_per_share": p.cost_per_share,
                "proceeds_per_share": p.proceeds_per_share, "proceeds": p.proceeds, "pnl": p.pnl,
            })
            mark(p.exit_ts, "exit")

    if len(candidates):
        mark(first_ts, "start")

    opened = 0
    stake_hist: list[float] = []
    by_regime_opened: dict[str, int] = {}
    by_regime_pnl: dict[str, float] = {}

    for r in candidates.itertuples(index=False):
        ts = int(r.entry_ts)
        settle_until(ts)
        eq = known_equity()
        stake = target_stake(eq, INITIAL_CAPITAL, BASE_TRADE, SINGLE_CAP)
        cid = str(r.condition_id)
        mexp = market_exposure(cid)
        reason = None
        if mexp + stake > MARKET_CAP + 1e-9:
            skipped_market_cap += 1
            reason = "market_cap"
        elif cash + 1e-9 < stake:
            skipped_cash += 1
            reason = "cash"
        if reason is not None:
            trades.append({
                "slug": str(r.slug), "cid": cid, "regime": str(r.regime), "event": f"skip_{reason}",
                "entry_ts": ts, "exit_ts": int(r.exit_ts), "stake": 0.0, "shares": 0.0,
                "cost_per_share": float(r.cost_per_share), "proceeds_per_share": float(r.proceeds_per_share),
                "proceeds": 0.0, "pnl": 0.0,
            })
            mark(ts, f"skip_{reason}")
            continue

        p = Position(
            cid=cid, slug=str(r.slug), entry_ts=ts, exit_ts=int(r.exit_ts), stake=float(stake),
            cost_per_share=float(r.cost_per_share), proceeds_per_share=float(r.proceeds_per_share),
            regime=str(r.regime),
        )
        cash -= p.stake
        active.append(p)
        opened += 1
        stake_hist.append(float(stake))
        by_regime_opened[p.regime] = by_regime_opened.get(p.regime, 0) + 1
        mark(ts, "open")

    if active:
        settle_until(max(p.exit_ts for p in active) + 1)
    if active:
        raise AssertionError("positions remain")

    # Collect realized PnL by regime from exit rows.
    tdf = pd.DataFrame(trades)
    exits = tdf[tdf["event"] == "exit"] if len(tdf) else tdf
    if len(exits):
        for regime, g in exits.groupby("regime"):
            by_regime_pnl[str(regime)] = float(g["pnl"].sum())

    final = float(cash)
    cdf = pd.DataFrame(curve)
    elapsed_s = max(int(last_ts) - int(first_ts), 0) if first_ts is not None else 0
    years = elapsed_s / YEAR_S if elapsed_s > 0 else 0.0
    cagr = (final / INITIAL_CAPITAL) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else None
    elapsed_days = elapsed_s / 86400.0
    simple_ann = (final / INITIAL_CAPITAL - 1.0) * 365.0 / elapsed_days if elapsed_days > 0 else None
    stake_counts = {f"{k:g}": int(v) for k, v in pd.Series(stake_hist).value_counts().sort_index().items()} if stake_hist else {}

    summary = {
        "layer": layer,
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": final,
        "profit": final - INITIAL_CAPITAL,
        "multiple": final / INITIAL_CAPITAL,
        "total_return": final / INITIAL_CAPITAL - 1.0,
        "elapsed_days": elapsed_days,
        "simple_annualized": simple_ann,
        "cagr": cagr,
        "candidate_signals": int(len(candidates)),
        "trades_opened": int(opened),
        "skipped_cash": int(skipped_cash),
        "skipped_market_cap": int(skipped_market_cap),
        "max_locked_cost": float(max_locked),
        "max_single_market_exposure": float(max_market),
        "max_drawdown_realized_cost_basis": max_drawdown(cdf["known_equity"].tolist()) if len(cdf) else 0.0,
        "first_reached_100_trade_cap_ts": cap_reached_ts,
        "first_reached_100_trade_cap_utc": (pd.to_datetime(cap_reached_ts, unit="s", utc=True).isoformat() if cap_reached_ts is not None else None),
        "stake_counts": stake_counts,
        "opened_by_regime": by_regime_opened,
        "realized_pnl_by_regime": by_regime_pnl,
        "rules": {
            "initial_capital": INITIAL_CAPITAL,
            "base_trade": BASE_TRADE,
            "double_trade_only_when_realized_known_equity_doubles": True,
            "trade_tiers": [5.0, 10.0, 20.0, 40.0, 80.0, 100.0],
            "single_trade_cap": SINGLE_CAP,
            "single_market_window_cap": MARKET_CAP,
            "leverage": False,
            "partial_fill_for_cash": False,
            "equity_for_sizing": "cash + unresolved positions at cost basis; only realized exits change tier",
            "capacity_note": "Hard $100 stake cap approximates founder-described PM 15m opportunity capacity. Current witness replay does not prove $100 of historical depth at every signal; full-depth replay is still required for queue/depth confirmation.",
        },
    }
    return summary, tdf, cdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.root)
    final = {}
    for layer in ["witness", "strict5"]:
        c = layer_candidates(records, layer)
        summary, trades, curve = replay(c, layer)
        final[layer] = summary
        trades.to_csv(args.out / f"{layer}_bankroll_trades.csv", index=False)
        curve.to_csv(args.out / f"{layer}_bankroll_curve.csv", index=False)
        (args.out / f"{layer}_bankroll_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
