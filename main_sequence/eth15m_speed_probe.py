from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# Importing safe installs the frozen exact-price same-second tape scorer and
# causal Deribit temporal-universe anchor into r.
from main_sequence import eth15m_conservative_replay_safe as safe

r = safe.r


def bankroll(trades: pd.DataFrame, start: str, stake_mode: str) -> dict:
    eq = 50.0
    peak = eq
    mdd = 0.0
    hits = {}
    first = pd.Timestamp(start, tz="UTC")
    taken = 0
    for _, row in trades.sort_values("decision", kind="mergesort").iterrows():
        if stake_mode == "fixed5":
            stake = min(5.0, eq)
        elif stake_mode == "fixed10":
            stake = min(10.0, eq)
        elif stake_mode == "20pct_cap10":
            stake = min(0.20 * eq, 10.0, eq)
        else:
            raise ValueError(stake_mode)
        if stake <= 0:
            break
        cost = float(row["cost"])
        pnl = float(row["pnl"]) * (stake / cost)
        eq += pnl
        taken += 1
        peak = max(peak, eq)
        if peak > 0:
            mdd = min(mdd, eq / peak - 1.0)
        ts = pd.to_datetime(int(row["decision"]), unit="s", utc=True)
        for target in (100.0, 150.0, 500.0):
            if str(int(target)) not in hits and eq >= target:
                hits[str(int(target))] = {
                    "timestamp": ts.isoformat(),
                    "days_from_window_start": float((ts - first).total_seconds() / 86400.0),
                    "trade_number": taken,
                }
        if eq <= 0:
            break
    return {
        "mode": stake_mode,
        "initial": 50.0,
        "final": float(eq),
        "mdd_pct": float(mdd * 100.0),
        "trades_taken": int(taken),
        "milestones": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=float, required=True)
    ap.add_argument("--edge", type=float, required=True)
    ap.add_argument("--ask", type=float, default=0.20)
    ap.add_argument("--volume", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    # The experiment deliberately varies only admission gates. Fair construction,
    # fees, decision window, exit band, market mapping, and causal anchors remain frozen.
    r.RAW_GAP_FLOOR = float(a.raw)
    r.EDGE_FLOOR = float(a.edge)
    r.ASK_FLOOR = float(a.ask)
    r.VOLUME_MULT = float(a.volume)
    safe.r.RAW_GAP_FLOOR = float(a.raw)
    safe.r.EDGE_FLOOR = float(a.edge)
    safe.r.ASK_FLOOR = float(a.ask)
    safe.r.VOLUME_MULT = float(a.volume)

    r.run(a.start, a.end, a.out, a.workers)
    tp = a.out / "trades.csv"
    trades = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
    result = {
        "period": [a.start, a.end],
        "gates": {
            "raw_gap_floor": a.raw,
            "net_edge_floor": a.edge,
            "ask_floor": a.ask,
            "volume_mult": a.volume,
        },
        "execution_note": "fixed10 is historically size-witnessed only because this sweep keeps exact-price same-second volume_mult=2.0 relative to the frozen fixed-$5 qty",
        "trades": int(len(trades)),
        "fixed5_pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
        "bankroll": [bankroll(trades, a.start, m) for m in ("fixed5", "fixed10", "20pct_cap10")] if len(trades) else [],
    }
    (a.out / "speed_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ETH15M_SPEED_FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
