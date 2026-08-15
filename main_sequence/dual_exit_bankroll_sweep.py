from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from main_sequence import dual_exit_capacity_bankroll as b

INITIALS = [10.0, 20.0, 50.0, 75.0, 100.0]
SINGLE_CAPS = [100.0, 200.0, 300.0]


def run_grid(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = b.load_records(root)
    results: dict[str, list[dict]] = {"witness": [], "strict5": []}

    for layer in ["witness", "strict5"]:
        candidates = b.layer_candidates(records, layer)
        for initial in INITIALS:
            base_trade = initial * 0.10
            for single_cap in SINGLE_CAPS:
                market_cap = single_cap * 2.0
                b.INITIAL_CAPITAL = initial
                b.BASE_TRADE = base_trade
                b.SINGLE_CAP = single_cap
                b.MARKET_CAP = market_cap

                summary, _, curve = b.replay(candidates, layer)
                cap_ts = summary.get("first_reached_100_trade_cap_ts")
                first_ts = int(candidates["entry_ts"].min()) if len(candidates) else None
                cap_days = None if cap_ts is None or first_ts is None else (int(cap_ts) - first_ts) / 86400.0
                linear_runrate = summary["profit"] * 365.0 / summary["elapsed_days"] if summary["elapsed_days"] else None
                post_cap_days = None
                post_cap_profit = None
                post_cap_runrate = None
                if cap_ts is not None and len(curve):
                    c = curve.sort_values(["ts"], kind="mergesort")
                    pre = c[c["ts"] <= int(cap_ts)]
                    if len(pre):
                        eq_at_cap = float(pre.iloc[-1]["known_equity"])
                        post_cap_profit = float(summary["final_capital"] - eq_at_cap)
                        last_ts = int(c["ts"].max())
                        post_cap_days = max((last_ts - int(cap_ts)) / 86400.0, 0.0)
                        if post_cap_days > 0:
                            post_cap_runrate = post_cap_profit * 365.0 / post_cap_days

                row = {
                    "layer": layer,
                    "initial_capital": initial,
                    "base_trade": base_trade,
                    "single_trade_cap": single_cap,
                    "market_cap": market_cap,
                    "final_capital_93d": float(summary["final_capital"]),
                    "profit_93d": float(summary["profit"]),
                    "multiple_93d": float(summary["multiple"]),
                    "return_93d": float(summary["total_return"]),
                    "trades_opened": int(summary["trades_opened"]),
                    "skipped_cash": int(summary["skipped_cash"]),
                    "max_dd_realized_cost_basis": float(summary["max_drawdown_realized_cost_basis"]),
                    "cap_reached": cap_ts is not None,
                    "days_to_cap": cap_days,
                    "cap_reached_utc": summary.get("first_reached_100_trade_cap_utc"),
                    "stake_counts": summary.get("stake_counts", {}),
                    "linearized_full_period_pnl_per_year": linear_runrate,
                    "post_cap_profit": post_cap_profit,
                    "post_cap_days": post_cap_days,
                    "post_cap_pnl_runrate_per_year": post_cap_runrate,
                    "large_convergence_pnl": float(summary.get("realized_pnl_by_regime", {}).get("large_convergence", 0.0)),
                    "small_settlement_pnl": float(summary.get("realized_pnl_by_regime", {}).get("small_settlement", 0.0)),
                }
                results[layer].append(row)
                print("SWEEP", json.dumps(row, sort_keys=True), flush=True)

    for layer, rows in results.items():
        pd.DataFrame(rows).to_csv(out / f"{layer}_sweep.csv", index=False)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run_grid(args.root, args.out)
    print("FINAL_SWEEP", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
