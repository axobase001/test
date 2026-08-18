from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from main_sequence import eth5m_chainlink_tail as runner
from main_sequence.qualification_stats import daily_pnl_stats


if __name__ == "__main__":
    runner.main()
    out = Path("eth5m_tail_out")
    raw_path = out / "market_rows.csv"
    surface_path = out / "surface.csv"
    manifest_path = out / "manifest.json"
    if raw_path.exists() and surface_path.exists():
        raw = pd.read_csv(raw_path)
        surf = pd.read_csv(surface_path)
        stat_rows = []
        for r in surf.itertuples(index=False):
            fair_floor = float(r.fair_floor)
            barrier_floor = float(r.barrier_bps_floor)
            z = raw[(raw["fair"] >= fair_floor) &
                    (raw["barrier_bps"] >= barrier_floor) &
                    (raw["net_settlement_edge_ps"] > 0) &
                    (raw["depth_headroom_x"] >= 1.0)].copy()
            seed_offset = int(round(fair_floor * 1000)) * 100 + int(round(barrier_floor * 10))
            st = daily_pnl_stats(z, "decision_ts_ms", "pnl_fixed5", unit="ms", seed_offset=seed_offset)
            stat_rows.append(st)
        surf["qualification_grade"] = [x["grade"] for x in stat_rows]
        surf["bootstrap_days"] = [x["days"] for x in stat_rows]
        surf["mean_daily_pnl"] = [x["mean_daily_pnl"] for x in stat_rows]
        surf["daily_pnl_ci95_low"] = [x["mean_daily_pnl_ci95"][0] for x in stat_rows]
        surf["daily_pnl_ci95_high"] = [x["mean_daily_pnl_ci95"][1] for x in stat_rows]
        surf["positive_day_rate"] = [x["positive_day_rate"] for x in stat_rows]
        surf.to_csv(surface_path, index=False)
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            m["qualification_gate"] = {
                "GREEN": "cell total PnL >0, >=20 observed trading days, >=100 trades, and day-bootstrap mean daily PnL CI95 lower bound >0",
                "YELLOW": "positive PnL but sample or CI not strong enough",
                "RED": "cell total PnL <=0",
                "selection_policy": "all 20 predeclared fair-floor x barrier-distance cells are reported and graded; no post-hoc winner is substituted for the surface",
            }
            manifest_path.write_text(json.dumps(m, indent=2))
