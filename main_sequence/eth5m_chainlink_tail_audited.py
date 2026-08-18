from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from main_sequence import eth5m_chainlink_tail as runner
from main_sequence.qualification_stats import daily_pnl_stats


def latest_row_books(paths: list[Path]) -> pd.DataFrame:
    """Qualification-only selector: rank every capture row first, then validate ask/depth.

    A newer empty/price-only row invalidates any older executable-looking snapshot.
    This prevents borrowing stale price or stale size from the past.
    """
    con = duckdb.connect()
    lo, hi = runner.utc_ms(runner.START), runner.utc_ms(runner.END)
    target_offset = runner.DECISION_S2C * 1000
    q = f"""
    WITH src AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms,
             CAST(best_bid AS DOUBLE) AS best_bid,
             CAST(best_ask AS DOUBLE) AS best_ask,
             CAST(bid_sz AS DOUBLE) AS bid_sz,
             CAST(ask_sz AS DOUBLE) AS ask_sz,
             lower(outcome) AS outcome, slug, CAST(cond AS VARCHAR) AS cond,
             CAST(win_start AS BIGINT) AS win_start, CAST(end_ts AS BIGINT) AS end_ts
      FROM read_parquet({runner.sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{runner.ASSET}' AND lower(slug) LIKE '%-updown-5m-%'
        AND end_ts*1000 >= {lo} AND end_ts*1000 < {hi}
        AND ts_ms <= end_ts*1000 - {target_offset}
        AND ts_ms >= end_ts*1000 - {target_offset + 7000}
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY slug, outcome ORDER BY ts_ms DESC, cond DESC) AS rn
      FROM src
    )
    SELECT * FROM ranked
    WHERE rn=1
      AND best_ask IS NOT NULL AND best_ask > 0 AND best_ask < 1
      AND ask_sz IS NOT NULL AND ask_sz > 0
    ORDER BY end_ts, slug, outcome
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


runner.load_decision_books = latest_row_books


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
            m["qualification_audit"] = {
                "book_selector": "latest capture row at/before T-90 is ranked before any price/depth validation",
                "stale_price_borrowing": False,
                "stale_depth_borrowing": False,
                "gamma_terminal_resolution_required": True,
                "max_book_staleness_ms": runner.MAX_BOOK_STALENESS_MS,
                "max_chainlink_source_lag_ms": runner.MAX_CHAINLINK_LAG_MS,
            }
            manifest_path.write_text(json.dumps(m, indent=2))
