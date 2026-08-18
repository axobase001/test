from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from main_sequence import eth5m_chainlink_tail as runner
from main_sequence.qualification_stats import daily_pnl_stats


_SELECTED_PATHS: dict[str, list[Path]] = {}


def boundary_safe_list_selected_files(revision: str, table: str, start: str, end: str, cache: Path) -> list[Path]:
    """Select rotation increments by rotation timestamp, plus the first file after END.

    Daily filenames are rotation timestamps, not calendar-day partitions. The first
    rotation after an end-exclusive boundary can contain rows from immediately before
    that boundary, so it must be downloaded and the parquet rows themselves must do
    the final time filtering.
    """
    api = HfApi()
    files = api.list_repo_files(runner.HF_REPO, repo_type="dataset", revision=revision)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    selected: list[str] = []

    root = f"{table}.parquet"
    if date.fromisoformat(start) <= date(2026, 6, 15) and date.fromisoformat(end) > date(2026, 6, 4) and root in files:
        selected.append(root)

    rotations: list[tuple[pd.Timestamp, str]] = []
    pat = re.compile(rf"^daily/{re.escape(table)}/(20\d\d-\d\d-\d\dT\d{{6}}Z)\.parquet$")
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        stamp = pd.to_datetime(m.group(1), format="%Y-%m-%dT%H%M%SZ", utc=True)
        rotations.append((stamp, f))
    rotations.sort(key=lambda x: x[0])

    # A rotation at/after START may contain rows since the previous rotation.
    selected.extend(f for stamp, f in rotations if start_ts <= stamp < end_ts)
    post_end = next((f for stamp, f in rotations if stamp >= end_ts), None)
    if post_end is not None:
        selected.append(post_end)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError(f"no {table} source files overlap requested window {start}..{end} at {revision}")

    paths = [Path(hf_hub_download(
        runner.HF_REPO, f, repo_type="dataset", revision=revision,
        cache_dir=str(cache / "hf"),
    )) for f in selected]
    _SELECTED_PATHS[table] = paths
    return paths


def latest_row_books(paths: list[Path]) -> pd.DataFrame:
    """Rank every capture row first, then validate ask/depth on the latest row only."""
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


def table_coverage(paths: list[Path], table: str) -> dict:
    if not paths:
        return {
            "table": table, "rows_in_requested_window": 0,
            "first_ts_ms": None, "last_ts_ms": None,
            "observed_utc_days": [], "missing_requested_utc_days": [],
        }
    lo, hi = runner.utc_ms(runner.START), runner.utc_ms(runner.END)
    if table == "cap_book":
        filt = f"upper(asset)='{runner.ASSET}' AND lower(slug) LIKE '%-updown-5m-%'"
    elif table == "cap_prices":
        filt = f"upper(asset)='{runner.ASSET}' AND lower(src)='chainlink'"
    else:
        raise ValueError(table)

    con = duckdb.connect()
    files_sql = runner.sql_files(paths)
    base = f"read_parquet({files_sql}, union_by_name=true)"
    row = con.execute(f"""
        SELECT count(*) AS n, min(CAST(ts_ms AS BIGINT)) AS first_ts, max(CAST(ts_ms AS BIGINT)) AS last_ts
        FROM {base}
        WHERE {filt} AND ts_ms >= {lo} AND ts_ms < {hi}
    """).fetchone()
    ddf = con.execute(f"""
        SELECT CAST(to_timestamp(CAST(ts_ms AS DOUBLE)/1000.0) AS DATE) AS day, count(*) AS n
        FROM {base}
        WHERE {filt} AND ts_ms >= {lo} AND ts_ms < {hi}
        GROUP BY day ORDER BY day
    """).fetchdf()
    con.close()

    observed = [str(x) for x in ddf["day"].tolist()] if len(ddf) else []
    expected = [d.strftime("%Y-%m-%d") for d in pd.date_range(
        runner.START, pd.Timestamp(runner.END) - pd.Timedelta(days=1), freq="D", tz="UTC"
    )]
    observed_set = set(observed)
    missing = [d for d in expected if d not in observed_set]
    return {
        "table": table,
        "rows_in_requested_window": int(row[0] or 0),
        "first_ts_ms": int(row[1]) if row[1] is not None else None,
        "last_ts_ms": int(row[2]) if row[2] is not None else None,
        "observed_utc_days": observed,
        "observed_utc_day_count": len(observed),
        "requested_utc_day_count": len(expected),
        "missing_requested_utc_days": missing,
        "complete_requested_day_coverage": len(missing) == 0,
    }


runner.list_selected_files = boundary_safe_list_selected_files
runner.load_decision_books = latest_row_books


if __name__ == "__main__":
    runner.main()
    out = Path("eth5m_tail_out")
    raw_path = out / "market_rows.csv"
    surface_path = out / "surface.csv"
    manifest_path = out / "manifest.json"

    book_cov = table_coverage(_SELECTED_PATHS.get("cap_book", []), "cap_book")
    price_cov = table_coverage(_SELECTED_PATHS.get("cap_prices", []), "cap_prices")
    source_complete = bool(
        book_cov.get("complete_requested_day_coverage")
        and price_cov.get("complete_requested_day_coverage")
    )

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

        stat_grades = [x["grade"] for x in stat_rows]
        overall = []
        for g in stat_grades:
            if str(g).startswith("RED"):
                overall.append("RED")
            elif not source_complete:
                overall.append("YELLOW_SOURCE_COVERAGE")
            else:
                overall.append(g)

        surf["statistical_grade"] = stat_grades
        surf["overall_qualification_grade"] = overall
        surf["source_requested_day_coverage_complete"] = source_complete
        surf["bootstrap_days"] = [x["days"] for x in stat_rows]
        surf["mean_daily_pnl"] = [x["mean_daily_pnl"] for x in stat_rows]
        surf["daily_pnl_ci95_low"] = [x["mean_daily_pnl_ci95"][0] for x in stat_rows]
        surf["daily_pnl_ci95_high"] = [x["mean_daily_pnl_ci95"][1] for x in stat_rows]
        surf["positive_day_rate"] = [x["positive_day_rate"] for x in stat_rows]
        surf.to_csv(surface_path, index=False)

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        m["requested_period"] = [runner.START, runner.END]
        m["source_coverage"] = {
            "cap_book_eth5m": book_cov,
            "cap_prices_eth_chainlink": price_cov,
            "complete_requested_day_coverage": source_complete,
            "policy": "statistical alpha is reported on observed trades, but an otherwise positive/GREEN cell is demoted to YELLOW_SOURCE_COVERAGE when either required source is missing any requested UTC day",
        }
        m["qualification_gate"] = {
            "GREEN": "cell total PnL >0, >=20 observed trading days, >=100 trades, day-bootstrap mean daily PnL CI95 lower bound >0, and complete requested-day coverage in ETH5m cap_book plus ETH Chainlink cap_prices",
            "YELLOW": "positive PnL but statistical sample/CI or requested source coverage is insufficient",
            "RED": "cell total PnL <=0",
            "selection_policy": "all 20 predeclared fair-floor x barrier-distance cells are reported and graded; no post-hoc winner is substituted for the surface",
        }
        m["qualification_audit"] = {
            "book_selector": "latest capture row at/before T-90 is ranked before any price/depth validation",
            "rotation_file_selection": "window rotations plus first post-END rotation, with parquet timestamps performing final row filtering",
            "stale_price_borrowing": False,
            "stale_depth_borrowing": False,
            "gamma_terminal_resolution_required": True,
            "max_book_staleness_ms": runner.MAX_BOOK_STALENESS_MS,
            "max_chainlink_source_lag_ms": runner.MAX_CHAINLINK_LAG_MS,
        }
        manifest_path.write_text(json.dumps(m, indent=2))
