from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence import eth5m_chainlink_tail_audited as audited
from main_sequence.qualification_stats import daily_pnl_stats


def at_source_time_bounded(
    self: audited.DualClockPriceSeries,
    target_source_ms: int,
    available_by_ms: int,
    max_source_staleness_ms: int = audited.runner.MAX_CHAINLINK_LAG_MS,
) -> tuple[float, int, int]:
    """Exact-equivalent bounded lookup for the audited dual-clock contract.

    Any admissible row has source in [target-staleness, target] and
    |capture-source| <= MAX_CAPTURE_SOURCE_DELTA_MS. Therefore its capture
    timestamp must lie in
      [target-staleness-delta, target+delta].
    Searching outside that interval can never change the answer.
    """
    target = int(target_source_ms)
    available = int(available_by_ms)
    delta = audited.MAX_CAPTURE_SOURCE_DELTA_MS
    lo_cap = target - int(max_source_staleness_ms) - delta
    hi_cap = min(available, target + delta)
    lo = int(np.searchsorted(self.capture_ts, lo_cap, side="left"))
    hi = int(np.searchsorted(self.capture_ts, hi_cap, side="right"))
    if hi <= lo:
        return float("nan"), -1, -1

    src = self.source_ts[lo:hi]
    cap = self.capture_ts[lo:hi]
    px = self.px[lo:hi]
    mask = (
        (cap <= available)
        & (src <= target)
        & (src >= target - int(max_source_staleness_ms))
        & (np.abs(cap - src) <= delta)
        & np.isfinite(px)
        & (px > 0)
    )
    idx = np.flatnonzero(mask)
    if not len(idx):
        return float("nan"), -1, -1
    cand_src = src[idx]
    best_src = np.max(cand_src)
    same = idx[cand_src == best_src]
    j = int(same[np.argmax(cap[same])])
    return float(px[j]), int(src[j]), int(cap[j])


def postprocess() -> None:
    out = Path("eth5m_tail_out")
    raw_path = out / "market_rows.csv"
    surface_path = out / "surface.csv"
    manifest_path = out / "manifest.json"

    book_cov = audited.table_coverage(audited._SELECTED_PATHS.get("cap_book", []), "cap_book")
    price_cov = audited.table_coverage(audited._SELECTED_PATHS.get("cap_prices", []), "cap_prices")
    day_coverage_complete = bool(
        book_cov.get("complete_requested_day_coverage")
        and price_cov.get("complete_requested_day_coverage")
    )
    source_clock_complete = bool(
        float(audited._DUAL_CLOCK_AUDIT.get("payload_source_timestamp_parse_rate") or 0.0)
        >= audited.MIN_CLOCK_PARSE_RATE_FOR_GREEN
        and float(audited._DUAL_CLOCK_AUDIT.get("source_clock_valid_rate") or 0.0)
        >= audited.MIN_CLOCK_PARSE_RATE_FOR_GREEN
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
            stat_rows.append(daily_pnl_stats(
                z, "decision_ts_ms", "pnl_fixed5", unit="ms", seed_offset=seed_offset
            ))

        stat_grades = [x["grade"] for x in stat_rows]
        overall = []
        for grade in stat_grades:
            if str(grade).startswith("RED"):
                overall.append("RED")
            elif not day_coverage_complete:
                overall.append("YELLOW_SOURCE_COVERAGE")
            elif not source_clock_complete:
                overall.append("YELLOW_SOURCE_CLOCK")
            else:
                overall.append(grade)

        surf["statistical_grade"] = stat_grades
        surf["overall_qualification_grade"] = overall
        surf["source_requested_day_coverage_complete"] = day_coverage_complete
        surf["source_dual_clock_complete"] = source_clock_complete
        surf["bootstrap_days"] = [x["days"] for x in stat_rows]
        surf["mean_daily_pnl"] = [x["mean_daily_pnl"] for x in stat_rows]
        surf["daily_pnl_ci95_low"] = [x["mean_daily_pnl_ci95"][0] for x in stat_rows]
        surf["daily_pnl_ci95_high"] = [x["mean_daily_pnl_ci95"][1] for x in stat_rows]
        surf["positive_day_rate"] = [x["positive_day_rate"] for x in stat_rows]
        if len(raw):
            surf["median_threshold_capture_minus_source_ms"] = float(raw["threshold_capture_minus_source_ms"].median())
            surf["median_spot_capture_minus_source_ms"] = float(raw["spot_capture_minus_source_ms"].median())
        surf.to_csv(surface_path, index=False)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["requested_period"] = [audited.runner.START, audited.runner.END]
        manifest["source_coverage"] = {
            "cap_book_eth5m": book_cov,
            "cap_prices_eth_chainlink": price_cov,
            "complete_requested_day_coverage": day_coverage_complete,
            "dual_clock_chainlink": audited._DUAL_CLOCK_AUDIT,
            "dual_clock_complete_for_green": source_clock_complete,
            "policy": "statistical alpha is reported on observed trades, but an otherwise positive/GREEN cell is demoted when requested-day coverage or Chainlink source-clock extraction is insufficient",
        }
        manifest["qualification_gate"] = {
            "GREEN": "cell total PnL >0, >=20 observed trading days, >=100 trades, day-bootstrap mean daily PnL CI95 lower bound >0, complete requested-day source coverage, and >=99.5% valid Chainlink payload/source timestamp coverage",
            "YELLOW": "positive PnL but statistical sample/CI, requested source coverage, or source-clock audit is insufficient",
            "RED": "cell total PnL <=0",
            "selection_policy": "all 20 predeclared fair-floor x barrier-distance cells are reported and graded; no post-hoc winner is substituted for the surface",
        }
        manifest["qualification_audit"] = {
            "runtime_entrypoint": "eth5m_chainlink_tail_qualified.py",
            "book_selector": "latest capture row at/before T-90 is ranked before any price/depth validation",
            "rotation_file_selection": "window rotations plus first post-END rotation, with parquet timestamps performing final row filtering",
            "chainlink_clock": "collector capture clock gates availability; nested RTDS payload timestamp defines oracle/source time",
            "threshold_clock": "latest source timestamp <= window start, captured no later than the chosen T-90 decision book timestamp",
            "source_time_lookup": "bounded exact-equivalent capture slice derived from the 5s capture-source invariant",
            "stale_price_borrowing": False,
            "stale_depth_borrowing": False,
            "gamma_terminal_resolution_required": True,
            "max_book_staleness_ms": audited.runner.MAX_BOOK_STALENESS_MS,
            "max_chainlink_capture_or_source_lag_ms": audited.runner.MAX_CHAINLINK_LAG_MS,
            "max_capture_minus_source_abs_ms": audited.MAX_CAPTURE_SOURCE_DELTA_MS,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))


# Patch only the expensive source-time lookup. All strategy/data gates remain
# those of the frozen audited module.
audited.DualClockPriceSeries.at_source_time = at_source_time_bounded


if __name__ == "__main__":
    audited.runner.main()
    postprocess()
