from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence import eth5m_chainlink_tail_audited as audited
from main_sequence.qualification_stats import daily_pnl_stats

MIN_PTB_AUDIT_MARKETS = 100
MIN_PTB_AUDIT_COVERAGE_FOR_GREEN = 0.90
MAX_PTB_P95_ABS_BPS_FOR_GREEN = 0.10


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


def _json_like(x):
    if isinstance(x, str):
        s = x.strip()
        if s[:1] in ("{", "["):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def extract_price_to_beat(obj, path: str = "$", depth: int = 0) -> tuple[float | None, str | None]:
    """Find an exact `priceToBeat`-style key anywhere in Gamma metadata.

    This value is audit-only. It is populated after close in historical Gamma
    metadata and must never enter the T-90 signal path.
    """
    if depth > 10:
        return None, None
    obj = _json_like(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if norm == "pricetobeat":
                try:
                    v = float(value)
                    if math.isfinite(v) and v > 0:
                        return v, f"{path}.{key}"
                except Exception:
                    pass
        for key, value in obj.items():
            v, p = extract_price_to_beat(value, f"{path}.{key}", depth + 1)
            if v is not None:
                return v, p
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            v, p = extract_price_to_beat(value, f"{path}[{i}]", depth + 1)
            if v is not None:
                return v, p
    return None, None


def fetch_gamma_resolutions_with_ptb(book: pd.DataFrame) -> tuple[dict[str, bool], pd.DataFrame]:
    """Same terminal-resolution contract as core runner, plus audit-only PTB extraction."""
    runner = audited.runner
    ids = sorted(book["cond"].dropna().astype(str).unique().tolist()) if len(book) else []
    sess = runner.requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-eth5m-resolution-ptb-audit/1.0"})
    resolved: dict[str, bool] = {}
    audit: dict[str, dict] = {}

    def consume(raw: dict | None, cid: str) -> None:
        up = runner.parse_resolution(raw) if raw is not None else None
        ptb, ptb_path = extract_price_to_beat(raw) if raw is not None else (None, None)
        audit[cid] = {
            "cond": cid,
            "slug_gamma": str(raw.get("slug") or "") if raw else "",
            "closed": bool(raw.get("closed")) if raw else False,
            "gamma_up": up,
            "outcomes": raw.get("outcomes") if raw else None,
            "outcomePrices": raw.get("outcomePrices") if raw else None,
            "gamma_price_to_beat": ptb,
            "gamma_price_to_beat_path": ptb_path,
        }
        if up is not None:
            resolved[cid] = up

    for off in range(0, len(ids), runner.RESOLUTION_BATCH):
        chunk = ids[off:off + runner.RESOLUTION_BATCH]
        params = [("condition_ids", x) for x in chunk] + [("closed", "true"), ("limit", len(chunk))]
        rows = runner.gamma_get(sess, params)
        by_id = {str(raw.get("conditionId") or ""): raw for raw in rows}
        for cid in chunk:
            if cid in by_id:
                consume(by_id[cid], cid)
        runner.time.sleep(0.05)

    # Missing batch results are fetched one-by-one exactly as in the core contract.
    missing = [cid for cid in ids if cid not in audit]
    for cid in missing:
        rows = runner.gamma_get(sess, [("condition_ids", cid), ("closed", "true"), ("limit", 1)])
        raw = next((x for x in rows if str(x.get("conditionId") or "") == cid), None)
        consume(raw, cid)
        runner.time.sleep(0.05)

    cols = [
        "cond", "slug_gamma", "closed", "gamma_up", "outcomes", "outcomePrices",
        "gamma_price_to_beat", "gamma_price_to_beat_path",
    ]
    adf = pd.DataFrame(list(audit.values())) if audit else pd.DataFrame(columns=cols)
    return resolved, adf.sort_values("cond", kind="mergesort") if len(adf) else adf


def build_ptb_audit(raw: pd.DataFrame, gamma_path: Path) -> tuple[pd.DataFrame, dict, bool, str | None]:
    audit = {
        "candidate_markets": int(len(raw)),
        "gamma_ptb_available": 0,
        "gamma_ptb_coverage": 0.0,
        "abs_mismatch_bps_median": None,
        "abs_mismatch_bps_p95": None,
        "abs_mismatch_bps_max": None,
        "minimum_markets_for_green": MIN_PTB_AUDIT_MARKETS,
        "minimum_coverage_for_green": MIN_PTB_AUDIT_COVERAGE_FOR_GREEN,
        "maximum_p95_abs_bps_for_green": MAX_PTB_P95_ABS_BPS_FOR_GREEN,
        "role": "post-close ground-truth audit only; gamma priceToBeat never enters the signal",
    }
    if raw.empty or not gamma_path.exists():
        return raw, audit, False, "YELLOW_PTB_AUDIT"

    gamma = pd.read_csv(gamma_path)
    if "gamma_price_to_beat" not in gamma.columns:
        return raw, audit, False, "YELLOW_PTB_AUDIT"
    ptb_map = gamma.drop_duplicates("cond", keep="last").set_index("cond")["gamma_price_to_beat"]
    out = raw.copy()
    out["gamma_price_to_beat"] = out["cond"].astype(str).map(ptb_map)
    ptb = pd.to_numeric(out["gamma_price_to_beat"], errors="coerce")
    threshold = pd.to_numeric(out["threshold_chainlink"], errors="coerce")
    valid = ptb.notna() & threshold.notna() & (ptb > 0) & (threshold > 0)
    out["threshold_vs_gamma_ptb_bps"] = np.where(valid, (threshold / ptb - 1.0) * 10000.0, np.nan)
    diff = out.loc[valid, "threshold_vs_gamma_ptb_bps"].astype(float).to_numpy()
    absdiff = np.abs(diff)
    coverage = float(valid.mean()) if len(out) else 0.0
    audit.update({
        "gamma_ptb_available": int(valid.sum()),
        "gamma_ptb_coverage": coverage,
        "abs_mismatch_bps_median": float(np.median(absdiff)) if len(absdiff) else None,
        "abs_mismatch_bps_p95": float(np.quantile(absdiff, 0.95)) if len(absdiff) else None,
        "abs_mismatch_bps_max": float(np.max(absdiff)) if len(absdiff) else None,
    })
    enough = int(valid.sum()) >= MIN_PTB_AUDIT_MARKETS and coverage >= MIN_PTB_AUDIT_COVERAGE_FOR_GREEN
    if not enough:
        return out, audit, False, "YELLOW_PTB_AUDIT"
    if float(audit["abs_mismatch_bps_p95"]) > MAX_PTB_P95_ABS_BPS_FOR_GREEN:
        return out, audit, False, "YELLOW_PTB_MISMATCH"
    return out, audit, True, None


def postprocess() -> None:
    out = Path("eth5m_tail_out")
    raw_path = out / "market_rows.csv"
    gamma_path = out / "gamma_resolutions.csv"
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

    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    raw, ptb_audit, ptb_audit_pass, ptb_yellow_reason = build_ptb_audit(raw, gamma_path)
    if raw_path.exists():
        raw.to_csv(raw_path, index=False)

    if surface_path.exists():
        surf = pd.read_csv(surface_path)
        stat_rows = []
        for r in surf.itertuples(index=False):
            fair_floor = float(r.fair_floor)
            barrier_floor = float(r.barrier_bps_floor)
            z = raw[(raw["fair"] >= fair_floor) &
                    (raw["barrier_bps"] >= barrier_floor) &
                    (raw["net_settlement_edge_ps"] > 0) &
                    (raw["depth_headroom_x"] >= 1.0)].copy() if len(raw) else pd.DataFrame()
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
            elif not ptb_audit_pass:
                overall.append(ptb_yellow_reason or "YELLOW_PTB_AUDIT")
            else:
                overall.append(grade)

        surf["statistical_grade"] = stat_grades
        surf["overall_qualification_grade"] = overall
        surf["source_requested_day_coverage_complete"] = day_coverage_complete
        surf["source_dual_clock_complete"] = source_clock_complete
        surf["ptb_audit_pass"] = ptb_audit_pass
        surf["ptb_audit_coverage"] = ptb_audit.get("gamma_ptb_coverage")
        surf["ptb_p95_abs_mismatch_bps"] = ptb_audit.get("abs_mismatch_bps_p95")
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
        manifest["ptb_reconstruction_audit"] = ptb_audit
        manifest["qualification_gate"] = {
            "GREEN": "cell total PnL >0, >=20 observed trading days, >=100 trades, day-bootstrap mean daily PnL CI95 lower bound >0, complete requested-day source coverage, >=99.5% valid Chainlink payload/source timestamp coverage, and PTB audit >=100 markets / >=90% coverage / p95 absolute mismatch <=0.10bp",
            "YELLOW": "positive PnL but statistical sample/CI, requested source coverage, source-clock audit, or PTB reconstruction audit is insufficient",
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
            "gamma_price_to_beat_role": "post-close audit only; never a signal input",
            "stale_price_borrowing": False,
            "stale_depth_borrowing": False,
            "gamma_terminal_resolution_required": True,
            "max_book_staleness_ms": audited.runner.MAX_BOOK_STALENESS_MS,
            "max_chainlink_capture_or_source_lag_ms": audited.runner.MAX_CHAINLINK_LAG_MS,
            "max_capture_minus_source_abs_ms": audited.MAX_CAPTURE_SOURCE_DELTA_MS,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))


# Frozen qualification monkey patches. Gamma PTB is audit-only and is written
# after the runner has already generated the trading rows.
audited.DualClockPriceSeries.at_source_time = at_source_time_bounded
audited.runner.fetch_gamma_resolutions = fetch_gamma_resolutions_with_ptb


if __name__ == "__main__":
    audited.runner.main()
    postprocess()
