from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from main_sequence import eth5m_chainlink_tail_qualified as qualified


def load_chainlink_no_future_source(paths):
    """Use only Chainlink rows whose oracle/source clock is not after capture clock."""
    audited = qualified.audited
    ps = audited.load_chainlink_dual_clock(paths)
    keep = ps.capture_ts >= ps.source_ts
    total_rows = int(audited._DUAL_CLOCK_AUDIT.get("chainlink_rows") or 0)
    kept = int(np.sum(keep))
    audited._DUAL_CLOCK_AUDIT["source_clock_rows_nonfuture_vs_capture"] = kept
    audited._DUAL_CLOCK_AUDIT["future_source_rows_rejected"] = int(np.sum(~keep))
    audited._DUAL_CLOCK_AUDIT["source_clock_valid_rate"] = (kept / total_rows) if total_rows else 0.0
    audited._DUAL_CLOCK_AUDIT["future_source_policy"] = "require source_ts <= capture_ts; collector clock skew never creates admissible future oracle state"
    if kept < 100:
        raise RuntimeError(f"insufficient nonfuture dual-clock Chainlink ETH observations: {kept}")
    return audited.DualClockPriceSeries(
        ps.capture_ts[keep],
        ps.source_ts[keep],
        ps.px[keep],
    )


# Freeze one common bootstrap unit across all three lanes: market-start UTC day.
_base_daily_pnl_stats = qualified.daily_pnl_stats


def market_start_daily_pnl_stats(df, time_col, pnl_col, *, unit="s", seed_offset=0):
    if df is not None and hasattr(df, "columns") and "win_start" in df.columns:
        return _base_daily_pnl_stats(df, "win_start", pnl_col, unit="s", seed_offset=seed_offset)
    return _base_daily_pnl_stats(df, time_col, pnl_col, unit=unit, seed_offset=seed_offset)


qualified.audited.runner.load_chainlink = load_chainlink_no_future_source
qualified.daily_pnl_stats = market_start_daily_pnl_stats


if __name__ == "__main__":
    qualified.audited.runner.main()
    qualified.postprocess()
    manifest_path = Path("eth5m_tail_out/manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("qualification_audit", {})["bootstrap_day_key"] = "market start UTC day"
        manifest_path.write_text(json.dumps(manifest, indent=2))
