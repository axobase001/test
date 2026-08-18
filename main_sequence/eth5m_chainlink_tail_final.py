from __future__ import annotations

import numpy as np

from main_sequence import eth5m_chainlink_tail_qualified as qualified


def load_chainlink_no_future_source(paths):
    """Use only Chainlink rows whose oracle/source clock is not after capture clock.

    The audited loader already requires a parsed payload timestamp and <=5s absolute
    capture/source separation. Qualification additionally rejects source_ts>capture_ts
    rather than treating collector clock skew as usable alpha.
    """
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


qualified.audited.runner.load_chainlink = load_chainlink_no_future_source


if __name__ == "__main__":
    qualified.audited.runner.main()
    qualified.postprocess()
