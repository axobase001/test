from __future__ import annotations

# Final pre-July release guard.  Statistical/data preprocessing semantics are
# inherited from `prejuly_5m_official_final`; this file only turns any partial
# phase retrieval into a hard failure so an incomplete calendar interval can
# never masquerade as the declared full train/validation/test interval.

import prejuly_5m_official_final  # applies corrected size/order + finance-anchor patches
import prejuly_5m_official_stream as stream

_original_fetch_phase_stream = stream.fetch_phase_stream


def fetch_phase_stream_complete(start: str, end: str, workers: int = 20):
    micros, cov = _original_fetch_phase_stream(start, end, workers)
    failures = cov.get("failed_hours") or []
    mapped = int(cov.get("mapped_markets", -1))
    expected = int(cov.get("expected_markets", -2))
    if failures:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: failed_hours={failures[:10]} count={len(failures)}"
        )
    if mapped != expected:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: mapped_markets={mapped} expected_markets={expected}"
        )
    print(
        "PHASE_COVERAGE_GREEN",
        start,
        end,
        "mapped",
        mapped,
        "expected",
        expected,
        flush=True,
    )
    return micros, cov


stream.fetch_phase_stream = fetch_phase_stream_complete

if __name__ == "__main__":
    stream.main()
