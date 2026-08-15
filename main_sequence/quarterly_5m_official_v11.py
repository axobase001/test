from __future__ import annotations

"""Quarter OOS protocol v1.1.

Amendments are based only on pre-OOS transport/coverage metadata from the aborted
v1 run.  No March-May outcome/test shard was fetched before these changes.

1. The historical 5m archive is sparse/absent through much of early February,
   so the train/validation boundary is moved to Feb 26 to guarantee a useful
   pre-March fit sample while preserving a fully pre-OOS validation tail.
2. The exhaustive primary Gamma /markets census is retained, but independent
   /events corroboration is deterministically sampled instead of querying every
   archive-absent slug (which rate-limited at 429 on the v1 pre-OOS run).

Model architecture, features, fee model, residual bound, edge threshold,
execution proxy, March-May OOS envelope, and statistical gates are unchanged.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np

import quarterly_5m_official_release as q
import prejuly_5m_official as core
import prejuly_5m_official_release as release

# Coverage-only protocol amendment, before any March-May shard is opened.
q.TRAIN_START = "2026-02-01"
q.TRAIN_END = "2026-02-26"
q.VAL_START = "2026-02-26"
q.VAL_END = "2026-03-01"
q.EVALUATION_CONTRACT["protocol_name"] = "Main Sequence quarterly OOS retrospective confirmation v1.1"
q.EVALUATION_CONTRACT["train"] = [q.TRAIN_START, q.TRAIN_END]
q.EVALUATION_CONTRACT["validation"] = [q.VAL_START, q.VAL_END]
q.EVALUATION_CONTRACT["pre_oos_protocol_amendment"] = (
    "v1 aborted before any March-May OOS shard fetch. Pre-OOS Gamma coverage showed the 5m archive "
    "is absent/sparse through much of early February and exhaustive /events absence corroboration "
    "hit HTTP 429 after the full primary /markets census. v1.1 moves only the train/validation split "
    "to Feb 26 and changes only the secondary coverage corroboration to deterministic sampling. "
    "Architecture, features, thresholds, fees, execution proxy, OOS envelope and gates are unchanged."
)


def _safe_event(slug: str, closed_filter: bool | None = None):
    """Low-rate independent archive corroboration with explicit retry."""
    last = None
    for attempt in range(4):
        try:
            out = release._event_exact(slug, closed_filter)
            time.sleep(0.15)
            return out
        except Exception as exc:  # transport only; never convert an error into absence
            last = exc
            time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"sampled /events corroboration failed for {slug}: {last!r}")


def _even_sample(seq, n: int):
    seq = list(seq)
    if len(seq) <= n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n, dtype=int)
    return [seq[int(i)] for i in sorted(set(idx.tolist()))]


def _verify_listing_coverage_sampled(start: str, end: str, stream_mapped: int, workers: int = 12):
    lo, hi = core.ts(start), core.ts(end)
    hours = list(range(lo, hi, 3600))
    by_hour = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(release._gamma_census_hour, h): h for h in hours}
        for f in as_completed(futs):
            h, present, missing = f.result()
            by_hour[h] = {"present": present, "missing": missing}

    expected = len(hours) * 12 * len(core.ASSETS)
    census_mapped = sum(len(v["present"]) for v in by_hour.values())
    missing = []
    anomalous_hours = []
    for h in sorted(by_hour):
        miss = by_hour[h]["missing"]
        if miss:
            anomalous_hours.append(h)
            missing.extend(miss)

    if census_mapped != stream_mapped:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: stream_mapped={stream_mapped} "
            f"independent_census_mapped={census_mapped}"
        )
    if census_mapped + len(missing) != expected:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: census arithmetic mismatch "
            f"mapped={census_mapped} missing={len(missing)} expected={expected}"
        )

    # Deterministically cover the full time span and every asset in the missing sample.
    missing_sample = []
    for asset in core.ASSETS:
        aa = sorted([x for x in missing if x[1] == asset], key=lambda x: x[2])
        missing_sample.extend(_even_sample(aa, 8))
    # Deduplicate while retaining deterministic order.
    seen = set()
    missing_sample = [x for x in missing_sample if not (x[0] in seen or seen.add(x[0]))]

    # Positive controls: sampled anomalous hours must resolve a listing through /events.
    controls = []
    for h in _even_sample(anomalous_hours, 16):
        present = by_hour[h]["present"]
        if present:
            controls.append((h, present[0]))

    control_results = []
    for h, item in controls:
        slug, asset, t0, condition_id = item
        exact = _safe_event(slug, None)
        ok = any(x.get("conditionId") == condition_id for x in exact)
        control_results.append({"hour": h, "slug": slug, "condition_id": condition_id, "ok": ok})
        if not ok:
            raise RuntimeError(f"PHASE_COVERAGE_RED {start}..{end}: /events positive control failed {slug}")

    missing_results = []
    for slug, asset, t0 in missing_sample:
        exact_any = _safe_event(slug, None)
        exact_closed = _safe_event(slug, True)
        rec = {
            "slug": slug,
            "asset": asset,
            "start": t0,
            "event_any_exact": exact_any,
            "event_closed_exact": exact_closed,
        }
        missing_results.append(rec)
        if exact_any or exact_closed:
            raise RuntimeError(
                f"PHASE_COVERAGE_RED {start}..{end}: primary-census-missing slug recovered in /events {slug}"
            )

    print(
        "SAMPLED_EVENT_COVERAGE_GREEN",
        start,
        end,
        "mapped",
        census_mapped,
        "census_absent",
        len(missing),
        "event_missing_sample",
        len(missing_results),
        "event_controls",
        len(control_results),
        flush=True,
    )
    return {
        "theoretical_markets": expected,
        "census_mapped_markets": census_mapped,
        # Exhaustive absence is established by the exact per-hour /markets census;
        # /events is a sampled independent corroborator in v1.1.
        "confirmed_absent_count": len(missing),
        "confirmed_absent_slugs": [x[0] for x in missing],
        "event_missing_sample": missing_results,
        "event_control_checks": control_results,
        "method": (
            "Exhaustive Gamma /markets per-hour exact-slug census. Every census-missing slug counts as "
            "archive-absent; an evenly spaced per-asset sample is independently required absent from "
            "Gamma /events under unfiltered and closed=true queries, with positive same-period controls. "
            "Sampling replaces exhaustive /events corroboration solely to avoid API 429 rate limiting."
        ),
    }


# fetch_phase_stream_complete resolves this global at call time.
release._verify_listing_coverage = _verify_listing_coverage_sampled

if __name__ == "__main__":
    q.main()
