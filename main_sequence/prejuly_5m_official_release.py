from __future__ import annotations

# Final pre-July release guard. Statistical/data preprocessing semantics are
# inherited from `prejuly_5m_official_final`; this file only enforces transport
# completeness. A theoretical 5m slot is allowed to be absent only when the
# independent Gamma /events index confirms that no exact market/event existed.

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import prejuly_5m_official as core
import prejuly_5m_official_final  # applies corrected size/order + finance-anchor patches
import prejuly_5m_official_stream as stream

_original_fetch_phase_stream = stream.fetch_phase_stream


def _gamma_census_hour(hour_start: int):
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-coverage-census/1.0"})
    wanted = []
    for asset in core.ASSETS:
        for t0 in range(hour_start, hour_start + 3600, 300):
            wanted.append((f"{asset.lower()}-updown-5m-{t0}", asset, t0))
    params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
    js = core.get_json(sess, core.GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js}
    present = []
    missing = []
    for slug, asset, t0 in wanted:
        m = core.parse_market(byslug.get(slug, {}), asset, t0)
        if m is None:
            missing.append((slug, asset, t0))
        else:
            present.append((slug, asset, t0, m.condition_id))
    return hour_start, present, missing


def _event_exact(slug: str, closed_filter: bool | None = None):
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-coverage-event-check/1.0"})
    params = [("slug", slug), ("limit", 20)]
    if closed_filter is not None:
        params.append(("closed", "true" if closed_filter else "false"))
    js = core.get_json(sess, core.GAMMA + "/events", params=params)
    exact = []
    for ev in js if isinstance(js, list) else []:
        for m in (ev.get("markets") or []):
            if str(m.get("slug")) == slug:
                exact.append({
                    "slug": str(m.get("slug")),
                    "conditionId": str(m.get("conditionId") or ""),
                    "closed": m.get("closed"),
                    "active": m.get("active"),
                })
    return exact


def _verify_listing_coverage(start: str, end: str, stream_mapped: int, workers: int = 20):
    lo, hi = core.ts(start), core.ts(end)
    hours = list(range(lo, hi, 3600))
    by_hour = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_gamma_census_hour, h): h for h in hours}
        for f in as_completed(futs):
            h, present, missing = f.result()
            by_hour[h] = {"present": present, "missing": missing}

    expected = len(hours) * 12 * len(core.ASSETS)
    census_mapped = sum(len(v["present"]) for v in by_hour.values())
    missing = []
    control_specs = []
    for h in sorted(by_hour):
        present = by_hour[h]["present"]
        miss = by_hour[h]["missing"]
        missing.extend(miss)
        if miss:
            # Prove the independent /events index is alive in the same anomalous hour.
            controls = present[:1] + (present[-1:] if len(present) > 1 else [])
            control_specs.extend((h, *x) for x in controls)

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

    control_results = []
    missing_results = []
    with ThreadPoolExecutor(max_workers=min(20, max(1, len(control_specs) + len(missing)))) as ex:
        control_futs = {
            ex.submit(_event_exact, slug, None): (h, slug, condition_id)
            for h, slug, asset, t0, condition_id in control_specs
        }
        missing_futs_any = {
            ex.submit(_event_exact, slug, None): (slug, asset, t0)
            for slug, asset, t0 in missing
        }
        for f, spec in control_futs.items():
            h, slug, condition_id = spec
            exact = f.result()
            ok = any(x.get("conditionId") == condition_id for x in exact)
            control_results.append({"hour": h, "slug": slug, "condition_id": condition_id, "ok": ok})
        for f, spec in missing_futs_any.items():
            slug, asset, t0 = spec
            exact_any = f.result()
            # Second query shape: a missing listing must also be absent from closed events.
            exact_closed = _event_exact(slug, True)
            missing_results.append({
                "slug": slug,
                "asset": asset,
                "start": t0,
                "event_any_exact": exact_any,
                "event_closed_exact": exact_closed,
            })

    bad_controls = [x for x in control_results if not x["ok"]]
    recovered_missing = [x for x in missing_results if x["event_any_exact"] or x["event_closed_exact"]]
    if bad_controls:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: Gamma /events control failure {bad_controls[:10]}"
        )
    if recovered_missing:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: batch-missing slugs exist in independent /events index "
            f"sample={recovered_missing[:5]} count={len(recovered_missing)}"
        )

    confirmed_absent = [x["slug"] for x in missing_results]
    return {
        "theoretical_markets": expected,
        "census_mapped_markets": census_mapped,
        "confirmed_absent_count": len(confirmed_absent),
        "confirmed_absent_slugs": confirmed_absent,
        "event_control_checks": control_results,
        "method": "Gamma /markets per-hour exact slug census; every missing slug must be absent from Gamma /events under both unfiltered and closed=true queries; same-hour present controls must resolve through /events.",
    }


def fetch_phase_stream_complete(start: str, end: str, workers: int = 20):
    micros, cov = _original_fetch_phase_stream(start, end, workers)
    failures = cov.get("failed_hours") or []
    mapped = int(cov.get("mapped_markets", -1))
    expected = int(cov.get("expected_markets", -2))
    if failures:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: failed_hours={failures[:10]} count={len(failures)}"
        )

    census = _verify_listing_coverage(start, end, mapped, workers)
    if census["theoretical_markets"] != expected:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: stream_expected={expected} "
            f"census_expected={census['theoretical_markets']}"
        )
    if mapped + census["confirmed_absent_count"] != expected:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: mapped={mapped} "
            f"confirmed_absent={census['confirmed_absent_count']} expected={expected}"
        )

    cov["official_listing_census"] = census
    print(
        "PHASE_COVERAGE_GREEN",
        start,
        end,
        "mapped",
        mapped,
        "confirmed_absent",
        census["confirmed_absent_count"],
        "theoretical",
        expected,
        flush=True,
    )
    return micros, cov


stream.fetch_phase_stream = fetch_phase_stream_complete

if __name__ == "__main__":
    stream.main()
