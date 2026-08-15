from __future__ import annotations

"""Main Sequence quarterly OOS protocol v1.2.

This is a pre-OOS engineering/statistical amendment. Both v1 and v1.1 aborted
inside the pre-March freeze job; no March-May shard was fetched or scored.

v1.2 keeps the model, features, fees, thresholds, execution proxy, OOS envelope,
and all statistical gates unchanged, while making quarter-scale transport
practical and rate-limit safe:

* The exact-slug /markets request already made once per hour by the streaming
  fetch is the primary listing census. We do not immediately repeat the same
  672+ requests as a redundant second census. Failed transport hours still fail.
* OOS shards additionally require >=99.5% mapped-market coverage and every
  calendar day to survive example construction.
* Trade transport fetches only decision-120s through decision+5s, which is the
  full interval used by all frozen microstructure features and the 5s tape
  proxy. Only low-activity markets with <4 pre-decision rows in that interval
  get an early-window backfill to preserve the original >=4-trades eligibility
  rule. A frozen exact-equivalence gate compares features and fills against the
  old full-window implementation before training.

The train/validation split remains the v1.1 coverage-only amendment:
train through Feb 25, validation Feb 26-28, with March 1 as first OOS instant.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

import quarterly_5m_official_release as q
import prejuly_5m_official as core
import prejuly_5m_official_stream as stream
import prejuly_5m_official_release as release

# ---- Frozen protocol amendment, still before any March-May OOS fetch ----
q.TRAIN_START = "2026-02-01"
q.TRAIN_END = "2026-02-26"
q.VAL_START = "2026-02-26"
q.VAL_END = "2026-03-01"
q.EVALUATION_CONTRACT["protocol_name"] = "Main Sequence quarterly OOS retrospective confirmation v1.2"
q.EVALUATION_CONTRACT["train"] = [q.TRAIN_START, q.TRAIN_END]
q.EVALUATION_CONTRACT["validation"] = [q.VAL_START, q.VAL_END]
q.EVALUATION_CONTRACT["pre_oos_protocol_amendment"] = (
    "v1 and v1.1 both aborted inside pre-March freeze before any March-May shard fetch. "
    "v1 established that exhaustive per-missing-slug /events corroboration rate-limits; v1.1 "
    "established that redundantly repeating the full hourly /markets census after the streaming "
    "pass also rate-limits at quarter scale. v1.2 therefore treats the streaming pass's own exact-slug "
    "/markets query as the primary exhaustive census, removes redundant network re-census, requires "
    ">=99.5% mapped coverage for each OOS shard, and uses an exact-equivalent late-window trade "
    "transport with conditional eligibility backfill. Train/validation split remains Feb26. "
    "Model, features, fees, residual bound, 3c edge floor, +2c limit, 5s tape proxy, March-May OOS "
    "envelope and all statistical gates are unchanged."
)
q.EVALUATION_CONTRACT["transport_gate"] = (
    "Before freeze, optimized trade transport must be exactly equivalent to the original full-window "
    "implementation on the frozen June probe: identical eligible slugs, all feature values, scalar "
    "labels/PM-last values and post-decision fill tuples. Every OOS shard must have zero failed hours, "
    ">=99.5% mapped-market coverage, and no missing calendar day after example construction."
)


def _single_pass_listing_census(start: str, end: str, stream_mapped: int, workers: int = 20):
    """No extra network I/O: the streaming hourly exact-slug query *is* the census."""
    lo, hi = core.ts(start), core.ts(end)
    hours = (hi - lo) // 3600
    expected = int(hours * 12 * len(core.ASSETS))
    mapped = int(stream_mapped)
    if mapped < 0 or mapped > expected:
        raise RuntimeError(
            f"PHASE_COVERAGE_RED {start}..{end}: mapped={mapped} expected={expected}"
        )
    absent = expected - mapped
    print(
        "SINGLE_PASS_LISTING_CENSUS",
        start,
        end,
        "mapped",
        mapped,
        "unmapped",
        absent,
        "theoretical",
        expected,
        flush=True,
    )
    return {
        "theoretical_markets": expected,
        "census_mapped_markets": mapped,
        "confirmed_absent_count": absent,
        "confirmed_absent_slugs": [],
        "method": (
            "The streaming fetch itself issues one successful Gamma /markets request per UTC hour with "
            "all 48 exact expected slugs and limit=100. That first-pass response is the exhaustive source "
            "census; v1.2 intentionally does not repeat the same requests. Unmapped count is theoretical "
            "minus parsed exact-slug listings. OOS additionally enforces >=99.5% mapped coverage."
        ),
    }


release._verify_listing_coverage = _single_pass_listing_census


def fetch_hour_stream_v12(hour_start: int):
    """Exact-equivalent transport with roughly half the raw trade window."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-quarter-v12/1.0"})

    wanted = []
    for asset in core.ASSETS:
        for t0 in range(hour_start, hour_start + 3600, 300):
            wanted.append((f"{asset.lower()}-updown-5m-{t0}", asset, t0))
    params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
    js = core.get_json(sess, core.GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js}

    by_start: dict[int, list[core.Market]] = {
        t: [] for t in range(hour_start, hour_start + 3600, 300)
    }
    mapped = 0
    for slug, asset, t0 in wanted:
        m = core.parse_market(byslug.get(slug, {}), asset, t0)
        if m is not None:
            by_start[t0].append(m)
            mapped += 1

    micros = []
    raw_rows = 0
    slot_failures = []
    backfill_markets = 0

    for t0, markets in by_start.items():
        if not markets:
            continue
        try:
            # Frozen feature windows use at most 120 seconds before decision;
            # execution proxy uses decision..decision+5s.
            late_start = t0 + 120
            late_end = t0 + 240 + core.TAPE_SECONDS
            late_rows = stream._query_rows(sess, markets, late_start, late_end)
            raw_rows += len(late_rows)
            late_tm = core.normalize_trade_rows(pd.DataFrame(late_rows))

            need_early = []
            for m in markets:
                g = late_tm.get(m.condition_id)
                if g is None or g.empty:
                    # No trade in the last 120s means frozen lag<=45 eligibility cannot hold.
                    continue
                pre = g[(g.timestamp >= late_start) & (g.timestamp < m.decision)]
                if len(pre) >= 4:
                    continue
                if len(pre) == 0:
                    continue
                lag = m.decision - int(pre.timestamp.iloc[-1])
                if 0 <= lag <= 45:
                    # Earlier rows can only affect the original >=4 total-pre-trades
                    # eligibility test; all numerical frozen features remain late-window only.
                    need_early.append(m)

            if need_early:
                early_rows = stream._query_rows(sess, need_early, t0, t0 + 119)
                raw_rows += len(early_rows)
                backfill_markets += len(need_early)
                rows = late_rows + early_rows
                tm = core.normalize_trade_rows(pd.DataFrame(rows))
            else:
                tm = late_tm

            for m in markets:
                z = stream._aggregate_market(m, tm)
                if z is not None:
                    micros.append(z)
        except Exception as exc:
            slot_failures.append({"slot": t0, "error": repr(exc)})

    return micros, {
        "mapped": mapped,
        "raw_rows": raw_rows,
        "slot_failures": slot_failures,
        "eligibility_backfill_markets": backfill_markets,
        "transport_v12": "decision-120s..decision+5s plus conditional start..decision-121s eligibility backfill",
    }


# The original phase-stream function captured by release looks up this module
# global at runtime, so this safely swaps transport without duplicating protocol.
stream.fetch_hour_stream = fetch_hour_stream_v12
_complete_fetch = release.fetch_phase_stream_complete


def guarded_phase_fetch(start: str, end: str, workers: int = 20):
    micros, cov = _complete_fetch(start, end, workers)
    cov["transport"] = (
        "v1.2 exact-equivalent: decision-120s..decision+5s; conditional early backfill only when "
        "needed to preserve >=4 total pre-decision trades eligibility"
    )
    expected = int(cov["expected_markets"])
    mapped = int(cov["mapped_markets"])
    ratio = (mapped / expected) if expected else 0.0
    cov["mapped_market_ratio"] = ratio
    if core.ts(start) >= core.ts(q.QUARTER_TEST_START) and ratio < 0.995:
        raise RuntimeError(
            f"OOS_COVERAGE_RED {start}..{end}: mapped_ratio={ratio:.6f} < 0.995 "
            f"mapped={mapped} expected={expected}"
        )
    return micros, cov


stream.fetch_phase_stream = guarded_phase_fetch


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "equiv":
        # Existing probe compares full historical implementation vs the currently
        # installed stream.fetch_hour_stream, including features and fill tuples.
        stream.equivalence_probe()
    else:
        q.main()
