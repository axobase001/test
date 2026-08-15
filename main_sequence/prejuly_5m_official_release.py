from __future__ import annotations

# Final pre-July release guard. Statistical/data preprocessing semantics are
# inherited from `prejuly_5m_official_final`; this file enforces transport
# completeness and freezes the evaluation comparator before July is requested.

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

import prejuly_5m_official as core
import prejuly_5m_official_final  # applies corrected size/order + finance-anchor patches
import prejuly_5m_official_stream as stream

_original_fetch_phase_stream = stream.fetch_phase_stream
_original_probability_summary = core.probability_summary
_original_day_bootstrap = core.paired_day_bootstrap

EVALUATION_CONTRACT = {
    "primary_probability_comparator": "finance_p hard anchor",
    "primary_probability_gate": (
        "GREEN iff model Brier is lower than finance_p and the 95% day-block bootstrap CI for "
        "Brier(finance_p)-Brier(model) has lower bound > 0; RED iff aggregate Brier gain vs finance_p <= 0; "
        "otherwise YELLOW."
    ),
    "secondary_probability_comparator": "PM last pre-decision trade; descriptive/secondary, never allowed to override the finance-anchor primary gate.",
    "execution_role": "5-second public taker-BUY tape proxy is confirmatory tradeability evidence only; it cannot rescue a RED primary probability gate.",
    "freeze_timing": "Declared before any July 1-14 market/trade/Binance test data is requested.",
}


def probability_summary_with_finance_gate(xs, p):
    out = _original_probability_summary(xs, p)
    out["brier_gain_vs_finance"] = float(out["brier_finance"] - out["brier_model"])
    out["logloss_gain_vs_finance"] = float(out["logloss_finance"] - out["logloss_model"])
    out["primary_comparator"] = "finance_p"
    return out


def paired_day_bootstrap_with_finance(xs, p, reps=5000, seed=20260815):
    # Preserve the pre-existing PM-last day bootstrap and add the hard-anchor comparator.
    pm_out = _original_day_bootstrap(xs, p, reps=reps, seed=seed)
    y = np.asarray([e.label for e in xs], dtype=float)
    pp = np.asarray(p, dtype=float)
    fi = core.FEATURES.index("finance_p")
    finance = np.asarray([float(e.x[fi]) for e in xs], dtype=float)
    days = np.asarray([
        pd.Timestamp(e.start, unit="s", tz="UTC").strftime("%Y-%m-%d") for e in xs
    ])
    vals = []
    for d in sorted(set(days)):
        m = days == d
        vals.append(float(np.mean((y[m] - finance[m]) ** 2 - (y[m] - pp[m]) ** 2)))
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    z = np.empty(reps, dtype=float)
    for i in range(reps):
        z[i] = rng.choice(vals, len(vals), replace=True).mean()
    fin_out = {
        "day_mean_gain": float(vals.mean()),
        "ci95": [float(np.quantile(z, .025)), float(np.quantile(z, .975))],
        "days": int(len(vals)),
    }
    out = dict(pm_out)
    out["vs_pm_last"] = {
        "day_mean_gain": pm_out["day_mean_gain"],
        "ci95": pm_out["ci95"],
        "days": pm_out["days"],
    }
    out["vs_finance"] = fin_out
    out["primary_comparator"] = "finance_p"
    return out


# Freeze evaluation semantics before train/validation, so the same functions are
# serialized in the June artifact and later reused by the sealed July job.
core.probability_summary = probability_summary_with_finance_gate
core.paired_day_bootstrap = paired_day_bootstrap_with_finance


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


def _arg_after(flag: str) -> Path | None:
    try:
        return Path(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def _stamp_train_contract(out: Path):
    cp = out / "FROZEN_CONTRACT.json"
    sp = out / "train_summary.json"
    if not cp.exists():
        raise RuntimeError("frozen contract missing after train")
    contract = json.loads(cp.read_text())
    contract["evaluation_contract"] = EVALUATION_CONTRACT
    cp.write_text(json.dumps(contract, indent=2))
    if sp.exists():
        summary = json.loads(sp.read_text())
        summary["contract"] = contract
        summary["evaluation_contract"] = EVALUATION_CONTRACT
        sp.write_text(json.dumps(summary, indent=2))
    print("EVALUATION_CONTRACT_FROZEN", json.dumps(EVALUATION_CONTRACT, ensure_ascii=False), flush=True)


def _stamp_test_verdict(out: Path, model_dir: Path):
    sp = out / "test_summary.json"
    cp = model_dir / "FROZEN_CONTRACT.json"
    if not sp.exists() or not cp.exists():
        return
    summary = json.loads(sp.read_text())
    contract = json.loads(cp.read_text())
    ec = contract.get("evaluation_contract")
    if ec != EVALUATION_CONTRACT:
        raise RuntimeError("sealed evaluation contract mismatch")
    prob = summary["probability"]
    boot = summary["brier_gain_day_bootstrap"]["vs_finance"]
    gain = float(prob["brier_gain_vs_finance"])
    lower = float(boot["ci95"][0])
    if gain <= 0:
        verdict = "RED"
    elif lower > 0:
        verdict = "GREEN"
    else:
        verdict = "YELLOW"
    summary["evaluation_contract"] = ec
    summary["predeclared_primary_probability_verdict"] = {
        "verdict": verdict,
        "brier_gain_vs_finance": gain,
        "day_bootstrap_vs_finance": boot,
    }
    sp.write_text(json.dumps(summary, indent=2))
    print("PREDECLARED_PRIMARY_PROBABILITY_VERDICT", verdict, "gain", gain, "ci95", boot["ci95"], flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    stream.main()
    if phase == "train":
        out = _arg_after("--out")
        if out is not None:
            _stamp_train_contract(out)
    elif phase == "test":
        out = _arg_after("--out")
        model_dir = _arg_after("--model-dir")
        if out is not None and model_dir is not None:
            _stamp_test_verdict(out, model_dir)
