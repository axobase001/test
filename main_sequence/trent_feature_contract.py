from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "trentmkelly/polymarket_crypto_derivatives"
REV = "6be20463ce33795178c121e7bd15ed428904b5bd"
EPISODE = "btc15m_market1572828_2026-03-14_00-00-00_all"
FILES = ["steps.parquet", "book_levels.parquet"]


def q(x: np.ndarray | pd.Series) -> dict:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)), "min": float(np.min(a)), "p10": float(np.quantile(a, .1)),
        "median": float(np.median(a)), "p90": float(np.quantile(a, .9)),
        "max": float(np.max(a)), "mean": float(np.mean(a)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)

    local = {}
    for name in FILES:
        rel = f"{EPISODE}/{name}"
        local[name] = Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=REV, cache_dir=args.cache))

    step_cols = [
        "step_index", "ts", "progress", "chainlink_price", "binance_price",
        "up_best_bid", "up_best_ask", "up_bid_size_total", "up_ask_size_total",
        "down_best_bid", "down_best_ask", "down_bid_size_total", "down_ask_size_total",
    ]
    steps = pd.read_parquet(local["steps.parquet"], columns=step_cols).sort_values("step_index").reset_index(drop=True)
    # Pull only level-0 rows. These are the candidate true top-of-book prices/sizes.
    levels = pq.read_table(
        local["book_levels.parquet"],
        columns=["step_index", "outcome", "side", "level_index", "price", "size"],
        filters=[("level_index", "=", 0)],
    ).to_pandas()
    levels = levels.sort_values(["step_index", "outcome", "side"]).drop_duplicates(["step_index", "outcome", "side"], keep="last")

    # Infer outcome/side integer semantics from price agreement rather than assuming docs.
    mappings = []
    for up_outcome in (0, 1):
        down_outcome = 1 - up_outcome
        for bid_side in (0, 1):
            ask_side = 1 - bid_side
            spec = {
                "up_best_bid": (up_outcome, bid_side), "up_best_ask": (up_outcome, ask_side),
                "down_best_bid": (down_outcome, bid_side), "down_best_ask": (down_outcome, ask_side),
            }
            joined = steps[["step_index", *spec.keys()]].copy()
            errors = []
            match_stats = {}
            for col, (outcome, side) in spec.items():
                z = levels[(levels.outcome == outcome) & (levels.side == side)][["step_index", "price", "size"]].rename(columns={"price": f"{col}_l0", "size": f"{col}_l0_size"})
                joined = joined.merge(z, on="step_index", how="left")
                a = pd.to_numeric(joined[col], errors="coerce")
                b = pd.to_numeric(joined[f"{col}_l0"], errors="coerce")
                ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
                err = np.abs(a[ok] - b[ok])
                errors.extend(err.tolist())
                match_stats[col] = {
                    "n": int(ok.sum()),
                    "mae": float(err.mean()) if len(err) else None,
                    "exact_1e6_rate": float((err <= 1e-6).mean()) if len(err) else None,
                    "within_1c_rate": float((err <= .01 + 1e-12).mean()) if len(err) else None,
                }
            mappings.append({
                "up_outcome": up_outcome, "bid_side": bid_side,
                "mean_abs_error": float(np.mean(errors)) if errors else float("inf"),
                "matches": match_stats,
            })
    mappings.sort(key=lambda x: x["mean_abs_error"])
    best = mappings[0]

    # Reconstruct top sizes using the empirically verified mapping.
    up_outcome = int(best["up_outcome"]); down_outcome = 1 - up_outcome
    bid_side = int(best["bid_side"]); ask_side = 1 - bid_side
    size_map = {
        "up_bid_top_size": (up_outcome, bid_side), "up_ask_top_size": (up_outcome, ask_side),
        "down_bid_top_size": (down_outcome, bid_side), "down_ask_top_size": (down_outcome, ask_side),
    }
    s = steps[["step_index", "ts", "up_bid_size_total", "up_ask_size_total", "down_bid_size_total", "down_ask_size_total"]].copy()
    for col, (outcome, side) in size_map.items():
        z = levels[(levels.outcome == outcome) & (levels.side == side)][["step_index", "size"]].rename(columns={"size": col})
        s = s.merge(z, on="step_index", how="left")

    size_ratio = {}
    for prefix in ("up_bid", "up_ask", "down_bid", "down_ask"):
        total = pd.to_numeric(s[f"{prefix}_size_total"], errors="coerce").to_numpy(float)
        top = pd.to_numeric(s[f"{prefix}_top_size"], errors="coerce").to_numpy(float)
        ok = np.isfinite(total) & np.isfinite(top) & (top > 0)
        size_ratio[prefix] = {
            "coverage": float(ok.mean()),
            "top_size": q(top[ok]),
            "total_over_top": q((total[ok] / top[ok])),
        }

    ts = pd.to_numeric(steps.ts, errors="coerce").to_numpy(np.int64)
    dt = np.diff(ts)
    # Episode filename is the canonical 00:00 UTC market start; close is +15m.
    open_ms = int(pd.Timestamp("2026-03-14T00:00:00Z").timestamp() * 1000)
    close_ms = open_ms + 900_000
    s2c = (close_ms - ts) / 1000.0
    decision_idx = np.flatnonzero((s2c >= 60) & (s2c <= 600))
    raw_spans = np.asarray([ts[i] - ts[i - 31] for i in decision_idx if i >= 31], dtype=float)

    # Training-cadence reconstruction: 30s close-anchored causal samples. Measure
    # history span and sequence length available at each canonical decision grid.
    cadence_lens = []
    cadence_spans = []
    causal_ages = []
    for dec_s2c in range(600, 59, -30):
        decision_t = close_ms - dec_s2c * 1000
        hist_targets = list(range(open_ms, decision_t + 1, 30_000))
        idxs = []
        for target in hist_targets:
            j = int(np.searchsorted(ts, target, side="right") - 1)
            if j < 0:
                continue
            age = target - int(ts[j])
            if age < 0 or age > 500:
                continue
            causal_ages.append(age)
            if not idxs or j != idxs[-1]:
                idxs.append(j)
        idxs = idxs[-32:]
        if idxs:
            cadence_lens.append(len(idxs))
            cadence_spans.append(int(ts[idxs[-1]] - ts[idxs[0]]))

    report = {
        "status": "TRENT_V1_FEATURE_CONTRACT_AUDIT",
        "dataset": REPO, "revision": REV, "episode": EPISODE,
        "steps_rows": int(len(steps)), "level0_rows": int(len(levels)),
        "raw_step_dt_ms": q(dt),
        "raw_last32_span_ms_decision_region": q(raw_spans),
        "mapping_candidates": mappings,
        "verified_mapping": {
            "up_outcome": up_outcome, "down_outcome": down_outcome,
            "bid_side": bid_side, "ask_side": ask_side,
            "mean_abs_price_error": float(best["mean_abs_error"]),
            "matches": best["matches"],
        },
        "top_size_contract": size_ratio,
        "training_cadence_reconstruction": {
            "grid_ms": 30000,
            "decision_s2c": "60..600 every 30s",
            "causal_step_age_ms": q(causal_ages),
            "sequence_length": q(cadence_lens),
            "sequence_span_ms": q(cadence_spans),
        },
        "boundary": (
            "This audit verifies reconstructability and cadence only. It does not use resolved outcome for signals, "
            "does not run frozen weights, and does not claim executable PnL."
        ),
    }
    (args.out / "trent_feature_contract.json").write_text(json.dumps(report, indent=2))
    s.head(500).to_csv(args.out / "top_size_sample.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
