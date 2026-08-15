from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "trentmkelly/polymarket_crypto_derivatives"
REV = "6be20463ce33795178c121e7bd15ed428904b5bd"
EPISODE = "btc15m_market1572828_2026-03-14_00-00-00_all"


def stats(a) -> dict:
    x = np.asarray(list(a), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def best(book: dict[float, float], is_ask: bool) -> tuple[float | None, float | None]:
    live = [(float(p), float(s)) for p, s in book.items() if math.isfinite(p) and math.isfinite(s) and s > 1e-12]
    if not live:
        return None, None
    p = min(x[0] for x in live) if is_ask else max(x[0] for x in live)
    return p, float(book[p])


def init_books(levels0: pd.DataFrame) -> dict[tuple[int, int], dict[float, float]]:
    books: dict[tuple[int, int], dict[float, float]] = defaultdict(dict)
    for r in levels0.itertuples(index=False):
        p = float(r.price); s = float(r.size)
        if 0 < p < 1 and s > 0:
            books[(int(r.outcome), int(r.side))][p] = s
    return books


def ground_truth_top(level0: pd.DataFrame) -> dict[tuple[int, int, int], tuple[float, float]]:
    out = {}
    for r in level0.itertuples(index=False):
        out[(int(r.step_index), int(r.outcome), int(r.side))] = (float(r.price), float(r.size))
    return out


def run_candidate(
    steps: pd.DataFrame,
    events: pd.DataFrame,
    levels0_truth: dict,
    initial_levels: pd.DataFrame,
    side_col: str,
    update_mode: str,
) -> dict:
    books = init_books(initial_levels)
    base_step = int(initial_levels.step_index.iloc[0])
    ev = events[(events.event_type == 2) & (events.following_step_index > base_step)].copy()
    ev = ev.dropna(subset=["following_step_index", "price", "size", "is_down", side_col])
    ev = ev.sort_values(["following_step_index", "event_index", "ts"])
    grouped = {int(k): g for k, g in ev.groupby("following_step_index", sort=True)}

    price_err = []
    size_log_err = []
    exact_price = 0
    within_1c = 0
    size_rel_10pct = 0
    comparisons = 0
    missing_state = 0
    examples = []

    for step in steps.step_index.astype(int).tolist():
        if step <= base_step:
            continue
        g = grouped.get(step)
        if g is not None:
            for r in g.itertuples(index=False):
                outcome = int(bool(r.is_down))  # verified level mapping: 0 UP, 1 DOWN
                side = int(bool(getattr(r, side_col)))  # candidate: 0 bid, 1 ask
                p = float(r.price); sz = float(r.size)
                if not (0 < p < 1) or not math.isfinite(sz):
                    continue
                b = books[(outcome, side)]
                if update_mode == "absolute":
                    if sz <= 1e-12:
                        b.pop(p, None)
                    else:
                        b[p] = sz
                elif update_mode == "delta":
                    new = float(b.get(p, 0.0)) + sz
                    if new <= 1e-12:
                        b.pop(p, None)
                    else:
                        b[p] = new
                else:
                    raise ValueError(update_mode)

        for outcome in (0, 1):
            for side in (0, 1):
                truth = levels0_truth.get((step, outcome, side))
                if truth is None:
                    continue
                rp, rs = best(books[(outcome, side)], is_ask=bool(side))
                if rp is None or rs is None:
                    missing_state += 1
                    continue
                tp, tsz = truth
                pe = abs(rp - tp)
                price_err.append(pe)
                exact_price += int(pe <= 1e-9)
                within_1c += int(pe <= 0.01 + 1e-12)
                comparisons += 1
                if rs > 0 and tsz > 0 and pe <= 1e-9:
                    le = abs(math.log(max(rs, 1e-12) / max(tsz, 1e-12)))
                    size_log_err.append(le)
                    size_rel_10pct += int(abs(rs / tsz - 1.0) <= 0.10)
                if len(examples) < 12 and (pe > 0.01 or (pe <= 1e-9 and rs > 0 and tsz > 0 and abs(rs / tsz - 1) > .25)):
                    examples.append({
                        "step": step, "outcome": outcome, "side": side,
                        "recon_price": rp, "truth_price": tp,
                        "recon_size": rs, "truth_size": tsz,
                        "price_error": pe,
                    })

    score = (
        (exact_price / comparisons if comparisons else 0.0)
        - 0.05 * float(np.mean(price_err) if price_err else 1.0)
        - 0.01 * float(np.mean(size_log_err) if size_log_err else 10.0)
    )
    return {
        "side_col": side_col,
        "update_mode": update_mode,
        "base_step": base_step,
        "book_update_events": int(len(ev)),
        "comparisons": comparisons,
        "missing_state": missing_state,
        "exact_price_rate": float(exact_price / comparisons) if comparisons else None,
        "within_1c_price_rate": float(within_1c / comparisons) if comparisons else None,
        "price_abs_error": stats(price_err),
        "size_log_abs_error_when_exact_price": stats(size_log_err),
        "size_within_10pct_rate_when_exact_price": float(size_rel_10pct / len(size_log_err)) if size_log_err else None,
        "score": float(score),
        "mismatch_examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)

    local = {}
    for name in ("steps.parquet", "events.parquet", "book_levels.parquet"):
        rel = f"{EPISODE}/{name}"
        local[name] = Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=REV, cache_dir=args.cache))

    steps = pd.read_parquet(local["steps.parquet"], columns=["step_index", "ts"]).sort_values("step_index")
    events = pd.read_parquet(
        local["events.parquet"],
        columns=["following_step_index", "event_index", "event_type", "ts", "is_down", "is_sell", "is_sell_side", "price", "size"],
    )
    levels0 = pq.read_table(
        local["book_levels.parquet"],
        columns=["step_index", "outcome", "side", "level_index", "price", "size"],
        filters=[("level_index", "=", 0)],
    ).to_pandas().sort_values(["step_index", "outcome", "side"]).drop_duplicates(["step_index", "outcome", "side"], keep="last")

    first_complete = (
        levels0.groupby("step_index").size().loc[lambda s: s >= 4].index.astype(int).min()
    )
    # Full book only for the one initialization step. The whole-run truth comparison
    # remains level-0 only, so production reconstruction would not need full-depth
    # snapshots after initialization.
    initial_levels = pq.read_table(
        local["book_levels.parquet"],
        columns=["step_index", "outcome", "side", "level_index", "price", "size"],
        filters=[("step_index", "=", int(first_complete))],
    ).to_pandas()
    truth = ground_truth_top(levels0)

    candidates = []
    for side_col in ("is_sell_side", "is_sell"):
        for mode in ("absolute", "delta"):
            candidates.append(run_candidate(steps, events, truth, initial_levels, side_col, mode))
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best_c = candidates[0]

    report = {
        "status": "TRENT_EVENT_RECONSTRUCTION_AUDIT",
        "dataset": REPO,
        "revision": REV,
        "episode": EPISODE,
        "steps": int(len(steps)),
        "events": int(len(events)),
        "level0_truth_rows": int(len(levels0)),
        "initial_step": int(first_complete),
        "initial_full_book_rows": int(len(initial_levels)),
        "candidates": candidates,
        "best": best_c,
        "production_gate": {
            "exact_price_rate_min": 0.99,
            "within_1c_rate_min": 0.995,
            "size_within_10pct_rate_min": 0.95,
            "passed": bool(
                (best_c.get("exact_price_rate") or 0) >= 0.99
                and (best_c.get("within_1c_price_rate") or 0) >= 0.995
                and (best_c.get("size_within_10pct_rate_when_exact_price") or 0) >= 0.95
            ),
        },
        "boundary": (
            "This reconstructability audit uses book_levels only as initialization and ground truth. "
            "It does not use resolved outcome, does not run policy weights, and does not claim PnL."
        ),
    }
    (args.out / "trent_event_reconstruction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
