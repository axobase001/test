from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from main_sequence import external_vinayak_v1 as base
from main_sequence.train_v0 import SEQ_LEN, SEQ_NAMES, safe_mid

CADENCE_MS = 30_000
SEQ_LENGTHS: list[int] = []
SEQ_SPANS_MS: list[int] = []
CLOCK_AUDIT: list[dict] = []


def epoch_to_ms(v: int | float) -> int:
    """Normalize epoch seconds/ms/us/ns to epoch milliseconds by magnitude."""
    x = int(float(v))
    ax = abs(x)
    while ax >= 10**14:
        x //= 1000
        ax = abs(x)
    if ax < 10**11:
        x *= 1000
    return int(x)


def build_sequence_cadence_matched(ctx, i: int) -> np.ndarray:
    """Rebuild only the GRU history on the training corpus' steady ~30 s cadence.

    Structural signal timestamps, current/static features, frozen scaler, frozen
    weights, threshold, and execution grading remain unchanged. Historical states
    are selected causally from the full external SlotCtx, not from an already
    truncated 32x1s Example sequence.
    """
    out = np.full((SEQ_LEN, len(SEQ_NAMES)), np.nan, dtype=np.float32)
    decision_ts = int(ctx.ts[i])
    close_ms = epoch_to_ms(ctx.meta.close_ts)
    open_ms = epoch_to_ms(ctx.meta.open_ts)

    # SlotMeta/SlotCtx implementations may expose boundaries in seconds or ms.
    # Audit the normalized boundary against the observed context instead of
    # assuming a unit. A valid BTC15m slot is approximately 900 seconds long.
    if len(CLOCK_AUDIT) < 8:
        CLOCK_AUDIT.append({
            "raw_open": int(ctx.meta.open_ts),
            "raw_close": int(ctx.meta.close_ts),
            "open_ms": open_ms,
            "close_ms": close_ms,
            "duration_s": (close_ms - open_ms) / 1000.0,
            "ctx_first_ms": int(ctx.ts[0]),
            "ctx_last_ms": int(ctx.ts[-1]),
            "decision_ms": decision_ts,
            "decision_s2c": (close_ms - decision_ts) / 1000.0,
        })

    if not (840_000 <= close_ms - open_ms <= 960_000):
        # Preserve causality and current-state inference, but do not fabricate
        # history if the slot boundary itself is inconsistent.
        idxs = [i]
    else:
        # Use a slot-clock-anchored 30s grid, matching the original collector's
        # steady cadence semantics. Every selected row must exist at or before the
        # target and be no more than 5s stale on this nominal 1s collector.
        first_target = open_ms
        last_target = min(decision_ts, close_ms)
        targets = np.arange(first_target, last_target + 1, CADENCE_MS, dtype=np.int64)
        idxs: list[int] = []
        for t in targets:
            j = int(np.searchsorted(ctx.ts, int(t), side="right") - 1)
            if j < 0 or j > i:
                continue
            age = int(t) - int(ctx.ts[j])
            if age < 0 or age > 5_000:
                continue
            if not idxs or j != idxs[-1]:
                idxs.append(j)
        if not idxs or idxs[-1] != i:
            idxs.append(i)
        idxs = idxs[-SEQ_LEN:]

    SEQ_LENGTHS.append(len(idxs))
    SEQ_SPANS_MS.append(int(ctx.ts[idxs[-1]]) - int(ctx.ts[idxs[0]]) if len(idxs) > 1 else 0)

    open_spot = float(ctx.meta.spot_at_open or 0.0)
    if open_spot <= 0 and idxs:
        open_spot = float(ctx.spot[idxs[0]])

    rows = []
    for j in idxs:
        yb, ya = float(ctx.yb[j]), float(ctx.ya[j])
        nb, na = float(ctx.nb[j]), float(ctx.na[j])
        ymid = safe_mid(yb, ya)
        ysp = ya - yb if 0 < yb < ya < 1 else float("nan")
        sp = float(ctx.spot[j])
        lr = math.log(sp / open_spot) if sp > 0 and open_spot > 0 else 0.0
        rows.append([
            yb, ya, nb, na,
            math.log1p(max(float(ctx.ybs[j]), 0.0)),
            math.log1p(max(float(ctx.yas[j]), 0.0)),
            math.log1p(max(float(ctx.nbs[j]), 0.0)),
            math.log1p(max(float(ctx.nas[j]), 0.0)),
            lr, float(ctx.s2c[j]) / 900.0, ymid, ysp,
        ])
    if rows:
        out[: len(rows)] = np.asarray(rows, dtype=np.float32)
    return out


def _arg_path(flag: str) -> Path | None:
    try:
        k = sys.argv.index(flag)
        return Path(sys.argv[k + 1])
    except (ValueError, IndexError):
        return None


def _dist(xs: list[int], scale: float = 1.0) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float) / scale
    return {
        "n": int(len(a)),
        "min": float(np.min(a)),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def main() -> None:
    # build_examples_external resolves this symbol from the base module at runtime.
    base.build_sequence = build_sequence_cadence_matched
    out = _arg_path("--out")
    base.main()

    audit = {
        "audit": "frozen_v1_sequence_cadence_sensitivity_v2",
        "cadence_ms": CADENCE_MS,
        "history_grid": "30 s slot-clock-anchored causal samples from full SlotCtx + unchanged current decision state",
        "signal_timing_changed": False,
        "static_features_changed": False,
        "weights_retrained": False,
        "scaler_refit": False,
        "threshold_changed": False,
        "execution_lens_changed": False,
        "clock_unit_parser": "magnitude-aware seconds/ms/us/ns -> ms",
        "clock_audit_examples": CLOCK_AUDIT,
        "sequence_lengths": _dist(SEQ_LENGTHS),
        "sequence_span_seconds": _dist(SEQ_SPANS_MS, 1000.0),
        "interpretation_boundary": (
            "Cadence-matched recurrent-history sensitivity only: structural decisions/static features remain on the independent 1 s collector. "
            "The recurrent history is rebuilt causally from the full SlotCtx at 30 s cadence; this is still not a retrained model or a new policy."
        ),
    }
    print("CADENCE_AUDIT", json.dumps(audit, sort_keys=True), flush=True)
    if out is not None:
        (out / "sequence_cadence_audit.json").write_text(json.dumps(audit, indent=2))
        summary_path = out / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["status"] = "EXTERNAL_BACKWARD_OOD_SEQUENCE_CADENCE_MATCHED_SENSITIVITY"
            summary["sequence_cadence_audit"] = audit
            summary.setdefault("caveats", []).append(audit["interpretation_boundary"])
            summary_path.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
