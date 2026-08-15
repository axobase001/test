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


def build_sequence_cadence_matched(ctx, i: int) -> np.ndarray:
    """Rebuild only the GRU sequence on the training corpus' steady ~30 s cadence.

    This is intentionally a sensitivity audit, not a new strategy. Structural signal
    timestamps, current/static features, frozen scaler, frozen weights, threshold,
    and execution grading remain unchanged from external_vinayak_v1.py.

    The original public training corpus samples top-of-book roughly every 30 seconds
    outside the final ~5 seconds. Main Sequence decisions live 60--600 seconds from
    close, so a close-anchored 30 s causal grid is the closest reproducible cadence
    match for the recurrent history on this independent 1 s collector.
    """
    out = np.full((SEQ_LEN, len(SEQ_NAMES)), np.nan, dtype=np.float32)
    decision_ts = int(ctx.ts[i])
    close_ms = int(ctx.meta.close_ts) * 1000
    open_ms = int(ctx.meta.open_ts) * 1000

    # Training snapshots are effectively anchored to the 15 m slot clock. Select
    # only timestamps that existed by the decision; never look forward.
    first_k = max(0, int(math.ceil((open_ms - close_ms) / CADENCE_MS)))
    last_k = int(math.floor((decision_ts - close_ms) / CADENCE_MS))
    targets = []
    if last_k >= first_k:
        targets = [close_ms + k * CADENCE_MS for k in range(first_k, last_k + 1)]

    idxs: list[int] = []
    for t in targets:
        j = int(np.searchsorted(ctx.ts, int(t), side="right") - 1)
        if j < 0 or j > i:
            continue
        # Do not silently reuse very stale rows when the independent collector has
        # a gap. 5 s is already much looser than its nominal 1 s cadence.
        if int(t) - int(ctx.ts[j]) > 5_000:
            continue
        if not idxs or j != idxs[-1]:
            idxs.append(j)

    # Preserve the causal current state as the last recurrent observation, matching
    # train_v0.build_sequence(), while thinning only its history. This isolates the
    # sequence-cadence effect without changing signal timing/static features.
    if not idxs or idxs[-1] != i:
        idxs.append(i)
    idxs = idxs[-SEQ_LEN:]
    SEQ_LENGTHS.append(len(idxs))

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


def main() -> None:
    # build_examples_external resolves this symbol from the base module at runtime.
    base.build_sequence = build_sequence_cadence_matched
    out = _arg_path("--out")
    base.main()

    audit = {
        "audit": "frozen_v1_sequence_cadence_sensitivity",
        "cadence_ms": CADENCE_MS,
        "history_grid": "30 s close-anchored causal samples + unchanged current decision state",
        "signal_timing_changed": False,
        "static_features_changed": False,
        "weights_retrained": False,
        "scaler_refit": False,
        "threshold_changed": False,
        "execution_lens_changed": False,
        "sequence_lengths": {
            "n": len(SEQ_LENGTHS),
            "min": int(np.min(SEQ_LENGTHS)) if SEQ_LENGTHS else None,
            "p10": float(np.quantile(SEQ_LENGTHS, 0.10)) if SEQ_LENGTHS else None,
            "median": float(np.median(SEQ_LENGTHS)) if SEQ_LENGTHS else None,
            "p90": float(np.quantile(SEQ_LENGTHS, 0.90)) if SEQ_LENGTHS else None,
            "max": int(np.max(SEQ_LENGTHS)) if SEQ_LENGTHS else None,
            "mean": float(np.mean(SEQ_LENGTHS)) if SEQ_LENGTHS else None,
        },
        "interpretation_boundary": (
            "Sensitivity audit only: structural decisions/static features remain on the independent 1 s collector. "
            "This isolates recurrent-history cadence; it is not yet a full 30 s signal-context replay."
        ),
    }
    print("CADENCE_AUDIT", json.dumps(audit, sort_keys=True), flush=True)
    if out is not None:
        (out / "sequence_cadence_audit.json").write_text(json.dumps(audit, indent=2))
        summary_path = out / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["status"] = "EXTERNAL_BACKWARD_OOD_SEQUENCE_CADENCE_SENSITIVITY"
            summary["sequence_cadence_audit"] = audit
            summary.setdefault("caveats", []).append(audit["interpretation_boundary"])
            summary_path.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
