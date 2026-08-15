from __future__ import annotations

import pandas as pd

from main_sequence import original_93d_dual_exit as replay

_original_replay_large = replay.replay_large


def replay_large_empty_safe(row, g, spot, bn, der):
    # Transport-only robustness fix: a condition can legitimately have no
    # post-decision public tape. Represent that as an empty normalized schema.
    # The frozen semantics then naturally produce no fresh entry/convergence
    # fill and fall back to settlement where applicable.
    if g is None or g.empty:
        g = pd.DataFrame(columns=["timestamp", "price", "size", "side_u", "outcome_l"])
    return _original_replay_large(row, g, spot, bn, der)


replay.replay_large = replay_large_empty_safe

if __name__ == "__main__":
    replay.main()
