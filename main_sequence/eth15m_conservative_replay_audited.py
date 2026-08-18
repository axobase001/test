from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from main_sequence import eth15m_conservative_replay_safe as safe
from main_sequence.qualification_stats import daily_pnl_stats


def requested_out() -> Path | None:
    try:
        i = sys.argv.index("--out")
        return Path(sys.argv[i + 1])
    except Exception:
        return None


if __name__ == "__main__":
    safe.r.main()
    out = requested_out()
    if out is not None:
        p = out / "summary.json"
        if p.exists():
            data = json.loads(p.read_text())
            data["qualification_audit"] = {
                "anchor": "no-lookahead ETH Deribit temporal instrument universe; contemporaneous trade index_price moneyness filter",
                "spot_used_for_deribit_universe_selection": False,
                "execution_evidence": "same-second public taker-tape exact-price volume; chosen BUY/SELL level alone must contain >=2x required qty",
                "execution_evidence_is_resting_l1_depth": False,
                "fixed_ticket_usd": 5.0,
                "raw_gap_floor": 0.10,
                "net_edge_floor": 0.05,
                "ask_floor": 0.20,
            }
            trades_path = out / "trades.csv"
            if trades_path.exists():
                trades = pd.read_csv(trades_path)
                data["qualification_stats"] = daily_pnl_stats(trades, "exit", "pnl", unit="s", seed_offset=15)
            p.write_text(json.dumps(data, indent=2))
