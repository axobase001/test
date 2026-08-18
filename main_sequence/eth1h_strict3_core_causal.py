from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from main_sequence import eth15m_conservative_replay as eth
from main_sequence.eth_causal_anchor import build_eth_anchors_causal
from main_sequence import eth1h_strict3_core as runner
from main_sequence.qualification_stats import daily_pnl_stats

eth.build_anchors = build_eth_anchors_causal
runner.eth.build_anchors = build_eth_anchors_causal

if __name__ == "__main__":
    runner.main()
    p = Path("eth1h_strict3_out/summary.json")
    if p.exists():
        data = json.loads(p.read_text())
        data["qualification_audit"] = {
            "anchor": "no-lookahead ETH Deribit temporal instrument universe; contemporaneous trade index_price moneyness filter",
            "spot_used_for_deribit_universe_selection": False,
            "execution_evidence": "same-second public taker-tape exact-price volume; entry, convergence exit and 0.5G stop each require >=3x required qty at that exact level",
            "execution_evidence_is_resting_l1_depth": False,
            "same_second_reentry": False,
            "fixed_ticket_usd": 5.0,
            "net_convergence_edge_floor": 0.05,
            "stop_multiple_G": 0.5,
            "ask_floor": 0.20,
        }
        events_path = Path("eth1h_strict3_out/events.csv")
        if events_path.exists():
            events = pd.read_csv(events_path)
            exits = events[events["event"].isin(["convergence", "stop_0.5G", "settlement"])].copy()
            data["qualification_stats"] = daily_pnl_stats(exits, "time", "pnl", unit="s", seed_offset=60)
        p.write_text(json.dumps(data, indent=2))
