from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence import eth15m_conservative_replay as eth
from main_sequence.eth_causal_anchor import build_eth_anchors_causal
from main_sequence import eth1h_strict3_core as runner
from main_sequence.qualification_stats import daily_pnl_stats

MIN_MAPPING_COVERAGE = 0.995

eth.build_anchors = build_eth_anchors_causal
runner.eth.build_anchors = build_eth_anchors_causal


def score_market_strict_stop(m, spot, bn, der):
    """Qualification score: a frozen 0.5G stop never depends on current fair availability."""
    g = runner.v4.market_tape(m)
    if g is None or g.empty:
        return [], {"slug": m.slug, "tape": False}

    secs = sorted(int(x) for x in g.timestamp.dropna().unique() if m.start <= int(x) < m.close)
    pos = None
    events = []
    entries = conv = stops = settles = gate_hits = size_reject = 0
    pnl_total = 0.0
    last_exit_sec = -1
    stops_without_current_fair = 0

    for sec in secs:
        qsec = g[g.timestamp == sec]
        fb = runner.fair_boundary(m, sec, spot, bn, der)
        exited = False

        if pos is not None:
            lv = runner.v4.top_level(qsec, "SELL", pos["outcome"])
            if lv is not None:
                bid, avail = map(float, lv)
                qty = float(pos["qty"])
                if avail + 1e-12 >= runner.SIZE_MULT * qty:
                    proceeds = qty * bid - runner.v4.fee_total(m, bid, qty)
                    pnl = proceeds - float(pos["cost"])
                    reason = None
                    if fb is not None:
                        fair = float(fb[pos["outcome"]])
                        if bid >= fair - runner.FAIR_BAND - 1e-12 and pnl > 1e-12:
                            reason = "convergence"
                    if reason is None and -pnl + 1e-12 >= runner.STOP_MULT_G * float(pos["G"]):
                        reason = "stop_0.5G"
                        if fb is None:
                            stops_without_current_fair += 1
                    if reason:
                        pnl_total += pnl
                        conv += int(reason == "convergence")
                        stops += int(reason == "stop_0.5G")
                        events.append({
                            "time": sec, "event": reason, "side": pos["outcome"],
                            "entry": pos["ask"], "exit": bid, "qty": qty,
                            "pnl": pnl, "G": pos["G"],
                            "depth_x": avail / max(qty, 1e-12),
                            "fair_available_at_exit": bool(fb is not None),
                        })
                        pos = None
                        last_exit_sec = sec
                        exited = True

        if pos is None and not exited and sec > last_exit_sec and fb is not None:
            cands = []
            for outcome in ("up", "down"):
                lv = runner.v4.top_level(qsec, "BUY", outcome)
                if lv is None:
                    continue
                ask, avail = map(float, lv)
                if not (runner.ASK_FLOOR <= ask < 1.0):
                    continue
                qty = runner.qty_for_budget(m, ask)
                if qty <= 0:
                    continue
                cost = qty * ask + runner.v4.fee_total(m, ask, qty)
                fair = float(fb[outcome])
                target = qty * fair - runner.v4.fee_total(m, min(max(fair, 1e-6), 1 - 1e-6), qty)
                G = target - cost
                edge_ps = G / qty
                if edge_ps + 1e-12 >= runner.EDGE_FLOOR:
                    gate_hits += 1
                    if avail + 1e-12 < runner.SIZE_MULT * qty:
                        size_reject += 1
                        continue
                    cands.append((G, outcome, ask, avail, qty, cost, fair, edge_ps))
            if cands:
                G, outcome, ask, avail, qty, cost, fair, edge_ps = max(cands, key=lambda x: x[0])
                pos = {
                    "outcome": outcome, "ask": ask, "qty": qty, "cost": cost,
                    "G": G, "entry": sec, "fair": fair, "edge_ps": edge_ps,
                }
                entries += 1
                events.append({
                    "time": sec, "event": "entry", "side": outcome,
                    "entry": ask, "exit": np.nan, "qty": qty, "pnl": 0.0,
                    "G": G, "edge_ps": edge_ps,
                    "depth_x": avail / max(qty, 1e-12),
                    "fair_available_at_exit": np.nan,
                })

    if pos is not None:
        won = (m.label_up >= 0.5) if pos["outcome"] == "up" else (m.label_up < 0.5)
        payout = pos["qty"] if won else 0.0
        pnl = payout - pos["cost"]
        pnl_total += pnl
        settles += 1
        events.append({
            "time": m.close, "event": "settlement", "side": pos["outcome"],
            "entry": pos["ask"], "exit": 1.0 if won else 0.0,
            "qty": pos["qty"], "pnl": pnl, "G": pos["G"],
            "edge_ps": pos["edge_ps"], "depth_x": np.nan,
            "fair_available_at_exit": np.nan,
        })

    return events, {
        "slug": m.slug, "tape": True, "entries": entries,
        "convergence": int(conv), "stops": int(stops), "settlements": settles,
        "pnl": pnl_total, "gate_hits": gate_hits, "size_reject": size_reject,
        "stops_without_current_fair": int(stops_without_current_fair),
    }


runner.score_market = score_market_strict_stop


if __name__ == "__main__":
    runner.main()
    p = Path("eth1h_strict3_out/summary.json")
    if p.exists():
        data = json.loads(p.read_text())
        expected = int(data.get("markets_expected") or 0)
        mapped = int(data.get("markets_mapped") or 0)
        mapping_coverage = mapped / expected if expected else 0.0
        data["qualification_audit"] = {
            "anchor": "no-lookahead ETH Deribit temporal instrument universe; contemporaneous trade index_price moneyness filter",
            "spot_used_for_deribit_universe_selection": False,
            "execution_evidence": "same-second public taker-tape exact-price volume; entry, convergence exit and 0.5G stop each require >=3x required qty at that exact level",
            "execution_evidence_is_resting_l1_depth": False,
            "same_second_reentry": False,
            "stop_requires_current_fair": False,
            "fixed_ticket_usd": 5.0,
            "net_convergence_edge_floor": 0.05,
            "stop_multiple_G": 0.5,
            "ask_floor": 0.20,
            "mapping_coverage": mapping_coverage,
            "minimum_mapping_coverage_for_green": MIN_MAPPING_COVERAGE,
            "bootstrap_day_key": "market start UTC day",
        }
        events_path = Path("eth1h_strict3_out/events.csv")
        if events_path.exists():
            events = pd.read_csv(events_path)
            exits = events[events["event"].isin(["convergence", "stop_0.5G", "settlement"])].copy()
            stats = daily_pnl_stats(exits, "start", "pnl", unit="s", seed_offset=60)
            data["qualification_stats"] = stats
            stat_grade = str(stats.get("grade"))
            if stat_grade.startswith("RED"):
                overall = "RED"
            elif mapping_coverage < MIN_MAPPING_COVERAGE:
                overall = "YELLOW_MAPPING_COVERAGE"
            else:
                overall = stat_grade
            data["overall_qualification_grade"] = overall
            if "fair_available_at_exit" in events.columns:
                stop_rows = events[events["event"] == "stop_0.5G"].copy()
                data["stops_without_current_fair"] = int((stop_rows["fair_available_at_exit"] == False).sum())  # noqa: E712
        p.write_text(json.dumps(data, indent=2))
