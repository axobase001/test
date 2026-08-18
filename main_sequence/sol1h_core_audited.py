from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from main_sequence import eth15m_conservative_replay as asset_base
from main_sequence import eth1h_strict3_core as runner
from main_sequence import eth1h_strict3_core_causal as causal  # installs frozen strict 0.5G scoring
from main_sequence.qualification_stats import daily_pnl_stats
from main_sequence.sol_core_common import build_sol_anchors_causal, configure_sol_base

MIN_MAPPING_COVERAGE = 0.995
OUT = Path("eth1h_strict3_out")


def sol_slug_candidates(start: int) -> list[str]:
    d = runner.local_dt(start)
    month = d.strftime("%B").lower()
    h = d.hour % 12 or 12
    ap = "am" if d.hour < 12 else "pm"
    return list(dict.fromkeys([
        f"solana-up-or-down-{month}-{d.day}-{d.year}-{h}{ap}-et",
        f"solana-up-or-down-{month}-{d.day}-{d.year}-{h}-{ap}-et",
        f"solana-up-or-down-{month}-{d.day}-{h}{ap}-et",
        f"solana-up-or-down-{month}-{d.day}-{h}-{ap}-et",
        f"sol-updown-1h-{int(start)}",
    ]))


def configure() -> None:
    configure_sol_base()
    asset_base.build_anchors = build_sol_anchors_causal
    runner.eth.build_anchors = build_sol_anchors_causal
    runner.slug_candidates = sol_slug_candidates
    # importing causal already installs score_market_strict_stop; reassign explicitly
    # so this entrypoint documents the frozen stop semantics mechanically.
    runner.score_market = causal.score_market_strict_stop


def audit() -> None:
    p = OUT / "summary.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    data["asset"] = "SOL"
    if isinstance(data.get("rules"), dict):
        data["rules"]["fair"] = "conservative min/max boundary from Binance SOLUSDT 60m RV + backward Deribit SOL_USDC actual option-trade IV"
        data["rules"]["reference"] = "Binance SOLUSDT 1H open/close"
    expected = int(data.get("markets_expected") or 0)
    mapped = int(data.get("markets_mapped") or 0)
    mapping_coverage = mapped / expected if expected else 0.0
    data["qualification_audit"] = {
        "anchor": "no-lookahead Deribit SOL_USDC temporal instrument universe; contemporaneous actual-trade index_price moneyness filter",
        "spot_used_for_deribit_universe_selection": False,
        "execution_evidence": "same-second public taker-tape exact-price volume; entry, convergence exit and 0.5G stop each require >=3x required qty at the exact level",
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
    events_path = OUT / "events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        exits = events[events["event"].isin(["convergence", "stop_0.5G", "settlement"])].copy()
        stats = daily_pnl_stats(exits, "start", "pnl", unit="s", seed_offset=6060)
        data["qualification_stats"] = stats
        stat_grade = str(stats.get("grade"))
        if stat_grade.startswith("RED"):
            overall = "RED"
        elif mapping_coverage < MIN_MAPPING_COVERAGE:
            overall = "YELLOW_MAPPING_COVERAGE"
        else:
            overall = stat_grade
        data["overall_qualification_grade"] = overall
        recent = exits[pd.to_datetime(exits["start"], unit="s", utc=True) >= pd.Timestamp("2026-07-15", tz="UTC")]
        data["recent_2026_07_15_to_08_15"] = {
            "exits": int(len(recent)),
            "pnl": float(recent["pnl"].sum()),
        }
        if "fair_available_at_exit" in events.columns:
            stop_rows = events[events["event"] == "stop_0.5G"]
            data["stops_without_current_fair"] = int((stop_rows["fair_available_at_exit"] == False).sum())  # noqa: E712
    p.write_text(json.dumps(data, indent=2))
    print("SOL1H_AUDITED_FINAL", json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    configure()
    runner.main()
    audit()
