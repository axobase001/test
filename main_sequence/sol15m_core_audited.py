from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

from main_sequence import eth15m_conservative_replay as r
from main_sequence import eth15m_conservative_replay_safe as safe
from main_sequence import final_recent_replay as pm_base
from main_sequence.qualification_stats import daily_pnl_stats
from main_sequence.sol_core_common import build_sol_anchors_causal, configure_sol_base

MIN_MAPPING_COVERAGE = 0.995


def fetch_sol_markets_hour(hour: int):
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-sol15m/1.0"})
    wanted = [(f"sol-updown-15m-{t0}", t0) for t0 in range(hour, hour + 3600, 900)]
    params = [("slug", s) for s, _ in wanted] + [("closed", "true"), ("limit", 20)]
    js = pm_base.get_json(sess, r.GAMMA + "/markets", params=params)
    by = {str(x.get("slug")): x for x in js if isinstance(x, dict)}
    mm, inv = [], []
    for slug, t0 in wanted:
        raw = by.get(slug)
        m = r.parse_market(raw or {}, t0)
        inv.append({"slug": slug, "start": t0, "exists": raw is not None, "mapped": m is not None})
        if m:
            mm.append(m)
    return mm, inv


def requested_out() -> Path | None:
    try:
        i = sys.argv.index("--out")
        return Path(sys.argv[i + 1])
    except Exception:
        return None


def configure() -> None:
    configure_sol_base()
    # Safe module has already replaced r.score_hour with the frozen exact-level
    # 2x taker-tape implementation. Only the asset/data adapters change here.
    r.fetch_markets_hour = fetch_sol_markets_hour
    r.build_anchors = build_sol_anchors_causal
    safe.r.fetch_markets_hour = fetch_sol_markets_hour
    safe.r.build_anchors = build_sol_anchors_causal


def audit(out: Path) -> None:
    p = out / "summary.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    data["asset"] = "SOL"
    expected = int(data.get("markets_expected") or 0)
    mapped = int(data.get("markets_mapped") or 0)
    mapping_coverage = mapped / expected if expected else 0.0
    data["qualification_audit"] = {
        "anchor": "no-lookahead Deribit SOL_USDC temporal instrument universe; contemporaneous actual-trade index_price moneyness filter",
        "spot_used_for_deribit_universe_selection": False,
        "official_resolution": "Gamma-resolved Polymarket outcome",
        "fair_anchor_role": "external Binance SOLUSDT + Deribit SOL_USDC actual option-trade IV cross-market valuation signal",
        "execution_evidence": "same-second public taker-tape exact-price volume; chosen BUY/SELL level alone must contain >=2x required qty",
        "execution_evidence_is_resting_l1_depth": False,
        "fixed_ticket_usd": 5.0,
        "raw_gap_floor": 0.10,
        "net_edge_floor": 0.05,
        "ask_floor": 0.20,
        "mapping_coverage": mapping_coverage,
        "minimum_mapping_coverage_for_green": MIN_MAPPING_COVERAGE,
        "bootstrap_day_key": "market start UTC day",
    }
    trades_path = out / "trades.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        stats = daily_pnl_stats(trades, "start", "pnl", unit="s", seed_offset=1515)
        data["qualification_stats"] = stats
        stat_grade = str(stats.get("grade"))
        if stat_grade.startswith("RED"):
            overall = "RED"
        elif mapping_coverage < MIN_MAPPING_COVERAGE:
            overall = "YELLOW_MAPPING_COVERAGE"
        else:
            overall = stat_grade
        data["overall_qualification_grade"] = overall
        if len(trades):
            recent = trades[pd.to_datetime(trades["decision"], unit="s", utc=True) >= pd.Timestamp("2026-07-15", tz="UTC")]
            data["recent_2026_07_15_to_08_15"] = {
                "trades": int(len(recent)),
                "pnl": float(recent["pnl"].sum()),
            }
    p.write_text(json.dumps(data, indent=2))
    print("SOL15M_AUDITED_FINAL", json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    configure()
    safe.r.main()
    out = requested_out()
    if out is not None:
        audit(out)
