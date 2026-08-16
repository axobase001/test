from __future__ import annotations

from main_sequence import final_recent_no_lookahead as v3

base = v3.base

CORE_NET_EDGE = 0.03
TAIL_FAIR_FLOOR = 0.95

base.PROTOCOL["name"] = "Main Sequence V3 core+tail no-lookahead replay / 2026-08-16 freeze"
base.PROTOCOL["entry"]["net_edge_floor"] = {
    "core": CORE_NET_EDGE,
    "tail_favorite": "strictly_positive_after_historical_fee",
}
base.PROTOCOL["entry"]["tail_favorite_rule"] = (
    "The selected side's conservative external fair must be >=0.95 and its post-historical-fee edge must be strictly >0. "
    "This adds extreme favorite mispricing without admitting generic sub-3c noise around 50/50."
)
base.PROTOCOL["entry"]["selection"] = (
    "At each causal second in the frozen 60-600s window evaluate Up and Down independently. A side qualifies if "
    "post-fee conservative edge >=3c (legacy V3 core) OR conservative fair >=95% with post-fee edge >0 (tail favorite). "
    "If both sides qualify choose the larger post-fee edge; enter at the first qualifying second; once per market."
)
base.PROTOCOL["entry"]["directionality"] = (
    "Two-sided by complement. Underpriced Up is bought as Up. If Up is overvalued relative to the external fair, "
    "the complementary Down side can qualify and is bought instead, and vice versa. No shorting is required."
)
base.PROTOCOL["analysis_bins_predeclared"] = {
    "net_edge_cents": ["(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "[3,5]", "(5,10]", "(10,+inf)"],
    "selected_conservative_fair": ["[0.50,0.90)", "[0.90,0.95)", "[0.95,0.975)", "[0.975,0.99)", "[0.99,0.995)", "[0.995,1]"],
    "entry_price": ["[0,0.05]", "(0.05,0.10]", "(0.10,0.25]", "(0.25,0.40]", "(0.40,0.60]", "(0.60,0.75]", "(0.75,0.90]", "(0.90,0.95]", "(0.95,1]"],
    "signal_family": ["core_3c", "tail_favorite_sub3c"],
}
base.PROTOCOL["anti_lookahead"].extend([
    "The legacy 3c core and the >=95% tail-favorite sub-3c admission rule are frozen before opening this replay's results.",
    "The 95/97.5/99/99.5% probability bins are descriptive only; no threshold is selected after observing returns.",
    "Final market outcome is settlement-only and cannot influence signal family, side, fair, admission time, limit, or exit fair.",
])

for col in ("signal_family", "selected_conservative_fair", "signal_fee_per_share"):
    if col not in base.RECORD_COLUMNS:
        base.RECORD_COLUMNS.append(col)


def score_market_tail_core(m, g, spot, bn, der):
    if g is None or g.empty:
        return None
    pre = g[
        (g["timestamp"] >= m.close - base.MAX_S2C)
        & (g["timestamp"] <= m.close - base.MIN_S2C)
        & (g["side_u"] == "BUY")
    ]
    if pre.empty:
        return None

    for sec in sorted(int(x) for x in pre["timestamp"].unique().tolist()):
        qsec = pre[pre["timestamp"] == sec]
        fb = base.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        candidates = []
        for outcome in ("up", "down"):
            lim = base.witness_limit(qsec[qsec["outcome_l"] == outcome], base.SIZE)
            if lim is None:
                continue
            fair = float(fb[outcome])
            raw = fair - float(lim)
            fee_ps = float(base.fee_per_share_for_signal(m, float(lim)))
            edge = raw - fee_ps
            is_core = edge >= CORE_NET_EDGE
            is_tail = fair >= TAIL_FAIR_FLOOR and edge > 0.0
            if not (is_core or is_tail):
                continue
            family = "core_3c" if is_core else "tail_favorite_sub3c"
            candidates.append((float(edge), outcome, float(lim), float(raw), family, fair, fee_ps))

        if candidates:
            edge, outcome, limit, raw_gap, family, fair, fee_ps = max(
                candidates, key=lambda x: (x[0], x[1] == "up")
            )
            rec = base.execute_signal(
                m, g, sec, outcome, limit, edge, raw_gap, fb, spot, bn, der
            )
            rec["signal_family"] = family
            rec["selected_conservative_fair"] = fair
            rec["signal_fee_per_share"] = fee_ps
            return rec
    return None


base.score_market = score_market_tail_core


if __name__ == "__main__":
    base.main()
