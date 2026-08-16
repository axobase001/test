from __future__ import annotations

import math

# Importing the authoritative v3 module applies the causal Deribit universe,
# contemporaneous trade filtering, historical-fee policy, and paced Data API
# transport to final_recent_replay.base.
from main_sequence import final_recent_no_lookahead as v3

base = v3.base

# The old V3 used a fixed 3c post-fee hurdle.  This audit changes exactly one
# admission rule: any STRICTLY POSITIVE post-fee conservative edge is eligible.
# math.nextafter is the smallest representable positive float, so score_market's
# existing `edge >= THRESHOLD` condition becomes mathematically `edge > 0`
# without introducing a fitted epsilon or a new economic threshold.
STRICT_POSITIVE = math.nextafter(0.0, 1.0)
base.THRESHOLD = STRICT_POSITIVE

base.PROTOCOL["name"] = "Main Sequence V3 all-positive no-lookahead replay / 2026-08-16 freeze"
base.PROTOCOL["entry"]["net_edge_floor"] = "strictly_positive_after_historical_fee"
base.PROTOCOL["entry"]["net_edge_floor_numeric"] = STRICT_POSITIVE
base.PROTOCOL["entry"]["selection"] = (
    "At each causal decision second, evaluate Up and Down independently using the conservative external fair boundary, "
    "historical fee and same-second 5-share public BUY witness. Admit only sides with strictly positive net edge; "
    "if both are positive, choose the larger net edge. Enter at the first qualifying second in the frozen 60-600s window."
)
base.PROTOCOL["entry"]["directionality"] = (
    "Two-sided by complement: underpriced Up is expressed by buying Up; overvalued Up is expressed when Down is underpriced "
    "relative to 1-p, and vice versa. No shorting and no future outcome is used."
)
base.PROTOCOL["analysis_bins_predeclared"] = {
    "net_edge_cents": ["(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "(3,5]", "(5,10]", "(10,+inf)"],
    "entry_price": ["[0,0.05]", "(0.05,0.10]", "(0.10,0.25]", "(0.25,0.40]", "(0.40,0.60]", "(0.60,0.75]", "(0.75,0.90]", "(0.90,0.95]", "(0.95,1]"],
    "selected_conservative_fair": ["[0,0.05]", "(0.05,0.10]", "(0.10,0.25]", "(0.25,0.40]", "(0.40,0.60]", "(0.60,0.75]", "(0.75,0.90]", "(0.90,0.95]", "(0.95,1]"],
    "tail_favorite_definition": "selected conservative fair >= 0.95; descriptive bin only, never an entry gate",
}
base.PROTOCOL["anti_lookahead"].extend([
    "The 3c threshold is removed before replay. No sub-3c edge threshold is selected after observing returns.",
    "Edge and price bins are descriptive and predeclared before the new replay; they do not change entry, exit or sizing.",
    "Only observations timestamped <= each decision/exit timestamp may influence fair value or trade selection; final outcome is settlement-only.",
])


if __name__ == "__main__":
    base.main()
