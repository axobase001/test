# Main Sequence execution candidate E1 — edge-preserving marketable limit

Frozen UTC date: 2026-08-16. Frozen before reading final-v3 July 2026 or February-April 2026 economics.

## Motivation from discovery data

Already-inspected final-v3 data in 2026-05-15..06-30 and 2026-08-01..08-14 show a material endogenous selection effect under the existing `strict5` lens: the current execution proof requires fresh decision+1..+5s BUY tape at or below the exact decision-witness price. In May/August, losing signals were substantially more likely than winning signals to satisfy this future-fill condition.

This file does not change or relabel v3. Existing strict5 remains the frozen v3 execution result.

## E1 rule frozen before untouched validation

Signal generation is **identical to final-v3**:

- BTC 15m only.
- Conservative fair from Binance 60m realized volatility and causal backward 30m median Deribit option-trade IV.
- Net edge floor = 0.03.
- Decision window = 60..600 seconds to close.
- Once per market.
- Side and decision timestamp are exactly those selected by the frozen v3 signal rule.
- Historical fee semantics are unchanged.

The only execution change is the entry limit:

1. At the decision timestamp, let `fair_decision` be the frozen conservative fair for the selected outcome.
2. Compute `p_cap` using only decision-time information as the largest admissible price in `[decision_witness_price, 1)` satisfying the same frozen entry inequality:

   `fair_decision - p_cap - fee_per_share_for_signal(market, p_cap) >= 0.03`.

   Solve deterministically to numerical tolerance <= 1e-6. No future fair, future price, outcome, or validation-month data may affect `p_cap`.
3. The decision-second witness is still **not** counted as our fill.
4. E1 entry requires fresh public taker-BUY tape for the chosen outcome during decision+1..+5 seconds, accumulating at least 5 shares at prices `<= p_cap`.
5. Fill cost uses the actual fresh tape prices and the same historical fee implementation as v3.
6. Large/small regime classification and all exit rules remain exactly v3. In particular, E1 does not imply the separate LARGE-only v4 candidate.
7. No latency search, threshold search, direction filter, time-of-day filter, month deletion, fee change, or sizing change.

## Untouched validation

Primary untouched E1 validation is:

- July 2026 final-v3 signal set;
- February, March, and April 2026 final-v3 signal sets.

May/June/August are discovery months for E1 and must not be counted as untouched validation.

## Required reporting

Report v3 strict5 and E1 side by side for every untouched month and pooled untouched data:

- original signals;
- strict entry count and fill rate;
- P(fill | eventual win) and P(fill | eventual loss) as a diagnostic only;
- PnL, cost, ROI, edge/share;
- 7-day moving-block bootstrap ROI interval where applicable;
- large vs small regime results;
- existing frozen $50/$5-base/$100-single-cap/$200-market-cap no-leverage bankroll stress path.

E1 is an execution-policy candidate, not permission to overwrite the v3 headline. If E1 fails untouched validation, it remains failed without further adjustment.
