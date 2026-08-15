# Main Sequence LARGE-regime exit-control preregistration

Frozen UTC date: 2026-08-16, before reading final-v3 July 2026 or February-April 2026 economics.

This is a diagnostic control only. It does not modify final-v3 or the preregistered LARGE-only v4 candidate.

## Discovery observation

On already-inspected discovery slices, the causal convergence exit had non-stationary value relative to holding the exact same strict-filled LARGE entries to binary settlement:

- 2026-05-15..05-31 strict LARGE: convergence-policy ROI about +6.88%; same entries held to settlement about -11.83%.
- 2026-06 strict LARGE: convergence-policy ROI about +3.16%; settlement about -2.74%.
- 2026-08-01..08-14 strict LARGE: convergence-policy ROI about -1.02%; settlement about +9.47%.

Therefore exit value must be measured explicitly rather than assumed positive.

## Untouched validation control

For July 2026 and February-April 2026, for the exact same preregistered LARGE strict5 entry set:

1. Report actual frozen-v3 convergence/fallback reward.
2. Report a settlement-only counterfactual using the same actual strict entry cost and final binary payout, with no alternative entry selection.
3. Report `exit_uplift = actual_convergence_policy_pnl - settlement_counterfactual_pnl`.
4. Report counts of convergence exits and settlement fallbacks.
5. Do not select between convergence and settlement after seeing the validation set. The v4 candidate remains the previously frozen convergence-exit policy; settlement is only a mechanism control.

No signal, threshold, fee, direction, timing, sizing, or fill rule changes are permitted in this control.
