# Main Sequence v4 candidate — LARGE-only preregistration

Frozen UTC date: 2026-08-16.

## Status

Exploratory candidate generated from already-inspected final-v3 strict5 results in 2026-05-15..2026-06-30 and 2026-08-01..2026-08-14. This file does **not** rewrite the v3 result and does **not** authorize post-hoc deletion of losing v3 trades.

## Discovery evidence already seen before this preregistration

Under the unchanged final-v3 strict5 execution lens:

- large_convergence (initial raw gap >= 0.10): 95 entries, aggregate PnL approximately +5.11 on cost approximately 168.10, ROI approximately +3.04%.
- small_settlement: 1057 entries, aggregate PnL approximately -95.01, ROI approximately -5.11%.
- large-convergence monthly slices already inspected: 2026-05-15..05-31 approximately +6.88% ROI; 2026-06 approximately +3.16%; 2026-08-01..08-14 approximately -1.02%.

These figures are discovery data only and must not be counted as untouched validation evidence for v4.

## Candidate policy frozen before reading validation months

Starting from the exact frozen final-v3 protocol:

1. Keep BTC 15m only.
2. Keep conservative fair construction unchanged: Binance BTCUSDT 60m realized volatility plus causal backward 30m median Deribit option-trade IV.
3. Keep net entry edge floor = 0.03.
4. Keep decision window = 60..600 seconds to close.
5. Keep once-per-market rule unchanged.
6. Keep historical fee model and metadata/fallback policy unchanged.
7. Keep strict5 entry execution unchanged: decision-second witness is not a fill; entry requires fresh full-size BUY evidence during decision+1..+5 seconds at or below the frozen entry limit.
8. **New and only policy change:** trade only markets whose already-defined `initial_raw_gap >= 0.10` (`large_convergence`). All `small_settlement` signals abstain.
9. Keep large-regime exit unchanged: recompute fair causally at subsequent SELL-witness seconds, exit at first sellable price within 0.01 of fair, with the existing strict fresh SELL confirmation; fallback to settlement.
10. No new threshold search, no direction filter, no time-of-day filter, no month deletion, no latency tuning, no fee change, no sizing change.

## Untouched validation set

Primary untouched validation after this timestamp:

- 2026-07-01..2026-07-31 final-v3 shard (job was still running and its economics had not been read when this file was frozen).
- 2026-02-01..2026-04-30 final-v3 shards (jobs were still running and their economics had not been read when this file was frozen).

Discovery months May/June/August must remain visibly separated from these validation months in any report.

## Validation outputs to report without selection

For each untouched month and pooled untouched set, report:

- number of large signals and strict5 entries;
- strict5 PnL, cost, ROI, edge/share;
- 7-day moving-block bootstrap ROI interval where enough calendar days exist;
- convergence-exit rate and settlement fallback count;
- $50 / $5-base / $100-single-cap / $200-market-cap no-leverage bankroll replay using the already-frozen sizing rule;
- month sign consistency.

No rule change is allowed after seeing July or February-April results if those results are to retain validation status.

## Separate execution-research boundary

The known difference between decision-witness opportunity and future-tape strict5 fill selection is an execution-measurement question, not part of this v4 policy change. Historical order-book / ordered-fill audits may be used to characterize that gap, but may not retroactively relabel strict5 v3 or v4 trades.
