# Main Sequence dual-exit bankroll sweep — 93d

Authoritative sweep workflow: `31887019595`.
Artifact: `main-sequence-dual-exit-bankroll-sweep`, artifact ID `9247527549`, SHA256 `2cc6b265f9e21e46b67109e277a55cc35e272493e0d2675782e6bfd18901b058`.
Source corrected dual-exit replay: run `31885683821`.

## Sweep protocol
- Initial capital: $10 / $20 / $50 / $75 / $100.
- Base trade = 10% of initial capital.
- Single-trade hard cap: $100 / $200 / $300.
- Single-market/window cap = 2x single-trade cap: $200 / $400 / $600.
- Trade size doubles only when realized/known equity doubles, then clips at the hard cap.
- No leverage; unresolved positions stay at cost basis for sizing.

## Witness/book-like layer

| Initial | Trade cap | 93d final capital | 93d profit | Days to cap | Post-cap annual PnL run-rate |
|---:|---:|---:|---:|---:|---:|
| $10 | $100 | $132,703 | $132,693 | 1.87 | $526,359 |
| $10 | $200 | $263,282 | $263,272 | 2.17 | $1,046,948 |
| $10 | $300 | $392,316 | $392,306 | 2.30 | $1,558,269 |
| $20 | $100 | $133,733 | $133,713 | 1.56 | $528,770 |
| $20 | $200 | $265,407 | $265,387 | 1.87 | $1,052,718 |
| $20 | $300 | $396,308 | $396,288 | 2.17 | $1,570,422 |
| $50 | $100 | $135,595 | $135,545 | 1.55 | $529,720 |
| $50 | $200 | $268,157 | $268,107 | 1.56 | $1,057,735 |
| $50 | $300 | $400,509 | $400,459 | 1.56 | $1,586,117 |
| $75 | $100 | $137,041 | $136,966 | 1.55 | $529,720 |
| $75 | $200 | $269,744 | $269,669 | 1.55 | $1,059,440 |
| $75 | $300 | $402,236 | $402,161 | 1.56 | $1,586,602 |
| $100 | $100 | $138,488 | $138,388 | 1.55 | $529,720 |
| $100 | $200 | $271,190 | $271,090 | 1.55 | $1,059,440 |
| $100 | $300 | $403,892 | $403,792 | 1.55 | $1,589,160 |

All 15 witness combinations opened all 7,661 candidate signals and had zero cash skips. The realized-cost-basis Max DD was about 39.4% for nearly all combinations; the very small $10 initial cases with $200/$300 caps showed larger transient DD (43.3% / 50.8%) while climbing through the early tiers.

The capacity-saturated post-cap run-rate is therefore approximately:
- $100 cap: ~$0.53m/year
- $200 cap: ~$1.06m/year
- $300 cap: ~$1.59m/year

This is almost exactly linear in the cap because the account reaches the cap within roughly two days and spends the rest of the sample capacity-limited.

## Strict fresh-5s trade-print lower-bound layer

The deliberately punitive strict proxy is highly path-dependent and should not be used to select the bankroll parameters. With 10% initial sizing, $10/$20/$50 starts ended near -91% because cash starvation caused 1,642 signals to be skipped. The $100 initial / $100 cap combination was the notable exception: it opened all 2,294 strict candidates and ended at $17,035.96, but still suffered a 77.0% realized-cost-basis Max DD. This instability is another reason the public fresh-print proxy is unsuitable as the primary capacity-sizing estimator; it is an intentionally harsh lower bound rather than a full-depth execution replay.
