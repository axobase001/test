# Main Sequence dual-exit capacity-capped bankroll replay — 93d

Authoritative bankroll workflow: `31886342759`
Artifact: `main-sequence-dual-exit-capacity-bankroll`, artifact ID `9247351998`, SHA256 `ec13bfbb676ebe64a4f7afaf9a8c5f1229bb422d86ed0972ce2d1d4141eade38`.
Source corrected dual-exit replay: run `31885683821`.

## Frozen bankroll rules
- initial capital: $50
- base trade: $5
- target trade doubles only when realized/known accounting equity doubles
- tiers: $5 / $10 / $20 / $40 / $80 / $100
- single-trade hard cap: $100
- single-market/window hard cap: $200
- no leverage
- unresolved positions stay at cost basis for sizing; only realized exits change the tier
- no partial fill for lack of cash

These reproduce the historical `main_sequence/bankroll.py` rules. The $100 cap is interpreted as the founder-described PM15m opportunity-capacity ceiling, not as a risk-preference cap.

## Corrected dual-exit witness/book-like layer
- candidates: 7,661
- trades opened: 7,661
- skipped for cash: 0
- skipped market cap: 0
- initial capital: **$50.00**
- final capital after 92.992 days: **$135,595.01**
- realized profit: **$135,545.01**
- capital multiple: **2,711.90x**
- total return: **+271,090.02%**
- realized-cost-basis max drawdown: **39.43%**
- maximum simultaneously locked cost: **$100**
- maximum single-market exposure: **$100**

The account first reached the $100-per-trade capacity ceiling at `2026-03-02T13:30:00Z`. Stake counts were:
- $5: 24 trades
- $10: 11 trades
- $20: 85 trades
- $40: 13 trades
- $80: 0 trades (equity jumped across this tier)
- $100: **7,528 trades**

Thus 98.3% of all trades were already capacity-capped at $100; after the first ~1.6 days, additional account equity no longer increased per-opportunity stake. Growth therefore becomes opportunity-count/capacity limited rather than unconstrained exponential compounding.

Realized PnL by payoff regime under this bankroll replay:
- large-convergence book: **+$55,403.61**
- small-settlement book: **+$80,141.40**

Mechanical CAGR from the 93d $50→$135,595 path is astronomically large and should **not** be used as a forecast, because repeating the CAGR would incorrectly assume the early $5→$100 stake scaling can recur after the account is already at the $100 market-capacity ceiling. A more meaningful capacity-limited run-rate is the realized absolute PnL per unit time once the cap is saturated.

Linearizing the observed 93d absolute profit over 365 days gives roughly **$532k/year of gross historical replay PnL** at this BTC15m opportunity set and $100 hard per-trade cap, assuming comparable opportunity frequency/economics persist. This is an extrapolation, not a forward guarantee.

## Deliberately punitive fresh-5s trade-print layer
Using the same bankroll rules on the strict public-tape proxy:
- candidates: 2,294
- trades opened: 652
- skipped for lack of cash: 1,642
- final capital: **$4.42**
- total return: **-91.16%**
- max realized-cost-basis drawdown: **99.66%**
- never reached the $100 trade tier

This strict result is not directly comparable to the old full-depth local replay: it requires fresh public taker prints after both entry and exit witnesses and therefore intentionally rejects many orders that may have been executable through historical resting depth. The witness/book-like result, conversely, does not prove $100 of historical depth at every signal. A full-depth/queue-aware replay remains necessary to locate the true executable result between these bounds.
