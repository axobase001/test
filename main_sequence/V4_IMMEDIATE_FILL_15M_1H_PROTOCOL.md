# Main Sequence V4 — Immediate-Quote Execution + 1H CORE/TAIL

Frozen 2026-08-16 before opening the new 1H result.

## Portfolio sizing
- Initial bankroll: $50.
- Base ticket: $5 total entry cost.
- Current-equity power-of-two tiers are inherited from the prior frozen bankroll protocol: $5 at $50, $10 at $100, $20 at $200, etc.; symmetric tier downshift after drawdown remains unchanged from the accepted prior protocol.
- No leverage.
- Combined open entry capital in one Polymarket market is capped at $200 across CORE + TAIL.
- Released capital is immediately reusable.

## Execution model — replaces future-5-second strict proxy
The old rule requiring a fresh print at the old limit during decision+1..+5s is retired as an execution model because it mechanically selects adverse post-signal moves.

Primary historical execution is `same_second_immediate`:
1. At second t, compute fair value using information timestamped <= t.
2. Observe the best executable first-level price at t. When full historical L1 is unavailable, a same-second public taker print/depth witness at the admitted price is the conservative observable proxy that the quote existed at order-send time.
3. If the signal qualifies and the same-second observable quantity supports the dollar-ticket-derived shares, submit immediately and count the order as filled at that observed level. A quote disappearing at t+1 does not invalidate a t fill.
4. No future print is allowed to decide whether the t order filled.
5. Full historical L1/depth archives, where available, are used only as an execution-proxy calibration check, not to retune signal thresholds.

## 15m policy
Signal/fair rules remain unchanged from frozen V3 CORE+TAIL:
- CORE: conservative post-entry-fee edge >= 3c.
- TAIL: conservative selected outcome fair >=95% and positive post-entry-fee edge below the 3c CORE threshold.
- CORE and TAIL may both trade the same 15m market.
- Existing exit logic remains: large raw gap >=10c seeks causal convergence; smaller CORE/TAIL positions may settle under the prior frozen policy.
- Only the execution admission layer changes from future-5s strict to same-second immediate.

## 1h market reference
BTC hourly Up/Down resolves against the Binance BTC/USDT 1H candle open/close, so the BS reference open and live underlying are kept in the same Binance price system.

## 1h fair value
- Underlying: Binance BTCUSDT.
- Reference strike/open: Binance BTCUSDT 1H candle open for the hourly market.
- Remaining maturity: seconds to the hourly market close.
- Two causal volatility anchors are retained for robustness: Binance realized volatility and backward-looking Deribit option-trade IV.
- Up conservative fair = min(p_RV, p_IV); Down conservative fair = 1 - max(p_RV, p_IV).
- No future price/outcome enters the signal.

## 1h CORE — repeatable convergence book
- A market may contain multiple sequential CORE round trips.
- At most one CORE position is open at a time.
- Entry: buy the underpriced outcome whenever expected round-trip profit is strictly positive after both the taker entry fee at the observed ask and the estimated taker exit fee at fair value:
  `fair - ask - entry_fee_per_share - exit_fee_per_share(fair) > 0`.
- No arbitrary 3c hurdle is imposed on the 1h convergence book; transaction costs themselves are the hurdle.
- Entry must satisfy same-second immediate execution evidence for the full current ticket.
- Exit: once a same-second executable bid has returned to the current causal fair-value band and realized proceeds after exit fee exceed entry cost, sell immediately.
- After exit, capital is released immediately and a new CORE cycle may be opened later in the same hourly market.
- If no profitable convergence exit occurs before expiry, the remaining CORE position settles.

## 1h TAIL — dominant lottery-premium book
- TAIL is settlement-only and at most once per hourly market.
- “BS longshot has gone to zero” is frozen operationally as BOTH causal external anchors valuing the longshot below one standard 1c probability tick; equivalently the selected conservative favorite fair is >=99%.
- Entry requires positive post-entry-fee settlement edge at the observed favorite ask.
- This is the primary threshold. Diagnostic sensitivity at 99.5% and 98% may be reported but cannot replace the frozen 99% result post hoc.
- If TAIL and CORE qualify at the same timestamp, TAIL has priority and CORE is not opened.
- Once a TAIL position is opened in an hourly market, no NEW CORE entry may be opened for the rest of that market.
- A CORE position opened before TAIL may continue only to its normal convergence/settlement exit; TAIL does not use future information to cancel an already-open CORE position.

## Statistical/output requirements
Report 15m and 1h separately and combined only after separate results are visible. For each available 3m/6m/12m window report: final bankroll, total return, calendar CAGR (descriptive), MaxDD, turnover, trade/round-trip count, CORE and TAIL PnL, per-market cap hits, capacity rejects, same-market re-entry count, and witness/execution-proxy diagnostics. Do not annualize incomplete shards. Do not tune rules after seeing a window result.