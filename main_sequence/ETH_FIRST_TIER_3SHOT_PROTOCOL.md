# ETH First-Tier Three-Shot Qualification — Frozen 2026-08-18

This document freezes the final protocol before any valid execution result is observed. GitHub hosted jobs were still queued when this freeze was written.

## Shared statistical gate
- Fixed ticket qualification only; no $100 cap in this experiment.
- UTC-day aggregation; 4,000-replicate day bootstrap with fixed seed 20260818.
- GREEN: total PnL > 0, >=20 observed trading days, >=100 trades, and bootstrap CI95 lower bound for mean daily PnL > 0.
- YELLOW: total PnL > 0 but sample or CI gate is not strong enough.
- RED: total PnL <= 0.

## ETH 15m CORE
- Period: 2026-05-15..2026-08-15 end-exclusive.
- Fixed $5 ticket.
- Conservative raw gap >=10c.
- Net fair-value edge >=5c/share.
- Ask >=20c.
- Entry and convergence-exit execution evidence: same-second public taker-tape exact-price volume; the chosen level alone must contain >=2x required qty.
- This evidence is **not** represented as resting L1 order-book depth.
- Fair: conservative boundary from Binance ETHUSDT 60m realized volatility and backward 30m median Deribit ETH option-trade IV.
- Deribit candidate universe is selected only by creation/expiration timestamps; contemporaneous trade `index_price` supplies moneyness filtering. No spot-selected option universe.
- Large dislocation exits through causal convergence to the 1c fair band; otherwise settlement fallback.

## ETH 5m TAIL
- Period: 2026-06-04..2026-07-15 end-exclusive, matching the frozen capture envelope used for this qualification.
- Settlement only.
- Fixed $5 ticket including fee.
- Fair floors reported independently: 95%, 97%, 98%, 99%.
- Barrier-distance floors reported independently: 0, 2, 5, 10, 20 bps.
- All 20 cells are reported; no post-hoc replacement of the surface by the best-looking cell.
- Reference/threshold: Chainlink ETH; threshold is the Chainlink window-start price.
- Fair: digital N(d2), using Chainlink spot/threshold and causal 60m realized volatility.
- Chainlink causality uses raw `cap_prices.ts_ms` directly. No 5-second bucket start is allowed to stand in for a later tick; every fair-value spot must have `spot_source_ts <= chosen_book_ts`.
- Realized-volatility minute points retain the actual last-tick timestamp, and a partial current minute can only contribute via the latest raw tick already observed by the chosen book timestamp.
- Execution: actual captured `cap_book.best_ask` and best-level `ask_sz`; full fixed-$5 quantity must fit at that ask.
- The chosen outcome is valued using its **own captured book timestamp**. No other outcome's later timestamp may advance fair value.
- Target decision point: T-90s; chosen book snapshot must be no more than 4,000 ms stale versus target.
- `cap_book.ts_ms` is collector capture time (~2s cadence/token), not exchange event time; this limitation remains explicit.

## ETH 1h CORE
- Period: 2026-05-15..2026-08-15 end-exclusive.
- Fixed $5 ticket.
- Reference: Binance ETHUSDT 1H open/close.
- Fair: conservative boundary from Binance ETHUSDT 60m realized volatility and backward 30m median Deribit ETH option-trade IV.
- Deribit candidate universe uses the same no-lookahead temporal-universe rule as ETH15m.
- Net convergence edge at entry >=5c/share.
- Ask >=20c.
- 0.5G stop: freeze entry expected convergence profit G; stop when executable net liquidation loss >=0.5G.
- strict3 execution evidence: same-second public taker-tape exact-price volume; entry, convergence exit, and 0.5G stop each require >=3x required qty at the exact level.
- This evidence is **not** represented as resting L1 order-book depth.
- No same-second re-entry after an exit.
- Settlement fallback only if neither convergence nor executable 0.5G stop occurs before close.

## Interpretation boundary
A positive tape-witness lane is an execution-proxy qualification, not proof of resting historical L1 queue fill. The 5m lane is stronger on book evidence because it uses captured best ask and ask size directly. Any later $100-cap test is a separate capacity experiment and cannot retroactively change this fixed-$5 qualification.
