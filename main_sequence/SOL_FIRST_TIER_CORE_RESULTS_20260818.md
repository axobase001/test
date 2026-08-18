# SOL First-Tier CORE Qualification — Frozen 2026-08-18

## Status

`SOL_15M_CORE = GREEN`

`SOL_1H_CORE = GREEN`

Primary full-run workflow: `32133488492` on head commit `7006c6d9d4168c81feb8337af32d7a1b03e4d2fd`.

The experiment is a direct asset substitution of the already-frozen ETH CORE protocols. No SOL-specific threshold search or post-result tuning was performed.

## Data / causal contract

- Polymarket: real closed SOL Up/Down markets and public same-second taker tape.
- Spot / realized-volatility anchor: Binance `SOLUSDT`.
- Option-IV anchor: actual historical Deribit `SOL_USDC-*` linear option trades from the Deribit history API.
- Deribit option universe is selected only from creation/expiration timestamps. Spot is not used to select instruments.
- IV uses only backward/contemporaneous actual option trades; no DVOL or other volatility proxy.
- The shared full-window anchor contained 3,993 temporal-universe SOL option instruments, 39,595 cleaned raw historical option trades, and 16,097 usable IV-anchor trades.

## SOL 15m CORE

Period: `2026-05-15` to `2026-08-15` end-exclusive.

Frozen rules:
- fixed ticket: $5
- raw fair/ask gap >= 10c
- net fair-value edge >= 5c/share
- ask >= 20c
- exact-price same-second public taker-tape volume >= 2x required quantity
- convergence band: 1c; otherwise settlement fallback

Results:
- expected markets: 8,832
- mapped markets: 8,830 (99.9774%)
- trades: **544**
- PnL: **+$373.02006973258**
- $50 fixed-ticket equity: **$423.02006973258**
- MDD: **-33.4634%**
- convergence exits: 113
- settlement fallbacks: 431
- trading days: 83
- mean daily PnL: +$4.4942
- positive-day rate: 66.27%
- day-bootstrap mean-PnL CI95: **[+$1.2026, +$7.7871]** (4,000 reps)
- overall qualification: **GREEN**
- recent `2026-07-15..2026-08-15`: 89 trades, **+$53.5314**
- opportunity density by frozen calendar blocks: 6.45/day -> 8.50/day -> 2.87/day

Primary artifact:
- name: `sol-first-tier-15m-core`
- artifact ID: `9326018428`
- SHA-256: `527a1f59217a35d9a62c774c0e9cda1fc6bf6c6aa32a1026da51bb91d36ed93e`

## SOL 1h CORE

Period: `2026-05-15` to `2026-08-15` end-exclusive.

Frozen rules:
- fixed ticket: $5
- net convergence edge >= 5c/share
- ask >= 20c
- frozen 0.5G stop
- strict3: exact-price same-second public taker-tape volume >= 3x required quantity for entry/exit/stop
- no same-second re-entry
- settlement fallback if neither convergence nor executable stop occurs

Results:
- expected markets: 2,208
- mapped markets: 2,202 (99.7283%)
- markets with tape: 2,202
- entries/exits: **154**
- PnL: **+$133.98173533622**
- $50 fixed-ticket equity: **$183.98173533622**
- MDD: **-54.9732%**
- convergence exits: 23
- 0.5G stops: 24
- settlements: 107
- trading days: 72
- mean daily PnL: +$1.8609
- positive-day rate: 55.56%
- day-bootstrap mean-PnL CI95: **[+$0.1282, +$3.6533]** (4,000 reps)
- overall qualification: **GREEN**
- recent `2026-07-15..2026-08-15`: 22 exits, **+$17.3326**
- mean entry edge: 9.13c/share
- median entry depth headroom: 4.95x

Primary artifact:
- name: `sol-first-tier-1h-core`
- artifact ID: `9324306190`
- SHA-256: `0abb60b5365c9163f61dce5f49ae7e8c13613532e82b788d96561e743bb91f83`

## Independent split reproduction

A second workflow reran the identical SOL 15m protocol in disjoint 7-day slices, each rebuilding its own prior-day Binance/Deribit anchor history. At freeze time, eight independently completed slices covered 343 of the 544 primary trades, including both winning and losing periods.

Completed exact-match slices:
- `2026-05-15..05-22`: 29 trades, +$24.1657
- `2026-05-22..05-29`: 36 trades, +$51.5860
- `2026-05-29..06-05`: 42 trades, +$43.3663
- `2026-06-05..06-12`: 72 trades, +$67.7036
- `2026-06-19..06-26`: 64 trades, +$54.2220
- `2026-07-10..07-17`: 53 trades, **-$22.9128**
- `2026-07-17..07-24`: 33 trades, +$40.8685
- `2026-07-31..08-07`: 14 trades, +$14.0353

For every completed slice, the slice runner and the corresponding rows of the full 93-day runner matched exactly in trade count and PnL, and matched column-by-column for `slug`, `condition_id`, `start`, `decision`, `exit`, `exit_kind`, `side`, `entry_px`, `qty`, `cost`, `pnl`, `raw_gap`, `edge_share`, and `entry_rv`.

## Combined interpretation

The arithmetic sum of the two independently qualified fixed-$5 lanes is **+$507.00180506880**. This is not a shared-cash portfolio result and must not be presented as one; each lane is a separate fixed-ticket qualification.

SOL is therefore a real positive-alpha extension of the frozen CORE architecture, but it is not stronger than ETH in this sample. Its opportunity density and total PnL are materially lower, and risk is worse, especially the 1h MDD. `GREEN` here means the frozen statistical qualification gate passed; it is not deployment approval.
