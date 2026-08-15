from __future__ import annotations

import math
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from main_sequence import final_recent_replay as base
from main_sequence.original_93d_anchor_freeze import fetch_deribit_trades_parallel
from pm_structural import recalc as original

# ---------------------------------------------------------------------------
# FINAL v3 CAUSAL PATCH
#
# v2 selected Deribit instruments using a full-day Binance median spot. That is
# an implicit universe-selection lookahead even though the final 30-minute IV
# window itself was backward-looking. v3 removes spot from universe selection
# completely. We fetch every BTC option instrument that could possibly be in
# the 7-65 DTE window during the shard, then apply the original per-trade
# contemporaneous index-price moneyness filter in DeribitAnchor.from_trades.
# ---------------------------------------------------------------------------
DAY_MS = 86_400_000

base.PROTOCOL["name"] = "Main Sequence FINAL v3 no-lookahead dual-exit replay / 2026-08-15 freeze"
base.PROTOCOL["deribit_universe"] = (
    "No spot-selected instrument universe. For each shard, include every BTC option instrument whose "
    "creation/expiration metadata makes it capable of entering the 7-65 DTE window. Fetch its actual "
    "historical trades, then filter each trade by contemporaneous Deribit index_price moneyness <=15%, "
    "TTE 7-65d and IV 1-300%. The fair-value IV is the backward 30-minute median only."
)
base.PROTOCOL["anti_lookahead"].append(
    "Deribit option candidates are selected only by creation/expiration timestamps, never by Binance or later spot; moneyness is evaluated from each historical trade's own contemporaneous index_price"
)
base.PROTOCOL["transport"] = (
    "Polymarket Data API requests are globally paced per runner; any unrecovered hour fails the shard. "
    "No partial-hour or silent truncation is admitted."
)


def _causal_instrument_universe(inst: pd.DataFrame, start: str, end: str) -> list[str]:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    # We need one pre-start day only so the first decisions have a full 30-minute
    # IV lookback. These metadata inequalities are purely temporal and do not
    # depend on any market price, future return, outcome, or volatility.
    anchor_start_ms = start_ms - DAY_MS
    x = inst.copy()
    creation = pd.to_numeric(x.get("creation_timestamp"), errors="coerce")
    expiry = pd.to_numeric(x.get("expiration_timestamp"), errors="coerce")
    keep = (
        creation.notna()
        & expiry.notna()
        & (creation < end_ms)
        & (expiry >= anchor_start_ms + 7 * DAY_MS)
        & (expiry <= end_ms + 65 * DAY_MS)
    )
    return sorted(x.loc[keep, "instrument_name"].astype(str).unique().tolist())


def build_anchors_causal(start: str, end: str, out: Path):
    cache = out / "anchor_cache_v3"; cache.mkdir(parents=True, exist_ok=True)
    start_d = date.fromisoformat(start); end_d = date.fromisoformat(end) - timedelta(days=1)
    anchor_start = start_d - timedelta(days=1)

    print("V3_ANCHOR_BINANCE_1M", start, end, flush=True)
    bn_df = original.download_binance_1m(anchor_start, end_d, cache / "binance_1m")
    bn = original.BinanceAnchor.from_df(bn_df)

    print("V3_ANCHOR_DERIBIT_INSTRUMENTS", flush=True)
    inst = original.deribit_instruments()
    selected = _causal_instrument_universe(inst, start, end)
    if not selected:
        raise RuntimeError(f"v3 causal Deribit universe empty for {start} {end}")
    print("V3_ANCHOR_DERIBIT_TEMPORAL_UNIVERSE", len(selected), flush=True)

    trades = fetch_deribit_trades_parallel(
        selected,
        anchor_start,
        end_d,
        cache / "deribit_all_temporal_universe_trades.parquet",
    )
    # This original routine does NOT use Binance or future spot. It joins strike
    # and expiry metadata and filters each actual historical trade using that
    # trade's own timestamp + index_price, then sorts by timestamp.
    der = original.DeribitAnchor.from_trades(trades, inst)
    if len(der.ts) == 0:
        raise RuntimeError(f"v3 causal Deribit anchor empty after contemporaneous filtering for {start} {end}")
    meta = {
        "method": "temporal_universe_plus_contemporaneous_trade_filter_v3",
        "temporal_universe_instruments": int(len(selected)),
        "raw_historical_option_trades": int(len(trades)),
        "usable_deribit_anchor_trades": int(len(der.ts)),
        "spot_used_for_universe_selection": False,
    }
    return bn, der, meta


# Pace Data-API requests below the public burst limit. The old transport already
# retries 429/5xx; this limiter prevents normal concurrency from manufacturing
# avoidable 429 gaps in busy months.
_ORIG_GET_JSON = base.get_json
_PM_LOCK = threading.Lock()
_PM_LAST = 0.0
_PM_MIN_INTERVAL = 0.12  # <= 8.34 requests/sec per runner


def paced_get_json(sess, url: str, *, params=None, timeout=60, tries=7):
    global _PM_LAST
    if "data-api.polymarket.com" in str(url):
        with _PM_LOCK:
            now = time.monotonic()
            delay = _PM_MIN_INTERVAL - (now - _PM_LAST)
            if delay > 0:
                time.sleep(delay)
            _PM_LAST = time.monotonic()
    return _ORIG_GET_JSON(sess, url, params=params, timeout=timeout, tries=tries)


base.build_anchors = build_anchors_causal
base.get_json = paced_get_json


if __name__ == "__main__":
    base.main()
