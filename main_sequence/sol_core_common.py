from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from main_sequence import eth15m_conservative_replay as base15
from main_sequence.original_93d_anchor_freeze import fetch_deribit_trades_parallel
from pm_structural import recalc as original

DAY_MS = 86_400_000


def configure_sol_base() -> None:
    # Frozen ETH implementation, asset substitution only.
    base15.SYMBOL = "SOLUSDT"
    base15.ASSET = "SOL"


def sol_deribit_instruments() -> pd.DataFrame:
    rows: list[dict] = []
    # SOL linear options live inside Deribit's USDC currency namespace.
    # The history endpoint exposes the full expired universe; live is included
    # for completeness but the qualification window ends before today.
    for expired in ("true", "false"):
        obj = original.fetch_json(
            f"{original.DERIBIT_BASE}/get_instruments",
            {"currency": "USDC", "kind": "option", "expired": expired},
        )
        rows.extend(
            x for x in (obj.get("result", []) or [])
            if str(x.get("instrument_name") or "").startswith("SOL_USDC-")
        )
    if not rows:
        raise RuntimeError("Deribit returned no SOL_USDC option instruments")
    df = pd.DataFrame(rows).drop_duplicates("instrument_name")
    for c in ["expiration_timestamp", "creation_timestamp", "strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["instrument_name", "expiration_timestamp", "creation_timestamp", "strike"])


def causal_instrument_universe(inst: pd.DataFrame, start: str, end: str) -> list[str]:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    anchor_start_ms = start_ms - DAY_MS
    creation = pd.to_numeric(inst["creation_timestamp"], errors="coerce")
    expiry = pd.to_numeric(inst["expiration_timestamp"], errors="coerce")
    keep = (
        creation.notna()
        & expiry.notna()
        & (creation < end_ms)
        & (expiry >= anchor_start_ms + 7 * DAY_MS)
        & (expiry <= end_ms + 65 * DAY_MS)
    )
    return sorted(inst.loc[keep, "instrument_name"].astype(str).unique().tolist())


def build_sol_anchors_causal(start: str, end: str, out: Path):
    configure_sol_base()
    cache = out / "anchors_causal"
    cache.mkdir(parents=True, exist_ok=True)
    sd = date.fromisoformat(start)
    ed = date.fromisoformat(end) - timedelta(days=1)
    anchor_start = sd - timedelta(days=1)

    bn_df = base15.download_eth_1m(sd, ed, cache / "binance1m")
    bn = original.BinanceAnchor.from_df(bn_df)

    inst = sol_deribit_instruments()
    selected = causal_instrument_universe(inst, start, end)
    if not selected:
        raise RuntimeError(f"empty causal SOL_USDC Deribit universe for {start}..{end}")

    trades = fetch_deribit_trades_parallel(
        selected,
        anchor_start,
        ed,
        cache / "deribit_temporal_universe_trades.parquet",
    )
    der = original.DeribitAnchor.from_trades(trades, inst)
    if len(der.ts) == 0:
        raise RuntimeError(f"no usable SOL_USDC Deribit anchor trades for {start}..{end}")

    return bn, der, {
        "venue": "Deribit",
        "product": "SOL_USDC linear options",
        "method": "temporal_universe_plus_contemporaneous_trade_filter",
        "spot_used_for_universe_selection": False,
        "temporal_universe_instruments": int(len(selected)),
        "raw_historical_option_trades": int(len(trades)),
        "usable_deribit_anchor_trades": int(len(der.ts)),
    }
