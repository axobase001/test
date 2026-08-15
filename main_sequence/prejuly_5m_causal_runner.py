"""Causality-hardening runner for the sealed pre-July / July validation.

The core loader downsamples raw price ticks into 5-second buckets for tractability.
A bucket label is not an observation timestamp: using it for as-of lookup can expose
a later tick from the same bucket.  This runner replaces only that conversion so all
as-of lookups use the actual source timestamp of the retained tick.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import prejuly_5m_multiasset as core


def build_price_series_causal(df: pd.DataFrame) -> dict[str, core.PriceSeries]:
    out: dict[str, core.PriceSeries] = {}
    for asset in core.ASSETS:
        ad = df[df.asset == asset]
        if ad.empty:
            continue
        counts = ad.groupby("src").size().to_dict()
        src = "chainlink" if counts.get("chainlink", 0) >= 100 else "binance"
        x = ad[ad.src == src].sort_values("source_ts").drop_duplicates("source_ts", keep="last")
        ts = x.source_ts.to_numpy(np.int64)
        px = x.value.to_numpy(float)
        order = np.argsort(ts)
        ts, px = ts[order], px[order]
        mins = (ts // 60_000) * 60_000
        tmp = pd.DataFrame({"m": mins, "ts": ts, "px": px}).groupby("m", sort=True).tail(1)
        out[asset] = core.PriceSeries(
            ts=ts,
            px=px,
            minute_ts=tmp.ts.to_numpy(np.int64),
            minute_px=tmp.px.to_numpy(float),
            source=src,
        )
    missing = [a for a in core.ASSETS if a not in out]
    if missing:
        raise RuntimeError(f"missing price series: {missing}")
    return out


core.build_price_series = build_price_series_causal

if __name__ == "__main__":
    core.main()
