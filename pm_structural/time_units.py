from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _datetime_ints_to_ms(dt: pd.Series) -> pd.Series:
    """Convert a pandas datetime Series to epoch ms without assuming ns storage."""
    unit = getattr(dt.dtype, "unit", "ns")
    raw = dt.astype("int64")
    if unit == "ns":
        vals = raw // 1_000_000
    elif unit == "us":
        vals = raw // 1_000
    elif unit == "ms":
        vals = raw
    elif unit == "s":
        vals = raw * 1_000
    else:
        raise RuntimeError(f"unsupported pandas datetime unit: {unit!r}")
    return vals.astype(np.int64)


def epoch_series_to_ms(s: pd.Series) -> pd.Series:
    """Normalize numeric epoch s/ms/us/ns or datetime-like strings to nullable epoch ms.

    Numeric magnitudes are detected row-wise. Datetime-like strings are converted
    using the actual pandas datetime dtype unit, so pandas 2/3 ns/us resolution
    differences cannot silently shrink timestamps by 1000x.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)
    mask = numeric.notna()
    if mask.any():
        v = numeric[mask].astype(float)
        av = v.abs()
        vals = np.full(len(v), np.nan, dtype=float)
        a = av.to_numpy(float); x = v.to_numpy(float)
        sec = (a >= 1e8) & (a < 1e11)
        ms = (a >= 1e11) & (a < 1e14)
        us = (a >= 1e14) & (a < 1e17)
        ns = a >= 1e17
        vals[sec] = x[sec] * 1000.0
        vals[ms] = x[ms]
        vals[us] = x[us] / 1000.0
        vals[ns] = x[ns] / 1_000_000.0
        ok = np.isfinite(vals)
        idx = v.index.to_numpy()[ok]
        out.loc[idx] = np.rint(vals[ok]).astype(np.int64)

    text_mask = ~mask & s.notna()
    if text_mask.any():
        dt = pd.to_datetime(s[text_mask], utc=True, errors="coerce")
        good = dt.notna()
        if good.any():
            good_dt = dt[good]
            out.loc[good_dt.index] = _datetime_ints_to_ms(good_dt)
    return out.astype("Int64")


def market_slug_open_s(slug: str) -> int:
    m = re.search(r"-(\d{10})$", str(slug))
    if not m:
        raise ValueError(f"no 10-digit epoch suffix in market slug: {slug}")
    x = int(m.group(1))
    if not (1_500_000_000 <= x <= 2_000_000_000):
        raise ValueError(f"implausible slug epoch: {slug}")
    return x


def canonical_btc15m_clock(slug: str) -> tuple[int, int]:
    open_s = market_slug_open_s(slug)
    return open_s, (open_s + 15 * 60) * 1000


def audit_btc15m_metadata(slug: str, start_raw, end_raw, tolerance_ms: int = 2000) -> dict:
    ss = pd.Series([start_raw, end_raw])
    ms = epoch_series_to_ms(ss)
    if ms.isna().any():
        raise ValueError(f"could not normalize metadata clock for {slug}: start={start_raw} end={end_raw}")
    start_ms, end_ms = int(ms.iloc[0]), int(ms.iloc[1])
    open_s, close_ms = canonical_btc15m_clock(slug)
    open_ms = open_s * 1000
    start_delta = start_ms - open_ms
    end_delta = end_ms - close_ms
    duration_ms = end_ms - start_ms
    ok = abs(start_delta) <= tolerance_ms and abs(end_delta) <= tolerance_ms and abs(duration_ms - 900_000) <= 2 * tolerance_ms
    return {
        "slug": str(slug), "canonical_open_ms": open_ms, "canonical_close_ms": close_ms,
        "metadata_start_ms": start_ms, "metadata_end_ms": end_ms,
        "start_delta_ms": start_delta, "end_delta_ms": end_delta,
        "metadata_duration_ms": duration_ms, "ok": bool(ok),
    }
