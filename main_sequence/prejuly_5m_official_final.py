from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import prejuly_5m_official as core

# ---------------------------------------------------------------------------
# PRE-JULY DATA-PREPROCESSING CORRECTIONS ONLY
# ---------------------------------------------------------------------------
# 1) core.normalize_trade_rows used `x.size`, which is pandas DataFrame.size
#    (row_count * column_count), not the Data API's "size" column.  That made
#    log_vol60 and pressure weights depend on the shape of the API query.
# 2) Data API timestamps are second-resolution.  pandas' default quicksort is
#    unstable for equal timestamps.  The API raw row order is empirically
#    stable across 48-market, 4-market, single-market, and repeated identical
#    June queries, so use stable mergesort to preserve that observed order.
# These corrections are frozen before any July test data is requested.

def normalize_trade_rows_fixed(td: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if td.empty:
        return {}
    need = {"conditionId", "timestamp", "price", "size", "side", "outcome"}
    if not need.issubset(td.columns):
        raise RuntimeError(f"trade schema missing {need-set(td.columns)}")
    x = td.copy()
    x["timestamp"] = pd.to_numeric(x["timestamp"], errors="coerce")
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"] = pd.to_numeric(x["size"], errors="coerce")
    x = x.dropna(subset=["timestamp", "price", "size"])
    x = x[(x["price"] > 0) & (x["price"] < 1) & (x["size"] > 0)]
    x["timestamp"] = x["timestamp"].astype(np.int64)
    x["outcome_l"] = x["outcome"].astype(str).str.lower().str.strip()
    x["side_u"] = x["side"].astype(str).str.upper().str.strip()
    x = x[x["outcome_l"].isin(["up", "down"])]
    x["p_up"] = np.where(x["outcome_l"].eq("up"), x["price"], 1.0 - x["price"])
    x["pressure"] = np.where(
        ((x["outcome_l"].eq("up")) & (x["side_u"].eq("BUY")))
        | ((x["outcome_l"].eq("down")) & (x["side_u"].eq("SELL"))),
        1.0,
        -1.0,
    )
    return {
        str(cid): g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        for cid, g in x.groupby("conditionId", sort=False)
    }


core.normalize_trade_rows = normalize_trade_rows_fixed

# Mechanical field-name typo in the original full-fetch baseline: Market defines
# label_up while build_examples reads m.label.  Expose the intended read-only alias.
if not hasattr(core.Market, "label"):
    core.Market.label = property(lambda self: self.label_up)

# Import after patching core so every downstream transport/statistical function sees
# the corrected preprocessing contract.
import prejuly_5m_official_sealed as sealed
import prejuly_5m_official_stream as stream


@dataclass
class FinalMicro:
    market: core.Market
    pm_last: float
    micro14: np.ndarray
    fill_up: tuple[tuple[int, float], ...]
    fill_down: tuple[tuple[int, float], ...]


def aggregate_market_fixed(m: core.Market, trade_map: dict[str, pd.DataFrame]) -> FinalMicro | None:
    g = trade_map.get(m.condition_id)
    if g is None or g.empty:
        return None
    pre = g[(g.timestamp >= m.start) & (g.timestamp < m.decision)].copy()
    if len(pre) < 4:
        return None
    lag = m.decision - int(pre.timestamp.iloc[-1])
    if lag < 0 or lag > 45:
        return None
    pm = float(pre.p_up.iloc[-1])
    if not (0.01 < pm < 0.99):
        return None

    def w(sec: int):
        return pre[pre.timestamp >= m.decision - sec]

    def mom(sec: int) -> float:
        q = w(sec)
        return float(pm - q.p_up.iloc[0]) if len(q) >= 2 else 0.0

    q30, q60, q120 = w(30), w(60), w(120)
    size60 = float(q60["size"].sum()) if len(q60) else 0.0
    press60 = float((q60.pressure * q60["size"]).sum() / (size60 + 1e-9)) if len(q60) else 0.0
    vals = q60.p_up.to_numpy(float) if len(q60) else np.asarray([pm])
    micro14 = np.asarray([
        pm,
        float(lag),
        math.log1p(len(q30)),
        math.log1p(len(q60)),
        math.log1p(len(q120)),
        math.log1p(size60),
        press60,
        mom(15),
        mom(30),
        mom(60),
        mom(120),
        float(np.std(vals)) if len(vals) > 1 else 0.0,
        float(np.max(vals) - np.min(vals)),
        float(np.mean(vals) - pm),
    ], dtype=np.float32)

    post = g[(g.timestamp >= m.decision) & (g.timestamp <= m.decision + core.TAPE_SECONDS) & (g.side_u == "BUY")]
    up = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "up"].itertuples(index=False))
    down = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "down"].itertuples(index=False))
    return FinalMicro(m, pm, micro14, up, down)


def build_stream_examples_fixed(micros, bs: dict[str, core.BinanceSeries]):
    out = []
    for z in micros:
        m = z.market
        bf = bs[m.asset].features(m.start, m.decision)
        if bf is None:
            continue
        feat = np.asarray([
            *z.micro14.tolist(),
            bf["finance_p"],
            bf["finance_p"] - z.pm_last,
            bf["log_rel_spot"],
            bf["ret1m"],
            bf["ret3m"],
            bf["rv60"],
            float(m.asset == "BTC"),
            float(m.asset == "ETH"),
            float(m.asset == "SOL"),
            float(m.asset == "XRP"),
        ], dtype=np.float32)
        if np.isfinite(feat).all():
            out.append(stream.StreamExample(
                m.slug, m.condition_id, m.asset, m.start, m.decision, m.label_up,
                z.pm_last, feat, z.fill_up, z.fill_down,
            ))
    out.sort(key=lambda e: (e.start, e.asset))
    return out


stream._aggregate_market = aggregate_market_fixed
stream.build_stream_examples = build_stream_examples_fixed

if __name__ == "__main__":
    stream.main()
