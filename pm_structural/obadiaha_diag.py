from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pm_structural.obadiaha_strict import hf_file, spot_at, to_ms
from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    digital_prob_up,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)


def q(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "min": float(np.min(a)),
        "p01": float(np.quantile(a, 0.01)),
        "p10": float(np.quantile(a, 0.10)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--day", default="2026-03-14")
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(args.day); hf_cache = args.cache / "hf"

    markets = pd.read_parquet(hf_file("markets/all.parquet", hf_cache))
    markets["market_id"] = markets["market_id"].astype(str)
    markets["start_dt"] = pd.to_datetime(markets["start_time"], utc=True, errors="coerce")
    markets["end_dt"] = pd.to_datetime(markets["end_time"], utc=True, errors="coerce")
    m = markets[(markets["asset"] == "BTC") & (markets["market_type"] == "crypto_15m")].copy()
    market_meta = {
        str(r.market_id): {
            "open_ts_s": int(r.start_dt.timestamp()),
            "close_ts_ms": int(r.end_dt.timestamp() * 1000),
        }
        for r in m.itertuples() if pd.notna(r.start_dt) and pd.notna(r.end_dt)
    }

    bn_df = download_binance_1m(d, d, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)
    inst = deribit_instruments(); selected = select_deribit_instruments(inst, bn_df, d, d)
    der_trades = fetch_deribit_trades(selected, d, d, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)

    bcols = ["timestamp", "asset", "market_id", "condition_id", "token_id", "best_bid", "best_ask", "mid_price"]
    book = pd.read_parquet(hf_file(f"orderbooks/{d.isoformat()}.parquet", hf_cache), columns=bcols)
    book = book[(book["asset"] == "BTC") & book["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    book["market_id"] = book["market_id"].astype(str); book["token_id"] = book["token_id"].astype(str)
    book["ts_ms"] = to_ms(book["timestamp"])
    for c in ["best_bid", "best_ask", "mid_price"]:
        book[c] = pd.to_numeric(book[c], errors="coerce")
    book = book.dropna(subset=["ts_ms", "best_ask"]).copy(); book["ts_ms"] = book["ts_ms"].astype(np.int64)
    book = book.sort_values(["market_id", "ts_ms", "token_id"]).drop_duplicates(["market_id", "ts_ms", "token_id"], keep="last")

    counts = {
        "book_rows": int(len(book)), "book_markets": int(book["market_id"].nunique()),
        "paired_snapshots": 0, "s2c_60_600": 0, "valid_asks": 0,
        "valid_rv": 0, "valid_deribit_iv": 0, "valid_both_anchors": 0,
        "valid_spot": 0, "valid_probs": 0, "cross_gross_ge_3c": 0,
    }
    ask_vals: list[float] = []; bid_vals: list[float] = []; ask_sum: list[float] = []; bid_sum: list[float] = []
    rv_vals: list[float] = []; iv_vals: list[float] = []; p_rv_vals: list[float] = []; p_iv_vals: list[float] = []
    cross_gross: list[float] = []; side_blind_pair_gross: list[float] = []
    tops: list[dict] = []

    for market_id, g in book.groupby("market_id", sort=True):
        meta = market_meta.get(market_id)
        if not meta:
            continue
        open_bn = bn.open_price(int(meta["open_ts_s"]))
        if not (open_bn > 0 and math.isfinite(open_bn)):
            continue
        close_ts_ms = int(meta["close_ts_ms"])
        for ts_ms, snap in g.groupby("ts_ms", sort=True):
            if len(snap) != 2:
                continue
            counts["paired_snapshots"] += 1
            s2c = int(round((close_ts_ms - int(ts_ms)) / 1000.0))
            if not 60 <= s2c <= 600:
                continue
            counts["s2c_60_600"] += 1
            asks = pd.to_numeric(snap["best_ask"], errors="coerce").to_numpy(float)
            bids = pd.to_numeric(snap["best_bid"], errors="coerce").to_numpy(float)
            if len(asks) != 2 or not np.all(np.isfinite(asks)) or np.any(asks <= 0) or np.any(asks >= 1):
                continue
            counts["valid_asks"] += 1
            ask_vals.extend(asks.tolist()); ask_sum.append(float(np.sum(asks)))
            if len(bids) == 2 and np.all(np.isfinite(bids)):
                bid_vals.extend(bids.tolist()); bid_sum.append(float(np.sum(bids)))
            rv = bn.rv_annualized(int(ts_ms), 60)
            div = der.median_iv(int(ts_ms), 30)
            if math.isfinite(rv) and 0.05 <= rv <= 3.0:
                counts["valid_rv"] += 1; rv_vals.append(float(rv))
            if math.isfinite(div) and 0.05 <= div <= 3.0:
                counts["valid_deribit_iv"] += 1; iv_vals.append(float(div))
            if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
                continue
            counts["valid_both_anchors"] += 1
            spot = spot_at(bn, int(ts_ms))
            if not (spot > 0 and math.isfinite(spot)):
                continue
            counts["valid_spot"] += 1
            rel = spot / open_bn
            p_rv = digital_prob_up(rel, s2c, rv); p_iv = digital_prob_up(rel, s2c, div)
            if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
                continue
            counts["valid_probs"] += 1; p_rv_vals.append(float(p_rv)); p_iv_vals.append(float(p_iv))
            fair_up = min(p_rv, p_iv); fair_down = 1.0 - max(p_rv, p_iv)
            # Deliberately mapping-agnostic and maximally permissive: any fair paired with any ask.
            gross = max(fair_up - asks[0], fair_up - asks[1], fair_down - asks[0], fair_down - asks[1])
            cross_gross.append(float(gross))
            # If collector rows happened to be ordered consistently, keep both possible mappings as diagnostics.
            map_a = max(fair_up - asks[0], fair_down - asks[1])
            map_b = max(fair_up - asks[1], fair_down - asks[0])
            side_blind_pair_gross.append(float(max(map_a, map_b)))
            if gross >= args.threshold:
                counts["cross_gross_ge_3c"] += 1
            tops.append({
                "market_id": market_id, "ts_ms": int(ts_ms), "s2c": s2c,
                "ask0": float(asks[0]), "ask1": float(asks[1]),
                "bid0": float(bids[0]) if len(bids) else None, "bid1": float(bids[1]) if len(bids) > 1 else None,
                "p_rv": float(p_rv), "p_iv": float(p_iv), "fair_up": float(fair_up), "fair_down": float(fair_down),
                "rv": float(rv), "deribit_iv": float(div), "rel_spot": float(rel), "cross_gross_edge": float(gross),
            })

    tops = sorted(tops, key=lambda x: x["cross_gross_edge"], reverse=True)[:100]
    report = {
        "day": d.isoformat(), "threshold": args.threshold, "counts": counts,
        "selected_deribit_instruments": len(selected), "usable_deribit_trades": len(der.ts),
        "quantiles": {
            "ask": q(ask_vals), "bid": q(bid_vals), "ask_sum_two_tokens": q(ask_sum), "bid_sum_two_tokens": q(bid_sum),
            "rv_annualized": q(rv_vals), "deribit_iv": q(iv_vals), "p_rv": q(p_rv_vals), "p_iv": q(p_iv_vals),
            "max_mapping_agnostic_gross_edge": q(cross_gross),
            "max_of_two_possible_token_mappings_gross_edge": q(side_blind_pair_gross),
        },
        "top_gross_opportunities": tops,
        "interpretation_gate": "No CLOB mapping, fee, resolution, tape, or outcome data is used. This is only a causal pre-gate diagnostic.",
    }
    (args.out / "diag.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(tops).to_csv(args.out / "top_gross_opportunities.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
