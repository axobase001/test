from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pm_structural.recalc import BinanceAnchor, download_binance_1m
from pm_structural.time_units import canonical_btc15m_clock


def binance_close_before(anchor: BinanceAnchor, close_ts_ms: int) -> float:
    i = int(np.searchsorted(anchor.close_times, int(close_ts_ms) - 1, side="right") - 1)
    if i < 0:
        return math.nan
    return float(anchor.closes[i])


def fetch_price_to_beat(slug: str) -> dict:
    url = f"https://polymarket.com/api/equity/price-to-beat/{slug}"
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(1 + attempt); continue
            last = {"status_code": int(r.status_code), "text": r.text[:1000]}
            if r.ok:
                obj = r.json()
                return {"ok": True, "status_code": int(r.status_code), "json": obj}
            return {"ok": False, **last}
        except Exception as exc:
            last = {"error": repr(exc)}
            time.sleep(0.5 + attempt)
    return {"ok": False, **(last or {})}


def extract_numeric(obj) -> float | None:
    if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        x = float(obj)
        return x if x > 1000 else None
    if isinstance(obj, str):
        try:
            x = float(obj.replace(",", "").replace("$", ""))
            return x if x > 1000 else None
        except Exception:
            return None
    if isinstance(obj, dict):
        preferred = ["price", "priceToBeat", "price_to_beat", "value", "open", "openPrice", "open_price"]
        for k in preferred:
            if k in obj:
                x = extract_numeric(obj[k])
                if x is not None: return x
        for v in obj.values():
            x = extract_numeric(v)
            if x is not None: return x
    if isinstance(obj, list):
        for v in obj:
            x = extract_numeric(v)
            if x is not None: return x
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-03-02")
    ap.add_argument("--end", default="2026-03-18")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    fills = pd.read_csv(args.fills).copy()
    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)

    ptb_cache: dict[str, dict] = {}
    rows = []
    for r in fills.itertuples(index=False):
        slug = str(r.cid)
        open_s, close_ms = canonical_btc15m_clock(slug)
        open_bn = bn.open_price(open_s)
        close_bn = binance_close_before(bn, close_ms)
        ret = close_bn / open_bn - 1.0 if open_bn > 0 and close_bn > 0 else math.nan
        bn_out = "Up" if math.isfinite(ret) and ret >= 0 else "Down"
        resolved = str(r.resolved_outcome)
        decision_side = str(r.outcome_token)
        if slug not in ptb_cache:
            ptb_cache[slug] = fetch_price_to_beat(slug)
            time.sleep(0.03)
        resp = ptb_cache[slug]
        ptb = extract_numeric(resp.get("json")) if resp.get("ok") else None
        basis_bps = (ptb / open_bn - 1.0) * 1e4 if ptb is not None and open_bn > 0 else None
        rows.append({
            "cid": slug, "ts_ms": int(r.ts_ms), "s2c": int(r.s2c),
            "decision_side": decision_side, "target_px": float(r.target_px),
            "fair_conservative": float(r.fair_conservative), "base_pnl": float(r.base_pnl),
            "resolved_chainlink_outcome": resolved,
            "binance_open": open_bn, "binance_close": close_bn, "binance_return_bps": ret * 1e4,
            "binance_outcome": bn_out, "chainlink_vs_binance_direction_mismatch": bool(bn_out != resolved),
            "decision_would_win_on_binance": bool(decision_side == bn_out),
            "decision_won_on_chainlink": bool(decision_side == resolved),
            "price_to_beat": ptb, "ptb_minus_binance_open_bps": basis_bps,
            "ptb_http_ok": bool(resp.get("ok")), "ptb_status_code": resp.get("status_code"),
        })
    x = pd.DataFrame(rows)
    x.to_csv(args.out / "oracle_direction_rows.csv", index=False)
    losses = x[x["base_pnl"] < 0].copy()
    mism = x[x["chainlink_vs_binance_direction_mismatch"]].copy()
    basis = pd.to_numeric(x["ptb_minus_binance_open_bps"], errors="coerce").dropna().to_numpy(float)
    def q(a):
        if not len(a): return None
        return {"n": int(len(a)), "min": float(np.min(a)), "p10": float(np.quantile(a,.1)),
                "p50": float(np.quantile(a,.5)), "p90": float(np.quantile(a,.9)), "max": float(np.max(a))}
    summary = {
        "fills": int(len(x)), "losses": int(len(losses)),
        "binance_chainlink_direction_mismatches": int(len(mism)),
        "direction_mismatch_rate": float(len(mism)/len(x)) if len(x) else None,
        "losses_explained_by_oracle_direction_mismatch": int(losses["chainlink_vs_binance_direction_mismatch"].sum()) if len(losses) else 0,
        "losses_where_binance_also_went_against_decision": int((~losses["decision_would_win_on_binance"]).sum()) if len(losses) else 0,
        "losses_where_binance_would_have_validated_decision_but_chainlink_flipped": int((losses["decision_would_win_on_binance"] & ~losses["decision_won_on_chainlink"]).sum()) if len(losses) else 0,
        "price_to_beat_endpoint_successes": int(x["ptb_http_ok"].sum()),
        "price_to_beat_numeric_extracted": int(x["price_to_beat"].notna().sum()),
        "ptb_minus_binance_open_bps": q(basis),
        "interpretation": {
            "if_most_losses_mismatch": "settlement-oracle mismatch is a major cause; payoff state must be Chainlink-native",
            "if_most_losses_same_direction": "oracle source is architecturally wrong but not the main March loss cause; short-horizon probability/volatility calibration is the priority",
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "price_to_beat_raw.json").write_text(json.dumps(ptb_cache, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(losses):
        print("LOSS ROWS", flush=True)
        print(losses.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
