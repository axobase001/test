from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pm_structural import recalc

PTB_API = "https://polymarket.com/api/crypto/crypto-price"


def iso_s(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_one(start: int, tries: int = 8) -> dict:
    end = int(start) + 900
    params = {
        "symbol": "BTC",
        "eventStartTime": iso_s(start),
        "variant": "fifteen",
        "endDate": iso_s(end),
    }
    last = None
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 main-sequence-chainlink-ptb-audit/1.0", "Accept": "application/json", "Referer": "https://polymarket.com/"})
    for i in range(tries):
        try:
            r = sess.get(PTB_API, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:160]}")
            r.raise_for_status()
            obj = r.json()
            op = obj.get("openPrice")
            cp = obj.get("closePrice")
            return {
                "start": int(start), "close": end,
                "slug": f"btc-updown-15m-{int(start)}",
                "ptb_open": float(op) if op is not None else math.nan,
                "ptb_close": float(cp) if cp is not None else math.nan,
                "completed": obj.get("completed"), "status": 200, "error": None,
                "raw": json.dumps(obj, separators=(",", ":"), sort_keys=True),
            }
        except Exception as exc:
            last = repr(exc)
            time.sleep(min(0.35 * (2 ** i), 8.0))
    return {"start": int(start), "close": end, "slug": f"btc-updown-15m-{int(start)}",
            "ptb_open": math.nan, "ptb_close": math.nan, "completed": None, "status": None,
            "error": last, "raw": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-16")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end) - timedelta(days=1)
    bn_df = recalc.download_binance_1m(d0, d1, args.out / "binance_1m")
    bn = recalc.BinanceAnchor.from_df(bn_df)

    start_ts = int(pd.Timestamp(args.start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(args.end, tz="UTC").timestamp())
    starts = list(range(start_ts, end_ts, 900))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(fetch_one, s): s for s in starts}
        for i, f in enumerate(as_completed(fut), 1):
            row = f.result()
            bop = bn.open_price(int(row["start"]))
            row["binance_open"] = float(bop) if math.isfinite(bop) else math.nan
            if math.isfinite(row["ptb_open"]) and math.isfinite(row["binance_open"]) and row["binance_open"] > 0:
                row["ptb_minus_binance_bp"] = 10000.0 * math.log(row["ptb_open"] / row["binance_open"])
                row["abs_basis_bp"] = abs(row["ptb_minus_binance_bp"])
            else:
                row["ptb_minus_binance_bp"] = math.nan
                row["abs_basis_bp"] = math.nan
            rows.append(row)
            if i % 96 == 0:
                ok = sum(math.isfinite(float(x.get("ptb_open", math.nan))) for x in rows)
                print("PTB_AUDIT", i, "/", len(starts), "ok", ok, flush=True)

    df = pd.DataFrame(rows).sort_values("start", kind="mergesort")
    df.to_csv(args.out / "ptb_vs_binance.csv", index=False)
    ok = df[np.isfinite(pd.to_numeric(df["ptb_minus_binance_bp"], errors="coerce"))].copy()
    b = ok["ptb_minus_binance_bp"].to_numpy(float)
    summary = {
        "period": [args.start, args.end],
        "markets_expected": len(starts), "ptb_ok": int(len(ok)), "ptb_failed": int(len(df) - len(ok)),
        "basis_bp": {
            "mean": float(np.mean(b)) if len(b) else None,
            "median": float(np.median(b)) if len(b) else None,
            "std": float(np.std(b, ddof=1)) if len(b) > 1 else None,
            "p01": float(np.quantile(b, .01)) if len(b) else None,
            "p05": float(np.quantile(b, .05)) if len(b) else None,
            "p25": float(np.quantile(b, .25)) if len(b) else None,
            "p75": float(np.quantile(b, .75)) if len(b) else None,
            "p95": float(np.quantile(b, .95)) if len(b) else None,
            "p99": float(np.quantile(b, .99)) if len(b) else None,
            "max_abs": float(np.max(np.abs(b))) if len(b) else None,
            "share_abs_ge_1bp": float(np.mean(np.abs(b) >= 1.0)) if len(b) else None,
            "share_abs_ge_2bp": float(np.mean(np.abs(b) >= 2.0)) if len(b) else None,
            "share_abs_ge_3bp": float(np.mean(np.abs(b) >= 3.0)) if len(b) else None,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
