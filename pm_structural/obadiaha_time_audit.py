from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

PM_REPO = "obadiaha/polymarket-crypto-5m-15m"
PM_REV = "11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"


def hf_file(rel: str, cache: Path) -> Path:
    return Path(hf_hub_download(repo_id=PM_REPO, repo_type="dataset", revision=PM_REV, filename=rel, cache_dir=cache))


def to_ms(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return (x.astype("int64") // 1_000_000).astype("Int64")


def q(vals) -> dict:
    a = np.asarray(list(vals), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)), "min": float(a.min()), "p01": float(np.quantile(a, .01)),
        "p10": float(np.quantile(a, .10)), "p50": float(np.quantile(a, .50)),
        "p90": float(np.quantile(a, .90)), "p99": float(np.quantile(a, .99)), "max": float(a.max()),
    }


def tail_epoch(slug: str) -> int | None:
    try:
        x = int(str(slug).rsplit("-", 1)[-1])
        if 1_500_000_000 <= x <= 2_000_000_000:
            return x
    except Exception:
        pass
    return None


def count_window(vals, lo=60, hi=600) -> int:
    a = np.asarray(list(vals), dtype=float)
    return int(np.sum((a >= lo) & (a <= hi)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--day", default="2026-03-14")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    day = args.day

    markets = pd.read_parquet(hf_file("markets/all.parquet", args.cache))
    markets["market_id"] = markets["market_id"].astype(str)
    markets = markets[(markets["asset"] == "BTC") & (markets["market_type"] == "crypto_15m")].copy()
    markets["start_ms"] = to_ms(markets["start_time"])
    markets["end_ms"] = to_ms(markets["end_time"])
    markets["slug_epoch_s"] = markets["market_id"].map(tail_epoch)
    markets["slug_ms"] = pd.to_numeric(markets["slug_epoch_s"], errors="coerce") * 1000

    bcols = ["timestamp", "asset", "market_id", "token_id", "best_bid", "best_ask"]
    book = pd.read_parquet(hf_file(f"orderbooks/{day}.parquet", args.cache), columns=bcols)
    book = book[(book["asset"] == "BTC") & book["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    book["market_id"] = book["market_id"].astype(str)
    book["ts_ms"] = to_ms(book["timestamp"])
    book = book.dropna(subset=["ts_ms"]).copy(); book["ts_ms"] = book["ts_ms"].astype(np.int64)

    tcols = ["timestamp", "asset", "market_id", "token_id", "side", "price"]
    trades = pd.read_parquet(hf_file(f"trades/{day}.parquet", args.cache), columns=tcols)
    trades = trades[(trades["asset"] == "BTC") & trades["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    trades["market_id"] = trades["market_id"].astype(str)
    trades["ts_ms"] = to_ms(trades["timestamp"])
    trades = trades.dropna(subset=["ts_ms"]).copy(); trades["ts_ms"] = trades["ts_ms"].astype(np.int64)

    bg = book.groupby("market_id")["ts_ms"].agg(["min", "max", "median", "count"]).reset_index()
    tg = trades.groupby("market_id")["ts_ms"].agg(trade_min="min", trade_max="max", trade_median="median", trade_count="count").reset_index()
    x = bg.merge(tg, on="market_id", how="left").merge(
        markets[["market_id", "start_ms", "end_ms", "slug_ms"]], on="market_id", how="left"
    )

    # Three candidate semantics for the slug epoch:
    # A) slug epoch is market OPEN -> close = slug + 15m
    # B) slug epoch is market CLOSE -> close = slug
    # C) metadata end_time is close (the current strict replay assumption)
    x["slug_open_close_ms"] = x["slug_ms"] + 15 * 60_000
    for prefix, close_col in [("meta", "end_ms"), ("slug_open", "slug_open_close_ms"), ("slug_close", "slug_ms")]:
        x[f"book_mid_s2c_{prefix}"] = (x[close_col] - x["median"]) / 1000.0
        x[f"book_last_s2c_{prefix}"] = (x[close_col] - x["max"]) / 1000.0
        x[f"trade_last_s2c_{prefix}"] = (x[close_col] - x["trade_max"]) / 1000.0

    # Snapshot-level window counts under each candidate close-time interpretation.
    meta_map = x.set_index("market_id")["end_ms"].to_dict()
    slug_open_map = x.set_index("market_id")["slug_open_close_ms"].to_dict()
    slug_close_map = x.set_index("market_id")["slug_ms"].to_dict()
    s2c_meta = []; s2c_slug_open = []; s2c_slug_close = []
    for r in book[["market_id", "ts_ms"]].itertuples(index=False):
        m = str(r.market_id); ts = int(r.ts_ms)
        if pd.notna(meta_map.get(m)): s2c_meta.append((float(meta_map[m]) - ts) / 1000.0)
        if pd.notna(slug_open_map.get(m)): s2c_slug_open.append((float(slug_open_map[m]) - ts) / 1000.0)
        if pd.notna(slug_close_map.get(m)): s2c_slug_close.append((float(slug_close_map[m]) - ts) / 1000.0)

    # Absolute offsets make systematic timezone/semantic errors obvious.
    report = {
        "day": day,
        "source": {"repo": PM_REPO, "revision": PM_REV},
        "rows": {"book": int(len(book)), "trades": int(len(trades)), "markets_joined": int(len(x))},
        "market_level_offsets_seconds": {
            "meta_start_minus_slug": q((x["start_ms"] - x["slug_ms"]) / 1000.0),
            "meta_end_minus_slug": q((x["end_ms"] - x["slug_ms"]) / 1000.0),
            "book_median_minus_slug": q((x["median"] - x["slug_ms"]) / 1000.0),
            "book_last_minus_slug": q((x["max"] - x["slug_ms"]) / 1000.0),
            "trade_last_minus_slug": q((x["trade_max"] - x["slug_ms"]) / 1000.0),
            "meta_end_minus_book_last": q((x["end_ms"] - x["max"]) / 1000.0),
        },
        "snapshot_s2c_seconds": {
            "using_metadata_end": q(s2c_meta),
            "using_slug_as_open_plus_15m": q(s2c_slug_open),
            "using_slug_as_close": q(s2c_slug_close),
        },
        "snapshot_counts_in_60_600s": {
            "using_metadata_end": count_window(s2c_meta),
            "using_slug_as_open_plus_15m": count_window(s2c_slug_open),
            "using_slug_as_close": count_window(s2c_slug_close),
        },
        "sample_markets": [],
    }

    for r in x.head(20).itertuples(index=False):
        def iso(ms):
            if pd.isna(ms): return None
            return datetime.fromtimestamp(float(ms)/1000.0, tz=timezone.utc).isoformat()
        report["sample_markets"].append({
            "market_id": r.market_id,
            "slug_epoch_utc": iso(r.slug_ms),
            "metadata_start_utc": iso(r.start_ms), "metadata_end_utc": iso(r.end_ms),
            "book_first_utc": iso(r.min), "book_last_utc": iso(r.max),
            "trade_first_utc": iso(r.trade_min), "trade_last_utc": iso(r.trade_max),
        })

    x.to_csv(args.out / "market_time_rows.csv", index=False)
    (args.out / "time_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
