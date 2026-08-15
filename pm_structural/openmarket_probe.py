from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from pm_structural.obadiaha_strict import PM_REPO, PM_REV, hf_file, to_ms

OPEN_REPO = "gregyoung14/openmarket-btc-polymarket"


def q(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
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


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def open_file(path: str, revision: str, cache: Path) -> Path:
    return Path(hf_hub_download(
        repo_id=OPEN_REPO,
        repo_type="dataset",
        revision=revision,
        filename=path,
        cache_dir=cache,
    ))


def concat_parquet(paths: list[str], revision: str, cache: Path, columns: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        local = open_file(p, revision, cache)
        frames.append(pd.read_parquet(local, columns=columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def build_quote_groups(df: pd.DataFrame, slug_col: str, token_col: str, ts_col: str,
                       bid_col: str, ask_col: str, size_col: str | None = None) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for (slug, token), g in df.groupby([slug_col, token_col], sort=False):
        g = g.sort_values(ts_col).drop_duplicates(ts_col, keep="last")
        row = {
            "ts": g[ts_col].to_numpy(np.int64),
            "bid": pd.to_numeric(g[bid_col], errors="coerce").to_numpy(float),
            "ask": pd.to_numeric(g[ask_col], errors="coerce").to_numpy(float),
        }
        if size_col is not None and size_col in g.columns:
            row["size"] = pd.to_numeric(g[size_col], errors="coerce").to_numpy(float)
        out[(str(slug), str(token))] = row
    return out


def compare_trades(trades: pd.DataFrame, groups: dict[tuple[str, str], dict[str, np.ndarray]],
                   max_lag_ms: int, label: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    for r in trades.itertuples(index=False):
        key = (str(r.market_id), str(r.token_id))
        g = groups.get(key)
        if g is None or not len(g["ts"]):
            continue
        i = int(np.searchsorted(g["ts"], int(r.ts_ms), side="right") - 1)
        if i < 0:
            continue
        lag = int(r.ts_ms) - int(g["ts"][i])
        if lag < 0 or lag > max_lag_ms:
            continue
        side = str(r.side).upper()
        quote = float(g["ask"][i]) if side == "BUY" else float(g["bid"][i])
        if not math.isfinite(quote) or quote <= 0 or quote >= 1:
            continue
        px = float(r.price)
        abs_gap = abs(px - quote)
        signed_gap = px - quote
        row = {
            "source": label, "market_id": key[0], "token_id": key[1], "side": side,
            "trade_ts_ms": int(r.ts_ms), "quote_ts_ms": int(g["ts"][i]), "lag_ms": lag,
            "trade_price": px, "quote_price": quote, "signed_gap": signed_gap, "abs_gap": abs_gap,
            "tx_hash": str(r.tx_hash),
        }
        if "size" in g:
            row["top_size"] = float(g["size"][i])
            row["top_notional"] = float(g["size"][i]) * quote
        rows.append(row)
    x = pd.DataFrame(rows)
    if x.empty:
        return {"matched": 0}, x
    buy = x[x["side"] == "BUY"]
    sell = x[x["side"] == "SELL"]
    def stats(y: pd.DataFrame) -> dict:
        if y.empty:
            return {"n": 0}
        a = y["abs_gap"].to_numpy(float)
        return {
            "n": int(len(y)),
            "lag_ms": q(y["lag_ms"].tolist()),
            "abs_price_gap": q(a.tolist()),
            "within_0p5c": float(np.mean(a <= 0.005 + 1e-12)),
            "within_1c": float(np.mean(a <= 0.01 + 1e-12)),
            "within_2c": float(np.mean(a <= 0.02 + 1e-12)),
            "within_5c": float(np.mean(a <= 0.05 + 1e-12)),
        }
    return {"matched": int(len(x)), "all": stats(x), "buy": stats(buy), "sell": stats(sell)}, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--day", default="2026-03-14")
    ap.add_argument("--coverage-start", default="2026-03-02")
    ap.add_argument("--coverage-end", default="2026-03-18")
    ap.add_argument("--max-lag-ms", type=int, default=2000)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(args.day)
    c0 = date.fromisoformat(args.coverage_start); c1 = date.fromisoformat(args.coverage_end)

    api = HfApi()
    info = api.dataset_info(OPEN_REPO, revision="main")
    open_rev = str(info.sha)
    files = api.list_repo_files(OPEN_REPO, repo_type="dataset", revision=open_rev)

    coverage = {}
    for day in daterange(c0, c1):
        key = day.isoformat()
        prefix = f"unified/polymarket_ticks_ms/date={key}/"
        hits = sorted(p for p in files if p.startswith(prefix) and p.endswith(".parquet"))
        coverage[key] = hits

    target_prefix = f"unified/polymarket_ticks_ms/date={d.isoformat()}/"
    tick_paths = sorted(p for p in files if p.startswith(target_prefix) and p.endswith(".parquet"))
    if not tick_paths:
        raise RuntimeError(f"OpenMarket has no unified polymarket tick partition for {d}")

    tick_cols = ["source_ts_ms", "market_slug", "asset_id", "side_label", "event_type",
                 "price", "best_bid", "best_ask", "size", "paired"]
    ticks = concat_parquet(tick_paths, open_rev, args.cache / "openmarket", tick_cols)
    ticks = ticks[ticks["market_slug"].astype(str).str.startswith("btc-updown-15m-")].copy()
    ticks["market_slug"] = ticks["market_slug"].astype(str); ticks["asset_id"] = ticks["asset_id"].astype(str)
    ticks["source_ts_ms"] = pd.to_numeric(ticks["source_ts_ms"], errors="coerce")
    ticks = ticks.dropna(subset=["source_ts_ms", "best_bid", "best_ask"]).copy()
    ticks["source_ts_ms"] = ticks["source_ts_ms"].astype(np.int64)

    # Independent real trade tape from Obadiaha Data API archive.
    tcols = ["timestamp", "asset", "market_id", "condition_id", "token_id", "side", "price", "size", "tx_hash"]
    trades = pd.read_parquet(hf_file(f"trades/{d.isoformat()}.parquet", args.cache / "obadiaha"), columns=tcols)
    trades = trades[(trades["asset"] == "BTC") & trades["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    trades["market_id"] = trades["market_id"].astype(str); trades["token_id"] = trades["token_id"].astype(str)
    trades["ts_ms"] = to_ms(trades["timestamp"])
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce"); trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    trades = trades.dropna(subset=["ts_ms", "price", "size"]).copy(); trades["ts_ms"] = trades["ts_ms"].astype(np.int64)

    open_groups = build_quote_groups(ticks, "market_slug", "asset_id", "source_ts_ms", "best_bid", "best_ask", "size")
    open_stats, open_matches = compare_trades(trades, open_groups, args.max_lag_ms, "openmarket")

    # Same test against Obadiaha's /book-derived snapshots: this is the suspected ghost-book source.
    bcols = ["timestamp", "asset", "market_id", "token_id", "best_bid", "best_ask"]
    book = pd.read_parquet(hf_file(f"orderbooks/{d.isoformat()}.parquet", args.cache / "obadiaha"), columns=bcols)
    book = book[(book["asset"] == "BTC") & book["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    book["market_id"] = book["market_id"].astype(str); book["token_id"] = book["token_id"].astype(str)
    book["ts_ms"] = to_ms(book["timestamp"]); book = book.dropna(subset=["ts_ms", "best_bid", "best_ask"]).copy()
    book["ts_ms"] = book["ts_ms"].astype(np.int64)
    ob_groups = build_quote_groups(book, "market_id", "token_id", "ts_ms", "best_bid", "best_ask")
    ob_stats, ob_matches = compare_trades(trades, ob_groups, max(args.max_lag_ms, 15000), "obadiaha_book")

    open_matches.to_csv(args.out / "openmarket_trade_matches.csv", index=False)
    ob_matches.to_csv(args.out / "obadiaha_book_trade_matches.csv", index=False)

    report = {
        "day": d.isoformat(),
        "openmarket": {
            "repo": OPEN_REPO, "revision_sha": open_rev,
            "dataset_version_claim": "v0.4.3-unified", "source_tag_claim": "v0.5.2",
            "tick_paths": tick_paths, "tick_rows_btc15m": int(len(ticks)),
            "markets": int(ticks["market_slug"].nunique()), "tokens": int(ticks["asset_id"].nunique()),
            "side_labels": ticks["side_label"].astype(str).value_counts().to_dict(),
            "event_types": ticks["event_type"].astype(str).value_counts().head(20).to_dict(),
            "best_ask": q(pd.to_numeric(ticks["best_ask"], errors="coerce").dropna().tolist()),
            "best_bid": q(pd.to_numeric(ticks["best_bid"], errors="coerce").dropna().tolist()),
        },
        "obadiaha": {"repo": PM_REPO, "revision_sha": PM_REV, "trade_rows_btc15m": int(len(trades)),
                      "book_rows_btc15m": int(len(book))},
        "coverage_openmarket_unified_pm_ticks": {k: len(v) for k, v in coverage.items()},
        "coverage_missing_dates": [k for k, v in coverage.items() if not v],
        "trade_quote_validation": {
            "causal_previous_quote_max_lag_ms_openmarket": args.max_lag_ms,
            "openmarket": open_stats,
            "obadiaha_book": ob_stats,
        },
        "decision_rule": "If OpenMarket causal top-of-book is materially closer to independent real tape than Obadiaha /book snapshots, promote OpenMarket for historical quote truth and retain Obadiaha for independent tape corroboration.",
    }
    (args.out / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
