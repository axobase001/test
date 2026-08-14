from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

import pm_structural.openmarket_strict as base
import pm_structural.obadiaha_strict as legacy
from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)


def load_open_day(day: date, files: list[str], cache: Path):
    """Load PM market ticks plus the *global* synchronized Binance BTC tape.

    OpenMarket's polymarket_ticks_ms is market-partitioned by market_slug, while
    binance_ticks_ms is a single BTC trade stream and intentionally has no
    market_slug column. The first strict adapter incorrectly assumed otherwise.
    """
    key = day.isoformat()
    pp = sorted(p for p in files if p.startswith(f"unified/polymarket_ticks_ms/date={key}/") and p.endswith(".parquet"))
    bp = sorted(p for p in files if p.startswith(f"unified/binance_ticks_ms/date={key}/") and p.endswith(".parquet"))
    if not pp or not bp:
        return pd.DataFrame(), pd.DataFrame(), pp, bp

    pm_cols = ["source_ts_ms", "market_slug", "asset_id", "side_label", "event_type",
               "best_bid", "best_ask", "size", "paired"]
    # Actual frozen v0.4.3 schema: id, source_ts_ms, ingest_ts_ms,
    # trade_time_ms, price, volume, date. No market_slug by design.
    bn_cols = ["source_ts_ms", "trade_time_ms", "price", "volume"]
    pm = base.concat_parquet(pp, cache / "openmarket", pm_cols)
    bn = base.concat_parquet(bp, cache / "openmarket", bn_cols)

    pm = pm[pm["market_slug"].astype(str).str.startswith("btc-updown-15m-")].copy()
    if pm.empty or bn.empty:
        return pm, bn, pp, bp

    pm["market_slug"] = pm["market_slug"].astype(str)
    pm["asset_id"] = pm["asset_id"].astype(str)
    pm["side_label"] = pm["side_label"].astype(str).str.upper()
    for c in ["source_ts_ms", "best_bid", "best_ask", "size"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    for c in ["source_ts_ms", "trade_time_ms", "price", "volume"]:
        bn[c] = pd.to_numeric(bn[c], errors="coerce")

    pm = pm.dropna(subset=["source_ts_ms", "best_ask", "size"]).copy()
    bn = bn.dropna(subset=["source_ts_ms", "price"]).copy()
    pm = pm[(pm["side_label"].isin(["UP", "DOWN"])) & (pm["best_ask"] > 0) & (pm["best_ask"] < 1) & (pm["size"] > 0)].copy()
    bn = bn[bn["price"] > 0].copy()
    pm["source_ts_ms"] = pm["source_ts_ms"].astype("int64")
    bn["source_ts_ms"] = bn["source_ts_ms"].astype("int64")
    pm = pm.sort_values(["market_slug", "side_label", "source_ts_ms"]).drop_duplicates(
        ["market_slug", "side_label", "source_ts_ms"], keep="last")
    bn = bn.sort_values("source_ts_ms").drop_duplicates("source_ts_ms", keep="last")
    return pm, bn, pp, bp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-03-02")
    ap.add_argument("--end", default="2026-03-18")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--grid-ms", type=int, default=1000)
    ap.add_argument("--max-quote-age-ms", type=int, default=2000)
    ap.add_argument("--max-spot-age-ms", type=int, default=2000)
    ap.add_argument("--latency-ms", type=int, default=1000)
    ap.add_argument("--tape-window-ms", type=int, default=15000)
    ap.add_argument("--base-stake", type=float, default=5.0)
    args = ap.parse_args()
    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(base.OPEN_REPO, revision=base.OPEN_REV)
    if str(info.sha) != base.OPEN_REV:
        raise RuntimeError(f"OpenMarket revision did not resolve immutably: {info.sha}")
    files = api.list_repo_files(base.OPEN_REPO, repo_type="dataset", revision=base.OPEN_REV)

    meta, audit_df, resolutions = base.load_market_meta(args.cache / "obadiaha", start, end)
    audit_df.to_csv(args.out / "clock_audit.csv", index=False)
    print(f"Pinned OpenMarket {base.OPEN_REPO}@{base.OPEN_REV}; eligible meta={len(meta)}", flush=True)

    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn_anchor = BinanceAnchor.from_df(bn_df)
    inst = deribit_instruments()
    selected = select_deribit_instruments(inst, bn_df, start, end)
    (args.out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n")
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"Deribit selected={len(selected)} usable={len(der.ts)}", flush=True)

    clob = legacy.ClobStaticCache(args.cache / "clob_static.json")
    signals: list[dict] = []; fills: list[dict] = []; coverage = {}; market_counts = {}
    for d in base.daterange(start, end):
        pm, bnt, pp, bp = load_open_day(d, files, args.cache)
        coverage[d.isoformat()] = {"pm_parts": pp, "bn_parts": bp, "available": bool(pp and bp)}
        if pm.empty or bnt.empty:
            print(f"DAY {d} skipped: OpenMarket partition missing/empty", flush=True)
            continue
        tape = base.load_obadiaha_trades(d, args.cache)
        pm_markets = sorted(set(pm["market_slug"].unique()) & set(meta))
        market_counts[d.isoformat()] = len(pm_markets)
        print(f"DAY {d} pm_rows={len(pm)} bn_rows={len(bnt)} tape_rows={len(tape)} markets={len(pm_markets)}", flush=True)
        day_s = day_f = 0
        for slug in pm_markets:
            pmg = pm[pm["market_slug"] == slug]
            sig, fill = base.process_market(
                slug, pmg, bnt, tape, meta[slug], resolutions.get(slug),
                bn_anchor, der, clob, args.threshold, args.grid_ms,
                args.max_quote_age_ms, args.max_spot_age_ms,
                args.latency_ms, args.tape_window_ms, args.base_stake,
            )
            if sig is not None:
                signals.append(sig); day_s += 1
            if fill is not None:
                fills.append(fill); day_f += 1
        clob.save()
        print(f"DAY {d} signals={day_s} strict_base_fills={day_f} cumulative={len(fills)}", flush=True)

    sig_df = pd.DataFrame(signals); fill_df = pd.DataFrame(fills)
    sig_df.to_csv(args.out / "signals_openmarket_strict.csv", index=False)
    base_cols = ["ts_ms", "close_ts_ms", "cid", "action", "cost", "real_reward", "capacity_usd"]
    if fill_df.empty:
        pd.DataFrame(columns=base_cols).to_csv(args.out / "fills_openmarket_strict.csv", index=False)
    else:
        fill_df.to_csv(args.out / "fills_openmarket_strict.csv", index=False)

    total_stake = float(fill_df["base_stake"].sum()) if len(fill_df) else 0.0
    total_pnl = float(fill_df["base_pnl"].sum()) if len(fill_df) else 0.0
    summary = {
        "classification": base.EXECUTION_GRADE,
        "adapter_revision": "global-binance-tape-v2",
        "headline_period": [start.isoformat(), end.isoformat()],
        "sources": {
            "quote_truth": {"repo": base.OPEN_REPO, "revision": base.OPEN_REV, "split": "v0.4.3-unified"},
            "real_tape_and_resolution": {"repo": legacy.PM_REPO, "revision": legacy.PM_REV},
            "binance_rv": "official BTCUSDT 1m; only closed candles <= decision time",
            "spot_now": "OpenMarket global synchronized Binance ms tick; causal previous tick <= decision time",
            "deribit": "historical BTC option trades; backward-only 30m IV median",
        },
        "coverage": coverage,
        "markets_per_available_day": market_counts,
        "missing_openmarket_dates": [k for k, v in coverage.items() if not v["available"]],
        "clock": {
            "canonical_market_boundary": "btc-updown-15m-<unix_open_seconds>; close=open+900s",
            "metadata_audit_rows": int(len(audit_df)),
            "metadata_all_consistent": bool(len(audit_df) and audit_df["ok"].all()),
        },
        "policy": {
            "family": "frozen structural two-anchor fade",
            "threshold": args.threshold,
            "decision_window_s2c": [60, 600],
            "decision_grid_ms": args.grid_ms,
            "max_quote_age_ms": args.max_quote_age_ms,
            "max_spot_age_ms": args.max_spot_age_ms,
            "no_outcome_in_decision": True,
            "once_per_market": True,
        },
        "execution": {
            "latency_ms": args.latency_ms,
            "tape_window_ms": args.tape_window_ms,
            "required": "OpenMarket top ask/depth at decision; fresh causal state at +latency still <= limit; independent same-token Obadiaha BUY tape <= limit",
            "capacity_shares": "min(OpenMarket decision top depth, OpenMarket latency top depth, largest qualifying independent tape trade)",
            "partial_fills": False,
            "base_fill_requires_usd": args.base_stake,
        },
        "historical_fee": {
            "formula": "V1 BUY fee_tokens/gross_share=(tbf/10000)*min(p,1-p)/p",
            "taker_base_fee_bps_seen": sorted({int(x["v1_taker_base_fee_bps"]) for x in signals if x.get("v1_taker_base_fee_bps") is not None}),
            "source": "per-market CLOB tbf plus archived V1 on-chain CalculatorHelper semantics; independently reconciled to raw OrderFilled",
        },
        "signals": int(len(sig_df)),
        "strict_base_fills": int(len(fill_df)),
        "fill_rate_given_signal": float(len(fill_df) / len(sig_df)) if len(sig_df) else None,
        "wins": int((fill_df["payout_per_net_share"] > 0.5).sum()) if len(fill_df) else 0,
        "losses": int((fill_df["payout_per_net_share"] < 0.5).sum()) if len(fill_df) else 0,
        "base_total_stake": total_stake,
        "base_total_pnl": total_pnl,
        "base_fee_adjusted_roi": total_pnl / total_stake if total_stake else None,
        "base_day_cluster_95ci_roi": legacy.cluster_bootstrap_roi(fill_df),
        "median_capacity_usd": float(fill_df["capacity_usd"].median()) if len(fill_df) else None,
        "min_capacity_usd": float(fill_df["capacity_usd"].min()) if len(fill_df) else None,
        "mean_edge_equiv": float(fill_df["edge_equiv"].mean()) if len(fill_df) else None,
        "mean_edge_exact_net_ev": float(fill_df["edge_exact_net_ev"].mean()) if len(fill_df) else None,
        "deribit_selected_instruments": len(selected),
        "deribit_usable_trades": len(der.ts),
        "caveat": "OpenMarket has missing unified PM-tick partitions on some dates in this window. Headline statistics use only explicitly listed available dates. Execution requires an independent real-tape witness and full target-size capacity; no partial fills are assumed.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.out / "EXECUTION_GRADE.txt").write_text(base.EXECUTION_GRADE + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
