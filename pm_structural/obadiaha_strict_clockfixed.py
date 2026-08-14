from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import pm_structural.obadiaha_strict as legacy
from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)
from pm_structural.time_units import audit_btc15m_metadata, canonical_btc15m_clock, epoch_series_to_ms


# Repair the historical collector's epoch-us columns everywhere inside the frozen
# strict implementation without changing its signal or execution policy.
legacy.to_ms = epoch_series_to_ms
EXECUTION_GRADE = legacy.EXECUTION_GRADE + "_CANONICAL_SLUG_CLOCK"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_market_meta(markets: pd.DataFrame, start: date, end: date) -> tuple[dict[str, dict], pd.DataFrame]:
    x = markets[(markets["asset"] == "BTC") & (markets["market_type"] == "crypto_15m")].copy()
    x["market_id"] = x["market_id"].astype(str)
    lo_s = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    end2 = end + timedelta(days=1)
    hi_s = int(datetime(end2.year, end2.month, end2.day, tzinfo=timezone.utc).timestamp())

    rows = []
    out: dict[str, dict] = {}
    for r in x.itertuples(index=False):
        slug = str(r.market_id)
        try:
            open_s, close_ms = canonical_btc15m_clock(slug)
        except ValueError:
            continue
        if not (lo_s <= open_s < hi_s):
            continue
        audit = audit_btc15m_metadata(slug, r.start_time, r.end_time)
        rows.append(audit)
        if not audit["ok"]:
            # A canonical slug clock is independently encoded in the market name,
            # but a material disagreement with metadata must be investigated rather
            # than silently admitted into a PnL run.
            continue
        out[slug] = {
            "condition_id": str(r.condition_id),
            "open_ts_s": int(open_s),
            "close_ts_ms": int(close_ms),
        }
    audit_df = pd.DataFrame(rows)
    bad = audit_df[~audit_df["ok"]] if len(audit_df) else audit_df
    if len(bad):
        examples = bad.head(10).to_dict("records")
        raise RuntimeError(f"BTC15m metadata disagrees with canonical slug clock: n={len(bad)} examples={examples}")
    return out, audit_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-03-02")
    ap.add_argument("--end", default="2026-03-18")
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--latency-ms", type=int, default=1000)
    ap.add_argument("--tape-window-ms", type=int, default=15000)
    ap.add_argument("--base-stake", type=float, default=5.0)
    args = ap.parse_args()
    start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    hf_cache = args.cache / "hf"

    print(f"Pinned PM source {legacy.PM_REPO}@{legacy.PM_REV}", flush=True)
    markets = pd.read_parquet(legacy.hf_file("markets/all.parquet", hf_cache))
    resolutions_df = pd.read_parquet(legacy.hf_file("resolutions/all.parquet", hf_cache))
    market_meta, clock_audit = build_market_meta(markets, start, end)
    clock_audit.to_csv(args.out / "clock_audit.csv", index=False)
    clock_summary = {
        "selected_markets": int(len(market_meta)),
        "audit_rows": int(len(clock_audit)),
        "all_metadata_consistent_with_slug_clock": bool(len(clock_audit) and clock_audit["ok"].all()),
        "max_abs_start_delta_ms": int(clock_audit["start_delta_ms"].abs().max()) if len(clock_audit) else None,
        "max_abs_end_delta_ms": int(clock_audit["end_delta_ms"].abs().max()) if len(clock_audit) else None,
        "duration_ms_min": int(clock_audit["metadata_duration_ms"].min()) if len(clock_audit) else None,
        "duration_ms_max": int(clock_audit["metadata_duration_ms"].max()) if len(clock_audit) else None,
        "canonical_source": "btc-updown-15m-<unix_open_seconds>; close=open+900s",
        "raw_timestamp_parser": "magnitude-aware s/ms/us/ns -> epoch ms",
    }
    (args.out / "clock_summary.json").write_text(json.dumps(clock_summary, indent=2))
    print(json.dumps({"clock": clock_summary}, indent=2), flush=True)

    resolutions = legacy.settlement_map(resolutions_df)
    print(f"Eligible BTC15m markets on canonical clock: {len(market_meta)}", flush=True)

    print("Downloading official Binance BTCUSDT 1m...", flush=True)
    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)
    print("Discovering Deribit option instruments...", flush=True)
    inst = deribit_instruments()
    selected = select_deribit_instruments(inst, bn_df, start, end)
    (args.out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n")
    print(f"Selected Deribit instruments: {len(selected)}", flush=True)
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"Deribit usable anchor trades: {len(der.ts)}", flush=True)

    clob = legacy.ClobStaticCache(args.cache / "clob_static.json")
    signals: list[dict] = []; fills: list[dict] = []
    for d in daterange(start, end):
        print(f"DAY {d.isoformat()} start", flush=True)
        s, f = legacy.process_day(
            d, market_meta, resolutions, bn, der, clob,
            args.threshold, args.latency_ms, args.tape_window_ms,
            args.base_stake, hf_cache,
        )
        for row in s:
            row["execution_grade"] = EXECUTION_GRADE
            row["clock_source"] = "slug_epoch_open_plus_900s"
        for row in f:
            row["execution_grade"] = EXECUTION_GRADE
            row["clock_source"] = "slug_epoch_open_plus_900s"
        signals.extend(s); fills.extend(f); clob.save()
        print(f"DAY {d.isoformat()} signals={len(s)} strict_base_fills={len(f)} cumulative={len(fills)}", flush=True)
    clob.save()

    fill_columns = ["ts_ms", "close_ts_ms", "cid", "action", "cost", "real_reward", "capacity_usd"]
    sig_df = pd.DataFrame(signals); fill_df = pd.DataFrame(fills)
    sig_df.to_csv(args.out / "signals_strict_coarse.csv", index=False)
    if fill_df.empty:
        pd.DataFrame(columns=fill_columns).to_csv(args.out / "fills_strict_coarse.csv", index=False)
    else:
        fill_df.to_csv(args.out / "fills_strict_coarse.csv", index=False)

    base_fee_values = sorted({int(x["v1_taker_base_fee_bps"]) for x in signals if x.get("v1_taker_base_fee_bps") is not None})
    fd_values = sorted({(x.get("fd_rate_metadata_only"), x.get("fd_exponent_metadata_only")) for x in signals})
    total_stake = float(fill_df["base_stake"].sum()) if len(fill_df) else 0.0
    total_pnl = float(fill_df["base_pnl"].sum()) if len(fill_df) else 0.0
    summary = {
        "classification": EXECUTION_GRADE,
        "headline_period": [start.isoformat(), end.isoformat()],
        "source": {"repo_id": legacy.PM_REPO, "revision": legacy.PM_REV},
        "clock": clock_summary,
        "policy": {
            "family": "frozen structural two-anchor fade", "threshold": args.threshold,
            "decision_window_s2c": [60, 600],
            "signal_edge": "conservative fair minus ask minus V1 on-chain USDC-equivalent taker fee",
            "once_per_market": True, "no_outcome_in_decision": True,
        },
        "execution": {
            "latency_ms": args.latency_ms, "tape_window_ms": args.tape_window_ms,
            "required": "post-latency book persistence plus same-token BUY tape at or below limit",
            "capacity_shares": "min(decision ask depth, post-latency ask depth, largest single qualifying BUY tape trade)",
            "base_fill_requires_usd": args.base_stake, "partial_fills": False,
        },
        "historical_fee": {
            "v1_taker_base_fee_bps_seen": base_fee_values,
            "formula": "BUY fee_tokens/gross_share=(tbf/10000)*min(p,1-p)/p; USDC-equivalent fee/gross_share=(tbf/10000)*min(p,1-p)",
            "source": "archived V1 CTF Exchange CalculatorHelper semantics + per-market CLOB tbf",
            "fd_metadata_seen_but_not_used_for_v1_pnl": fd_values,
        },
        "markets_metadata": len(market_meta), "mapping_api_calls": clob.calls,
        "signals": int(len(sig_df)), "strict_base_fills": int(len(fill_df)),
        "fill_rate_given_signal": float(len(fill_df) / len(sig_df)) if len(sig_df) else None,
        "wins": int((fill_df["payout_per_net_share"] > 0.5).sum()) if len(fill_df) else 0,
        "losses": int((fill_df["payout_per_net_share"] < 0.5).sum()) if len(fill_df) else 0,
        "base_total_stake": total_stake, "base_total_pnl": total_pnl,
        "base_fee_adjusted_roi": total_pnl / total_stake if total_stake else None,
        "base_day_cluster_95ci_roi": legacy.cluster_bootstrap_roi(fill_df),
        "mean_edge_equiv": float(fill_df["edge_equiv"].mean()) if len(fill_df) else None,
        "mean_edge_exact_net_ev": float(fill_df["edge_exact_net_ev"].mean()) if len(fill_df) else None,
        "median_capacity_usd": float(fill_df["capacity_usd"].median()) if len(fill_df) else None,
        "min_capacity_usd": float(fill_df["capacity_usd"].min()) if len(fill_df) else None,
        "deribit_selected_instruments": len(selected), "deribit_usable_trades": len(der.ts),
        "caveat": "Obadiaha order books are coarse snapshots. Real same-token tape is required. Raw collector timestamps are epoch microseconds and are normalized explicitly; canonical market boundaries come from the slug epoch invariant.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.out / "EXECUTION_GRADE.txt").write_text(EXECUTION_GRADE + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
