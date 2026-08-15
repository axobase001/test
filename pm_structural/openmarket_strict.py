from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

import pm_structural.obadiaha_strict as legacy
from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    digital_prob_up,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)
from pm_structural.time_units import audit_btc15m_metadata, canonical_btc15m_clock, epoch_series_to_ms

OPEN_REPO = "gregyoung14/openmarket-btc-polymarket"
OPEN_REV = "74502466d1a7cef56395bfd8d0b465fbebc849cf"
EXECUTION_GRADE = "STRICT_TAPE_OPENMARKET_TOPDEPTH_GRID1S_V1_ONCHAIN_FEE"
EPS = 1e-9

# The legacy V1 fee and CLOB metadata implementation is retained, but all
# historical timestamps are normalized through the audited unit-aware parser.
legacy.to_ms = epoch_series_to_ms


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def open_file(rel: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(
        repo_id=OPEN_REPO,
        repo_type="dataset",
        revision=OPEN_REV,
        filename=rel,
        cache_dir=cache_dir,
    ))


def concat_parquet(paths: list[str], cache_dir: Path, columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(open_file(p, cache_dir), columns=columns) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def state_indices(ts: np.ndarray, query: np.ndarray, max_age_ms: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(ts, query, side="right") - 1
    valid = idx >= 0
    safe = np.maximum(idx, 0)
    age = query - ts[safe]
    valid &= (age >= 0) & (age <= max_age_ms)
    return safe, valid


def one_state(ts: np.ndarray, query: int, max_age_ms: int) -> int | None:
    i = int(np.searchsorted(ts, int(query), side="right") - 1)
    if i < 0:
        return None
    age = int(query) - int(ts[i])
    if age < 0 or age > max_age_ms:
        return None
    return i


def load_market_meta(hf_cache: Path, start: date, end: date) -> tuple[dict[str, dict], pd.DataFrame, dict[str, str]]:
    markets = pd.read_parquet(legacy.hf_file("markets/all.parquet", hf_cache))
    resolutions = pd.read_parquet(legacy.hf_file("resolutions/all.parquet", hf_cache))
    x = markets[(markets["asset"] == "BTC") & (markets["market_type"] == "crypto_15m")].copy()
    x["market_id"] = x["market_id"].astype(str)
    lo_s = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
    end2 = end + timedelta(days=1)
    hi_s = int(datetime(end2.year, end2.month, end2.day, tzinfo=timezone.utc).timestamp())

    meta: dict[str, dict] = {}
    audits = []
    for r in x.itertuples(index=False):
        slug = str(r.market_id)
        try:
            open_s, close_ms = canonical_btc15m_clock(slug)
        except ValueError:
            continue
        if not (lo_s <= open_s < hi_s):
            continue
        a = audit_btc15m_metadata(slug, r.start_time, r.end_time)
        audits.append(a)
        if not a["ok"]:
            continue
        meta[slug] = {
            "condition_id": str(r.condition_id),
            "open_ts_s": int(open_s),
            "close_ts_ms": int(close_ms),
        }
    audit_df = pd.DataFrame(audits)
    if len(audit_df) and not bool(audit_df["ok"].all()):
        bad = audit_df[~audit_df["ok"]].head(10).to_dict("records")
        raise RuntimeError(f"metadata/slug clock mismatch: {bad}")
    res = legacy.settlement_map(resolutions)
    return meta, audit_df, res


def load_open_day(day: date, files: list[str], cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    key = day.isoformat()
    pp = sorted(p for p in files if p.startswith(f"unified/polymarket_ticks_ms/date={key}/") and p.endswith(".parquet"))
    bp = sorted(p for p in files if p.startswith(f"unified/binance_ticks_ms/date={key}/") and p.endswith(".parquet"))
    if not pp or not bp:
        return pd.DataFrame(), pd.DataFrame(), pp, bp

    pm_cols = ["source_ts_ms", "market_slug", "asset_id", "side_label", "event_type",
               "best_bid", "best_ask", "size", "paired"]
    bn_cols = ["source_ts_ms", "market_slug", "price", "best_bid", "best_ask"]
    pm = concat_parquet(pp, cache / "openmarket", pm_cols)
    bn = concat_parquet(bp, cache / "openmarket", bn_cols)

    pm = pm[pm["market_slug"].astype(str).str.startswith("btc-updown-15m-")].copy()
    bn = bn[bn["market_slug"].astype(str).str.startswith("btc-updown-15m-")].copy()
    if pm.empty or bn.empty:
        return pm, bn, pp, bp

    pm["market_slug"] = pm["market_slug"].astype(str)
    pm["asset_id"] = pm["asset_id"].astype(str)
    pm["side_label"] = pm["side_label"].astype(str).str.upper()
    bn["market_slug"] = bn["market_slug"].astype(str)
    for c in ["source_ts_ms", "best_bid", "best_ask", "size"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    for c in ["source_ts_ms", "price", "best_bid", "best_ask"]:
        bn[c] = pd.to_numeric(bn[c], errors="coerce")
    pm = pm.dropna(subset=["source_ts_ms", "best_ask", "size"]).copy()
    bn = bn.dropna(subset=["source_ts_ms", "price"]).copy()
    pm = pm[(pm["side_label"].isin(["UP", "DOWN"])) & (pm["best_ask"] > 0) & (pm["best_ask"] < 1) & (pm["size"] > 0)].copy()
    bn = bn[bn["price"] > 0].copy()
    pm["source_ts_ms"] = pm["source_ts_ms"].astype(np.int64)
    bn["source_ts_ms"] = bn["source_ts_ms"].astype(np.int64)
    pm = pm.sort_values(["market_slug", "side_label", "source_ts_ms"]).drop_duplicates(
        ["market_slug", "side_label", "source_ts_ms"], keep="last")
    bn = bn.sort_values(["market_slug", "source_ts_ms"]).drop_duplicates(
        ["market_slug", "source_ts_ms"], keep="last")
    return pm, bn, pp, bp


def load_obadiaha_trades(day: date, cache: Path) -> pd.DataFrame:
    cols = ["timestamp", "asset", "market_id", "condition_id", "token_id", "side", "price", "size", "tx_hash"]
    t = pd.read_parquet(legacy.hf_file(f"trades/{day.isoformat()}.parquet", cache / "obadiaha"), columns=cols)
    t = t[(t["asset"] == "BTC") & t["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    t["market_id"] = t["market_id"].astype(str)
    t["token_id"] = t["token_id"].astype(str)
    t["ts_ms"] = epoch_series_to_ms(t["timestamp"])
    t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t["size"] = pd.to_numeric(t["size"], errors="coerce")
    t = t.dropna(subset=["ts_ms", "price", "size"]).copy()
    t["ts_ms"] = t["ts_ms"].astype(np.int64)
    return t.sort_values(["market_id", "token_id", "ts_ms"])


def process_market(
    slug: str,
    pmg: pd.DataFrame,
    bng: pd.DataFrame,
    tape: pd.DataFrame,
    meta: dict,
    outcome: str | None,
    bn_anchor: BinanceAnchor,
    der: DeribitAnchor,
    clob: legacy.ClobStaticCache,
    threshold: float,
    grid_ms: int,
    max_quote_age_ms: int,
    max_spot_age_ms: int,
    latency_ms: int,
    tape_window_ms: int,
    base_stake: float,
) -> tuple[dict | None, dict | None]:
    open_s = int(meta["open_ts_s"])
    close_ms = int(meta["close_ts_ms"])
    open_bn = bn_anchor.open_price(open_s)
    if not (open_bn > 0 and math.isfinite(open_bn)):
        return None, None

    side_stream = {}
    side_token = {}
    for side in ("UP", "DOWN"):
        s = pmg[pmg["side_label"] == side].sort_values("source_ts_ms")
        if s.empty:
            return None, None
        toks = s["asset_id"].astype(str).value_counts()
        if toks.empty:
            return None, None
        token = str(toks.index[0])
        s = s[s["asset_id"].astype(str) == token]
        side_token[side] = token
        side_stream[side] = {
            "ts": s["source_ts_ms"].to_numpy(np.int64),
            "ask": s["best_ask"].to_numpy(float),
            "size": s["size"].to_numpy(float),
        }

    bs = bng.sort_values("source_ts_ms")
    bts = bs["source_ts_ms"].to_numpy(np.int64)
    bpx = bs["price"].to_numpy(float)
    if not len(bts):
        return None, None

    grid = np.arange(close_ms - 600_000, close_ms - 60_000 + 1, int(grid_ms), dtype=np.int64)
    ui, uv = state_indices(side_stream["UP"]["ts"], grid, max_quote_age_ms)
    di, dv = state_indices(side_stream["DOWN"]["ts"], grid, max_quote_age_ms)
    bi, bv = state_indices(bts, grid, max_spot_age_ms)
    valid = uv & dv & bv
    if not valid.any():
        return None, None

    up_ask = side_stream["UP"]["ask"][ui]
    dn_ask = side_stream["DOWN"]["ask"][di]
    up_size = side_stream["UP"]["size"][ui]
    dn_size = side_stream["DOWN"]["size"][di]
    spot_arr = bpx[bi]

    static = None
    chosen = None
    for k in np.flatnonzero(valid):
        ts = int(grid[k])
        s2c = int(round((close_ms - ts) / 1000.0))
        rv = bn_anchor.rv_annualized(ts, 60)
        div = der.median_iv(ts, 30)
        if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
            continue
        spot = float(spot_arr[k])
        rel = spot / open_bn
        p_rv = digital_prob_up(rel, s2c, rv)
        p_iv = digital_prob_up(rel, s2c, div)
        if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
            continue
        fair_up = min(p_rv, p_iv)
        fair_dn = 1.0 - max(p_rv, p_iv)
        asks = {"UP": float(up_ask[k]), "DOWN": float(dn_ask[k])}
        fairs = {"UP": fair_up, "DOWN": fair_dn}
        if max(fairs[s] - asks[s] for s in ("UP", "DOWN")) < threshold:
            continue

        if static is None:
            static = clob.get(str(meta["condition_id"]))
            if static is None or static.taker_base_fee_bps <= 0:
                return None, None
            # Side labels come from OpenMarket itself, but require exact CLOB token
            # agreement before using a historical quote for PnL.
            for side in ("UP", "DOWN"):
                mapped = static.token_outcome.get(side_token[side])
                if str(mapped).upper() != side:
                    return None, None

        candidates = []
        for side in ("UP", "DOWN"):
            ask = asks[side]
            fee_token_fraction, fee_usdc_share = legacy.v1_buy_fee(ask, static.taker_base_fee_bps)
            net_ratio = 1.0 - fee_token_fraction
            if not (math.isfinite(net_ratio) and net_ratio > 0):
                continue
            edge_equiv = fairs[side] - ask - fee_usdc_share
            edge_exact = fairs[side] * net_ratio - ask
            if edge_equiv >= threshold:
                qidx = int(ui[k] if side == "UP" else di[k])
                qstream = side_stream[side]
                candidates.append((edge_equiv, edge_exact, side, ask, fairs[side], net_ratio,
                                   fee_token_fraction, fee_usdc_share, qidx, qstream))
        if not candidates:
            continue
        edge_equiv, edge_exact, side, ask, fair, net_ratio, ftok, fusd, qidx, qstream = max(
            candidates, key=lambda x: (x[0], x[1]))
        chosen = {
            "ts_ms": ts,
            "close_ts_ms": close_ms,
            "cid": slug,
            "condition_id": str(meta["condition_id"]),
            "token_id": side_token[side],
            "action": f"buy_{side.lower()}",
            "outcome_token": side.title(),
            "target_px": ask,
            "decision_depth_shares": float(qstream["size"][qidx]),
            "decision_quote_ts_ms": int(qstream["ts"][qidx]),
            "decision_quote_age_ms": int(ts - int(qstream["ts"][qidx])),
            "spot": spot,
            "spot_ts_ms": int(bts[bi[k]]),
            "spot_age_ms": int(ts - int(bts[bi[k]])),
            "open_spot": float(open_bn),
            "rel_spot": float(rel),
            "p_rv": float(p_rv),
            "p_deribit": float(p_iv),
            "fair_conservative": float(fair),
            "rv": float(rv),
            "deribit_iv": float(div),
            "s2c": s2c,
            "edge_equiv": float(edge_equiv),
            "edge_exact_net_ev": float(edge_exact),
            "v1_maker_base_fee_bps": int(static.maker_base_fee_bps),
            "v1_taker_base_fee_bps": int(static.taker_base_fee_bps),
            "fee_tokens_per_gross_share": float(ftok),
            "fee_usdc_equiv_per_gross_share": float(fusd),
            "net_share_ratio": float(net_ratio),
            "execution_grade": EXECUTION_GRADE,
            "clock_source": "slug_epoch_open_plus_900s",
            "decision_grid_ms": int(grid_ms),
        }
        break

    if chosen is None:
        return None, None

    side = "UP" if chosen["action"] == "buy_up" else "DOWN"
    stream = side_stream[side]
    latency_ts = int(chosen["ts_ms"]) + int(latency_ms)
    li = one_state(stream["ts"], latency_ts, max_quote_age_ms)
    if li is None:
        chosen.update({"filled": False, "reason": "no_fresh_latency_quote"})
        return chosen, None
    latency_ask = float(stream["ask"][li])
    latency_depth = float(stream["size"][li])
    chosen["latency_quote_ts_ms"] = int(stream["ts"][li])
    chosen["latency_quote_age_ms"] = int(latency_ts - int(stream["ts"][li]))
    chosen["latency_ask"] = latency_ask
    chosen["latency_depth_shares"] = latency_depth
    if latency_ask > float(chosen["target_px"]) + EPS or latency_depth <= 0:
        chosen.update({"filled": False, "reason": "quote_not_persistent"})
        return chosen, None

    tg = tape[(tape["market_id"] == slug) & (tape["token_id"] == str(chosen["token_id"]))].copy()
    tg = tg[
        (tg["ts_ms"] >= latency_ts)
        & (tg["ts_ms"] <= int(chosen["ts_ms"]) + int(tape_window_ms))
        & (tg["side"].astype(str).str.upper() == "BUY")
        & (tg["price"] <= float(chosen["target_px"]) + EPS)
        & (tg["size"] > 0)
    ]
    if tg.empty:
        chosen.update({"filled": False, "reason": "no_real_buy_tape_witness"})
        return chosen, None
    witness = tg.sort_values(["size", "ts_ms"], ascending=[False, True]).iloc[0]
    tape_size = float(witness["size"])
    cap_shares = max(min(float(chosen["decision_depth_shares"]), latency_depth, tape_size), 0.0)
    cap_usd = cap_shares * float(chosen["target_px"])
    chosen.update({
        "filled": cap_shares > 0,
        "reason": "strict_witness" if cap_shares > 0 else "zero_capacity",
        "tape_ts_ms": int(witness.ts_ms),
        "tape_price": float(witness.price),
        "tape_size_shares": tape_size,
        "tape_tx_hash": str(witness.tx_hash),
        "capacity_shares": cap_shares,
        "capacity_usd": cap_usd,
    })
    if cap_usd + EPS < base_stake or outcome not in ("Up", "Down"):
        return chosen, None

    payout = 1.0 if chosen["outcome_token"] == outcome else 0.0
    effective_cost = float(chosen["target_px"]) / float(chosen["net_share_ratio"])
    base_net_shares = base_stake / effective_cost
    base_payout = base_net_shares * payout
    base_pnl = base_payout - base_stake
    fill = dict(chosen)
    fill.update({
        "resolved_outcome": outcome,
        "cost": effective_cost,
        "real_reward": payout - effective_cost,
        "payout_per_net_share": payout,
        "base_stake": float(base_stake),
        "base_net_shares": base_net_shares,
        "base_payout": base_payout,
        "base_pnl": base_pnl,
        "base_roi": base_pnl / base_stake,
    })
    return chosen, fill


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
    info = api.dataset_info(OPEN_REPO, revision=OPEN_REV)
    if str(info.sha) != OPEN_REV:
        raise RuntimeError(f"OpenMarket revision did not resolve immutably: {info.sha}")
    files = api.list_repo_files(OPEN_REPO, repo_type="dataset", revision=OPEN_REV)

    meta, audit_df, resolutions = load_market_meta(args.cache / "obadiaha", start, end)
    audit_df.to_csv(args.out / "clock_audit.csv", index=False)
    print(f"Pinned OpenMarket {OPEN_REPO}@{OPEN_REV}; eligible meta={len(meta)}", flush=True)

    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn_anchor = BinanceAnchor.from_df(bn_df)
    inst = deribit_instruments()
    selected = select_deribit_instruments(inst, bn_df, start, end)
    (args.out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n")
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"Deribit selected={len(selected)} usable={len(der.ts)}", flush=True)

    clob = legacy.ClobStaticCache(args.cache / "clob_static.json")
    signals = []; fills = []; coverage = {}; market_counts = {}
    for d in daterange(start, end):
        pm, bnt, pp, bp = load_open_day(d, files, args.cache)
        coverage[d.isoformat()] = {"pm_parts": pp, "bn_parts": bp, "available": bool(pp and bp)}
        if pm.empty or bnt.empty:
            print(f"DAY {d} skipped: OpenMarket partition missing/empty", flush=True)
            continue
        tape = load_obadiaha_trades(d, args.cache)
        pm_markets = sorted(set(pm["market_slug"].unique()) & set(meta))
        market_counts[d.isoformat()] = len(pm_markets)
        by_bn = {m: g for m, g in bnt.groupby("market_slug", sort=False)}
        print(f"DAY {d} pm_rows={len(pm)} bn_rows={len(bnt)} tape_rows={len(tape)} markets={len(pm_markets)}", flush=True)
        day_s = day_f = 0
        for slug in pm_markets:
            pmg = pm[pm["market_slug"] == slug]
            bng = by_bn.get(slug)
            if bng is None or bng.empty:
                continue
            sig, fill = process_market(
                slug, pmg, bng, tape, meta[slug], resolutions.get(slug),
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
        "classification": EXECUTION_GRADE,
        "headline_period": [start.isoformat(), end.isoformat()],
        "sources": {
            "quote_truth": {"repo": OPEN_REPO, "revision": OPEN_REV, "split": "v0.4.3-unified"},
            "real_tape_and_resolution": {"repo": legacy.PM_REPO, "revision": legacy.PM_REV},
            "binance_rv": "official BTCUSDT 1m; only closed candles <= decision time",
            "spot_now": "OpenMarket synchronized Binance ms tick; causal previous tick <= decision time",
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
    (args.out / "EXECUTION_GRADE.txt").write_text(EXECUTION_GRADE + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
