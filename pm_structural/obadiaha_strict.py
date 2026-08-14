from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    digital_prob_up,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)

PM_REPO = "obadiaha/polymarket-crypto-5m-15m"
PM_REV = "11793901f0ac89c5a6c51123a6ccd29a3aaf8f4c"
CLOB_BASE = "https://clob.polymarket.com"
EXECUTION_GRADE = "STRICT_TAPE_COARSE_BOOK_10S_V1_ONCHAIN_FEE"
EPS = 1e-9


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def hf_file(rel: str, cache_dir: Path) -> Path:
    return Path(hf_hub_download(
        repo_id=PM_REPO,
        repo_type="dataset",
        revision=PM_REV,
        filename=rel,
        cache_dir=cache_dir,
    ))


def to_ms(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return (x.astype("int64") // 1_000_000).astype("Int64")


def spot_at(bn: BinanceAnchor, ts_ms: int) -> float:
    i = int(np.searchsorted(bn.close_times, int(ts_ms), side="right") - 1)
    return float(bn.closes[i]) if i >= 0 else math.nan


def parse_levels(obj: Any) -> list[dict]:
    if obj is None or (isinstance(obj, float) and math.isnan(obj)):
        return []
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return []
    if not isinstance(obj, list):
        return []
    out = []
    for x in obj:
        try:
            p = float(x.get("price")); q = float(x.get("size"))
        except Exception:
            continue
        if math.isfinite(p) and math.isfinite(q) and 0 < p < 1 and q > 0:
            out.append({"price": p, "size": q})
    return out


def ask_depth_to_limit(levels: Any, limit: float) -> float:
    return float(sum(x["size"] for x in parse_levels(levels) if x["price"] <= limit + 1e-12))


def v1_buy_fee(price: float, taker_base_fee_bps: int) -> tuple[float, float]:
    """Return (fee_tokens_per_gross_share, USDC-equivalent fee per gross share).

    This mirrors the archived V1 CTF Exchange CalculatorHelper for BUY orders:
      fee_tokens = bps/10000 * min(p,1-p)/p * gross_outcome_tokens
    The USD-equivalent fee per gross share is therefore bps/10000 * min(p,1-p).
    """
    if not (0 < price < 1 and taker_base_fee_bps >= 0):
        return math.nan, math.nan
    rate = float(taker_base_fee_bps) / 10_000.0
    fee_token_fraction = rate * min(price, 1.0 - price) / price
    fee_usdc_per_gross_share = fee_token_fraction * price
    return float(fee_token_fraction), float(fee_usdc_per_gross_share)


@dataclass
class MarketStatic:
    condition_id: str
    token_outcome: dict[str, str]
    maker_base_fee_bps: int
    taker_base_fee_bps: int
    fd_rate: float | None
    fd_exponent: float | None
    fd_taker_only: bool | None


class ClobStaticCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.data: dict[str, dict] = {}
        if cache_path.exists():
            try:
                self.data = json.loads(cache_path.read_text())
            except Exception:
                self.data = {}
        self.calls = 0

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.data, indent=2, sort_keys=True))

    def get(self, condition_id: str) -> MarketStatic | None:
        if condition_id not in self.data:
            obj = None
            for attempt in range(7):
                try:
                    r = requests.get(f"{CLOB_BASE}/clob-markets/{condition_id}", timeout=30)
                    if r.status_code == 429:
                        time.sleep(min(2 ** attempt, 20)); continue
                    r.raise_for_status(); obj = r.json(); break
                except Exception:
                    if attempt == 6:
                        obj = None
                    else:
                        time.sleep(min(2 ** attempt, 20))
            self.calls += 1
            if obj is None:
                self.data[condition_id] = {"status": "error"}
            else:
                tokens = {str(x.get("t")): str(x.get("o")) for x in obj.get("t", []) if x.get("t") is not None}
                fd = obj.get("fd") or {}
                self.data[condition_id] = {
                    "status": "ok",
                    "tokens": tokens,
                    "maker_base_fee_bps": obj.get("mbf"),
                    "taker_base_fee_bps": obj.get("tbf"),
                    "fee_descriptor": {"r": fd.get("r"), "e": fd.get("e"), "to": fd.get("to")},
                    "activated_at": obj.get("aot"),
                }
                if self.calls % 25 == 0:
                    self.save()
        row = self.data.get(condition_id, {})
        if row.get("status") != "ok":
            return None
        toks = row.get("tokens") or {}
        if not toks or not any(v.lower() == "up" for v in toks.values()) or not any(v.lower() == "down" for v in toks.values()):
            return None
        try:
            mbf = int(row.get("maker_base_fee_bps") or 0)
            tbf = int(row.get("taker_base_fee_bps") or 0)
        except Exception:
            return None
        fd = row.get("fee_descriptor") or {}
        try:
            fdr = float(fd["r"]) if fd.get("r") is not None else None
            fde = float(fd["e"]) if fd.get("e") is not None else None
            fdto = bool(fd["to"]) if fd.get("to") is not None else None
        except Exception:
            fdr = fde = fdto = None
        return MarketStatic(condition_id, toks, mbf, tbf, fdr, fde, fdto)


def conservative_fair(side: str, p_rv: float, p_iv: float) -> float:
    p_lo, p_hi = min(p_rv, p_iv), max(p_rv, p_iv)
    return p_lo if side == "Up" else 1.0 - p_hi


def settlement_map(resolutions: pd.DataFrame) -> dict[str, str]:
    x = resolutions.copy(); x["market_id"] = x["market_id"].astype(str); x["outcome"] = x["outcome"].astype(str)
    return dict(zip(x["market_id"], x["outcome"]))


def cluster_bootstrap_roi(fills: pd.DataFrame, n: int = 10000, seed: int = 20260815) -> list[float] | None:
    if fills.empty:
        return None
    x = fills.copy(); x["day"] = pd.to_datetime(x["ts_ms"], unit="ms", utc=True).dt.date.astype(str)
    agg = x.groupby("day", as_index=False).agg(pnl=("base_pnl", "sum"), stake=("base_stake", "sum"))
    if len(agg) < 2:
        return None
    pnl = agg["pnl"].to_numpy(float); stake = agg["stake"].to_numpy(float)
    rng = np.random.default_rng(seed); vals = np.empty(n, dtype=float)
    for i in range(n):
        j = rng.integers(0, len(agg), size=len(agg)); vals[i] = pnl[j].sum() / stake[j].sum()
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def grade_execution(token_rows: pd.DataFrame, trade_rows: pd.DataFrame, decision_ts: int,
                    target_px: float, decision_levels: Any, latency_ms: int, tape_window_ms: int) -> dict:
    decision_depth = ask_depth_to_limit(decision_levels, target_px)
    later = token_rows[(token_rows["ts_ms"] >= decision_ts + latency_ms)
                       & (token_rows["ts_ms"] <= decision_ts + tape_window_ms)].sort_values("ts_ms")
    if later.empty:
        return {"filled": False, "reason": "no_persist_snapshot", "decision_depth_shares": decision_depth}
    p = later.iloc[0]; persist_ask = float(p.best_ask)
    persist_depth = ask_depth_to_limit(p.ask_levels, target_px) if persist_ask <= target_px + EPS else 0.0
    if persist_ask > target_px + EPS or persist_depth <= 0:
        return {"filled": False, "reason": "book_not_persistent", "decision_depth_shares": decision_depth,
                "persist_ts_ms": int(p.ts_ms), "persist_ask": persist_ask, "persist_depth_shares": persist_depth}
    tape = trade_rows[(trade_rows["ts_ms"] >= decision_ts + latency_ms)
                      & (trade_rows["ts_ms"] <= decision_ts + tape_window_ms)
                      & (trade_rows["side"].astype(str).str.upper() == "BUY")
                      & (trade_rows["price"] <= target_px + EPS)
                      & (trade_rows["size"] > 0)].copy()
    if tape.empty:
        return {"filled": False, "reason": "no_buy_tape_corroboration", "decision_depth_shares": decision_depth,
                "persist_ts_ms": int(p.ts_ms), "persist_ask": persist_ask, "persist_depth_shares": persist_depth}
    witness = tape.sort_values(["size", "ts_ms"], ascending=[False, True]).iloc[0]
    tape_size = float(witness["size"]); capacity_shares = max(min(decision_depth, persist_depth, tape_size), 0.0)
    return {"filled": capacity_shares > 0, "reason": "strict_witness" if capacity_shares > 0 else "zero_capacity",
            "decision_depth_shares": decision_depth, "persist_ts_ms": int(p.ts_ms), "persist_ask": persist_ask,
            "persist_depth_shares": persist_depth, "tape_ts_ms": int(witness.ts_ms),
            "tape_price": float(witness.price), "tape_size_shares": tape_size,
            "tape_tx_hash": str(witness.tx_hash), "capacity_shares": capacity_shares,
            "capacity_usd": capacity_shares * target_px}


def process_day(d: date, market_meta: dict[str, dict], resolutions: dict[str, str], bn: BinanceAnchor,
                der: DeribitAnchor, clob: ClobStaticCache, threshold: float, latency_ms: int,
                tape_window_ms: int, base_stake: float, hf_cache: Path) -> tuple[list[dict], list[dict]]:
    key = d.isoformat()
    book_path = hf_file(f"orderbooks/{key}.parquet", hf_cache)
    trade_path = hf_file(f"trades/{key}.parquet", hf_cache)
    bcols = ["timestamp", "asset", "market_id", "condition_id", "token_id", "best_ask", "ask_levels"]
    tcols = ["timestamp", "asset", "market_id", "condition_id", "token_id", "side", "price", "size", "tx_hash"]
    book = pd.read_parquet(book_path, columns=bcols); trades = pd.read_parquet(trade_path, columns=tcols)
    book = book[(book["asset"] == "BTC") & book["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    trades = trades[(trades["asset"] == "BTC") & trades["market_id"].astype(str).str.startswith("btc-updown-15m-")].copy()
    if book.empty:
        return [], []
    for df in (book, trades):
        df["market_id"] = df["market_id"].astype(str); df["token_id"] = df["token_id"].astype(str)
        df["ts_ms"] = to_ms(df["timestamp"])
    book["best_ask"] = pd.to_numeric(book["best_ask"], errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce"); trades["size"] = pd.to_numeric(trades["size"], errors="coerce")
    book = book.dropna(subset=["ts_ms", "best_ask"]).copy(); trades = trades.dropna(subset=["ts_ms", "price", "size"]).copy()
    book["ts_ms"] = book["ts_ms"].astype(np.int64); trades["ts_ms"] = trades["ts_ms"].astype(np.int64)
    book = book.sort_values(["market_id", "ts_ms", "token_id"]).drop_duplicates(["market_id", "ts_ms", "token_id"], keep="last")

    signals: list[dict] = []; fills: list[dict] = []
    by_trade_market = {m: g for m, g in trades.groupby("market_id", sort=False)}
    for market_id, g in book.groupby("market_id", sort=True):
        meta = market_meta.get(market_id); outcome = resolutions.get(market_id)
        if not meta or outcome not in ("Up", "Down"):
            continue
        open_ts_s = int(meta["open_ts_s"]); close_ts_ms = int(meta["close_ts_ms"]); condition_id = str(meta["condition_id"])
        open_bn = bn.open_price(open_ts_s)
        if not (open_bn > 0 and math.isfinite(open_bn)):
            continue
        g = g.sort_values(["ts_ms", "token_id"]); chosen = None
        for ts_ms, snap in g.groupby("ts_ms", sort=True):
            if len(snap) != 2:
                continue
            s2c = int(round((close_ts_ms - int(ts_ms)) / 1000.0))
            if s2c < 60 or s2c > 600:
                continue
            asks = pd.to_numeric(snap["best_ask"], errors="coerce").to_numpy(float)
            if len(asks) != 2 or not np.all(np.isfinite(asks)) or np.any(asks <= 0) or np.any(asks >= 1):
                continue
            rv = bn.rv_annualized(int(ts_ms), 60); div = der.median_iv(int(ts_ms), 30)
            if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
                continue
            spot = spot_at(bn, int(ts_ms))
            if not (spot > 0 and math.isfinite(spot)):
                continue
            rel = spot / open_bn; p_rv = digital_prob_up(rel, s2c, rv); p_iv = digital_prob_up(rel, s2c, div)
            if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
                continue
            fairs = [min(p_rv, p_iv), 1.0 - max(p_rv, p_iv)]
            if max(f - a for f in fairs for a in asks) < threshold:
                continue
            static = clob.get(condition_id)
            if static is None or static.taker_base_fee_bps <= 0:
                continue
            row_by_outcome = {}
            for _, r in snap.iterrows():
                label = static.token_outcome.get(str(r.token_id))
                if label in ("Up", "Down"):
                    row_by_outcome[label] = r
            if set(row_by_outcome) != {"Up", "Down"}:
                continue
            candidates = []
            for side in ("Up", "Down"):
                r = row_by_outcome[side]; ask = float(r.best_ask); fair = conservative_fair(side, p_rv, p_iv)
                fee_token_fraction, fee_usdc_share = v1_buy_fee(ask, static.taker_base_fee_bps)
                net_ratio = 1.0 - fee_token_fraction
                if not (math.isfinite(net_ratio) and net_ratio > 0):
                    continue
                # Preserve the frozen 3c meaning: probability-value edge after the actual USDC-equivalent entry fee.
                edge_equiv = fair - ask - fee_usdc_share
                # Exact settlement EV per gross share after the token fee is deducted.
                edge_exact = fair * net_ratio - ask
                candidates.append((edge_equiv, edge_exact, side, r, ask, fair, fee_token_fraction, fee_usdc_share, net_ratio))
            eligible = [x for x in candidates if x[0] >= threshold]
            if not eligible:
                continue
            edge_equiv, edge_exact, side, r, ask, fair, fee_token_fraction, fee_usdc_share, net_ratio = max(eligible, key=lambda x: (x[0], x[1]))
            chosen = {
                "ts_ms": int(ts_ms), "close_ts_ms": close_ts_ms, "cid": market_id, "condition_id": condition_id,
                "token_id": str(r.token_id), "action": f"buy_{side.lower()}", "outcome_token": side,
                "resolved_outcome": outcome, "target_px": ask,
                "v1_maker_base_fee_bps": static.maker_base_fee_bps,
                "v1_taker_base_fee_bps": static.taker_base_fee_bps,
                "fee_tokens_per_gross_share": fee_token_fraction,
                "fee_usdc_equiv_per_gross_share": fee_usdc_share,
                "net_share_ratio": net_ratio,
                "fd_rate_metadata_only": static.fd_rate,
                "fd_exponent_metadata_only": static.fd_exponent,
                "fd_taker_only_metadata_only": static.fd_taker_only,
                "p_rv": p_rv, "p_deribit": p_iv, "fair_conservative": fair,
                "edge_equiv": edge_equiv, "edge_exact_net_ev": edge_exact,
                "rv": rv, "deribit_iv": div, "spot": spot, "open_spot": open_bn,
                "rel_spot": rel, "s2c": s2c, "decision_ask_levels": r.ask_levels,
                "execution_grade": EXECUTION_GRADE,
            }
            break
        if chosen is None:
            continue
        token_rows = g[g["token_id"] == chosen["token_id"]][["ts_ms", "best_ask", "ask_levels"]].copy()
        tg = by_trade_market.get(market_id, trades.iloc[0:0])
        tg = tg[tg["token_id"] == chosen["token_id"]][["ts_ms", "side", "price", "size", "tx_hash"]].copy()
        grade = grade_execution(token_rows, tg, chosen["ts_ms"], chosen["target_px"], chosen["decision_ask_levels"], latency_ms, tape_window_ms)
        chosen.update(grade); chosen.pop("decision_ask_levels", None); signals.append(dict(chosen))
        if not grade.get("filled") or float(grade.get("capacity_usd", 0.0)) + EPS < base_stake:
            continue
        net_ratio = float(chosen["net_share_ratio"]); effective_cost = float(chosen["target_px"]) / net_ratio
        payout = 1.0 if chosen["outcome_token"] == chosen["resolved_outcome"] else 0.0
        base_net_shares = base_stake / effective_cost; base_payout = base_net_shares * payout; base_pnl = base_payout - base_stake
        f = dict(chosen); f.update({"cost": effective_cost, "real_reward": payout - effective_cost,
            "payout_per_net_share": payout, "base_stake": base_stake, "base_net_shares": base_net_shares,
            "base_payout": base_payout, "base_pnl": base_pnl, "base_roi": base_pnl / base_stake})
        fills.append(f)
    return signals, fills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-03-02"); ap.add_argument("--end", default="2026-03-18")
    ap.add_argument("--threshold", type=float, default=0.03); ap.add_argument("--latency-ms", type=int, default=1000)
    ap.add_argument("--tape-window-ms", type=int, default=15000); ap.add_argument("--base-stake", type=float, default=5.0)
    args = ap.parse_args(); start = date.fromisoformat(args.start); end = date.fromisoformat(args.end)
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True); hf_cache = args.cache / "hf"

    print(f"Pinned PM source {PM_REPO}@{PM_REV}", flush=True)
    markets = pd.read_parquet(hf_file("markets/all.parquet", hf_cache)); resolutions_df = pd.read_parquet(hf_file("resolutions/all.parquet", hf_cache))
    markets["market_id"] = markets["market_id"].astype(str); markets["start_dt"] = pd.to_datetime(markets["start_time"], utc=True, errors="coerce")
    markets["end_dt"] = pd.to_datetime(markets["end_time"], utc=True, errors="coerce")
    m = markets[(markets["asset"] == "BTC") & (markets["market_type"] == "crypto_15m")].copy()
    lo = pd.Timestamp(start, tz="UTC"); hi = pd.Timestamp(end + timedelta(days=1), tz="UTC"); m = m[(m["start_dt"] >= lo) & (m["start_dt"] < hi)].copy()
    market_meta = {str(r.market_id): {"condition_id": str(r.condition_id), "open_ts_s": int(r.start_dt.timestamp()),
        "close_ts_ms": int(r.end_dt.timestamp() * 1000)} for r in m.itertuples() if pd.notna(r.start_dt) and pd.notna(r.end_dt)}
    resolutions = settlement_map(resolutions_df); print(f"Eligible BTC15m markets in metadata: {len(market_meta)}", flush=True)

    print("Downloading official Binance BTCUSDT 1m...", flush=True); bn_df = download_binance_1m(start, end, args.cache / "binance"); bn = BinanceAnchor.from_df(bn_df)
    print("Discovering Deribit option instruments...", flush=True); inst = deribit_instruments(); selected = select_deribit_instruments(inst, bn_df, start, end)
    (args.out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n"); print(f"Selected Deribit instruments: {len(selected)}", flush=True)
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet"); der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"Deribit usable anchor trades: {len(der.ts)}", flush=True)

    clob = ClobStaticCache(args.cache / "clob_static.json"); signals: list[dict] = []; fills: list[dict] = []
    for d in daterange(start, end):
        print(f"DAY {d.isoformat()} start", flush=True)
        s, f = process_day(d, market_meta, resolutions, bn, der, clob, args.threshold, args.latency_ms, args.tape_window_ms, args.base_stake, hf_cache)
        signals.extend(s); fills.extend(f); clob.save(); print(f"DAY {d.isoformat()} signals={len(s)} strict_base_fills={len(f)} cumulative={len(fills)}", flush=True)
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
    total_stake = float(fill_df["base_stake"].sum()) if len(fill_df) else 0.0; total_pnl = float(fill_df["base_pnl"].sum()) if len(fill_df) else 0.0
    summary = {
        "classification": EXECUTION_GRADE, "headline_period": [start.isoformat(), end.isoformat()],
        "source": {"repo_id": PM_REPO, "revision": PM_REV},
        "policy": {"family": "frozen structural two-anchor fade", "threshold": args.threshold,
            "decision_window_s2c": [60, 600], "signal_edge": "conservative fair minus ask minus V1 on-chain USDC-equivalent taker fee",
            "once_per_market": True, "no_outcome_in_decision": True},
        "execution": {"latency_ms": args.latency_ms, "tape_window_ms": args.tape_window_ms,
            "required": "post-latency book persistence plus same-token BUY tape at or below limit",
            "capacity_shares": "min(decision ask depth, post-latency ask depth, largest single qualifying BUY tape trade)",
            "base_fill_requires_usd": args.base_stake, "partial_fills": False},
        "historical_fee": {"v1_taker_base_fee_bps_seen": base_fee_values,
            "formula": "BUY fee_tokens/gross_share = (tbf/10000)*min(p,1-p)/p; USDC-equivalent fee/gross_share = (tbf/10000)*min(p,1-p)",
            "source": "archived V1 CTF Exchange CalculatorHelper semantics + per-market CLOB tbf",
            "empirical_validation": "Independent mapping audit cross-matched Data API taker trades to raw V1 OrderFilled txs; 20/20 inspected trades infer approximately 1000 bps to integer-rounding precision on the audited 2026-03-14 BTC15m window.",
            "fd_metadata_seen_but_not_used_for_v1_pnl": fd_values,
            "reason_fd_not_used": "The audited V1 raw settlement fees match tbf=1000 plus the archived on-chain CalculatorHelper exactly; fd metadata/documented curve does not reproduce those historical OrderFilled fee amounts."},
        "markets_metadata": len(market_meta), "mapping_api_calls": clob.calls, "signals": int(len(sig_df)), "strict_base_fills": int(len(fill_df)),
        "fill_rate_given_signal": float(len(fill_df) / len(sig_df)) if len(sig_df) else None,
        "wins": int((fill_df["payout_per_net_share"] > 0.5).sum()) if len(fill_df) else 0,
        "losses": int((fill_df["payout_per_net_share"] < 0.5).sum()) if len(fill_df) else 0,
        "base_total_stake": total_stake, "base_total_pnl": total_pnl,
        "base_fee_adjusted_roi": total_pnl / total_stake if total_stake else None,
        "base_day_cluster_95ci_roi": cluster_bootstrap_roi(fill_df),
        "mean_edge_equiv": float(fill_df["edge_equiv"].mean()) if len(fill_df) else None,
        "mean_edge_exact_net_ev": float(fill_df["edge_exact_net_ev"].mean()) if len(fill_df) else None,
        "median_capacity_usd": float(fill_df["capacity_usd"].median()) if len(fill_df) else None,
        "min_capacity_usd": float(fill_df["capacity_usd"].min()) if len(fill_df) else None,
        "deribit_selected_instruments": len(selected), "deribit_usable_trades": len(der.ts),
        "caveat": "Order books are about 10-second snapshots. Real tape is required, but execution is coarse-book rather than millisecond-exact. Historical V1 fee settlement is intentionally modeled from on-chain mechanics rather than later documentation."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str)); (args.out / "EXECUTION_GRADE.txt").write_text(EXECUTION_GRADE + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
