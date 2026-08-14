from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from honest_backtest import Decision, Signal, evaluate, run_signal
from honest_backtest.adapters.parquet_pm import load_corpus

YEAR_SECONDS = 365.0 * 24 * 3600
DAY_MS = 86_400_000
DERIBIT_BASE = "https://history.deribit.com/api/v2/public"
BINANCE_BASE = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_json(url: str, params: dict, retries: int = 6) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 30))
                continue
            r.raise_for_status()
            obj = r.json()
            if "error" in obj:
                raise RuntimeError(obj["error"])
            return obj
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET failed {url} params={params}: {last}")


def download_binance_1m(start: date, end: date, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ]
    for d in daterange(start - timedelta(days=1), end):
        key = d.isoformat()
        csv_cache = cache_dir / f"BTCUSDT-1m-{key}.csv"
        if not csv_cache.exists():
            url = f"{BINANCE_BASE}/BTCUSDT-1m-{key}.zip"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
                if len(names) != 1:
                    raise RuntimeError(f"unexpected Binance archive {url}: {names}")
                csv_cache.write_bytes(zf.read(names[0]))
        f = pd.read_csv(csv_cache, header=None, names=cols)
        frames.append(f[["open_time", "open", "close", "close_time"]])
    out = pd.concat(frames, ignore_index=True)
    for c in ["open_time", "close_time"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ["open", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()
    # Binance Vision spot archives use microsecond timestamps in newer files;
    # normalize to milliseconds without guessing the archive vintage.
    for c in ["open_time", "close_time"]:
        vals = out[c].astype(np.int64)
        if int(vals.abs().median()) > 10**14:
            vals = vals // 1000
        out[c] = vals
    out = out.sort_values("open_time").drop_duplicates("open_time")
    return out.reset_index(drop=True)


@dataclass
class BinanceAnchor:
    open_times: np.ndarray
    opens: np.ndarray
    close_times: np.ndarray
    closes: np.ndarray
    log_returns: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        close_times = df["close_time"].to_numpy(np.int64)
        closes = df["close"].to_numpy(float)
        lr = np.full_like(closes, np.nan, dtype=float)
        lr[1:] = np.diff(np.log(closes))
        return cls(
            df["open_time"].to_numpy(np.int64),
            df["open"].to_numpy(float),
            close_times,
            closes,
            lr,
        )

    def open_price(self, open_ts_s: int) -> float:
        t = int(open_ts_s) * 1000
        i = int(np.searchsorted(self.open_times, t, side="right") - 1)
        if i < 0:
            return math.nan
        return float(self.opens[i])

    def rv_annualized(self, ts_ms: int, minutes: int = 60) -> float:
        # A close is usable only after its close_time; strict no-lookahead.
        hi = int(np.searchsorted(self.close_times, int(ts_ms), side="right"))
        lo_t = int(ts_ms) - minutes * 60_000
        lo = int(np.searchsorted(self.close_times, lo_t, side="left"))
        vals = self.log_returns[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size < max(20, minutes // 3):
            return math.nan
        sig_1m = float(np.std(vals, ddof=1))
        return sig_1m * math.sqrt(365 * 24 * 60)


def deribit_instruments() -> pd.DataFrame:
    rows = []
    for expired in ("true", "false"):
        obj = fetch_json(
            f"{DERIBIT_BASE}/get_instruments",
            {"currency": "BTC", "kind": "option", "expired": expired},
        )
        rows.extend(obj.get("result", []))
    if not rows:
        raise RuntimeError("Deribit returned no BTC option instruments")
    df = pd.DataFrame(rows).drop_duplicates("instrument_name")
    for c in ["expiration_timestamp", "creation_timestamp", "strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["instrument_name", "expiration_timestamp", "strike"])


def select_deribit_instruments(inst: pd.DataFrame, bn: pd.DataFrame,
                               start: date, end: date) -> list[str]:
    selected: set[str] = set()
    bn2 = bn.copy()
    bn2["day"] = pd.to_datetime(bn2["open_time"], unit="ms", utc=True).dt.date
    daily = bn2.groupby("day")["close"].median().to_dict()
    for d in daterange(start, end):
        spot = daily.get(d)
        if not spot or not math.isfinite(float(spot)):
            continue
        noon = int(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc).timestamp() * 1000)
        x = inst[(inst["expiration_timestamp"] > noon + 7 * DAY_MS)
                 & (inst["expiration_timestamp"] < noon + 65 * DAY_MS)].copy()
        if x.empty:
            continue
        expiries = np.array(sorted(x["expiration_timestamp"].unique()), dtype=float)
        chosen_exp = set()
        for target_days in (10, 30, 60):
            target = noon + target_days * DAY_MS
            chosen_exp.add(float(expiries[np.argmin(np.abs(expiries - target))]))
        x = x[x["expiration_timestamp"].isin(chosen_exp)].copy()
        x["mny"] = np.abs(np.log(x["strike"].astype(float) / float(spot)))
        for (_, opt_type), g in x.groupby(["expiration_timestamp", "option_type"], dropna=False):
            for name in g.nsmallest(4, "mny")["instrument_name"]:
                selected.add(str(name))
    return sorted(selected)


def fetch_deribit_trades(instruments: list[str], start: date, end: date,
                          cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_d = end + timedelta(days=1)
    end_ms = int(datetime(end_d.year, end_d.month, end_d.day, tzinfo=timezone.utc).timestamp() * 1000) - 1
    all_rows = []
    for idx, name in enumerate(instruments, 1):
        start_seq = None
        while True:
            params = {
                "instrument_name": name,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "count": 1000,
                "sorting": "asc",
            }
            if start_seq is not None:
                params["start_seq"] = start_seq
            obj = fetch_json(f"{DERIBIT_BASE}/get_last_trades_by_instrument", params)
            result = obj.get("result", {})
            rows = result.get("trades", [])
            if not rows:
                break
            all_rows.extend(rows)
            if not result.get("has_more"):
                break
            seqs = [r.get("trade_seq") for r in rows if r.get("trade_seq") is not None]
            if not seqs:
                break
            new_start = max(seqs) + 1
            if start_seq is not None and new_start <= start_seq:
                raise RuntimeError(f"Deribit pagination stalled for {name}")
            start_seq = new_start
        if idx % 20 == 0:
            print(f"Deribit instruments fetched: {idx}/{len(instruments)} rows={len(all_rows)}", flush=True)
    if not all_rows:
        raise RuntimeError("No Deribit option trades returned for selected instruments/date range")
    df = pd.DataFrame(all_rows)
    # Keep lit/simple trades; block/combo prints can be negotiated and are not a clean anchor.
    for c in ["block_trade_id", "block_rfq_id", "combo_id", "combo_trade_id"]:
        if c in df.columns:
            df = df[df[c].isna()]
    for c in ["timestamp", "iv", "index_price", "price", "trade_seq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "iv", "index_price", "instrument_name"])
    df = df[(df["iv"] > 0) & (df["index_price"] > 0)].copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


@dataclass
class DeribitAnchor:
    ts: np.ndarray
    iv: np.ndarray

    @classmethod
    def from_trades(cls, trades: pd.DataFrame, instruments: pd.DataFrame):
        meta = instruments[["instrument_name", "strike", "expiration_timestamp"]].drop_duplicates("instrument_name")
        t = trades.merge(meta, on="instrument_name", how="left")
        t["tte_days"] = (t["expiration_timestamp"] - t["timestamp"]) / DAY_MS
        t["abs_log_mny"] = np.abs(np.log(t["strike"] / t["index_price"]))
        t = t[(t["tte_days"] >= 7) & (t["tte_days"] <= 65)
              & (t["abs_log_mny"] <= 0.15) & (t["iv"] > 1) & (t["iv"] < 300)]
        t = t.sort_values("timestamp")
        return cls(t["timestamp"].to_numpy(np.int64), t["iv"].to_numpy(float) / 100.0)

    def median_iv(self, ts_ms: int, lookback_min: int = 30) -> float:
        hi = int(np.searchsorted(self.ts, int(ts_ms), side="right"))
        if hi <= 0:
            return math.nan
        lo = int(np.searchsorted(self.ts, int(ts_ms) - lookback_min * 60_000, side="left"))
        vals = self.iv[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return math.nan
        return float(np.median(vals))


def digital_prob_up(rel_spot: float, tau_s: int, sigma: float) -> float:
    if not (rel_spot > 0 and tau_s > 0 and sigma > 0 and math.isfinite(sigma)):
        return math.nan
    tau = tau_s / YEAR_SECONDS
    den = sigma * math.sqrt(tau)
    if den <= 0:
        return math.nan
    d2 = (math.log(rel_spot) - 0.5 * sigma * sigma * tau) / den
    return float(norm.cdf(d2))


class StructuralMispricing(Signal):
    family = "pm_structural_two_anchor"
    mode = "taker"
    coins = ("btc",)
    durations = ("15m",)
    once = True

    def __init__(self, bn: BinanceAnchor, der: DeribitAnchor, threshold: float,
                 min_s2c: int = 60, max_s2c: int = 600, size: float = 5.0):
        self.bn = bn
        self.der = der
        self.threshold = float(threshold)
        self.min_s2c = int(min_s2c)
        self.max_s2c = int(max_s2c)
        self.size = float(size)
        self.name = f"structural_{int(round(threshold*100)):02d}c"

    def decide(self, ctx, i):
        if ctx.meta.coin != "btc" or ctx.meta.duration != "15m":
            return None
        s2c = int(ctx.s2c[i])
        if s2c < self.min_s2c or s2c > self.max_s2c or not ctx.book_ok(i):
            return None
        ts = int(ctx.ts[i])
        rv = self.bn.rv_annualized(ts, 60)
        div = self.der.median_iv(ts, 30)
        if not (math.isfinite(rv) and math.isfinite(div)):
            return None
        if rv < 0.05 or rv > 3.0 or div < 0.05 or div > 3.0:
            return None
        open_bn = self.bn.open_price(ctx.meta.open_ts)
        spot_now = float(ctx.spot[i])
        if not (open_bn > 0 and spot_now > 0):
            return None
        # Relative spot move cancels the Binance-vs-Polymarket oracle level basis.
        rel = spot_now / open_bn
        p_rv = digital_prob_up(rel, s2c, rv)
        p_iv = digital_prob_up(rel, s2c, div)
        if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
            return None
        p_lo, p_hi = min(p_rv, p_iv), max(p_rv, p_iv)

        ya, na = float(ctx.ya[i]), float(ctx.na[i])
        yas, nas = float(ctx.yas[i]), float(ctx.nas[i])
        fr = float(ctx.meta.fee_rate or 0.0)
        fee_y = fr * ya * (1.0 - ya) if 0 < ya < 1 else math.inf
        fee_n = fr * na * (1.0 - na) if 0 < na < 1 else math.inf
        edge_y = p_lo - ya - fee_y
        edge_n = (1.0 - p_hi) - na - fee_n

        # Require enough displayed top-of-book size for the full 5-share order.
        buy_yes = edge_y >= self.threshold and yas >= self.size
        buy_no = edge_n >= self.threshold and nas >= self.size
        if not buy_yes and not buy_no:
            return None
        if buy_yes and (not buy_no or edge_y >= edge_n):
            yes, ask, edge = True, ya, edge_y
        else:
            yes, ask, edge = False, na, edge_n
        tag = json.dumps({
            "edge_signal": round(edge, 6), "p_rv": round(p_rv, 6),
            "p_deribit": round(p_iv, 6), "rv": round(rv, 6),
            "deribit_iv": round(div, 6), "rel_spot": round(rel, 8),
            "s2c": s2c,
        }, separators=(",", ":"))
        return Decision(i=i, ts_ms=ts, token_yes=yes, action="taker",
                        target_px=ask, size=self.size, tag=tag)


def split_ctxs(ctxs):
    closes = sorted(ctx.meta.close_ts for ctx in ctxs)
    cut = closes[len(closes)//2]
    a = [c for c in ctxs if c.meta.close_ts <= cut]
    b = [c for c in ctxs if c.meta.close_ts > cut]
    return a, b, cut


def save_records(records, path: Path):
    rows = []
    for r in records:
        x = dict(r)
        try:
            x.update(json.loads(x.get("tag", "{}")))
        except Exception:
            pass
        rows.append(x)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--start", default="2026-05-27")
    ap.add_argument("--end", default="2026-06-24")
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print("Loading Polymarket BTC 15m corpus...", flush=True)
    ctxs = list(load_corpus(str(args.pm_dir), coins=("btc",), durations=("15m",)))
    if not ctxs:
        raise RuntimeError("No BTC 15m Polymarket contexts loaded")
    print(f"PM slots: {len(ctxs)}", flush=True)

    print("Downloading official Binance BTCUSDT 1m...", flush=True)
    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)

    print("Discovering Deribit option instruments...", flush=True)
    inst = deribit_instruments()
    selected = select_deribit_instruments(inst, bn_df, start, end)
    print(f"Selected Deribit instruments: {len(selected)}", flush=True)
    (args.out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n")

    print("Fetching actual Deribit option trades...", flush=True)
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"Deribit usable anchor trades: {len(der.ts)}", flush=True)

    train, holdout, cut = split_ctxs(ctxs)
    thresholds = [0.02, 0.03, 0.05, 0.08, 0.10]
    results = {
        "method": "PM structural residual outside both BN-RV and backward Deribit trade-IV digital fair-value anchors",
        "no_lookahead": True,
        "execution": "honest-backtest v0.2.0; 1000ms book-persistence and 1500ms real-tape corroboration; 5 shares; decision ask as limit; historical slot fee_rate",
        "pm_slots": len(ctxs),
        "deribit_selected_instruments": len(selected),
        "deribit_usable_trades": len(der.ts),
        "split_close_ts": int(cut),
        "split_close_utc": datetime.fromtimestamp(cut, tz=timezone.utc).isoformat(),
        "thresholds": {},
    }
    for th in thresholds:
        sig = StructuralMispricing(bn, der, th)
        print(f"Running {sig.name}...", flush=True)
        all_row = evaluate(sig, ctxs, latency_ms=1000, tape_window_ms=1500)
        train_row = evaluate(sig, train, latency_ms=1000, tape_window_ms=1500)
        hold_row = evaluate(sig, holdout, latency_ms=1000, tape_window_ms=1500)
        recs = run_signal(sig, ctxs, latency_ms=1000, tape_window_ms=1500)
        save_records(recs, args.out / f"records_{sig.name}.csv")
        results["thresholds"][sig.name] = {
            "all": all_row, "first_half": train_row, "holdout_second_half": hold_row,
        }
        print(json.dumps({"name": sig.name, "all": all_row, "holdout": hold_row}, default=str), flush=True)

    (args.out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    # Compact markdown summary for auditability.
    lines = [
        "# BTC 15m structural-mispricing replay",
        "",
        f"PM slots: {len(ctxs)}; Deribit usable trades: {len(der.ts)}; split: {results['split_close_utc']}",
        "",
        "| threshold | all persist n | all edge | all fee-ROI | holdout persist n | holdout edge | holdout fee-ROI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, x in results["thresholds"].items():
        a, h = x["all"], x["holdout_second_half"]
        ap = a.get("honest_persist", {})
        hp = h.get("honest_persist", {})
        ar = a.get("roi_real", [None])[0] if a.get("roi_real") else None
        hr = h.get("roi_real", [None])[0] if h.get("roi_real") else None
        lines.append(f"| {name} | {ap.get('n')} | {ap.get('edge_real')} | {ar} | {hp.get('n')} | {hp.get('edge_real')} | {hr} |")
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
