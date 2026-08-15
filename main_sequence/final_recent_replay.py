from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from main_sequence import bankroll as bankroll_base
from main_sequence.original_93d_anchor_freeze import fetch_deribit_trades_parallel
from main_sequence import original_93d_tape_replay as stats_base
from pm_structural import recalc as original

# ---------------------------------------------------------------------------
# FINAL PRE-PRODUCTION REPLAY. Freeze policy before opening the recent tape.
# ---------------------------------------------------------------------------
CUTOFF = "2026-08-15"  # last complete UTC day; end-exclusive
WINDOWS = {
    "3m": ["2026-05-15", CUTOFF],
    "6m": ["2026-02-15", CUTOFF],
    "9m": ["2025-11-15", CUTOFF],
    "12m": ["2025-08-15", CUTOFF],
}
THRESHOLD = 0.03
SIZE = 5.0
MIN_S2C = 60
MAX_S2C = 600
LARGE_RAW_GAP = 0.10
EXIT_BAND = 0.01
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_1S = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s"
BINANCE_REST = "https://data-api.binance.vision/api/v3/klines"
LEGACY_FEE_START = int(pd.Timestamp("2026-01-05", tz="UTC").timestamp())
FEE_V2_START = int(pd.Timestamp("2026-03-30", tz="UTC").timestamp())

PROTOCOL = {
    "name": "Main Sequence FINAL recent causal dual-exit replay / 2026-08-15 freeze",
    "cutoff_exclusive": CUTOFF,
    "windows": WINDOWS,
    "asset": "BTC",
    "timeframe": "15m",
    "entry": {
        "net_edge_floor": THRESHOLD,
        "reference_size_shares": SIZE,
        "decision_window_s2c": [MIN_S2C, MAX_S2C],
        "once_per_market": True,
        "fair": "conservative boundary from Binance BTCUSDT 60m realized vol and causal backward 30m median Deribit option trade IV",
        "quote_witness": "same-second public Polymarket taker BUY tape must contain >=5 shares at or below the admitted limit",
    },
    "exit": {
        "large_if_initial_raw_gap_gte": LARGE_RAW_GAP,
        "large": "recompute fair causally at subsequent SELL-witness seconds and exit at first sellable price within 1c of fair; fallback settlement",
        "small": "binary settlement",
        "fair_band": EXIT_BAND,
    },
    "execution_layers": {
        "witness": "decision-second 5-share public BUY witness; large exit uses first same-second >=5-share public SELL witness satisfying fair band",
        "strict5": "decision witness is not fill; entry needs fresh +1..+5s full-size BUY; large exit needs a qualifying SELL witness plus fresh +1..+5s full-size SELL",
    },
    "historical_fees": {
        "policy": "read feesEnabled/feeType/feeSchedule from each historical Gamma market; only use dated fallback when metadata is absent",
        "pre_2026_01_05": "fee-free fallback for BTC15m",
        "2026_01_05_to_fee_v2": "legacy 15m crypto fee-curve fallback: rate=.25 exponent=2 fee-equivalent curve",
        "fee_v2_from_2026_03_30": "crypto fee schedule v2 fallback: rate=.07 exponent=1",
    },
    "anti_lookahead": [
        "entry signal consumes only Polymarket trades at the decision second and anchor observations timestamped <= decision",
        "strict entry begins decision+1s; no decision witness is reused",
        "final market outcome is used only for settlement payout/fallback, never for entry side/fair/threshold",
        "large convergence scans subsequent timestamps strictly after entry and recomputes fair using only anchor data available at each exit timestamp",
        "no threshold search, month deletion, asset deletion, or per-window policy changes",
    ],
    "bankroll_headline": {"initial": 50.0, "base_trade": 5.0, "single_trade_cap": 100.0, "market_cap": 200.0, "leverage": False},
}


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def get_json(sess: requests.Session, url: str, *, params=None, timeout=60, tries=7):
    last = None
    for i in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:160]}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(min(0.5 * (2 ** i), 12.0) + np.random.default_rng(i + 31).random() * 0.1)
    raise RuntimeError(f"GET failed {url}: {last!r}")


def jsonish(v, default):
    if isinstance(v, type(default)):
        return v
    if isinstance(v, str):
        try:
            z = json.loads(v)
            return z if isinstance(z, type(default)) else default
        except Exception:
            return default
    return default


def boolish(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return bool(v)
    if isinstance(v, str):
        if v.strip().lower() in ("true", "1", "yes"):
            return True
        if v.strip().lower() in ("false", "0", "no"):
            return False
    return None


@dataclass(frozen=True)
class Market:
    slug: str
    start: int
    condition_id: str
    label_up: float
    fee_enabled: bool
    fee_type: str
    fee_rate: float | None
    fee_exponent: float | None
    fee_source: str

    @property
    def close(self) -> int:
        return self.start + 900


def parse_market(raw: dict, start: int) -> Market | None:
    try:
        outcomes = jsonish(raw["outcomes"], [])
        prices = jsonish(raw.get("outcomePrices") or "[]", [])
        oi = {str(x).strip().lower(): i for i, x in enumerate(outcomes)}
        if "up" not in oi or "down" not in oi:
            return None
        ui, di = oi["up"], oi["down"]
        if len(prices) <= max(ui, di):
            return None
        pp = [float(x) for x in prices]
        if max(pp) < 0.99:
            return None
        cid = str(raw.get("conditionId") or "")
        if not cid:
            return None
        label = 1.0 if pp[ui] > pp[di] else 0.0
        explicit_enabled = boolish(raw.get("feesEnabled"))
        fs = jsonish(raw.get("feeSchedule"), {})
        rate = fs.get("rate")
        expo = fs.get("exponent")
        try:
            rate = float(rate) if rate is not None else None
        except Exception:
            rate = None
        try:
            expo = float(expo) if expo is not None else None
        except Exception:
            expo = None
        if explicit_enabled is not None:
            enabled = explicit_enabled
            source = "gamma_feesEnabled"
        elif start < LEGACY_FEE_START:
            enabled = False
            source = "dated_fallback_prefee"
        else:
            enabled = True
            source = "dated_fallback_enabled"
        return Market(
            slug=str(raw["slug"]), start=int(start), condition_id=cid, label_up=label,
            fee_enabled=enabled, fee_type=str(raw.get("feeType") or ""),
            fee_rate=rate, fee_exponent=expo, fee_source=source,
        )
    except Exception:
        return None


def fee_total(m: Market, p: float, qty: float) -> float:
    """USDC fee-equivalent for the requested historical taker quantity.

    Gamma's per-market fee schedule is authoritative when present. The old 15m
    crypto program (Jan-Mar 2026) and Fee Structure V2 used different curves;
    preserve that distinction instead of projecting today's 0.07 curve backward.
    """
    if not m.fee_enabled or qty <= 0:
        return 0.0
    p = min(max(float(p), 0.0), 1.0)
    rate = m.fee_rate
    expo = m.fee_exponent
    ft = m.fee_type.lower()
    is_v2 = "v2" in ft or (expo is not None and expo <= 1.000001 and m.start >= FEE_V2_START)
    if is_v2:
        r = 0.07 if rate is None else float(rate)
        raw = float(qty) * r * p * (1.0 - p)
    elif rate is not None and expo is not None:
        # Legacy Jan-2026 crypto maker-rebate curve. Fee-equivalent in USDC.
        raw = float(qty) * p * float(rate) * (p * (1.0 - p)) ** float(expo)
    elif m.start < FEE_V2_START:
        raw = float(qty) * p * 0.25 * (p * (1.0 - p)) ** 2.0
    else:
        raw = float(qty) * 0.07 * p * (1.0 - p)
    # Current protocol rounds USDC fees to 5 decimals. This is immaterial at our
    # scale but keeps replay accounting deterministic.
    out = round(max(raw, 0.0), 5)
    return 0.0 if out < 0.00001 else out


def fee_per_share_for_signal(m: Market, p: float) -> float:
    return fee_total(m, p, SIZE) / SIZE


def fetch_markets_hour(hour: int) -> tuple[list[Market], list[dict]]:
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-final-recent-gamma/1.0"})
    wanted = [(f"btc-updown-15m-{t0}", t0) for t0 in range(hour, hour + 3600, 900)]
    params = [("slug", slug) for slug, _ in wanted] + [("closed", "true"), ("limit", 20)]
    js = get_json(sess, GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js if isinstance(x, dict)}
    markets, inventory = [], []
    for slug, t0 in wanted:
        raw = byslug.get(slug)
        m = parse_market(raw or {}, t0)
        inventory.append({
            "slug": slug, "start": int(t0), "exists_in_gamma": raw is not None,
            "mapped": m is not None,
            "condition_id": (m.condition_id if m else None),
            "fee_enabled": (m.fee_enabled if m else None),
            "fee_type": (m.fee_type if m else None),
            "fee_rate": (m.fee_rate if m else None),
            "fee_exponent": (m.fee_exponent if m else None),
            "fee_source": (m.fee_source if m else None),
        })
        if m is not None:
            markets.append(m)
    return markets, inventory


def discover(start: str, end: str, workers: int = 12):
    hours = list(range(ts(start), ts(end), 3600))
    by_hour: dict[int, list[Market]] = {}
    inv: list[dict] = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fetch_markets_hour, h): h for h in hours}
        for k, f in enumerate(as_completed(fut), 1):
            h = fut[f]
            try:
                mm, ii = f.result(); by_hour[h] = mm; inv.extend(ii)
            except Exception as exc:
                failures.append({"hour": h, "error": repr(exc)})
            if k % 72 == 0:
                print("DISCOVER", start, end, k, "/", len(hours), "mapped", sum(len(x) for x in by_hour.values()), "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"Gamma discovery transport failures: {failures[:10]} count={len(failures)}")
    return hours, by_hour, pd.DataFrame(inv).sort_values("start", kind="mergesort")


def query_trade_rows(sess: requests.Session, markets: list[Market], start: int, end: int) -> list[dict]:
    """Complete Data-API retrieval under the documented 500-row page cap.

    Split by markets, then time, then offsets only for a single-market single-second
    overflow. This avoids silently truncating busy recent BTC15m hours.
    """
    if not markets or end < start:
        return []
    q = {"market": ",".join(m.condition_id for m in markets), "start": int(start), "end": int(end),
         "limit": 500, "offset": 0, "takerOnly": "true"}
    rows = get_json(sess, DATA_API + "/trades", params=q, timeout=60)
    if len(rows) < 500:
        return rows
    if len(markets) > 1:
        mid = len(markets) // 2
        return query_trade_rows(sess, markets[:mid], start, end) + query_trade_rows(sess, markets[mid:], start, end)
    if start < end:
        mid_t = (start + end) // 2
        return query_trade_rows(sess, markets, start, mid_t) + query_trade_rows(sess, markets, mid_t + 1, end)
    # One market, one second: use all legal offset pages; fail rather than truncate.
    out = list(rows)
    for off in (500, 1000):
        q2 = dict(q); q2["offset"] = off
        rr = get_json(sess, DATA_API + "/trades", params=q2, timeout=60)
        out.extend(rr)
        if len(rr) < 500:
            return out
    raise RuntimeError(f"Data API >1500 trades in one market-second {markets[0].slug} {start}; refusing truncation")


def normalize_trades(rows: list[dict]) -> dict[str, pd.DataFrame]:
    return stats_base.normalize_trades(rows)


def load_binance_1s(start: str, end: str, cache: Path) -> stats_base.BinanceSecond:
    cache.mkdir(parents=True, exist_ok=True)
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-final-recent-binance/1.0"})
    frames = []
    for d in pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D"):
        key = d.strftime("%Y-%m-%d"); cp = cache / f"BTCUSDT-1s-{key}.parquet"
        if cp.exists():
            frames.append(pd.read_parquet(cp)); continue
        url = f"{BINANCE_1S}/BTCUSDT-1s-{key}.zip"
        r = sess.get(url, timeout=120)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names = [n for n in z.namelist() if n.endswith(".csv")]
                if len(names) != 1:
                    raise RuntimeError(f"unexpected Binance 1s archive {url}: {names}")
                df = pd.read_csv(z.open(names[0]), header=None, usecols=[0, 4, 6])
            df.columns = ["open_time", "close", "close_time"]
        else:
            # Last complete UTC day may not yet be in data.binance.vision. Use
            # the public market-data REST endpoint causally, paging 1000 1s bars.
            day0 = int(pd.Timestamp(key, tz="UTC").timestamp() * 1000)
            day1 = day0 + 86_400_000 - 1
            chunks = []
            cur = day0
            while cur <= day1:
                js = get_json(sess, BINANCE_REST, params={"symbol": "BTCUSDT", "interval": "1s", "startTime": cur, "endTime": day1, "limit": 1000})
                if not js:
                    break
                z = pd.DataFrame(js)
                chunks.append(z.iloc[:, [0, 4, 6]])
                nxt = int(z.iloc[-1, 6]) + 1
                if nxt <= cur:
                    raise RuntimeError("Binance REST 1s pagination stalled")
                cur = nxt
                if len(js) < 1000:
                    break
            if not chunks:
                raise RuntimeError(f"no Binance 1s data for {key}; archive status={r.status_code}")
            df = pd.concat(chunks, ignore_index=True); df.columns = ["open_time", "close", "close_time"]
        for c in ["open_time", "close_time"]:
            v = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            if len(v.dropna()) and int(v.dropna().abs().median()) > 10**14:
                v = v // 1000
            df[c] = v
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().astype({"open_time": "int64", "close_time": "int64", "close": "float64"})
        df.to_parquet(cp, index=False); frames.append(df)
        print("BINANCE_1S", key, len(df), flush=True)
    if not frames:
        raise RuntimeError("empty Binance 1s range")
    return stats_base.BinanceSecond(pd.concat(frames, ignore_index=True))


def build_anchors(start: str, end: str, out: Path):
    cache = out / "anchor_cache"; cache.mkdir(parents=True, exist_ok=True)
    start_d = date.fromisoformat(start); end_d = date.fromisoformat(end) - timedelta(days=1)
    print("ANCHOR_BINANCE_1M", start, end, flush=True)
    bn_df = original.download_binance_1m(start_d, end_d, cache / "binance_1m")
    bn = original.BinanceAnchor.from_df(bn_df)
    print("ANCHOR_DERIBIT_INSTRUMENTS", flush=True)
    inst = original.deribit_instruments()
    anchor_start = start_d - timedelta(days=1)
    selected = original.select_deribit_instruments(inst, bn_df, anchor_start, end_d)
    print("ANCHOR_DERIBIT_SELECTED", len(selected), flush=True)
    trades = fetch_deribit_trades_parallel(selected, anchor_start, end_d, cache / "deribit_trades.parquet")
    der = original.DeribitAnchor.from_trades(trades, inst)
    meta = {"selected_deribit_instruments": len(selected), "usable_deribit_anchor_trades": int(len(der.ts))}
    return bn, der, meta


def fair_boundary(m: Market, sec: int, spot: stats_base.BinanceSecond, bn, der):
    if sec >= m.close:
        return None
    rv = bn.rv_annualized(sec * 1000, 60)
    div = der.median_iv(sec * 1000, 30)
    if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
        return None
    op = bn.open_price(m.start); sp = spot.at(sec)
    if not (op > 0 and sp > 0):
        return None
    s2c = m.close - sec
    p_rv = original.digital_prob_up(sp / op, s2c, rv)
    p_iv = original.digital_prob_up(sp / op, s2c, div)
    if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
        return None
    lo, hi = min(p_rv, p_iv), max(p_rv, p_iv)
    return {"up": float(lo), "down": float(1.0 - hi), "p_rv": float(p_rv), "p_iv": float(p_iv),
            "rv": float(rv), "div": float(div), "spot": float(sp), "open_spot": float(op)}


def witness_limit(q: pd.DataFrame, qty: float = SIZE):
    return stats_base.witness_limit(q, qty)


def buy_fill(post: pd.DataFrame, m: Market, outcome: str, limit: float, qty: float = SIZE):
    z = post[(post["side_u"] == "BUY") & (post["outcome_l"] == outcome) & (post["price"] <= limit)].copy()
    if z.empty:
        return None
    z = z.sort_values(["timestamp", "price", "size"], kind="mergesort")
    left = float(qty); gross = fees = 0.0; done = None
    for r in z.itertuples(index=False):
        take = min(left, float(r.size))
        if take <= 0: continue
        px = float(r.price); gross += take * px; fees += fee_total(m, px, take); left -= take; done = int(r.timestamp)
        if left <= 1e-12:
            return {"cost": gross + fees, "avg": gross / qty, "done": done}
    return None


def sell_witness(qsec: pd.DataFrame, m: Market, outcome: str, min_price: float, qty: float = SIZE):
    z = qsec[(qsec["side_u"] == "SELL") & (qsec["outcome_l"] == outcome) & (qsec["price"] >= min_price) & (qsec["size"] > 0)].copy()
    if z.empty: return None
    z = z.sort_values(["price", "size"], ascending=[False, True], kind="mergesort")
    left = float(qty); gross = fees = 0.0
    for r in z.itertuples(index=False):
        take = min(left, float(r.size)); px = float(r.price)
        gross += take * px; fees += fee_total(m, px, take); left -= take
        if left <= 1e-12:
            return {"proceeds": gross - fees, "avg": gross / qty}
    return None


def strict_sell_fill(g: pd.DataFrame, m: Market, outcome: str, limit: float, witness_sec: int, close: int, qty: float = SIZE):
    z = g[(g["timestamp"] >= witness_sec + 1) & (g["timestamp"] <= min(witness_sec + 5, close - 1)) &
          (g["side_u"] == "SELL") & (g["outcome_l"] == outcome) & (g["price"] >= limit) & (g["size"] > 0)].copy()
    if z.empty: return None
    z = z.sort_values(["timestamp", "price", "size"], ascending=[True, False, True], kind="mergesort")
    left = float(qty); gross = fees = 0.0; done = None
    for r in z.itertuples(index=False):
        take = min(left, float(r.size)); px = float(r.price)
        gross += take * px; fees += fee_total(m, px, take); left -= take; done = int(r.timestamp)
        if left <= 1e-12:
            return {"proceeds": gross - fees, "avg": gross / qty, "done": done}
    return None


def execute_signal(m: Market, g: pd.DataFrame, sec: int, outcome: str, limit: float, edge: float, raw_gap: float, fb: dict,
                   spot: stats_base.BinanceSecond, bn, der):
    won = (m.label_up >= 0.5) if outcome == "up" else (m.label_up < 0.5)
    payout = SIZE if won else 0.0
    paper_cost = SIZE * limit + fee_total(m, limit, SIZE)
    post5 = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] <= sec + 5)]
    entry = buy_fill(post5, m, outcome, limit, SIZE)
    regime = "large_convergence" if raw_gap >= LARGE_RAW_GAP else "small_settlement"

    witness_reward = payout - paper_cost
    witness_exit_time = m.close
    witness_exit_kind = "settlement" if regime == "small_settlement" else "settlement_fallback"
    witness_hold_s = m.close - sec
    witness_exit_px = 1.0 if won else 0.0
    strict_cost = strict_reward = strict_entry_time = strict_exit_time = strict_hold_s = strict_exit_px = None
    strict_exit_kind = "no_entry_fill" if entry is None else ("settlement" if regime == "small_settlement" else "settlement_fallback")
    if entry is not None:
        strict_cost = float(entry["cost"]); strict_entry_time = int(entry["done"])
        strict_reward = payout - strict_cost; strict_exit_time = m.close
        strict_hold_s = m.close - strict_entry_time; strict_exit_px = 1.0 if won else 0.0

    first_conv = None
    if regime == "large_convergence":
        sells = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] < m.close) & (g["side_u"] == "SELL") & (g["outcome_l"] == outcome)]
        witness_exit = None; strict_exit = None
        for sx in sorted(int(x) for x in sells["timestamp"].unique().tolist()):
            fbx = fair_boundary(m, sx, spot, bn, der)
            if fbx is None: continue
            min_price = max(0.0, min(1.0, fbx[outcome] - EXIT_BAND))
            w = sell_witness(sells[sells["timestamp"] == sx], m, outcome, min_price, SIZE)
            if w is None: continue
            if first_conv is None: first_conv = sx
            if witness_exit is None:
                witness_exit = {**w, "sec": sx, "limit": min_price}
            if entry is not None and sx >= int(entry["done"]) + 1 and strict_exit is None:
                sf = strict_sell_fill(g, m, outcome, min_price, sx, m.close, SIZE)
                if sf is not None:
                    strict_exit = {**sf, "witness_sec": sx, "limit": min_price}
                    break
        if witness_exit is not None:
            witness_reward = float(witness_exit["proceeds"]) - paper_cost
            witness_exit_time = int(witness_exit["sec"]); witness_exit_kind = "convergence"
            witness_hold_s = witness_exit_time - sec; witness_exit_px = float(witness_exit["avg"])
        if entry is not None and strict_exit is not None:
            strict_reward = float(strict_exit["proceeds"]) - float(entry["cost"])
            strict_exit_time = int(strict_exit["done"]); strict_exit_kind = "convergence"
            strict_hold_s = strict_exit_time - int(entry["done"]); strict_exit_px = float(strict_exit["avg"])

    return {
        "slug": m.slug, "condition_id": m.condition_id, "start": m.start, "close": m.close,
        "decision": int(sec), "s2c": int(m.close - sec), "side": "Up" if outcome == "up" else "Down", "won": bool(won),
        "limit": float(limit), "signal_edge": float(edge), "initial_raw_gap": float(raw_gap), "regime": regime,
        "p_rv": fb["p_rv"], "p_deribit": fb["p_iv"], "rv": fb["rv"], "deribit_iv": fb["div"], "spot": fb["spot"], "open_spot": fb["open_spot"],
        "fee_enabled": m.fee_enabled, "fee_type": m.fee_type, "fee_rate": m.fee_rate, "fee_exponent": m.fee_exponent, "fee_source": m.fee_source,
        "witness_cost": float(paper_cost), "witness_reward": float(witness_reward), "witness_exit_time": int(witness_exit_time),
        "witness_exit_kind": witness_exit_kind, "witness_hold_s": int(witness_hold_s), "witness_exit_px": float(witness_exit_px),
        "strict_cost": strict_cost, "strict_reward": strict_reward, "strict_entry_time": strict_entry_time,
        "strict_exit_time": strict_exit_time, "strict_exit_kind": strict_exit_kind, "strict_hold_s": strict_hold_s, "strict_exit_px": strict_exit_px,
        "first_convergence_witness_sec": first_conv,
    }


def score_market(m: Market, g: pd.DataFrame, spot: stats_base.BinanceSecond, bn, der):
    if g is None or g.empty: return None
    pre = g[(g["timestamp"] >= m.close - MAX_S2C) & (g["timestamp"] <= m.close - MIN_S2C) & (g["side_u"] == "BUY")]
    if pre.empty: return None
    for sec in sorted(int(x) for x in pre["timestamp"].unique().tolist()):
        qsec = pre[pre["timestamp"] == sec]
        fb = fair_boundary(m, sec, spot, bn, der)
        if fb is None: continue
        candidates = []
        for outcome in ("up", "down"):
            lim = witness_limit(qsec[qsec["outcome_l"] == outcome], SIZE)
            if lim is None: continue
            fair = float(fb[outcome]); raw = fair - float(lim)
            edge = raw - fee_per_share_for_signal(m, float(lim))
            if edge >= THRESHOLD:
                candidates.append((float(edge), outcome, float(lim), float(raw)))
        if candidates:
            edge, outcome, limit, raw_gap = max(candidates, key=lambda x: (x[0], x[1] == "up"))
            return execute_signal(m, g, sec, outcome, limit, edge, raw_gap, fb, spot, bn, der)
    return None


def score_hour(hour: int, markets: list[Market], spot, bn, der):
    if not markets: return [], 0
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-final-recent-trades/1.0"})
    raw = query_trade_rows(sess, markets, hour + 300, hour + 3599)
    tm = normalize_trades(raw); out = []
    for m in markets:
        z = score_market(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der)
        if z is not None: out.append(z)
    return out, len(raw)


RECORD_COLUMNS = [
    "slug","condition_id","start","close","decision","s2c","side","won","limit","signal_edge","initial_raw_gap","regime",
    "p_rv","p_deribit","rv","deribit_iv","spot","open_spot","fee_enabled","fee_type","fee_rate","fee_exponent","fee_source",
    "witness_cost","witness_reward","witness_exit_time","witness_exit_kind","witness_hold_s","witness_exit_px",
    "strict_cost","strict_reward","strict_entry_time","strict_exit_time","strict_exit_kind","strict_hold_s","strict_exit_px","first_convergence_witness_sec",
]


def score_shard(start: str, end: str, shard: str, out: Path, workers: int = 12):
    if not (ts("2025-08-15") <= ts(start) < ts(end) <= ts(CUTOFF)):
        raise RuntimeError(f"shard outside frozen range: {start} {end}")
    out.mkdir(parents=True, exist_ok=True)
    hours, by_hour, inventory = discover(start, end, workers)
    inventory.to_csv(out / "market_inventory.csv", index=False)
    mapped = int(inventory["mapped"].sum()) if len(inventory) else 0
    if mapped == 0:
        pd.DataFrame(columns=RECORD_COLUMNS).to_csv(out / "dual_exit_records.csv", index=False)
        summary = {"shard": shard, "period": [start,end], "expected_markets": int(len(inventory)), "mapped_markets": 0,
                   "gamma_existing_unparsed": int((inventory["exists_in_gamma"] & ~inventory["mapped"]).sum()) if len(inventory) else 0,
                   "signals": 0, "raw_trade_rows": 0, "anchor_meta": None, "protocol": PROTOCOL}
        (out / "summary.json").write_text(json.dumps(summary, indent=2)); print("SHARD_NO_MARKETS", json.dumps(summary), flush=True); return
    bn, der, anchor_meta = build_anchors(start, end, out)
    spot = load_binance_1s(start, end, out / "binance_1s_cache")
    records = []; raw_rows = 0; failures = []
    active_hours = [h for h in hours if by_hour.get(h)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(score_hour, h, by_hour[h], spot, bn, der): h for h in active_hours}
        for k, f in enumerate(as_completed(fut), 1):
            h = fut[f]
            try:
                rr, nr = f.result(); records.extend(rr); raw_rows += int(nr)
            except Exception as exc:
                failures.append({"hour": h, "error": repr(exc)})
            if k % 48 == 0:
                print("SCORE", shard, k, "/", len(active_hours), "signals", len(records), "raw", raw_rows, "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"hour score failures {failures[:10]} count={len(failures)}")
    df = pd.DataFrame(records, columns=RECORD_COLUMNS).sort_values(["decision","slug"], kind="mergesort") if records else pd.DataFrame(columns=RECORD_COLUMNS)
    df.to_csv(out / "dual_exit_records.csv", index=False)
    fee_counts = inventory[inventory["mapped"]].groupby(["fee_enabled","fee_type","fee_rate","fee_exponent"], dropna=False).size().astype(int).to_dict()
    summary = {
        "shard": shard, "period": [start,end], "expected_markets": int(len(inventory)), "mapped_markets": mapped,
        "mapped_ratio_all_slots": float(mapped / len(inventory)) if len(inventory) else None,
        "gamma_existing_unparsed": int((inventory["exists_in_gamma"] & ~inventory["mapped"]).sum()),
        "signals": int(len(df)), "large": int((df["regime"] == "large_convergence").sum()) if len(df) else 0,
        "strict_entries": int(df["strict_cost"].notna().sum()) if len(df) else 0, "raw_trade_rows": raw_rows,
        "fee_market_counts": {str(k): v for k,v in fee_counts.items()}, "anchor_meta": anchor_meta, "protocol": PROTOCOL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("FINAL_RECENT_SHARD_DONE", json.dumps(summary, indent=2, default=float), flush=True)


def convergence_stats(z: pd.DataFrame, prefix: str):
    g = z[z["regime"] == "large_convergence"].copy()
    if prefix == "strict":
        g = g[g["strict_cost"].notna()].copy(); kind = g["strict_exit_kind"]; hold = pd.to_numeric(g["strict_hold_s"], errors="coerce")
    else:
        kind = g["witness_exit_kind"]; hold = pd.to_numeric(g["witness_hold_s"], errors="coerce")
    conv = kind == "convergence"; ch = hold[conv & hold.notna()]
    return {"eligible_large": int(len(g)), "convergence_exits": int(conv.sum()), "convergence_rate": float(conv.mean()) if len(g) else None,
            "within_60s": float((ch <= 60).sum()/len(g)) if len(g) else None, "within_300s": float((ch <= 300).sum()/len(g)) if len(g) else None,
            "median_convergence_s": float(ch.median()) if len(ch) else None,
            "settlement_fallbacks": int((kind == "settlement_fallback").sum()),
            "no_entry_fill": int((kind == "no_entry_fill").sum()) if prefix == "strict" else 0}


def layer_stats(z: pd.DataFrame, prefix: str, start: str, end: str):
    cost_col = f"{prefix}_cost"; reward_col = f"{prefix}_reward"; exit_col = f"{prefix}_exit_time"
    g = z.dropna(subset=[cost_col,reward_col,exit_col]).copy()
    if g.empty: return {"n": 0}
    g["entry_month"] = pd.to_datetime(g["decision"], unit="s", utc=True).dt.strftime("%Y-%m")
    g["exit_day"] = pd.to_datetime(g[exit_col], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    days = pd.date_range(start, pd.Timestamp(end)-pd.Timedelta(days=1), freq="D", tz="UTC").strftime("%Y-%m-%d")
    daily = g.groupby("exit_day").agg(reward=(reward_col,"sum"), cost=(cost_col,"sum"), n=(reward_col,"size")).reindex(days, fill_value=0.0)
    nday = daily["n"].to_numpy(float); edge_day = np.divide(daily["reward"].to_numpy(float), nday*SIZE, out=np.zeros(len(daily)), where=nday>0)
    monthly = g.groupby("entry_month").agg(pnl=(reward_col,"sum"), cost=(cost_col,"sum"), n=(reward_col,"size"))
    by_month = {m: {"n":int(r.n),"pnl":float(r.pnl),"roi":float(r.pnl/r.cost)} for m,r in monthly.iterrows()}
    by_regime = {}
    for reg, q in g.groupby("regime"):
        by_regime[str(reg)] = {"n":int(len(q)),"pnl":float(q[reward_col].sum()),"roi":float(q[reward_col].sum()/q[cost_col].sum())}
    return {"n":int(len(g)),"pnl":float(g[reward_col].sum()),"cost_sum":float(g[cost_col].sum()),
            "roi":float(g[reward_col].sum()/g[cost_col].sum()),"edge_share":float(g[reward_col].sum()/(len(g)*SIZE)),
            "iid_day_edge_ci95": stats_base.iid_boot(edge_day, reps=5000), "mbb7_edge_ci95": stats_base.mbb_boot(edge_day,7,reps=5000),
            "mbb7_roi_ci95": stats_base.mbb_ratio(daily["reward"].to_numpy(float), daily["cost"].to_numpy(float),7,reps=5000),
            "newey_west_t_edge_lag7": stats_base.nw_tstat(edge_day,7), "by_month":by_month,"by_regime":by_regime}


def bankroll_replay(z: pd.DataFrame, prefix: str, initial=50.0, base_trade=5.0, single_cap=100.0, market_cap=200.0):
    cost_col=f"{prefix}_cost"; reward_col=f"{prefix}_reward"; exit_col=f"{prefix}_exit_time"; entry_col="strict_entry_time" if prefix=="strict" else "decision"
    g=z.dropna(subset=[cost_col,reward_col,exit_col]).copy()
    if g.empty: return {"trades_opened":0}
    g[entry_col]=g[entry_col].fillna(g["decision"]).astype(np.int64); g[exit_col]=g[exit_col].astype(np.int64)
    g=g.sort_values([entry_col,"slug"],kind="mergesort")
    cash=float(initial); active=[]; curve=[]; skipped=0; opened=0; cap_ts=None
    def locked(): return float(sum(p[3] for p in active))
    def eq(): return cash+locked()
    def settle(t):
        nonlocal cash, active
        due=sorted([p for p in active if p[2]<=t], key=lambda p:(p[2],p[1]))
        for p in due:
            cash += p[4]; active.remove(p); curve.append((p[2],cash+locked()))
    for r in g.itertuples(index=False):
        d=r._asdict(); t=int(d[entry_col]); settle(t); e=eq(); stake=bankroll_base.target_stake(e,initial,base_trade,single_cap)
        if cap_ts is None and stake >= single_cap-1e-12: cap_ts=t
        cid=str(d["condition_id"]); mexp=sum(p[3] for p in active if p[0]==cid)
        if cash+1e-9<stake or mexp+stake>market_cap+1e-9: skipped+=1; curve.append((t,e)); continue
        cps=float(d[cost_col])/SIZE; pps=(float(d[cost_col])+float(d[reward_col]))/SIZE
        shares=stake/cps; proceeds=shares*pps
        cash-=stake; active.append((cid,str(d["slug"]),int(d[exit_col]),float(stake),float(proceeds))); opened+=1; curve.append((t,cash+locked()))
    if active: settle(max(p[2] for p in active)+1)
    final=float(cash); vals=[x[1] for x in sorted(curve,key=lambda x:x[0])]
    first=int(g[entry_col].min()); last=max(int(g[exit_col].max()), first); days=(last-first)/86400.0
    post=None
    if cap_ts is not None:
        pre=[x for x in curve if x[0]<=cap_ts]; eqcap=pre[-1][1] if pre else initial; pdays=(last-cap_ts)/86400.0
        post=(final-eqcap)*365.0/pdays if pdays>0 else None
    return {"initial_capital":initial,"final_capital":final,"profit":final-initial,"multiple":final/initial,"trades_opened":opened,"skipped":skipped,
            "max_drawdown_realized_cost_basis":bankroll_base.max_drawdown(vals),"cap_reached_ts":cap_ts,
            "days_to_cap":((cap_ts-first)/86400.0 if cap_ts is not None else None),"post_cap_pnl_runrate_per_year":post,"elapsed_days":days}


def aggregate(root: Path, out: Path):
    rec_files=sorted(root.glob("**/dual_exit_records.csv")); inv_files=sorted(root.glob("**/market_inventory.csv"))
    if not rec_files or not inv_files: raise RuntimeError(f"missing shard artifacts records={len(rec_files)} inventory={len(inv_files)}")
    records=pd.concat([pd.read_csv(p) for p in rec_files],ignore_index=True); inventory=pd.concat([pd.read_csv(p) for p in inv_files],ignore_index=True)
    if len(records): records=records.sort_values(["decision","slug"],kind="mergesort").drop_duplicates("slug",keep="last")
    inventory=inventory.sort_values("start",kind="mergesort").drop_duplicates("slug",keep="last")
    out.mkdir(parents=True,exist_ok=True); records.to_csv(out/"all_records.csv",index=False); inventory.to_csv(out/"all_market_inventory.csv",index=False)
    mapped_all=inventory[inventory["mapped"].astype(bool)]; first_active=int(mapped_all["start"].min()) if len(mapped_all) else None
    result={"protocol":PROTOCOL,"first_active_market_utc":(pd.to_datetime(first_active,unit="s",utc=True).isoformat() if first_active is not None else None),"windows":{}}
    for name,(start,end) in WINDOWS.items():
        a,b=ts(start),ts(end); inv=inventory[(inventory["start"]>=a)&(inventory["start"]<b)].copy(); z=records[(records["decision"]>=a)&(records["decision"]<b)].copy() if len(records) else records.copy()
        mapped=int(inv["mapped"].astype(bool).sum()) if len(inv) else 0; expected=int(len(inv))
        active_floor=max(a,first_active) if first_active is not None else b; active_inv=inv[inv["start"]>=active_floor]
        active_cov=float(active_inv["mapped"].astype(bool).mean()) if len(active_inv) else None
        win={"period":[start,end],"calendar_days":float((pd.Timestamp(end)-pd.Timestamp(start)).days),"expected_slots":expected,"mapped_markets":mapped,
             "mapped_ratio_all_calendar_slots":float(mapped/expected) if expected else None,"active_market_coverage":active_cov,"signals":int(len(z)),
             "regime_counts":z["regime"].value_counts().to_dict() if len(z) else {},
             "witness":layer_stats(z,"witness",start,end),"strict5":layer_stats(z,"strict",start,end),
             "convergence_witness":convergence_stats(z,"witness") if len(z) else {},"convergence_strict5":convergence_stats(z,"strict") if len(z) else {},
             "bankroll_witness_50_5_100":bankroll_replay(z,"witness"),"bankroll_strict_50_5_100":bankroll_replay(z,"strict")}
        result["windows"][name]=win
    (out/"summary.json").write_text(json.dumps(result,indent=2,default=float))
    lines=["# Main Sequence FINAL recent replay",""]
    for name,w in result["windows"].items():
        ww=w["witness"]; ss=w["strict5"]; cw=w["convergence_witness"]; bw=w["bankroll_witness_50_5_100"]
        lines += [f"## {name} — {w['period'][0]} → {w['period'][1]}",
                  f"Mapped {w['mapped_markets']}/{w['expected_slots']} calendar slots; active coverage={w['active_market_coverage']}",
                  f"Signals={w['signals']} regimes={w['regime_counts']}",
                  f"Witness: n={ww.get('n')} PnL={ww.get('pnl')} ROI={ww.get('roi')} edge/share={ww.get('edge_share')}",
                  f"Strict5: n={ss.get('n')} PnL={ss.get('pnl')} ROI={ss.get('roi')} edge/share={ss.get('edge_share')}",
                  f"Large convergence witness: {cw}",
                  f"$50->$100-cap witness bankroll: {bw}",""]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("FINAL_RECENT_AGGREGATE",json.dumps(result,indent=2,default=float),flush=True)


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("protocol"); p.add_argument("--out",type=Path,required=True)
    s=sp.add_parser("score"); s.add_argument("--start",required=True); s.add_argument("--end",required=True); s.add_argument("--shard",required=True); s.add_argument("--out",type=Path,required=True); s.add_argument("--workers",type=int,default=12)
    a=sp.add_parser("aggregate"); a.add_argument("--root",type=Path,required=True); a.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.cmd=="protocol":
        args.out.mkdir(parents=True,exist_ok=True); (args.out/"FROZEN_FINAL_RECENT_PROTOCOL.json").write_text(json.dumps(PROTOCOL,indent=2)); print(json.dumps(PROTOCOL,indent=2))
    elif args.cmd=="score": score_shard(args.start,args.end,args.shard,args.out,args.workers)
    else: aggregate(args.root,args.out)


if __name__=="__main__":
    main()
