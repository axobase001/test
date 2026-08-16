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

from pm_structural import recalc as original

# ---------------------------------------------------------------------------
# Frozen protocol: original structural_03c policy, no training/tuning.
# ---------------------------------------------------------------------------
START = "2026-03-01"
END_EXCLUSIVE = "2026-06-02"  # 93 calendar days, includes 2026-06-01
DAYS = 93
THRESHOLD = 0.03
SIZE = 5.0
MIN_S2C = 60
MAX_S2C = 600
FEE_RATE = 0.07
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_1S = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s"

SHARDS = {
    "mar": ["2026-03-01", "2026-04-01"],
    "apr": ["2026-04-01", "2026-05-01"],
    "may": ["2026-05-01", "2026-06-01"],
    "jun1": ["2026-06-01", "2026-06-02"],
}

CONTRACT = {
    "name": "Main Sequence original structural_03c / 93-day official-tape replay v1",
    "period": [START, END_EXCLUSIVE],
    "days": DAYS,
    "asset": "BTC",
    "timeframe": "15m",
    "strategy": "original structural_03c: conservative interval outside Binance-60m-RV and backward 30m Deribit trade-IV digital anchors",
    "threshold": THRESHOLD,
    "size_shares": SIZE,
    "decision_window_s2c": [MIN_S2C, MAX_S2C],
    "fee_rate": FEE_RATE,
    "once_per_market": True,
    "quote_witness": (
        "A decision price is admitted only when public taker-BUY tape in that exact integer second "
        "contains >=5 shares in the outcome at or below the witness limit. This replaces incomplete "
        "third-party quarter book archives with a single-source conservative executable price witness; "
        "it does not alter either fair-value anchor or the 3c policy threshold."
    ),
    "execution": (
        "The decision-second witness is never counted as our fill. tape_next_second requires a new "
        "public taker-BUY print in the chosen outcome during decision+1s at prices <= frozen limit; "
        "tape_5s requires sufficient new public taker-BUY volume during decision+1..+5s."
    ),
    "statistics": (
        "All 93 calendar days retained. iid-day and 7-day moving-block bootstrap; Newey-West lag7 "
        "diagnostic; Mar/Apr/May month-sign consistency; actual sequential capital path and ex-post "
        "10/20/30% historical-max-DD annualization."
    ),
    "anti_tuning": (
        "No network fitting, no asset selection, no threshold search, no use of the 92-day MLP result "
        "to alter the original 3c policy. Transport fixes may only preserve or make execution evidence "
        "more conservative."
    ),
}


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def fee_per_share(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    return FEE_RATE * p * (1.0 - p)


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
            time.sleep(min(0.5 * (2 ** i), 12.0) + np.random.default_rng(i + 17).random() * 0.1)
    raise RuntimeError(f"GET failed {url}: {last!r}")


@dataclass(frozen=True)
class Market:
    slug: str
    start: int
    condition_id: str
    label_up: float

    @property
    def close(self) -> int:
        return self.start + 900


def parse_market(raw: dict, start: int) -> Market | None:
    try:
        outcomes = json.loads(raw["outcomes"])
        prices = json.loads(raw.get("outcomePrices") or "[]")
        oi = {str(x).strip().lower(): i for i, x in enumerate(outcomes)}
        if "up" not in oi or "down" not in oi:
            return None
        ui, di = oi["up"], oi["down"]
        if len(prices) <= max(ui, di):
            return None
        pp = [float(x) for x in prices]
        if max(pp) < 0.99:
            return None
        label = 1.0 if pp[ui] > pp[di] else 0.0
        cid = str(raw.get("conditionId") or "")
        if not cid:
            return None
        return Market(str(raw["slug"]), int(start), cid, label)
    except Exception:
        return None


def fetch_markets_hour(hour: int) -> tuple[list[Market], list[str]]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-original-93d/1.0"})
    wanted = [(f"btc-updown-15m-{t0}", t0) for t0 in range(hour, hour + 3600, 900)]
    params = [("slug", slug) for slug, _ in wanted] + [("closed", "true"), ("limit", 20)]
    js = get_json(sess, GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js if isinstance(x, dict)}
    out, missing = [], []
    for slug, t0 in wanted:
        m = parse_market(byslug.get(slug, {}), t0)
        if m is None:
            missing.append(slug)
        else:
            out.append(m)
    return out, missing


def query_trade_rows(sess: requests.Session, markets: list[Market], start: int, end: int) -> list[dict]:
    if not markets:
        return []
    q = {
        "market": ",".join(m.condition_id for m in markets),
        "start": int(start),
        "end": int(end),
        "limit": 10000,
        "offset": 0,
        "takerOnly": "true",
    }
    try:
        r = sess.get(DATA_API + "/trades", params=q, timeout=60)
    except requests.RequestException:
        r = None
    if r is not None and r.status_code == 200:
        rows = r.json()
    elif r is not None and r.status_code == 500 and len(markets) > 1:
        mid = len(markets) // 2
        return query_trade_rows(sess, markets[:mid], start, end) + query_trade_rows(sess, markets[mid:], start, end)
    else:
        rows = get_json(sess, DATA_API + "/trades", params=q, timeout=60)
    if len(rows) < 10000:
        return rows
    if len(markets) > 1:
        mid = len(markets) // 2
        return query_trade_rows(sess, markets[:mid], start, end) + query_trade_rows(sess, markets[mid:], start, end)
    if end <= start:
        raise RuntimeError(f"single-second trade cap {markets[0].slug} at {start}")
    mid_t = (start + end) // 2
    return query_trade_rows(sess, markets, start, mid_t) + query_trade_rows(sess, markets, mid_t + 1, end)


def normalize_trades(rows: list[dict]) -> dict[str, pd.DataFrame]:
    if not rows:
        return {}
    x = pd.DataFrame(rows)
    need = {"conditionId", "timestamp", "price", "size", "side", "outcome"}
    if not need.issubset(x.columns):
        raise RuntimeError(f"trade schema missing {need - set(x.columns)}")
    x["timestamp"] = pd.to_numeric(x["timestamp"], errors="coerce")
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"] = pd.to_numeric(x["size"], errors="coerce")
    x = x.dropna(subset=["timestamp", "price", "size"])
    if len(x) and float(x["timestamp"].abs().median()) > 1e12:
        x["timestamp"] = np.floor(x["timestamp"] / 1000.0)
    x["timestamp"] = x["timestamp"].astype(np.int64)
    x = x[(x["price"] > 0) & (x["price"] < 1) & (x["size"] > 0)]
    x["side_u"] = x["side"].astype(str).str.upper().str.strip()
    x["outcome_l"] = x["outcome"].astype(str).str.lower().str.strip()
    x = x[x["outcome_l"].isin(["up", "down"])]
    # Deterministic order for equal-second rows; transaction hash (when present) is only a tie-breaker.
    tie = "transactionHash" if "transactionHash" in x.columns else "conditionId"
    x = x.sort_values(["timestamp", tie, "price", "size"], kind="mergesort")
    return {str(cid): g.reset_index(drop=True) for cid, g in x.groupby("conditionId", sort=False)}


class BinanceSecond:
    def __init__(self, df: pd.DataFrame):
        x = df.sort_values("close_time").drop_duplicates("close_time", keep="last")
        self.t = x["close_time"].to_numpy(np.int64)
        self.c = x["close"].to_numpy(float)

    def at(self, sec: int) -> float:
        i = int(np.searchsorted(self.t, int(sec) * 1000, side="right") - 1)
        return float(self.c[i]) if i >= 0 else math.nan


def load_binance_1s(start: str, end: str, cache: Path) -> BinanceSecond:
    cache.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-original-93d-binance/1.0"})
    frames = []
    for d in pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D"):
        key = d.strftime("%Y-%m-%d")
        cp = cache / f"BTCUSDT-1s-{key}.parquet"
        if cp.exists():
            frames.append(pd.read_parquet(cp)); continue
        url = f"{BINANCE_1S}/BTCUSDT-1s-{key}.zip"
        r = sess.get(url, timeout=120); r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise RuntimeError(f"unexpected Binance 1s archive {url}: {names}")
            df = pd.read_csv(z.open(names[0]), header=None, usecols=[0, 4, 6])
        df.columns = ["open_time", "close", "close_time"]
        for c in ["open_time", "close_time"]:
            v = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            if len(v.dropna()) and int(v.dropna().abs().median()) > 10**14:
                v = v // 1000
            df[c] = v
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().astype({"open_time": "int64", "close_time": "int64", "close": "float64"})
        df.to_parquet(cp, index=False)
        frames.append(df)
        print("BINANCE_1S", key, len(df), flush=True)
    return BinanceSecond(pd.concat(frames, ignore_index=True))


def save_anchors(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    start_d = date.fromisoformat(START)
    end_d = date.fromisoformat("2026-06-01")
    cache = out / "cache"
    print("ANCHOR_BINANCE_1M", flush=True)
    bn_df = original.download_binance_1m(start_d, end_d, cache / "binance_1m")
    bn_df.to_parquet(out / "binance_1m.parquet", index=False)
    print("ANCHOR_DERIBIT_INSTRUMENTS", flush=True)
    inst = original.deribit_instruments()
    selected = original.select_deribit_instruments(inst, bn_df, start_d, end_d)
    (out / "selected_deribit_instruments.txt").write_text("\n".join(selected) + "\n")
    print("ANCHOR_DERIBIT_TRADES", len(selected), flush=True)
    trades = original.fetch_deribit_trades(selected, start_d, end_d, cache / "deribit_trades.parquet")
    der = original.DeribitAnchor.from_trades(trades, inst)
    pd.DataFrame({"ts": der.ts, "iv": der.iv}).to_parquet(out / "deribit_anchor.parquet", index=False)
    contract = dict(CONTRACT)
    contract["selected_deribit_instruments"] = len(selected)
    contract["deribit_usable_anchor_trades"] = int(len(der.ts))
    (out / "FROZEN_CONTRACT.json").write_text(json.dumps(contract, indent=2))
    print("ORIGINAL_93D_CONTRACT_FROZEN", json.dumps(contract, indent=2), flush=True)


def load_anchors(model_dir: Path):
    contract = json.loads((model_dir / "FROZEN_CONTRACT.json").read_text())
    for k in ["period", "threshold", "size_shares", "decision_window_s2c", "fee_rate"]:
        if contract.get(k) != CONTRACT.get(k):
            raise RuntimeError(f"contract mismatch {k}: {contract.get(k)} != {CONTRACT.get(k)}")
    bn = original.BinanceAnchor.from_df(pd.read_parquet(model_dir / "binance_1m.parquet"))
    d = pd.read_parquet(model_dir / "deribit_anchor.parquet")
    der = original.DeribitAnchor(d["ts"].to_numpy(np.int64), d["iv"].to_numpy(float))
    return contract, bn, der


def witness_limit(q: pd.DataFrame, qty: float = SIZE) -> float | None:
    if q.empty:
        return None
    z = q[(q["side_u"] == "BUY") & (q["size"] > 0)].sort_values(["price", "size"], kind="mergesort")
    if z.empty:
        return None
    c = z["size"].cumsum().to_numpy(float)
    idx = np.flatnonzero(c >= qty)
    return float(z.iloc[int(idx[0])]["price"]) if len(idx) else None


def fill_cost(post: pd.DataFrame, outcome: str, limit: float, qty: float = SIZE) -> tuple[float, float] | None:
    z = post[(post["side_u"] == "BUY") & (post["outcome_l"] == outcome) & (post["price"] <= limit)].copy()
    if z.empty:
        return None
    z = z.sort_values(["timestamp", "price", "size"], kind="mergesort")
    left = float(qty)
    gross = 0.0
    fees = 0.0
    for r in z.itertuples(index=False):
        take = min(left, float(r.size))
        if take <= 0:
            continue
        px = float(r.price)
        gross += take * px
        fees += take * fee_per_share(px)
        left -= take
        if left <= 1e-12:
            return gross + fees, gross / qty
    return None


def score_market(m: Market, g: pd.DataFrame, spot: BinanceSecond, bn, der) -> dict | None:
    if g is None or g.empty:
        return None
    lo_t = m.close - MAX_S2C
    hi_t = m.close - MIN_S2C
    pre = g[(g["timestamp"] >= lo_t) & (g["timestamp"] <= hi_t) & (g["side_u"] == "BUY")]
    if pre.empty:
        return None
    for sec in sorted(pre["timestamp"].unique().tolist()):
        sec = int(sec)
        qsec = pre[pre["timestamp"] == sec]
        candidates = []
        rv = bn.rv_annualized(sec * 1000, 60)
        div = der.median_iv(sec * 1000, 30)
        if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
            continue
        op = bn.open_price(m.start)
        sp = spot.at(sec)
        if not (op > 0 and sp > 0):
            continue
        s2c = m.close - sec
        rel = sp / op
        p_rv = original.digital_prob_up(rel, s2c, rv)
        p_iv = original.digital_prob_up(rel, s2c, div)
        if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
            continue
        p_lo, p_hi = min(p_rv, p_iv), max(p_rv, p_iv)
        for outcome in ("up", "down"):
            lim = witness_limit(qsec[qsec["outcome_l"] == outcome], SIZE)
            if lim is None:
                continue
            edge = (p_lo - lim - fee_per_share(lim)) if outcome == "up" else ((1.0 - p_hi) - lim - fee_per_share(lim))
            if edge >= THRESHOLD:
                candidates.append((float(edge), outcome, float(lim)))
        if not candidates:
            continue
        edge, outcome, limit = max(candidates, key=lambda x: (x[0], x[1] == "up"))
        won = (m.label_up >= 0.5) if outcome == "up" else (m.label_up < 0.5)
        paper_cost = SIZE * (limit + fee_per_share(limit))
        payout = SIZE if won else 0.0
        paper_reward = payout - paper_cost
        # Never reuse the decision-second witness as the fill.
        next_second = g[(g["timestamp"] == sec + 1)]
        post5 = g[(g["timestamp"] >= sec + 1) & (g["timestamp"] <= sec + 5)]
        f1 = fill_cost(next_second, outcome, limit, SIZE)
        f5 = fill_cost(post5, outcome, limit, SIZE)
        return {
            "slug": m.slug, "condition_id": m.condition_id, "start": m.start, "close": m.close,
            "decision": sec, "s2c": s2c, "side": "Up" if outcome == "up" else "Down",
            "won": bool(won), "limit": limit, "signal_edge": edge, "p_rv": p_rv,
            "p_deribit": p_iv, "rv": rv, "deribit_iv": div, "spot": sp, "open_spot": op,
            "paper_cost": paper_cost, "paper_reward": paper_reward,
            "next_second_fill": (f1[1] if f1 else None),
            "next_second_cost": (f1[0] if f1 else None),
            "next_second_reward": (payout - f1[0] if f1 else None),
            "tape5_fill": (f5[1] if f5 else None),
            "tape5_cost": (f5[0] if f5 else None),
            "tape5_reward": (payout - f5[0] if f5 else None),
        }
    return None


def score_hour(hour: int, spot: BinanceSecond, bn, der):
    markets, missing = fetch_markets_hour(hour)
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-original-93d-trades/1.0"})
    raw = query_trade_rows(sess, markets, hour + 300, hour + 3545) if markets else []
    tm = normalize_trades(raw)
    records = []
    for m in markets:
        z = score_market(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der)
        if z is not None:
            records.append(z)
    return records, {"mapped": len(markets), "missing": missing, "raw_rows": len(raw)}


def score_shard(model_dir: Path, out: Path, start: str, end: str, shard: str, workers: int = 12) -> None:
    if SHARDS.get(shard) != [start, end]:
        raise RuntimeError(f"undeclared shard {shard} {start} {end}")
    contract, bn, der = load_anchors(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    spot = load_binance_1s(start, end, out / "binance_1s_cache")
    hours = list(range(ts(start), ts(end), 3600))
    records, missing = [], []
    raw_rows = mapped = 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(score_hour, h, spot, bn, der): h for h in hours}
        for k, f in enumerate(as_completed(fut), 1):
            h = fut[f]
            try:
                rr, meta = f.result()
                records.extend(rr); mapped += int(meta["mapped"]); raw_rows += int(meta["raw_rows"])
                missing.extend(meta["missing"])
            except Exception as exc:
                failures.append({"hour": h, "error": repr(exc)})
            if k % 48 == 0:
                print("ORIGINAL_93D_HOURS", shard, k, "/", len(hours), "mapped", mapped, "signals", len(records), "raw", raw_rows, "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"hour transport failures {failures[:10]} count={len(failures)}")
    expected = len(hours) * 4
    ratio = mapped / expected if expected else 0.0
    if ratio < 0.995:
        raise RuntimeError(f"market coverage RED mapped={mapped} expected={expected} ratio={ratio}")
    df = pd.DataFrame(records).sort_values(["decision", "slug"]) if records else pd.DataFrame()
    df.to_csv(out / "signals.csv", index=False)
    summary = {
        "shard": shard, "period": [start, end], "hours": len(hours), "expected_markets": expected,
        "mapped_markets": mapped, "mapped_ratio": ratio, "missing_market_count": len(missing),
        "missing_market_sample": missing[:50], "raw_trade_rows": raw_rows, "signals": len(df),
        "next_second_fills": int(df["next_second_cost"].notna().sum()) if len(df) else 0,
        "tape5_fills": int(df["tape5_cost"].notna().sum()) if len(df) else 0,
        "contract": contract,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("ORIGINAL_93D_SHARD_DONE", json.dumps(summary, indent=2, default=float), flush=True)


def iid_boot(x: np.ndarray, reps: int = 20000, seed: int = 20260815):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); z = np.empty(reps)
    for i in range(reps): z[i] = rng.choice(x, len(x), replace=True).mean()
    return [float(np.quantile(z, .025)), float(np.quantile(z, .975))]


def mbb_boot(x: np.ndarray, block: int = 7, reps: int = 20000, seed: int = 20260816):
    x = np.asarray(x, float); n = len(x)
    if n < block: return iid_boot(x, reps, seed)
    starts = np.arange(n - block + 1); k = int(math.ceil(n / block)); rng = np.random.default_rng(seed); z = np.empty(reps)
    for i in range(reps):
        s = rng.choice(starts, k, replace=True)
        y = np.concatenate([x[j:j+block] for j in s])[:n]
        z[i] = y.mean()
    return [float(np.quantile(z, .025)), float(np.quantile(z, .975))]


def mbb_ratio(reward: np.ndarray, cost: np.ndarray, block: int = 7, reps: int = 20000, seed: int = 20260817):
    reward = np.asarray(reward, float); cost = np.asarray(cost, float); n = len(reward)
    starts = np.arange(max(1, n - block + 1)); k = int(math.ceil(n / block)); rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        s = rng.choice(starts, k, replace=True)
        ix = np.concatenate([np.arange(j, min(j+block, n)) for j in s])[:n]
        den = cost[ix].sum()
        if den > 0: vals.append(float(reward[ix].sum() / den))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def nw_tstat(x: np.ndarray, lag: int = 7):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return None
    u = x - x.mean(); gamma0 = float(np.dot(u, u) / n); s = gamma0
    for l in range(1, min(lag, n-1) + 1):
        g = float(np.dot(u[l:], u[:-l]) / n); w = 1.0 - l / (lag + 1.0); s += 2*w*g
    se = math.sqrt(max(s, 0.0) / n)
    return float(x.mean() / se) if se > 0 else None


def max_dd_for_capital(cum: np.ndarray, capital: float):
    eq = capital + np.asarray(cum, float); peak = np.maximum.accumulate(np.concatenate([[capital], eq]))[1:]
    return float(np.max((peak - eq) / np.maximum(peak, 1e-12))) if len(eq) else 0.0


def capital_for_dd(cost: np.ndarray, reward: np.ndarray, target: float):
    pnl_before = np.concatenate([[0.0], np.cumsum(reward)[:-1]])
    min_sol = max(1e-9, float(np.max(cost - pnl_before)))
    cum = np.cumsum(reward)
    if max_dd_for_capital(cum, min_sol) <= target:
        return min_sol, max_dd_for_capital(cum, min_sol)
    lo, hi = min_sol, max(2*min_sol, 1.0)
    while max_dd_for_capital(cum, hi) > target: hi *= 2
    for _ in range(80):
        mid = (lo + hi)/2
        if max_dd_for_capital(cum, mid) > target: lo = mid
        else: hi = mid
    return hi, max_dd_for_capital(cum, hi)


def layer_stats(df: pd.DataFrame, cost_col: str, reward_col: str, all_days: list[str]):
    z = df[df[cost_col].notna() & df[reward_col].notna()].copy()
    if z.empty:
        return {"n": 0}
    z["day"] = pd.to_datetime(z["decision"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    z["month"] = pd.to_datetime(z["decision"], unit="s", utc=True).dt.strftime("%Y-%m")
    daily = z.groupby("day").agg(reward=(reward_col,"sum"), cost=(cost_col,"sum"), n=(reward_col,"size")).reindex(all_days, fill_value=0.0)
    edge_day = np.where(daily["n"].to_numpy(float)>0, daily["reward"].to_numpy(float)/(daily["n"].to_numpy(float)*SIZE), 0.0)
    dr, dc = daily["reward"].to_numpy(float), daily["cost"].to_numpy(float)
    monthly = z.groupby("month").agg(reward=(reward_col,"sum"), cost=(cost_col,"sum"), n=(reward_col,"size"))
    month_out = {m: {"n": int(r.n), "pnl": float(r.reward), "roi": float(r.reward/r.cost), "edge_share": float(r.reward/(r.n*SIZE))} for m,r in monthly.iterrows()}
    reward = z[reward_col].to_numpy(float); cost = z[cost_col].to_numpy(float)
    order = np.argsort(z["close"].to_numpy(np.int64), kind="mergesort"); reward_o, cost_o = reward[order], cost[order]
    total = float(reward.sum()); annual = {}
    for target in (0.10,0.20,0.30):
        cap, dd = capital_for_dd(cost_o, reward_o, target); ret = total/cap
        annual[f"target_dd_{int(target*100)}pct"] = {"capital":cap,"period_return":ret,"max_drawdown":dd,
            "simple_annualized": ret*365.0/DAYS,
            "cagr": (1+ret)**(365.0/DAYS)-1 if ret>-1 else None}
    return {
        "n": int(len(z)), "pnl": total, "cost_sum": float(cost.sum()), "roi": float(total/cost.sum()),
        "edge_share": float(total/(len(z)*SIZE)), "win_rate": float(z["won"].astype(bool).mean()),
        "iid_day_edge_ci95": iid_boot(edge_day), "mbb7_edge_ci95": mbb_boot(edge_day),
        "mbb7_roi_ci95": mbb_ratio(dr,dc), "newey_west_t_edge_lag7": nw_tstat(edge_day,7),
        "by_month": month_out, "annualization": annual,
    }


def aggregate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames=[]; shard_summaries={}
    for shard in SHARDS:
        dirs=list(root.glob(f"**/main-sequence-original-93d-{shard}")) + list(root.glob(f"**/original93_{shard}"))
        if not dirs: raise RuntimeError(f"missing shard dir {shard}")
        d=dirs[0]; sp=d/"signals.csv"; jp=d/"summary.json"
        if sp.exists() and sp.stat().st_size>1: frames.append(pd.read_csv(sp))
        shard_summaries[shard]=json.loads(jp.read_text())
    df=pd.concat(frames,ignore_index=True).sort_values(["decision","slug"]) if frames else pd.DataFrame()
    df.to_csv(out/"all_signals.csv",index=False)
    all_days=pd.date_range(START,pd.Timestamp(END_EXCLUSIVE)-pd.Timedelta(days=1),freq="D",tz="UTC").strftime("%Y-%m-%d").tolist()
    paper=layer_stats(df,"paper_cost","paper_reward",all_days)
    nexts=layer_stats(df,"next_second_cost","next_second_reward",all_days)
    tape5=layer_stats(df,"tape5_cost","tape5_reward",all_days)
    expected=DAYS*96; mapped=sum(int(x["mapped_markets"]) for x in shard_summaries.values())
    if mapped/expected < .995: raise RuntimeError(f"aggregate market coverage RED {mapped}/{expected}")
    positive_full_months=sum(1 for m in ("2026-03","2026-04","2026-05") if tape5.get("by_month",{}).get(m,{}).get("edge_share",-math.inf)>0)
    if tape5.get("n",0)==0 or tape5.get("edge_share",0)<=0:
        verdict="RED"
    elif (tape5["iid_day_edge_ci95"][0]>0 and tape5["mbb7_edge_ci95"][0]>0 and tape5["mbb7_roi_ci95"][0]>0 and positive_full_months>=2):
        verdict="GREEN"
    else:
        verdict="YELLOW"
    summary={"contract":CONTRACT,"coverage":{"expected_markets":expected,"mapped_markets":mapped,"ratio":mapped/expected,"shards":shard_summaries},
             "signals":int(len(df)),"paper":paper,"tape_next_second":nexts,"tape_5s":tape5,
             "positive_full_months_tape5":positive_full_months,"execution_verdict":verdict}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    lines=["# Main Sequence original structural_03c — 93-day tape replay","",f"Period: {START} to {END_EXCLUSIVE} exclusive ({DAYS} days).",f"Execution verdict (5s tape): **{verdict}**","", "## Paper","```json",json.dumps(paper,indent=2,default=float),"```","","## Next-second tape","```json",json.dumps(nexts,indent=2,default=float),"```","","## 5s tape","```json",json.dumps(tape5,indent=2,default=float),"```",""]
    (out/"SUMMARY.md").write_text("\n".join(lines)); print((out/"SUMMARY.md").read_text(),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["anchors","score","aggregate"]); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--model-dir",type=Path); ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--shard"); ap.add_argument("--root",type=Path)
    args=ap.parse_args()
    if args.phase=="anchors": save_anchors(args.out)
    elif args.phase=="score":
        if not all([args.model_dir,args.start,args.end,args.shard]): raise SystemExit("score args missing")
        score_shard(args.model_dir,args.out,args.start,args.end,args.shard)
    else:
        if args.root is None: raise SystemExit("--root required")
        aggregate(args.root,args.out)

if __name__=="__main__": main()
