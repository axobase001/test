from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from main_sequence import final_recent_replay as base

GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
NY = ZoneInfo("America/New_York")

# Frozen before opening 1H results.
TAIL_FAVORITE_FAIR = 0.99   # both anchors imply longshot below one 1c probability tick
FAIR_BAND = 0.01            # one standard probability tick
MARKET_CAP_USD = 200.0
TICKETS = (0.3125, 0.625, 1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 200.0)

PROTOCOL = {
    "name": "Main Sequence V4 hourly immediate-quote CORE+TAIL / 2026-08-16 freeze",
    "market": "Polymarket BTC Up/Down 1h",
    "reference": "Binance BTCUSDT 1H candle open/close",
    "execution": "same-second immediate top-level public quote witness; never require a future print at the old price",
    "fair": "conservative boundary from causal Binance RV and backward Deribit trade IV",
    "core": {
        "repeatable": True,
        "max_open_core_positions": 1,
        "entry": "fair-ask-entry_fee-estimated_exit_fee_at_fair > 0",
        "exit": "same-second sellable bid within 1c of current causal fair AND proceeds after fee > entry cost",
        "fallback": "settlement",
    },
    "tail": {
        "once_per_market": True,
        "favorite_fair_floor": TAIL_FAVORITE_FAIR,
        "interpretation": "both anchors value longshot below 1c",
        "entry": "positive post-entry-fee settlement edge",
        "exit": "settlement",
        "priority": "TAIL beats a simultaneous CORE entry and blocks all NEW CORE entries after TAIL fill",
    },
    "portfolio": {
        "initial_equity": 50.0,
        "base_ticket": 5.0,
        "power_of_two_current_equity_tiers": True,
        "max_combined_open_capital_per_market": MARKET_CAP_USD,
        "no_leverage": True,
    },
    "anti_lookahead": [
        "fair uses only anchor observations timestamped <= decision/exit second",
        "entry existence and size use only same-second public tape evidence",
        "a quote disappearing on the next second does not invalidate a same-second immediate order",
        "final outcome is read only for settlement of positions still open at market close",
        "TAIL cannot retroactively cancel a CORE opened earlier",
    ],
}


@dataclass(frozen=True)
class HourMarket:
    slug: str
    event_slug: str
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
        return self.start + 3600


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
        s = v.strip().lower()
        if s in ("true", "1", "yes"): return True
        if s in ("false", "0", "no"): return False
    return None


def get_json(sess: requests.Session, url: str, *, params=None, tries=7, timeout=60):
    last = None
    for i in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(min(0.4 * 2**i, 8.0))
    raise RuntimeError(f"GET failed {url} params={params}: {last!r}")


def hour_slug_candidates(start: int) -> list[str]:
    d = datetime.fromtimestamp(int(start), tz=timezone.utc).astimezone(NY)
    month = d.strftime("%B").lower()
    h = d.hour % 12 or 12
    ap = "am" if d.hour < 12 else "pm"
    # Polymarket has used both yearless and year-explicit event slugs.
    return [
        f"bitcoin-up-or-down-{month}-{d.day}-{h}{ap}-et",
        f"bitcoin-up-or-down-{month}-{d.day}-{d.year}-{h}{ap}-et",
        f"btc-updown-1h-{int(start)}",
    ]


def parse_event(event: dict, start: int, event_slug: str) -> HourMarket | None:
    for raw in event.get("markets") or []:
        outcomes = jsonish(raw.get("outcomes") or "[]", [])
        oi = {str(x).strip().lower(): i for i, x in enumerate(outcomes)}
        if "up" not in oi or "down" not in oi:
            continue
        prices = jsonish(raw.get("outcomePrices") or "[]", [])
        if len(prices) <= max(oi["up"], oi["down"]):
            continue
        try:
            pp = [float(x) for x in prices]
        except Exception:
            continue
        if max(pp) < 0.99:
            continue
        cid = str(raw.get("conditionId") or "")
        if not cid:
            continue
        enabled = boolish(raw.get("feesEnabled"))
        fs = jsonish(raw.get("feeSchedule"), {})
        try: rate = float(fs.get("rate")) if fs.get("rate") is not None else None
        except Exception: rate = None
        try: expo = float(fs.get("exponent")) if fs.get("exponent") is not None else None
        except Exception: expo = None
        if enabled is None:
            # Historical hourly crypto markets created before 2026-03-06 were fee-free;
            # after expansion use dated fallback when Gamma lacks an explicit flag.
            enabled = int(start) >= int(pd.Timestamp("2026-03-06", tz="UTC").timestamp())
            source = "dated_hourly_fallback"
        else:
            source = "gamma_feesEnabled"
        label = 1.0 if pp[oi["up"]] > pp[oi["down"]] else 0.0
        return HourMarket(
            slug=str(raw.get("slug") or event_slug), event_slug=event_slug, start=int(start), condition_id=cid,
            label_up=label, fee_enabled=bool(enabled), fee_type=str(raw.get("feeType") or ""),
            fee_rate=rate, fee_exponent=expo, fee_source=source,
        )
    return None


def fetch_hour_market(start: int) -> tuple[HourMarket | None, dict]:
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-v4-hourly/1.0"})
    errors = []
    for slug in hour_slug_candidates(start):
        try:
            js = get_json(sess, GAMMA + "/events", params={"slug": slug, "closed": "true", "limit": 5}, tries=4)
            if js:
                m = parse_event(js[0], int(start), slug)
                if m is not None:
                    return m, {"start": int(start), "event_slug": slug, "mapped": True, "market_slug": m.slug, "condition_id": m.condition_id}
        except Exception as exc:
            errors.append(repr(exc))
    return None, {"start": int(start), "event_slug": None, "mapped": False, "market_slug": None, "condition_id": None, "errors": " | ".join(errors[:3])}


def discover(start: str, end: str, workers: int = 16):
    s0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    s1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    starts = list(range(s0, s1, 3600))
    markets, inv = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fetch_hour_market, s): s for s in starts}
        for i, f in enumerate(as_completed(fut), 1):
            m, row = f.result(); inv.append(row)
            if m is not None: markets.append(m)
            if i % 120 == 0:
                print("DISCOVER_1H", i, "/", len(starts), "mapped", len(markets), flush=True)
    markets.sort(key=lambda x: x.start)
    return markets, pd.DataFrame(inv).sort_values("start", kind="mergesort")


def fee_total(m: HourMarket, p: float, qty: float) -> float:
    # Reuse historical fee chronology/formula from the frozen recent engine.
    return float(base.fee_total(m, float(p), float(qty)))


def fee_ps(m: HourMarket, p: float, qty: float) -> float:
    return fee_total(m, p, qty) / qty if qty > 0 else math.inf


def qty_for_budget(m: HourMarket, p: float, budget: float) -> float:
    if not (0 < p < 1 and budget > 0): return 0.0
    fps = fee_total(m, p, 1000.0) / 1000.0
    q = budget / max(p + fps, 1e-12)
    for _ in range(5):
        cost = q * p + fee_total(m, p, q)
        if cost <= 0: return 0.0
        q *= budget / cost
    cost = q * p + fee_total(m, p, q)
    if cost > budget: q *= budget / cost
    return max(float(q), 0.0)


def top_level(q: pd.DataFrame, action: str, outcome: str):
    z = q[(q["side_u"] == action) & (q["outcome_l"] == outcome) & (q["size"] > 0)].copy()
    if z.empty: return None
    if action == "BUY":
        px = float(z["price"].min())
    else:
        px = float(z["price"].max())
    size = float(z.loc[np.isclose(z["price"].astype(float), px, rtol=0, atol=1e-12), "size"].sum())
    return px, size


def fair_boundary(m: HourMarket, sec: int, spot, bn, der):
    if sec >= m.close: return None
    rv = bn.rv_annualized(sec * 1000, 60)
    div = der.median_iv(sec * 1000, 30)
    if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
        return None
    op = bn.open_price(m.start); sp = spot.at(sec)
    if not (op > 0 and sp > 0): return None
    tau = m.close - sec
    p_rv = base.original.digital_prob_up(sp / op, tau, rv)
    p_iv = base.original.digital_prob_up(sp / op, tau, div)
    if not (math.isfinite(p_rv) and math.isfinite(p_iv)): return None
    lo, hi = min(p_rv, p_iv), max(p_rv, p_iv)
    return {
        "up": float(lo), "down": float(1.0 - hi), "p_rv": float(p_rv), "p_iv": float(p_iv),
        "rv": float(rv), "iv": float(div), "spot": float(sp), "open": float(op),
    }


def market_tape(m: HourMarket) -> pd.DataFrame:
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-v4-hourly-data/1.0"})
    rows = base.query_trade_rows(sess, [m], m.start, m.close - 1)
    by = base.normalize_trades(rows)
    return by.get(m.condition_id, pd.DataFrame(columns=["timestamp", "side_u", "outcome_l", "price", "size"]))


def simulate_market_ticket(m: HourMarket, g: pd.DataFrame, spot, bn, der, ticket: float):
    if g is None or g.empty:
        return {"ticket": ticket, "pnl": 0.0, "turnover": 0.0, "core_pnl": 0.0, "tail_pnl": 0.0,
                "core_entries": 0, "core_round_trips": 0, "tail_entries": 0, "cap_rejects": 0,
                "max_open_capital": 0.0, "events": []}

    secs = sorted(int(x) for x in g["timestamp"].dropna().unique().tolist() if m.start <= int(x) < m.close)
    core = None
    tail = None
    tail_blocks_core = False
    core_reentry_after = m.start
    core_pnl = tail_pnl = turnover = 0.0
    core_entries = core_round_trips = tail_entries = cap_rejects = 0
    max_open_cap = 0.0
    events = []
    fair_cache = {}

    def open_capital():
        return (float(core["cost"]) if core else 0.0) + (float(tail["cost"]) if tail else 0.0)

    for sec in secs:
        fb = fair_cache.get(sec)
        if fb is None:
            fb = fair_boundary(m, sec, spot, bn, der); fair_cache[sec] = fb
        if fb is None: continue
        qsec = g[g["timestamp"] == sec]

        # 1) Existing CORE may exit first; capital becomes reusable immediately.
        if core is not None:
            lv = top_level(qsec, "SELL", core["outcome"])
            if lv is not None:
                bid, avail = lv
                qty = float(core["qty"])
                fair = float(fb[core["outcome"]])
                proceeds = qty * bid - fee_total(m, bid, qty)
                if avail + 1e-12 >= qty and bid >= fair - FAIR_BAND - 1e-12 and proceeds > float(core["cost"]) + 1e-12:
                    pnl = proceeds - float(core["cost"])
                    core_pnl += pnl; turnover += proceeds; core_round_trips += 1
                    events.append({"time": sec, "family": "core", "event": "convergence_exit", "outcome": core["outcome"], "pnl": pnl,
                                   "entry": core["price"], "exit": bid, "qty": qty})
                    core = None; core_reentry_after = sec + 1

        # 2) TAIL has priority over a simultaneous NEW CORE entry.
        if tail is None:
            tail_candidates = []
            for outcome in ("up", "down"):
                lv = top_level(qsec, "BUY", outcome)
                if lv is None: continue
                ask, avail = lv
                fair = float(fb[outcome])
                qty = qty_for_budget(m, ask, ticket)
                if qty <= 0 or avail + 1e-12 < qty: continue
                cost = qty * ask + fee_total(m, ask, qty)
                edge = fair - ask - fee_ps(m, ask, qty)
                if fair >= TAIL_FAVORITE_FAIR and edge > 0:
                    tail_candidates.append((edge, outcome, ask, avail, qty, cost, fair))
            if tail_candidates:
                edge, outcome, ask, avail, qty, cost, fair = max(tail_candidates, key=lambda x: x[0])
                if open_capital() + cost <= MARKET_CAP_USD + 1e-9:
                    tail = {"outcome": outcome, "price": ask, "qty": qty, "cost": cost, "entry_time": sec, "fair": fair}
                    tail_entries += 1; turnover += cost; tail_blocks_core = True
                    max_open_cap = max(max_open_cap, open_capital())
                    events.append({"time": sec, "family": "tail", "event": "entry", "outcome": outcome, "pnl": 0.0,
                                   "entry": ask, "qty": qty, "fair": fair, "edge": edge})
                else:
                    cap_rejects += 1

        # 3) Repeatable CORE if no CORE is open and TAIL has not blocked new CORE.
        if core is None and not tail_blocks_core and sec >= core_reentry_after:
            core_candidates = []
            for outcome in ("up", "down"):
                lv = top_level(qsec, "BUY", outcome)
                if lv is None: continue
                ask, avail = lv
                fair = float(fb[outcome])
                qty = qty_for_budget(m, ask, ticket)
                if qty <= 0 or avail + 1e-12 < qty: continue
                buy_fee = fee_ps(m, ask, qty)
                # Estimate taker exit cost at fair; this is only a hurdle, actual exit uses actual bid+fee.
                exit_fee = fee_ps(m, min(max(fair, 1e-6), 1-1e-6), qty)
                rt_edge = fair - ask - buy_fee - exit_fee
                if rt_edge > 0:
                    cost = qty * ask + fee_total(m, ask, qty)
                    core_candidates.append((rt_edge, outcome, ask, avail, qty, cost, fair))
            if core_candidates:
                edge, outcome, ask, avail, qty, cost, fair = max(core_candidates, key=lambda x: x[0])
                if open_capital() + cost <= MARKET_CAP_USD + 1e-9:
                    core = {"outcome": outcome, "price": ask, "qty": qty, "cost": cost, "entry_time": sec, "fair": fair}
                    core_entries += 1; turnover += cost
                    max_open_cap = max(max_open_cap, open_capital())
                    events.append({"time": sec, "family": "core", "event": "entry", "outcome": outcome, "pnl": 0.0,
                                   "entry": ask, "qty": qty, "fair": fair, "edge": edge})
                else:
                    cap_rejects += 1

    # Settlement fallback / TAIL settlement. Outcome is read only here.
    won_up = m.label_up >= 0.5
    if core is not None:
        won = won_up if core["outcome"] == "up" else (not won_up)
        payout = float(core["qty"]) if won else 0.0
        pnl = payout - float(core["cost"])
        core_pnl += pnl; core_round_trips += 1
        events.append({"time": m.close, "family": "core", "event": "settlement", "outcome": core["outcome"], "pnl": pnl,
                       "entry": core["price"], "exit": 1.0 if won else 0.0, "qty": core["qty"]})
        core = None
    if tail is not None:
        won = won_up if tail["outcome"] == "up" else (not won_up)
        payout = float(tail["qty"]) if won else 0.0
        pnl = payout - float(tail["cost"])
        tail_pnl += pnl
        events.append({"time": m.close, "family": "tail", "event": "settlement", "outcome": tail["outcome"], "pnl": pnl,
                       "entry": tail["price"], "exit": 1.0 if won else 0.0, "qty": tail["qty"]})
        tail = None

    return {
        "ticket": float(ticket), "pnl": float(core_pnl + tail_pnl), "turnover": float(turnover),
        "core_pnl": float(core_pnl), "tail_pnl": float(tail_pnl),
        "core_entries": int(core_entries), "core_round_trips": int(core_round_trips), "tail_entries": int(tail_entries),
        "cap_rejects": int(cap_rejects), "max_open_capital": float(max_open_cap), "events": events,
    }


def score_market(m: HourMarket, spot, bn, der):
    g = market_tape(m)
    out = []
    for ticket in TICKETS:
        r = simulate_market_ticket(m, g, spot, bn, der, float(ticket))
        out.append({"start": m.start, "close": m.close, "slug": m.slug, "event_slug": m.event_slug,
                    "condition_id": m.condition_id, "label_up": m.label_up, **r, "events_json": json.dumps(r["events"], separators=(",", ":"))})
    return out


def score_range(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = discover(start, end, workers=min(workers, 24))
    inv.to_csv(out / "inventory.csv", index=False)
    if not markets:
        raise RuntimeError("no hourly markets mapped")
    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(score_market, m, spot, bn, der): m for m in markets}
        for i, f in enumerate(as_completed(fut), 1):
            m = fut[f]
            try: rows.extend(f.result())
            except Exception as exc: failures.append({"slug": m.slug, "error": repr(exc)})
            if i % 24 == 0:
                print("SCORE_1H", i, "/", len(markets), "rows", len(rows), "fail", len(failures), flush=True)
    if failures:
        raise RuntimeError(f"hourly score failures {failures[:10]} count={len(failures)}")
    df = pd.DataFrame(rows).sort_values(["start", "ticket"], kind="mergesort")
    df.to_csv(out / "market_ticket_results.csv", index=False)
    summary = {"period": [start, end], "markets_expected": int(len(inv)), "markets_mapped": int(inv["mapped"].sum()),
               "markets_scored": int(df["start"].nunique()), "rows": int(len(df)), "anchor_meta": anchor_meta, "protocol": PROTOCOL}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def stake_for_equity(eq: float):
    if not (eq > 0 and math.isfinite(eq)): return None
    exp = int(math.floor(math.log(eq / 50.0, 2.0)))
    nominal = 5.0 * (2.0 ** exp)
    if nominal < min(TICKETS): return None
    if nominal >= MARKET_CAP_USD: return MARKET_CAP_USD
    # Tickets include all prior power-of-two tiers.
    return float(nominal)


def aggregate(root: Path, out: Path, start: str, end: str):
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("market_ticket_results.csv"))
    if not files: raise RuntimeError("no market_ticket_results.csv")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True).drop_duplicates(["start", "ticket"], keep="last")
    s0 = int(pd.Timestamp(start, tz="UTC").timestamp()); s1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    df = df[(df["start"] >= s0) & (df["start"] < s1)].sort_values(["start", "ticket"], kind="mergesort")
    eq = 50.0; peak = eq; maxdd = 0.0; turnover = core_pnl = tail_pnl = 0.0
    core_entries = core_round_trips = tail_entries = cap_rejects = 0
    paths = [{"time": s0, "equity": eq, "event": "start"}]
    selected = []
    for start_ts, g in df.groupby("start", sort=True):
        ticket = stake_for_equity(eq)
        if ticket is None: break
        z = g[np.isclose(g["ticket"].astype(float), ticket, rtol=0, atol=1e-10)]
        if z.empty: continue
        r = z.iloc[0]
        selected.append(r)
        turnover += float(r.turnover); core_pnl += float(r.core_pnl); tail_pnl += float(r.tail_pnl)
        core_entries += int(r.core_entries); core_round_trips += int(r.core_round_trips); tail_entries += int(r.tail_entries); cap_rejects += int(r.cap_rejects)
        events = json.loads(r.events_json) if isinstance(r.events_json, str) else []
        for ev in sorted(events, key=lambda x: (int(x["time"]), 0 if x["event"] != "entry" else 1)):
            pnl = float(ev.get("pnl", 0.0))
            if pnl != 0:
                eq += pnl; peak = max(peak, eq); maxdd = min(maxdd, eq / peak - 1.0)
                paths.append({"time": int(ev["time"]), "equity": eq, "event": f"{ev['family']}:{ev['event']}"})
    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    ret = eq / 50.0 - 1.0
    cagr = (eq / 50.0) ** (365.0 / days) - 1.0 if eq > 0 else math.nan
    summary = {
        "period": [start, end], "days": days, "initial_equity": 50.0, "final_equity": eq,
        "total_return": ret, "calendar_cagr": cagr, "max_dd": maxdd, "turnover": turnover,
        "core_pnl": core_pnl, "tail_pnl": tail_pnl, "core_entries": core_entries,
        "core_round_trips": core_round_trips, "tail_entries": tail_entries, "cap_rejects": cap_rejects,
        "markets_selected": len(selected), "protocol": PROTOCOL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(paths).to_csv(out / "equity_path.csv", index=False)
    if selected: pd.DataFrame(selected).to_csv(out / "selected_market_results.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("protocol"); p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--out", type=Path, required=True); s.add_argument("--workers", type=int, default=8)
    a = sub.add_parser("aggregate"); a.add_argument("--root", type=Path, required=True); a.add_argument("--out", type=Path, required=True); a.add_argument("--start", required=True); a.add_argument("--end", required=True)
    args = ap.parse_args()
    if args.cmd == "protocol":
        args.out.mkdir(parents=True, exist_ok=True); (args.out / "FROZEN_V4_1H_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2)); print(json.dumps(PROTOCOL, indent=2))
    elif args.cmd == "score": score_range(args.start, args.end, args.out, args.workers)
    else: aggregate(args.root, args.out, args.start, args.end)

if __name__ == "__main__": main()
