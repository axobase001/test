from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from scipy.special import ndtr

HF_REPO = "Alezanello/polymarket-arena-capture"
GAMMA = "https://gamma-api.polymarket.com"
START = "2026-06-04"
END = "2026-07-15"  # end-exclusive; frozen capture coverage used by prior Main Sequence B
ASSET = "ETH"
DECISION_S2C = 90
MAX_BOOK_STALENESS_MS = 4_000
MAX_CHAINLINK_LAG_MS = 5_000
TICKET = 5.0
FEE_RATE = 0.07
FAIR_FLOORS = (0.95, 0.97, 0.98, 0.99)
BARRIER_BPS_FLOORS = (0.0, 2.0, 5.0, 10.0, 20.0)
YEAR_SECONDS = 365.0 * 24.0 * 3600.0
RESOLUTION_BATCH = 50


def utc_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_selected_files(revision: str, table: str, start: str, end: str, cache: Path) -> list[Path]:
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    selected: list[str] = []
    root = f"{table}.parquet"
    if s <= date(2026, 6, 15) and e > date(2026, 6, 4) and root in files:
        selected.append(root)
    for f in files:
        if not (f.startswith(f"daily/{table}/") and f.endswith(".parquet")):
            continue
        m = re.search(r"(20\d\d-\d\d-\d\d)", f)
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        if s <= d < e:
            selected.append(f)
    if not selected:
        raise RuntimeError(f"no {table} files for {start}..{end} at {revision}")
    return [Path(hf_hub_download(HF_REPO, f, repo_type="dataset", revision=revision,
                                 cache_dir=str(cache / "hf"))) for f in sorted(set(selected))]


def sql_files(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def fee_ps(p: float) -> float:
    return FEE_RATE * p * (1.0 - p)


def qty_for_budget(p: float) -> float:
    return TICKET / max(p + fee_ps(p), 1e-12)


def digital_prob_up(rel_spot: float, seconds: float, sigma: float) -> float:
    if not (rel_spot > 0 and seconds > 0 and sigma > 0 and math.isfinite(sigma)):
        return float("nan")
    t = seconds / YEAR_SECONDS
    den = sigma * math.sqrt(t)
    if den <= 0:
        return float(rel_spot >= 1.0)
    d2 = (math.log(rel_spot) - 0.5 * sigma * sigma * t) / den
    return float(ndtr(d2))


@dataclass
class PriceSeries:
    ts: np.ndarray
    px: np.ndarray
    minute_ts: np.ndarray
    minute_px: np.ndarray

    def at(self, t_ms: int, tolerance_ms: int = MAX_CHAINLINK_LAG_MS) -> tuple[float, int]:
        i = int(np.searchsorted(self.ts, t_ms, side="right") - 1)
        if i < 0 or t_ms - int(self.ts[i]) > tolerance_ms:
            return float("nan"), -1
        return float(self.px[i]), int(self.ts[i])

    def rv(self, t_ms: int, minutes: int = 60, min_obs: int = 30) -> float:
        hi = int(np.searchsorted(self.minute_ts, t_ms, side="right"))
        lo = int(np.searchsorted(self.minute_ts, t_ms - minutes * 60_000, side="left"))
        vals = self.minute_px[lo:hi].astype(float, copy=True)
        mts = self.minute_ts[lo:hi]
        cur_px, cur_ts = self.at(t_ms, tolerance_ms=MAX_CHAINLINK_LAG_MS)
        if cur_ts >= 0 and cur_px > 0:
            cur_min = cur_ts // 60_000
            if len(mts) == 0 or int(mts[-1]) // 60_000 != cur_min:
                vals = np.append(vals, cur_px)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < min_obs + 1:
            return float("nan")
        r = np.diff(np.log(vals))
        if r.size < min_obs:
            return float("nan")
        return float(np.std(r, ddof=1)) * math.sqrt(365.0 * 24.0 * 60.0)


def load_chainlink(paths: list[Path]) -> PriceSeries:
    con = duckdb.connect()
    lo = utc_ms(START) - 75 * 60_000
    hi = utc_ms(END) + 30_000
    q = f"""
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms, CAST(value AS DOUBLE) AS value
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{ASSET}' AND lower(src)='chainlink'
        AND ts_ms >= {lo} AND ts_ms <= {hi} AND value > 0
      ORDER BY ts_ms
    """
    df = con.execute(q).fetchdf(); con.close()
    if len(df) < 100:
        raise RuntimeError(f"insufficient Chainlink ETH observations: {len(df)}")
    df = df.drop_duplicates("ts_ms", keep="last").sort_values("ts_ms", kind="mergesort")
    ts = df.ts_ms.to_numpy(np.int64)
    px = df.value.to_numpy(float)
    minute_id = ts // 60_000
    tmp = pd.DataFrame({"minute_id": minute_id, "ts": ts, "px": px}).groupby("minute_id", sort=True).tail(1)
    return PriceSeries(ts, px, tmp.ts.to_numpy(np.int64), tmp.px.to_numpy(float))


def load_decision_books(paths: list[Path]) -> pd.DataFrame:
    con = duckdb.connect()
    lo, hi = utc_ms(START), utc_ms(END)
    target_offset = DECISION_S2C * 1000
    q = f"""
    WITH src AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms,
             CAST(best_bid AS DOUBLE) AS best_bid,
             CAST(best_ask AS DOUBLE) AS best_ask,
             CAST(bid_sz AS DOUBLE) AS bid_sz,
             CAST(ask_sz AS DOUBLE) AS ask_sz,
             lower(outcome) AS outcome, slug, CAST(cond AS VARCHAR) AS cond,
             CAST(win_start AS BIGINT) AS win_start, CAST(end_ts AS BIGINT) AS end_ts
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{ASSET}' AND lower(slug) LIKE '%-updown-5m-%'
        AND end_ts*1000 >= {lo} AND end_ts*1000 < {hi}
        AND ts_ms <= end_ts*1000 - {target_offset}
        AND ts_ms >= end_ts*1000 - {target_offset + 7000}
        AND best_ask > 0 AND best_ask < 1
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY slug, outcome ORDER BY ts_ms DESC) AS rn
      FROM src
    )
    SELECT * FROM ranked
    WHERE rn=1 AND ask_sz IS NOT NULL AND ask_sz > 0
    ORDER BY end_ts, slug, outcome
    """
    df = con.execute(q).fetchdf(); con.close()
    return df


def parse_resolution(raw: dict) -> bool | None:
    if not bool(raw.get("closed")):
        return None
    try:
        outcomes = raw.get("outcomes")
        prices = raw.get("outcomePrices")
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        if isinstance(prices, str): prices = json.loads(prices)
        oi = {str(x).strip().lower(): i for i, x in enumerate(outcomes or [])}
        if "up" not in oi or "down" not in oi:
            return None
        pp = [float(x) for x in prices]
        ui, di = oi["up"], oi["down"]
        if len(pp) <= max(ui, di) or max(pp[ui], pp[di]) < 0.99 or min(pp[ui], pp[di]) > 0.01:
            return None
        return bool(pp[ui] > pp[di])
    except Exception:
        return None


def gamma_get(sess: requests.Session, params, tries: int = 6):
    last = None
    for i in range(tries):
        try:
            r = sess.get(GAMMA + "/markets", params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:120]}")
            r.raise_for_status()
            out = r.json()
            if not isinstance(out, list):
                raise RuntimeError(f"unexpected Gamma response {type(out)}")
            return out
        except Exception as exc:
            last = exc
            time.sleep(min(0.5 * (2 ** i), 8.0))
    raise RuntimeError(f"Gamma resolution fetch failed: {last!r}")


def fetch_gamma_resolutions(book: pd.DataFrame) -> tuple[dict[str, bool], pd.DataFrame]:
    ids = sorted(book["cond"].dropna().astype(str).unique().tolist()) if len(book) else []
    sess = requests.Session(); sess.headers.update({"User-Agent": "main-sequence-eth5m-resolution/1.0"})
    resolved: dict[str, bool] = {}
    audit: dict[str, dict] = {}
    for off in range(0, len(ids), RESOLUTION_BATCH):
        chunk = ids[off:off + RESOLUTION_BATCH]
        params = [("condition_ids", x) for x in chunk] + [("closed", "true"), ("limit", len(chunk))]
        rows = gamma_get(sess, params)
        for raw in rows:
            cid = str(raw.get("conditionId") or "")
            if cid not in chunk:
                continue
            up = parse_resolution(raw)
            audit[cid] = {
                "cond": cid, "slug_gamma": str(raw.get("slug") or ""),
                "closed": bool(raw.get("closed")), "gamma_up": up,
                "outcomes": raw.get("outcomes"), "outcomePrices": raw.get("outcomePrices"),
            }
            if up is not None:
                resolved[cid] = up
        time.sleep(0.05)

    # Batch response must never silently create labels for missing markets.
    missing = [x for x in ids if x not in audit]
    for cid in missing:
        rows = gamma_get(sess, [("condition_ids", cid), ("closed", "true"), ("limit", 1)])
        raw = next((x for x in rows if str(x.get("conditionId") or "") == cid), None)
        up = parse_resolution(raw) if raw is not None else None
        audit[cid] = {
            "cond": cid, "slug_gamma": str(raw.get("slug") or "") if raw else "",
            "closed": bool(raw.get("closed")) if raw else False, "gamma_up": up,
            "outcomes": raw.get("outcomes") if raw else None,
            "outcomePrices": raw.get("outcomePrices") if raw else None,
        }
        if up is not None:
            resolved[cid] = up
        time.sleep(0.05)

    adf = pd.DataFrame(list(audit.values())) if audit else pd.DataFrame(columns=["cond","slug_gamma","closed","gamma_up","outcomes","outcomePrices"])
    return resolved, adf.sort_values("cond", kind="mergesort") if len(adf) else adf


def one_market_rows(book: pd.DataFrame, ps: PriceSeries, resolutions: dict[str, bool]) -> pd.DataFrame:
    rows = []
    for slug, g in book.groupby("slug", sort=False):
        first = g.iloc[0]
        cond = str(first.cond)
        if cond not in resolutions:
            continue
        win_start = int(first.win_start); end_ts = int(first.end_ts)
        target_ms = end_ts * 1000 - DECISION_S2C * 1000
        open_target_ms = win_start * 1000
        open_px, open_src = ps.at(open_target_ms, MAX_CHAINLINK_LAG_MS)
        if not (open_px > 0 and 0 <= open_target_ms - open_src <= MAX_CHAINLINK_LAG_MS):
            continue
        close_px, close_src = ps.at(end_ts * 1000, 10_000)  # diagnostic only; never determines PnL label
        settle_up = bool(resolutions[cond])
        captured_chainlink_up = bool(close_px >= open_px) if close_px > 0 else None
        candidates = []
        for rr in g.itertuples(index=False):
            outcome = str(rr.outcome).lower()
            if not (outcome.startswith("up") or outcome.startswith("down")):
                continue
            decision_ts = int(rr.ts_ms)
            book_staleness_ms = int(target_ms - decision_ts)
            if book_staleness_ms < 0 or book_staleness_ms > MAX_BOOK_STALENESS_MS:
                continue
            spot, spot_src = ps.at(decision_ts, MAX_CHAINLINK_LAG_MS)
            rv60 = ps.rv(decision_ts)
            if not (spot > 0 and 0 <= decision_ts - spot_src <= MAX_CHAINLINK_LAG_MS and math.isfinite(rv60) and 0.02 < rv60 < 5.0):
                continue
            tau = max((end_ts * 1000 - decision_ts) / 1000.0, 1.0)
            p_up = digital_prob_up(spot / open_px, tau, rv60)
            if not math.isfinite(p_up):
                continue
            side = "up" if outcome.startswith("up") else "down"
            fair = p_up if side == "up" else 1.0 - p_up
            if fair < 0.5:
                continue
            ask = float(rr.best_ask); ask_sz = float(rr.ask_sz)
            q = qty_for_budget(ask); cost = q * (ask + fee_ps(ask))
            barrier_bps = abs(spot / open_px - 1.0) * 10000.0
            sigma_distance = abs(math.log(spot / open_px)) / max(rv60 * math.sqrt(tau / YEAR_SECONDS), 1e-12)
            won = settle_up if side == "up" else not settle_up
            mismatch = (captured_chainlink_up != settle_up) if captured_chainlink_up is not None else None
            candidates.append({
                "slug": slug, "cond": cond, "win_start": win_start, "end_ts": end_ts,
                "decision_ts_ms": decision_ts, "decision_s2c_s": tau, "book_staleness_ms": book_staleness_ms,
                "side": side, "fair": fair, "p_up": p_up,
                "ask": ask, "ask_sz": ask_sz, "qty_fixed5": q, "cost": cost,
                "depth_headroom_x": ask_sz / max(q, 1e-12),
                "net_settlement_edge_ps": fair - ask - fee_ps(ask),
                "barrier_bps": barrier_bps, "sigma_distance": sigma_distance,
                "rv60": rv60, "threshold_chainlink": open_px, "spot_chainlink": spot,
                "close_chainlink_diagnostic": close_px,
                "gamma_settle_up": bool(settle_up), "captured_chainlink_reconstructed_up": captured_chainlink_up,
                "gamma_vs_captured_chainlink_mismatch": mismatch, "won": bool(won),
                "pnl_fixed5": (q if won else 0.0) - cost,
                "open_source_ts": open_src, "spot_source_ts": spot_src, "close_source_ts": close_src,
                "threshold_source_lag_ms": int(open_target_ms - open_src),
                "spot_source_lag_ms": int(decision_ts - spot_src),
            })
        if candidates:
            rows.append(max(candidates, key=lambda x: (x["net_settlement_edge_ps"], x["fair"])))
    return pd.DataFrame(rows).sort_values("decision_ts_ms", kind="mergesort") if rows else pd.DataFrame()


def surface(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fair_floor in FAIR_FLOORS:
        for barrier_floor in BARRIER_BPS_FLOORS:
            z = raw[(raw.fair >= fair_floor) &
                    (raw.barrier_bps >= barrier_floor) &
                    (raw.net_settlement_edge_ps > 0) &
                    (raw.depth_headroom_x >= 1.0)].copy()
            if z.empty:
                rows.append({"fair_floor":fair_floor,"barrier_bps_floor":barrier_floor,"n":0})
                continue
            eq = 50.0 + z.pnl_fixed5.cumsum(); peak = eq.cummax(); mdd = float((eq/peak-1.0).min())
            day = pd.to_datetime(z.decision_ts_ms, unit="ms", utc=True).dt.floor("D")
            mismatch = z["gamma_vs_captured_chainlink_mismatch"].dropna()
            rows.append({
                "fair_floor":fair_floor,"barrier_bps_floor":barrier_floor,"n":int(len(z)),
                "days":int(day.nunique()),"trades_per_day":float(len(z)/max(day.nunique(),1)),
                "win_rate":float(z.won.mean()),"mean_fair":float(z.fair.mean()),
                "mean_ask":float(z.ask.mean()),"mean_exante_edge_c":float(z.net_settlement_edge_ps.mean()*100),
                "median_barrier_bps":float(z.barrier_bps.median()),
                "median_sigma_distance":float(z.sigma_distance.median()),
                "median_book_staleness_ms":float(z.book_staleness_ms.median()),
                "median_threshold_source_lag_ms":float(z.threshold_source_lag_ms.median()),
                "median_spot_source_lag_ms":float(z.spot_source_lag_ms.median()),
                "median_depth_headroom_x":float(z.depth_headroom_x.median()),
                "captured_chainlink_vs_gamma_mismatch_rate":float(mismatch.astype(bool).mean()) if len(mismatch) else None,
                "fixed5_pnl":float(z.pnl_fixed5.sum()),"fixed5_final":float(50.0+z.pnl_fixed5.sum()),
                "fixed5_mdd_pct":float(mdd*100.0),
            })
    return pd.DataFrame(rows)


def main():
    out = Path("eth5m_tail_out"); out.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(HF_REPO).sha
    book_paths = list_selected_files(revision, "cap_book", START, END, out)
    price_paths = list_selected_files(revision, "cap_prices", START, END, out)
    ps = load_chainlink(price_paths)
    book = load_decision_books(book_paths)
    resolutions, resolution_audit = fetch_gamma_resolutions(book)
    resolution_audit.to_csv(out/"gamma_resolutions.csv", index=False)
    raw = one_market_rows(book, ps, resolutions)
    raw.to_csv(out/"market_rows.csv", index=False)
    surf = surface(raw); surf.to_csv(out/"surface.csv", index=False)
    expected_resolution_ids = int(book["cond"].nunique()) if len(book) else 0
    manifest = {
        "repo":HF_REPO,"revision":revision,"period":[START,END],"asset":ASSET,
        "rules":{
            "settlement_only":True,"decision_s2c_target":DECISION_S2C,
            "max_book_staleness_ms":MAX_BOOK_STALENESS_MS,"max_chainlink_source_lag_ms":MAX_CHAINLINK_LAG_MS,
            "ticket":TICKET,"fair_floors":list(FAIR_FLOORS),"barrier_bps_floors":list(BARRIER_BPS_FLOORS),
            "reference":"Chainlink ETH price stream; threshold is window-start Chainlink price",
            "fair":"digital N(d2) using each chosen outcome snapshot's own timestamp, raw-tick-causal Chainlink spot/threshold, and causal Chainlink 60m realized volatility",
            "settlement_label":"official closed Gamma market outcomePrices; captured Chainlink close is diagnostic only and never determines PnL",
            "execution":"latest captured book row at/before T-90 must itself contain best ask + best-level ask size; no borrowing depth from an older row after a newer price-only update; full fixed-$5 qty required at that ask",
            "capture_clock_note":"cap_book ts_ms is collector capture time (~2s cadence/token), not exchange event time",
            "chainlink_clock_note":"raw cap_prices Chainlink ts_ms is used directly; no 5s bucket label is allowed to stand in for a later tick",
            "fee":"0.07*p*(1-p)",
            "barrier":"absolute Chainlink spot-to-threshold distance in bps; full predeclared sensitivity surface, no post-hoc threshold selection",
            "anti_lookahead":"each outcome ask is valued only with Chainlink/RV state timestamped at or before that same outcome's own captured book timestamp; final Gamma outcome is read only for settlement PnL",
        },
        "files":{
            "cap_book":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in book_paths],
            "cap_prices":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in price_paths],
        },
        "markets_in_decision_book":int(book.slug.nunique()) if len(book) else 0,
        "resolution_condition_ids_expected":expected_resolution_ids,
        "resolution_condition_ids_gamma_terminal":int(len(resolutions)),
        "resolution_condition_ids_rejected":int(expected_resolution_ids-len(resolutions)),
        "markets_with_causal_fresh_candidate":int(len(raw)),
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("ETH5M_TAIL_SURFACE")
    print(surf.to_string(index=False))
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__":
    main()
