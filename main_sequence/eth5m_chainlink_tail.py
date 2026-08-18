from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from scipy.special import ndtr

HF_REPO = "Alezanello/polymarket-arena-capture"
START = "2026-06-04"
END = "2026-07-15"  # end-exclusive; frozen capture coverage used by prior Main Sequence B
ASSET = "ETH"
DECISION_S2C = 90
MAX_BOOK_STALENESS_MS = 4_000
TICKET = 5.0
FEE_RATE = 0.07
FAIR_FLOORS = (0.95, 0.97, 0.98, 0.99)
BARRIER_BPS_FLOORS = (0.0, 2.0, 5.0, 10.0, 20.0)
YEAR_SECONDS = 365.0 * 24.0 * 3600.0


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

    def at(self, t_ms: int, tolerance_ms: int = 20_000) -> tuple[float, int]:
        i = int(np.searchsorted(self.ts, t_ms, side="right") - 1)
        if i < 0 or t_ms - int(self.ts[i]) > tolerance_ms:
            return float("nan"), -1
        return float(self.px[i]), int(self.ts[i])

    def rv(self, t_ms: int, minutes: int = 60, min_obs: int = 30) -> float:
        hi = int(np.searchsorted(self.minute_ts, t_ms, side="right"))
        lo = int(np.searchsorted(self.minute_ts, t_ms - minutes * 60_000, side="left"))
        vals = self.minute_px[lo:hi]
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
    WITH p AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms, CAST(value AS DOUBLE) AS value
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{ASSET}' AND lower(src)='chainlink'
        AND ts_ms >= {lo} AND ts_ms <= {hi} AND value > 0
    ), b AS (
      SELECT CAST(floor(ts_ms/5000)*5000 AS BIGINT) AS bar_ms,
             arg_max(value, ts_ms) AS value, max(ts_ms) AS source_ts
      FROM p GROUP BY bar_ms
    )
    SELECT * FROM b ORDER BY bar_ms
    """
    df = con.execute(q).fetchdf(); con.close()
    if len(df) < 100:
        raise RuntimeError(f"insufficient Chainlink ETH observations: {len(df)}")
    ts = df.bar_ms.to_numpy(np.int64); px = df.value.to_numpy(float)
    mins = (ts // 60_000) * 60_000
    tmp = pd.DataFrame({"m": mins, "ts": ts, "px": px}).groupby("m", sort=True).tail(1)
    return PriceSeries(ts, px, tmp.m.to_numpy(np.int64), tmp.px.to_numpy(float))


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
        AND best_ask > 0 AND best_ask < 1 AND ask_sz > 0
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY slug, outcome ORDER BY ts_ms DESC) AS rn
      FROM src
    )
    SELECT * FROM ranked WHERE rn=1 ORDER BY end_ts, slug, outcome
    """
    df = con.execute(q).fetchdf(); con.close()
    return df


def one_market_rows(book: pd.DataFrame, ps: PriceSeries) -> pd.DataFrame:
    rows = []
    for slug, g in book.groupby("slug", sort=False):
        first = g.iloc[0]
        win_start = int(first.win_start); end_ts = int(first.end_ts)
        target_ms = end_ts * 1000 - DECISION_S2C * 1000
        open_px, open_src = ps.at(win_start * 1000, 30_000)
        close_px, close_src = ps.at(end_ts * 1000, 30_000)
        if not (open_px > 0 and close_px > 0):
            continue
        settle_up = close_px >= open_px
        candidates = []
        for rr in g.itertuples(index=False):
            outcome = str(rr.outcome).lower()
            if not (outcome.startswith("up") or outcome.startswith("down")):
                continue
            decision_ts = int(rr.ts_ms)
            book_staleness_ms = int(target_ms - decision_ts)
            if book_staleness_ms < 0 or book_staleness_ms > MAX_BOOK_STALENESS_MS:
                continue
            spot, spot_src = ps.at(decision_ts, 20_000)
            rv60 = ps.rv(decision_ts)
            if not (spot > 0 and math.isfinite(rv60) and 0.02 < rv60 < 5.0):
                continue
            tau = max((end_ts * 1000 - decision_ts) / 1000.0, 1.0)
            p_up = digital_prob_up(spot / open_px, tau, rv60)
            if not math.isfinite(p_up):
                continue
            side = "up" if outcome.startswith("up") else "down"
            fair = p_up if side == "up" else 1.0 - p_up
            # Only the contemporaneous favorite can become a TAIL candidate.
            if fair < 0.5:
                continue
            ask = float(rr.best_ask); ask_sz = float(rr.ask_sz)
            q = qty_for_budget(ask); cost = q * (ask + fee_ps(ask))
            barrier_bps = abs(spot / open_px - 1.0) * 10000.0
            sigma_distance = abs(math.log(spot / open_px)) / max(rv60 * math.sqrt(tau / YEAR_SECONDS), 1e-12)
            won = settle_up if side == "up" else not settle_up
            candidates.append({
                "slug": slug, "cond": str(rr.cond), "win_start": win_start, "end_ts": end_ts,
                "decision_ts_ms": decision_ts, "decision_s2c_s": tau, "book_staleness_ms": book_staleness_ms,
                "side": side, "fair": fair, "p_up": p_up,
                "ask": ask, "ask_sz": ask_sz, "qty_fixed5": q, "cost": cost,
                "depth_headroom_x": ask_sz / max(q, 1e-12),
                "net_settlement_edge_ps": fair - ask - fee_ps(ask),
                "barrier_bps": barrier_bps, "sigma_distance": sigma_distance,
                "rv60": rv60, "threshold_chainlink": open_px, "spot_chainlink": spot,
                "close_chainlink": close_px, "settle_up": bool(settle_up), "won": bool(won),
                "pnl_fixed5": (q if won else 0.0) - cost,
                "open_source_ts": open_src, "spot_source_ts": spot_src, "close_source_ts": close_src,
            })
        if candidates:
            # Ex-ante only: choose the strongest contemporaneous settlement edge; never use the label/PnL.
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
            rows.append({
                "fair_floor":fair_floor,"barrier_bps_floor":barrier_floor,"n":int(len(z)),
                "days":int(day.nunique()),"trades_per_day":float(len(z)/max(day.nunique(),1)),
                "win_rate":float(z.won.mean()),"mean_fair":float(z.fair.mean()),
                "mean_ask":float(z.ask.mean()),"mean_exante_edge_c":float(z.net_settlement_edge_ps.mean()*100),
                "median_barrier_bps":float(z.barrier_bps.median()),
                "median_sigma_distance":float(z.sigma_distance.median()),
                "median_book_staleness_ms":float(z.book_staleness_ms.median()),
                "median_depth_headroom_x":float(z.depth_headroom_x.median()),
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
    raw = one_market_rows(book, ps)
    raw.to_csv(out/"market_rows.csv", index=False)
    surf = surface(raw); surf.to_csv(out/"surface.csv", index=False)
    manifest = {
        "repo":HF_REPO,"revision":revision,"period":[START,END],"asset":ASSET,
        "rules":{
            "settlement_only":True,"decision_s2c_target":DECISION_S2C,"max_book_staleness_ms":MAX_BOOK_STALENESS_MS,"ticket":TICKET,
            "fair_floors":list(FAIR_FLOORS),"barrier_bps_floors":list(BARRIER_BPS_FLOORS),
            "reference":"Chainlink ETH price stream; threshold is window-start Chainlink price",
            "fair":"digital N(d2) using each chosen outcome snapshot's own timestamp, Chainlink spot/threshold, and causal Chainlink 60m realized volatility",
            "execution":"actual captured best ask + best-level ask size; full fixed-$5 qty required at that ask",
            "capture_clock_note":"cap_book ts_ms is collector capture time (~2s cadence/token), not exchange event time",
            "fee":"0.07*p*(1-p)",
            "barrier":"absolute Chainlink spot-to-threshold distance in bps; full predeclared sensitivity surface, no post-hoc threshold selection",
            "anti_lookahead":"each outcome ask is valued only with Chainlink/RV state timestamped at or before that same outcome's own captured book timestamp; final label is used only for settlement PnL",
        },
        "files":{
            "cap_book":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in book_paths],
            "cap_prices":[{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in price_paths],
        },
        "markets_in_decision_book":int(book.slug.nunique()) if len(book) else 0,
        "markets_with_causal_fresh_candidate":int(len(raw)),
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("ETH5M_TAIL_SURFACE")
    print(surf.to_string(index=False))
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__":
    main()
