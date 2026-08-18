from __future__ import annotations

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

from main_sequence import eth5m_chainlink_tail as runner
from main_sequence.qualification_stats import daily_pnl_stats


_SELECTED_PATHS: dict[str, list[Path]] = {}
_DUAL_CLOCK_AUDIT: dict = {}
MIN_CLOCK_PARSE_RATE_FOR_GREEN = 0.995
MAX_CAPTURE_SOURCE_DELTA_MS = 5_000


def _normalize_epoch_ms(x) -> int | None:
    if x in (None, ""):
        return None
    try:
        v = int(float(x))
    except Exception:
        return None
    a = abs(v)
    if a > 10**17:       # ns -> ms
        v //= 1_000_000
    elif a > 10**14:     # us -> ms
        v //= 1_000
    elif 10**8 < a < 10**11:  # s -> ms
        v *= 1_000
    return int(v)


def _source_ts_from_full(raw) -> int | None:
    """Extract Chainlink's payload/source timestamp from the lossless RTDS JSON.

    `ts_ms` in the Alezanello archive is collector capture time. Polymarket RTDS
    Chainlink updates contain a separate payload timestamp. If `full` stores the
    complete event we use `payload.timestamp`; if it stores the payload object
    alone we accept its timestamp only when symbol/value fields identify it as
    a price payload. We never substitute the outer event timestamp as source time.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    try:
        obj = json.loads(str(raw))
        if isinstance(obj, str):
            obj = json.loads(obj)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    payload = obj.get("payload")
    if isinstance(payload, dict):
        return _normalize_epoch_ms(payload.get("timestamp"))
    if "symbol" in obj and "value" in obj:
        return _normalize_epoch_ms(obj.get("timestamp"))
    return None


@dataclass
class DualClockPriceSeries:
    capture_ts: np.ndarray
    source_ts: np.ndarray
    px: np.ndarray

    def latest_available(
        self,
        available_by_ms: int,
        max_capture_lag_ms: int = runner.MAX_CHAINLINK_LAG_MS,
        max_source_age_ms: int = runner.MAX_CHAINLINK_LAG_MS,
    ) -> tuple[float, int, int]:
        """Latest source-time tick that was actually captured by `available_by_ms`."""
        t = int(available_by_ms)
        hi = int(np.searchsorted(self.capture_ts, t, side="right"))
        lo = int(np.searchsorted(self.capture_ts, t - max_capture_lag_ms, side="left"))
        if hi <= lo:
            return float("nan"), -1, -1
        cap = self.capture_ts[lo:hi]
        src = self.source_ts[lo:hi]
        px = self.px[lo:hi]
        mask = (
            (src <= t)
            & (src >= t - max_source_age_ms)
            & (np.abs(cap - src) <= MAX_CAPTURE_SOURCE_DELTA_MS)
            & np.isfinite(px)
            & (px > 0)
        )
        idx = np.flatnonzero(mask)
        if not len(idx):
            return float("nan"), -1, -1
        # Prefer the latest oracle/source timestamp; break ties by latest capture.
        cand_src = src[idx]
        best_src = np.max(cand_src)
        same = idx[cand_src == best_src]
        j = int(same[np.argmax(cap[same])])
        return float(px[j]), int(src[j]), int(cap[j])

    def at_source_time(
        self,
        target_source_ms: int,
        available_by_ms: int,
        max_source_staleness_ms: int = runner.MAX_CHAINLINK_LAG_MS,
    ) -> tuple[float, int, int]:
        """Reconstruct an oracle-time value using only messages known by decision time.

        This is the crucial PTB clock: a source tick belonging to T=0 may arrive
        milliseconds after T=0 and is still causal for a T-90s trading decision.
        """
        target = int(target_source_ms)
        available = int(available_by_ms)
        hi = int(np.searchsorted(self.capture_ts, available, side="right"))
        if hi <= 0:
            return float("nan"), -1, -1
        # Source time, not capture time, defines the requested oracle boundary.
        src = self.source_ts[:hi]
        cap = self.capture_ts[:hi]
        px = self.px[:hi]
        mask = (
            (src <= target)
            & (src >= target - max_source_staleness_ms)
            & (np.abs(cap - src) <= MAX_CAPTURE_SOURCE_DELTA_MS)
            & np.isfinite(px)
            & (px > 0)
        )
        idx = np.flatnonzero(mask)
        if not len(idx):
            return float("nan"), -1, -1
        cand_src = src[idx]
        best_src = np.max(cand_src)
        same = idx[cand_src == best_src]
        j = int(same[np.argmax(cap[same])])
        return float(px[j]), int(src[j]), int(cap[j])

    def rv(self, available_by_ms: int, source_target_ms: int | None = None, minutes: int = 60, min_obs: int = 30) -> float:
        available = int(available_by_ms)
        target = available if source_target_ms is None else int(source_target_ms)
        hi = int(np.searchsorted(self.capture_ts, available, side="right"))
        lo_cap = available - (minutes * 60_000 + MAX_CAPTURE_SOURCE_DELTA_MS + runner.MAX_CHAINLINK_LAG_MS)
        lo = int(np.searchsorted(self.capture_ts, lo_cap, side="left"))
        if hi <= lo:
            return float("nan")
        cap = self.capture_ts[lo:hi]
        src = self.source_ts[lo:hi]
        px = self.px[lo:hi]
        mask = (
            (src <= target)
            & (src >= target - minutes * 60_000)
            & (np.abs(cap - src) <= MAX_CAPTURE_SOURCE_DELTA_MS)
            & np.isfinite(px)
            & (px > 0)
        )
        idx = np.flatnonzero(mask)
        if not len(idx):
            return float("nan")
        src2 = src[idx]
        px2 = px[idx]
        order = np.argsort(src2, kind="mergesort")
        src2 = src2[order]
        px2 = px2[order]
        minute_id = src2 // 60_000
        keep = np.r_[minute_id[1:] != minute_id[:-1], True]
        vals = px2[keep]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < min_obs + 1:
            return float("nan")
        r = np.diff(np.log(vals))
        if r.size < min_obs:
            return float("nan")
        return float(np.std(r, ddof=1)) * math.sqrt(365.0 * 24.0 * 60.0)


def boundary_safe_list_selected_files(revision: str, table: str, start: str, end: str, cache: Path) -> list[Path]:
    """Select rotation increments by rotation timestamp, plus the first file after END."""
    api = HfApi()
    files = api.list_repo_files(runner.HF_REPO, repo_type="dataset", revision=revision)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    selected: list[str] = []

    root = f"{table}.parquet"
    if date.fromisoformat(start) <= date(2026, 6, 15) and date.fromisoformat(end) > date(2026, 6, 4) and root in files:
        selected.append(root)

    rotations: list[tuple[pd.Timestamp, str]] = []
    pat = re.compile(rf"^daily/{re.escape(table)}/(20\d\d-\d\d-\d\dT\d{{6}}Z)\.parquet$")
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        stamp = pd.to_datetime(m.group(1), format="%Y-%m-%dT%H%M%SZ", utc=True)
        rotations.append((stamp, f))
    rotations.sort(key=lambda x: x[0])

    selected.extend(f for stamp, f in rotations if start_ts <= stamp < end_ts)
    post_end = next((f for stamp, f in rotations if stamp >= end_ts), None)
    if post_end is not None:
        selected.append(post_end)

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError(f"no {table} source files overlap requested window {start}..{end} at {revision}")

    paths = [Path(hf_hub_download(
        runner.HF_REPO, f, repo_type="dataset", revision=revision,
        cache_dir=str(cache / "hf"),
    )) for f in selected]
    _SELECTED_PATHS[table] = paths
    return paths


def load_chainlink_dual_clock(paths: list[Path]) -> DualClockPriceSeries:
    con = duckdb.connect()
    lo = runner.utc_ms(runner.START) - 75 * 60_000
    hi = runner.utc_ms(runner.END) + 30_000
    q = f"""
      SELECT CAST(ts_ms AS BIGINT) AS capture_ts_ms,
             CAST(value AS DOUBLE) AS value,
             CAST(full AS VARCHAR) AS full
      FROM read_parquet({runner.sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{runner.ASSET}' AND lower(src)='chainlink'
        AND ts_ms >= {lo} AND ts_ms <= {hi} AND value > 0
      ORDER BY ts_ms
    """
    df = con.execute(q).fetchdf()
    con.close()
    if len(df) < 100:
        raise RuntimeError(f"insufficient Chainlink ETH observations: {len(df)}")

    source = np.array([_source_ts_from_full(x) or -1 for x in df["full"].tolist()], dtype=np.int64)
    capture = df["capture_ts_ms"].to_numpy(np.int64)
    px = df["value"].to_numpy(float)
    parsed = source > 0
    deltas = capture[parsed] - source[parsed]
    clock_valid = parsed.copy()
    clock_valid[parsed] = np.abs(deltas) <= MAX_CAPTURE_SOURCE_DELTA_MS
    valid = clock_valid & np.isfinite(px) & (px > 0)

    parse_rate = float(parsed.mean()) if len(parsed) else 0.0
    valid_rate = float(valid.mean()) if len(valid) else 0.0
    _DUAL_CLOCK_AUDIT.clear()
    _DUAL_CLOCK_AUDIT.update({
        "chainlink_rows": int(len(df)),
        "payload_source_timestamp_parsed": int(parsed.sum()),
        "payload_source_timestamp_parse_rate": parse_rate,
        "source_clock_rows_within_5s_of_capture": int(valid.sum()),
        "source_clock_valid_rate": valid_rate,
        "source_after_capture_rows": int((parsed & (source > capture)).sum()),
        "capture_minus_source_ms_median": float(np.median(deltas)) if len(deltas) else None,
        "capture_minus_source_ms_p95_abs": float(np.quantile(np.abs(deltas), 0.95)) if len(deltas) else None,
        "policy": "capture clock gates information availability; payload/source clock defines Chainlink oracle time",
    })
    if int(valid.sum()) < 100:
        raise RuntimeError(f"insufficient dual-clock Chainlink ETH observations after source timestamp audit: {int(valid.sum())}")

    out = pd.DataFrame({"capture": capture[valid], "source": source[valid], "px": px[valid]})
    out = out.sort_values(["capture", "source"], kind="mergesort").drop_duplicates(["capture", "source"], keep="last")
    return DualClockPriceSeries(
        out["capture"].to_numpy(np.int64),
        out["source"].to_numpy(np.int64),
        out["px"].to_numpy(float),
    )


def latest_row_books(paths: list[Path]) -> pd.DataFrame:
    """Rank every capture row first, then validate ask/depth on the latest row only."""
    con = duckdb.connect()
    lo, hi = runner.utc_ms(runner.START), runner.utc_ms(runner.END)
    target_offset = runner.DECISION_S2C * 1000
    q = f"""
    WITH src AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms,
             CAST(best_bid AS DOUBLE) AS best_bid,
             CAST(best_ask AS DOUBLE) AS best_ask,
             CAST(bid_sz AS DOUBLE) AS bid_sz,
             CAST(ask_sz AS DOUBLE) AS ask_sz,
             lower(outcome) AS outcome, slug, CAST(cond AS VARCHAR) AS cond,
             CAST(win_start AS BIGINT) AS win_start, CAST(end_ts AS BIGINT) AS end_ts
      FROM read_parquet({runner.sql_files(paths)}, union_by_name=true)
      WHERE upper(asset)='{runner.ASSET}' AND lower(slug) LIKE '%-updown-5m-%'
        AND end_ts*1000 >= {lo} AND end_ts*1000 < {hi}
        AND ts_ms <= end_ts*1000 - {target_offset}
        AND ts_ms >= end_ts*1000 - {target_offset + 7000}
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY slug, outcome ORDER BY ts_ms DESC, cond DESC) AS rn
      FROM src
    )
    SELECT * FROM ranked
    WHERE rn=1
      AND best_ask IS NOT NULL AND best_ask > 0 AND best_ask < 1
      AND ask_sz IS NOT NULL AND ask_sz > 0
    ORDER BY end_ts, slug, outcome
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def one_market_rows_dual_clock(book: pd.DataFrame, ps: DualClockPriceSeries, resolutions: dict[str, bool]) -> pd.DataFrame:
    rows = []
    for slug, g in book.groupby("slug", sort=False):
        first = g.iloc[0]
        cond = str(first.cond)
        if cond not in resolutions:
            continue
        win_start = int(first.win_start)
        end_ts = int(first.end_ts)
        open_target_ms = win_start * 1000
        end_target_ms = end_ts * 1000
        target_decision_ms = end_target_ms - runner.DECISION_S2C * 1000
        settle_up = bool(resolutions[cond])

        # Diagnostic close reconstruction only. It never determines PnL.
        close_px, close_src, close_cap = ps.at_source_time(
            end_target_ms, end_target_ms + 10_000, runner.MAX_CHAINLINK_LAG_MS
        )
        candidates = []
        for rr in g.itertuples(index=False):
            outcome = str(rr.outcome).lower()
            if not (outcome.startswith("up") or outcome.startswith("down")):
                continue
            decision_ts = int(rr.ts_ms)
            book_staleness_ms = int(target_decision_ms - decision_ts)
            if book_staleness_ms < 0 or book_staleness_ms > runner.MAX_BOOK_STALENESS_MS:
                continue

            # PTB reconstruction is keyed to source time T=0, but only from a
            # message that had actually arrived by this trading decision.
            open_px, open_src, open_cap = ps.at_source_time(
                open_target_ms, decision_ts, runner.MAX_CHAINLINK_LAG_MS
            )
            spot, spot_src, spot_cap = ps.latest_available(
                decision_ts, runner.MAX_CHAINLINK_LAG_MS, runner.MAX_CHAINLINK_LAG_MS
            )
            rv60 = ps.rv(decision_ts, source_target_ms=decision_ts)
            if not (
                open_px > 0 and spot > 0
                and 0 <= open_target_ms - open_src <= runner.MAX_CHAINLINK_LAG_MS
                and open_cap <= decision_ts
                and 0 <= decision_ts - spot_cap <= runner.MAX_CHAINLINK_LAG_MS
                and 0 <= decision_ts - spot_src <= runner.MAX_CHAINLINK_LAG_MS
                and math.isfinite(rv60) and 0.02 < rv60 < 5.0
            ):
                continue

            tau = max((end_target_ms - decision_ts) / 1000.0, 1.0)
            p_up = runner.digital_prob_up(spot / open_px, tau, rv60)
            if not math.isfinite(p_up):
                continue
            side = "up" if outcome.startswith("up") else "down"
            fair = p_up if side == "up" else 1.0 - p_up
            if fair < 0.5:
                continue

            ask = float(rr.best_ask)
            ask_sz = float(rr.ask_sz)
            q = runner.qty_for_budget(ask)
            cost = q * (ask + runner.fee_ps(ask))
            barrier_bps = abs(spot / open_px - 1.0) * 10000.0
            sigma_distance = abs(math.log(spot / open_px)) / max(rv60 * math.sqrt(tau / runner.YEAR_SECONDS), 1e-12)
            won = settle_up if side == "up" else not settle_up
            captured_chainlink_up = bool(close_px >= open_px) if close_px > 0 else None
            mismatch = (captured_chainlink_up != settle_up) if captured_chainlink_up is not None else None
            candidates.append({
                "slug": slug, "cond": cond, "win_start": win_start, "end_ts": end_ts,
                "decision_ts_ms": decision_ts, "decision_s2c_s": tau,
                "book_staleness_ms": book_staleness_ms,
                "side": side, "fair": fair, "p_up": p_up,
                "ask": ask, "ask_sz": ask_sz, "qty_fixed5": q, "cost": cost,
                "depth_headroom_x": ask_sz / max(q, 1e-12),
                "net_settlement_edge_ps": fair - ask - runner.fee_ps(ask),
                "barrier_bps": barrier_bps, "sigma_distance": sigma_distance,
                "rv60": rv60,
                "threshold_chainlink": open_px, "spot_chainlink": spot,
                "close_chainlink_diagnostic": close_px,
                "gamma_settle_up": bool(settle_up),
                "captured_chainlink_reconstructed_up": captured_chainlink_up,
                "gamma_vs_captured_chainlink_mismatch": mismatch,
                "won": bool(won), "pnl_fixed5": (q if won else 0.0) - cost,
                "threshold_source_ts": open_src, "threshold_capture_ts": open_cap,
                "spot_source_ts": spot_src, "spot_capture_ts": spot_cap,
                "close_source_ts": close_src, "close_capture_ts": close_cap,
                "threshold_source_lag_ms": int(open_target_ms - open_src),
                "threshold_capture_minus_source_ms": int(open_cap - open_src),
                "threshold_known_before_decision_ms": int(decision_ts - open_cap),
                "spot_source_lag_ms": int(decision_ts - spot_src),
                "spot_capture_lag_ms": int(decision_ts - spot_cap),
                "spot_capture_minus_source_ms": int(spot_cap - spot_src),
            })
        if candidates:
            rows.append(max(candidates, key=lambda x: (x["net_settlement_edge_ps"], x["fair"])))
    return pd.DataFrame(rows).sort_values("decision_ts_ms", kind="mergesort") if rows else pd.DataFrame()


def table_coverage(paths: list[Path], table: str) -> dict:
    if not paths:
        return {
            "table": table, "rows_in_requested_window": 0,
            "first_ts_ms": None, "last_ts_ms": None,
            "observed_utc_days": [], "missing_requested_utc_days": [],
        }
    lo, hi = runner.utc_ms(runner.START), runner.utc_ms(runner.END)
    if table == "cap_book":
        filt = f"upper(asset)='{runner.ASSET}' AND lower(slug) LIKE '%-updown-5m-%'"
    elif table == "cap_prices":
        filt = f"upper(asset)='{runner.ASSET}' AND lower(src)='chainlink'"
    else:
        raise ValueError(table)

    con = duckdb.connect()
    files_sql = runner.sql_files(paths)
    base = f"read_parquet({files_sql}, union_by_name=true)"
    row = con.execute(f"""
        SELECT count(*) AS n, min(CAST(ts_ms AS BIGINT)) AS first_ts, max(CAST(ts_ms AS BIGINT)) AS last_ts
        FROM {base}
        WHERE {filt} AND ts_ms >= {lo} AND ts_ms < {hi}
    """).fetchone()
    ddf = con.execute(f"""
        SELECT CAST(to_timestamp(CAST(ts_ms AS DOUBLE)/1000.0) AS DATE) AS day, count(*) AS n
        FROM {base}
        WHERE {filt} AND ts_ms >= {lo} AND ts_ms < {hi}
        GROUP BY day ORDER BY day
    """).fetchdf()
    con.close()

    observed = [str(x) for x in ddf["day"].tolist()] if len(ddf) else []
    expected = [d.strftime("%Y-%m-%d") for d in pd.date_range(
        runner.START, pd.Timestamp(runner.END) - pd.Timedelta(days=1), freq="D", tz="UTC"
    )]
    observed_set = set(observed)
    missing = [d for d in expected if d not in observed_set]
    return {
        "table": table,
        "rows_in_requested_window": int(row[0] or 0),
        "first_ts_ms": int(row[1]) if row[1] is not None else None,
        "last_ts_ms": int(row[2]) if row[2] is not None else None,
        "observed_utc_days": observed,
        "observed_utc_day_count": len(observed),
        "requested_utc_day_count": len(expected),
        "missing_requested_utc_days": missing,
        "complete_requested_day_coverage": len(missing) == 0,
    }


runner.list_selected_files = boundary_safe_list_selected_files
runner.load_chainlink = load_chainlink_dual_clock
runner.load_decision_books = latest_row_books
runner.one_market_rows = one_market_rows_dual_clock


if __name__ == "__main__":
    runner.main()
    out = Path("eth5m_tail_out")
    raw_path = out / "market_rows.csv"
    surface_path = out / "surface.csv"
    manifest_path = out / "manifest.json"

    book_cov = table_coverage(_SELECTED_PATHS.get("cap_book", []), "cap_book")
    price_cov = table_coverage(_SELECTED_PATHS.get("cap_prices", []), "cap_prices")
    day_coverage_complete = bool(
        book_cov.get("complete_requested_day_coverage")
        and price_cov.get("complete_requested_day_coverage")
    )
    source_clock_complete = bool(
        float(_DUAL_CLOCK_AUDIT.get("payload_source_timestamp_parse_rate") or 0.0) >= MIN_CLOCK_PARSE_RATE_FOR_GREEN
        and float(_DUAL_CLOCK_AUDIT.get("source_clock_valid_rate") or 0.0) >= MIN_CLOCK_PARSE_RATE_FOR_GREEN
    )

    if raw_path.exists() and surface_path.exists():
        raw = pd.read_csv(raw_path)
        surf = pd.read_csv(surface_path)
        stat_rows = []
        for r in surf.itertuples(index=False):
            fair_floor = float(r.fair_floor)
            barrier_floor = float(r.barrier_bps_floor)
            z = raw[(raw["fair"] >= fair_floor) &
                    (raw["barrier_bps"] >= barrier_floor) &
                    (raw["net_settlement_edge_ps"] > 0) &
                    (raw["depth_headroom_x"] >= 1.0)].copy()
            seed_offset = int(round(fair_floor * 1000)) * 100 + int(round(barrier_floor * 10))
            st = daily_pnl_stats(z, "decision_ts_ms", "pnl_fixed5", unit="ms", seed_offset=seed_offset)
            stat_rows.append(st)

        stat_grades = [x["grade"] for x in stat_rows]
        overall = []
        for g in stat_grades:
            if str(g).startswith("RED"):
                overall.append("RED")
            elif not day_coverage_complete:
                overall.append("YELLOW_SOURCE_COVERAGE")
            elif not source_clock_complete:
                overall.append("YELLOW_SOURCE_CLOCK")
            else:
                overall.append(g)

        surf["statistical_grade"] = stat_grades
        surf["overall_qualification_grade"] = overall
        surf["source_requested_day_coverage_complete"] = day_coverage_complete
        surf["source_dual_clock_complete"] = source_clock_complete
        surf["bootstrap_days"] = [x["days"] for x in stat_rows]
        surf["mean_daily_pnl"] = [x["mean_daily_pnl"] for x in stat_rows]
        surf["daily_pnl_ci95_low"] = [x["mean_daily_pnl_ci95"][0] for x in stat_rows]
        surf["daily_pnl_ci95_high"] = [x["mean_daily_pnl_ci95"][1] for x in stat_rows]
        surf["positive_day_rate"] = [x["positive_day_rate"] for x in stat_rows]
        if len(raw):
            surf["median_threshold_capture_minus_source_ms"] = float(raw["threshold_capture_minus_source_ms"].median())
            surf["median_spot_capture_minus_source_ms"] = float(raw["spot_capture_minus_source_ms"].median())
        surf.to_csv(surface_path, index=False)

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        m["requested_period"] = [runner.START, runner.END]
        m["source_coverage"] = {
            "cap_book_eth5m": book_cov,
            "cap_prices_eth_chainlink": price_cov,
            "complete_requested_day_coverage": day_coverage_complete,
            "dual_clock_chainlink": _DUAL_CLOCK_AUDIT,
            "dual_clock_complete_for_green": source_clock_complete,
            "policy": "statistical alpha is reported on observed trades, but an otherwise positive/GREEN cell is demoted when requested-day coverage or Chainlink source-clock extraction is insufficient",
        }
        m["qualification_gate"] = {
            "GREEN": "cell total PnL >0, >=20 observed trading days, >=100 trades, day-bootstrap mean daily PnL CI95 lower bound >0, complete requested-day source coverage, and >=99.5% valid Chainlink payload/source timestamp coverage",
            "YELLOW": "positive PnL but statistical sample/CI, requested source coverage, or source-clock audit is insufficient",
            "RED": "cell total PnL <=0",
            "selection_policy": "all 20 predeclared fair-floor x barrier-distance cells are reported and graded; no post-hoc winner is substituted for the surface",
        }
        m["qualification_audit"] = {
            "book_selector": "latest capture row at/before T-90 is ranked before any price/depth validation",
            "rotation_file_selection": "window rotations plus first post-END rotation, with parquet timestamps performing final row filtering",
            "chainlink_clock": "collector capture clock gates availability; nested RTDS payload timestamp defines oracle/source time",
            "threshold_clock": "latest source timestamp <= window start, captured no later than the chosen T-90 decision book timestamp",
            "stale_price_borrowing": False,
            "stale_depth_borrowing": False,
            "gamma_terminal_resolution_required": True,
            "max_book_staleness_ms": runner.MAX_BOOK_STALENESS_MS,
            "max_chainlink_capture_or_source_lag_ms": runner.MAX_CHAINLINK_LAG_MS,
            "max_capture_minus_source_abs_ms": MAX_CAPTURE_SOURCE_DELTA_MS,
        }
        manifest_path.write_text(json.dumps(m, indent=2))
