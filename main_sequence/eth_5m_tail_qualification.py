from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from scipy.stats import norm

from main_sequence.prejuly_5m_multiasset import (
    FEE_RATE,
    HF_REPO,
    YEAR_SECONDS,
    build_price_series,
    digital_prob_up,
    list_selected_files,
    load_price_bars,
    sql_files,
)

START = "2026-06-04"
END = "2026-07-15"
DECISION_S2C = 60
TICKET_USD = 5.0
THRESHOLDS = (0.95, 0.97, 0.98, 0.99)


def utc_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def normalize_outcome(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("up") or s == "yes":
        return "up"
    if s.startswith("down") or s == "no":
        return "down"
    return s


def load_eth_decisions(paths: list[Path], start: str, end: str) -> pd.DataFrame:
    lo, hi = utc_ms(start), utc_ms(end)
    con = duckdb.connect()
    q = f"""
    WITH src AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms,
             CAST(asset_id AS VARCHAR) AS asset_id,
             CAST(best_bid AS DOUBLE) AS best_bid,
             CAST(best_ask AS DOUBLE) AS best_ask,
             CAST(bid_sz AS DOUBLE) AS bid_sz,
             CAST(ask_sz AS DOUBLE) AS ask_sz,
             upper(asset) AS asset, outcome, slug, cond,
             CAST(win_start AS BIGINT) AS win_start,
             CAST(end_ts AS BIGINT) AS end_ts
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset) = 'ETH'
        AND lower(slug) LIKE '%-updown-5m-%'
        AND end_ts * 1000 >= {lo} AND end_ts * 1000 < {hi}
        AND ts_ms <= end_ts * 1000 - {DECISION_S2C * 1000}
        AND ts_ms >= end_ts * 1000 - {(DECISION_S2C + 15) * 1000}
        AND best_ask > 0 AND best_ask < 1
        AND best_bid >= 0 AND best_bid < 1
        AND ask_sz >= 0
    ), ranked AS (
      SELECT *, row_number() OVER (
        PARTITION BY slug, lower(outcome) ORDER BY ts_ms DESC
      ) AS rn
      FROM src
    )
    SELECT * FROM ranked WHERE rn = 1
    ORDER BY end_ts, slug, lower(outcome)
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def mdd_from_pnls(pnls: list[float], initial: float = 50.0) -> float | None:
    if not pnls:
        return None
    eq = initial + np.cumsum(np.asarray(pnls, dtype=float))
    eq = np.r_[initial, eq]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    revision = HfApi().dataset_info(HF_REPO).sha
    book_paths = list_selected_files(revision, "cap_book", START, END, args.cache)
    price_paths = list_selected_files(revision, "cap_prices", START, END, args.cache)
    book = load_eth_decisions(book_paths, START, END)
    price_bars = load_price_bars(price_paths, START, END)
    prices = build_price_series(price_bars)
    ps = prices["ETH"]
    if ps.source != "chainlink":
        raise RuntimeError(f"ETH reference source is {ps.source}, expected chainlink")

    rows: list[dict] = []
    for slug, g in book.groupby("slug", sort=False):
        up = g[g.outcome.map(normalize_outcome) == "up"]
        dn = g[g.outcome.map(normalize_outcome) == "down"]
        if len(up) != 1 or len(dn) != 1:
            continue
        u, d = up.iloc[0], dn.iloc[0]
        if abs(int(u.ts_ms) - int(d.ts_ms)) > 7000:
            continue
        end_ts = int(u.end_ts)
        win_start = int(u.win_start)
        decision_ts = max(int(u.ts_ms), int(d.ts_ms))
        target = end_ts * 1000 - DECISION_S2C * 1000
        if target - decision_ts > 7000:
            continue
        anchor, anchor_ts = ps.at(win_start * 1000, tolerance_ms=30_000)
        close, close_ts = ps.at(end_ts * 1000, tolerance_ms=30_000)
        spot, spot_ts = ps.at(decision_ts, tolerance_ms=20_000)
        if not (anchor > 0 and close > 0 and spot > 0):
            continue
        rv60 = ps.rv(decision_ts, 60, min_obs=30)
        if not (math.isfinite(rv60) and 0.02 < rv60 < 5.0):
            continue
        tau_s = max((end_ts * 1000 - decision_ts) / 1000.0, 1.0)
        rel = spot / anchor
        p_up = digital_prob_up(rel, tau_s, rv60)
        if not math.isfinite(p_up):
            continue
        settle_up = close > anchor
        side_up = p_up >= 0.5
        fair = p_up if side_up else 1.0 - p_up
        chosen = u if side_up else d
        ask = float(chosen.best_ask)
        ask_sz = max(float(chosen.ask_sz), 0.0)
        depth_usd = ask * ask_sz
        fee_per_share = FEE_RATE * ask * (1.0 - ask)
        net_edge = fair - ask - fee_per_share
        raw_barrier_sigma = abs(math.log(rel)) / max(rv60 * math.sqrt(tau_s / YEAR_SECONDS), 1e-12)
        d2_abs = abs(float(norm.ppf(min(max(p_up, 1e-9), 1.0 - 1e-9))))
        won = bool(settle_up == side_up)
        shares = TICKET_USD / ask
        fee_usd = shares * fee_per_share
        pnl = (shares if won else 0.0) - TICKET_USD - fee_usd
        rows.append({
            "slug": str(slug), "decision_ts": decision_ts, "end_ts": end_ts,
            "side": "Up" if side_up else "Down", "settle_up": bool(settle_up), "won": won,
            "anchor": anchor, "spot": spot, "close": close, "rel_spot": rel,
            "rv60": rv60, "tau_s": tau_s, "p_up": p_up, "fair": fair,
            "ask": ask, "ask_sz": ask_sz, "depth_usd": depth_usd,
            "fee_per_share": fee_per_share, "net_edge": net_edge,
            "raw_barrier_sigma": raw_barrier_sigma, "d2_abs": d2_abs,
            "ticket_usd": TICKET_USD, "fee_usd": fee_usd, "pnl": pnl,
            "anchor_ts": anchor_ts, "spot_ts": spot_ts, "close_ts": close_ts,
        })

    df = pd.DataFrame(rows).sort_values(["decision_ts", "slug"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no ETH 5m decision rows survived causal/reference checks")
    df.to_csv(args.out / "eth_5m_tail_all_decisions.csv", index=False)

    summary: dict = {
        "contract": "ETH 5m TAIL settlement-only; Chainlink barrier; actual displayed ask/top-level depth; fixed $5 qualification ticket",
        "window": [START, END],
        "dataset": HF_REPO,
        "dataset_revision": revision,
        "reference_source": ps.source,
        "decision_s2c": DECISION_S2C,
        "ticket_usd": TICKET_USD,
        "fee_rate": FEE_RATE,
        "markets_with_causal_decision": int(len(df)),
        "thresholds": {},
        "barrier_filter": "independent raw standardized barrier distance |log(S/K)|/(sigma*sqrt(tau)) >= N^{-1}(fair_threshold); d2 also reported separately",
        "execution": "first/only 60s decision snapshot per market; require top ask notional depth >= $5 and positive fair-minus-ask-minus-fee; hold to Chainlink settlement",
    }

    for th in THRESHOLDS:
        z_floor = float(norm.ppf(th))
        fair_gate = df[df.fair >= th].copy()
        barrier_gate = fair_gate[fair_gate.raw_barrier_sigma >= z_floor].copy()
        depth_gate = barrier_gate[barrier_gate.depth_usd >= TICKET_USD - 1e-12].copy()
        trades = depth_gate[depth_gate.net_edge > 0].copy()
        trades.to_csv(args.out / f"eth_5m_tail_fair_{int(th*100)}.csv", index=False)
        pnls = trades.pnl.tolist()
        total_cost = float((trades.ticket_usd + trades.fee_usd).sum()) if not trades.empty else 0.0
        summary["thresholds"][str(th)] = {
            "z_floor": z_floor,
            "fair_gate": int(len(fair_gate)),
            "barrier_gate": int(len(barrier_gate)),
            "depth_gate": int(len(depth_gate)),
            "positive_net_edge_trades": int(len(trades)),
            "win_rate": None if trades.empty else float(trades.won.mean()),
            "mean_fair": None if trades.empty else float(trades.fair.mean()),
            "mean_ask": None if trades.empty else float(trades.ask.mean()),
            "mean_depth_usd": None if trades.empty else float(trades.depth_usd.mean()),
            "mean_net_edge": None if trades.empty else float(trades.net_edge.mean()),
            "pnl_fixed5": float(sum(pnls)),
            "roi_on_cash_plus_fee": None if total_cost <= 0 else float(sum(pnls) / total_cost),
            "mdd_from_50": mdd_from_pnls(pnls, 50.0),
            "losses": int((~trades.won).sum()) if not trades.empty else 0,
        }

    (args.out / "eth_5m_tail_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# ETH 5m TAIL qualification", "",
        f"Window: {START}..{END} UTC; reference: {ps.source}; fixed qualification ticket: ${TICKET_USD:.0f}.", "",
        "| fair floor | fair gate | barrier gate | depth gate | EV+ trades | WR | PnL | ROI | MDD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for th in THRESHOLDS:
        s = summary["thresholds"][str(th)]
        fmt = lambda x: "NA" if x is None else f"{100*x:.2f}%"
        lines.append(
            f"| {th:.0%} | {s['fair_gate']} | {s['barrier_gate']} | {s['depth_gate']} | {s['positive_net_edge_trades']} | "
            f"{fmt(s['win_rate'])} | ${s['pnl_fixed5']:.2f} | {fmt(s['roi_on_cash_plus_fee'])} | {fmt(s['mdd_from_50'])} |"
        )
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((args.out / "SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
