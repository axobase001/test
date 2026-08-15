from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from main_sequence import original_93d_tape_replay as base
from pm_structural import recalc as original

DAYS = 93
SIZE = 5.0
LARGE_RAW_GAP = 0.10
EXIT_BAND = 0.01
PRIOR_RUN = 31883052702
SHARDS = {
    "mar": ["2026-03-01", "2026-04-01"],
    "apr": ["2026-04-01", "2026-05-01"],
    "may": ["2026-05-01", "2026-06-01"],
    "jun1": ["2026-06-01", "2026-06-02"],
}

PROTOCOL = {
    "name": "Main Sequence corrected dual-exit structural_03c / 93-day replay v1",
    "period": ["2026-03-01", "2026-06-02"],
    "days": DAYS,
    "source_decisions": {
        "authoritative_run": PRIOR_RUN,
        "policy": "frozen original structural_03c BTC15m decisions",
        "entry_floor_net_edge": 0.03,
        "size_shares": SIZE,
        "decision_window_s2c": [60, 600],
    },
    "regime_split": {
        "definition": "initial raw structural gap = conservative fair boundary - witnessed PM buy price, before fee",
        "large_convergence_if_raw_gap_gte": LARGE_RAW_GAP,
        "small_settlement_if_raw_gap_lt": LARGE_RAW_GAP,
        "note": "10pp split is frozen from the user's corrected strategy example; it is not selected by inspecting 93-day PnL.",
    },
    "large_exit": {
        "definition": "recompute Binance-60m-RV + backward-30m Deribit-trade-IV conservative fair boundary causally; exit once a sellable PM price is within 1c of that boundary",
        "exit_band": EXIT_BAND,
        "fallback": "if no executable convergence exit occurs before close, hold to binary settlement",
    },
    "small_exit": "hold to binary settlement",
    "layers": {
        "witness_book": "entry uses the frozen 5-share decision-second BUY witness; convergence exit uses the first same-second >=5-share public taker-SELL witness meeting the fair-band condition",
        "strict5": "entry must reconstruct the prior full-size fresh +1..+5s BUY fill exactly; convergence requires a >=5-share SELL witness and then fresh +1..+5s SELL volume at or above the frozen convergence limit; witness trades are not reused as fills",
    },
    "statistics": "all 93 days retained; iid-day and 7-day moving-block bootstrap; Newey-West lag7; realized-time capital ledger with solvency plus 10/20/30% historical realized Max-DD annualization",
    "anti_tuning": "no threshold search, no month deletion, no post-hoc asset selection; large/small split and convergence band are frozen before running the corrected PnL replay",
}


def fee(p: float) -> float:
    return base.fee_per_share(float(p))


def raw_gap(row) -> float:
    return float(row.signal_edge) + fee(float(row.limit))


def side_outcome(row) -> str:
    return "up" if str(row.side).lower() == "up" else "down"


def label_up_from_row(row) -> float:
    won = bool(row.won)
    return 1.0 if ((str(row.side).lower() == "up" and won) or (str(row.side).lower() == "down" and not won)) else 0.0


def load_spot_from_prior(prior_shard: Path) -> base.BinanceSecond:
    files = sorted((prior_shard / "binance_1s_cache").glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no prior Binance 1s cache under {prior_shard}")
    frames = [pd.read_parquet(p) for p in files]
    return base.BinanceSecond(pd.concat(frames, ignore_index=True))


def reconstruct_buy_fill(g: pd.DataFrame, outcome: str, limit: float, decision: int, qty: float = SIZE):
    z = g[
        (g["timestamp"] >= decision + 1)
        & (g["timestamp"] <= decision + 5)
        & (g["side_u"] == "BUY")
        & (g["outcome_l"] == outcome)
        & (g["price"] <= limit)
    ].copy()
    if z.empty:
        return None
    z = z.sort_values(["timestamp", "price", "size"], kind="mergesort")
    left, gross, fees = float(qty), 0.0, 0.0
    done = None
    for r in z.itertuples(index=False):
        take = min(left, float(r.size))
        if take <= 0:
            continue
        px = float(r.price)
        gross += take * px
        fees += take * fee(px)
        left -= take
        done = int(r.timestamp)
        if left <= 1e-12:
            return {"cost": gross + fees, "avg": gross / qty, "done": done}
    return None


def sell_witness(qsec: pd.DataFrame, outcome: str, min_price: float, qty: float = SIZE):
    z = qsec[
        (qsec["side_u"] == "SELL")
        & (qsec["outcome_l"] == outcome)
        & (qsec["price"] >= min_price)
        & (qsec["size"] > 0)
    ].copy()
    if z.empty:
        return None
    z = z.sort_values(["price", "size"], ascending=[False, True], kind="mergesort")
    left, gross, fees = float(qty), 0.0, 0.0
    for r in z.itertuples(index=False):
        take = min(left, float(r.size))
        if take <= 0:
            continue
        px = float(r.price)
        gross += take * px
        fees += take * fee(px)
        left -= take
        if left <= 1e-12:
            return {"proceeds": gross - fees, "avg": gross / qty}
    return None


def strict_sell_fill(g: pd.DataFrame, outcome: str, limit: float, witness_sec: int, close: int, qty: float = SIZE):
    z = g[
        (g["timestamp"] >= witness_sec + 1)
        & (g["timestamp"] <= min(witness_sec + 5, close - 1))
        & (g["side_u"] == "SELL")
        & (g["outcome_l"] == outcome)
        & (g["price"] >= limit)
        & (g["size"] > 0)
    ].copy()
    if z.empty:
        return None
    z = z.sort_values(["timestamp", "price", "size"], ascending=[True, False, True], kind="mergesort")
    left, gross, fees = float(qty), 0.0, 0.0
    done = None
    for r in z.itertuples(index=False):
        take = min(left, float(r.size))
        if take <= 0:
            continue
        px = float(r.price)
        gross += take * px
        fees += take * fee(px)
        left -= take
        done = int(r.timestamp)
        if left <= 1e-12:
            return {"proceeds": gross - fees, "avg": gross / qty, "done": done}
    return None


def fair_boundary(row, sec: int, spot: base.BinanceSecond, bn, der):
    close = int(row.close)
    if sec >= close:
        return None
    rv = bn.rv_annualized(sec * 1000, 60)
    div = der.median_iv(sec * 1000, 30)
    if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
        return None
    op = bn.open_price(int(row.start))
    sp = spot.at(sec)
    if not (op > 0 and sp > 0):
        return None
    s2c = close - sec
    p_rv = original.digital_prob_up(sp / op, s2c, rv)
    p_iv = original.digital_prob_up(sp / op, s2c, div)
    if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
        return None
    p_lo, p_hi = min(p_rv, p_iv), max(p_rv, p_iv)
    fair = p_lo if str(row.side).lower() == "up" else 1.0 - p_hi
    return {"fair": float(fair), "p_rv": float(p_rv), "p_iv": float(p_iv), "rv": float(rv), "div": float(div), "spot": float(sp)}


def replay_large(row, g: pd.DataFrame, spot: base.BinanceSecond, bn, der):
    outcome = side_outcome(row)
    decision, close = int(row.decision), int(row.close)
    payout = SIZE if bool(row.won) else 0.0
    paper_cost = float(row.paper_cost)

    # Reconstruct the strict entry exactly from the same public tape semantics as the prior replay.
    entry = reconstruct_buy_fill(g, outcome, float(row.limit), decision, SIZE)
    prior_cost = None if pd.isna(row.tape5_cost) else float(row.tape5_cost)
    if (entry is None) != (prior_cost is None):
        raise RuntimeError(f"entry equivalence mismatch {row.slug}: prior={prior_cost} replay={entry}")
    if entry is not None and abs(float(entry["cost"]) - prior_cost) > 1e-8:
        raise RuntimeError(f"entry cost mismatch {row.slug}: prior={prior_cost} replay={entry['cost']}")

    sells = g[
        (g["timestamp"] >= decision + 1)
        & (g["timestamp"] < close)
        & (g["side_u"] == "SELL")
        & (g["outcome_l"] == outcome)
    ]
    secs = sorted(int(x) for x in sells["timestamp"].unique().tolist()) if not sells.empty else []

    witness_exit = None
    strict_exit = None
    strict_start = int(entry["done"]) + 1 if entry is not None else None
    first_market_convergence_sec = None

    for sec in secs:
        fb = fair_boundary(row, sec, spot, bn, der)
        if fb is None:
            continue
        min_price = max(0.0, min(1.0, fb["fair"] - EXIT_BAND))
        qsec = sells[sells["timestamp"] == sec]
        w = sell_witness(qsec, outcome, min_price, SIZE)
        if w is None:
            continue
        if first_market_convergence_sec is None:
            first_market_convergence_sec = sec
        if witness_exit is None:
            witness_exit = {**w, **fb, "sec": sec, "limit": min_price}
        if entry is not None and sec >= strict_start and strict_exit is None:
            f = strict_sell_fill(g, outcome, min_price, sec, close, SIZE)
            if f is not None:
                strict_exit = {**f, **fb, "witness_sec": sec, "limit": min_price}
                break

    if witness_exit is not None:
        witness_reward = float(witness_exit["proceeds"]) - paper_cost
        witness_exit_time = int(witness_exit["sec"])
        witness_kind = "convergence"
        witness_hold = witness_exit_time - decision
        witness_exit_px = float(witness_exit["avg"])
    else:
        witness_reward = payout - paper_cost
        witness_exit_time = close
        witness_kind = "settlement_fallback"
        witness_hold = close - decision
        witness_exit_px = 1.0 if bool(row.won) else 0.0

    if entry is None:
        strict_reward = strict_cost = strict_exit_time = strict_hold = strict_exit_px = None
        strict_kind = "no_entry_fill"
    elif strict_exit is not None:
        strict_cost = float(entry["cost"])
        strict_reward = float(strict_exit["proceeds"]) - strict_cost
        strict_exit_time = int(strict_exit["done"])
        strict_kind = "convergence"
        strict_hold = strict_exit_time - int(entry["done"])
        strict_exit_px = float(strict_exit["avg"])
    else:
        strict_cost = float(entry["cost"])
        strict_reward = payout - strict_cost
        strict_exit_time = close
        strict_kind = "settlement_fallback"
        strict_hold = close - int(entry["done"])
        strict_exit_px = 1.0 if bool(row.won) else 0.0

    return {
        "witness_reward": witness_reward,
        "witness_cost": paper_cost,
        "witness_exit_time": witness_exit_time,
        "witness_exit_kind": witness_kind,
        "witness_hold_s": witness_hold,
        "witness_exit_px": witness_exit_px,
        "strict_reward": strict_reward,
        "strict_cost": strict_cost,
        "strict_entry_time": (int(entry["done"]) if entry is not None else None),
        "strict_exit_time": strict_exit_time,
        "strict_exit_kind": strict_kind,
        "strict_hold_s": strict_hold,
        "strict_exit_px": strict_exit_px,
        "first_convergence_witness_sec": first_market_convergence_sec,
    }


def replay_hour(hour: int, rows: pd.DataFrame, spot: base.BinanceSecond, bn, der):
    markets = []
    for r in rows.itertuples(index=False):
        markets.append(base.Market(str(r.slug), int(r.start), str(r.condition_id), label_up_from_row(r)))
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-dual-exit-93d/1.0"})
    start = int(rows["decision"].min()) + 1
    end = int(rows["close"].max()) - 1
    raw = base.query_trade_rows(sess, markets, start, end)
    tm = base.normalize_trades(raw)
    out = []
    for r in rows.itertuples(index=False):
        g = tm.get(str(r.condition_id), pd.DataFrame())
        z = replay_large(r, g, spot, bn, der)
        out.append({"slug": str(r.slug), **z})
    return out, len(raw)


def score_shard(prior_shard: Path, anchor_dir: Path, out: Path, shard: str, workers: int = 10):
    if shard not in SHARDS:
        raise RuntimeError(shard)
    sig_path = prior_shard / "signals.csv"
    if not sig_path.exists():
        raise RuntimeError(f"missing {sig_path}")
    df = pd.read_csv(sig_path)
    if df.empty:
        raise RuntimeError("empty prior signals")
    _, bn, der = base.load_anchors(anchor_dir)
    spot = load_spot_from_prior(prior_shard)

    df["initial_raw_gap"] = df.apply(lambda r: float(r["signal_edge"]) + fee(float(r["limit"])), axis=1)
    df["regime"] = np.where(df["initial_raw_gap"] >= LARGE_RAW_GAP, "large_convergence", "small_settlement")

    # Small-gap book is literal settlement using the already-frozen prior entry fills.
    small = df[df["regime"] == "small_settlement"].copy()
    df["witness_reward"] = np.nan
    df["witness_cost"] = np.nan
    df["witness_exit_time"] = np.nan
    df["witness_exit_kind"] = None
    df["witness_hold_s"] = np.nan
    df["witness_exit_px"] = np.nan
    df["strict_reward"] = np.nan
    df["strict_cost"] = np.nan
    df["strict_entry_time"] = np.nan
    df["strict_exit_time"] = np.nan
    df["strict_exit_kind"] = None
    df["strict_hold_s"] = np.nan
    df["strict_exit_px"] = np.nan
    df["first_convergence_witness_sec"] = np.nan

    if len(small):
        ix = small.index
        df.loc[ix, "witness_reward"] = small["paper_reward"].to_numpy(float)
        df.loc[ix, "witness_cost"] = small["paper_cost"].to_numpy(float)
        df.loc[ix, "witness_exit_time"] = small["close"].to_numpy(float)
        df.loc[ix, "witness_exit_kind"] = "settlement"
        df.loc[ix, "witness_hold_s"] = small["close"].to_numpy(float) - small["decision"].to_numpy(float)
        df.loc[ix, "witness_exit_px"] = np.where(small["won"].astype(bool), 1.0, 0.0)
        filled = small["tape5_cost"].notna() & small["tape5_reward"].notna()
        fi = small.index[filled]
        df.loc[fi, "strict_reward"] = small.loc[fi, "tape5_reward"].to_numpy(float)
        df.loc[fi, "strict_cost"] = small.loc[fi, "tape5_cost"].to_numpy(float)
        # Entry completed somewhere in +1..+5s; use +1s for conservative capital occupancy.
        df.loc[fi, "strict_entry_time"] = small.loc[fi, "decision"].to_numpy(float) + 1.0
        df.loc[fi, "strict_exit_time"] = small.loc[fi, "close"].to_numpy(float)
        df.loc[fi, "strict_exit_kind"] = "settlement"
        df.loc[fi, "strict_hold_s"] = small.loc[fi, "close"].to_numpy(float) - (small.loc[fi, "decision"].to_numpy(float) + 1.0)
        df.loc[fi, "strict_exit_px"] = np.where(small.loc[fi, "won"].astype(bool), 1.0, 0.0)
        df.loc[small.index[~filled], "strict_exit_kind"] = "no_entry_fill"

    large = df[df["regime"] == "large_convergence"].copy()
    raw_rows = 0
    large_updates = []
    if len(large):
        large["hour"] = (large["start"].astype(np.int64) // 3600) * 3600
        groups = [(int(h), g.drop(columns=["hour"])) for h, g in large.groupby("hour", sort=True)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(replay_hour, h, g, spot, bn, der): h for h, g in groups}
            for k, f in enumerate(as_completed(fut), 1):
                rr, nraw = f.result()
                large_updates.extend(rr)
                raw_rows += int(nraw)
                if k % 48 == 0:
                    print("DUAL_EXIT_HOURS", shard, k, "/", len(groups), "updates", len(large_updates), "raw", raw_rows, flush=True)
        upd = pd.DataFrame(large_updates).set_index("slug")
        for c in upd.columns:
            if c == "slug":
                continue
            mapper = upd[c].to_dict()
            mask = df["regime"] == "large_convergence"
            df.loc[mask, c] = df.loc[mask, "slug"].map(mapper)

    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["decision", "slug"], kind="mergesort")
    df.to_csv(out / "dual_exit_records.csv", index=False)
    summary = {
        "shard": shard,
        "period": SHARDS[shard],
        "signals": int(len(df)),
        "large": int((df["regime"] == "large_convergence").sum()),
        "small": int((df["regime"] == "small_settlement").sum()),
        "strict_entries": int(df["strict_cost"].notna().sum()),
        "witness_large_convergence_exits": int(((df["regime"] == "large_convergence") & (df["witness_exit_kind"] == "convergence")).sum()),
        "strict_large_convergence_exits": int(((df["regime"] == "large_convergence") & (df["strict_exit_kind"] == "convergence")).sum()),
        "raw_exit_trade_rows": raw_rows,
        "protocol": PROTOCOL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("DUAL_EXIT_SHARD_DONE", json.dumps(summary, indent=2), flush=True)


def capital_path(trades: pd.DataFrame, cost_col: str, reward_col: str, entry_col: str, exit_col: str, target_dd: float):
    z = trades.dropna(subset=[cost_col, reward_col, exit_col]).copy()
    if z.empty:
        return None
    # Entry time is known exactly for large strict fills; small strict fills use decision+1s conservatively.
    if entry_col not in z.columns:
        z[entry_col] = z["decision"]
    z[entry_col] = z[entry_col].fillna(z["decision"]).astype(np.int64)
    z[exit_col] = z[exit_col].astype(np.int64)
    events = []
    for r in z.itertuples(index=False):
        d = r._asdict()
        cost = float(d[cost_col]); reward = float(d[reward_col]); proceeds = cost + reward
        events.append((int(d[entry_col]), 1, -cost))
        events.append((int(d[exit_col]), 0, proceeds))  # exits before entries on a tie
    events.sort(key=lambda x: (x[0], x[1]))
    cash = 0.0; min_cash = 0.0
    for _, _, delta in events:
        cash += delta; min_cash = min(min_cash, cash)
    solvency = max(1e-9, -min_cash)

    exits = z.sort_values(exit_col, kind="mergesort")
    cum = np.cumsum(exits[reward_col].to_numpy(float))
    def dd(k):
        eq = k + cum
        peak = np.maximum.accumulate(np.concatenate([[k], eq]))[1:]
        return float(np.max((peak - eq) / np.maximum(peak, 1e-12))) if len(eq) else 0.0
    if dd(solvency) <= target_dd:
        cap = solvency
    else:
        lo, hi = solvency, max(2 * solvency, 1.0)
        while dd(hi) > target_dd:
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if dd(mid) > target_dd:
                lo = mid
            else:
                hi = mid
        cap = hi
    total = float(z[reward_col].sum())
    ret = total / cap
    return {
        "capital": float(cap), "solvency_floor": float(solvency), "period_return": float(ret),
        "max_drawdown": float(dd(cap)), "simple_annualized": float(ret * 365.0 / DAYS),
        "cagr": float((1.0 + ret) ** (365.0 / DAYS) - 1.0) if ret > -1 else None,
    }


def layer_stats(df: pd.DataFrame, prefix: str):
    cost_col, reward_col = f"{prefix}_cost", f"{prefix}_reward"
    exit_col = f"{prefix}_exit_time"
    entry_col = "strict_entry_time" if prefix == "strict" else "decision"
    z = df.dropna(subset=[cost_col, reward_col, exit_col]).copy()
    if z.empty:
        return {"n": 0}
    z["exit_day"] = pd.to_datetime(z[exit_col], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    z["entry_month"] = pd.to_datetime(z["decision"], unit="s", utc=True).dt.strftime("%Y-%m")
    all_days = pd.date_range("2026-03-01", "2026-06-01", freq="D", tz="UTC").strftime("%Y-%m-%d").tolist()
    daily = z.groupby("exit_day").agg(reward=(reward_col, "sum"), cost=(cost_col, "sum"), n=(reward_col, "size")).reindex(all_days, fill_value=0.0)
    nday = daily["n"].to_numpy(float)
    edge_day = np.divide(daily["reward"].to_numpy(float), nday * SIZE, out=np.zeros(len(daily)), where=nday > 0)
    dr, dc = daily["reward"].to_numpy(float), daily["cost"].to_numpy(float)
    monthly = z.groupby("entry_month").agg(reward=(reward_col, "sum"), cost=(cost_col, "sum"), n=(reward_col, "size"))
    month_out = {m: {"n": int(r.n), "pnl": float(r.reward), "roi": float(r.reward / r.cost), "edge_share": float(r.reward / (r.n * SIZE))} for m, r in monthly.iterrows()}
    by_regime = {}
    for reg, g in z.groupby("regime"):
        by_regime[str(reg)] = {"n": int(len(g)), "pnl": float(g[reward_col].sum()), "roi": float(g[reward_col].sum() / g[cost_col].sum())}
    ann = {f"target_dd_{int(dd*100)}pct": capital_path(z, cost_col, reward_col, entry_col, exit_col, dd) for dd in (0.10, 0.20, 0.30)}
    return {
        "n": int(len(z)), "pnl": float(z[reward_col].sum()), "cost_sum": float(z[cost_col].sum()),
        "roi": float(z[reward_col].sum() / z[cost_col].sum()), "edge_share": float(z[reward_col].sum() / (len(z) * SIZE)),
        "win_rate": float((z[reward_col] > 0).mean()),
        "iid_day_edge_ci95": base.iid_boot(edge_day),
        "mbb7_edge_ci95": base.mbb_boot(edge_day, 7),
        "mbb7_roi_ci95": base.mbb_ratio(dr, dc, 7),
        "newey_west_t_edge_lag7": base.nw_tstat(edge_day, 7),
        "by_month": month_out, "by_regime": by_regime, "annualization": ann,
    }


def convergence_stats(df: pd.DataFrame, prefix: str):
    z = df[df["regime"] == "large_convergence"].copy()
    if prefix == "strict":
        z = z[z["strict_cost"].notna()].copy()
        kind = z["strict_exit_kind"]
        hold = pd.to_numeric(z["strict_hold_s"], errors="coerce")
    else:
        kind = z["witness_exit_kind"]
        hold = pd.to_numeric(z["witness_hold_s"], errors="coerce")
    conv = kind == "convergence"
    conv_hold = hold[conv & hold.notna()]
    return {
        "eligible_large": int(len(z)), "convergence_exits": int(conv.sum()),
        "convergence_rate": float(conv.mean()) if len(z) else None,
        "within_60s": float((conv_hold <= 60).sum() / len(z)) if len(z) else None,
        "within_300s": float((conv_hold <= 300).sum() / len(z)) if len(z) else None,
        "median_convergence_s": float(conv_hold.median()) if len(conv_hold) else None,
        "settlement_fallbacks": int((kind == "settlement_fallback").sum()),
        "no_entry_fill": int((kind == "no_entry_fill").sum()) if prefix == "strict" else 0,
    }


def aggregate(root: Path, out: Path):
    files = sorted(root.glob("**/dual_exit_records.csv"))
    if len(files) != 4:
        raise RuntimeError(f"expected 4 shards, found {files}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = df.sort_values(["decision", "slug"], kind="mergesort")
    if len(df) != 7661:
        raise RuntimeError(f"authoritative signal count changed: {len(df)} != 7661")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dual_exit_records.csv", index=False)
    summary = {
        "protocol": PROTOCOL,
        "signals": int(len(df)),
        "regime_counts": df["regime"].value_counts().to_dict(),
        "witness_book": layer_stats(df, "witness"),
        "strict5": layer_stats(df, "strict"),
        "convergence_witness": convergence_stats(df, "witness"),
        "convergence_strict5": convergence_stats(df, "strict"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    lines = [
        "# Main Sequence corrected dual-exit — 93-day replay",
        "",
        f"Signals: {summary['signals']}; regimes: {summary['regime_counts']}",
        "",
        "## Witness-book layer",
        "```json", json.dumps(summary["witness_book"], indent=2), "```",
        "", "## Strict fresh-5s layer", "```json", json.dumps(summary["strict5"], indent=2), "```",
        "", "## Convergence", "```json", json.dumps({"witness": summary["convergence_witness"], "strict5": summary["convergence_strict5"]}, indent=2), "```",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("DUAL_EXIT_FINAL", json.dumps(summary, indent=2, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("protocol"); p.add_argument("--out", type=Path, required=True)
    s = sp.add_parser("score"); s.add_argument("--prior-shard", type=Path, required=True); s.add_argument("--anchor-dir", type=Path, required=True); s.add_argument("--out", type=Path, required=True); s.add_argument("--shard", required=True); s.add_argument("--workers", type=int, default=10)
    a = sp.add_parser("aggregate"); a.add_argument("--root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "protocol":
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "FROZEN_DUAL_EXIT_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))
        print("DUAL_EXIT_PROTOCOL_FROZEN", json.dumps(PROTOCOL, indent=2))
    elif args.cmd == "score":
        score_shard(args.prior_shard, args.anchor_dir, args.out, args.shard, args.workers)
    else:
        aggregate(args.root, args.out)


if __name__ == "__main__":
    main()
