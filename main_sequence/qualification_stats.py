from __future__ import annotations

import math

import numpy as np
import pandas as pd

BOOTSTRAP_REPS = 4000
BOOTSTRAP_SEED = 20260818
MIN_GREEN_DAYS = 20
MIN_GREEN_TRADES = 100


def daily_pnl_stats(df: pd.DataFrame, time_col: str, pnl_col: str, *, unit: str = "s", seed_offset: int = 0) -> dict:
    if df is None or df.empty:
        return {
            "trades": 0, "days": 0, "total_pnl": 0.0,
            "mean_daily_pnl": None, "median_daily_pnl": None,
            "positive_day_rate": None, "mean_daily_pnl_ci95": [None, None],
            "grade": "RED_EMPTY",
        }
    x = df[[time_col, pnl_col]].copy()
    x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    x[pnl_col] = pd.to_numeric(x[pnl_col], errors="coerce")
    x = x.dropna()
    if x.empty:
        return daily_pnl_stats(pd.DataFrame(), time_col, pnl_col, unit=unit, seed_offset=seed_offset)
    x["day"] = pd.to_datetime(x[time_col], unit=unit, utc=True).dt.floor("D")
    daily = x.groupby("day", sort=True)[pnl_col].sum().to_numpy(float)
    n_days = int(len(daily)); total = float(x[pnl_col].sum()); n = int(len(x))
    if n_days == 1:
        lo = hi = float(daily[0])
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset))
        means = np.empty(BOOTSTRAP_REPS, dtype=float)
        for i in range(BOOTSTRAP_REPS):
            means[i] = float(rng.choice(daily, size=n_days, replace=True).mean())
        lo, hi = map(float, np.quantile(means, [0.025, 0.975]))
    if total <= 0:
        grade = "RED"
    elif n_days < MIN_GREEN_DAYS or n < MIN_GREEN_TRADES:
        grade = "YELLOW_LOW_N"
    elif lo > 0:
        grade = "GREEN"
    else:
        grade = "YELLOW_CI"
    return {
        "trades": n,
        "days": n_days,
        "total_pnl": total,
        "mean_daily_pnl": float(np.mean(daily)),
        "median_daily_pnl": float(np.median(daily)),
        "positive_day_rate": float(np.mean(daily > 0)),
        "mean_daily_pnl_ci95": [lo, hi],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED + int(seed_offset),
        "green_gate": {"min_days": MIN_GREEN_DAYS, "min_trades": MIN_GREEN_TRADES, "ci95_lower_gt_zero": True, "total_pnl_gt_zero": True},
        "grade": grade,
    }
