from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

import prejuly_5m_official as core
import prejuly_5m_official_sealed as sealed
import prejuly_5m_official_stream as stream
import prejuly_5m_official_release as release  # applies corrected transport/statistics patches

TRAIN_START = "2026-02-01"
TRAIN_END = "2026-02-22"
VAL_START = "2026-02-22"
VAL_END = "2026-03-01"
QUARTER_TEST_START = "2026-03-01"
QUARTER_TEST_END = "2026-06-01"
QUARTER_DAYS = 92

SHARDS = {
    "mar": ["2026-03-01", "2026-04-01"],
    "apr": ["2026-04-01", "2026-05-01"],
    "may": ["2026-05-01", "2026-06-01"],
}

EVALUATION_CONTRACT = {
    "protocol_name": "Main Sequence quarterly OOS retrospective confirmation v1",
    "train": [TRAIN_START, TRAIN_END],
    "validation": [VAL_START, VAL_END],
    "quarter_test": [QUARTER_TEST_START, QUARTER_TEST_END],
    "quarter_days": QUARTER_DAYS,
    "shards": SHARDS,
    "primary_probability_comparator": "finance_p hard anchor",
    "secondary_probability_comparator": "PM last pre-decision trade; descriptive only",
    "probability_gate": (
        "GREEN iff aggregate Brier(finance)-Brier(model)>0, both 95% iid-day and "
        "7-day moving-block bootstrap lower bounds are >0, and at least 2/3 calendar "
        "months have positive finance-anchor Brier gain. RED iff aggregate gain<=0; otherwise YELLOW."
    ),
    "execution_gate": (
        "GREEN iff tape_5s aggregate edge/share>0, both 95% iid-day and 7-day moving-block "
        "bootstrap lower bounds for daily edge/share are >0, the 7-day moving-block ROI lower "
        "bound is >0, and at least 2/3 calendar months have positive tape edge/share. "
        "RED iff aggregate tape edge/share<=0; otherwise YELLOW."
    ),
    "execution_role": (
        "5-second public taker-BUY tape-compatible proxy, frozen ref+2c limit and original fee model; "
        "not queue/depth proof."
    ),
    "anti_tuning": (
        "Architecture, features, residual bound, 3c edge floor, +2c limit, 5s tape window and gates "
        "are frozen before the March-May 5m shard data are fetched by this workflow. July 1-14 is burned "
        "and is not used for fitting, validation, threshold selection, or gate selection."
    ),
    "interpretation": (
        "This is a 92-day causal historical OOS confirmation with Feb train/validation and Mar-May test. "
        "Because the architecture was developed later in calendar time, treat it as retrospective "
        "holdout evidence, not a prospective sealed quarter."
    ),
}


def _set_train_dates() -> None:
    core.TRAIN_START = TRAIN_START
    core.TRAIN_END = TRAIN_END
    core.VAL_START = VAL_START
    core.VAL_END = VAL_END
    core.TEST_START = QUARTER_TEST_START
    core.TEST_END = QUARTER_TEST_END


def _set_test_dates(start: str, end: str) -> None:
    core.TEST_START = start
    core.TEST_END = end


def _months_for_range(start: str, end: str) -> list[str]:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) - pd.Timedelta(seconds=1)
    p0 = lo.to_period("M") - 1
    p1 = hi.to_period("M")
    return [str(p) for p in pd.period_range(p0, p1, freq="M")]


def load_binance_range(start: str, end: str) -> dict[str, core.BinanceSeries]:
    months = _months_for_range(start, end)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-quarter-oos/1.0"})
    out: dict[str, core.BinanceSeries] = {}
    for asset, symbol in core.SYMBOLS.items():
        parts = []
        for ym in months:
            print("BINANCE_RANGE", asset, symbol, ym, flush=True)
            parts.append(core.load_binance_month(symbol, ym, sess))
        out[asset] = core.BinanceSeries(pd.concat(parts, ignore_index=True))
    return out


def _save_examples(xs, path: Path) -> None:
    rows = []
    for e in xs:
        r = {
            "slug": e.slug,
            "condition_id": e.condition_id,
            "asset": e.asset,
            "start": e.start,
            "decision": e.decision,
            "label": e.label,
            "pm_last": e.pm_last,
        }
        r.update({k: float(v) for k, v in zip(core.FEATURES, e.x)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_phase(out: Path) -> None:
    _set_train_dates()
    out.mkdir(parents=True, exist_ok=True)
    micros, cov = stream.fetch_phase_stream(TRAIN_START, VAL_END)
    bs = load_binance_range(TRAIN_START, VAL_END)
    xs = stream.build_stream_examples(micros, bs)
    train = [e for e in xs if core.ts(TRAIN_START) <= e.start < core.ts(TRAIN_END)]
    val = [e for e in xs if core.ts(VAL_START) <= e.start < core.ts(VAL_END)]
    if len(train) < 8000 or len(val) < 2000:
        raise RuntimeError(f"insufficient examples train={len(train)} val={len(val)} coverage={cov}")

    sc = core.Scaler.fit(train)
    models, metas, states = [], [], {}
    for seed in core.SEEDS:
        print("TRAIN_SEED", seed, flush=True)
        m, meta = sealed.train_one_finance_anchor(train, val, sc, seed)
        models.append(m)
        metas.append(meta)
        states[str(seed)] = m.state_dict()

    pv = sealed.predict_finance_anchor(models, val, sc)
    vsummary = core.probability_summary(val, pv)
    vboot = core.paired_day_bootstrap(val, pv)

    contract = {
        "train": [TRAIN_START, TRAIN_END],
        "validation": [VAL_START, VAL_END],
        "sealed_test": [QUARTER_TEST_START, QUARTER_TEST_END],
        "shards": SHARDS,
        "assets": list(core.ASSETS),
        "decision_s2c": core.DECISION_S2C,
        "fee_rate": core.FEE_RATE,
        "edge_floor": core.EDGE_FLOOR,
        "limit_slip": core.LIMIT_SLIP,
        "tape_seconds": core.TAPE_SECONDS,
        "residual_bound": core.RESIDUAL_BOUND,
        "seeds": list(core.SEEDS),
        "features": core.FEATURES,
        "model": (
            "causal Binance RV finance hard anchor + bounded +/-0.50 logit residual MLP; "
            "PM trade microstructure and Binance state are correction features"
        ),
        "hard_anchor": "finance_p: causal Binance 1m spot/open + trailing RV digital probability",
        "pm_last_role": "microstructure feature and execution-price reference; never the hard probability anchor",
        "transport": cov["transport"],
        "evaluation_contract": EVALUATION_CONTRACT,
        "isolation": (
            "Fit/validation requests only Feb 1-Mar 1 data. March-May test shards are fetched only "
            "in downstream jobs after this frozen artifact is uploaded."
        ),
    }

    np.savez(out / "scaler.npz", mean=sc.mean, std=sc.std)
    torch.save({"states": states, "features": core.FEATURES, "seeds": core.SEEDS}, out / "model.pt")
    (out / "FROZEN_CONTRACT.json").write_text(json.dumps(contract, indent=2))
    summary = {
        "name": "Main Sequence / quarterly OOS frozen Feb train",
        "coverage": cov,
        "counts": {
            "train": len(train),
            "validation": len(val),
            "train_by_asset": pd.Series([e.asset for e in train]).value_counts().to_dict(),
            "val_by_asset": pd.Series([e.asset for e in val]).value_counts().to_dict(),
        },
        "validation": vsummary,
        "validation_gain_day_bootstrap": vboot,
        "models": metas,
        "contract": contract,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _save_examples(train, out / "train_examples.csv")
    _save_examples(val, out / "validation_examples.csv")
    print("QUARTER_FROZEN_CONTRACT", json.dumps(contract, indent=2), flush=True)


def test_shard(model_dir: Path, out: Path, start: str, end: str, shard_name: str) -> None:
    if shard_name not in SHARDS or SHARDS[shard_name] != [start, end]:
        raise RuntimeError(f"undeclared shard {shard_name} {start} {end}")
    contract = json.loads((model_dir / "FROZEN_CONTRACT.json").read_text())
    if contract.get("evaluation_contract") != EVALUATION_CONTRACT:
        raise RuntimeError("evaluation contract mismatch")
    if contract.get("sealed_test") != [QUARTER_TEST_START, QUARTER_TEST_END]:
        raise RuntimeError("quarter test envelope mismatch")
    if contract.get("validation") != [VAL_START, VAL_END]:
        raise RuntimeError("validation window mismatch")
    if contract.get("shards", {}).get(shard_name) != [start, end]:
        raise RuntimeError("frozen shard mismatch")
    if contract.get("features") != core.FEATURES:
        raise RuntimeError("feature contract mismatch")
    if not str(contract.get("hard_anchor", "")).startswith("finance_p: causal Binance"):
        raise RuntimeError("hard anchor mismatch")

    _set_test_dates(start, end)
    out.mkdir(parents=True, exist_ok=True)
    print("QUARTER_SHARD_CONTRACT_OK", shard_name, start, end, flush=True)

    micros, cov = stream.fetch_phase_stream(start, end)
    bs = load_binance_range(start, end)
    test = [e for e in stream.build_stream_examples(micros, bs) if core.ts(start) <= e.start < core.ts(end)]

    expected_days = pd.date_range(
        start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D", tz="UTC"
    ).strftime("%Y-%m-%d").tolist()
    got_days = sorted({
        pd.Timestamp(e.start, unit="s", tz="UTC").strftime("%Y-%m-%d") for e in test
    })
    missing = [d for d in expected_days if d not in got_days]
    if len(test) < 5000 or missing:
        raise RuntimeError(
            f"quarter shard coverage insufficient n={len(test)} missing_days={missing} coverage={cov}"
        )

    z = np.load(model_dir / "scaler.npz")
    sc = core.Scaler(z["mean"], z["std"])
    ck = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    models = []
    for seed in ck["seeds"]:
        m = core.ResidualMLP(len(core.FEATURES))
        m.load_state_dict(ck["states"][str(seed)])
        models.append(m)

    p = sealed.predict_finance_anchor(models, test, sc)
    ps = core.probability_summary(test, p)
    boot = core.paired_day_bootstrap(test, p)
    policy, pdf = stream.policy_summary_stream(test, p)
    fi = core.FEATURES.index("finance_p")

    by_asset = {}
    for asset in core.ASSETS:
        idx = [i for i, e in enumerate(test) if e.asset == asset]
        by_asset[asset] = core.probability_summary(
            [test[i] for i in idx], p[idx]
        ) if idx else {"n": 0}

    summary = {
        "name": f"Main Sequence quarterly OOS shard {shard_name}",
        "shard": {"name": shard_name, "start": start, "end": end},
        "coverage": cov,
        "counts": {
            "test": len(test),
            "by_asset": pd.Series([e.asset for e in test]).value_counts().to_dict(),
            "missing_calendar_days": missing,
        },
        "probability": ps,
        "brier_gain_day_bootstrap": boot,
        "by_asset": by_asset,
        "policy": policy,
        "evaluation_contract": EVALUATION_CONTRACT,
    }
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _save_examples(test, out / "test_examples.csv")
    pd.DataFrame({
        "slug": [e.slug for e in test],
        "asset": [e.asset for e in test],
        "start": [e.start for e in test],
        "decision": [e.decision for e in test],
        "label": [e.label for e in test],
        "pm_last": [e.pm_last for e in test],
        "finance_p": [float(e.x[fi]) for e in test],
        "model_p": p,
        "shard": shard_name,
    }).to_csv(out / "test_probabilities.csv", index=False)
    pdf["shard"] = shard_name
    pdf.to_csv(out / "policy_trades.csv", index=False)
    print("QUARTER_SHARD_DONE", shard_name, "examples", len(test), "signals", policy["signals"], flush=True)


def _brier(y, p) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((y - p) ** 2))


def _logloss(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            out += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)


def _iid_boot(x: np.ndarray, reps: int = 20000, seed: int = 20260815) -> list[float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    z = np.empty(reps, float)
    for i in range(reps):
        z[i] = rng.choice(x, len(x), replace=True).mean()
    return [float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))]


def _mbb_boot(x: np.ndarray, block: int = 7, reps: int = 20000, seed: int = 20260816) -> list[float]:
    x = np.asarray(x, float)
    n = len(x)
    if n < block:
        return _iid_boot(x, reps=reps, seed=seed)
    starts = np.arange(n - block + 1)
    k = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    z = np.empty(reps, float)
    for i in range(reps):
        picks = rng.choice(starts, k, replace=True)
        sample = np.concatenate([x[j:j + block] for j in picks])[:n]
        z[i] = sample.mean()
    return [float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))]


def _mbb_ratio(
    reward: np.ndarray,
    cost: np.ndarray,
    block: int = 7,
    reps: int = 20000,
    seed: int = 20260817,
) -> list[float]:
    reward = np.asarray(reward, float)
    cost = np.asarray(cost, float)
    n = len(reward)
    if n < block:
        block = 1
    starts = np.arange(n - block + 1)
    k = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    z = np.empty(reps, float)
    for i in range(reps):
        picks = rng.choice(starts, k, replace=True)
        idx = np.concatenate([np.arange(j, j + block) for j in picks])[:n]
        denom = cost[idx].sum()
        z[i] = reward[idx].sum() / denom if denom > 0 else np.nan
    z = z[np.isfinite(z)]
    return [float(np.quantile(z, 0.025)), float(np.quantile(z, 0.975))]


def _nw_tstat(x: np.ndarray, lag: int = 7) -> float | None:
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return None
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    L = min(lag, n - 1)
    for ell in range(1, L + 1):
        gamma = float(np.dot(u[ell:], u[:-ell]) / n)
        weight = 1.0 - ell / (L + 1.0)
        lrv += 2.0 * weight * gamma
    if lrv <= 0:
        return None
    se = math.sqrt(lrv / n)
    return float(x.mean() / se) if se > 0 else None


def _max_drawdown_for_capital(cum_pnl: np.ndarray, capital: float) -> float:
    equity = capital + np.concatenate([[0.0], np.asarray(cum_pnl, float)])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def _capital_for_target_dd(
    slot_cost: np.ndarray,
    slot_reward: np.ndarray,
    target_dd: float,
) -> tuple[float, float]:
    reward = np.asarray(slot_reward, float)
    cost = np.asarray(slot_cost, float)
    pnl_before = np.concatenate([[0.0], np.cumsum(reward)[:-1]])
    min_solvency = max(1e-9, float(np.max(cost - pnl_before)))
    cum = np.cumsum(reward)
    if _max_drawdown_for_capital(cum, min_solvency) <= target_dd:
        return min_solvency, _max_drawdown_for_capital(cum, min_solvency)
    lo = min_solvency
    hi = max(2 * lo, 1.0)
    while _max_drawdown_for_capital(cum, hi) > target_dd:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _max_drawdown_for_capital(cum, mid) > target_dd:
            lo = mid
        else:
            hi = mid
    return hi, _max_drawdown_for_capital(cum, hi)


def _trade_summary(z: pd.DataFrame) -> dict:
    if z.empty:
        return {"n": 0, "edge_share": None, "roi": None, "win_rate": None}
    reward = z["tape_reward"].astype(float)
    cost = z["tape_cost"].astype(float)
    won = z["won"]
    if won.dtype == object:
        won = won.astype(str).str.lower().isin(["true", "1", "yes"])
    return {
        "n": int(len(z)),
        "edge_share": float(reward.mean()),
        "roi": float(reward.sum() / cost.sum()),
        "win_rate": float(won.astype(float).mean()),
        "pnl": float(reward.sum()),
        "cost_sum": float(cost.sum()),
    }


def aggregate_phase(shard_root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prob_files = sorted(shard_root.glob("**/test_probabilities.csv"))
    trade_files = sorted(shard_root.glob("**/policy_trades.csv"))
    if len(prob_files) != 3 or len(trade_files) != 3:
        raise RuntimeError(f"expected 3 probability/trade shards, got {len(prob_files)} / {len(trade_files)}")

    probs = pd.concat([pd.read_csv(p) for p in prob_files], ignore_index=True)
    trades = pd.concat([pd.read_csv(p) for p in trade_files], ignore_index=True)
    if probs["slug"].duplicated().any():
        raise RuntimeError("duplicate probability rows across quarter shards")
    probs = probs.sort_values(["start", "asset"]).reset_index(drop=True)
    trades = trades.sort_values(["decision", "asset"]).reset_index(drop=True)

    days = pd.to_datetime(probs["start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    expected_days = pd.date_range(
        QUARTER_TEST_START,
        pd.Timestamp(QUARTER_TEST_END) - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    ).strftime("%Y-%m-%d").tolist()
    got_days = sorted(days.unique().tolist())
    missing_days = sorted(set(expected_days) - set(got_days))
    if missing_days or len(got_days) != QUARTER_DAYS:
        raise RuntimeError(f"quarter day coverage RED: got={len(got_days)} missing={missing_days}")

    y = probs["label"].to_numpy(float)
    pm = probs["pm_last"].to_numpy(float)
    fi = probs["finance_p"].to_numpy(float)
    mp = probs["model_p"].to_numpy(float)

    prob_summary = {
        "n": int(len(probs)),
        "days": int(len(got_days)),
        "brier_model": _brier(y, mp),
        "brier_pm_last": _brier(y, pm),
        "brier_finance": _brier(y, fi),
        "brier_gain_vs_pm_last": _brier(y, pm) - _brier(y, mp),
        "brier_gain_vs_finance": _brier(y, fi) - _brier(y, mp),
        "logloss_model": _logloss(y, mp),
        "logloss_pm_last": _logloss(y, pm),
        "logloss_finance": _logloss(y, fi),
        "logloss_gain_vs_finance": _logloss(y, fi) - _logloss(y, mp),
        "ece_model": _ece(y, mp),
        "ece_pm_last": _ece(y, pm),
    }

    q = probs.assign(
        day=days,
        gain_fin=(y - fi) ** 2 - (y - mp) ** 2,
        gain_pm=(y - pm) ** 2 - (y - mp) ** 2,
        month=pd.to_datetime(probs["start"], unit="s", utc=True).dt.strftime("%Y-%m"),
    )
    daily_fin = q.groupby("day", sort=True)["gain_fin"].mean().reindex(expected_days).to_numpy(float)
    daily_pm = q.groupby("day", sort=True)["gain_pm"].mean().reindex(expected_days).to_numpy(float)
    month_prob = {}
    for month, g in q.groupby("month", sort=True):
        gy = g["label"].to_numpy(float)
        gm = g["model_p"].to_numpy(float)
        gf = g["finance_p"].to_numpy(float)
        gp = g["pm_last"].to_numpy(float)
        month_prob[month] = {
            "n": int(len(g)),
            "brier_gain_vs_finance": _brier(gy, gf) - _brier(gy, gm),
            "brier_gain_vs_pm_last": _brier(gy, gp) - _brier(gy, gm),
        }

    prob_inference = {
        "vs_finance": {
            "daily_mean_gain": float(daily_fin.mean()),
            "iid_day_ci95": _iid_boot(daily_fin),
            "mbb7_ci95": _mbb_boot(daily_fin),
            "newey_west_t_lag7": _nw_tstat(daily_fin, 7),
        },
        "vs_pm_last": {
            "daily_mean_gain": float(daily_pm.mean()),
            "iid_day_ci95": _iid_boot(daily_pm, seed=20260818),
            "mbb7_ci95": _mbb_boot(daily_pm, seed=20260819),
            "newey_west_t_lag7": _nw_tstat(daily_pm, 7),
        },
    }
    positive_prob_months = sum(v["brier_gain_vs_finance"] > 0 for v in month_prob.values())
    pgain = prob_summary["brier_gain_vs_finance"]
    p_iid_lo = prob_inference["vs_finance"]["iid_day_ci95"][0]
    p_mbb_lo = prob_inference["vs_finance"]["mbb7_ci95"][0]
    if pgain <= 0:
        prob_verdict = "RED"
    elif p_iid_lo > 0 and p_mbb_lo > 0 and positive_prob_months >= 2:
        prob_verdict = "GREEN"
    else:
        prob_verdict = "YELLOW"

    tape = trades[trades["tape_fill"].notna() & trades["tape_reward"].notna() & trades["tape_cost"].notna()].copy()
    tape["day"] = pd.to_datetime(tape["decision"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    tape["month"] = pd.to_datetime(tape["decision"], unit="s", utc=True).dt.strftime("%Y-%m")
    tape_daily = tape.groupby("day", sort=True).agg(
        edge=("tape_reward", "mean"),
        reward_sum=("tape_reward", "sum"),
        cost_sum=("tape_cost", "sum"),
        n=("tape_reward", "size"),
    ).reindex(expected_days, fill_value=0.0)
    daily_edge = tape_daily["edge"].to_numpy(float)
    daily_reward = tape_daily["reward_sum"].to_numpy(float)
    daily_cost = tape_daily["cost_sum"].to_numpy(float)

    tape_summary = _trade_summary(tape)
    tape_summary.update({
        "days": QUARTER_DAYS,
        "fill_rate": float(len(tape) / len(trades)) if len(trades) else None,
        "iid_day_edge_ci95": _iid_boot(daily_edge, seed=20260820),
        "mbb7_edge_ci95": _mbb_boot(daily_edge, seed=20260821),
        "mbb7_roi_ci95": _mbb_ratio(daily_reward, daily_cost, seed=20260822),
        "newey_west_t_edge_lag7": _nw_tstat(daily_edge, 7),
    })

    month_tape = {}
    for month in sorted(q["month"].unique()):
        z = tape[tape["month"] == month]
        month_tape[month] = _trade_summary(z)
    positive_tape_months = sum(
        (v["edge_share"] is not None and v["edge_share"] > 0) for v in month_tape.values()
    )
    e = tape_summary["edge_share"]
    if e is None or e <= 0:
        exec_verdict = "RED"
    elif (
        tape_summary["iid_day_edge_ci95"][0] > 0
        and tape_summary["mbb7_edge_ci95"][0] > 0
        and tape_summary["mbb7_roi_ci95"][0] > 0
        and positive_tape_months >= 2
    ):
        exec_verdict = "GREEN"
    else:
        exec_verdict = "YELLOW"

    by_asset_tape = {
        asset: _trade_summary(tape[tape["asset"] == asset]) for asset in core.ASSETS
    }

    slot = tape.groupby("decision", sort=True).agg(
        cost=("tape_cost", "sum"),
        reward=("tape_reward", "sum"),
    )
    capital = {}
    total_pnl = float(slot["reward"].sum())
    if len(slot):
        reward = slot["reward"].to_numpy(float)
        cost = slot["cost"].to_numpy(float)
        pnl_before = np.concatenate([[0.0], np.cumsum(reward)[:-1]])
        min_solvency = max(1e-9, float(np.max(cost - pnl_before)))
        min_dd = _max_drawdown_for_capital(np.cumsum(reward), min_solvency)
        min_ret = total_pnl / min_solvency
        capital["minimum_solvency"] = {
            "capital": min_solvency,
            "period_return": min_ret,
            "max_drawdown": min_dd,
            "simple_annualized": min_ret * 365.0 / QUARTER_DAYS,
            "cagr": ((1 + min_ret) ** (365.0 / QUARTER_DAYS) - 1) if min_ret > -1 else None,
        }
        for target in (0.10, 0.20, 0.30):
            cap, dd = _capital_for_target_dd(cost, reward, target)
            ret = total_pnl / cap
            capital[f"target_dd_{int(target * 100)}pct"] = {
                "capital": cap,
                "period_return": ret,
                "max_drawdown": dd,
                "simple_annualized": ret * 365.0 / QUARTER_DAYS,
                "cagr": ((1 + ret) ** (365.0 / QUARTER_DAYS) - 1) if ret > -1 else None,
            }

    summary = {
        "name": "Main Sequence quarterly OOS aggregate / Mar-May 2026",
        "period": [QUARTER_TEST_START, QUARTER_TEST_END],
        "days": QUARTER_DAYS,
        "evaluation_contract": EVALUATION_CONTRACT,
        "probability": prob_summary,
        "probability_inference": prob_inference,
        "probability_by_month": month_prob,
        "positive_probability_months_vs_finance": positive_prob_months,
        "probability_verdict": prob_verdict,
        "execution_tape": tape_summary,
        "execution_by_month": month_tape,
        "execution_by_asset": by_asset_tape,
        "positive_execution_months": positive_tape_months,
        "execution_verdict": exec_verdict,
        "capital_path_one_share": capital,
        "notes": {
            "annualization": (
                "Drawdown-normalized implied annualization from the actual one-share tape-compatible "
                "PnL path. Capital is solved ex post to satisfy the stated historical max-DD target; "
                "therefore it is a descriptive risk-normalized statistic, not a deployable forward guarantee."
            ),
            "dependence": (
                "Primary confidence uses 92 calendar-day blocks plus a 7-day moving-block bootstrap; "
                "Newey-West lag-7 t-stat is diagnostic only."
            ),
        },
    }
    (out / "quarter_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    probs.to_csv(out / "quarter_probabilities.csv", index=False)
    trades.to_csv(out / "quarter_policy_trades.csv", index=False)
    lines = [
        "# Main Sequence — 92-day quarterly OOS aggregate",
        "",
        f"Period: {QUARTER_TEST_START} to {QUARTER_TEST_END} (exclusive), {QUARTER_DAYS} days.",
        "",
        f"Probability verdict: **{prob_verdict}**",
        f"Execution tape verdict: **{exec_verdict}**",
        "",
        "## Probability",
        "```json",
        json.dumps(prob_summary, indent=2),
        "```",
        "",
        "## Probability inference",
        "```json",
        json.dumps(prob_inference, indent=2),
        "```",
        "",
        "## Tape execution",
        "```json",
        json.dumps(tape_summary, indent=2),
        "```",
        "",
        "## Capital / implied annualization",
        "```json",
        json.dumps(capital, indent=2),
        "```",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print((out / "SUMMARY.md").read_text(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--out", type=Path, required=True)

    p_test = sub.add_parser("test")
    p_test.add_argument("--model-dir", type=Path, required=True)
    p_test.add_argument("--out", type=Path, required=True)
    p_test.add_argument("--start", required=True)
    p_test.add_argument("--end", required=True)
    p_test.add_argument("--shard-name", required=True)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--shard-root", type=Path, required=True)
    p_agg.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.phase == "train":
        train_phase(args.out)
    elif args.phase == "test":
        test_shard(args.model_dir, args.out, args.start, args.end, args.shard_name)
    else:
        aggregate_phase(args.shard_root, args.out)


if __name__ == "__main__":
    main()
