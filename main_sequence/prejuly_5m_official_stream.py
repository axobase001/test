from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

import prejuly_5m_official as core
import prejuly_5m_official_retryfix as retryfix
import prejuly_5m_official_sealed as sealed

# The statistical protocol remains entirely in core/sealed.  This file only
# streams/aggregates rows that core.build_examples would otherwise retain in RAM.


@dataclass
class Micro:
    market: core.Market
    micro14: np.ndarray
    fill_up: tuple[tuple[int, float], ...]
    fill_down: tuple[tuple[int, float], ...]


@dataclass
class StreamExample:
    slug: str
    condition_id: str
    asset: str
    start: int
    decision: int
    label: float
    pm_last: float
    x: np.ndarray
    fill_up: tuple[tuple[int, float], ...]
    fill_down: tuple[tuple[int, float], ...]


def _query_rows(sess: requests.Session, markets: list[core.Market], start: int, end: int) -> list[dict]:
    """Fetch the exact condition/time rectangle, losslessly splitting 500/cap cases."""
    if not markets:
        return []
    q = {
        "market": ",".join(m.condition_id for m in markets),
        "start": start,
        "end": end,
        "limit": 10000,
        "offset": 0,
        "takerOnly": "true",
    }
    try:
        r = sess.get(core.DATA_API + "/trades", params=q, timeout=60)
    except requests.RequestException:
        r = None
    if r is not None and r.status_code == 200:
        rows = r.json()
    elif r is not None and r.status_code == 500 and len(markets) > 1:
        mid = len(markets) // 2
        return _query_rows(sess, markets[:mid], start, end) + _query_rows(sess, markets[mid:], start, end)
    else:
        rows = core.get_json(sess, core.DATA_API + "/trades", params=q, timeout=60)

    if len(rows) < 10000:
        return rows
    if len(markets) > 1:
        mid = len(markets) // 2
        return _query_rows(sess, markets[:mid], start, end) + _query_rows(sess, markets[mid:], start, end)
    if end <= start:
        raise RuntimeError(f"single-second trade cap for {markets[0].slug} at {start}")
    mid_t = (start + end) // 2
    return _query_rows(sess, markets, start, mid_t) + _query_rows(sess, markets, mid_t + 1, end)


def _aggregate_market(m: core.Market, trade_map: dict[str, pd.DataFrame]) -> Micro | None:
    g = trade_map.get(m.condition_id)
    if g is None or g.empty:
        return None
    # Exact same eligibility and feature math as core.build_examples.
    pre = g[(g.timestamp >= m.start) & (g.timestamp < m.decision)].copy()
    if len(pre) < 4:
        return None
    lag = m.decision - int(pre.timestamp.iloc[-1])
    if lag < 0 or lag > 45:
        return None
    pm = float(pre.p_up.iloc[-1])
    if not (0.01 < pm < 0.99):
        return None

    def w(sec: int):
        return pre[pre.timestamp >= m.decision - sec]

    def mom(sec: int) -> float:
        q = w(sec)
        return float(pm - q.p_up.iloc[0]) if len(q) >= 2 else 0.0

    q30, q60, q120 = w(30), w(60), w(120)
    size60 = float(q60["size"].sum()) if len(q60) else 0.0
    press60 = float((q60.pressure * q60["size"]).sum() / (size60 + 1e-9)) if len(q60) else 0.0
    vals = q60.p_up.to_numpy(float) if len(q60) else np.asarray([pm])
    micro14 = np.asarray([
        pm,
        float(lag),
        math.log1p(len(q30)),
        math.log1p(len(q60)),
        math.log1p(len(q120)),
        math.log1p(size60),
        press60,
        mom(15),
        mom(30),
        mom(60),
        mom(120),
        float(np.std(vals)) if len(vals) > 1 else 0.0,
        float(np.max(vals) - np.min(vals)),
        float(np.mean(vals) - pm),
    ], dtype=np.float32)

    post = g[(g.timestamp >= m.decision) & (g.timestamp <= m.decision + core.TAPE_SECONDS) & (g.side_u == "BUY")]
    up = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "up"].itertuples(index=False))
    down = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "down"].itertuples(index=False))
    return Micro(m, micro14, up, down)


def fetch_hour_stream(hour_start: int) -> tuple[list[Micro], dict]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-sealed-stream/1.0"})
    wanted = []
    for a in core.ASSETS:
        for t0 in range(hour_start, hour_start + 3600, 300):
            wanted.append((f"{a.lower()}-updown-5m-{t0}", a, t0))
    params = [("slug", x[0]) for x in wanted] + [("closed", "true"), ("limit", 100)]
    js = core.get_json(sess, core.GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js}
    by_start: dict[int, list[core.Market]] = {t: [] for t in range(hour_start, hour_start + 3600, 300)}
    mapped = 0
    for slug, asset, t0 in wanted:
        m = core.parse_market(byslug.get(slug, {}), asset, t0)
        if m is not None:
            by_start[t0].append(m)
            mapped += 1

    micros: list[Micro] = []
    raw_rows = 0
    slot_failures = []
    for t0, markets in by_start.items():
        if not markets:
            continue
        # This is exactly the only PM-tape interval consumed by core.build_examples
        # plus policy_summary: feature eligibility starts at market start, and
        # execution proxy ends at decision + 5 seconds.
        try:
            rows = _query_rows(sess, markets, t0, t0 + 240 + core.TAPE_SECONDS)
            raw_rows += len(rows)
            td = pd.DataFrame(rows)
            tm = core.normalize_trade_rows(td)
            for m in markets:
                z = _aggregate_market(m, tm)
                if z is not None:
                    micros.append(z)
        except Exception as e:
            slot_failures.append({"slot": t0, "error": repr(e)})
    return micros, {"mapped": mapped, "raw_rows": raw_rows, "slot_failures": slot_failures}


def fetch_phase_stream(start: str, end: str, workers: int = 20):
    lo, hi = core.ts(start), core.ts(end)
    hours = list(range(lo, hi, 3600))
    micros: list[Micro] = []
    failures = []
    mapped = 0
    raw_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fetch_hour_stream, h): h for h in hours}
        for k, f in enumerate(as_completed(fut), 1):
            h = fut[f]
            try:
                mm, meta = f.result()
                micros.extend(mm)
                mapped += int(meta["mapped"])
                raw_rows += int(meta["raw_rows"])
                if meta["slot_failures"]:
                    failures.append({"hour": h, "slot_failures": meta["slot_failures"]})
            except Exception as e:
                failures.append({"hour": h, "error": repr(e)})
            if k % 48 == 0:
                print("STREAM_HOURS", k, "/", len(hours), "mapped", mapped, "micro", len(micros), "raw_rows", raw_rows, "fail", len(failures), flush=True)
    micros.sort(key=lambda z: (z.market.start, z.market.asset))
    cov = {
        "hours": len(hours),
        "failed_hours": failures,
        "expected_markets": len(hours) * 12 * len(core.ASSETS),
        "mapped_markets": mapped,
        "micro_eligible": len(micros),
        "raw_trade_rows_streamed": raw_rows,
        "transport": "per-5m-slot start..decision+5s; aggregate immediately; no raw month tape retained",
    }
    return micros, cov


def build_stream_examples(micros: list[Micro], bs: dict[str, core.BinanceSeries]) -> list[StreamExample]:
    out = []
    for z in micros:
        m = z.market
        bf = bs[m.asset].features(m.start, m.decision)
        if bf is None:
            continue
        feat = np.asarray([
            *z.micro14.tolist(),
            bf["finance_p"],
            bf["finance_p"] - float(z.micro14[0]),
            bf["log_rel_spot"],
            bf["ret1m"],
            bf["ret3m"],
            bf["rv60"],
            float(m.asset == "BTC"),
            float(m.asset == "ETH"),
            float(m.asset == "SOL"),
            float(m.asset == "XRP"),
        ], dtype=np.float32)
        if np.isfinite(feat).all():
            out.append(StreamExample(m.slug, m.condition_id, m.asset, m.start, m.decision, m.label_up,
                                     float(z.micro14[0]), feat, z.fill_up, z.fill_down))
    out.sort(key=lambda e: (e.start, e.asset))
    return out


def _summ(z: pd.DataFrame, reward: str, cost: str):
    if z.empty:
        return {"n": 0, "edge_share": None, "roi": None, "win_rate": None, "ci95_day": [None, None]}
    vals = z[reward].astype(float)
    costs = z[cost].astype(float)
    day = pd.to_datetime(z.decision, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    daily = pd.DataFrame({"day": day, "r": vals}).groupby("day").r.mean().to_numpy()
    rng = np.random.default_rng(20260815)
    boot = np.asarray([rng.choice(daily, len(daily), replace=True).mean() for _ in range(4000)]) if len(daily) > 1 else daily
    return {"n": int(len(z)), "edge_share": float(vals.mean()), "roi": float(vals.sum() / costs.sum()),
            "win_rate": float(z.won.mean()), "ci95_day": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def policy_summary_stream(xs: list[StreamExample], p: np.ndarray):
    rows = []
    for e, prob in zip(xs, p):
        up_ref = e.pm_last
        dn_ref = 1 - up_ref
        up_edge = float(prob) - up_ref - core.fee_per_share(up_ref)
        dn_edge = (1 - float(prob)) - dn_ref - core.fee_per_share(dn_ref)
        if max(up_edge, dn_edge) < core.EDGE_FLOOR:
            continue
        choose_up = up_edge >= dn_edge
        ref = up_ref if choose_up else dn_ref
        fair = float(prob) if choose_up else 1 - float(prob)
        limit = min(0.99, ref + core.LIMIT_SLIP)
        won = (e.label >= .5) == choose_up
        paper_cost = ref + core.fee_per_share(ref)
        paper_reward = (1.0 if won else 0.0) - paper_cost
        tape = e.fill_up if choose_up else e.fill_down
        fill = None
        for _, px in tape:
            if px <= limit:
                fill = float(px)
                break
        tape_cost = None
        tape_reward = None
        if fill is not None:
            tape_cost = fill + core.fee_per_share(fill)
            tape_reward = (1.0 if won else 0.0) - tape_cost
        rows.append({"slug": e.slug, "asset": e.asset, "decision": e.decision,
                     "side": "Up" if choose_up else "Down", "fair": fair, "ref": ref,
                     "signal_edge": max(up_edge, dn_edge), "limit": limit, "won": won,
                     "paper_cost": paper_cost, "paper_reward": paper_reward,
                     "tape_fill": fill, "tape_cost": tape_cost, "tape_reward": tape_reward})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"signals": 0, "paper": _summ(df, "paper_reward", "paper_cost"),
                "tape_5s": _summ(df, "paper_reward", "paper_cost")}, df
    tape = df[df.tape_fill.notna()].copy()
    out = {"signals": int(len(df)), "tape_fill_rate": float(len(tape) / len(df)),
           "paper": _summ(df, "paper_reward", "paper_cost"), "tape_5s": _summ(tape, "tape_reward", "tape_cost")}
    out["by_asset_tape"] = {a: _summ(tape[tape.asset == a], "tape_reward", "tape_cost") for a in core.ASSETS}
    return out, df


def save_stream_examples(xs, path: Path):
    rows = []
    for e in xs:
        r = {"slug": e.slug, "condition_id": e.condition_id, "asset": e.asset, "start": e.start,
             "decision": e.decision, "label": e.label, "pm_last": e.pm_last}
        r.update({k: float(v) for k, v in zip(core.FEATURES, e.x)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_phase(args):
    args.out.mkdir(parents=True, exist_ok=True)
    micros, cov = fetch_phase_stream(core.TRAIN_START, core.VAL_END)
    bs = core.load_binance("train")
    xs = build_stream_examples(micros, bs)
    train = [e for e in xs if core.ts(core.TRAIN_START) <= e.start < core.ts(core.TRAIN_END)]
    val = [e for e in xs if core.ts(core.VAL_START) <= e.start < core.ts(core.VAL_END)]
    if len(train) < 8000 or len(val) < 2000:
        raise RuntimeError(f"insufficient examples train={len(train)} val={len(val)} coverage={cov}")
    sc = core.Scaler.fit(train)
    models, metas, states = [], [], {}
    for seed in core.SEEDS:
        print("TRAIN_SEED", seed, flush=True)
        m, meta = sealed.train_one_finance_anchor(train, val, sc, seed)
        models.append(m); metas.append(meta); states[str(seed)] = m.state_dict()
    pv = sealed.predict_finance_anchor(models, val, sc)
    vsummary = core.probability_summary(val, pv)
    contract = {
        "train": [core.TRAIN_START, core.TRAIN_END], "validation": [core.VAL_START, core.VAL_END],
        "sealed_test": [core.TEST_START, core.TEST_END], "assets": list(core.ASSETS),
        "decision_s2c": core.DECISION_S2C, "fee_rate": core.FEE_RATE, "edge_floor": core.EDGE_FLOOR,
        "limit_slip": core.LIMIT_SLIP, "tape_seconds": core.TAPE_SECONDS,
        "residual_bound": core.RESIDUAL_BOUND, "seeds": list(core.SEEDS), "features": core.FEATURES,
        "model": "causal Binance RV finance hard anchor + bounded +/-0.50 logit residual MLP; PM trade microstructure and Binance state are correction features",
        "hard_anchor": "finance_p: causal Binance 1m spot/open + trailing RV digital probability",
        "pm_last_role": "microstructure feature and execution-price reference; never the hard probability anchor",
        "isolation": "train phase requests only June market/trade/Binance data; July is first requested by downstream sealed test job",
        "transport": cov["transport"],
    }
    np.savez(args.out / "scaler.npz", mean=sc.mean, std=sc.std)
    torch.save({"states": states, "features": core.FEATURES, "seeds": core.SEEDS}, args.out / "model.pt")
    (args.out / "FROZEN_CONTRACT.json").write_text(json.dumps(contract, indent=2))
    summary = {"name": "Main Sequence B / pre-July 5m multiasset frozen train", "coverage": cov,
               "counts": {"train": len(train), "validation": len(val),
                          "train_by_asset": pd.Series([e.asset for e in train]).value_counts().to_dict(),
                          "val_by_asset": pd.Series([e.asset for e in val]).value_counts().to_dict()},
               "validation": vsummary, "validation_gain_day_bootstrap": core.paired_day_bootstrap(val, pv),
               "models": metas, "contract": contract,
               "implementation_attestation": "Streaming transport retains exactly the PM-tape interval consumed by frozen features/policy and aggregates before month-level retention; statistical protocol unchanged."}
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    save_stream_examples(train, args.out / "train_examples.csv")
    save_stream_examples(val, args.out / "validation_examples.csv")
    print(json.dumps(summary, indent=2, default=float), flush=True)


def test_phase(args):
    args.out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((args.model_dir / "FROZEN_CONTRACT.json").read_text())
    assert contract["sealed_test"] == [core.TEST_START, core.TEST_END]
    assert contract["validation"][1] == core.TEST_START
    assert contract["features"] == core.FEATURES
    assert contract["hard_anchor"].startswith("finance_p: causal Binance")
    micros, cov = fetch_phase_stream(core.TEST_START, core.TEST_END)
    bs = core.load_binance("test")
    test = [e for e in build_stream_examples(micros, bs) if core.ts(core.TEST_START) <= e.start < core.ts(core.TEST_END)]
    expected_days = pd.date_range(core.TEST_START, pd.Timestamp(core.TEST_END) - pd.Timedelta(days=1), freq="D", tz="UTC").strftime("%Y-%m-%d").tolist()
    got_days = sorted(set(pd.Timestamp(e.start, unit="s", tz="UTC").strftime("%Y-%m-%d") for e in test))
    missing = [d for d in expected_days if d not in got_days]
    if len(test) < 5000 or missing:
        raise RuntimeError(f"sealed test coverage insufficient n={len(test)} missing_days={missing} coverage={cov}")
    z = np.load(args.model_dir / "scaler.npz")
    sc = core.Scaler(z["mean"], z["std"])
    ck = torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=False)
    models = []
    for seed in ck["seeds"]:
        m = core.ResidualMLP(len(core.FEATURES)); m.load_state_dict(ck["states"][str(seed)]); models.append(m)
    p = sealed.predict_finance_anchor(models, test, sc)
    ps = core.probability_summary(test, p)
    boot = core.paired_day_bootstrap(test, p)
    policy, pdf = policy_summary_stream(test, p)
    by_asset = {}
    for a in core.ASSETS:
        idx = [i for i, e in enumerate(test) if e.asset == a]
        by_asset[a] = core.probability_summary([test[i] for i in idx], p[idx]) if idx else {"n": 0}
    summary = {"name": "Main Sequence B / SEALED July 1-14 5m multiasset test", "coverage": cov,
               "counts": {"test": len(test), "by_asset": pd.Series([e.asset for e in test]).value_counts().to_dict(),
                          "missing_calendar_days": missing},
               "probability": ps, "brier_gain_day_bootstrap": boot, "by_asset": by_asset, "policy": policy,
               "execution_note": "paper uses last pre-decision public trade as reference; tape_5s requires a real public taker BUY print in the chosen outcome within 5 seconds at or below frozen ref+2c limit; tape-compatible proxy, not queue/depth proof.",
               "leakage_note": "all PM trade features are < decision; Binance uses completed 1m bars ending <= decision; July was absent from frozen train job."}
    (args.out / "test_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    save_stream_examples(test, args.out / "test_examples.csv")
    pd.DataFrame({"slug": [e.slug for e in test], "asset": [e.asset for e in test], "start": [e.start for e in test],
                  "label": [e.label for e in test], "pm_last": [e.pm_last for e in test], "model_p": p}).to_csv(args.out / "test_probabilities.csv", index=False)
    pdf.to_csv(args.out / "policy_trades.csv", index=False)
    lines = ["# Main Sequence B — sealed July 1–14 test", "", f"Test examples: {len(test)} / mapped markets {cov['mapped_markets']}.", "",
             "## Probability", "", "```json", json.dumps(ps, indent=2), "```", "", "## Day-block Brier gain vs PM last trade", "", "```json", json.dumps(boot, indent=2), "```", "", "## Policy", "", "```json", json.dumps(policy, indent=2, default=float), "```", ""]
    (args.out / "SUMMARY.md").write_text("\n".join(lines))
    print((args.out / "SUMMARY.md").read_text(), flush=True)


def equivalence_probe():
    hour = 1782388800  # 2026-06-25 12:00 UTC, pre-July only
    print("EQUIV_FETCH_FULL", flush=True)
    full_markets, full_rows = retryfix.fetch_hour_fixed(hour)
    full_td = pd.DataFrame(full_rows)
    print("EQUIV_FETCH_STREAM", flush=True)
    micros, meta = fetch_hour_stream(hour)
    bs = core.load_binance("train")
    full_ex = core.build_examples(full_markets, full_td, bs)
    stream_ex = build_stream_examples(micros, bs)
    A = {e.slug: e for e in full_ex}
    B = {e.slug: e for e in stream_ex}
    only_a = sorted(set(A) - set(B)); only_b = sorted(set(B) - set(A))
    max_abs = 0.0
    worst = None
    for slug in sorted(set(A) & set(B)):
        d = float(np.max(np.abs(A[slug].x.astype(float) - B[slug].x.astype(float))))
        if d > max_abs:
            max_abs, worst = d, slug
        if A[slug].pm_last != B[slug].pm_last or A[slug].label != B[slug].label:
            raise RuntimeError(f"scalar mismatch {slug}")
    # Compare exact post-decision BUY prints used by policy proxy.
    tm = core.normalize_trade_rows(full_td)
    fill_mismatch = []
    bm = {z.market.slug: z for z in micros}
    for m in full_markets:
        if m.slug not in bm:
            continue
        g = tm.get(m.condition_id)
        if g is None:
            continue
        post = g[(g.timestamp >= m.decision) & (g.timestamp <= m.decision + core.TAPE_SECONDS) & (g.side_u == "BUY")]
        up = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "up"].itertuples(index=False))
        down = tuple((int(r.timestamp), float(r.price)) for r in post[post.outcome_l == "down"].itertuples(index=False))
        z = bm[m.slug]
        if up != z.fill_up or down != z.fill_down:
            fill_mismatch.append(m.slug)
    result = {"full_rows": len(full_rows), "stream_raw_rows": meta["raw_rows"], "full_examples": len(A), "stream_examples": len(B),
              "only_full": only_a, "only_stream": only_b, "max_abs_feature_diff": max_abs, "worst_slug": worst,
              "fill_mismatch_count": len(fill_mismatch), "fill_mismatch_sample": fill_mismatch[:10], "stream_meta": meta}
    print("EQUIVALENCE", json.dumps(result, indent=2), flush=True)
    if only_a or only_b or max_abs > 1e-7 or fill_mismatch:
        raise RuntimeError("stream transport is not exactly equivalent on June probe")
    print("EQUIVALENCE_PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "test", "equiv"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model-dir", type=Path)
    args = ap.parse_args()
    if args.phase == "equiv":
        equivalence_probe(); return
    if args.out is None:
        raise SystemExit("--out required")
    if args.phase == "train":
        train_phase(args)
    else:
        if args.model_dir is None:
            raise SystemExit("--model-dir required")
        test_phase(args)


if __name__ == "__main__":
    main()
