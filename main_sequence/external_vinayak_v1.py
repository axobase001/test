from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from honest_backtest import Decision, Signal, SlotCtx, SlotMeta, evaluate, grade_taker, run_signal

from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    StructuralMispricing,
    deribit_instruments,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)
from main_sequence.train_v0 import (
    EPS,
    YEAR_SECONDS,
    Example,
    Scaler,
    bn_return,
    bn_rv,
    build_sequence,
    safe_mid,
)
from main_sequence.train_v1 import MainSequenceV1, action_rewards, brier, logloss, predict


REQUIRED = [
    "ts_ms", "window_slug", "window_end_ts", "outcome",
    "yes_token_id", "no_token_id",
    "yes_bid", "yes_ask", "yes_ask_size", "yes_bid_size",
    "no_bid", "no_ask", "no_ask_size", "no_bid_size", "btc_price",
]


def norm_epoch_s(v) -> int:
    x = int(float(v))
    while abs(x) > 10**11:
        x //= 1000
    return x


def norm_epoch_ms(v) -> int:
    x = int(float(v))
    while abs(x) > 10**14:
        x //= 1000
    if abs(x) < 10**11:
        x *= 1000
    return x


def side_is_yes(v: object) -> bool:
    s = str(v).strip().lower()
    return s in {"up", "yes", "1", "true", "higher"}


def read_external_ctxs(csv_dir: Path, start: date, end: date, bn: BinanceAnchor, fee_rate: float) -> tuple[list[SlotCtx], dict]:
    ctxs: list[SlotCtx] = []
    audit = {"files": [], "rows": 0, "groups": 0, "kept": 0, "dropped": {}}

    def drop(reason: str):
        audit["dropped"][reason] = audit["dropped"].get(reason, 0) + 1

    for p in sorted(csv_dir.glob("orderbook_2026-*.csv")):
        try:
            d = date.fromisoformat(p.stem.replace("orderbook_", ""))
        except ValueError:
            continue
        if d < start or d > end:
            continue
        print(f"read external {p.name}", flush=True)
        f = pd.read_csv(p, usecols=lambda c: c in REQUIRED, low_memory=False)
        missing = [c for c in REQUIRED if c not in f.columns]
        if missing:
            raise RuntimeError(f"{p}: missing columns {missing}; got {list(f.columns)}")
        audit["files"].append({"name": p.name, "rows": int(len(f))})
        audit["rows"] += int(len(f))
        for slug, g in f.groupby("window_slug", sort=False):
            audit["groups"] += 1
            g = g.copy()
            for c in ["ts_ms", "window_end_ts", "yes_bid", "yes_ask", "yes_ask_size", "yes_bid_size",
                      "no_bid", "no_ask", "no_ask_size", "no_bid_size", "btc_price"]:
                g[c] = pd.to_numeric(g[c], errors="coerce")
            g = g.dropna(subset=["ts_ms", "yes_bid", "yes_ask", "no_bid", "no_ask", "btc_price"])
            if len(g) < 3:
                drop("too_few_rows"); continue
            g["ts_ms"] = g["ts_ms"].map(norm_epoch_ms)
            g = g.sort_values("ts_ms").drop_duplicates("ts_ms", keep="last")
            tail = str(slug).rsplit("-", 1)[-1]
            if tail.isdigit():
                open_ts = norm_epoch_s(tail)
            else:
                vals = g["window_end_ts"].dropna()
                if vals.empty:
                    drop("no_open_or_end"); continue
                open_ts = norm_epoch_s(vals.iloc[0]) - 900
            vals = g["window_end_ts"].dropna()
            close_ts = norm_epoch_s(vals.iloc[0]) if not vals.empty else open_ts + 900
            if not (840 <= close_ts - open_ts <= 960):
                close_ts = open_ts + 900
            yes_tok = str(g["yes_token_id"].dropna().iloc[0]) if g["yes_token_id"].notna().any() else ""
            no_tok = str(g["no_token_id"].dropna().iloc[0]) if g["no_token_id"].notna().any() else ""
            if not yes_tok or not no_tok:
                drop("missing_tokens"); continue
            outs = g["outcome"].dropna()
            if outs.empty:
                drop("missing_outcome"); continue
            resolved = "Yes" if side_is_yes(outs.iloc[0]) else "No"
            open_spot = bn.open_price(open_ts)
            if not (open_spot > 0 and math.isfinite(open_spot)):
                open_spot = float(g["btc_price"].iloc[0])
            close_spot = float(g["btc_price"].iloc[-1])
            rows = []
            for r in g.itertuples(index=False):
                ts = norm_epoch_ms(r.ts_ms)
                s2c = int(round(close_ts - ts / 1000.0))
                if s2c < -10 or s2c > 920:
                    continue
                rows.append((
                    ts, max(s2c, 0),
                    float(r.yes_bid), float(r.yes_ask), max(float(r.yes_bid_size or 0), 0.0), max(float(r.yes_ask_size or 0), 0.0),
                    float(r.no_bid), float(r.no_ask), max(float(r.no_bid_size or 0), 0.0), max(float(r.no_ask_size or 0), 0.0),
                    float(r.btc_price),
                ))
            if len(rows) < 3:
                drop("no_inwindow_rows"); continue
            meta = SlotMeta(
                condition_id=str(slug), coin="btc", duration="15m", duration_s=900,
                open_ts=int(open_ts), close_ts=int(close_ts), strike=float(open_spot),
                spot_at_open=float(open_spot), spot_at_close=float(close_spot),
                yes_token_id=yes_tok, no_token_id=no_tok, resolved_side=resolved,
                fee_rate=float(fee_rate), rebate_rate=None,
            )
            ctxs.append(SlotCtx.from_rows(meta, rows, []))
            audit["kept"] += 1
    ctxs.sort(key=lambda c: (c.meta.close_ts, c.meta.condition_id))
    return ctxs, audit


def save_records(records: list[dict], path: Path) -> pd.DataFrame:
    rows = []
    for r in records:
        x = dict(r)
        try:
            x.update(json.loads(x.get("tag", "{}")))
        except Exception:
            pass
        rows.append(x)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


@dataclass
class Ref:
    ctx: SlotCtx
    i: int
    fade_yes: bool
    fade_ask: float
    follow_ask: float


def build_examples_external(ctxs: list[SlotCtx], recs: pd.DataFrame, bn: BinanceAnchor) -> tuple[list[Example], list[Ref]]:
    by_cid = {str(c.meta.condition_id): c for c in ctxs}
    examples: list[Example] = []
    refs: list[Ref] = []
    for r in recs.itertuples(index=False):
        ctx = by_cid.get(str(r.cid))
        if ctx is None:
            continue
        ts = int(r.ts_ms)
        i = int(np.searchsorted(ctx.ts, ts, side="right") - 1)
        if i < 0 or abs(int(ctx.ts[i]) - ts) > 1500:
            continue
        fade_yes = bool(r.yes)
        fade_ask = float(ctx.ya[i] if fade_yes else ctx.na[i])
        follow_ask = float(ctx.na[i] if fade_yes else ctx.ya[i])
        if not (0 < fade_ask < 1 and 0 < follow_ask < 1):
            continue
        p_rv, p_d = float(r.p_rv), float(r.p_deribit)
        rv60, div, rel_spot, s2c = float(r.rv), float(r.deribit_iv), float(r.rel_spot), float(r.s2c)
        yb, ya, nb, na = float(ctx.yb[i]), float(ctx.ya[i]), float(ctx.nb[i]), float(ctx.na[i])
        ymid, nmid = safe_mid(yb, ya), safe_mid(nb, na)
        ysp = ya - yb if 0 < yb < ya < 1 else 0.0
        nsp = na - nb if 0 < nb < na < 1 else 0.0
        ybs, yas = max(float(ctx.ybs[i]), 0.0), max(float(ctx.yas[i]), 0.0)
        nbs, nas = max(float(ctx.nbs[i]), 0.0), max(float(ctx.nas[i]), 0.0)
        yimb = (ybs - yas) / (ybs + yas + EPS)
        nimb = (nbs - nas) / (nbs + nas + EPS)
        anchor_mean = 0.5 * (p_rv + p_d)
        ret1, ret5, ret15 = bn_return(bn, ts, 60), bn_return(bn, ts, 300), bn_return(bn, ts, 900)
        rv15 = bn_rv(bn, ts, 15, min_obs=8)
        if not math.isfinite(rv15):
            rv15 = rv60
        vol_exp = rv15 / max(rv60, 1e-6)
        sigma5 = rv60 * math.sqrt(300.0 / YEAR_SECONDS)
        shockz = ret5 / max(sigma5, 1e-6)
        static = np.asarray([
            float(r.edge_signal), p_rv, p_d, anchor_mean, abs(p_rv-p_d),
            rv60, div, div-rv60, rel_spot, math.log(max(rel_spot,1e-8)),
            s2c/900.0, yb, ya, nb, na, ysp, nsp, ymid, nmid,
            math.log1p(ybs), math.log1p(yas), math.log1p(nbs), math.log1p(nas),
            yimb, nimb, ymid-anchor_mean if math.isfinite(ymid) else 0.0,
            ret1, ret5, ret15, rv15, vol_exp, shockz,
        ], dtype=np.float32)
        static[~np.isfinite(static)] = 0.0
        settle_yes = 1.0 if ctx.meta.resolved_side == "Yes" else 0.0
        fr = float(ctx.meta.fee_rate or 0.0)
        fade_cost = fade_ask + fr*fade_ask*(1-fade_ask)
        follow_cost = follow_ask + fr*follow_ask*(1-follow_ask)
        fade_won = 1.0 if ((ctx.meta.resolved_side == "Yes") == fade_yes) else 0.0
        follow_won = 1.0 - fade_won
        examples.append(Example(
            ts_ms=ts, close_ts=int(ctx.meta.close_ts), cid=str(ctx.meta.condition_id),
            seq=build_sequence(ctx, i), static=static, settle_yes=settle_yes,
            fade_fill=0.0, follow_fill=0.0,
            fade_reward=fade_won-fade_cost, follow_reward=follow_won-follow_cost,
            fade_cost=fade_cost, follow_cost=follow_cost, fade_yes=fade_yes,
        ))
        refs.append(Ref(ctx=ctx, i=i, fade_yes=fade_yes, fade_ask=fade_ask, follow_ask=follow_ask))
    return examples, refs


def load_frozen(model_dir: Path):
    z = np.load(model_dir / "scaler.npz", allow_pickle=True)
    scaler = Scaler(z["static_mean"], z["static_std"], z["seq_mean"], z["seq_std"])
    payload = torch.load(model_dir / "main_sequence_v1.pt", map_location="cpu", weights_only=False)
    models = []
    for seed in payload["seeds"]:
        m = MainSequenceV1(len(payload["seq_names"]), len(payload["static_names"]))
        m.load_state_dict(payload["states"][str(seed)])
        m.eval(); models.append(m)
    return scaler, models


class PrecomputedSignal(Signal):
    family = "main_sequence_v1_external"
    mode = "taker"
    coins = ("btc",)
    durations = ("15m",)
    once = True

    def __init__(self, name: str, decisions: dict[str, tuple[int, bool, float, str]]):
        self.name = name
        self.decisions = decisions

    def decide(self, ctx, i):
        d = self.decisions.get(str(ctx.meta.condition_id))
        if d is None:
            return None
        di, yes, px, tag = d
        if i != di:
            return None
        return Decision(i=i, ts_ms=int(ctx.ts[i]), token_yes=yes, action="taker", target_px=float(px), size=5.0, tag=tag)


def precomputed_policies(exs: list[Example], refs: list[Ref], probs: np.ndarray, fill_probs: np.ndarray):
    out = {}
    for mode in ("value", "exec"):
        decisions = {}
        counts = {"fade": 0, "follow": 0, "abstain": 0}
        for k, (e, ref) in enumerate(zip(exs, refs)):
            er = action_rewards(e, probs[k])
            if mode == "value":
                a = int(np.argmax(er))
                if er[a] <= 0:
                    counts["abstain"] += 1; continue
            else:
                score = fill_probs[k] * np.maximum(er, 0.0)
                a = int(np.argmax(score))
                if score[a] <= 0:
                    counts["abstain"] += 1; continue
            if a == 0:
                yes, px, aname = ref.fade_yes, ref.fade_ask, "fade"
            else:
                yes, px, aname = (not ref.fade_yes), ref.follow_ask, "follow"
            counts[aname] += 1
            decisions[e.cid] = (ref.i, yes, px, json.dumps({"pred_p_yes":float(probs[k]),"pred_edge":float(er[a]),"action":aname},separators=(",",":")))
        out[mode] = (decisions, counts)
    return out


def stability(records: pd.DataFrame) -> dict:
    if records.empty:
        return {}
    x = records.copy()
    x["day"] = pd.to_datetime(x.ts_ms, unit="ms", utc=True).dt.date.astype(str)
    x["reward_paper"] = x["won"].astype(float) - x["fill_px"].astype(float) - x["fee_rate"].astype(float)*x["fill_px"].astype(float)*(1-x["fill_px"].astype(float))
    return {
        "days": int(x.day.nunique()),
        "signals": int(len(x)),
        "win_rate": float(x.won.mean()),
        "mean_signal_edge": float(x.edge_signal.mean()),
        "anchor_gap_mean": float((x.p_rv-x.p_deribit).abs().mean()),
        "rel_spot_abs_bps_median": float(((x.rel_spot-1).abs()*1e4).median()),
        "signals_by_day": {str(k):int(v) for k,v in x.groupby("day").size().to_dict().items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-03-24")
    ap.add_argument("--end", default="2026-03-25")
    ap.add_argument("--fee-rate", type=float, default=0.07)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    print("download official Binance 1m", flush=True)
    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)
    print("load independent Vinayak 1s BBO", flush=True)
    ctxs, audit = read_external_ctxs(args.csv_dir, start, end, bn, args.fee_rate)
    if not ctxs:
        raise RuntimeError("no external contexts")
    print(f"external slots={len(ctxs)} rows={audit['rows']}", flush=True)

    print("build historical Deribit traded-IV anchor", flush=True)
    inst = deribit_instruments()
    selected = select_deribit_instruments(inst, bn_df, start, end)
    der_trades = fetch_deribit_trades(selected, start, end, args.cache / "deribit_trades.parquet")
    der = DeribitAnchor.from_trades(der_trades, inst)
    print(f"deribit instruments={len(selected)} usable_trades={len(der.ts)}", flush=True)

    sig = StructuralMispricing(bn, der, threshold=0.03, size=5.0)
    base_eval = evaluate(sig, ctxs, latency_ms=1000, tape_window_ms=1500)
    base_records = save_records(run_signal(sig, ctxs, latency_ms=1000, tape_window_ms=1500), args.out / "external_3c_records.csv")
    print("3c baseline", json.dumps(base_eval, default=str), flush=True)

    exs, refs = build_examples_external(ctxs, base_records, bn)
    if not exs:
        raise RuntimeError("3c signal produced no model examples")
    scaler, models = load_frozen(args.model_dir)
    ps=[]; fps=[]; ds=[]
    for m in models:
        p, fp, d = predict(m, exs, scaler); ps.append(p); fps.append(fp); ds.append(d)
    prob = np.mean(ps, axis=0); fill_prob=np.mean(fps,axis=0); delta=np.mean(ds,axis=0)
    anchor=np.asarray([float(e.static[3]) for e in exs]); y=np.asarray([e.settle_yes for e in exs])
    probability = {
        "n":len(exs), "brier_model":brier(y,prob), "brier_anchor":brier(y,anchor),
        "logloss_model":logloss(y,prob), "logloss_anchor":logloss(y,anchor),
        "delta_logit_mean":float(delta.mean()), "delta_logit_abs_mean":float(np.abs(delta).mean()),
    }
    print("frozen probability", json.dumps(probability), flush=True)

    policies = precomputed_policies(exs, refs, prob, fill_prob)
    value_sig = PrecomputedSignal("main_sequence_v1_value_external", policies["value"][0])
    exec_sig = PrecomputedSignal("main_sequence_v1_exec_external", policies["exec"][0])
    value_eval = evaluate(value_sig, ctxs, latency_ms=1000, tape_window_ms=1500)
    exec_eval = evaluate(exec_sig, ctxs, latency_ms=1000, tape_window_ms=1500)
    save_records(run_signal(value_sig, ctxs, latency_ms=1000, tape_window_ms=1500), args.out / "external_v1_value_records.csv")
    save_records(run_signal(exec_sig, ctxs, latency_ms=1000, tape_window_ms=1500), args.out / "external_v1_exec_records.csv")

    pd.DataFrame({
        "cid":[e.cid for e in exs], "ts_ms":[e.ts_ms for e in exs], "anchor_p":anchor,
        "model_p":prob, "settle_yes":y, "delta_logit":delta,
        "anchor_gap":[float(e.static[4]) for e in exs], "edge_signal":[float(e.static[0]) for e in exs],
    }).to_csv(args.out/"external_probabilities.csv",index=False)

    result = {
        "status":"EXTERNAL_BACKWARD_OOD",
        "source":"Vinayak19112003/Polymarket-orderbooks independent 1-second BTC15m BBO + official Binance 1m + backward-only Deribit traded IV",
        "period":{"start":args.start,"end":args.end},
        "frozen":{"threshold":0.03,"model":"Main Sequence v1 run 31786389773","weights_retrained":False,"scaler_refit":False,"fee_rate_assumption":args.fee_rate},
        "execution":{"library":"honest-backtest 0.2.0","lens":"1-second book persistence at 1000ms; no trade tape in this source","tape_verdict":"NOT_AVAILABLE"},
        "source_audit":audit,
        "slots":len(ctxs), "deribit_selected_instruments":len(selected), "deribit_usable_trades":len(der.ts),
        "signals":stability(base_records), "probability":probability,
        "policies":{
            "frozen_3c_fade":base_eval,
            "main_sequence_v1_value":{"decisions":policies["value"][1],"eval":value_eval},
            "main_sequence_v1_exec":{"decisions":policies["exec"][1],"eval":exec_eval},
        },
        "caveats":[
            "This period predates the v1 training period, so it is an independent backward/OOD transfer test, not a chronological deployable forward backtest.",
            "The independent source has 1-second BBO but no authenticated PM trade tape; honest book-persistence is the executable lens and strict tape corroboration is unavailable.",
            "Fee curve is held fixed to the same 0.07 crypto curve used by the frozen v1 replay rather than inferred post hoc from returns.",
            "No v1 weights, scaler, 3c threshold, or policy parameters are fit on this external dataset.",
        ],
    }
    (args.out/"summary.json").write_text(json.dumps(result,indent=2,default=str))
    lines=[
        "# Frozen Main Sequence v1 — independent collector OOD", "",
        f"Period: {args.start} to {args.end}; slots={len(ctxs)}; structural signals={len(base_records)}.",
        "Weights/scaler/3c threshold are frozen. No external labels are used for fitting.", "",
        f"Probability: Brier model {probability['brier_model']:.6f} vs anchor {probability['brier_anchor']:.6f}; logloss model {probability['logloss_model']:.6f} vs anchor {probability['logloss_anchor']:.6f}.", "",
        "Execution uses honest-backtest 0.2.0 book-persistence at 1000 ms. This collector has no PM trade tape, so strict tape fills are not claimed.", "",
        "## Raw policy summaries", "", "```json", json.dumps(result['policies'],indent=2,default=str), "```", "",
    ]
    (args.out/"SUMMARY.md").write_text("\n".join(lines))
    print((args.out/"SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
