from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from honest_backtest import Decision
from honest_backtest.adapters.parquet_pm import load_corpus
from honest_backtest.fills import grade_taker

from recalc import BinanceAnchor, YEAR_SECONDS, download_binance_1m

SIZE = 5.0
LATENCY_MS = 1000
TAPE_WINDOW_MS = 1500


def log_return(anchor: BinanceAnchor, ts_ms: int, minutes: int) -> float:
    hi = int(np.searchsorted(anchor.close_times, int(ts_ms), side="right"))
    if hi <= 0:
        return math.nan
    end_i = hi - 1
    lo_t = int(ts_ms) - minutes * 60_000
    j = int(np.searchsorted(anchor.close_times, lo_t, side="right") - 1)
    if j < 0 or j >= end_i:
        return math.nan
    a, b = float(anchor.closes[j]), float(anchor.closes[end_i])
    if a <= 0 or b <= 0:
        return math.nan
    return float(math.log(b / a))


def rv_annualized(anchor: BinanceAnchor, ts_ms: int, minutes: int) -> float:
    hi = int(np.searchsorted(anchor.close_times, int(ts_ms), side="right"))
    lo = int(np.searchsorted(anchor.close_times, int(ts_ms) - minutes * 60_000, side="left"))
    vals = anchor.log_returns[lo:hi]
    vals = vals[np.isfinite(vals)]
    if vals.size < max(5, minutes // 3):
        return math.nan
    return float(np.std(vals, ddof=1)) * math.sqrt(365 * 24 * 60)


def fee_ps(px: float, fr: float) -> float:
    return float(fr) * float(px) * (1.0 - float(px))


def trade(won: bool, px: float, fr: float, sz: float = SIZE) -> dict:
    fee = fee_ps(px, fr)
    w = 1.0 if won else 0.0
    return {"won": w, "px": px, "sz": sz, "pnl_ps": w - px - fee,
            "cost_ps": px + fee, "pnl": (w - px - fee) * sz, "cost": (px + fee) * sz}


def strict_trade(g: dict) -> dict | None:
    if not (g.get("valid") and g.get("crossable") and g.get("has_tape") and g.get("fillable")):
        return None
    sz = float(g["honest_sz"])
    if sz <= 0:
        return None
    return trade(bool(g["won"]), float(g["best_ask"]), float(g.get("fee_rate") or 0.0), sz)


def agg(xs: list[dict]) -> dict:
    if not xs:
        return {"n": 0, "wr": None, "edge_ps": None, "roi": None, "pnl": 0.0, "cost": 0.0}
    pnl, cost = sum(x["pnl"] for x in xs), sum(x["cost"] for x in xs)
    return {"n": len(xs), "wr": round(sum(x["won"] for x in xs) / len(xs), 4),
            "edge_ps": round(sum(x["pnl_ps"] for x in xs) / len(xs), 4),
            "roi": round(pnl / cost, 4) if cost > 0 else None,
            "pnl": round(pnl, 4), "cost": round(cost, 4)}


def cluster_ci(df: pd.DataFrame, col: str, reps: int = 20000) -> dict:
    x = df.dropna(subset=[col]).copy()
    if x.empty:
        return {"days": 0, "n": 0, "lo": None, "mean": None, "hi": None}
    x["day"] = pd.to_datetime(x.ts_ms, unit="ms", utc=True).dt.date
    days = list(x.day.unique())
    mean = float(x[col].mean())
    if len(days) < 2:
        return {"days": len(days), "n": len(x), "lo": None, "mean": round(mean, 4), "hi": None}
    groups = {d: x.loc[x.day == d, col].to_numpy(float) for d in days}
    rng = np.random.default_rng(42)
    vals = np.empty(reps)
    for k in range(reps):
        ds = rng.choice(days, len(days), replace=True)
        vals[k] = np.concatenate([groups[d] for d in ds]).mean()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"days": len(days), "n": len(x), "lo": round(float(lo), 4),
            "mean": round(mean, 4), "hi": round(float(hi), 4)}


def decode(series: pd.Series) -> list[dict]:
    return [json.loads(v) for v in series if isinstance(v, str) and v]


def summarize(g: pd.DataFrame) -> dict:
    return {"signals": int(len(g)),
            "fade_paper": agg(decode(g.fade_paper_json)),
            "follow_paper": agg(decode(g.follow_paper_json)),
            "fade_tape": agg(decode(g.fade_tape_json)),
            "follow_tape": agg(decode(g.follow_tape_json)),
            "fade_tape_ci": cluster_ci(g[g.fade_tape_pnl_ps.notna()], "fade_tape_pnl_ps"),
            "follow_tape_ci": cluster_ci(g[g.follow_tape_pnl_ps.notna()], "follow_tape_pnl_ps")}


def build(pm_dir: Path, records_path: Path, summary_path: Path, start: date, end: date, cache: Path):
    records = pd.read_csv(records_path)
    prior = json.loads(summary_path.read_text())
    cut = int(prior["split_close_ts"])
    ctxs = list(load_corpus(str(pm_dir), coins=("btc",), durations=("15m",)))
    by_cid = {c.meta.condition_id: c for c in ctxs}
    bn = BinanceAnchor.from_df(download_binance_1m(start, end, cache / "binance"))
    rows = []
    for r in records.itertuples(index=False):
        ctx = by_cid.get(r.cid)
        if ctx is None:
            continue
        ts = int(r.ts_ms)
        i = int(np.searchsorted(ctx.ts, ts, side="left"))
        if i >= ctx.n or int(ctx.ts[i]) != ts:
            continue
        fade_yes = bool(r.yes)
        follow_yes = not fade_yes
        fade_ask = float(ctx.ask(i, fade_yes))
        follow_ask = float(ctx.ask(i, follow_yes))
        fr = float(ctx.meta.fee_rate or 0.0)
        fg = grade_taker(ctx, Decision(i, ts, fade_yes, "taker", fade_ask, SIZE, "fade"),
                         latency_ms=LATENCY_MS, tape_window_ms=TAPE_WINDOW_MS)
        ft = strict_trade(fg)
        follow_ok = 0.0 < follow_ask < 1.0 and float(ctx.ask_sz(i, follow_yes)) >= SIZE
        fot = None
        if follow_ok:
            gg = grade_taker(ctx, Decision(i, ts, follow_yes, "taker", follow_ask, SIZE, "follow"),
                             latency_ms=LATENCY_MS, tape_window_ms=TAPE_WINDOW_MS)
            fot = strict_trade(gg)
        fade_won = bool(fg["won"])
        rv60 = float(r.rv)
        rv15 = rv_annualized(bn, ts, 15)
        ret5 = log_return(bn, ts, 5)
        exp5 = rv60 * math.sqrt(300 / YEAR_SECONDS) if rv60 > 0 else math.nan
        z = ret5 / exp5 if math.isfinite(ret5) and exp5 > 0 else math.nan
        ve = rv15 / rv60 if math.isfinite(rv15) and rv60 > 0 else math.nan
        crowd_yes = follow_yes
        if not math.isfinite(z) or abs(z) < 1:
            pattern = "no_shock"
        elif z <= -1:
            pattern = "dip_buy" if crowd_yes else "panic_sell"
        else:
            pattern = "chase_up" if crowd_yes else "fade_rally"
        fp = trade(fade_won, fade_ask, fr)
        fop = trade(not fade_won, follow_ask, fr) if follow_ok else None
        rows.append({"cid": r.cid, "ts_ms": ts, "close_ts": int(ctx.meta.close_ts),
                     "split": "holdout" if int(ctx.meta.close_ts) > cut else "train",
                     "fade_yes": fade_yes, "crowd_yes": crowd_yes, "edge_signal": float(r.edge_signal),
                     "rv60": rv60, "rv15": rv15, "vol_expansion": ve,
                     "ret5": ret5, "shock_z_5m": z, "abs_shock_z_5m": abs(z) if math.isfinite(z) else math.nan,
                     "deribit_iv": float(r.deribit_iv), "iv_rv_ratio": float(r.deribit_iv) / rv60 if rv60 > 0 else math.nan,
                     "pattern": pattern,
                     "fade_paper_json": json.dumps(fp, separators=(",", ":")),
                     "follow_paper_json": json.dumps(fop, separators=(",", ":")) if fop else "",
                     "fade_tape_json": json.dumps(ft, separators=(",", ":")) if ft else "",
                     "follow_tape_json": json.dumps(fot, separators=(",", ":")) if fot else "",
                     "fade_tape_pnl_ps": ft["pnl_ps"] if ft else math.nan,
                     "follow_tape_pnl_ps": fot["pnl_ps"] if fot else math.nan})
    return pd.DataFrame(rows), prior


def regimes(df: pd.DataFrame):
    tr = df[df.split == "train"]
    rq = tr.rv60.quantile([1/3, 2/3]).to_numpy(float)
    vq = tr.vol_expansion.replace([np.inf, -np.inf], np.nan).dropna().quantile([1/3, 2/3]).to_numpy(float)
    def tert(v, q):
        if not math.isfinite(float(v)): return "missing"
        if v < q[0]: return "low"
        if v < q[1]: return "mid"
        return "high"
    x = df.copy()
    x["rv_regime"] = [tert(v, rq) for v in x.rv60]
    x["vol_exp_regime"] = [tert(v, vq) for v in x.vol_expansion]
    x["shock_regime"] = np.where(x.abs_shock_z_5m < 1, "quiet",
                                  np.where(x.abs_shock_z_5m < 2, "shock", "extreme"))
    return x, {"rv_q33": float(rq[0]), "rv_q67": float(rq[1]),
               "vol_exp_q33": float(vq[0]), "vol_exp_q67": float(vq[1]),
               "shock_fixed": [1.0, 2.0]}


def matrix(df: pd.DataFrame, cols: list[str], split: str):
    out = {}
    for keys, g in df[df.split == split].groupby(cols, dropna=False, sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        out["|".join(f"{c}={v}" for c, v in zip(cols, keys))] = summarize(g)
    return out


def flat(df: pd.DataFrame, cols: list[str]):
    rows = []
    for split in ("train", "holdout"):
        for keys, g in df[df.split == split].groupby(cols, dropna=False, sort=False):
            if not isinstance(keys, tuple): keys = (keys,)
            s = summarize(g)
            r = {"split": split, **{c: v for c, v in zip(cols, keys)}, "signals": s["signals"]}
            for side in ("fade_tape", "follow_tape"):
                r[f"{side}_n"] = s[side]["n"]
                r[f"{side}_edge_ps"] = s[side]["edge_ps"]
                r[f"{side}_roi"] = s[side]["roi"]
            r["fade_paper_edge_ps"] = s["fade_paper"]["edge_ps"]
            r["follow_paper_edge_ps"] = s["follow_paper"]["edge_ps"]
            rows.append(r)
    return pd.DataFrame(rows)


def policy_trade(r, name):
    high = r.rv_regime == "high" or r.shock_regime == "extreme"
    if name == "baseline_fade": s = r.fade_tape_json
    elif name == "high_abstain":
        if high: return None
        s = r.fade_tape_json
    elif name == "high_flip": s = r.follow_tape_json if high else r.fade_tape_json
    else: raise ValueError(name)
    return json.loads(s) if isinstance(s, str) and s else None


def policy(df: pd.DataFrame, split: str, name: str):
    xs, cir = [], []
    for r in df[df.split == split].itertuples(index=False):
        t = policy_trade(r, name)
        if t:
            xs.append(t); cir.append({"ts_ms": r.ts_ms, "pnl_ps": t["pnl_ps"]})
    a = agg(xs)
    a["cluster_ci"] = cluster_ci(pd.DataFrame(cir), "pnl_ps") if cir else {"days":0,"n":0,"lo":None,"mean":None,"hi":None}
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-dir", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--prior-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache_regime"))
    ap.add_argument("--start", default="2026-05-27"); ap.add_argument("--end", default="2026-06-24")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    df, prior = build(a.pm_dir, a.records, a.prior_summary, date.fromisoformat(a.start), date.fromisoformat(a.end), a.cache)
    df, cuts = regimes(df); df.to_csv(a.out / "regime_enriched_03c.csv", index=False)
    result = {"frozen_signal":"structural_03c","threshold":0.03,"no_threshold_tuning":True,
              "split_close_ts":int(prior["split_close_ts"]),"split_close_utc":prior["split_close_utc"],
              "regime_cutpoints_train_only":cuts,
              "definitions":{"shock_z_5m":"past-only Binance 5m return / past-only 60m RV scaled to 5m",
                             "rv_regime":"tertiles fixed from first-half signals only",
                             "shock_regime":"quiet |z|<1; shock 1<=|z|<2; extreme |z|>=2",
                             "crowd_side":"opposite of frozen 3c fade side",
                             "strict_execution":"honest-backtest 0.2.0; 1000ms latency; 1500ms real-tape corroboration; historical fee; 5 shares"},
              "counts":{"rows":len(df),"train":int((df.split=="train").sum()),"holdout":int((df.split=="holdout").sum())},
              "rv_x_shock":{"train":matrix(df,["rv_regime","shock_regime"],"train"),"holdout":matrix(df,["rv_regime","shock_regime"],"holdout")},
              "shock_pattern":{"train":matrix(df,["pattern"],"train"),"holdout":matrix(df,["pattern"],"holdout")},
              "vol_expansion":{"train":matrix(df,["vol_exp_regime"],"train"),"holdout":matrix(df,["vol_exp_regime"],"holdout")},
              "policies":{}}
    for sp in ("train","holdout"):
        result["policies"][sp] = {p: policy(df,sp,p) for p in ("baseline_fade","high_abstain","high_flip")}
    (a.out/"regime_matrix.json").write_text(json.dumps(result,indent=2,default=str))
    flat(df,["rv_regime","shock_regime"]).to_csv(a.out/"rv_shock_matrix.csv",index=False)
    flat(df,["pattern"]).to_csv(a.out/"shock_pattern_matrix.csv",index=False)
    flat(df,["vol_exp_regime"]).to_csv(a.out/"vol_expansion_matrix.csv",index=False)
    h=result["policies"]["holdout"]
    lines=["# Frozen 3c residual regime matrix","",f"Split: {result['split_close_utc']}; signals={len(df)}; train={result['counts']['train']}; holdout={result['counts']['holdout']}",
           f"Train-only RV tertiles: {cuts['rv_q33']:.4f}, {cuts['rv_q67']:.4f}; shock z fixed 1/2.","","## Holdout strict-tape policies","",
           "| policy | fills | edge/share | fee ROI | day-cluster 95% CI |","|---|---:|---:|---:|---:|"]
    for p in ("baseline_fade","high_abstain","high_flip"):
        x=h[p]; ci=x["cluster_ci"]; cis=f"[{ci['lo']}, {ci['hi']}]" if ci.get("lo") is not None else "sparse"
        lines.append(f"| {p} | {x['n']} | {x['edge_ps']} | {x['roi']} | {cis} |")
    lines += ["","## Holdout shock/crowd pattern","","| pattern | signals | fade n | fade edge | follow n | follow edge |","|---|---:|---:|---:|---:|---:|"]
    q=flat(df,["pattern"]); q=q[q.split=="holdout"]
    for r in q.itertuples(index=False): lines.append(f"| {r.pattern} | {r.signals} | {r.fade_tape_n} | {r.fade_tape_edge_ps} | {r.follow_tape_n} | {r.follow_tape_edge_ps} |")
    lines += ["","## Holdout RV x shock","","| RV | shock | signals | fade n | fade edge | follow n | follow edge |","|---|---|---:|---:|---:|---:|---:|"]
    q=flat(df,["rv_regime","shock_regime"]); q=q[q.split=="holdout"]
    for r in q.itertuples(index=False): lines.append(f"| {r.rv_regime} | {r.shock_regime} | {r.signals} | {r.fade_tape_n} | {r.fade_tape_edge_ps} | {r.follow_tape_n} | {r.follow_tape_edge_ps} |")
    (a.out/"REGIME_SUMMARY.md").write_text("\n".join(lines)+"\n"); print((a.out/"REGIME_SUMMARY.md").read_text())


if __name__ == "__main__": main()
