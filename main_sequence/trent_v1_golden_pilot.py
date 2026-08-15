from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from main_sequence.train_v0 import (
    EPS, YEAR_SECONDS, Example, SEQ_LEN, SEQ_NAMES, STATIC_NAMES,
    bn_return, bn_rv, safe_mid,
)
from main_sequence.external_vinayak_v1 import load_frozen
from main_sequence.train_v1 import brier, logloss, predict
from pm_structural.recalc import (
    BinanceAnchor, DeribitAnchor, deribit_instruments, digital_prob_up,
    download_binance_1m, fetch_deribit_trades, select_deribit_instruments,
)

REPO = "trentmkelly/polymarket_crypto_derivatives"
REV = "6be20463ce33795178c121e7bd15ed428904b5bd"
TRAINING_FEE_RATE = 0.07
CADENCE_MS = 30_000


def episode_open_ms(name: str) -> int:
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_all$", name)
    if not m:
        raise ValueError(name)
    dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def hf(rel: str, cache: Path) -> Path:
    return Path(hf_hub_download(REPO, rel, repo_type="dataset", revision=REV, cache_dir=cache))


def previous_step_indices(ts: np.ndarray, targets: np.ndarray, max_age_ms: int = 500) -> list[int]:
    out = []
    for t in targets:
        j = int(np.searchsorted(ts, int(t), side="right") - 1)
        if j < 0:
            continue
        age = int(t) - int(ts[j])
        if age < 0 or age > max_age_ms:
            continue
        if not out or j != out[-1]:
            out.append(j)
    return out


def load_episode(ep: str, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_cols = [
        "step_index", "ts", "chainlink_price", "binance_price",
        "up_best_bid", "up_best_ask", "down_best_bid", "down_best_ask",
    ]
    steps = pd.read_parquet(hf(f"{ep}/steps.parquet", cache), columns=step_cols).sort_values("step_index").reset_index(drop=True)
    l0 = pq.read_table(
        hf(f"{ep}/book_levels.parquet", cache),
        columns=["step_index", "outcome", "side", "level_index", "price", "size"],
        filters=[("level_index", "=", 0)],
    ).to_pandas()
    l0 = l0.sort_values(["step_index", "outcome", "side"]).drop_duplicates(["step_index", "outcome", "side"], keep="last")
    # Verified contract from the independent Trent audit: outcome 0=UP, 1=DOWN; side 0=bid, 1=ask.
    piv = {}
    for outcome, oname in ((0, "up"), (1, "down")):
        for side, sname in ((0, "bid"), (1, "ask")):
            z = l0[(l0.outcome == outcome) & (l0.side == side)][["step_index", "size"]].rename(columns={"size": f"{oname}_{sname}_size"})
            piv[f"{oname}_{sname}_size"] = z
    for z in piv.values():
        steps = steps.merge(z, on="step_index", how="left")
    return steps, l0


def row_valid(r: pd.Series) -> bool:
    vals = [r.up_best_bid, r.up_best_ask, r.down_best_bid, r.down_best_ask]
    return all(math.isfinite(float(x)) for x in vals) and 0 < r.up_best_bid < r.up_best_ask < 1 and 0 < r.down_best_bid < r.down_best_ask < 1


def seq_from_indices(steps: pd.DataFrame, idxs: list[int], open_spot: float, close_ms: int) -> np.ndarray:
    out = np.full((SEQ_LEN, len(SEQ_NAMES)), np.nan, dtype=np.float32)
    rows = []
    for i in idxs[-SEQ_LEN:]:
        r = steps.iloc[i]
        yb, ya, nb, na = map(float, [r.up_best_bid, r.up_best_ask, r.down_best_bid, r.down_best_ask])
        ymid = safe_mid(yb, ya)
        ysp = ya - yb if 0 < yb < ya < 1 else float("nan")
        sp = float(r.binance_price)
        lr = math.log(sp / open_spot) if sp > 0 and open_spot > 0 else 0.0
        s2c = max((close_ms - int(r.ts)) / 1000.0, 0.0)
        rows.append([
            yb, ya, nb, na,
            math.log1p(max(float(r.up_bid_size), 0.0)),
            math.log1p(max(float(r.up_ask_size), 0.0)),
            math.log1p(max(float(r.down_bid_size), 0.0)),
            math.log1p(max(float(r.down_ask_size), 0.0)),
            lr, s2c / 900.0, ymid, ysp,
        ])
    if rows:
        out[:len(rows)] = np.asarray(rows, dtype=np.float32)
    return out


def make_example(ep: str, steps: pd.DataFrame, bn: BinanceAnchor, der: DeribitAnchor) -> tuple[Example | None, dict]:
    open_ms = episode_open_ms(ep); close_ms = open_ms + 900_000; open_s = open_ms // 1000
    open_bn = bn.open_price(open_s)
    if not (open_bn > 0 and math.isfinite(open_bn)):
        return None, {"episode": ep, "reason": "no_open_binance"}
    ts = pd.to_numeric(steps.ts, errors="coerce").to_numpy(np.int64)
    targets = np.arange(open_ms + 300_000, close_ms - 60_000 + 1, CADENCE_MS, dtype=np.int64)
    scan_idxs = previous_step_indices(ts, targets, 500)
    chosen = None
    for i in scan_idxs:
        r = steps.iloc[i]
        if not row_valid(r):
            continue
        if not all(math.isfinite(float(r[c])) and float(r[c]) >= 5.0 for c in ["up_ask_size", "down_ask_size"]):
            continue
        t = int(r.ts); s2c = int(round((close_ms - t) / 1000.0))
        if not (60 <= s2c <= 600):
            continue
        rv = bn.rv_annualized(t, 60); div = der.median_iv(t, 30)
        if not (math.isfinite(rv) and math.isfinite(div) and 0.05 <= rv <= 3.0 and 0.05 <= div <= 3.0):
            continue
        spot = float(r.binance_price)
        if not (spot > 0):
            continue
        rel = spot / open_bn
        p_rv = digital_prob_up(rel, s2c, rv); p_d = digital_prob_up(rel, s2c, div)
        if not (math.isfinite(p_rv) and math.isfinite(p_d)):
            continue
        p_lo, p_hi = min(p_rv, p_d), max(p_rv, p_d)
        ya, na = float(r.up_best_ask), float(r.down_best_ask)
        edge_y = p_lo - ya - TRAINING_FEE_RATE * ya * (1 - ya)
        edge_n = (1 - p_hi) - na - TRAINING_FEE_RATE * na * (1 - na)
        buy_y = edge_y >= 0.03 and float(r.up_ask_size) >= 5.0
        buy_n = edge_n >= 0.03 and float(r.down_ask_size) >= 5.0
        if not buy_y and not buy_n:
            continue
        fade_yes = bool(buy_y and (not buy_n or edge_y >= edge_n))
        edge = edge_y if fade_yes else edge_n
        chosen = {
            "i": i, "ts_ms": t, "s2c": s2c, "fade_yes": fade_yes,
            "edge_signal": round(float(edge), 6),
            "p_rv": round(float(p_rv), 6), "p_deribit": round(float(p_d), 6),
            "rv": round(float(rv), 6), "deribit_iv": round(float(div), 6),
            "rel_spot": round(float(rel), 8),
        }
        break
    if chosen is None:
        return None, {"episode": ep, "reason": "no_3c_signal", "sampled_states": len(scan_idxs)}

    i = int(chosen["i"]); r = steps.iloc[i]; t = int(chosen["ts_ms"])
    p_rv, p_d = float(chosen["p_rv"]), float(chosen["p_deribit"])
    rv60, div, rel_spot, s2c = float(chosen["rv"]), float(chosen["deribit_iv"]), float(chosen["rel_spot"]), float(chosen["s2c"])
    yb, ya, nb, na = map(float, [r.up_best_bid, r.up_best_ask, r.down_best_bid, r.down_best_ask])
    ymid, nmid = safe_mid(yb, ya), safe_mid(nb, na)
    ysp, nsp = ya-yb, na-nb
    ybs, yas = max(float(r.up_bid_size),0.0), max(float(r.up_ask_size),0.0)
    nbs, nas = max(float(r.down_bid_size),0.0), max(float(r.down_ask_size),0.0)
    yimb=(ybs-yas)/(ybs+yas+EPS); nimb=(nbs-nas)/(nbs+nas+EPS)
    anchor_mean=0.5*(p_rv+p_d)
    ret1,ret5,ret15=bn_return(bn,t,60),bn_return(bn,t,300),bn_return(bn,t,900)
    rv15=bn_rv(bn,t,15,min_obs=8)
    if not math.isfinite(rv15): rv15=rv60
    vol_exp=rv15/max(rv60,1e-6); sigma5=rv60*math.sqrt(300.0/YEAR_SECONDS); shockz=ret5/max(sigma5,1e-6)
    static=np.asarray([
        float(chosen["edge_signal"]),p_rv,p_d,anchor_mean,abs(p_rv-p_d),
        rv60,div,div-rv60,rel_spot,math.log(max(rel_spot,1e-8)),
        s2c/900.0,yb,ya,nb,na,ysp,nsp,ymid,nmid,
        math.log1p(ybs),math.log1p(yas),math.log1p(nbs),math.log1p(nas),
        yimb,nimb,ymid-anchor_mean if math.isfinite(ymid) else 0.0,
        ret1,ret5,ret15,rv15,vol_exp,shockz,
    ],dtype=np.float32)
    static[~np.isfinite(static)]=0.0

    hist_targets=np.arange(open_ms,t+1,CADENCE_MS,dtype=np.int64)
    hist_idxs=previous_step_indices(ts,hist_targets,500)
    if not hist_idxs or hist_idxs[-1] != i:
        hist_idxs.append(i)
    hist_idxs=hist_idxs[-SEQ_LEN:]
    seq=seq_from_indices(steps,hist_idxs,float(open_bn),close_ms)

    # Label is terminal PM consensus and is used only for pilot scoring after the
    # decision/features are frozen. It is never used in signal construction.
    last=steps.iloc[-1]
    settle_yes=float(float(last.up_best_bid) > float(last.down_best_bid))
    fade_ask=ya if chosen["fade_yes"] else na
    follow_ask=na if chosen["fade_yes"] else ya
    ex=Example(
        ts_ms=t, close_ts=open_s+900, cid=ep, seq=seq, static=static,
        settle_yes=settle_yes, fade_fill=0.0, follow_fill=0.0,
        fade_reward=0.0, follow_reward=0.0,
        fade_cost=fade_ask+TRAINING_FEE_RATE*fade_ask*(1-fade_ask),
        follow_cost=follow_ask+TRAINING_FEE_RATE*follow_ask*(1-follow_ask),
        fade_yes=bool(chosen["fade_yes"]),
    )
    meta={
        "episode":ep,"decision_ts_ms":t,"s2c":int(s2c),"fade_yes":bool(chosen["fade_yes"]),
        "edge_signal":float(chosen["edge_signal"]),"p_rv":p_rv,"p_deribit":p_d,
        "anchor_mean":float(anchor_mean),"sequence_length":len(hist_idxs),
        "sequence_span_s":float((int(steps.iloc[hist_idxs[-1]].ts)-int(steps.iloc[hist_idxs[0]].ts))/1000.0) if len(hist_idxs)>1 else 0.0,
        "settle_yes_terminal_pm_only":settle_yes,
    }
    return ex,meta


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--day",default="2026-03-14")
    ap.add_argument("--episodes",type=int,default=8)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); args.cache.mkdir(parents=True,exist_ok=True)
    d=date.fromisoformat(args.day)
    api=HfApi(); files=api.list_repo_files(REPO,repo_type="dataset",revision=REV)
    eps=sorted({str(Path(p).parent) for p in files if p.startswith("btc15m_") and args.day in p and p.endswith("/steps.parquet")})[:args.episodes]
    if not eps: raise RuntimeError("no Trent BTC15m episodes")

    bn_df=download_binance_1m(d,d,args.cache/"binance"); bn=BinanceAnchor.from_df(bn_df)
    inst=deribit_instruments(); selected=select_deribit_instruments(inst,bn_df,d,d)
    der_trades=fetch_deribit_trades(selected,d,d,args.cache/"deribit_trades.parquet"); der=DeribitAnchor.from_trades(der_trades,inst)
    scaler,models=load_frozen(args.model_dir)

    exs=[]; metas=[]; rejects=[]
    for ep in eps:
        print(f"EPISODE {ep}",flush=True)
        steps,_=load_episode(ep,args.cache/"trent")
        ex,meta=make_example(ep,steps,bn,der)
        if ex is None: rejects.append(meta)
        else: exs.append(ex); metas.append(meta)
    if not exs:
        raise RuntimeError(f"pilot produced no model examples: {rejects}")

    ps=[];fps=[];ds=[]
    for m in models:
        p,fp,dlt=predict(m,exs,scaler); ps.append(p);fps.append(fp);ds.append(dlt)
    p=np.mean(ps,0); fp=np.mean(fps,0); delta=np.mean(ds,0)
    anchor=np.asarray([float(e.static[3]) for e in exs]); y=np.asarray([e.settle_yes for e in exs])
    rows=[]
    for k,(e,m) in enumerate(zip(exs,metas)):
        rows.append({**m,"model_p":float(p[k]),"delta_logit":float(delta[k]),"pred_fill_fade":float(fp[k,0]),"pred_fill_follow":float(fp[k,1])})
    pd.DataFrame(rows).to_csv(args.out/"trent_v1_golden_rows.csv",index=False)
    report={
        "status":"TRENT_FROZEN_V1_GOLDEN_FEATURE_PILOT",
        "dataset":REPO,"revision":REV,"day":args.day,
        "episodes_requested":len(eps),"signals":len(exs),"rejects":rejects,
        "frozen":{"weights_retrained":False,"scaler_refit":False,"threshold":0.03,"training_fee_rate_for_feature_contract":TRAINING_FEE_RATE},
        "sequence":{"cadence_ms":CADENCE_MS,"lengths":[int(x["sequence_length"]) for x in metas],"spans_s":[float(x["sequence_span_s"]) for x in metas]},
        "probability":{"n":len(exs),"brier_model":brier(y,p),"brier_anchor":brier(y,anchor),"logloss_model":logloss(y,p),"logloss_anchor":logloss(y,anchor),"delta_mean":float(delta.mean()),"delta_abs_mean":float(np.abs(delta).mean())},
        "boundary":"Golden feature/inference pilot only. Terminal PM state is used only as an after-the-fact label. No execution/PnL claim; historical March fee and oracle-basis execution are deliberately deferred.",
    }
    (args.out/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
