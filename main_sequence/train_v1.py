from __future__ import annotations

import argparse
import json
import math
import random
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pm_structural.recalc import BinanceAnchor, download_binance_1m
from main_sequence.train_v0 import (
    STATIC_NAMES, SEQ_NAMES, SEEDS, Dataset, Scaler, build_examples, cluster_ci, set_seed,
)

EPS = 1e-6


class MainSequenceV1(nn.Module):
    """Anchor-residual model: finance anchor stays hard; NN learns bounded logit correction.

    The settlement head is dense-supervised on every historical signal. The fill
    head is separate and never gates the value loss, avoiding logged-action bias.
    """
    def __init__(self, seq_dim: int, static_dim: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, 32, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_dim, 32), nn.SiLU(), nn.LayerNorm(32))
        self.fuse = nn.Sequential(
            nn.Linear(64, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 48), nn.SiLU()
        )
        self.delta = nn.Linear(48, 1)
        self.fill = nn.Linear(48, 2)

    def forward(self, seq, mask, static):
        lengths = mask.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        z = self.fuse(torch.cat([h[-1], self.static(static)], dim=1))
        # Bound correction to ±0.75 logit so the network refines rather than replaces finance.
        delta = 0.75 * torch.tanh(self.delta(z).squeeze(1))
        return delta, self.fill(z)


def implied_prob(anchor_p: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    p = anchor_p.clamp(EPS, 1 - EPS)
    return torch.sigmoid(torch.logit(p) + delta)


def pos_weights(train_ex):
    fade = np.asarray([e.fade_fill for e in train_ex], dtype=float)
    follow = np.asarray([e.follow_fill for e in train_ex], dtype=float)
    out = []
    for x in (fade, follow):
        pos = max(float(x.sum()), 1.0)
        neg = max(float(len(x) - x.sum()), 1.0)
        out.append(min(neg / pos, 25.0))
    return torch.tensor(out, dtype=torch.float32)


def batch_loss(model, batch, posw):
    seq, mask, stat, _, fill, settle = batch
    delta, fill_logit = model(seq, mask, stat)
    # static index 3 is the unscaled anchor_mean only before scaling, so recover it
    # via an explicit extra value carried by caller instead of trying to invert scaler.
    raise RuntimeError("batch_loss should not be called directly")


class DenseDataset(torch.utils.data.Dataset):
    def __init__(self, exs, scaler):
        self.base = Dataset(exs, scaler)
        self.exs = exs

    def __len__(self): return len(self.exs)

    def __getitem__(self, i):
        seq, mask, stat, _, fill, settle = self.base[i]
        anchor = torch.tensor(float(self.exs[i].static[3]), dtype=torch.float32)
        return seq, mask, stat, fill, settle, anchor


def compute_loss(model, batch, posw):
    seq, mask, stat, fill, settle, anchor = batch
    delta, fill_logit = model(seq, mask, stat)
    p = implied_prob(anchor, delta)
    settle_loss = F.binary_cross_entropy(p, settle)
    fill_loss = F.binary_cross_entropy_with_logits(fill_logit, fill, pos_weight=posw)
    # Small prior penalty: zero correction means "trust finance anchor".
    prior = (delta * delta).mean()
    return settle_loss + 0.20 * fill_loss + 0.02 * prior, settle_loss, fill_loss, prior


def train_one(seed, train_ex, val_ex, scaler):
    set_seed(seed)
    tr = DenseDataset(train_ex, scaler)
    va = DenseDataset(val_ex, scaler)
    posw = pos_weights(train_ex)
    model = MainSequenceV1(len(SEQ_NAMES), len(STATIC_NAMES))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(va, batch_size=len(va), shuffle=False)
    best, best_val, stale, hist = None, float("inf"), 0, []
    for epoch in range(120):
        model.train(); ls = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss, *_ = compute_loss(model, batch, posw)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ls.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            loss, sl, fl, pr = compute_loss(model, next(iter(vloader)), posw)
            v = float(loss)
        hist.append({"epoch": epoch+1, "train": float(np.mean(ls)), "val": v,
                     "val_settle": float(sl), "val_fill": float(fl), "val_prior": float(pr)})
        if v < best_val - 1e-5:
            best_val = v
            best = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 12: break
    model.load_state_dict(best)
    return model, hist, best_val, posw.tolist()


def predict(model, exs, scaler):
    ds = DenseDataset(exs, scaler)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    ps, fps, ds_ = [], [], []
    model.eval()
    with torch.no_grad():
        for seq, mask, stat, _, _, anchor in loader:
            delta, fl = model(seq, mask, stat)
            ps.append(implied_prob(anchor, delta).numpy())
            fps.append(torch.sigmoid(fl).numpy())
            ds_.append(delta.numpy())
    return np.concatenate(ps), np.concatenate(fps), np.concatenate(ds_)


def action_rewards(e, p):
    # Dense expected terminal rewards at the decision ask, fee included.
    # fade_yes indicates which side the frozen 3c structural system chose.
    py = float(p)
    pfade = py if e.fade_yes else (1.0 - py)
    pfollow = 1.0 - pfade
    return np.asarray([pfade - e.fade_cost, pfollow - e.follow_cost], dtype=float)


def realized_for(e, action):
    if action == 0:
        return e.fade_fill, e.fade_reward, e.fade_cost, "fade"
    return e.follow_fill, e.follow_reward, e.follow_cost, "follow"


def eval_policy(exs, probs, fill_probs, name, mode):
    rows = []
    decisions = {"fade": 0, "follow": 0, "abstain": 0}
    for i, e in enumerate(exs):
        er = action_rewards(e, probs[i])
        if mode == "baseline":
            a = 0
        elif mode == "value":
            a = int(np.argmax(er))
            if er[a] <= 0:
                decisions["abstain"] += 1; continue
        elif mode == "exec_weighted":
            score = fill_probs[i] * np.maximum(er, 0.0)
            a = int(np.argmax(score))
            if score[a] <= 0:
                decisions["abstain"] += 1; continue
        else: raise ValueError(mode)
        fill, rew, cost, aname = realized_for(e, a)
        decisions[aname] += 1
        if fill < 0.5: continue
        rows.append({"ts_ms": e.ts_ms, "cid": e.cid, "action": aname,
                     "real_reward": rew, "cost": cost, "pred_p_yes": float(probs[i]),
                     "pred_fill": float(fill_probs[i, a]), "pred_edge": float(er[a])})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"name": name, "fills": 0, "edge_share": None, "roi": None,
                "ci95": [None, None], "decisions": decisions}, df
    pnl, cost = float(df.real_reward.sum()), float(df.cost.sum())
    return {"name": name, "fills": int(len(df)), "edge_share": float(df.real_reward.mean()),
            "roi": pnl / cost if cost > 0 else None, "ci95": cluster_ci(df),
            "decisions": decisions,
            "filled_actions": {k: int(v) for k, v in df.action.value_counts().to_dict().items()}}, df


def brier(y, p): return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))

def logloss(y, p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pm-dir",type=Path,required=True); ap.add_argument("--records",type=Path,required=True)
    ap.add_argument("--prior-summary",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--start",default="2026-05-27"); ap.add_argument("--end",default="2026-06-24")
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True); args.cache.mkdir(parents=True,exist_ok=True)
    prior=json.loads(args.prior_summary.read_text()); split_ts=int(prior["split_close_ts"])
    start,end=date.fromisoformat(args.start),date.fromisoformat(args.end)
    bn=BinanceAnchor.from_df(download_binance_1m(start,end,args.cache/"binance"))
    exs=build_examples(args.pm_dir,args.records,bn)
    pre=[e for e in exs if e.close_ts<=split_ts]; hold=[e for e in exs if e.close_ts>split_ts]
    n=int(len(pre)*0.80); tr,val=pre[:n],pre[n:]; scaler=Scaler.fit(tr)
    print(f"examples={len(exs)} train={len(tr)} val={len(val)} holdout={len(hold)}",flush=True)
    print(f"fill labels train fade={sum(e.fade_fill for e in tr):.0f} follow={sum(e.follow_fill for e in tr):.0f}; value labels each action={len(tr)}",flush=True)

    P=[]; FP=[]; D=[]; metas=[]; states={}
    for seed in SEEDS:
        print(f"training v1 seed={seed}",flush=True)
        m,h,bv,pw=train_one(seed,tr,val,scaler); p,fp,d=predict(m,hold,scaler)
        P.append(p); FP.append(fp); D.append(d); metas.append({"seed":seed,"best_val_loss":bv,"epochs":len(h),"fill_pos_weight":pw})
        states[str(seed)]=m.state_dict()
    p=np.mean(P,0); fp=np.mean(FP,0); delta=np.mean(D,0)
    anchor=np.asarray([float(e.static[3]) for e in hold]); y=np.asarray([e.settle_yes for e in hold])

    policies=[]; frames={}
    for name,mode in [("frozen_3c_fade","baseline"),("main_sequence_v1_value","value"),("main_sequence_v1_exec","exec_weighted")]:
        met,df=eval_policy(hold,p,fp,name,mode); policies.append(met); frames[name]=df
        print(json.dumps(met),flush=True)
    summary={
        "name":"Main Sequence v1","split_close_utc":datetime.fromtimestamp(split_ts,tz=timezone.utc).isoformat(),
        "data":{"examples":len(exs),"train":len(tr),"validation":len(val),"holdout":len(hold)},
        "architecture":"GRU PM-book sequence + static cross-market state; bounded logit correction to hard BN/Deribit anchor; independent fill head",
        "value_supervision":"dense counterfactual terminal payoff available for both fade and follow on every training signal; settlement head trained on all signals",
        "seeds":metas,
        "holdout_probability":{"brier_model":brier(y,p),"brier_anchor":brier(y,anchor),"logloss_model":logloss(y,p),"logloss_anchor":logloss(y,anchor),"delta_logit_mean":float(delta.mean()),"delta_logit_abs_mean":float(np.abs(delta).mean())},
        "policies":policies,
        "notes":["3c structural threshold and chronological split remain frozen from the prior replay.","No holdout outcome is used in fitting, scaling, early stopping, or policy construction.","Value and execution are separated to avoid v0 logged-fill action-selection bias.","Both value-only and execution-weighted policies are predeclared and reported; neither is selected post hoc."],
    }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    for k,df in frames.items(): df.to_csv(args.out/f"{k}_fills.csv",index=False)
    pd.DataFrame({"ts_ms":[e.ts_ms for e in hold],"anchor_p":anchor,"model_p":p,"settle_yes":y,"delta_logit":delta}).to_csv(args.out/"holdout_probabilities.csv",index=False)
    np.savez(args.out/"scaler.npz",static_mean=scaler.static_mean,static_std=scaler.static_std,seq_mean=scaler.seq_mean,seq_std=scaler.seq_std,static_names=np.asarray(STATIC_NAMES),seq_names=np.asarray(SEQ_NAMES))
    torch.save({"states":states,"static_names":STATIC_NAMES,"seq_names":SEQ_NAMES,"seeds":SEEDS},args.out/"main_sequence_v1.pt")
    lines=["# Main Sequence v1","",f"Frozen 3c; holdout={len(hold)}; split={summary['split_close_utc']}.","","| policy | strict tape fills | edge/share | fee ROI | day-cluster 95% CI | decisions |","|---|---:|---:|---:|---|---|"]
    for m in policies: lines.append(f"| {m['name']} | {m['fills']} | {m['edge_share']} | {m['roi']} | {m['ci95']} | {m['decisions']} |")
    hp=summary['holdout_probability']; lines += ["",f"Probability OOS: Brier model={hp['brier_model']:.6f} vs anchor={hp['brier_anchor']:.6f}; logloss model={hp['logloss_model']:.6f} vs anchor={hp['logloss_anchor']:.6f}."]
    (args.out/"SUMMARY.md").write_text("\n".join(lines)+"\n"); print((args.out/"SUMMARY.md").read_text(),flush=True)

if __name__=="__main__": main()
