from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr

ASSETS = ("BTC", "ETH", "SOL", "XRP")
SYMBOLS = {a: f"{a}USDT" for a in ASSETS}
TRAIN_START = "2026-06-01"
TRAIN_END = "2026-06-24"
VAL_START = "2026-06-24"
VAL_END = "2026-07-01"
TEST_START = "2026-07-01"
TEST_END = "2026-07-15"
DECISION_S2C = 60
FEE_RATE = 0.07
EDGE_FLOOR = 0.03
LIMIT_SLIP = 0.02
TAPE_SECONDS = 5
RESIDUAL_BOUND = 0.50
SEEDS = (7, 19, 42, 73, 101)
YEAR_SECONDS = 365.0 * 24 * 3600
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE_VISION = "https://data.binance.vision/data/spot/monthly/klines"

FEATURES = [
    "pm_last", "last_lag_s", "log_n30", "log_n60", "log_n120",
    "log_vol60", "pressure60", "mom15", "mom30", "mom60", "mom120",
    "std60", "range60", "mean60_minus_last",
    "finance_p", "finance_minus_pm", "log_rel_spot", "ret1m", "ret3m", "rv60",
    "asset_BTC", "asset_ETH", "asset_SOL", "asset_XRP",
]


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def fee_per_share(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    return FEE_RATE * p * (1.0 - p)


def logloss(y, p) -> float:
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())


def brier(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((y-p)**2))


def ece(y, p, bins: int = 10) -> float:
    y=np.asarray(y,float); p=np.asarray(p,float); out=0.0
    edges=np.linspace(0,1,bins+1)
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): out += float(m.mean()) * abs(float(y[m].mean())-float(p[m].mean()))
    return out


def get_json(sess: requests.Session, url: str, *, params=None, timeout=45, tries=6):
    err=None
    for i in range(tries):
        try:
            r=sess.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err=e; time.sleep(min(0.4*(2**i), 5.0) + random.random()*0.1)
    raise RuntimeError(f"GET failed {url}: {err}")


@dataclass(frozen=True)
class Market:
    slug: str
    asset: str
    start: int
    end: int
    decision: int
    condition_id: str
    up_token: str
    down_token: str
    label_up: float


def parse_market(m: dict, asset: str, start: int) -> Market | None:
    try:
        outcomes=json.loads(m["outcomes"]); tokens=json.loads(m["clobTokenIds"])
        prices=json.loads(m.get("outcomePrices") or "[]")
        oi={str(o).strip().lower():i for i,o in enumerate(outcomes)}
        if "up" not in oi or "down" not in oi: return None
        ui,di=oi["up"],oi["down"]
        if len(tokens) <= max(ui,di): return None
        if len(prices) <= max(ui,di): return None
        pp=[float(x) for x in prices]
        if max(pp) < 0.99: return None
        label=1.0 if pp[ui] > pp[di] else 0.0
        return Market(str(m["slug"]), asset, start, start+300, start+300-DECISION_S2C,
                      str(m["conditionId"]), str(tokens[ui]), str(tokens[di]), label)
    except Exception:
        return None


def fetch_hour(hour_start: int) -> tuple[list[Market], list[dict]]:
    sess=requests.Session(); sess.headers.update({"User-Agent":"main-sequence-sealed-research/1.0"})
    wanted=[]; meta={}
    for a in ASSETS:
        for t0 in range(hour_start, hour_start+3600, 300):
            slug=f"{a.lower()}-updown-5m-{t0}"; wanted.append((slug,a,t0))
    params=[("slug",x[0]) for x in wanted] + [("closed","true"),("limit",100)]
    js=get_json(sess,GAMMA+"/markets",params=params)
    byslug={str(x.get("slug")):x for x in js}
    markets=[]
    for slug,a,t0 in wanted:
        m=parse_market(byslug.get(slug,{}),a,t0)
        if m is not None: markets.append(m)
    if not markets: return [],[]

    def pull_group(group: list[Market]) -> list[dict]:
        conds=",".join(m.condition_id for m in group)
        q={"market":conds,"start":hour_start,"end":hour_start+3600,"limit":10000,"offset":0,"takerOnly":"true"}
        rows=get_json(sess,DATA_API+"/trades",params=q,timeout=60)
        if len(rows) >= 10000 and len(group)>1:
            mid=len(group)//2
            return pull_group(group[:mid])+pull_group(group[mid:])
        if len(rows) >= 10000:
            # One five-minute market should not hit the 10k cap; fail rather than silently truncate.
            raise RuntimeError(f"trade cap hit for {group[0].slug}")
        return rows
    trades=[]
    for i in range(0,len(markets),24): trades.extend(pull_group(markets[i:i+24]))
    return markets,trades


def fetch_phase(start: str, end: str, workers: int = 20) -> tuple[list[Market], pd.DataFrame, dict]:
    lo,hi=ts(start),ts(end)
    hours=list(range(lo,hi,3600)); markets=[]; trades=[]; failures=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(fetch_hour,h):h for h in hours}
        for k,f in enumerate(as_completed(fut),1):
            h=fut[f]
            try:
                mm,tt=f.result(); markets.extend(mm); trades.extend(tt)
            except Exception as e:
                failures.append({"hour":h,"error":repr(e)})
            if k%48==0: print("FETCH_HOURS",k,"/",len(hours),"markets",len(markets),"trades",len(trades),"fail",len(failures),flush=True)
    markets.sort(key=lambda m:(m.start,m.asset))
    td=pd.DataFrame(trades)
    cov={"hours":len(hours),"failed_hours":failures,"expected_markets":len(hours)*12*len(ASSETS),"mapped_markets":len(markets),"trade_rows":len(td)}
    return markets,td,cov


def load_binance_month(symbol: str, ym: str, sess: requests.Session) -> pd.DataFrame:
    name=f"{symbol}-1m-{ym}.zip"; url=f"{BINANCE_VISION}/{symbol}/1m/{name}"
    r=sess.get(url,timeout=120); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name=z.namelist()[0]
        df=pd.read_csv(z.open(csv_name),header=None,usecols=[0,1,4])
    df.columns=["open_ts","open","close"]
    ot=df.open_ts.to_numpy(np.int64)
    if np.nanmedian(ot)>1e14: ot=ot//1000
    df["open_ts"]=(ot//1000).astype(np.int64)
    df["open"]=df.open.astype(float); df["close"]=df.close.astype(float)
    return df


class BinanceSeries:
    def __init__(self, df: pd.DataFrame):
        x=df.sort_values("open_ts").drop_duplicates("open_ts",keep="last")
        self.t=x.open_ts.to_numpy(np.int64); self.o=x.open.to_numpy(float); self.c=x.close.to_numpy(float)
        self.idx={int(t):i for i,t in enumerate(self.t)}
    def features(self,start: int,decision: int):
        i0=self.idx.get(start); idm=self.idx.get(decision-60)
        if i0 is None or idm is None or idm-i0<3: return None
        op=float(self.o[i0]); spot=float(self.c[idm])
        if not (op>0 and spot>0): return None
        lo=max(0,idm-60); closes=self.c[lo:idm+1]
        closes=closes[np.isfinite(closes)&(closes>0)]
        if len(closes)<30: return None
        r=np.diff(np.log(closes)); sigma=float(np.std(r,ddof=1))*math.sqrt(365*24*60)
        if not (0.02<sigma<6 and math.isfinite(sigma)): return None
        rel=spot/op; T=max(1,(start+300-decision))/YEAR_SECONDS
        den=sigma*math.sqrt(T)
        d2=(math.log(rel)-0.5*sigma*sigma*T)/den if den>0 else (50 if rel>=1 else -50)
        fp=float(ndtr(d2))
        c1=float(self.c[idm-1]); c3=float(self.c[max(i0,idm-3)])
        return {"finance_p":fp,"log_rel_spot":math.log(rel),"ret1m":math.log(spot/c1),"ret3m":math.log(spot/c3),"rv60":sigma}


def load_binance(phase: str) -> dict[str,BinanceSeries]:
    months=("2026-05","2026-06") if phase=="train" else ("2026-06","2026-07")
    sess=requests.Session(); sess.headers.update({"User-Agent":"main-sequence-sealed-research/1.0"})
    out={}
    for a,sym in SYMBOLS.items():
        parts=[]
        for ym in months:
            print("BINANCE",phase,sym,ym,flush=True); parts.append(load_binance_month(sym,ym,sess))
        out[a]=BinanceSeries(pd.concat(parts,ignore_index=True))
    return out


def normalize_trade_rows(td: pd.DataFrame) -> dict[str,pd.DataFrame]:
    if td.empty: return {}
    need={"conditionId","timestamp","price","size","side","outcome"}
    if not need.issubset(td.columns): raise RuntimeError(f"trade schema missing {need-set(td.columns)}")
    x=td.copy(); x["timestamp"]=pd.to_numeric(x.timestamp,errors="coerce"); x["price"]=pd.to_numeric(x.price,errors="coerce"); x["size"]=pd.to_numeric(x.size,errors="coerce")
    x=x.dropna(subset=["timestamp","price","size"]); x=x[(x.price>0)&(x.price<1)&(x.size>0)]
    x["timestamp"]=x.timestamp.astype(np.int64)
    x["outcome_l"]=x.outcome.astype(str).str.lower().str.strip(); x["side_u"]=x.side.astype(str).str.upper().str.strip()
    x=x[x.outcome_l.isin(["up","down"])]
    x["p_up"]=np.where(x.outcome_l.eq("up"),x.price,1.0-x.price)
    x["pressure"]=np.where(((x.outcome_l.eq("up"))&(x.side_u.eq("BUY")))|((x.outcome_l.eq("down"))&(x.side_u.eq("SELL"))),1.0,-1.0)
    return {str(cid):g.sort_values("timestamp").reset_index(drop=True) for cid,g in x.groupby("conditionId",sort=False)}


@dataclass
class Example:
    slug: str; condition_id: str; asset: str; start: int; decision: int; label: float; pm_last: float
    x: np.ndarray


def first_at_or_after(g: pd.DataFrame, cutoff: int) -> float:
    q=g[g.timestamp>=cutoff]
    return float(q.p_up.iloc[0]) if len(q) else float("nan")


def build_examples(markets: list[Market], td: pd.DataFrame, bs: dict[str,BinanceSeries]) -> list[Example]:
    tm=normalize_trade_rows(td); out=[]
    for m in markets:
        g=tm.get(m.condition_id)
        if g is None or g.empty: continue
        pre=g[(g.timestamp>=m.start)&(g.timestamp<m.decision)].copy()
        if len(pre)<4: continue
        lag=m.decision-int(pre.timestamp.iloc[-1])
        if lag<0 or lag>45: continue
        pm=float(pre.p_up.iloc[-1])
        if not (0.01<pm<0.99): continue
        bf=bs[m.asset].features(m.start,m.decision)
        if bf is None: continue
        def w(sec): return pre[pre.timestamp>=m.decision-sec]
        def mom(sec):
            q=w(sec); return float(pm-q.p_up.iloc[0]) if len(q)>=2 else 0.0
        q30,q60,q120=w(30),w(60),w(120)
        size60=float(q60["size"].sum()) if len(q60) else 0.0
        press60=float((q60.pressure*q60["size"]).sum()/(size60+1e-9)) if len(q60) else 0.0
        vals=q60.p_up.to_numpy(float) if len(q60) else np.asarray([pm])
        feat=np.asarray([
            pm, float(lag), math.log1p(len(q30)), math.log1p(len(q60)), math.log1p(len(q120)),
            math.log1p(size60), press60, mom(15), mom(30), mom(60), mom(120),
            float(np.std(vals)) if len(vals)>1 else 0.0, float(np.max(vals)-np.min(vals)), float(np.mean(vals)-pm),
            bf["finance_p"], bf["finance_p"]-pm, bf["log_rel_spot"], bf["ret1m"], bf["ret3m"], bf["rv60"],
            float(m.asset=="BTC"),float(m.asset=="ETH"),float(m.asset=="SOL"),float(m.asset=="XRP"),
        ],dtype=np.float32)
        if np.isfinite(feat).all(): out.append(Example(m.slug,m.condition_id,m.asset,m.start,m.decision,m.label,pm,feat))
    out.sort(key=lambda e:(e.start,e.asset)); return out


@dataclass
class Scaler:
    mean: np.ndarray; std: np.ndarray
    @classmethod
    def fit(cls,xs):
        a=np.stack([e.x for e in xs]); mu=a.mean(0); sd=a.std(0); sd[sd<1e-6]=1.0; return cls(mu,sd)
    def transform(self,xs): return ((np.stack([e.x for e in xs])-self.mean)/self.std).astype(np.float32)


class ResidualMLP(nn.Module):
    def __init__(self,nf: int):
        super().__init__(); self.net=nn.Sequential(nn.Linear(nf,48),nn.SiLU(),nn.LayerNorm(48),nn.Dropout(.10),nn.Linear(48,24),nn.SiLU(),nn.Linear(24,1))
    def forward(self,x): return RESIDUAL_BOUND*torch.tanh(self.net(x).squeeze(1))


def apply_residual(anchor,delta):
    a=anchor.clamp(1e-5,1-1e-5); return torch.sigmoid(torch.logit(a)+delta)


def train_one(train,val,sc,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    xt=torch.from_numpy(sc.transform(train)); yt=torch.tensor([e.label for e in train],dtype=torch.float32); at=torch.tensor([e.pm_last for e in train],dtype=torch.float32)
    xv=torch.from_numpy(sc.transform(val)); yv=torch.tensor([e.label for e in val],dtype=torch.float32); av=torch.tensor([e.pm_last for e in val],dtype=torch.float32)
    ds=torch.utils.data.TensorDataset(xt,yt,at); dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True)
    m=ResidualMLP(len(FEATURES)); opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=1e-3)
    best=None; bestv=1e9; stale=0; bestep=0
    for ep in range(80):
        m.train()
        for xb,yb,ab in dl:
            opt.zero_grad(); d=m(xb); p=apply_residual(ab,d); loss=F.binary_cross_entropy(p,yb)+0.02*(d*d).mean(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad():
            d=m(xv); p=apply_residual(av,d); v=float(F.binary_cross_entropy(p,yv)+0.02*(d*d).mean())
        if v<bestv-1e-5: bestv=v; best={k:z.detach().cpu().clone() for k,z in m.state_dict().items()}; stale=0; bestep=ep+1
        else:
            stale+=1
            if stale>=10: break
    m.load_state_dict(best); return m,{"seed":seed,"best_val_loss":bestv,"best_epoch":bestep}


def predict(models,xs,sc):
    x=torch.from_numpy(sc.transform(xs)); a=torch.tensor([e.pm_last for e in xs],dtype=torch.float32); ps=[]
    with torch.no_grad():
        for m in models: m.eval(); ps.append(apply_residual(a,m(x)).numpy())
    return np.mean(ps,axis=0)


def probability_summary(xs, p):
    y=np.asarray([e.label for e in xs]); pm=np.asarray([e.pm_last for e in xs]); fp=np.asarray([e.x[FEATURES.index("finance_p")] for e in xs])
    return {"n":len(xs),"brier_model":brier(y,p),"brier_pm_last":brier(y,pm),"brier_finance":brier(y,fp),
            "logloss_model":logloss(y,p),"logloss_pm_last":logloss(y,pm),"logloss_finance":logloss(y,fp),
            "ece_model":ece(y,p),"ece_pm_last":ece(y,pm),"brier_gain_vs_pm":brier(y,pm)-brier(y,p)}


def paired_day_bootstrap(xs,p,reps=5000,seed=20260815):
    y=np.asarray([e.label for e in xs]); pm=np.asarray([e.pm_last for e in xs]); p=np.asarray(p); days=np.asarray([pd.Timestamp(e.start,unit="s",tz="UTC").strftime("%Y-%m-%d") for e in xs])
    vals=[]
    for d in sorted(set(days)):
        m=days==d; vals.append(float(np.mean((y[m]-pm[m])**2-(y[m]-p[m])**2)))
    vals=np.asarray(vals); rng=np.random.default_rng(seed); z=np.empty(reps)
    for i in range(reps): z[i]=rng.choice(vals,len(vals),replace=True).mean()
    return {"day_mean_gain":float(vals.mean()),"ci95":[float(np.quantile(z,.025)),float(np.quantile(z,.975))],"days":len(vals)}


def policy_summary(xs,p,td):
    tm=normalize_trade_rows(td); rows=[]
    for e,prob in zip(xs,p):
        up_ref=e.pm_last; dn_ref=1-up_ref
        up_edge=float(prob)-up_ref-fee_per_share(up_ref); dn_edge=(1-float(prob))-dn_ref-fee_per_share(dn_ref)
        if max(up_edge,dn_edge)<EDGE_FLOOR: continue
        choose_up=up_edge>=dn_edge; ref=up_ref if choose_up else dn_ref; outcome="up" if choose_up else "down"; fair=float(prob) if choose_up else 1-float(prob)
        limit=min(0.99,ref+LIMIT_SLIP); won=(e.label>=.5)==choose_up
        paper_cost=ref+fee_per_share(ref); paper_reward=(1.0 if won else 0.0)-paper_cost
        fill=None; g=tm.get(e.condition_id)
        if g is not None:
            q=g[(g.timestamp>=e.decision)&(g.timestamp<=e.decision+TAPE_SECONDS)&(g.outcome_l==outcome)&(g.side_u=="BUY")&(g.price<=limit)]
            if len(q): fill=float(q.iloc[0].price)
        tape_reward=None; tape_cost=None
        if fill is not None:
            tape_cost=fill+fee_per_share(fill); tape_reward=(1.0 if won else 0.0)-tape_cost
        rows.append({"slug":e.slug,"asset":e.asset,"decision":e.decision,"side":"Up" if choose_up else "Down","fair":fair,"ref":ref,"signal_edge":max(up_edge,dn_edge),"limit":limit,"won":won,"paper_cost":paper_cost,"paper_reward":paper_reward,"tape_fill":fill,"tape_cost":tape_cost,"tape_reward":tape_reward})
    df=pd.DataFrame(rows)
    def summ(z,reward,cost):
        if z.empty: return {"n":0,"edge_share":None,"roi":None,"win_rate":None,"ci95_day":[None,None]}
        vals=z[reward].astype(float); costs=z[cost].astype(float); day=pd.to_datetime(z.decision,unit="s",utc=True).dt.strftime("%Y-%m-%d"); daily=pd.DataFrame({"day":day,"r":vals}).groupby("day").r.mean().to_numpy()
        rng=np.random.default_rng(20260815); boot=np.asarray([rng.choice(daily,len(daily),replace=True).mean() for _ in range(4000)]) if len(daily)>1 else daily
        return {"n":len(z),"edge_share":float(vals.mean()),"roi":float(vals.sum()/costs.sum()),"win_rate":float(z.won.mean()),"ci95_day":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
    if df.empty: return {"signals":0,"paper":summ(df,"paper_reward","paper_cost"),"tape_5s":summ(df,"paper_reward","paper_cost")},df
    tape=df[df.tape_fill.notna()].copy()
    out={"signals":len(df),"tape_fill_rate":float(len(tape)/len(df)),"paper":summ(df,"paper_reward","paper_cost"),"tape_5s":summ(tape,"tape_reward","tape_cost")}
    out["by_asset_tape"]={a:summ(tape[tape.asset==a],"tape_reward","tape_cost") for a in ASSETS}
    return out,df


def save_examples(xs,path):
    rows=[]
    for e in xs:
        r={"slug":e.slug,"condition_id":e.condition_id,"asset":e.asset,"start":e.start,"decision":e.decision,"label":e.label,"pm_last":e.pm_last}
        r.update({k:float(v) for k,v in zip(FEATURES,e.x)}); rows.append(r)
    pd.DataFrame(rows).to_csv(path,index=False)


def train_phase(args):
    args.out.mkdir(parents=True,exist_ok=True)
    markets,td,cov=fetch_phase(TRAIN_START,VAL_END); bs=load_binance("train"); xs=build_examples(markets,td,bs)
    train=[e for e in xs if ts(TRAIN_START)<=e.start<ts(TRAIN_END)]; val=[e for e in xs if ts(VAL_START)<=e.start<ts(VAL_END)]
    if len(train)<8000 or len(val)<2000: raise RuntimeError(f"insufficient examples train={len(train)} val={len(val)} coverage={cov}")
    sc=Scaler.fit(train); models=[]; metas=[]; states={}
    for seed in SEEDS:
        print("TRAIN_SEED",seed,flush=True); m,meta=train_one(train,val,sc,seed); models.append(m); metas.append(meta); states[str(seed)]=m.state_dict()
    pv=predict(models,val,sc); vsummary=probability_summary(val,pv)
    contract={"train":[TRAIN_START,TRAIN_END],"validation":[VAL_START,VAL_END],"sealed_test":[TEST_START,TEST_END],"assets":ASSETS,"decision_s2c":DECISION_S2C,
              "fee_rate":FEE_RATE,"edge_floor":EDGE_FLOOR,"limit_slip":LIMIT_SLIP,"tape_seconds":TAPE_SECONDS,"residual_bound":RESIDUAL_BOUND,"seeds":SEEDS,
              "features":FEATURES,"model":"PM-last hard anchor + bounded logit residual MLP; trade microstructure + Binance RV features",
              "isolation":"train phase generates/fetches only June markets/trades and May-June Binance data; July data is first requested by downstream sealed test job"}
    np.savez(args.out/"scaler.npz",mean=sc.mean,std=sc.std)
    torch.save({"states":states,"features":FEATURES,"seeds":SEEDS},args.out/"model.pt")
    (args.out/"FROZEN_CONTRACT.json").write_text(json.dumps(contract,indent=2))
    summary={"name":"Main Sequence B / pre-July 5m multiasset frozen train","coverage":cov,"counts":{"train":len(train),"validation":len(val),"train_by_asset":pd.Series([e.asset for e in train]).value_counts().to_dict(),"val_by_asset":pd.Series([e.asset for e in val]).value_counts().to_dict()},"validation":vsummary,"validation_gain_day_bootstrap":paired_day_bootstrap(val,pv),"models":metas,"contract":contract}
    (args.out/"train_summary.json").write_text(json.dumps(summary,indent=2,default=float)); save_examples(train,args.out/"train_examples.csv"); save_examples(val,args.out/"validation_examples.csv")
    print(json.dumps(summary,indent=2,default=float),flush=True)


def test_phase(args):
    args.out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((args.model_dir/"FROZEN_CONTRACT.json").read_text())
    assert contract["sealed_test"]==[TEST_START,TEST_END] and contract["validation"][1]==TEST_START
    assert contract["features"]==FEATURES and contract["edge_floor"]==EDGE_FLOOR and contract["limit_slip"]==LIMIT_SLIP
    markets,td,cov=fetch_phase(TEST_START,TEST_END); bs=load_binance("test"); xs=build_examples(markets,td,bs)
    test=[e for e in xs if ts(TEST_START)<=e.start<ts(TEST_END)]
    expected_days=pd.date_range(TEST_START,pd.Timestamp(TEST_END)-pd.Timedelta(days=1),freq="D",tz="UTC").strftime("%Y-%m-%d").tolist(); got_days=sorted(set(pd.Timestamp(e.start,unit="s",tz="UTC").strftime("%Y-%m-%d") for e in test)); missing=[d for d in expected_days if d not in got_days]
    if len(test)<5000 or missing: raise RuntimeError(f"sealed test coverage insufficient n={len(test)} missing_days={missing} coverage={cov}")
    z=np.load(args.model_dir/"scaler.npz"); sc=Scaler(z["mean"],z["std"]); ck=torch.load(args.model_dir/"model.pt",map_location="cpu",weights_only=False)
    models=[]
    for seed in ck["seeds"]:
        m=ResidualMLP(len(FEATURES)); m.load_state_dict(ck["states"][str(seed)]); models.append(m)
    p=predict(models,test,sc); ps=probability_summary(test,p); boot=paired_day_bootstrap(test,p); policy,pdf=policy_summary(test,p,td)
    by_asset={}
    for a in ASSETS:
        idx=[i for i,e in enumerate(test) if e.asset==a]
        by_asset[a]=probability_summary([test[i] for i in idx],p[idx]) if idx else {"n":0}
    summary={"name":"Main Sequence B / SEALED July 1-14 5m multiasset test","coverage":cov,"counts":{"test":len(test),"by_asset":pd.Series([e.asset for e in test]).value_counts().to_dict(),"missing_calendar_days":missing},"probability":ps,"brier_gain_day_bootstrap":boot,"by_asset":by_asset,"policy":policy,
             "execution_note":"paper uses last pre-decision public trade as reference; tape_5s requires a real public taker BUY print in the chosen outcome within 5 seconds at or below the predeclared ref+2c limit. It is a tape-compatible fill proxy, not queue/depth proof.",
             "leakage_note":"all trade features are timestamp < decision; Binance features use only completed 1m bars ending <= decision; July was not fetched by the train job."}
    (args.out/"test_summary.json").write_text(json.dumps(summary,indent=2,default=float)); save_examples(test,args.out/"test_examples.csv")
    pd.DataFrame({"slug":[e.slug for e in test],"asset":[e.asset for e in test],"start":[e.start for e in test],"label":[e.label for e in test],"pm_last":[e.pm_last for e in test],"model_p":p}).to_csv(args.out/"test_probabilities.csv",index=False)
    pdf.to_csv(args.out/"policy_trades.csv",index=False)
    lines=["# Main Sequence B — sealed July 1–14 test","",f"Test examples: {len(test)} / mapped markets {cov['mapped_markets']}.","","## Probability","","```json",json.dumps(ps,indent=2),"```","","## Day-block Brier gain vs PM last trade","","```json",json.dumps(boot,indent=2),"```","","## Policy","","```json",json.dumps(policy,indent=2,default=float),"```",""]
    (args.out/"SUMMARY.md").write_text("\n".join(lines)); print((args.out/"SUMMARY.md").read_text(),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["train","test"]); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--model-dir",type=Path)
    a=ap.parse_args()
    if a.phase=="train": train_phase(a)
    else:
        if a.model_dir is None: raise SystemExit("--model-dir required")
        test_phase(a)

if __name__=="__main__": main()
