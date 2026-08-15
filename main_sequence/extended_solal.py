from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from pm_structural.recalc import (
    BinanceAnchor,
    DeribitAnchor,
    deribit_instruments,
    digital_prob_up,
    download_binance_1m,
    fetch_deribit_trades,
    select_deribit_instruments,
)

HF_REPO = "Solal9/polymarket-crypto-updown-binary"
HF_REV = "c17800caa413042941bbd8a9f266b23b6f5f8ed4"
SNAP_FILE = "polymarket_btc_15m_snapshots.csv"
BOOK_FILE = "polymarket_btc_15m_orderbook.csv"
FEE_RATE = 0.07  # frozen from the prior audited BTC15m corpus
YEAR_SECONDS = 365.0 * 24 * 3600
SEQ_LEN = 10
SEEDS = (7, 19, 42, 73, 101)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            pass
    return []


def to_ms(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, utc=True, errors="coerce").astype("int64") // 1_000_000).astype("Int64")


def resolve_side(row) -> float:
    outs = [str(v).lower() for v in parse_list(row.outcomes)]
    ps = []
    for v in parse_list(row.outcome_prices):
        try:
            ps.append(float(v))
        except Exception:
            ps.append(float("nan"))
    if len(outs) != len(ps) or not outs:
        return float("nan")
    if not (max(ps) >= 0.999 and min(ps) <= 0.001):
        return float("nan")
    up_idx = None
    for i, o in enumerate(outs):
        if o in {"up", "yes"} or o.startswith("up"):
            up_idx = i
            break
    if up_idx is None:
        return float("nan")
    return 1.0 if ps[up_idx] >= 0.999 else 0.0


def stake_tier(realized_capital: float) -> float:
    if realized_capital < 100.0:
        return 5.0
    k = math.floor(math.log(realized_capital / 50.0, 2.0))
    return min(100.0, 5.0 * (2.0 ** max(k, 0)))


def fee_per_share(px: float, fr: float = FEE_RATE) -> float:
    return fr * px * (1.0 - px)


def last_closed_price(bn: BinanceAnchor, ts_ms: int) -> tuple[float, int]:
    i = int(np.searchsorted(bn.close_times, int(ts_ms), side="right") - 1)
    if i < 0:
        return float("nan"), -1
    return float(bn.closes[i]), int(bn.close_times[i])


def bn_return(bn: BinanceAnchor, ts_ms: int, seconds: int) -> float:
    hi = int(np.searchsorted(bn.close_times, int(ts_ms), side="right") - 1)
    lo = int(np.searchsorted(bn.close_times, int(ts_ms) - seconds * 1000, side="right") - 1)
    if hi < 0 or lo < 0 or hi <= lo:
        return 0.0
    a, b = float(bn.closes[lo]), float(bn.closes[hi])
    if a <= 0 or b <= 0:
        return 0.0
    return math.log(b / a)


def deribit_iv_and_ts(der: DeribitAnchor, ts_ms: int, lookback_min: int = 30) -> tuple[float, int]:
    hi = int(np.searchsorted(der.ts, int(ts_ms), side="right"))
    if hi <= 0:
        return float("nan"), -1
    lo = int(np.searchsorted(der.ts, int(ts_ms) - lookback_min * 60_000, side="left"))
    vals = der.iv[lo:hi]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), -1
    return float(np.median(vals)), int(der.ts[hi - 1])


def top_level(g: pd.DataFrame, side: str) -> tuple[float, float]:
    x = g[g.side.str.lower() == side]
    if x.empty:
        return float("nan"), 0.0
    if side == "bid":
        r = x.loc[x.price.astype(float).idxmax()]
    else:
        r = x.loc[x.price.astype(float).idxmin()]
    return float(r.price), float(r["size"])


def build_books(book: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    snap_prob = snaps[["market_id", "ts_ms", "implied_probability"]].dropna().copy()
    snap_prob["market_id"] = snap_prob.market_id.astype(str)
    snap_prob = snap_prob.sort_values(["market_id", "ts_ms"])
    rows = []
    for (mid, ts), g in book.groupby(["market_id", "ts_ms"], sort=False):
        toks = []
        for tok, tg in g.groupby("token_id"):
            bid, bsz = top_level(tg, "bid")
            ask, asz = top_level(tg, "ask")
            if not (0 < bid < 1 or 0 < ask < 1):
                continue
            if 0 < bid < 1 and 0 < ask < 1 and bid <= ask:
                midpx = 0.5 * (bid + ask)
            elif 0 < ask < 1:
                midpx = ask
            else:
                midpx = bid
            toks.append((str(tok), bid, ask, bsz, asz, midpx))
        if len(toks) < 2:
            continue
        sg = snap_prob[snap_prob.market_id == str(mid)]
        if sg.empty:
            continue
        j = int(np.searchsorted(sg.ts_ms.to_numpy(np.int64), int(ts), side="right") - 1)
        if j < 0:
            continue
        sr = sg.iloc[j]
        if int(ts) - int(sr.ts_ms) > 90_000:
            continue
        p = float(sr.implied_probability)
        toks = sorted(toks, key=lambda z: abs(z[5] - p))
        up = toks[0]
        dn = min(toks[1:], key=lambda z: abs(z[5] - (1.0 - p)))
        rows.append({
            "market_id": str(mid), "ts_ms": int(ts),
            "up_bid": up[1], "up_ask": up[2], "up_bid_sz": up[3], "up_ask_sz": up[4],
            "dn_bid": dn[1], "dn_ask": dn[2], "dn_bid_sz": dn[3], "dn_ask_sz": dn[4],
            "mapping_ref_ts": int(sr.ts_ms), "mapping_prob": p,
        })
    return pd.DataFrame(rows).sort_values(["market_id", "ts_ms"]).reset_index(drop=True)


def load_solal(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    sp = Path(hf_hub_download(HF_REPO, SNAP_FILE, repo_type="dataset", revision=HF_REV, cache_dir=str(cache / "hf")))
    bp = Path(hf_hub_download(HF_REPO, BOOK_FILE, repo_type="dataset", revision=HF_REV, cache_dir=str(cache / "hf")))
    snaps = pd.read_csv(sp)
    book = pd.read_csv(bp)
    snaps["market_id"] = snaps.market_id.astype(str)
    book["market_id"] = book.market_id.astype(str)
    book["token_id"] = book.token_id.astype(str)
    snaps["ts_ms"] = to_ms(snaps.collection_timestamp_utc)
    book["ts_ms"] = to_ms(book.collection_timestamp_utc)
    snaps = snaps.dropna(subset=["ts_ms"]).copy(); snaps.ts_ms = snaps.ts_ms.astype(np.int64)
    book = book.dropna(subset=["ts_ms", "price", "size"]).copy(); book.ts_ms = book.ts_ms.astype(np.int64)
    book["price"] = pd.to_numeric(book.price, errors="coerce"); book["size"] = pd.to_numeric(book["size"], errors="coerce")
    book = book.dropna(subset=["price", "size"])
    manifest = {
        "repo": HF_REPO, "revision": HF_REV,
        "files": {
            SNAP_FILE: {"sha256": sha256_file(sp), "bytes": sp.stat().st_size},
            BOOK_FILE: {"sha256": sha256_file(bp), "bytes": bp.stat().st_size},
        },
    }
    return snaps, book, manifest


def market_meta(snaps: pd.DataFrame) -> pd.DataFrame:
    s = snaps.copy()
    s["start_ms"] = to_ms(s.window_start_utc)
    s["end_ms"] = to_ms(s.window_end_utc)
    if "window_start_unix" in s:
        miss = s.start_ms.isna() & pd.to_numeric(s.window_start_unix, errors="coerce").notna()
        s.loc[miss, "start_ms"] = pd.to_numeric(s.loc[miss, "window_start_unix"], errors="coerce").astype("Int64") * 1000
    s["settle_yes"] = s.apply(resolve_side, axis=1)
    base = s.dropna(subset=["start_ms", "end_ms"]).sort_values("ts_ms").groupby("market_id").agg(
        start_ms=("start_ms", "first"), end_ms=("end_ms", "first"), slug=("slug", "first")
    ).reset_index()
    res = s[np.isfinite(s.settle_yes)].sort_values("ts_ms").groupby("market_id").tail(1)[["market_id", "settle_yes"]]
    out = base.merge(res, on="market_id", how="left")
    out.start_ms = out.start_ms.astype(np.int64); out.end_ms = out.end_ms.astype(np.int64)
    return out


@dataclass
class SignalRow:
    market_id: str
    ts_ms: int
    close_ts: int
    settle_yes: int
    fade_yes: bool
    up_bid: float
    up_ask: float
    dn_bid: float
    dn_ask: float
    up_ask_sz: float
    dn_ask_sz: float
    p_rv: float
    p_iv: float
    rv: float
    iv: float
    rel_spot: float
    s2c: int
    edge: float
    max_feature_ts: int
    seq: np.ndarray
    static: np.ndarray


SEQ_NAMES = ["up_bid","up_ask","dn_bid","dn_ask","log_up_bid_sz","log_up_ask_sz","log_dn_bid_sz","log_dn_ask_sz","spot_log_rel_open","s2c_norm"]
STATIC_NAMES = ["edge","p_rv","p_iv","anchor_mean","anchor_gap","rv60","deribit_iv","iv_minus_rv","rel_spot","log_rel_spot","s2c_norm","up_bid","up_ask","dn_bid","dn_ask","up_spread","dn_spread","up_imb","dn_imb","ret1","ret5","ret15"]


def build_signals(obs: pd.DataFrame, meta: pd.DataFrame, bn: BinanceAnchor, der: DeribitAnchor, threshold: float = 0.03) -> list[SignalRow]:
    mm = meta.set_index("market_id").to_dict("index")
    result = []
    for mid, g0 in obs.groupby("market_id", sort=False):
        m = mm.get(str(mid))
        if not m or not np.isfinite(m.get("settle_yes", np.nan)):
            continue
        start_ms, end_ms = int(m["start_ms"]), int(m["end_ms"])
        open_px = bn.open_price(start_ms // 1000)
        if not (open_px > 0):
            continue
        g = g0.sort_values("ts_ms").copy()
        feats = []
        for r in g.itertuples(index=False):
            ts = int(r.ts_ms); s2c = int((end_ms - ts) // 1000)
            if s2c < 60 or s2c > 600:
                continue
            spot, bn_ts = last_closed_price(bn, ts)
            rv = bn.rv_annualized(ts, 60)
            div, der_ts = deribit_iv_and_ts(der, ts, 30)
            if not (spot > 0 and math.isfinite(rv) and math.isfinite(div) and 0.05 < rv < 3 and 0.05 < div < 3):
                continue
            rel = spot / open_px
            p_rv = digital_prob_up(rel, s2c, rv); p_iv = digital_prob_up(rel, s2c, div)
            if not (math.isfinite(p_rv) and math.isfinite(p_iv)):
                continue
            if not (0 < r.up_ask < 1 and 0 < r.dn_ask < 1 and 0 <= r.up_bid < 1 and 0 <= r.dn_bid < 1):
                continue
            lo, hi = min(p_rv,p_iv), max(p_rv,p_iv)
            eu = lo - float(r.up_ask) - fee_per_share(float(r.up_ask))
            ed = (1-hi) - float(r.dn_ask) - fee_per_share(float(r.dn_ask))
            fade_yes = eu >= threshold and eu >= ed
            fade_dn = ed >= threshold and ed > eu
            ret1, ret5, ret15 = bn_return(bn,ts,60), bn_return(bn,ts,300), bn_return(bn,ts,900)
            ub,ua,db,da = map(float,[r.up_bid,r.up_ask,r.dn_bid,r.dn_ask])
            ubs,uas,dbs,das = map(float,[r.up_bid_sz,r.up_ask_sz,r.dn_bid_sz,r.dn_ask_sz])
            static = np.asarray([
                max(eu,ed), p_rv,p_iv,0.5*(p_rv+p_iv),abs(p_rv-p_iv),rv,div,div-rv,rel,math.log(rel),s2c/900,
                ub,ua,db,da,max(ua-ub,0),max(da-db,0),(ubs-uas)/(ubs+uas+1e-9),(dbs-das)/(dbs+das+1e-9),ret1,ret5,ret15
            ], dtype=np.float32)
            seqrow = np.asarray([ub,ua,db,da,math.log1p(max(ubs,0)),math.log1p(max(uas,0)),math.log1p(max(dbs,0)),math.log1p(max(das,0)),math.log(rel),s2c/900],dtype=np.float32)
            maxfts=max(ts,int(r.mapping_ref_ts),bn_ts,der_ts)
            if maxfts > ts:
                raise RuntimeError(f"future feature detected market={mid} decision={ts} feature={maxfts}")
            feats.append((r,ts,s2c,fade_yes,fade_dn,p_rv,p_iv,rv,div,rel,max(eu,ed),maxfts,static,seqrow))
        if not feats:
            continue
        chosen_idx=None
        for k,x in enumerate(feats):
            if x[3] or x[4]:
                chosen_idx=k; break
        if chosen_idx is None:
            continue
        x=feats[chosen_idx]; r,ts,s2c,fy,fd,p_rv,p_iv,rv,div,rel,edge,maxfts,static,_=x
        seq=np.full((SEQ_LEN,len(SEQ_NAMES)),np.nan,dtype=np.float32)
        hist=[z[-1] for z in feats[:chosen_idx+1]][-SEQ_LEN:]
        seq[:len(hist)]=np.stack(hist)
        result.append(SignalRow(
            market_id=str(mid),ts_ms=ts,close_ts=end_ms,settle_yes=int(m["settle_yes"]),fade_yes=bool(fy),
            up_bid=float(r.up_bid),up_ask=float(r.up_ask),dn_bid=float(r.dn_bid),dn_ask=float(r.dn_ask),
            up_ask_sz=float(r.up_ask_sz),dn_ask_sz=float(r.dn_ask_sz),p_rv=p_rv,p_iv=p_iv,rv=rv,iv=div,rel_spot=rel,s2c=s2c,edge=edge,max_feature_ts=maxfts,seq=seq,static=static
        ))
    return sorted(result,key=lambda x:(x.close_ts,x.ts_ms))


def action_quote(s: SignalRow, yes: bool):
    return (s.up_ask,s.up_ask_sz) if yes else (s.dn_ask,s.dn_ask_sz)


def bankroll(signals: Iterable[SignalRow], action_fn, start_cash: float = 50.0, tag: str = "policy"):
    cash=float(start_cash); open_pos=[]; curve=[]; trades=[]; skipped={"cash":0,"depth":0,"edge":0,"window_cap":0}
    max_window_exposure=0.0
    def settle_until(ts):
        nonlocal cash,open_pos
        due=[p for p in open_pos if p["close_ts"]<=ts]
        keep=[p for p in open_pos if p["close_ts"]>ts]
        for p in sorted(due,key=lambda q:q["close_ts"]):
            cash += p["shares"] if p["won"] else 0.0
            curve.append({"ts_ms":p["close_ts"],"equity":cash+sum(q["cost"] for q in keep),"event":"settle"})
        open_pos=keep
    for s in signals:
        settle_until(s.ts_ms)
        capital=cash+sum(p["cost"] for p in open_pos)
        planned=stake_tier(capital)
        if cash + 1e-9 < planned:
            skipped["cash"]+=1; continue
        act=action_fn(s)
        if act is None:
            skipped["edge"]+=1; continue
        yes=bool(act)
        px,shares_depth=action_quote(s,yes)
        cps=px+fee_per_share(px)
        if not (0 < cps < 2):
            skipped["edge"]+=1; continue
        shares=planned/cps
        if shares_depth+1e-9 < shares:
            skipped["depth"]+=1; continue
        exposure=sum(p["cost"] for p in open_pos if p["market_id"]==s.market_id)
        if exposure+planned > 200.0+1e-9:
            skipped["window_cap"]+=1; continue
        cash-=planned
        won=(s.settle_yes==1)==yes
        p={"market_id":s.market_id,"ts_ms":s.ts_ms,"close_ts":s.close_ts,"yes":yes,"px":px,"shares":shares,"cost":planned,"won":won,"capital_before":capital,"tier":planned}
        open_pos.append(p); trades.append(p.copy()); max_window_exposure=max(max_window_exposure,exposure+planned)
        curve.append({"ts_ms":s.ts_ms,"equity":cash+sum(q["cost"] for q in open_pos),"event":"open"})
    settle_until(10**18)
    final=cash
    cdf=pd.DataFrame(curve).sort_values("ts_ms") if curve else pd.DataFrame(columns=["ts_ms","equity","event"])
    tdf=pd.DataFrame(trades)
    if not cdf.empty:
        peak=cdf.equity.cummax(); dd=(cdf.equity/peak-1.0); mdd=float(dd.min())
        first=int(cdf.ts_ms.min()); last=int(cdf.ts_ms.max()); days=max((last-first)/86_400_000,1e-9)
        cagr=(final/start_cash)**(365.0/days)-1 if final>0 else -1.0
    else:
        mdd=0.0;days=0.0;cagr=0.0
    return {
        "tag":tag,"start_cash":start_cash,"final_cash":final,"return":final/start_cash-1,"cagr":cagr,"days":days,"max_drawdown":mdd,
        "trades":len(tdf),"wins":int(tdf.won.sum()) if not tdf.empty else 0,"win_rate":float(tdf.won.mean()) if not tdf.empty else None,
        "max_window_exposure":max_window_exposure,"skipped":skipped,"stake_counts":tdf.tier.value_counts().sort_index().to_dict() if not tdf.empty else {}
    },tdf,cdf


@dataclass
class Scaler:
    sm: np.ndarray; ss: np.ndarray; qm: np.ndarray; qs: np.ndarray
    @classmethod
    def fit(cls, xs):
        s=np.stack([x.static for x in xs]); sm=s.mean(0); ss=s.std(0); ss[ss<1e-6]=1
        q=np.concatenate([x.seq for x in xs]); qm=np.nanmean(q,0); qs=np.nanstd(q,0); qm[~np.isfinite(qm)]=0; qs[(~np.isfinite(qs))|(qs<1e-6)]=1
        return cls(sm,ss,qm,qs)
    def one(self,x):
        s=((x.static-self.sm)/self.ss).astype(np.float32)
        valid=np.isfinite(x.seq).all(1).astype(np.float32)
        q=(x.seq-self.qm)/self.qs; q[~np.isfinite(q)]=0
        return q.astype(np.float32),valid,s

class DS(torch.utils.data.Dataset):
    def __init__(self,xs,sc): self.xs=xs;self.sc=sc
    def __len__(self): return len(self.xs)
    def __getitem__(self,i):
        x=self.xs[i];q,m,s=self.sc.one(x);anchor=0.5*(x.p_rv+x.p_iv)
        return torch.from_numpy(q),torch.from_numpy(m),torch.from_numpy(s),torch.tensor(anchor,dtype=torch.float32),torch.tensor(float(x.settle_yes),dtype=torch.float32)

class ResidualGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(len(SEQ_NAMES),32,batch_first=True);self.st=nn.Sequential(nn.Linear(len(STATIC_NAMES),32),nn.SiLU(),nn.LayerNorm(32));self.f=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Dropout(.1),nn.Linear(48,1))
    def forward(self,q,m,s):
        lens=m.sum(1).clamp(min=1).long();p=nn.utils.rnn.pack_padded_sequence(q,lens.cpu(),batch_first=True,enforce_sorted=False);_,h=self.gru(p);d=.75*torch.tanh(self.f(torch.cat([h[-1],self.st(s)],1)).squeeze(1));return d

def model_prob(anchor,d):
    a=anchor.clamp(1e-6,1-1e-6);return torch.sigmoid(torch.logit(a)+d)

def train_model(train,val,sc,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    m=ResidualGRU();opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=1e-3)
    dl=torch.utils.data.DataLoader(DS(train,sc),batch_size=64,shuffle=True);vl=torch.utils.data.DataLoader(DS(val,sc),batch_size=max(1,len(val)))
    best=None;bestv=1e9;stale=0
    for ep in range(100):
        m.train()
        for q,mask,s,a,y in dl:
            opt.zero_grad();d=m(q,mask,s);p=model_prob(a,d);loss=F.binary_cross_entropy(p,y)+.02*(d*d).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        m.eval()
        with torch.no_grad():
            q,mask,s,a,y=next(iter(vl));d=m(q,mask,s);p=model_prob(a,d);v=float(F.binary_cross_entropy(p,y)+.02*(d*d).mean())
        if v<bestv-1e-5: bestv=v;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=12:break
    m.load_state_dict(best);return m,bestv

def predict(models,xs,sc):
    ds=DS(xs,sc);dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False);allp=[]
    for m in models:
        ps=[];m.eval()
        with torch.no_grad():
            for q,mask,s,a,y in dl:
                ps.append(model_prob(a,m(q,mask,s)).numpy())
        allp.append(np.concatenate(ps))
    return np.mean(allp,0)


def model_action(s: SignalRow,p_yes:float):
    eu=p_yes-s.up_ask-fee_per_share(s.up_ask); ed=(1-p_yes)-s.dn_ask-fee_per_share(s.dn_ask)
    if max(eu,ed)<=0:return None
    return eu>=ed


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("extended_results"));ap.add_argument("--cache",type=Path,default=Path("extended_cache"));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    snaps,book,manifest=load_solal(args.cache)
    meta=market_meta(snaps);obs=build_books(book,snaps)
    cov={
        "snap_rows":len(snaps),"book_rows":len(book),"obs_rows":len(obs),"markets_meta":int(meta.market_id.nunique()),"resolved_markets":int(meta.settle_yes.notna().sum()),
        "snapshot_min_utc":datetime.fromtimestamp(int(snaps.ts_ms.min())/1000,tz=timezone.utc).isoformat(),"snapshot_max_utc":datetime.fromtimestamp(int(snaps.ts_ms.max())/1000,tz=timezone.utc).isoformat(),
        "book_min_utc":datetime.fromtimestamp(int(book.ts_ms.min())/1000,tz=timezone.utc).isoformat(),"book_max_utc":datetime.fromtimestamp(int(book.ts_ms.max())/1000,tz=timezone.utc).isoformat(),
    }
    print("COVERAGE",json.dumps(cov),flush=True)
    start=date.fromtimestamp(int(min(snaps.ts_ms.min(),book.ts_ms.min()))/1000)
    end=date.fromtimestamp(int(max(snaps.ts_ms.max(),book.ts_ms.max()))/1000)
    bn_df=download_binance_1m(start,end,args.cache/"binance");bn=BinanceAnchor.from_df(bn_df)
    inst=deribit_instruments();sel=select_deribit_instruments(inst,bn_df,start,end);tr=fetch_deribit_trades(sel,start,end,args.cache/"deribit_trades.parquet");der=DeribitAnchor.from_trades(tr,inst)
    signals=build_signals(obs,meta,bn,der,0.03)
    if len(signals)<30: raise RuntimeError(f"too few causal structural signals: {len(signals)}")
    assert all(x.max_feature_ts<=x.ts_ms for x in signals)
    sigdf=pd.DataFrame([{k:getattr(x,k) for k in ["market_id","ts_ms","close_ts","settle_yes","fade_yes","up_bid","up_ask","dn_bid","dn_ask","up_ask_sz","dn_ask_sz","p_rv","p_iv","rv","iv","rel_spot","s2c","edge","max_feature_ts"]} for x in signals])
    sigdf.to_csv(args.out/"signals.csv",index=False)
    structural,st,sc=bankroll(signals,lambda s:s.fade_yes,50,"structural_3c_full")
    st.to_csv(args.out/"structural_trades.csv",index=False);sc.to_csv(args.out/"structural_curve.csv",index=False)
    n=len(signals);i1=int(n*.60);i2=int(n*.80);train,val,hold=signals[:i1],signals[i1:i2],signals[i2:]
    if min(len(train),len(val),len(hold))<10: raise RuntimeError("split too small")
    scaler=Scaler.fit(train);models=[];seed_meta=[]
    for seed in SEEDS:
        print("TRAIN",seed,flush=True);m,bv=train_model(train,val,scaler,seed);models.append(m);seed_meta.append({"seed":seed,"best_val":bv})
    p=predict(models,hold,scaler);y=np.asarray([x.settle_yes for x in hold]);anchor=np.asarray([.5*(x.p_rv+x.p_iv) for x in hold])
    model_map={x.market_id:float(pp) for x,pp in zip(hold,p)}
    model,mt,mc=bankroll(hold,lambda s:model_action(s,model_map[s.market_id]),50,"residual_gru_holdout")
    baseline_hold,bht,bhc=bankroll(hold,lambda s:s.fade_yes,50,"structural_3c_same_holdout")
    mt.to_csv(args.out/"model_trades.csv",index=False);mc.to_csv(args.out/"model_curve.csv",index=False);bht.to_csv(args.out/"holdout_structural_trades.csv",index=False);bhc.to_csv(args.out/"holdout_structural_curve.csv",index=False)
    brier=lambda yy,pp:float(np.mean((np.asarray(yy)-np.asarray(pp))**2))
    ll=lambda yy,pp:float(-(np.asarray(yy)*np.log(np.clip(pp,1e-6,1-1e-6))+(1-np.asarray(yy))*np.log(np.clip(1-np.asarray(pp),1e-6,1-1e-6))).mean())
    summary={
        "dataset_manifest":manifest,"coverage":cov,"fee_rate":FEE_RATE,"signal_threshold":0.03,"no_lookahead":{
            "book":"same collector snapshot only","binance":"only closed 1m candles with close_time <= decision_ts","deribit":"only option trades timestamp <= decision_ts","target":"terminal outcome used only for loss/PnL after chronological split","assertion":"max_feature_ts <= decision_ts for every example"
        },
        "bankroll_rule":"start $50; planned trade $5 at <$100, doubles at each capital doubling; per-trade cap $100; per-market/window exposure cap $200; full displayed ask depth required; no leverage; unsettled cost basis carries no unrealized gain",
        "execution_model":"causal snapshot-cross at observed top ask with displayed L1 depth; unlike kinzikdza strict tape this dataset cannot prove post-decision persistence, so treat as robustness rather than strict-fill headline",
        "signals":len(signals),"split":{"train":len(train),"val":len(val),"holdout":len(hold),"holdout_start_utc":datetime.fromtimestamp(hold[0].close_ts/1000,tz=timezone.utc).isoformat()},
        "structural_full":structural,"structural_holdout":baseline_hold,"model_holdout":model,"model_seeds":seed_meta,
        "probability_holdout":{"brier_anchor":brier(y,anchor),"brier_model":brier(y,p),"logloss_anchor":ll(y,anchor),"logloss_model":ll(y,p)}
    }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    lines=["# Main Sequence extended Solal replay","",f"Pinned: `{HF_REPO}@{HF_REV}`",f"Coverage: {cov['snapshot_min_utc']} -> {cov['snapshot_max_utc']}",f"Causal structural signals: {len(signals)}","", "## Bankroll", "", "```json",json.dumps({"structural_full":structural,"structural_holdout":baseline_hold,"model_holdout":model,"probability_holdout":summary['probability_holdout']},indent=2,default=float),"```", "", "Execution caveat: snapshot-cross robustness replay, not strict post-decision tape fill."]
    (args.out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print((args.out/"SUMMARY.md").read_text(),flush=True)

if __name__=="__main__":main()
