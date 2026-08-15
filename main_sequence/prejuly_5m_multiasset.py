from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from scipy.special import ndtr

HF_REPO = "Alezanello/polymarket-arena-capture"
ASSETS = ("BTC", "ETH", "SOL", "XRP")
TRAIN_START = "2026-06-04"
TRAIN_END = "2026-06-24"
VAL_START = "2026-06-24"
VAL_END = "2026-07-01"
TEST_START = "2026-07-01"
TEST_END = "2026-07-15"
DECISION_S2C = 60
SEQ_LEN = 16
FEE_RATE = 0.07
EDGE_FLOOR = 0.03
SEEDS = (7, 19, 42, 73, 101)
YEAR_SECONDS = 365.0 * 24 * 3600
EPS = 1e-8

SEQ_NAMES = [
    "up_bid", "up_ask", "dn_bid", "dn_ask",
    "log_up_bid_sz", "log_up_ask_sz", "log_dn_bid_sz", "log_dn_ask_sz",
    "up_mid", "dn_mid", "complement_gap", "spread_mean",
]
STATIC_NAMES = [
    "p_rv", "pm_up_mid", "pm_residual", "rel_spot", "log_rel_spot",
    "ret_1m", "ret_5m", "rv_15m", "rv_60m", "vol_expansion",
    "up_bid", "up_ask", "dn_bid", "dn_ask", "up_spread", "dn_spread",
    "up_mid", "dn_mid", "up_imb", "dn_imb", "complement_gap", "s2c_norm",
    "asset_BTC", "asset_ETH", "asset_SOL", "asset_XRP",
]


def utc_ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fee_per_share(px: float) -> float:
    return FEE_RATE * px * (1.0 - px)


def safe_mid(bid: float, ask: float) -> float:
    if 0 <= bid <= ask <= 1 and ask > 0 and bid < 1:
        return 0.5 * (bid + ask)
    return float("nan")


def digital_prob_up(rel_spot: float, seconds: float, annual_sigma: float) -> float:
    if not (rel_spot > 0 and seconds > 0 and annual_sigma > 0 and math.isfinite(annual_sigma)):
        return float("nan")
    t = seconds / YEAR_SECONDS
    den = annual_sigma * math.sqrt(t)
    if den <= 0:
        return float(rel_spot >= 1.0)
    d2 = (math.log(rel_spot) - 0.5 * annual_sigma * annual_sigma * t) / den
    return float(ndtr(d2))


def list_selected_files(revision: str, table: str, start: str, end: str, cache: Path) -> list[Path]:
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset", revision=revision)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    selected: list[str] = []
    root = f"{table}.parquet"
    if s <= date(2026, 6, 15) and e > date(2026, 6, 4) and root in files:
        selected.append(root)
    for f in files:
        if not (f.startswith(f"daily/{table}/") and f.endswith(".parquet")):
            continue
        m = re.search(r"(20\d\d-\d\d-\d\d)", f)
        if not m:
            continue
        d = date.fromisoformat(m.group(1))
        if s <= d < e:
            selected.append(f)
    if not selected:
        raise RuntimeError(f"no {table} files for {start}..{end} at {revision}")
    return [Path(hf_hub_download(HF_REPO, f, repo_type="dataset", revision=revision,
                                 cache_dir=str(cache / "hf"))) for f in sorted(set(selected))]


def sql_files(paths: Iterable[Path]) -> str:
    return "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def load_book_sequences(paths: list[Path], start: str, end: str) -> pd.DataFrame:
    lo, hi = utc_ms(start), utc_ms(end)
    con = duckdb.connect()
    q = f"""
    WITH src AS (
      SELECT ts_ms, asset_id, CAST(best_bid AS DOUBLE) AS best_bid, CAST(best_ask AS DOUBLE) AS best_ask,
             CAST(bid_sz AS DOUBLE) AS bid_sz, CAST(ask_sz AS DOUBLE) AS ask_sz,
             upper(asset) AS asset, outcome, slug, cond,
             CAST(win_start AS BIGINT) AS win_start, CAST(end_ts AS BIGINT) AS end_ts
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset) IN ('BTC','ETH','SOL','XRP')
        AND lower(slug) LIKE '%-updown-5m-%'
        AND end_ts * 1000 >= {lo} AND end_ts * 1000 < {hi}
        AND ts_ms <= end_ts * 1000 - {DECISION_S2C * 1000}
        AND ts_ms >= end_ts * 1000 - {(DECISION_S2C + 90) * 1000}
        AND best_ask > 0 AND best_ask < 1 AND best_bid >= 0 AND best_bid < 1
    ), ranked AS (
      SELECT *, row_number() OVER (PARTITION BY slug, lower(outcome) ORDER BY ts_ms DESC) AS rn
      FROM src
    )
    SELECT * FROM ranked WHERE rn <= {SEQ_LEN}
    ORDER BY end_ts, slug, lower(outcome), rn DESC
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def load_price_bars(paths: list[Path], start: str, end: str) -> pd.DataFrame:
    lo, hi = utc_ms(start) - 75 * 60_000, utc_ms(end) + 15_000
    con = duckdb.connect()
    q = f"""
    WITH p AS (
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms, lower(src) AS src, upper(asset) AS asset, CAST(value AS DOUBLE) AS value
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset) IN ('BTC','ETH','SOL','XRP') AND ts_ms >= {lo} AND ts_ms <= {hi}
        AND lower(src) IN ('chainlink','binance') AND value > 0
    ), b AS (
      SELECT asset, src, CAST(floor(ts_ms / 5000) * 5000 AS BIGINT) AS bar_ms,
             arg_max(value, ts_ms) AS value, max(ts_ms) AS source_ts
      FROM p GROUP BY asset, src, bar_ms
    )
    SELECT * FROM b ORDER BY asset, src, bar_ms
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def load_trades(paths: list[Path], start: str, end: str) -> pd.DataFrame:
    lo, hi = utc_ms(start), utc_ms(end) + 5000
    con = duckdb.connect()
    q = f"""
      SELECT CAST(ts_ms AS BIGINT) AS ts_ms, CAST(asset_id AS VARCHAR) AS asset_id,
             CAST(price AS DOUBLE) AS price, CAST(size AS DOUBLE) AS size,
             upper(asset) AS asset, outcome, slug
      FROM read_parquet({sql_files(paths)}, union_by_name=true)
      WHERE upper(asset) IN ('BTC','ETH','SOL','XRP')
        AND lower(slug) LIKE '%-updown-5m-%'
        AND ts_ms >= {lo} AND ts_ms < {hi}
        AND price > 0 AND price < 1 AND size > 0
      ORDER BY asset_id, ts_ms
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


@dataclass
class PriceSeries:
    ts: np.ndarray
    px: np.ndarray
    minute_ts: np.ndarray
    minute_px: np.ndarray
    source: str

    def at(self, t_ms: int, tolerance_ms: int = 20_000) -> tuple[float, int]:
        i = int(np.searchsorted(self.ts, t_ms, side="right") - 1)
        if i < 0 or t_ms - int(self.ts[i]) > tolerance_ms:
            return float("nan"), -1
        return float(self.px[i]), int(self.ts[i])

    def minute_at(self, t_ms: int) -> tuple[float, int]:
        i = int(np.searchsorted(self.minute_ts, t_ms, side="right") - 1)
        if i < 0:
            return float("nan"), -1
        return float(self.minute_px[i]), int(self.minute_ts[i])

    def ret(self, t_ms: int, seconds: int) -> float:
        a, _ = self.minute_at(t_ms - seconds * 1000)
        b, _ = self.minute_at(t_ms)
        return math.log(b / a) if a > 0 and b > 0 else 0.0

    def rv(self, t_ms: int, minutes: int, min_obs: int) -> float:
        hi = int(np.searchsorted(self.minute_ts, t_ms, side="right"))
        lo = int(np.searchsorted(self.minute_ts, t_ms - minutes * 60_000, side="left"))
        vals = self.minute_px[lo:hi]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < min_obs + 1:
            return float("nan")
        r = np.diff(np.log(vals))
        if r.size < min_obs:
            return float("nan")
        return float(np.std(r, ddof=1)) * math.sqrt(365 * 24 * 60)


def build_price_series(df: pd.DataFrame) -> dict[str, PriceSeries]:
    out: dict[str, PriceSeries] = {}
    for asset in ASSETS:
        ad = df[df.asset == asset]
        if ad.empty:
            continue
        counts = ad.groupby("src").size().to_dict()
        src = "chainlink" if counts.get("chainlink", 0) >= 100 else "binance"
        x = ad[ad.src == src].sort_values("bar_ms").drop_duplicates("bar_ms", keep="last")
        ts = x.bar_ms.to_numpy(np.int64)
        px = x.value.to_numpy(float)
        mins = (ts // 60_000) * 60_000
        tmp = pd.DataFrame({"m": mins, "ts": ts, "px": px}).groupby("m", sort=True).tail(1)
        out[asset] = PriceSeries(ts, px, tmp.m.to_numpy(np.int64), tmp.px.to_numpy(float), src)
    missing = [a for a in ASSETS if a not in out]
    if missing:
        raise RuntimeError(f"missing price series: {missing}")
    return out


@dataclass
class Example:
    slug: str
    asset: str
    cond: str
    up_asset_id: str
    dn_asset_id: str
    decision_ts: int
    win_start: int
    end_ts: int
    settle_up: float
    anchor: float
    pm_mid: float
    up_ask: float
    dn_ask: float
    seq: np.ndarray
    static: np.ndarray
    label_source: str


def normalize_outcome(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("up") or s == "yes": return "up"
    if s.startswith("down") or s == "no": return "down"
    return s


def make_examples(book: pd.DataFrame, prices: dict[str, PriceSeries]) -> list[Example]:
    examples: list[Example] = []
    for slug, g in book.groupby("slug", sort=False):
        asset = str(g.asset.iloc[0]).upper()
        ps = prices.get(asset)
        if ps is None:
            continue
        up = g[g.outcome.map(normalize_outcome) == "up"].sort_values("rn", ascending=False)
        dn = g[g.outcome.map(normalize_outcome) == "down"].sort_values("rn", ascending=False)
        if up.empty or dn.empty:
            continue
        n = min(len(up), len(dn), SEQ_LEN)
        up = up.tail(n).reset_index(drop=True); dn = dn.tail(n).reset_index(drop=True)
        ul, dl = up.iloc[-1], dn.iloc[-1]
        end_ts = int(ul.end_ts); win_start = int(ul.win_start)
        target = end_ts * 1000 - DECISION_S2C * 1000
        if target - int(ul.ts_ms) > 7000 or target - int(dl.ts_ms) > 7000:
            continue
        decision_ts = max(int(ul.ts_ms), int(dl.ts_ms))
        open_px, open_ts = ps.at(win_start * 1000, tolerance_ms=30_000)
        close_px, close_ts = ps.at(end_ts * 1000, tolerance_ms=30_000)
        spot, spot_ts = ps.at(decision_ts, tolerance_ms=20_000)
        if not (open_px > 0 and close_px > 0 and spot > 0):
            continue
        lr_close = math.log(close_px / open_px)
        if abs(lr_close) < 1e-8:
            continue
        settle = 1.0 if lr_close > 0 else 0.0
        rv60 = ps.rv(decision_ts, 60, min_obs=30)
        rv15 = ps.rv(decision_ts, 15, min_obs=8)
        if not (math.isfinite(rv60) and 0.02 < rv60 < 5.0):
            continue
        if not (math.isfinite(rv15) and 0.02 < rv15 < 5.0):
            rv15 = rv60
        rel = spot / open_px
        p_rv = digital_prob_up(rel, max((end_ts * 1000 - decision_ts) / 1000.0, 1.0), rv60)
        if not math.isfinite(p_rv):
            continue
        ub, ua, db, da = map(float, [ul.best_bid, ul.best_ask, dl.best_bid, dl.best_ask])
        if not (0 <= ub <= ua <= 1 and 0 <= db <= da <= 1 and 0 < ua < 1 and 0 < da < 1):
            continue
        um, dm = safe_mid(ub, ua), safe_mid(db, da)
        if not (math.isfinite(um) and math.isfinite(dm)):
            continue
        ubs, uas, dbs, das = map(lambda z: max(float(z), 0.0), [ul.bid_sz, ul.ask_sz, dl.bid_sz, dl.ask_sz])
        uimb = (ubs - uas) / (ubs + uas + EPS); dimb = (dbs - das) / (dbs + das + EPS)
        pm_mid = um / max(um + dm, EPS)
        static = np.asarray([
            p_rv, pm_mid, pm_mid - p_rv, rel, math.log(rel),
            ps.ret(decision_ts, 60), ps.ret(decision_ts, 300), rv15, rv60, rv15 / max(rv60, 1e-6),
            ub, ua, db, da, ua-ub, da-db, um, dm, uimb, dimb, um+dm-1.0,
            DECISION_S2C / 300.0,
            float(asset == "BTC"), float(asset == "ETH"), float(asset == "SOL"), float(asset == "XRP"),
        ], dtype=np.float32)
        seq = np.full((SEQ_LEN, len(SEQ_NAMES)), np.nan, dtype=np.float32)
        rows = []
        for i in range(n):
            u, d = up.iloc[i], dn.iloc[i]
            ub_i, ua_i, db_i, da_i = map(float, [u.best_bid, u.best_ask, d.best_bid, d.best_ask])
            um_i, dm_i = safe_mid(ub_i, ua_i), safe_mid(db_i, da_i)
            if not all(math.isfinite(z) for z in [um_i, dm_i]):
                continue
            ubs_i, uas_i, dbs_i, das_i = map(lambda z: max(float(z),0.0), [u.bid_sz,u.ask_sz,d.bid_sz,d.ask_sz])
            rows.append([ub_i, ua_i, db_i, da_i,
                         math.log1p(ubs_i), math.log1p(uas_i), math.log1p(dbs_i), math.log1p(das_i),
                         um_i, dm_i, um_i+dm_i-1.0, 0.5*((ua_i-ub_i)+(da_i-db_i))])
        if len(rows) < 4:
            continue
        seq[:len(rows)] = np.asarray(rows, dtype=np.float32)
        max_feature_ts = max(int(ul.ts_ms), int(dl.ts_ms), int(spot_ts))
        if max_feature_ts > decision_ts:
            raise RuntimeError(f"future feature {slug}: {max_feature_ts}>{decision_ts}")
        examples.append(Example(str(slug), asset, str(ul.cond), str(ul.asset_id), str(dl.asset_id),
                                decision_ts, win_start, end_ts, settle, p_rv, pm_mid, ua, da, seq, static,
                                f"{ps.source}_direction:{open_ts}->{close_ts}"))
    examples.sort(key=lambda x: (x.end_ts, x.asset, x.slug))
    return examples


@dataclass
class Scaler:
    sm: np.ndarray; ss: np.ndarray; qm: np.ndarray; qs: np.ndarray

    @classmethod
    def fit(cls, xs: list[Example]) -> "Scaler":
        s = np.stack([x.static for x in xs]); sm=s.mean(0); ss=s.std(0); ss[ss<1e-6]=1.0
        q = np.concatenate([x.seq for x in xs]); qm=np.nanmean(q,0); qs=np.nanstd(q,0)
        qm[~np.isfinite(qm)] = 0.0; qs[(~np.isfinite(qs)) | (qs<1e-6)] = 1.0
        return cls(sm,ss,qm,qs)

    def one(self, x: Example):
        s=((x.static-self.sm)/self.ss).astype(np.float32)
        valid=np.isfinite(x.seq).all(1).astype(np.float32)
        q=(x.seq-self.qm)/self.qs; q[~np.isfinite(q)]=0.0
        return q.astype(np.float32),valid,s


class DS(torch.utils.data.Dataset):
    def __init__(self,xs,sc): self.xs=xs; self.sc=sc
    def __len__(self): return len(self.xs)
    def __getitem__(self,i):
        x=self.xs[i]; q,m,s=self.sc.one(x)
        return torch.from_numpy(q),torch.from_numpy(m),torch.from_numpy(s),torch.tensor(x.anchor,dtype=torch.float32),torch.tensor(x.settle_up,dtype=torch.float32)


class ResidualGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(len(SEQ_NAMES),32,batch_first=True)
        self.st=nn.Sequential(nn.Linear(len(STATIC_NAMES),32),nn.SiLU(),nn.LayerNorm(32))
        self.f=nn.Sequential(nn.Linear(64,48),nn.SiLU(),nn.Dropout(.10),nn.Linear(48,1))
    def forward(self,q,m,s):
        lens=m.sum(1).clamp(min=1).long()
        p=nn.utils.rnn.pack_padded_sequence(q,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(p)
        return 0.75*torch.tanh(self.f(torch.cat([h[-1],self.st(s)],1)).squeeze(1))


def model_prob(anchor,delta):
    a=anchor.clamp(1e-6,1-1e-6)
    return torch.sigmoid(torch.logit(a)+delta)


def train_model(train,val,sc,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m=ResidualGRU(); opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=1e-3)
    dl=torch.utils.data.DataLoader(DS(train,sc),batch_size=128,shuffle=True)
    vl=torch.utils.data.DataLoader(DS(val,sc),batch_size=max(1,min(4096,len(val))),shuffle=False)
    best=None; bestv=float("inf"); stale=0; epochs=0
    for ep in range(100):
        m.train()
        for q,mask,s,a,y in dl:
            opt.zero_grad(); d=m(q,mask,s); p=model_prob(a,d)
            loss=F.binary_cross_entropy(p,y)+0.02*(d*d).mean(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval(); vals=[]
        with torch.no_grad():
            for q,mask,s,a,y in vl:
                d=m(q,mask,s); p=model_prob(a,d); vals.append(float(F.binary_cross_entropy(p,y)+0.02*(d*d).mean()))
        v=float(np.mean(vals)); epochs=ep+1
        if v < bestv-1e-5:
            bestv=v; best={k:z.detach().cpu().clone() for k,z in m.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=12: break
    if best is None: raise RuntimeError("training produced no checkpoint")
    m.load_state_dict(best); return m,bestv,epochs


def predict(models,xs,sc):
    dl=torch.utils.data.DataLoader(DS(xs,sc),batch_size=512,shuffle=False); allp=[]
    for m in models:
        ps=[]; m.eval()
        with torch.no_grad():
            for q,mask,s,a,y in dl: ps.append(model_prob(a,m(q,mask,s)).cpu().numpy())
        allp.append(np.concatenate(ps))
    return np.mean(allp,axis=0)


def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def logloss(y,p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())


def cluster_bootstrap(values: pd.DataFrame, reps: int = 4000, seed: int = 20260815):
    if values.empty: return [None,None]
    day = pd.to_datetime(values.decision_ts,unit="ms",utc=True).dt.date.astype(str)
    a=values.assign(day=day).groupby("day").reward.mean().to_numpy(float)
    if len(a)==1: return [float(a[0]),float(a[0])]
    rng=np.random.default_rng(seed); means=np.empty(reps)
    for i in range(reps): means[i]=rng.choice(a,size=len(a),replace=True).mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def grade_policy(examples: list[Example], probs: np.ndarray, trades: pd.DataFrame | None) -> tuple[dict,pd.DataFrame]:
    trade_map={}
    if trades is not None and not trades.empty:
        for aid,g in trades.groupby("asset_id",sort=False):
            trade_map[str(aid)] = (g.ts_ms.to_numpy(np.int64),g.price.to_numpy(float))
    rows=[]
    for x,p in zip(examples,probs):
        up_cost=x.up_ask+fee_per_share(x.up_ask); dn_cost=x.dn_ask+fee_per_share(x.dn_ask)
        eu=float(p)-up_cost; ed=(1-float(p))-dn_cost
        if max(eu,ed) < EDGE_FLOOR: continue
        yes=eu>=ed; ask=x.up_ask if yes else x.dn_ask; cost=up_cost if yes else dn_cost
        won=(x.settle_up>=0.5)==yes; reward=(1.0 if won else 0.0)-cost
        aid=x.up_asset_id if yes else x.dn_asset_id
        strict=False
        if aid in trade_map:
            ts,px=trade_map[aid]; lo=int(np.searchsorted(ts,x.decision_ts,side="left")); hi=int(np.searchsorted(ts,x.decision_ts+1500,side="right"))
            strict=bool(hi>lo and np.any(px[lo:hi] >= ask-1e-9))
        rows.append({"slug":x.slug,"asset":x.asset,"decision_ts":x.decision_ts,"side":"Up" if yes else "Down",
                     "ask":ask,"cost":cost,"pred_p_up":float(p),"pred_edge":max(eu,ed),"won":won,"reward":reward,"strict_tape_fill":strict})
    df=pd.DataFrame(rows)
    def summarize(z):
        if z.empty:return {"trades":0,"edge_share":None,"roi":None,"win_rate":None,"ci95_day":[None,None]}
        return {"trades":int(len(z)),"edge_share":float(z.reward.mean()),"roi":float(z.reward.sum()/z.cost.sum()),
                "win_rate":float(z.won.mean()),"ci95_day":cluster_bootstrap(z)}
    out={"cross_all":summarize(df),"strict_tape":summarize(df[df.strict_tape_fill]) if not df.empty else summarize(df)}
    out["by_asset"]={a:summarize(df[df.asset==a]) for a in ASSETS}
    return out,df


def phase_data(phase: str):
    if phase=="train": return TRAIN_START,VAL_END
    if phase=="test": return TEST_START,TEST_END
    raise ValueError(phase)


def download_phase(revision: str, phase: str, cache: Path, need_trades: bool):
    start,end=phase_data(phase)
    book=list_selected_files(revision,"cap_book",start,end,cache)
    price_start="2026-06-30" if phase=="test" else start
    prices=list_selected_files(revision,"cap_prices",price_start,end,cache)
    trades=list_selected_files(revision,"cap_trades",start,end,cache) if need_trades else []
    manifest={"repo":HF_REPO,"revision":revision,"phase":phase,"start":start,"end":end,"files":{}}
    for k,paths in [("cap_book",book),("cap_prices",prices),("cap_trades",trades)]:
        manifest["files"][k]=[{"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in paths]
    return book,prices,trades,manifest


def save_examples_index(xs: list[Example], path: Path):
    pd.DataFrame([{"slug":x.slug,"asset":x.asset,"decision_ts":x.decision_ts,"end_ts":x.end_ts,
                   "settle_up":x.settle_up,"anchor":x.anchor,"pm_mid":x.pm_mid,"up_ask":x.up_ask,"dn_ask":x.dn_ask,
                   "label_source":x.label_source} for x in xs]).to_csv(path,index=False)


def train_phase(args):
    revision=HfApi().dataset_info(HF_REPO).sha
    book_paths,price_paths,_,manifest=download_phase(revision,"train",args.cache,False)
    book=load_book_sequences(book_paths,TRAIN_START,VAL_END)
    prices=build_price_series(load_price_bars(price_paths,TRAIN_START,VAL_END))
    ex=make_examples(book,prices)
    tr=[x for x in ex if utc_ms(TRAIN_START)<=x.end_ts*1000<utc_ms(TRAIN_END)]
    va=[x for x in ex if utc_ms(VAL_START)<=x.end_ts*1000<utc_ms(VAL_END)]
    if len(tr)<500 or len(va)<100: raise RuntimeError(f"too few examples train={len(tr)} val={len(va)}")
    scaler=Scaler.fit(tr); models=[]; metas=[]; states={}
    for seed in SEEDS:
        print("TRAIN",seed,flush=True); m,bv,ep=train_model(tr,va,scaler,seed)
        models.append(m); states[str(seed)]=m.state_dict(); metas.append({"seed":seed,"best_val":bv,"epochs":ep})
    pv=predict(models,va,scaler); yv=np.asarray([x.settle_up for x in va]); av=np.asarray([x.anchor for x in va]); pm=np.asarray([x.pm_mid for x in va])
    summary={
        "name":"Main Sequence pre-July 5m multiasset temporal test / train artifact",
        "dataset":manifest,"assets":ASSETS,
        "partition":{"train":[TRAIN_START,TRAIN_END],"validation":[VAL_START,VAL_END],"sealed_test":[TEST_START,TEST_END]},
        "contract":{"decision_s2c":DECISION_S2C,"edge_floor":EDGE_FLOOR,"fee_rate":FEE_RATE,"seeds":SEEDS,"seq_len":SEQ_LEN,
                    "architecture":"causal 5m multiasset RV finance anchor + bounded +/-0.75 logit GRU residual",
                    "test_isolation":"test files are not downloaded in train phase; scaler, early stopping and weights are frozen before test phase"},
        "counts":{"train":len(tr),"validation":len(va),"by_asset_train":pd.Series([x.asset for x in tr]).value_counts().to_dict(),
                  "by_asset_validation":pd.Series([x.asset for x in va]).value_counts().to_dict()},
        "validation_probability":{"brier_model":brier(yv,pv),"brier_anchor_rv":brier(yv,av),"brier_pm_mid":brier(yv,pm),
                                  "logloss_model":logloss(yv,pv),"logloss_anchor_rv":logloss(yv,av),"logloss_pm_mid":logloss(yv,pm)},
        "model_seeds":metas,
    }
    args.out.mkdir(parents=True,exist_ok=True)
    save_examples_index(tr,args.out/"train_examples.csv"); save_examples_index(va,args.out/"validation_examples.csv")
    np.savez(args.out/"scaler.npz",sm=scaler.sm,ss=scaler.ss,qm=scaler.qm,qs=scaler.qs,
             static_names=np.asarray(STATIC_NAMES),seq_names=np.asarray(SEQ_NAMES))
    torch.save({"states":states,"seeds":SEEDS,"static_names":STATIC_NAMES,"seq_names":SEQ_NAMES,"dataset_revision":revision},args.out/"prejuly_5m_model.pt")
    (args.out/"train_summary.json").write_text(json.dumps(summary,indent=2,default=float))
    (args.out/"FROZEN_CONTRACT.json").write_text(json.dumps({"dataset_revision":revision,"train_end":TRAIN_END,"validation_end":VAL_END,
        "test_start":TEST_START,"test_end":TEST_END,"decision_s2c":DECISION_S2C,"edge_floor":EDGE_FLOOR,"fee_rate":FEE_RATE,
        "static_names":STATIC_NAMES,"seq_names":SEQ_NAMES,"seeds":SEEDS},indent=2))
    print(json.dumps(summary,indent=2,default=float),flush=True)


def test_phase(args):
    frozen=json.loads((args.model_dir/"FROZEN_CONTRACT.json").read_text()); revision=frozen["dataset_revision"]
    assert frozen["validation_end"]==TEST_START and frozen["test_start"]==TEST_START and frozen["test_end"]==TEST_END
    book_paths,price_paths,trade_paths,manifest=download_phase(revision,"test",args.cache,True)
    book=load_book_sequences(book_paths,TEST_START,TEST_END)
    prices=build_price_series(load_price_bars(price_paths,TEST_START,TEST_END))
    test=[x for x in make_examples(book,prices) if utc_ms(TEST_START)<=x.end_ts*1000<utc_ms(TEST_END)]
    if len(test)<500: raise RuntimeError(f"too few sealed test examples {len(test)}")
    scz=np.load(args.model_dir/"scaler.npz",allow_pickle=False); scaler=Scaler(scz["sm"],scz["ss"],scz["qm"],scz["qs"])
    ck=torch.load(args.model_dir/"prejuly_5m_model.pt",map_location="cpu",weights_only=False)
    if ck["dataset_revision"]!=revision: raise RuntimeError("dataset revision mismatch")
    models=[]
    for seed in ck["seeds"]:
        m=ResidualGRU();m.load_state_dict(ck["states"][str(seed)]);models.append(m)
    p=predict(models,test,scaler); y=np.asarray([x.settle_up for x in test]); a=np.asarray([x.anchor for x in test]); pm=np.asarray([x.pm_mid for x in test])
    trades=load_trades(trade_paths,TEST_START,TEST_END); policy,tdf=grade_policy(test,p,trades)
    by_asset={}
    for asset in ASSETS:
        idx=np.asarray([x.asset==asset for x in test])
        if idx.sum(): by_asset[asset]={"n":int(idx.sum()),"brier_model":brier(y[idx],p[idx]),"brier_anchor_rv":brier(y[idx],a[idx]),
                                      "brier_pm_mid":brier(y[idx],pm[idx]),"logloss_model":logloss(y[idx],p[idx])}
    ix=pd.DataFrame({"end_ts":[x.end_ts for x in test],"asset":[x.asset for x in test]});ix["day"]=pd.to_datetime(ix.end_ts,unit="s",utc=True).dt.strftime("%Y-%m-%d")
    daily=ix.groupby(["day","asset"]).size().unstack(fill_value=0).reindex(columns=ASSETS,fill_value=0)
    expected_days=pd.date_range(TEST_START,pd.Timestamp(TEST_END)-pd.Timedelta(days=1),freq="D",tz="UTC").strftime("%Y-%m-%d")
    missing_days=[d for d in expected_days if d not in daily.index]
    summary={
      "name":"Main Sequence 5m multiasset sealed July test","dataset":manifest,"frozen_contract":frozen,
      "counts":{"test":len(test),"by_asset":pd.Series([x.asset for x in test]).value_counts().to_dict(),"missing_calendar_days":missing_days},
      "probability":{"brier_model":brier(y,p),"brier_anchor_rv":brier(y,a),"brier_pm_mid":brier(y,pm),"logloss_model":logloss(y,p),
                     "logloss_anchor_rv":logloss(y,a),"logloss_pm_mid":logloss(y,pm)},
      "by_asset":by_asset,"policy":policy,
      "no_lookahead":"book snapshots and underlying prices are <= decision_ts; labels use only end-of-window reference direction; train phase never downloads July test files",
      "execution":"cross_all assumes taking displayed ask; strict_tape additionally requires an observed same-token trade at or through the decision ask within 1.5s",
    }
    args.out.mkdir(parents=True,exist_ok=True); save_examples_index(test,args.out/"test_examples.csv"); tdf.to_csv(args.out/"policy_trades.csv",index=False); daily.to_csv(args.out/"daily_counts.csv")
    pd.DataFrame({"slug":[x.slug for x in test],"asset":[x.asset for x in test],"decision_ts":[x.decision_ts for x in test],
                  "settle_up":y,"anchor_rv":a,"pm_mid":pm,"model_p":p}).to_csv(args.out/"test_probabilities.csv",index=False)
    (args.out/"test_summary.json").write_text(json.dumps(summary,indent=2,default=float))
    lines=["# Main Sequence 5m multiasset — sealed July test","",f"Frozen train: {TRAIN_START}..{TRAIN_END}; validation: {VAL_START}..{VAL_END}; sealed test: {TEST_START}..{TEST_END} UTC.","",
           f"Test examples: {len(test)}; missing calendar days: {missing_days}","","## Probability","","```json",json.dumps(summary["probability"],indent=2),"```","","## Policy","","```json",json.dumps(policy,indent=2,default=float),"```","",
           "No July sample was available to scaler fitting, early stopping, weights, or the fixed 3c policy floor."]
    (args.out/"SUMMARY.md").write_text("\n".join(lines)+"\n"); print((args.out/"SUMMARY.md").read_text(),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["train","test"]);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--cache",type=Path,required=True);ap.add_argument("--model-dir",type=Path)
    args=ap.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    if args.phase=="train": train_phase(args)
    else:
        if args.model_dir is None: raise SystemExit("--model-dir required for test")
        test_phase(args)

if __name__=="__main__": main()
