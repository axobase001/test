from __future__ import annotations

import argparse, io, json, math, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from main_sequence import final_recent_replay as base
from main_sequence import original_93d_tape_replay as stats_base
from main_sequence.original_93d_anchor_freeze import fetch_deribit_trades_parallel
from pm_structural import recalc as original

START='2026-05-15'; END='2026-08-15'
SYMBOL='ETHUSDT'; ASSET='ETH'
EDGE_FLOOR=0.05; RAW_GAP_FLOOR=0.10; ASK_FLOOR=0.20; EXIT_BAND=0.01
TICKET=5.0; VOLUME_MULT=2.0
MIN_S2C=60; MAX_S2C=600
GAMMA='https://gamma-api.polymarket.com'; BINANCE_REST='https://data-api.binance.vision/api/v3/klines'

@dataclass(frozen=True)
class Market:
    slug:str; start:int; condition_id:str; label_up:float
    fee_enabled:bool; fee_type:str; fee_rate:float|None; fee_exponent:float|None; fee_source:str
    @property
    def close(self): return self.start+900

def ts(s): return int(pd.Timestamp(s,tz='UTC').timestamp())

def parse_market(raw,start):
    try:
        outcomes=base.jsonish(raw['outcomes'],[]); prices=base.jsonish(raw.get('outcomePrices') or '[]',[])
        oi={str(x).strip().lower():i for i,x in enumerate(outcomes)}
        if 'up' not in oi or 'down' not in oi: return None
        ui,di=oi['up'],oi['down']; pp=[float(x) for x in prices]
        if len(pp)<=max(ui,di) or max(pp)<.99: return None
        cid=str(raw.get('conditionId') or '')
        if not cid: return None
        enabled=base.boolish(raw.get('feesEnabled')); fs=base.jsonish(raw.get('feeSchedule'),{})
        rate=fs.get('rate'); expo=fs.get('exponent')
        try: rate=float(rate) if rate is not None else None
        except: rate=None
        try: expo=float(expo) if expo is not None else None
        except: expo=None
        if enabled is None:
            enabled=start>=base.LEGACY_FEE_START; source='dated_fallback'
        else: source='gamma_feesEnabled'
        return Market(str(raw['slug']),int(start),cid,1.0 if pp[ui]>pp[di] else 0.0,bool(enabled),str(raw.get('feeType') or ''),rate,expo,source)
    except Exception: return None

def fetch_markets_hour(hour):
    sess=requests.Session(); sess.headers.update({'User-Agent':'main-sequence-eth15m/1.0'})
    wanted=[(f'eth-updown-15m-{t0}',t0) for t0 in range(hour,hour+3600,900)]
    params=[('slug',s) for s,_ in wanted]+[('closed','true'),('limit',20)]
    js=base.get_json(sess,GAMMA+'/markets',params=params)
    by={str(x.get('slug')):x for x in js if isinstance(x,dict)}
    mm=[]; inv=[]
    for slug,t0 in wanted:
        raw=by.get(slug); m=parse_market(raw or {},t0)
        inv.append({'slug':slug,'start':t0,'exists':raw is not None,'mapped':m is not None})
        if m: mm.append(m)
    return mm,inv

def download_eth_1m(start_d,end_d,cache):
    cache.mkdir(parents=True,exist_ok=True); frames=[]
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
    for d in original.daterange(start_d-timedelta(days=1),end_d):
        key=d.isoformat(); cp=cache/f'{SYMBOL}-1m-{key}.csv'
        if not cp.exists():
            url=f'https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{key}.zip'
            r=requests.get(url,timeout=90); r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names=[n for n in z.namelist() if n.endswith('.csv')]
                if len(names)!=1: raise RuntimeError((url,names))
                cp.write_bytes(z.read(names[0]))
        f=pd.read_csv(cp,header=None,names=cols); frames.append(f[['open_time','open','close','close_time']])
    x=pd.concat(frames,ignore_index=True)
    for c in ['open_time','close_time']:
        v=pd.to_numeric(x[c],errors='coerce').astype('Int64');
        if len(v.dropna()) and int(v.dropna().abs().median())>10**14: v=v//1000
        x[c]=v
    for c in ['open','close']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna().astype({'open_time':'int64','close_time':'int64','open':'float64','close':'float64'}).sort_values('open_time').drop_duplicates('open_time').reset_index(drop=True)

def load_eth_1s(start,end,cache):
    cache.mkdir(parents=True,exist_ok=True); sess=requests.Session(); frames=[]
    for d in pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='D'):
        key=d.strftime('%Y-%m-%d'); cp=cache/f'{SYMBOL}-1s-{key}.parquet'
        if cp.exists(): frames.append(pd.read_parquet(cp)); continue
        url=f'https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/1s/{SYMBOL}-1s-{key}.zip'; r=sess.get(url,timeout=120)
        if r.status_code==200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names=[n for n in z.namelist() if n.endswith('.csv')]; df=pd.read_csv(z.open(names[0]),header=None,usecols=[0,4,6])
            df.columns=['open_time','close','close_time']
        else:
            d0=int(pd.Timestamp(key,tz='UTC').timestamp()*1000); d1=d0+86_400_000-1; chunks=[]; cur=d0
            while cur<=d1:
                js=base.get_json(sess,BINANCE_REST,params={'symbol':SYMBOL,'interval':'1s','startTime':cur,'endTime':d1,'limit':1000})
                if not js: break
                z=pd.DataFrame(js); chunks.append(z.iloc[:,[0,4,6]]); nxt=int(z.iloc[-1,6])+1
                if nxt<=cur: raise RuntimeError('pagination stalled')
                cur=nxt
                if len(js)<1000: break
            if not chunks: raise RuntimeError(f'no {SYMBOL} 1s {key}')
            df=pd.concat(chunks,ignore_index=True); df.columns=['open_time','close','close_time']
        for c in ['open_time','close_time']:
            v=pd.to_numeric(df[c],errors='coerce').astype('Int64');
            if len(v.dropna()) and int(v.dropna().abs().median())>10**14: v=v//1000
            df[c]=v
        df['close']=pd.to_numeric(df['close'],errors='coerce'); df=df.dropna().astype({'open_time':'int64','close_time':'int64','close':'float64'}); df.to_parquet(cp,index=False); frames.append(df)
    return stats_base.BinanceSecond(pd.concat(frames,ignore_index=True))

def eth_deribit_instruments():
    rows=[]
    for expired in ('true','false'):
        obj=original.fetch_json(f'{original.DERIBIT_BASE}/get_instruments',{'currency':'ETH','kind':'option','expired':expired}); rows.extend(obj.get('result',[]))
    if not rows: raise RuntimeError('no ETH Deribit options')
    x=pd.DataFrame(rows).drop_duplicates('instrument_name')
    for c in ['expiration_timestamp','creation_timestamp','strike']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['instrument_name','expiration_timestamp','strike'])

def build_anchors(start,end,out):
    cache=out/'anchors'; cache.mkdir(parents=True,exist_ok=True); sd=date.fromisoformat(start); ed=date.fromisoformat(end)-timedelta(days=1)
    bn_df=download_eth_1m(sd,ed,cache/'binance1m'); bn=original.BinanceAnchor.from_df(bn_df)
    inst=eth_deribit_instruments(); selected=original.select_deribit_instruments(inst,bn_df,sd-timedelta(days=1),ed)
    trades=fetch_deribit_trades_parallel(selected,sd-timedelta(days=1),ed,cache/'deribit_trades.parquet'); der=original.DeribitAnchor.from_trades(trades,inst)
    return bn,der,{'selected':len(selected),'usable_iv_trades':len(der.ts)}

def fair_boundary(m,sec,spot,bn,der):
    rv=bn.rv_annualized(sec*1000,60); iv=der.median_iv(sec*1000,30)
    if not (math.isfinite(rv) and math.isfinite(iv) and .05<=rv<=3 and .05<=iv<=3): return None
    op=bn.open_price(m.start); sp=spot.at(sec); tau=m.close-sec
    if not(op>0 and sp>0 and tau>0): return None
    prv=original.digital_prob_up(sp/op,tau,rv); piv=original.digital_prob_up(sp/op,tau,iv)
    if not(math.isfinite(prv) and math.isfinite(piv)): return None
    lo,hi=min(prv,piv),max(prv,piv)
    return {'up':lo,'down':1-hi,'p_rv':prv,'p_iv':piv,'rv':rv,'iv':iv}

def fee_total(m,p,q):
    bm=base.Market(m.slug,m.start,m.condition_id,m.label_up,m.fee_enabled,m.fee_type,m.fee_rate,m.fee_exponent,m.fee_source)
    return base.fee_total(bm,p,q)

def qty_for_budget(m,p,budget=TICKET):
    fps=fee_total(m,p,1000)/1000; q=budget/max(p+fps,1e-12)
    for _ in range(5):
        cost=q*p+fee_total(m,p,q); q*=budget/cost
    return q

def exact_level(qsec,action,outcome,need_mult=1.0,fixed_qty=None):
    z=qsec[(qsec.side_u==action)&(qsec.outcome_l==outcome)&(qsec['size']>0)].copy()
    if z.empty:return None
    prices=sorted(z.price.astype(float).unique(),reverse=(action=='SELL'))
    for p in prices:
        if action=='BUY' and p<ASK_FLOOR: continue
        q=fixed_qty if fixed_qty is not None else qty_for_budget(CURRENT_MARKET,p)
        size=float(z[np.isclose(z.price.astype(float),p,rtol=0,atol=1e-12)]['size'].sum())
        if size+1e-12>=need_mult*q: return float(p),size,float(q)
    return None

CURRENT_MARKET=None

def score_market(m,g,spot,bn,der):
    global CURRENT_MARKET; CURRENT_MARKET=m
    if g is None or g.empty:return None, None
    med_rv=[]
    secs=sorted(int(x) for x in g.timestamp.dropna().unique() if m.start<=int(x)<m.close)
    entry=None
    for sec in secs:
        fb=fair_boundary(m,sec,spot,bn,der)
        if fb: med_rv.append(fb['rv'])
        if sec<m.close-MAX_S2C or sec>m.close-MIN_S2C or fb is None: continue
        qsec=g[g.timestamp==sec]; cand=[]
        for o in ('up','down'):
            lv=exact_level(qsec,'BUY',o,VOLUME_MULT)
            if lv is None: continue
            p,size,q=lv; cost=q*p+fee_total(m,p,q); fair=float(fb[o]); raw=fair-p; edge=(q*fair-fee_total(m,min(max(fair,1e-6),1-1e-6),q)-cost)/q
            if raw>=RAW_GAP_FLOOR and edge>=EDGE_FLOOR: cand.append((edge,o,p,q,cost,fair,raw))
        if cand:
            entry=max(cand,key=lambda x:x[0]); entry_sec=sec; break
    market_rv=float(np.median(med_rv)) if med_rv else math.nan
    if entry is None:return None,market_rv
    edge,o,p,q,cost,fair,raw=entry; exit_kind='settlement_fallback'; exit_sec=m.close; proceeds=q if ((m.label_up>=.5)==(o=='up')) else 0.0
    for sec in [x for x in secs if x>entry_sec]:
        fb=fair_boundary(m,sec,spot,bn,der)
        if fb is None: continue
        lv=exact_level(g[g.timestamp==sec],'SELL',o,VOLUME_MULT,fixed_qty=q)
        if lv is None:continue
        bid,size,_=lv
        if bid>=float(fb[o])-EXIT_BAND-1e-12:
            net=q*bid-fee_total(m,bid,q)
            if net>cost:
                proceeds=net; exit_sec=sec; exit_kind='convergence'; break
    return {'slug':m.slug,'condition_id':m.condition_id,'start':m.start,'decision':entry_sec,'exit':exit_sec,'exit_kind':exit_kind,'side':o,'entry_px':p,'qty':q,'cost':cost,'pnl':proceeds-cost,'raw_gap':raw,'edge_share':edge,'entry_rv':fair_boundary(m,entry_sec,spot,bn,der)['rv']}, market_rv

def score_hour(hour,markets,spot,bn,der):
    if not markets:return [],[]
    sess=requests.Session(); raw=base.query_trade_rows(sess,[base.Market(m.slug,m.start,m.condition_id,m.label_up,m.fee_enabled,m.fee_type,m.fee_rate,m.fee_exponent,m.fee_source) for m in markets],hour,hour+3599)
    tm=base.normalize_trades(raw); rec=[]; rv=[]
    for m in markets:
        r,mrv=score_market(m,tm.get(m.condition_id,pd.DataFrame()),spot,bn,der); rv.append({'start':m.start,'market_rv':mrv});
        if r: rec.append(r)
    return rec,rv

def density_stats(rec,rvdf,start,end):
    days=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='D',tz='UTC'); d=pd.DataFrame({'day':days})
    if len(rec):
        x=rec.copy(); x['day']=pd.to_datetime(x.decision,unit='s',utc=True).dt.floor('D'); cnt=x.groupby('day').size().rename('trades')
    else: cnt=pd.Series(dtype=float,name='trades')
    if len(rvdf):
        r=rvdf.copy(); r['day']=pd.to_datetime(r.start,unit='s',utc=True).dt.floor('D'); rv=r.groupby('day').market_rv.median().rename('median_rv')
    else: rv=pd.Series(dtype=float,name='median_rv')
    d=d.join(cnt,on='day').join(rv,on='day'); d['trades']=d.trades.fillna(0.0); d['t']=np.arange(len(d),dtype=float)
    z=d.dropna(subset=['median_rv']).copy(); out={}
    if len(z)>=20:
        X=np.column_stack([np.ones(len(z)),z.median_rv.to_numpy(float),z.t.to_numpy(float)]); y=z.trades.to_numpy(float); beta=np.linalg.lstsq(X,y,rcond=None)[0]; u=y-X@beta; L=7; S=np.zeros((3,3))
        for i in range(len(z)): S+=u[i]**2*np.outer(X[i],X[i])
        for lag in range(1,L+1):
            w=1-lag/(L+1); G=np.zeros((3,3))
            for i in range(lag,len(z)): G+=u[i]*u[i-lag]*np.outer(X[i],X[i-lag])
            S+=w*(G+G.T)
        inv=np.linalg.inv(X.T@X); cov=inv@S@inv; se=math.sqrt(max(cov[2,2],0)); tstat=beta[2]/se if se else math.nan; p=2*(1-norm.cdf(abs(tstat))) if math.isfinite(tstat) else math.nan
        out={'time_slope_trades_per_day_per_day':float(beta[2]),'nw7_t':float(tstat),'nw7_p':float(p),'rv_coef':float(beta[1])}
    seg=[]
    for a,b in [('2026-05-15','2026-06-15'),('2026-06-15','2026-07-15'),('2026-07-15','2026-08-15')]:
        q=d[(d.day>=pd.Timestamp(a,tz='UTC'))&(d.day<pd.Timestamp(b,tz='UTC'))]; seg.append({'period':[a,b],'trades_per_day':float(q.trades.mean()),'median_rv':float(q.median_rv.median())})
    return d,out,seg

def run(start,end,out,workers=6):
    out.mkdir(parents=True,exist_ok=True); hours=list(range(ts(start),ts(end),3600)); by={}; inv=[]
    with ThreadPoolExecutor(max_workers=min(workers,8)) as ex:
        fut={ex.submit(fetch_markets_hour,h):h for h in hours}
        for f in as_completed(fut):
            h=fut[f]; mm,ii=f.result(); by[h]=mm; inv.extend(ii)
    invdf=pd.DataFrame(inv); invdf.to_csv(out/'inventory.csv',index=False)
    bn,der,am=build_anchors(start,end,out); spot=load_eth_1s(start,end,out/'binance1s')
    rec=[]; rv=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(score_hour,h,by.get(h,[]),spot,bn,der):h for h in hours if by.get(h)}
        for k,f in enumerate(as_completed(fut),1):
            rr,vv=f.result(); rec.extend(rr); rv.extend(vv)
            if k%48==0: print('ETH15M',k,'/',len(fut),'trades',len(rec),flush=True)
    rdf=pd.DataFrame(rec).sort_values('decision') if rec else pd.DataFrame(); rvdf=pd.DataFrame(rv)
    rdf.to_csv(out/'trades.csv',index=False); rvdf.to_csv(out/'market_rv.csv',index=False)
    daily,trend,segs=density_stats(rdf,rvdf,start,end); daily.to_csv(out/'daily_density.csv',index=False)
    eq=50.0; peak=eq; mdd=0.0
    if len(rdf):
        for p in rdf.sort_values('exit').pnl.astype(float): eq+=p; peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    result={'asset':'ETH','timeframe':'15m','period':[start,end],'rules':{'ticket':TICKET,'edge_floor':EDGE_FLOOR,'raw_gap_floor':RAW_GAP_FLOOR,'ask_floor':ASK_FLOOR,'volume_mult':VOLUME_MULT,'exit_band':EXIT_BAND},'markets_expected':len(invdf),'markets_mapped':int(invdf.mapped.sum()),'trades':len(rdf),'pnl':float(rdf.pnl.sum()) if len(rdf) else 0.0,'final_fixed5_equity':eq,'mdd':mdd,'convergence':int((rdf.exit_kind=='convergence').sum()) if len(rdf) else 0,'settlement_fallback':int((rdf.exit_kind=='settlement_fallback').sum()) if len(rdf) else 0,'density_trend':trend,'segments':segs,'anchor_meta':am}
    (out/'summary.json').write_text(json.dumps(result,indent=2)); print('ETH15M_FINAL',json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',default=START); ap.add_argument('--end',default=END); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=6); a=ap.parse_args(); run(a.start,a.end,a.out,a.workers)
if __name__=='__main__': main()
