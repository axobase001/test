from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from main_sequence import v4_hourly_core_tail as v4
from main_sequence import eth15m_conservative_replay as eth

START = "2026-05-15"
END = "2026-08-15"
TICKET = 5.0
EDGE_FLOOR = 0.05
ASK_FLOOR = 0.20
SIZE_MULT = 3.0
STOP_MULT_G = 0.5
FAIR_BAND = 0.01
NY = ZoneInfo("America/New_York")
GAMMA = "https://gamma-api.polymarket.com"


def local_dt(start: int):
    return datetime.fromtimestamp(int(start), tz=timezone.utc).astimezone(NY)


def slug_candidates(start: int) -> list[str]:
    d = local_dt(start); month = d.strftime("%B").lower(); h = d.hour % 12 or 12; ap = "am" if d.hour < 12 else "pm"
    return list(dict.fromkeys([
        f"ethereum-up-or-down-{month}-{d.day}-{d.year}-{h}{ap}-et",
        f"ethereum-up-or-down-{month}-{d.day}-{d.year}-{h}-{ap}-et",
        f"ethereum-up-or-down-{month}-{d.day}-{h}{ap}-et",
        f"ethereum-up-or-down-{month}-{d.day}-{h}-{ap}-et",
        f"eth-updown-1h-{int(start)}",
    ]))


def event_years(event: dict) -> set[int]:
    years=set(); objs=[event]+[x for x in (event.get("markets") or []) if isinstance(x,dict)]
    for obj in objs:
        for k in ("startDate","endDate","startDateIso","endDateIso","createdAt","updatedAt","closedTime","startTime","endTime"):
            v=obj.get(k)
            if v in (None,""): continue
            try:
                t=pd.Timestamp(v)
                if t.tzinfo is None: t=t.tz_localize("UTC")
                years.add(int(t.tz_convert(NY).year))
            except Exception: pass
    return years


def fetch_market(start: int):
    sess=requests.Session(); sess.headers.update({"User-Agent":"main-sequence-eth1h-strict3/1.0"})
    expected=local_dt(start).year; wrong=[]; errors=[]
    for slug in slug_candidates(start):
        try:
            js=v4.get_json(sess,GAMMA+"/events",params={"slug":slug,"closed":"true","limit":5},tries=4)
            for event in js or []:
                yrs=event_years(event)
                if yrs and expected not in yrs:
                    wrong.append({"slug":slug,"years":sorted(yrs)}); continue
                m=v4.parse_event(event,int(start),slug)
                if m is not None:
                    return m,{"start":int(start),"mapped":True,"event_slug":slug,"market_slug":m.slug,"condition_id":m.condition_id,"event_years":"|".join(map(str,sorted(yrs)))}
        except Exception as exc: errors.append(repr(exc))
    return None,{"start":int(start),"mapped":False,"event_slug":None,"market_slug":None,"condition_id":None,"wrong_year":json.dumps(wrong[:4]),"errors":" | ".join(errors[:3])}


def discover(start: str,end: str,workers=20):
    s0=int(pd.Timestamp(start,tz="UTC").timestamp()); s1=int(pd.Timestamp(end,tz="UTC").timestamp())
    starts=list(range(s0,s1,3600)); markets=[]; inv=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(fetch_market,s):s for s in starts}
        for i,f in enumerate(as_completed(fut),1):
            m,row=f.result(); inv.append(row)
            if m is not None: markets.append(m)
            if i%120==0: print("ETH1H_DISCOVER",i,"/",len(starts),"mapped",len(markets),flush=True)
    markets.sort(key=lambda x:x.start)
    return markets,pd.DataFrame(inv).sort_values("start",kind="mergesort")


def fair_boundary(m,sec,spot,bn,der):
    if sec>=m.close: return None
    rv=bn.rv_annualized(sec*1000,60); iv=der.median_iv(sec*1000,30)
    if not (math.isfinite(rv) and math.isfinite(iv) and 0.05<=rv<=3.0 and 0.05<=iv<=3.0): return None
    op=bn.open_price(m.start); sp=spot.at(sec); tau=m.close-sec
    if not (op>0 and sp>0 and tau>0): return None
    prv=eth.original.digital_prob_up(sp/op,tau,rv); piv=eth.original.digital_prob_up(sp/op,tau,iv)
    if not (math.isfinite(prv) and math.isfinite(piv)): return None
    lo,hi=min(prv,piv),max(prv,piv)
    return {"up":float(lo),"down":float(1-hi),"p_rv":float(prv),"p_iv":float(piv),"rv":float(rv),"iv":float(iv),"spot":float(sp),"open":float(op)}


def qty_for_budget(m,p,budget=TICKET):
    return v4.qty_for_budget(m,float(p),float(budget))


def score_market(m,spot,bn,der):
    g=v4.market_tape(m)
    if g is None or g.empty: return [],{"slug":m.slug,"tape":False}
    secs=sorted(int(x) for x in g.timestamp.dropna().unique() if m.start<=int(x)<m.close)
    pos=None; events=[]; entries=conv=stops=settles=gate_hits=size_reject=0; pnl_total=0.0; last_exit_sec=-1
    for sec in secs:
        fb=fair_boundary(m,sec,spot,bn,der)
        if fb is None: continue
        qsec=g[g.timestamp==sec]
        exited=False
        if pos is not None:
            lv=v4.top_level(qsec,"SELL",pos["outcome"])
            if lv is not None:
                bid,avail=map(float,lv); qty=float(pos["qty"])
                if avail+1e-12>=SIZE_MULT*qty:
                    proceeds=qty*bid-v4.fee_total(m,bid,qty); pnl=proceeds-float(pos["cost"]); fair=float(fb[pos["outcome"]]); reason=None
                    if bid>=fair-FAIR_BAND-1e-12 and pnl>1e-12: reason="convergence"
                    elif -pnl+1e-12>=STOP_MULT_G*float(pos["G"]): reason="stop_0.5G"
                    if reason:
                        pnl_total+=pnl; conv+=reason=="convergence"; stops+=reason!="convergence"
                        events.append({"time":sec,"event":reason,"side":pos["outcome"],"entry":pos["ask"],"exit":bid,"qty":qty,"pnl":pnl,"G":pos["G"],"depth_x":avail/max(qty,1e-12)})
                        pos=None; last_exit_sec=sec; exited=True
        # strict3: never re-enter in the exact second used to exit.
        if pos is None and not exited and sec>last_exit_sec:
            cands=[]
            for outcome in ("up","down"):
                lv=v4.top_level(qsec,"BUY",outcome)
                if lv is None: continue
                ask,avail=map(float,lv)
                if not (ASK_FLOOR<=ask<1.0): continue
                qty=qty_for_budget(m,ask)
                if qty<=0: continue
                cost=qty*ask+v4.fee_total(m,ask,qty); fair=float(fb[outcome])
                target=qty*fair-v4.fee_total(m,min(max(fair,1e-6),1-1e-6),qty); G=target-cost; edge_ps=G/qty
                if edge_ps+1e-12>=EDGE_FLOOR:
                    gate_hits+=1
                    if avail+1e-12<SIZE_MULT*qty:
                        size_reject+=1; continue
                    cands.append((G,outcome,ask,avail,qty,cost,fair,edge_ps))
            if cands:
                G,outcome,ask,avail,qty,cost,fair,edge_ps=max(cands,key=lambda x:x[0])
                pos={"outcome":outcome,"ask":ask,"qty":qty,"cost":cost,"G":G,"entry":sec,"fair":fair,"edge_ps":edge_ps}
                entries+=1; events.append({"time":sec,"event":"entry","side":outcome,"entry":ask,"exit":np.nan,"qty":qty,"pnl":0.0,"G":G,"edge_ps":edge_ps,"depth_x":avail/max(qty,1e-12)})
    if pos is not None:
        won=(m.label_up>=0.5) if pos["outcome"]=="up" else (m.label_up<0.5)
        payout=pos["qty"] if won else 0.0; pnl=payout-pos["cost"]; pnl_total+=pnl; settles+=1
        events.append({"time":m.close,"event":"settlement","side":pos["outcome"],"entry":pos["ask"],"exit":1.0 if won else 0.0,"qty":pos["qty"],"pnl":pnl,"G":pos["G"],"edge_ps":pos["edge_ps"],"depth_x":np.nan})
    return events,{"slug":m.slug,"tape":True,"entries":entries,"convergence":int(conv),"stops":int(stops),"settlements":settles,"pnl":pnl_total,"gate_hits":gate_hits,"size_reject":size_reject}


def main():
    out=Path("eth1h_strict3_out"); out.mkdir(parents=True,exist_ok=True)
    markets,inv=discover(START,END); inv.to_csv(out/"inventory.csv",index=False)
    if not markets: raise RuntimeError("no ETH hourly markets mapped")
    bn,der,anchor_meta=eth.build_anchors(START,END,out); spot=eth.load_eth_1s(START,END,out/"binance1s")
    ev=[]; mr=[]; failures=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(score_market,m,spot,bn,der):m for m in markets}
        for i,f in enumerate(as_completed(fut),1):
            m=fut[f]
            try:
                ee,ss=f.result(); mr.append(ss)
                for x in ee: ev.append({"start":m.start,"slug":m.slug,"label_up":m.label_up,**x})
            except Exception as exc: failures.append({"slug":m.slug,"error":repr(exc)})
            if i%48==0: print("ETH1H_SCORE",i,"/",len(markets),"events",len(ev),"fail",len(failures),flush=True)
    if failures: raise RuntimeError(f"ETH1H failures {failures[:10]} count={len(failures)}")
    edf=pd.DataFrame(ev).sort_values(["time","event"],kind="mergesort") if ev else pd.DataFrame(); mdf=pd.DataFrame(mr)
    edf.to_csv(out/"events.csv",index=False); mdf.to_csv(out/"market_summary.csv",index=False)
    exits=edf[edf.event.isin(["convergence","stop_0.5G","settlement"])].copy() if len(edf) else pd.DataFrame()
    eq=50.0; peak=eq; mdd=0.0
    if len(exits):
        for p in exits.sort_values("time").pnl.astype(float): eq+=p; peak=max(peak,eq); mdd=min(mdd,eq/peak-1.0)
    entries=edf[edf.event=="entry"] if len(edf) else pd.DataFrame()
    summary={
        "asset":"ETH","timeframe":"1h","period":[START,END],"fixed_ticket":TICKET,
        "rules":{"entry_net_convergence_edge_ps":EDGE_FLOOR,"ask_floor":ASK_FLOOR,"stop_mult_G":STOP_MULT_G,"strict_depth_mult":SIZE_MULT,"fair_band":FAIR_BAND,"same_second_reentry":False,
                 "fair":"conservative min/max boundary from Binance ETHUSDT 60m RV + backward Deribit ETH option IV","reference":"Binance ETHUSDT 1H open/close"},
        "markets_expected":int(len(inv)),"markets_mapped":int(inv.mapped.sum()),"markets_with_tape":int(mdf.tape.sum()) if len(mdf) else 0,
        "entries":int(len(entries)),"exits":int(len(exits)),"convergence":int((exits.event=="convergence").sum()) if len(exits) else 0,
        "stops":int((exits.event=="stop_0.5G").sum()) if len(exits) else 0,"settlements":int((exits.event=="settlement").sum()) if len(exits) else 0,
        "pnl":float(exits.pnl.sum()) if len(exits) else 0.0,"final_fixed5_equity":float(eq),"mdd_pct":float(mdd*100),
        "entry_edge_mean_c":float(entries.edge_ps.mean()*100) if len(entries) else None,"entry_depth_headroom_median_x":float(entries.depth_x.median()) if len(entries) else None,
        "gate_hits":int(mdf.gate_hits.sum()) if len(mdf) else 0,"size_reject":int(mdf.size_reject.sum()) if len(mdf) else 0,"anchor_meta":anchor_meta,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print("ETH1H_STRICT3_FINAL",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
