from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

S = requests.Session()
S.headers.update({"User-Agent": "main-sequence-sol-probe/1.2"})
HOST = "https://api.manepa.jp"


def req(path, params=None):
    try:
        r=S.get(HOST+path,params=params,timeout=30)
        obj=None
        try: obj=r.json()
        except Exception: pass
        return {"status":r.status_code,"url":r.url,"json":obj,"text":r.text[:2000]}
    except Exception as e:
        return {"error":repr(e)}


def main():
    out={"probe_version":4}
    # Pull all Closed SOL option instruments, paginating if needed.
    closed=[]; cursor=None; pages=[]
    for page in range(20):
        p={"category":"option","baseCoin":"SOL","status":"Closed","limit":1000}
        if cursor: p["cursor"]=cursor
        x=req("/v5/market/instruments-info",p); pages.append(x)
        j=x.get("json") or {}; res=j.get("result") or {}; rows=res.get("list") or []
        closed.extend(rows)
        cursor=res.get("nextPageCursor") or ""
        if not cursor or not rows: break
    out["closed_pages"]=pages
    out["closed_count"]=len(closed)

    lo=1778803200000; hi=1786752000000  # 2026-05-15 .. 2026-08-15 UTC
    target=[z for z in closed if lo <= int(z.get("deliveryTime") or 0) <= hi]
    out["target_closed_count"]=len(target)
    out["target_instruments"]=target[:5000]

    # Sample across target expiries, prioritising strikes near rough midrange is unnecessary:
    # this probe only tests whether expired symbols still expose public recent trades.
    target_sorted=sorted(target,key=lambda z:(int(z.get("deliveryTime") or 0),z.get("symbol","")))
    picks=[]
    if target_sorted:
        idxs=sorted(set([0,len(target_sorted)//4,len(target_sorted)//2,3*len(target_sorted)//4,len(target_sorted)-1]))
        picks=[target_sorted[i] for i in idxs]
    out["expired_trade_probes"]={}
    for z in picks:
        sym=z["symbol"]
        out["expired_trade_probes"][sym]=req("/v5/market/recent-trade",{"category":"option","symbol":sym,"limit":1000})

    # Also query a current symbol as positive control.
    cur=req("/v5/market/instruments-info",{"category":"option","baseCoin":"SOL","limit":1})
    out["current_instrument"]=cur
    cj=cur.get("json") or {}; cl=(cj.get("result") or {}).get("list") or []
    if cl:
        sym=cl[0]["symbol"]
        out["current_trade_control"]=req("/v5/market/recent-trade",{"category":"option","symbol":sym,"limit":1000})

    print("CLOSED_SOL_OPTIONS",len(closed),"TARGET_MAY_AUG",len(target),flush=True)
    if target:
        dts=[int(z["deliveryTime"]) for z in target]
        print("TARGET_DELIVERY_RANGE",min(dts),max(dts),flush=True)
        print("TARGET_SAMPLE",json.dumps(target[:3]),flush=True)
    for sym,x in out["expired_trade_probes"].items():
        j=x.get("json") or {}; rows=(j.get("result") or {}).get("list") or []
        print("EXPIRED_TRADE",sym,"http",x.get("status"),"ret",j.get("retCode"),"n",len(rows),flush=True)
        if rows:
            times=[int(r.get("time") or 0) for r in rows]
            print("EXPIRED_TRADE_RANGE",sym,min(times),max(times),"sample",json.dumps(rows[:1]),flush=True)
    open("sol_probe.json","w",encoding="utf-8").write(json.dumps(out,indent=2))

if __name__=="__main__": main()
