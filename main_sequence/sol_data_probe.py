from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HOST="https://api.manepa.jp"
UA={"User-Agent":"main-sequence-sol-monthly-scan/1.0"}
EXPIRIES=["29MAY26","26JUN26","31JUL26","28AUG26","25SEP26","30OCT26"]
STRIKES=range(20,251)


def one(sym:str,limit:int=1):
    try:
        r=requests.get(HOST+"/v5/market/recent-trade",params={"category":"option","symbol":sym,"limit":limit},headers=UA,timeout=20)
        j=r.json()
        rows=(j.get("result") or {}).get("list") or []
        return {"symbol":sym,"http":r.status_code,"retCode":j.get("retCode"),"retMsg":j.get("retMsg"),"rows":rows}
    except Exception as e:
        return {"symbol":sym,"error":repr(e),"rows":[]}


def main():
    candidates=[]
    for exp in EXPIRIES:
        for k in STRIKES:
            for cp in ("C","P"):
                candidates.append(f"SOL-{exp}-{k}-{cp}-USDT")
                candidates.append(f"SOL-{exp}-{k}-{cp}")
    found=[]; errors=0
    with ThreadPoolExecutor(max_workers=48) as ex:
        futs=[ex.submit(one,s,1) for s in candidates]
        for i,f in enumerate(as_completed(futs),1):
            x=f.result()
            if x.get("rows"): found.append(x["symbol"])
            elif x.get("error"): errors+=1
            if i%1000==0: print("SCAN",i,"/",len(candidates),"found",len(found),"errors",errors,flush=True)
    found=sorted(set(found))
    print("MONTHLY_FOUND",len(found),flush=True)
    print("MONTHLY_SYMBOLS",json.dumps(found[:300]),flush=True)

    full={}; truncated=[]
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs={ex.submit(one,s,1000):s for s in found}
        for f in as_completed(futs):
            x=f.result(); rows=x.get("rows") or []; full[x["symbol"]]=x
            if len(rows)>=1000: truncated.append(x["symbol"])
    summary={}
    for exp in EXPIRIES:
        syms=[s for s in found if f"-{exp}-" in s]
        rows=[r for s in syms for r in (full.get(s,{}).get("rows") or [])]
        times=[int(r.get("time") or 0) for r in rows if r.get("time")]
        summary[exp]={"symbols":len(syms),"trades":len(rows),"min_time":min(times) if times else None,"max_time":max(times) if times else None,"truncated_symbols":sum(s in truncated for s in syms)}
        print("EXPIRY",exp,json.dumps(summary[exp]),flush=True)
        if rows: print("TRADE_SAMPLE",exp,json.dumps(rows[:2]),flush=True)
    out={"expiries":EXPIRIES,"strike_range":[20,250],"found_symbols":found,"truncated_symbols":truncated,"summary":summary,"full":full}
    open("sol_probe.json","w",encoding="utf-8").write(json.dumps(out))

if __name__=="__main__": main()
