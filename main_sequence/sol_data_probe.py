from __future__ import annotations

import json
import requests

BASE="https://history.deribit.com/api/v2/public"
S=requests.Session(); S.headers.update({"User-Agent":"main-sequence-sol-deribit-probe/1.0"})


def get(method,params):
    try:
        r=S.get(f"{BASE}/{method}",params=params,timeout=60)
        j=None
        try:j=r.json()
        except Exception:pass
        return {"status":r.status_code,"url":r.url,"json":j,"text":r.text[:4000]}
    except Exception as e:return {"error":repr(e)}


def main():
    out={}
    expired=get("get_instruments",{"currency":"USDC","kind":"option","expired":"true"})
    live=get("get_instruments",{"currency":"USDC","kind":"option","expired":"false"})
    out["expired_raw_meta"]={k:v for k,v in expired.items() if k!="json"}
    out["live_raw_meta"]={k:v for k,v in live.items() if k!="json"}
    er=(expired.get("json") or {}).get("result") or []
    lr=(live.get("json") or {}).get("result") or []
    sol=[x for x in er+lr if str(x.get("instrument_name") or "").startswith("SOL_USDC-")]
    # Dedup and target instruments that existed around the 93d replay window.
    by={x["instrument_name"]:x for x in sol}; sol=list(by.values())
    lo=1778716800000; hi=1790985600000 # allow expiries around target + 65d
    target=[x for x in sol if lo <= int(x.get("expiration_timestamp") or 0) <= hi]
    target=sorted(target,key=lambda x:(x.get("expiration_timestamp",0),x.get("strike",0),x.get("option_type","")))
    out["sol_total"]=len(sol); out["target_count"]=len(target); out["target_sample"]=target[:20]
    print("DERIBIT_HTTP",expired.get("status"),live.get("status"),flush=True)
    print("DERIBIT_USDC_ALL",len(er),len(lr),"SOL_USDC",len(sol),"TARGET",len(target),flush=True)
    if target: print("TARGET_SAMPLE",json.dumps(target[:3]),flush=True)

    picks=[]
    if target:
        idxs=sorted(set([0,len(target)//4,len(target)//2,3*len(target)//4,len(target)-1]))
        picks=[target[i] for i in idxs]
    out["trade_probes"]={}
    start=1778716800000; end=1786751999999
    for x in picks:
        name=x["instrument_name"]
        t=get("get_last_trades_by_instrument",{"instrument_name":name,"start_timestamp":start,"end_timestamp":end,"count":1000,"sorting":"asc"})
        out["trade_probes"][name]=t
        rows=((t.get("json") or {}).get("result") or {}).get("trades") or []
        print("TRADE_PROBE",name,"http",t.get("status"),"n",len(rows),flush=True)
        if rows: print("TRADE_SAMPLE",json.dumps(rows[:2]),flush=True)
    open("sol_probe.json","w",encoding="utf-8").write(json.dumps(out))

if __name__=="__main__":main()
