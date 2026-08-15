from __future__ import annotations

import json
import requests

BASE="https://clob.polymarket.com"
GAMMA="https://gamma-api.polymarket.com"
SLUG="btc-updown-5m-1782388800"  # 2026-06-25 12:00 UTC; deliberately pre-July only
START=1782388800
END=START+300
s=requests.Session()

r=s.get(GAMMA+"/markets",params=[("slug",SLUG),("closed","true"),("limit",5)],timeout=30)
r.raise_for_status(); ms=r.json(); assert len(ms)==1, len(ms)
m=ms[0]
outcomes=json.loads(m["outcomes"]); toks=json.loads(m["clobTokenIds"])
print("MARKET",{k:m.get(k) for k in ["slug","conditionId","outcomes","clobTokenIds"]},flush=True)
print("OUTCOMES",list(zip(outcomes,toks)),flush=True)

for tok in toks:
    tests=[
      ("prices-history", {"market":tok,"startTs":START,"endTs":END,"fidelity":1}),
      ("ohlc", {"asset_id":tok,"startTs":START,"endTs":END,"fidelity":"1m","limit":1000}),
      ("orderbook-history-asset", {"asset_id":tok,"startTs":START,"endTs":END,"limit":1000}),
    ]
    for name,params in tests:
        path=name.split("-asset")[0]
        rr=s.get(BASE+"/"+path,params=params,timeout=60)
        print("PROBE",name,"token",tok[:18],"status",rr.status_code,"url",rr.url,flush=True)
        print("BODY",rr.text[:4000],flush=True)

# Condition-ID form may return both outcome books in one call.
rr=s.get(BASE+"/orderbook-history",params={"market":m["conditionId"],"startTs":START,"endTs":END,"limit":1000},timeout=60)
print("PROBE orderbook-history-market status",rr.status_code,"url",rr.url,flush=True)
print("BODY",rr.text[:8000],flush=True)
