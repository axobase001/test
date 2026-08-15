from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import duckdb
import requests

ASSETS=("btc","eth","sol","xrp")
HOUR_START=int(datetime(2026,6,25,12,0,tzinfo=timezone.utc).timestamp())
SLUGS=[f"{a}-updown-5m-{t}" for a in ASSETS for t in range(HOUR_START,HOUR_START+3600,300)]

s=requests.Session()
markets=[]
for i in range(0,len(SLUGS),24):
    batch=SLUGS[i:i+24]
    params=[("slug",x) for x in batch]+[("closed","true"),("limit",100)]
    r=s.get("https://gamma-api.polymarket.com/markets",params=params,timeout=30)
    r.raise_for_status(); markets.extend(r.json())
print("GAMMA",len(markets),"of",len(SLUGS),flush=True)
for m in markets[:3]:
    print(json.dumps({k:m.get(k) for k in ["slug","conditionId","outcomes","clobTokenIds"]}),flush=True)

cids=[m["conditionId"] for m in markets]
assert len(cids)>=40, f"too few mapped markets {len(cids)}"
url="https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-25T12.parquet"
con=duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
print("SCHEMA",con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[url]).fetchall(),flush=True)
expr=",".join("encode(?)" for _ in cids)
q=f"""SELECT event_type,count(*) n,min(timestamp_received),max(timestamp_received),count(distinct market),count(distinct asset_id)
FROM read_parquet(?) WHERE market IN ({expr}) GROUP BY event_type ORDER BY event_type"""
t0=time.time(); rows=con.execute(q,[url,*cids]).fetchall(); dt=time.time()-t0
print("RESULT",rows,flush=True); print("SECONDS",round(dt,3),flush=True)
# Sample exact target rows around one market so we can validate quote/trade fields.
cid=cids[0]
rows=con.execute("""SELECT timestamp_received,event_type,asset_id,price,size,side,best_bid,best_ask,bids,asks
FROM read_parquet(?) WHERE market=encode(?) ORDER BY timestamp_received LIMIT 20""",[url,cid]).fetchall()
print("SAMPLE_ROWS",len(rows),flush=True)
for x in rows[:10]: print(x,flush=True)
