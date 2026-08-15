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
tokens=[]
for m in markets:
    outcomes=json.loads(m["outcomes"]); ids=json.loads(m["clobTokenIds"])
    assert len(outcomes)==len(ids)==2
    tokens.extend(ids)
assert len(cids)>=40 and len(tokens)>=80, (len(cids),len(tokens))
url="https://r2v2.pmxt.dev/polymarket_orderbook_2026-06-25T12.parquet"
con=duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
print("SCHEMA",con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[url]).fetchall(),flush=True)

# asset_id is a documented exact-match fast predicate and avoids any BLOB representation ambiguity on market.
expr=",".join("?" for _ in tokens)
q=f"""SELECT event_type,count(*) n,min(timestamp_received),max(timestamp_received),count(distinct market),count(distinct asset_id)
FROM read_parquet(?) WHERE asset_id IN ({expr}) GROUP BY event_type ORDER BY event_type"""
t0=time.time(); rows=con.execute(q,[url,*tokens]).fetchall(); dt=time.time()-t0
print("TOKEN_RESULT",rows,flush=True); print("TOKEN_SECONDS",round(dt,3),flush=True)

# Diagnose market BLOB representation separately; not needed by the production path if token filtering works.
print("BLOB_SAMPLE",con.execute("SELECT decode(market),hex(market),asset_id,event_type,timestamp_received FROM read_parquet(?) LIMIT 5",[url]).fetchall(),flush=True)
cid=cids[0]
q2="SELECT count(*) FROM read_parquet(?) WHERE decode(market)=?"
t0=time.time(); print("DECODE_CID_COUNT",con.execute(q2,[url,cid]).fetchone()[0],"seconds",round(time.time()-t0,3),flush=True)

# Sample exact target rows around one outcome token so quote/trade field semantics are visible.
tok=tokens[0]
rows=con.execute("""SELECT timestamp_received,event_type,asset_id,price,size,side,best_bid,best_ask,bids,asks
FROM read_parquet(?) WHERE asset_id=? ORDER BY timestamp_received LIMIT 20""",[url,tok]).fetchall()
print("SAMPLE_ROWS",len(rows),flush=True)
for x in rows[:10]: print(x,flush=True)
