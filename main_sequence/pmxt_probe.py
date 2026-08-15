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
print("FILE_TIME",con.execute("SELECT min(epoch_ms(timestamp_received)),max(epoch_ms(timestamp_received)),count(*) FROM read_parquet(?)",[url]).fetchone(),flush=True)

# pmxt stores condition IDs as raw bytes, not ASCII "0x..." strings. Compare by hex/raw bytes.
hexes=[x[2:].upper() for x in cids]
expr=",".join("?" for _ in hexes)
q=f"""SELECT event_type,count(*) n,count(distinct market),count(distinct asset_id)
FROM read_parquet(?) WHERE hex(market) IN ({expr}) GROUP BY event_type ORDER BY event_type"""
t0=time.time(); rows=con.execute(q,[url,*hexes]).fetchall(); print("MARKET_HEX_RESULT",rows,"seconds",round(time.time()-t0,3),flush=True)

# Token-ID predicate is independently useful if historical Gamma token IDs match the archive.
expr2=",".join("?" for _ in tokens)
q2=f"""SELECT event_type,count(*) n,count(distinct market),count(distinct asset_id)
FROM read_parquet(?) WHERE asset_id IN ({expr2}) GROUP BY event_type ORDER BY event_type"""
t0=time.time(); rows2=con.execute(q2,[url,*tokens]).fetchall(); print("TOKEN_RESULT",rows2,"seconds",round(time.time()-t0,3),flush=True)

# Inspect actual archive IDs without converting TIMESTAMPTZ through Python/pytz.
samples=con.execute("""SELECT hex(market) AS market_hex, asset_id, event_type,
                              epoch_ms(timestamp_received) AS recv_ms,
                              CAST(price AS VARCHAR), CAST(size AS VARCHAR), side,
                              CAST(best_bid AS VARCHAR), CAST(best_ask AS VARCHAR)
                       FROM read_parquet(?)
                       WHERE event_type IN ('book','price_change','last_trade_price')
                       LIMIT 12""",[url]).fetchall()
print("ARCHIVE_SAMPLE_ROWS",len(samples),flush=True)
for row in samples: print(row,flush=True)

# Ask Gamma what a few archive condition IDs are; this diagnoses historical-ID mismatch vs file coverage.
seen=[]
for row in samples:
    cid='0x'+str(row[0]).lower()
    if cid not in seen: seen.append(cid)
for cid in seen[:4]:
    r=s.get("https://gamma-api.polymarket.com/markets",params=[("condition_ids",cid),("closed","true"),("limit",5)],timeout=30)
    r.raise_for_status(); js=r.json()
    print("ARCHIVE_GAMMA",cid,[{k:m.get(k) for k in ["slug","conditionId","outcomes","clobTokenIds"]} for m in js[:3]],flush=True)

# Exact target spot-check using raw hex comparison.
target=hexes[0]
spot=con.execute("""SELECT epoch_ms(timestamp_received),event_type,asset_id,CAST(price AS VARCHAR),CAST(size AS VARCHAR),side,
                           CAST(best_bid AS VARCHAR),CAST(best_ask AS VARCHAR),bids,asks
                    FROM read_parquet(?) WHERE hex(market)=? ORDER BY timestamp_received LIMIT 20""",[url,target]).fetchall()
print("TARGET_ROWS",len(spot),flush=True)
for row in spot[:10]: print(row,flush=True)
