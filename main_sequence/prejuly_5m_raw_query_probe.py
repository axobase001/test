from __future__ import annotations

from collections import Counter
import json
import requests

import prejuly_5m_official as core
import prejuly_5m_official_retryfix as retryfix
import prejuly_5m_official_stream as stream

HOUR=1782388800
SLUG='btc-updown-5m-1782388800'

full_markets, full_rows = retryfix.fetch_hour_fixed(HOUR)
by_slug={m.slug:m for m in full_markets}
target=by_slug[SLUG]
slot_markets=[m for m in full_markets if m.start==target.start]

sess=requests.Session();sess.headers.update({'User-Agent':'main-sequence-raw-query-probe/1.0'})
slot_rows=stream._query_rows(sess,slot_markets,target.start,target.decision+core.TAPE_SECONDS)
single1=stream._query_rows(sess,[target],target.start,target.decision+core.TAPE_SECONDS)
single2=stream._query_rows(sess,[target],target.start,target.decision+core.TAPE_SECONDS)

# Only compare the same target-condition rectangle. Use the raw API dictionaries,
# not pandas-normalized values, to expose any query-shape-dependent response fields.
def target_rows(rows):
    out=[]
    for r in rows:
        if str(r.get('conditionId'))!=target.condition_id: continue
        try:t=int(r.get('timestamp'))
        except Exception:continue
        if target.start<=t<=target.decision+core.TAPE_SECONDS: out.append(dict(r))
    return out

sets={
 'full48_hour':target_rows(full_rows),
 'slot4':target_rows(slot_rows),
 'single1':target_rows(single1),
 'single2':target_rows(single2),
}
keys=sorted(set().union(*(set(r) for rows in sets.values() for r in rows)))
print('RAW_KEYS',keys,flush=True)

# Canonical whole-row representation: include every returned field. This is the
# strongest equality test; if this passes, query shapes return identical raw multisets.
def canon(r):
    return json.dumps(r,sort_keys=True,separators=(',',':'),default=str)
def ctr(rows):return Counter(canon(r) for r in rows)
C={k:ctr(v) for k,v in sets.items()}
base=C['full48_hour']
for name,c in C.items():
    plus=list((c-base).elements())[:5];minus=list((base-c).elements())[:5]
    print('COUNTER_COMPARE',json.dumps({'name':name,'rows':sum(c.values()),'equal_full':c==base,'extra_n':sum((c-base).values()),'missing_n':sum((base-c).values()),'extra_sample':[json.loads(x) for x in plus],'missing_sample':[json.loads(x) for x in minus]},default=str),flush=True)

# Compare a reduced economic identity counter too. This catches the API returning
# the same tx/outcome/price but changing `size` or another economically-used field.
ECON=('timestamp','price','size','side','outcome','transactionHash','asset')
def econ(r):return tuple(str(r.get(k)) for k in ECON)
E={k:Counter(econ(r) for r in v) for k,v in sets.items()}
eb=E['full48_hour']
for name,c in E.items():
    print('ECON_COMPARE',json.dumps({'name':name,'equal_full':c==eb,'extra_n':sum((c-eb).values()),'missing_n':sum((eb-c).values()),'extra_sample':list((c-eb).elements())[:10],'missing_sample':list((eb-c).elements())[:10]}),flush=True)

# Drill into the final pre-decision second and all transaction hashes that occur there.
pre=[r for r in sets['full48_hour'] if int(r['timestamp'])<target.decision]
max_ts=max(int(r['timestamp']) for r in pre)
txs=sorted({str(r.get('transactionHash')) for r in pre if int(r['timestamp'])==max_ts})
print('MAX_PRE_SECOND',max_ts,'TXS',txs,flush=True)
for name,rows in sets.items():
    rr=[r for r in rows if int(r.get('timestamp',-1))==max_ts]
    print(name+'_LAST_SECOND_RAW',json.dumps(rr,sort_keys=True,default=str),flush=True)
    for tx in txs:
        q=[r for r in rows if str(r.get('transactionHash'))==tx]
        print('TX_ROWS',name,tx,json.dumps(q,sort_keys=True,default=str),flush=True)

# Check window-boundary seconds that feed momentum first-observation selection.
for sec in (15,30,60,120):
    cutoff=target.decision-sec
    for name,rows in sets.items():
        q=[r for r in rows if cutoff<=int(r.get('timestamp',-1))<target.decision]
        if not q: continue
        first_ts=min(int(r['timestamp']) for r in q)
        first=[r for r in q if int(r['timestamp'])==first_ts]
        print('BOUNDARY',sec,name,'first_ts',first_ts,'rows',json.dumps(first,sort_keys=True,default=str),flush=True)

# Repeated identical single-market calls must themselves be stable. If not, the API
# is nondeterministic and a canonical same-second reduction is mandatory.
print('REPEAT_SINGLE_EQUAL',C['single1']==C['single2'],flush=True)
