from __future__ import annotations
import json,time,requests
from concurrent.futures import ThreadPoolExecutor,as_completed

GAMMA='https://gamma-api.polymarket.com'; DATA='https://data-api.polymarket.com'; ASSETS=('BTC','ETH','SOL','XRP')
HOUR=1782388800  # 2026-06-25 12:00 UTC, pre-July

def gamma_hour(h):
    s=requests.Session(); wanted=[(f'{a.lower()}-updown-5m-{t}',a,t) for a in ASSETS for t in range(h,h+3600,300)]
    r=s.get(GAMMA+'/markets',params=[('slug',x[0]) for x in wanted]+[('closed','true'),('limit',100)],timeout=45);r.raise_for_status();by={m['slug']:m for m in r.json()}
    out={t:[] for t in range(h,h+3600,300)}
    for slug,a,t in wanted:
        if slug in by: out[t].append(by[slug]['conditionId'])
    return out

def fetch_slot(t,cids):
    s=requests.Session(); decision=t+240
    q={'market':','.join(cids),'start':decision-120,'end':decision+5,'limit':10000,'offset':0,'takerOnly':'true'}
    t0=time.time();r=s.get(DATA+'/trades',params=q,timeout=90);dt=time.time()-t0
    if r.status_code!=200:return {'slot':t,'status':r.status_code,'seconds':dt,'rows':None,'bytes':len(r.content),'body':r.text[:200]}
    j=r.json();return {'slot':t,'status':200,'seconds':dt,'rows':len(j),'bytes':len(r.content),'cap':len(j)>=10000,'conditions':len(cids)}

if __name__=='__main__':
    slots=gamma_hour(HOUR); t0=time.time();out=[]
    with ThreadPoolExecutor(max_workers=12) as ex:
        fs=[ex.submit(fetch_slot,t,cids) for t,cids in slots.items()]
        for f in as_completed(fs):
            x=f.result();out.append(x);print('SLOT',json.dumps(x),flush=True)
    print('SUMMARY',json.dumps({'wall_s':time.time()-t0,'slots':len(out),'rows':sum(x.get('rows') or 0 for x in out),'bytes':sum(x['bytes'] for x in out),'http500':sum(x['status']==500 for x in out),'cap_hits':sum(bool(x.get('cap')) for x in out),'avg_s':sum(x['seconds'] for x in out)/len(out),'max_s':max(x['seconds'] for x in out)},indent=2),flush=True)
