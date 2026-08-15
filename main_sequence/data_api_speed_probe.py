from __future__ import annotations
import json,time,requests
from concurrent.futures import ThreadPoolExecutor,as_completed

GAMMA='https://gamma-api.polymarket.com'
DATA='https://data-api.polymarket.com'
ASSETS=('BTC','ETH','SOL','XRP')
DAY0=1782388800  # 2026-06-25 12:00 UTC; pre-July only


def one_hour(h):
    s=requests.Session(); s.headers.update({'User-Agent':'main-sequence-speed-probe/1.0'})
    wanted=[]
    for a in ASSETS:
        for t in range(h,h+3600,300): wanted.append((f'{a.lower()}-updown-5m-{t}',a,t))
    t0=time.time()
    r=s.get(GAMMA+'/markets',params=[('slug',x[0]) for x in wanted]+[('closed','true'),('limit',100)],timeout=45); r.raise_for_status(); js=r.json()
    gamma_s=time.time()-t0
    by={m['slug']:m for m in js}; cids=[by[x[0]]['conditionId'] for x in wanted if x[0] in by]
    stats=[]
    def pull(group,depth=0):
        q={'market':','.join(group),'start':h,'end':h+3600,'limit':10000,'offset':0,'takerOnly':'true'}
        t=time.time(); rr=s.get(DATA+'/trades',params=q,timeout=90); dt=time.time()-t
        row={'depth':depth,'nmarkets':len(group),'status':rr.status_code,'seconds':dt,'bytes':len(rr.content),'rows':None}
        if rr.status_code!=200:
            row['body']=rr.text[:300]; stats.append(row); return []
        j=rr.json(); row['rows']=len(j); stats.append(row)
        if len(j)>=10000 and len(group)>1:
            m=len(group)//2
            return pull(group[:m],depth+1)+pull(group[m:],depth+1)
        return j
    trades=[]
    for i in range(0,len(cids),24): trades.extend(pull(cids[i:i+24]))
    return {'hour':h,'gamma_s':gamma_s,'markets':len(cids),'trades':len(trades),'stats':stats,'total_s':gamma_s+sum(x['seconds'] for x in stats)}

if __name__=='__main__':
    # 6 representative hours across one pre-July day, using the exact sealed fetch logic.
    hours=[DAY0+i*2*3600 for i in range(6)]
    t=time.time(); out=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut={ex.submit(one_hour,h):h for h in hours}
        for f in as_completed(fut):
            x=f.result(); out.append(x); print('HOUR',json.dumps(x),flush=True)
    print('WALL_SECONDS',round(time.time()-t,3),flush=True)
    flat=[q for x in out for q in x['stats']]
    print('SUMMARY',json.dumps({'hours':len(out),'markets':sum(x['markets'] for x in out),'trades':sum(x['trades'] for x in out),'requests':len(flat),'cap_hits':sum((q.get('rows') or 0)>=10000 for q in flat),'avg_req_s':sum(q['seconds'] for q in flat)/max(len(flat),1),'max_req_s':max([q['seconds'] for q in flat] or [0]),'statuses':{str(k):sum(q['status']==k for q in flat) for k in sorted(set(q['status'] for q in flat))}},indent=2),flush=True)
