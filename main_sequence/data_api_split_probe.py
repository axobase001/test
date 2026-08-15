from __future__ import annotations
import json,time,requests
from concurrent.futures import ThreadPoolExecutor,as_completed

GAMMA='https://gamma-api.polymarket.com'; DATA='https://data-api.polymarket.com'; ASSETS=('BTC','ETH','SOL','XRP')
DAY0=1782388800

def one_hour(h):
    s=requests.Session(); s.headers.update({'User-Agent':'main-sequence-split-probe/1.0'})
    wanted=[(f'{a.lower()}-updown-5m-{t}',a,t) for a in ASSETS for t in range(h,h+3600,300)]
    r=s.get(GAMMA+'/markets',params=[('slug',x[0]) for x in wanted]+[('closed','true'),('limit',100)],timeout=45); r.raise_for_status(); by={m['slug']:m for m in r.json()}
    cids=[by[x[0]]['conditionId'] for x in wanted if x[0] in by]; stats=[]
    def pull(group,depth=0):
        q={'market':','.join(group),'start':h,'end':h+3600,'limit':10000,'offset':0,'takerOnly':'true'}
        t=time.time(); rr=s.get(DATA+'/trades',params=q,timeout=90); dt=time.time()-t
        row={'depth':depth,'nmarkets':len(group),'status':rr.status_code,'seconds':dt,'rows':None,'bytes':len(rr.content)}; stats.append(row)
        if rr.status_code==500 and len(group)>1:
            mid=len(group)//2; return pull(group[:mid],depth+1)+pull(group[mid:],depth+1)
        rr.raise_for_status(); j=rr.json(); row['rows']=len(j)
        if len(j)>=10000 and len(group)>1:
            mid=len(group)//2; return pull(group[:mid],depth+1)+pull(group[mid:],depth+1)
        if len(j)>=10000: raise RuntimeError('single market cap')
        return j
    out=[]
    for i in range(0,len(cids),24): out.extend(pull(cids[i:i+24]))
    return {'hour':h,'markets':len(cids),'trades':len(out),'stats':stats}

if __name__=='__main__':
    hours=[DAY0+i*2*3600 for i in range(6)]; t=time.time(); results=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        fs=[ex.submit(one_hour,h) for h in hours]
        for f in as_completed(fs):
            x=f.result(); results.append(x); print('HOUR',json.dumps(x),flush=True)
    flat=[q for x in results for q in x['stats']]
    print('SUMMARY',json.dumps({'wall_s':time.time()-t,'hours':len(results),'markets':sum(x['markets'] for x in results),'trades':sum(x['trades'] for x in results),'requests':len(flat),'http500':sum(q['status']==500 for q in flat),'by_batch':{str(n):{'calls':sum(q['nmarkets']==n for q in flat),'500s':sum(q['nmarkets']==n and q['status']==500 for q in flat),'avg_s':sum(q['seconds'] for q in flat if q['nmarkets']==n)/max(1,sum(q['nmarkets']==n for q in flat))} for n in sorted(set(q['nmarkets'] for q in flat),reverse=True)}},indent=2),flush=True)
