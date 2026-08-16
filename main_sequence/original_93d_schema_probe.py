from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = Path('original_93d_schema_probe')
OUT.mkdir(exist_ok=True)
api = HfApi()


def trent_probe():
    repo='trentmkelly/polymarket_crypto_derivatives'
    info=api.dataset_info(repo); rev=info.sha
    files=api.list_repo_files(repo, revision=rev, repo_type='dataset')
    btc=[x for x in files if 'btc15m' in x.lower() and ('2026-03-24' in x or '2026-03-23' in x)]
    print('TRENT_MATCHES', len(btc), btc[:20], flush=True)
    # Prefer steps/events parquet for a complete episode near source-switch overlap.
    steps=[x for x in btc if x.endswith('/steps.parquet')]
    if not steps:
        steps=[x for x in files if 'btc15m' in x.lower() and x.endswith('/steps.parquet')]
    if not steps: raise RuntimeError('no Trent BTC15m steps.parquet')
    sp=steps[-1]
    ep=sp.rsplit('/',1)[0]
    ev=ep+'/events.parquet'
    local_s=hf_hub_download(repo,sp,repo_type='dataset',revision=rev)
    local_e=hf_hub_download(repo,ev,repo_type='dataset',revision=rev)
    s=pd.read_parquet(local_s); e=pd.read_parquet(local_e)
    return {'repo':repo,'revision':rev,'episode':ep,'steps_path':sp,'events_path':ev,
            'steps_shape':list(s.shape),'steps_columns':list(s.columns),'steps_head':s.head(3).to_dict('records'),
            'events_shape':list(e.shape),'events_columns':list(e.columns),'events_head':e.head(5).to_dict('records')}


def pq_probe():
    repo='predict-quant/poly-btc-orderbook'
    info=api.dataset_info(repo); rev=info.sha
    files=api.list_repo_files(repo, revision=rev, repo_type='dataset')
    cand=sorted(x for x in files if x.startswith('15m/2026/03/25/') and ('btc-updown-15m-' in x))
    if not cand: raise RuntimeError('no predict-quant Mar25 BTC15m files')
    path=cand[len(cand)//2]
    local=hf_hub_download(repo,path,repo_type='dataset',revision=rev)
    p=Path(local)
    if p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:
            names=z.namelist()
            raw=z.read(names[0])
            member=names[0]
    else:
        raw=p.read_bytes(); member=p.name
    lines=raw.splitlines()
    sample=[]
    for b in lines[:200]:
        try: sample.append(json.loads(b))
        except Exception: pass
        if len(sample)>=12: break
    types={}
    for r in sample:
        t=str(r.get('event_type') or r.get('type') or r.get('event') or 'unknown')
        types[t]=types.get(t,0)+1
    return {'repo':repo,'revision':rev,'path':path,'local_size':p.stat().st_size,'member':member,
            'uncompressed_bytes':len(raw),'line_count':len(lines),'sample':sample,'sample_event_types':types}


def coverage_inventory():
    repo='predict-quant/poly-btc-orderbook'; rev=api.dataset_info(repo).sha
    files=api.list_repo_files(repo, revision=rev, repo_type='dataset')
    days={}
    for x in files:
        m=re.match(r'15m/2026/(\d{2})/(\d{2})/btc-updown-15m-(\d+)\.(?:zip|jsonl)$',x)
        if m:
            day=f'2026-{m.group(1)}-{m.group(2)}'; days.setdefault(day,[]).append(x)
    return {d:len(v) for d,v in sorted(days.items()) if '2026-03-25'<=d<='2026-06-01'}


def main():
    out={}
    for name,fn in [('trent',trent_probe),('predict_quant',pq_probe),('predict_quant_daily_counts',coverage_inventory)]:
        try: out[name]=fn()
        except Exception as e: out[name]={'error':repr(e)}
        (OUT/'probe.json').write_text(json.dumps(out,indent=2,default=str))
    print(json.dumps(out,indent=2,default=str),flush=True)

if __name__=='__main__': main()
