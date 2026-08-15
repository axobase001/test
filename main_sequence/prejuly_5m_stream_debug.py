from __future__ import annotations
import json
import pandas as pd
import numpy as np

import prejuly_5m_official as core
if not hasattr(core.Market,'label'):
    core.Market.label=property(lambda self:self.label_up)
import prejuly_5m_official_retryfix as retryfix
import prejuly_5m_official_stream as stream

HOUR=1782388800
SLUG='btc-updown-5m-1782388800'

full_markets,full_rows=retryfix.fetch_hour_fixed(HOUR)
fm={m.slug:m for m in full_markets}; m=fm[SLUG]
# Pull exactly the stream rectangle again, but preserve raw rows for diagnosis.
slot_markets=[x for x in full_markets if x.start==m.start]
import requests
s=requests.Session();s.headers.update({'User-Agent':'main-sequence-sealed-debug/1.0'})
slot_rows=stream._query_rows(s,slot_markets,m.start,m.decision+core.TAPE_SECONDS)

full=pd.DataFrame(full_rows); slot=pd.DataFrame(slot_rows)
def norm(df):
    x=df[df.conditionId.astype(str)==m.condition_id].copy()
    for c in ['timestamp','price','size']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=['timestamp','price','size']);x['timestamp']=x.timestamp.astype(np.int64)
    x=x[(x.timestamp>=m.start)&(x.timestamp<=m.decision+core.TAPE_SECONDS)]
    x['outcome_l']=x.outcome.astype(str).str.lower().str.strip();x['side_u']=x.side.astype(str).str.upper().str.strip()
    x['p_up']=np.where(x.outcome_l.eq('up'),x.price,1.0-x.price)
    return x
F=norm(full);S=norm(slot)
cols=[c for c in ['timestamp','price','size','side','outcome','transactionHash'] if c in F.columns and c in S.columns]
def keyset(x):
    return sorted(tuple(str(v) for v in r) for r in x[cols].itertuples(index=False,name=None))
print('ROWS',json.dumps({'full_total_market_window':len(F),'slot_total_market_window':len(S),'key_cols':cols,'multiset_equal':keyset(F)==keyset(S)},indent=2),flush=True)
# Reproduce old normalize order and show the exact conflicting scalar.
ft=core.normalize_trade_rows(full).get(m.condition_id); st=core.normalize_trade_rows(slot).get(m.condition_id)
for tag,g in [('FULL',ft),('SLOT',st)]:
    pre=g[(g.timestamp>=m.start)&(g.timestamp<m.decision)]
    mx=int(pre.timestamp.max()); tail=pre[pre.timestamp==mx]
    print(tag,'PRE_N',len(pre),'MAX_TS',mx,'LAST_PUP',float(pre.p_up.iloc[-1]),'TIES',len(tail),flush=True)
    show=[c for c in ['timestamp','price','size','side_u','outcome_l','p_up','transactionHash'] if c in tail.columns]
    print(tag+'_LAST_SECOND',tail[show].to_json(orient='records'),flush=True)

# Compare exact feature vectors after intended label alias.
bs=core.load_binance('train')
A={e.slug:e for e in core.build_examples(full_markets,full,bs)}
micros,_=stream.fetch_hour_stream(HOUR);B={e.slug:e for e in stream.build_stream_examples(micros,bs)}
a,b=A[SLUG],B[SLUG]
print('SCALARS',json.dumps({'full_pm_last':a.pm_last,'stream_pm_last':b.pm_last,'full_label':a.label,'stream_label':b.label,'max_abs_x':float(np.max(np.abs(a.x.astype(float)-b.x.astype(float)))),'diff_features':[{ 'name':n,'full':float(x),'stream':float(y),'diff':float(y-x)} for n,x,y in zip(core.FEATURES,a.x,b.x) if abs(float(y-x))>1e-8]},indent=2),flush=True)

# A row-set mismatch means transport is not equivalent.  Equal multiset + scalar mismatch proves order ambiguity.
if keyset(F)!=keyset(S):
    raise RuntimeError('RAW_ROW_MULTISET_MISMATCH')
if a.pm_last!=b.pm_last:
    print('DIAGNOSIS SAME_ROWS_DIFFERENT_ORDER',flush=True)
else:
    print('DIAGNOSIS OTHER_FEATURE_ORDER_EFFECT',flush=True)
