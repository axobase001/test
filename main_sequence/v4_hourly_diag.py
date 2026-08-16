from __future__ import annotations

import json, math
from pathlib import Path
import pandas as pd
from main_sequence import v4_hourly_core_tail as h

START='2026-08-01'; END='2026-08-02'; TICKET=5.0
out=Path('hourly_diag'); out.mkdir(exist_ok=True)
markets, inv = h.discover(START, END, workers=12)
bn, der, meta = h.base.build_anchors(START, END, out)
spot = h.base.load_binance_1s(START, END, out/'binance_1s_cache')
rows=[]
for m in markets:
    g=h.market_tape(m)
    secs=sorted(int(x) for x in g['timestamp'].dropna().unique().tolist()) if not g.empty else []
    fair_valid=0; buy_level_secs=0; max_core=-999.0; max_tail=-999.0; core_pos=0; tail_pos=0
    side_counts = g.groupby(['side_u','outcome_l']).size().to_dict() if not g.empty else {}
    for sec in secs:
        fb=h.fair_boundary(m,sec,spot,bn,der)
        if fb is None: continue
        fair_valid += 1
        q=g[g['timestamp']==sec]
        any_buy=False
        for outcome in ('up','down'):
            lv=h.top_level(q,'BUY',outcome)
            if lv is None: continue
            any_buy=True
            ask, avail=lv
            qty=h.qty_for_budget(m,ask,TICKET)
            if qty<=0 or avail+1e-12<qty: continue
            fair=float(fb[outcome])
            buy_fee=h.fee_ps(m,ask,qty)
            exit_fee=h.fee_ps(m,min(max(fair,1e-6),1-1e-6),qty)
            rt=fair-ask-buy_fee-exit_fee
            tail=fair-ask-buy_fee
            max_core=max(max_core,rt); max_tail=max(max_tail,tail)
            if rt>0: core_pos+=1
            if fair>=h.TAIL_FAVORITE_FAIR and tail>0: tail_pos+=1
        if any_buy: buy_level_secs += 1
    rows.append({
      'start':m.start,'slug':m.slug,'tape_rows':len(g),'unique_secs':len(secs),'fair_valid_secs':fair_valid,
      'buy_level_secs':buy_level_secs,'max_core_rt_edge':None if max_core<-100 else max_core,
      'max_tail_edge':None if max_tail<-100 else max_tail,'core_positive_seconds':core_pos,'tail_positive_seconds':tail_pos,
      'side_counts':json.dumps({f'{a}:{b}':int(v) for (a,b),v in side_counts.items()},sort_keys=True)
    })
df=pd.DataFrame(rows)
df.to_csv(out/'diag.csv',index=False)
summary={
 'markets':len(markets),'anchor_meta':meta,'total_tape_rows':int(df.tape_rows.sum()),
 'markets_with_tape':int((df.tape_rows>0).sum()),'total_fair_valid_secs':int(df.fair_valid_secs.sum()),
 'markets_with_positive_core':int((df.core_positive_seconds>0).sum()),
 'positive_core_seconds':int(df.core_positive_seconds.sum()),
 'markets_with_positive_tail':int((df.tail_positive_seconds>0).sum()),
 'positive_tail_seconds':int(df.tail_positive_seconds.sum()),
 'max_core_rt_edge':float(pd.to_numeric(df.max_core_rt_edge,errors='coerce').max()),
}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(df.to_string(index=False))
