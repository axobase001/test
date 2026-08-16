from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence.audit_orderbook_execution import load_books, QTY, roi

LATENCIES_MS = (100, 250, 500, 750, 1000, 1500, 2000)
MAX_SNAPSHOT_LAG_MS = 350


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signals', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('orderbook_cache'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.signals)
    lo = int(pd.Timestamp('2026-04-24', tz='UTC').timestamp())
    hi = int(pd.Timestamp('2026-04-27', tz='UTC').timestamp())
    x = x[(x['start'] >= lo) & (x['start'] < hi) & x['witness_exit_kind'].eq('settlement')].copy()
    books = load_books(args.cache)
    by_slug = {s: g.sort_values('ts_ms', kind='mergesort') for s, g in books.groupby('window_slug', sort=False)}

    rows = []
    for r in x.itertuples(index=False):
        g = by_slug.get(str(r.slug))
        rec = {'slug':r.slug,'decision':int(r.decision),'side':r.side,'limit':float(r.limit),'won':bool(r.won),
               'witness_cost':float(r.witness_cost),'witness_reward':float(r.witness_reward)}
        if g is None or g.empty:
            for lat in LATENCIES_MS:
                rec[f'l{lat}_snapshot']=False; rec[f'l{lat}_fill']=False
            rows.append(rec); continue
        tt = g['ts_ms'].to_numpy(np.int64)
        for lat in LATENCIES_MS:
            target = int(r.decision)*1000 + lat
            i = int(np.searchsorted(tt, target, side='left'))
            ok = i < len(g) and int(tt[i]) <= target + MAX_SNAPSHOT_LAG_MS
            rec[f'l{lat}_snapshot'] = bool(ok)
            if not ok:
                rec[f'l{lat}_fill']=False; rec[f'l{lat}_ask']=math.nan; rec[f'l{lat}_size']=math.nan; rec[f'l{lat}_snap_lag_ms']=math.nan
                continue
            q = g.iloc[i]
            if str(r.side).lower() == 'up':
                ask=float(q['yes_ask']); size=float(q['yes_ask_size'])
            else:
                ask=float(q['no_ask']); size=float(q['no_ask_size'])
            fill=math.isfinite(ask) and math.isfinite(size) and ask <= float(r.limit)+1e-10 and size+1e-10 >= QTY
            rec[f'l{lat}_fill']=bool(fill); rec[f'l{lat}_ask']=ask; rec[f'l{lat}_size']=size; rec[f'l{lat}_snap_lag_ms']=int(tt[i])-target
        rows.append(rec)

    a=pd.DataFrame(rows); a.to_csv(args.out/'latency_aligned.csv',index=False)
    summary={'period':['2026-04-24','2026-04-27'],'signals':int(len(a)),'qty_shares':QTY,'snapshot_max_lag_ms':MAX_SNAPSHOT_LAG_MS,
             'witness_same_second_roi':roi(a,'witness_reward','witness_cost'),'latencies':{}}
    for lat in LATENCIES_MS:
        have=a[a[f'l{lat}_snapshot']].copy(); f=have[have[f'l{lat}_fill']].copy()
        summary['latencies'][str(lat)]={
            'coverage':int(len(have)), 'fills':int(len(f)), 'fill_rate':float(len(f)/len(have)) if len(have) else None,
            'conservative_roi_frozen_limit':roi(f,'witness_reward','witness_cost'),
            'win_rate':float(f['won'].mean()) if len(f) else None,
            'avg_limit':float(f['limit'].mean()) if len(f) else None,
            'median_snapshot_lag_ms':float(have[f'l{lat}_snap_lag_ms'].median()) if len(have) else None,
        }
    (args.out/'latency_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
