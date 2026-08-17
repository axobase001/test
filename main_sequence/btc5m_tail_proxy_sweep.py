from pathlib import Path
import json, math
import pandas as pd
import numpy as np

DATA_URL='https://raw.githubusercontent.com/Filip303/Polymarket/main/data/sii_exec_matrix_btc5m.parquet'
OUT=Path('btc5m_tail_proxy_out'); OUT.mkdir(exist_ok=True)

# Public SII-derived matrix: official/on-chain outcome + causal last observed trade prices.
df=pd.read_parquet(DATA_URL)
# one row per market
thresholds=[0.80,0.85,0.90,0.92,0.95,0.96,0.97,0.98,0.99]
offsets=[30,60,120,180,240]
rows=[]
trades=[]
for off in offsets:
    col=f'p_{off}s'
    if col not in df: continue
    z=df[['condition_id','start_ts','label_up',col]].dropna().copy()
    z=z[(z[col]>0)&(z[col]<1)]
    for th in thresholds:
        # Market-implied favorite; this is a screening upper bound, not executable ask.
        sel=z[(z[col]>=th)|(z[col]<=1-th)].copy()
        if sel.empty:
            continue
        sel['side_up']=sel[col]>=0.5
        sel['entry_px']=np.where(sel['side_up'],sel[col],1-sel[col])
        sel=sel[sel['entry_px']>=th].copy()
        sel['won']=np.where(sel['side_up'],sel['label_up']==1,sel['label_up']==0)
        # Polymarket crypto taker fee per share.
        sel['fee_ps']=0.07*sel['entry_px']*(1-sel['entry_px'])
        sel['net_ev_ps']=sel['won'].astype(float)-sel['entry_px']-sel['fee_ps']
        # fixed $5 total entry cash including fee
        sel['qty']=5.0/(sel['entry_px']+sel['fee_ps'])
        sel['pnl_fixed5']=np.where(sel['won'],sel['qty'],0.0)-5.0
        # sequential equity without compounding: 50 + cumulative fixed-$5 pnl
        eq=50.0+sel.sort_values('start_ts')['pnl_fixed5'].cumsum()
        peak=eq.cummax(); dd=(eq/peak-1).min() if len(eq) else 0
        # day aggregation + day bootstrap normal-ish summary; store simple stats here
        sel['day']=pd.to_datetime(sel['start_ts'],unit='s',utc=True).dt.date.astype(str)
        dayp=sel.groupby('day')['pnl_fixed5'].sum()
        rows.append({
            'offset_s':off,'seconds_to_close':300-off,'threshold':th,'n':len(sel),'days':sel['day'].nunique(),
            'win_rate':sel['won'].mean(),'entry_mean':sel['entry_px'].mean(),
            'net_ev_c_per_share':sel['net_ev_ps'].mean()*100,
            'fixed5_pnl':sel['pnl_fixed5'].sum(),'fixed5_final':50+sel['pnl_fixed5'].sum(),
            'fixed5_mdd_pct':float(dd*100),'trades_per_day':len(sel)/max(sel['day'].nunique(),1),
            'mean_daily_pnl':dayp.mean(),'median_daily_pnl':dayp.median(),
        })
        s=sel[['condition_id','start_ts','label_up',col,'side_up','entry_px','won','fee_ps','net_ev_ps','qty','pnl_fixed5','day']].copy()
        s['offset_s']=off;s['seconds_to_close']=300-off;s['threshold']=th
        s.rename(columns={col:'p_up_proxy'},inplace=True)
        trades.append(s)

res=pd.DataFrame(rows).sort_values(['seconds_to_close','threshold'],ascending=[False,True])
res.to_csv(OUT/'surface.csv',index=False)
if trades: pd.concat(trades,ignore_index=True).to_csv(OUT/'all_selected_proxy_trades.csv',index=False)
# compact candidate ranking: require at least 100 trades, positive fixed5 and positive per-share EV
cand=res[(res.n>=100)&(res.net_ev_c_per_share>0)&(res.fixed5_pnl>0)].sort_values('net_ev_c_per_share',ascending=False)
summary={'rows':len(df),'coverage_start':pd.to_datetime(df.start_ts.min(),unit='s',utc=True).isoformat(),'coverage_end':pd.to_datetime(df.start_ts.max(),unit='s',utc=True).isoformat(),'candidate_cells':cand.to_dict('records'),'warning':'Prices are causal last-observed trades, not executable asks. Positive cells require L1/L2 validation; negative cells are strong no-go evidence.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(res.to_string(index=False))
print(json.dumps(summary,indent=2,default=str))
