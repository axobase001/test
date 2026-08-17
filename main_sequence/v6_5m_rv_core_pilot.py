from __future__ import annotations

import io, json, math, zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from main_sequence import final_recent_replay as base
from pm_structural import recalc as original

HF_BASE = "https://huggingface.co/datasets/kachoio/polymarket-5-minute-crypto-up-down-markets/resolve/main"
SII_MATRIX = "https://raw.githubusercontent.com/Filip303/Polymarket/main/data/sii_exec_matrix_btc5m.parquet"

START = "2026-04-01"
END = "2026-05-04"  # end-exclusive; overlap with official/on-chain SII labels
INITIAL = 50.0
TICKET = 5.0
GATE_PS = 0.05
ASK_FLOOR = 0.20
SIZE_MULT = 3.0
S2C_MIN = 60
S2C_MAX = 180
FAIR_BAND = 0.01
STOP_MULT = 0.5

@dataclass(frozen=True)
class M5:
    start: int
    label_up: float
    fee_enabled: bool = True
    fee_type: str = "v2"
    fee_rate: float | None = 0.07
    fee_exponent: float | None = 1.0
    @property
    def close(self): return self.start + 300


def dl(url: str, path: Path):
    if path.exists() and path.stat().st_size > 1000: return
    r=requests.get(url,timeout=300); r.raise_for_status(); path.write_bytes(r.content)


def fee_total(m,p,qty):
    return float(base.fee_total(m,float(p),float(qty)))


def qty_for_budget(m,p,budget):
    fps=fee_total(m,p,1000.0)/1000.0
    q=budget/max(p+fps,1e-12)
    for _ in range(5):
        cost=q*p+fee_total(m,p,q)
        q*=budget/max(cost,1e-12)
    cost=q*p+fee_total(m,p,q)
    if cost>budget: q*=budget/cost
    return max(float(q),0.0)


def fair(m, sec, spot, bn):
    if sec>=m.close: return None
    rv=bn.rv_annualized(sec*1000,60)
    op=bn.open_price(m.start); sp=spot.at(sec)
    if not (math.isfinite(rv) and 0.05<=rv<=3.0 and op>0 and sp>0): return None
    p=original.digital_prob_up(sp/op,m.close-sec,rv)
    if not math.isfinite(p): return None
    return {"up":float(p),"down":float(1-p),"rv":float(rv),"spot":float(sp),"open":float(op)}


def run(out: Path):
    out.mkdir(parents=True,exist_ok=True)
    dl(f"{HF_BASE}/btc_markets.parquet?download=true",out/"btc_markets.parquet")
    dl(f"{HF_BASE}/btc_ticks.parquet?download=true",out/"btc_ticks.parquet")
    dl(SII_MATRIX,out/"sii_exec_matrix_btc5m.parquet")

    mk=pd.read_parquet(out/"btc_markets.parquet")
    labels=pd.read_parquet(out/"sii_exec_matrix_btc5m.parquet")[["condition_id","start_ts","label_up"]]
    mk=mk.merge(labels,on="condition_id",how="inner")
    s0=int(pd.Timestamp(START,tz="UTC").timestamp()); s1=int(pd.Timestamp(END,tz="UTC").timestamp())
    mk=mk[(mk.start_ts>=s0)&(mk.start_ts<s1)].copy()
    wanted=set(mk.condition_id.astype(str))
    ticks=pd.read_parquet(out/"btc_ticks.parquet",filters=[("t",">=",s0),("t","<",s1)])
    ticks=ticks[ticks.condition_id.astype(str).isin(wanted)].copy()
    ticks=ticks.sort_values(["condition_id","t"],kind="mergesort")

    # Same causal anchors as the existing Main Sequence family: Binance 1m 60m RV + Binance 1s spot.
    start_d=pd.Timestamp(START).date(); end_d=(pd.Timestamp(END)-pd.Timedelta(days=1)).date()
    bn_df=original.download_binance_1m(start_d,end_d,out/"binance_1m")
    bn=original.BinanceAnchor.from_df(bn_df)
    spot=base.load_binance_1s(START,END,out/"binance_1s")

    meta=mk.set_index("condition_id")
    eq=INITIAL; peak=eq; mdd=0.0
    events=[]; entries=conv=stops=settles=0; pnl_conv=pnl_stop=pnl_settle=0.0
    markets_seen=0; gate_hits=0; size_reject=0; cash_reject=0
    for cid,g in ticks.groupby("condition_id",sort=False):
        if cid not in meta.index: continue
        r0=meta.loc[cid]
        if isinstance(r0,pd.DataFrame): r0=r0.iloc[0]
        m=M5(int(r0.start_ts),float(r0.label_up))
        markets_seen+=1
        pos=None
        for r in g.itertuples(index=False):
            sec=int(r.t); s2c=m.close-sec
            if sec<m.start or sec>=m.close: continue
            fb=fair(m,sec,spot,bn)
            if fb is None: continue
            # Existing position: convergence first, then executable 0.5G stop.
            if pos is not None:
                outcome=pos["outcome"]
                bid=float(r.bu if outcome=="up" else r.bd)
                bsz=float(r.su if outcome=="up" else r.sd) if pd.notna(r.su if outcome=="up" else r.sd) else 0.0
                qty=pos["qty"]
                if math.isfinite(bid) and bid>0 and bsz+1e-12>=SIZE_MULT*qty:
                    proceeds=qty*bid-fee_total(m,bid,qty); pnl=proceeds-pos["cost"]
                    f=float(fb[outcome]); reason=None
                    if bid>=f-FAIR_BAND-1e-12 and pnl>0: reason="convergence"
                    elif -pnl+1e-12>=STOP_MULT*pos["G"]: reason="stop"
                    if reason:
                        eq+=pnl; peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
                        if reason=="convergence": conv+=1; pnl_conv+=pnl
                        else: stops+=1; pnl_stop+=pnl
                        events.append({"start":m.start,"time":sec,"event":reason,"side":outcome,"entry":pos["ask"],"exit":bid,"pnl":pnl,"G":pos["G"],"rv":fb["rv"],"s2c":s2c})
                        pos=None
                        continue
            # One entry per 5m market.
            if pos is None and not any(e.get("start")==m.start and e.get("event")=="entry" for e in events[-4:]) and S2C_MIN<=s2c<=S2C_MAX:
                cands=[]
                for outcome in ("up","down"):
                    ask=float(r.au if outcome=="up" else r.ad)
                    asz=float(r.sau if outcome=="up" else r.sad) if pd.notna(r.sau if outcome=="up" else r.sad) else 0.0
                    if not (math.isfinite(ask) and ASK_FLOOR<=ask<1): continue
                    qty=qty_for_budget(m,ask,TICKET)
                    if qty<=0: continue
                    cost=qty*ask+fee_total(m,ask,qty)
                    f=float(fb[outcome]); target=qty*f-fee_total(m,min(max(f,1e-6),1-1e-6),qty)
                    G=target-cost; edge_ps=G/qty
                    if edge_ps>GATE_PS:
                        gate_hits+=1
                        if asz+1e-12<SIZE_MULT*qty:
                            size_reject+=1; continue
                        cands.append((G,outcome,ask,qty,cost,f,edge_ps))
                if cands:
                    G,outcome,ask,qty,cost,f,edge_ps=max(cands,key=lambda x:x[0])
                    if eq+1e-12<cost:
                        cash_reject+=1
                    else:
                        pos={"outcome":outcome,"ask":ask,"qty":qty,"cost":cost,"G":G,"entry":sec}
                        entries+=1
                        events.append({"start":m.start,"time":sec,"event":"entry","side":outcome,"entry":ask,"exit":np.nan,"pnl":0.0,"G":G,"edge_ps":edge_ps,"rv":fb["rv"],"s2c":s2c})
        if pos is not None:
            won=(m.label_up>=0.5) if pos["outcome"]=="up" else (m.label_up<0.5)
            payout=pos["qty"] if won else 0.0; pnl=payout-pos["cost"]
            eq+=pnl; peak=max(peak,eq); mdd=min(mdd,eq/peak-1); settles+=1; pnl_settle+=pnl
            events.append({"start":m.start,"time":m.close,"event":"settlement","side":pos["outcome"],"entry":pos["ask"],"exit":1.0 if won else 0.0,"pnl":pnl,"G":pos["G"],"rv":np.nan,"s2c":0})
            pos=None
    ev=pd.DataFrame(events); ev.to_csv(out/"events.csv",index=False)
    summary={
        "period":[START,END],"markets":markets_seen,"initial":INITIAL,"final":eq,"pnl":eq-INITIAL,"mdd":mdd,
        "entries":entries,"convergence":conv,"stops":stops,"settlements":settles,
        "pnl_convergence":pnl_conv,"pnl_stop":pnl_stop,"pnl_settlement":pnl_settle,
        "gate_hits":gate_hits,"size_reject":size_reject,"cash_reject":cash_reject,
        "rules":{"ticket":TICKET,"gate_ps":GATE_PS,"ask_floor":ASK_FLOOR,"size_mult":SIZE_MULT,"s2c":[S2C_MIN,S2C_MAX],"stop_mult_G":STOP_MULT,"fair_band":FAIR_BAND,"fair":"RV-only Binance 60m causal RV + 1s spot","execution":"1Hz historical top-of-book; require best-level resting size >=3x qty on entry and exit","fee":"0.07*p*(1-p), v2 assumption for Apr-May 2026"},
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": run(Path("btc5m_pilot"))
