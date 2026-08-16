from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from main_sequence import v4_hourly_core_tail as v4
from main_sequence import final_recent_replay as base

NY = ZoneInfo("America/New_York")
GAMMA = "https://gamma-api.polymarket.com"
TAIL_FAVORITE_FAIR = 0.99
FAIR_BAND = 0.01
INITIAL_EQUITY = 50.0
BASE_TICKET = 5.0
MAX_TICKET_USD = 100.0
MAX_MARKET_OPEN_USD = 100.0
WINDOWS = {
    "3m": ["2026-05-15", "2026-08-15"],
    "6m": ["2026-02-15", "2026-08-15"],
    "9m": ["2025-11-15", "2026-08-15"],
    "12m": ["2025-08-15", "2026-08-15"],
}

PROTOCOL = {
    "name": "Main Sequence V5 hourly symmetric expected-profit stop / 2026-08-16 freeze",
    "market": "Polymarket BTC Up/Down 1h",
    "reference": "Binance BTCUSDT 1H candle open/close",
    "fair": "conservative boundary from causal Binance 60m RV and backward 30m median Deribit trade IV",
    "execution_proxy": "same-second best observed public-tape price level only; exact level must alone contain enough size; no future-second fill test",
    "core": {
        "repeatable": True,
        "entry": "expected net convergence profit at entry fair > 0 after entry fee and estimated exit fee",
        "take_profit": "sell at same-second best bid when bid is within 1c of current causal fair and net proceeds exceed entry cost",
        "stop": "freeze entry expected net convergence profit G; if executable current net liquidation loss L >= G, sell immediately",
        "stop_formula": "G=(qty*entry_fair-exit_fee_at_entry_fair)-entry_cost; L=entry_cost-(qty*current_bid-current_sell_fee)",
        "fallback": "if neither executable convergence nor executable symmetric stop occurs before close, settle the still-open binary position",
    },
    "tail": {
        "favorite_fair_floor": TAIL_FAVORITE_FAIR,
        "entry": "favorite conservative fair >=99% and positive post-entry-fee settlement edge",
        "exit": "settlement",
        "priority": "TAIL beats simultaneous new CORE and blocks all later new CORE entries in that market; earlier CORE is not retroactively cancelled",
    },
    "portfolio": {
        "initial_equity": INITIAL_EQUITY,
        "base_ticket": BASE_TICKET,
        "ticket_rule": "current-equity power-of-two ladder; doubles/halves at each 2x equity boundary",
        "max_ticket_usd": MAX_TICKET_USD,
        "max_combined_open_capital_per_market_usd": MAX_MARKET_OPEN_USD,
        "no_leverage": True,
        "capital_reuse": "realized CORE exit capital/PnL is immediately reusable for later entries in the same 1h market",
    },
    "windows": WINDOWS,
    "anti_lookahead": [
        "market year is validated from Gamma event/market timestamps; wrong-year yearless slug matches are rejected",
        "fair uses only anchors timestamped <= the current second",
        "entry/exit existence and size use only the current-second public tape proxy",
        "entry expected-profit stop budget is frozen at entry and never uses future outcome",
        "final outcome is used only for positions still open at settlement",
    ],
}

STATE_COLS = [
    "start","close","slug","event_slug","condition_id","label_up","fee_enabled","fee_type","fee_rate","fee_exponent","fee_source",
    "sec","fair_up","fair_down","p_rv","p_iv","rv","iv","spot","open_spot",
    "buy_up_px","buy_up_size","sell_up_px","sell_up_size","buy_down_px","buy_down_size","sell_down_px","sell_down_size",
]


def local_dt(start: int):
    return datetime.fromtimestamp(int(start), tz=timezone.utc).astimezone(NY)


def slug_candidates(start: int) -> list[str]:
    d = local_dt(start)
    month = d.strftime("%B").lower(); h = d.hour % 12 or 12; ap = "am" if d.hour < 12 else "pm"
    year_explicit = [
        f"bitcoin-up-or-down-{month}-{d.day}-{d.year}-{h}{ap}-et",
        f"bitcoin-up-or-down-{month}-{d.day}-{d.year}-{h}-{ap}-et",
    ]
    yearless = [
        f"bitcoin-up-or-down-{month}-{d.day}-{h}{ap}-et",
        f"bitcoin-up-or-down-{month}-{d.day}-{h}-{ap}-et",
    ]
    # 2026 collides with 2025 yearless slugs, so prefer explicit year from 2026 onward.
    ordered = year_explicit + yearless if d.year >= 2026 else yearless + year_explicit
    ordered.append(f"btc-updown-1h-{int(start)}")
    return list(dict.fromkeys(ordered))


def event_years(event: dict) -> set[int]:
    years: set[int] = set()
    objs = [event] + [x for x in (event.get("markets") or []) if isinstance(x, dict)]
    keys = ("startDate","endDate","startDateIso","endDateIso","createdAt","updatedAt","closedTime","startTime","endTime")
    for obj in objs:
        for k in keys:
            v = obj.get(k)
            if v in (None, ""): continue
            try:
                t = pd.Timestamp(v)
                if t.tzinfo is None: t = t.tz_localize("UTC")
                years.add(int(t.tz_convert(NY).year))
            except Exception:
                pass
    return years


def fetch_hour_market_validated(start: int):
    sess = requests.Session(); sess.headers.update({"User-Agent":"main-sequence-v5-hourly/1.0"})
    expected_year = local_dt(start).year
    errors=[]; wrong_year=[]
    for slug in slug_candidates(start):
        try:
            js = v4.get_json(sess, GAMMA + "/events", params={"slug":slug,"closed":"true","limit":5}, tries=4)
            for event in js or []:
                yrs = event_years(event)
                if yrs and expected_year not in yrs:
                    wrong_year.append({"slug":slug,"years":sorted(yrs)}); continue
                m = v4.parse_event(event, int(start), slug)
                if m is not None:
                    return m, {"start":int(start),"event_slug":slug,"mapped":True,"market_slug":m.slug,
                               "condition_id":m.condition_id,"event_years":"|".join(map(str,sorted(yrs))),"wrong_year_rejects":json.dumps(wrong_year[:4])}
        except Exception as exc:
            errors.append(repr(exc))
    return None, {"start":int(start),"event_slug":None,"mapped":False,"market_slug":None,"condition_id":None,
                  "event_years":None,"wrong_year_rejects":json.dumps(wrong_year[:4]),"errors":" | ".join(errors[:3])}


def discover(start: str, end: str, workers: int=20):
    s0=int(pd.Timestamp(start,tz="UTC").timestamp()); s1=int(pd.Timestamp(end,tz="UTC").timestamp())
    starts=list(range(s0,s1,3600)); markets=[]; inv=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(fetch_hour_market_validated,s):s for s in starts}
        for i,f in enumerate(as_completed(fut),1):
            m,row=f.result(); inv.append(row)
            if m is not None: markets.append(m)
            if i%120==0: print("DISCOVER_V5",i,"/",len(starts),"mapped",len(markets),flush=True)
    markets.sort(key=lambda x:x.start)
    return markets,pd.DataFrame(inv).sort_values("start",kind="mergesort")


def pack_level(qsec: pd.DataFrame, action: str, outcome: str):
    lv=v4.top_level(qsec,action,outcome)
    return (math.nan,0.0) if lv is None else (float(lv[0]),float(lv[1]))


def score_market_state(m, spot, bn, der):
    g=v4.market_tape(m)
    if g is None or g.empty: return []
    rows=[]
    for sec in sorted(int(x) for x in g["timestamp"].dropna().unique().tolist() if m.start<=int(x)<m.close):
        fb=v4.fair_boundary(m,sec,spot,bn,der)
        if fb is None: continue
        q=g[g["timestamp"]==sec]
        bu=pack_level(q,"BUY","up"); su=pack_level(q,"SELL","up"); bd=pack_level(q,"BUY","down"); sd=pack_level(q,"SELL","down")
        if not any(math.isfinite(x) for x in (bu[0],su[0],bd[0],sd[0])): continue
        rows.append({
            "start":m.start,"close":m.close,"slug":m.slug,"event_slug":m.event_slug,"condition_id":m.condition_id,"label_up":m.label_up,
            "fee_enabled":m.fee_enabled,"fee_type":m.fee_type,"fee_rate":m.fee_rate,"fee_exponent":m.fee_exponent,"fee_source":m.fee_source,
            "sec":sec,"fair_up":fb["up"],"fair_down":fb["down"],"p_rv":fb["p_rv"],"p_iv":fb["p_iv"],"rv":fb["rv"],"iv":fb["iv"],"spot":fb["spot"],"open_spot":fb["open"],
            "buy_up_px":bu[0],"buy_up_size":bu[1],"sell_up_px":su[0],"sell_up_size":su[1],
            "buy_down_px":bd[0],"buy_down_size":bd[1],"sell_down_px":sd[0],"sell_down_size":sd[1],
        })
    return rows


def score_range(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True,exist_ok=True)
    markets,inv=discover(start,end,workers=min(24,max(4,workers*2)))
    inv.to_csv(out/"inventory.csv",index=False)
    if not markets:
        pd.DataFrame(columns=STATE_COLS).to_csv(out/"state_rows.csv",index=False)
        summary={"period":[start,end],"markets_expected":len(inv),"markets_mapped":0,"markets_scored":0,"state_rows":0,"protocol":PROTOCOL}
        (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); return
    bn,der,anchor_meta=base.build_anchors(start,end,out)
    spot=base.load_binance_1s(start,end,out/"binance_1s_cache")
    rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(score_market_state,m,spot,bn,der):m for m in markets}
        for i,f in enumerate(as_completed(fut),1):
            m=fut[f]
            try: rows.extend(f.result())
            except Exception as exc: failures.append({"slug":m.slug,"error":repr(exc)})
            if i%24==0: print("STATE_V5",i,"/",len(markets),"rows",len(rows),"fail",len(failures),flush=True)
    if failures: raise RuntimeError(f"state failures {failures[:10]} count={len(failures)}")
    sdf=pd.DataFrame(rows,columns=STATE_COLS)
    if len(sdf): sdf=sdf.sort_values(["start","sec"],kind="mergesort")
    sdf.to_csv(out/"state_rows.csv",index=False)
    summary={"period":[start,end],"markets_expected":int(len(inv)),"markets_mapped":int(inv["mapped"].sum()),
             "markets_scored":int(sdf["start"].nunique()) if len(sdf) else 0,"state_rows":int(len(sdf)),"anchor_meta":anchor_meta,"protocol":PROTOCOL}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)


def market_from_row(r):
    def fnum(x):
        try:
            v=float(x); return v if math.isfinite(v) else None
        except Exception: return None
    return v4.HourMarket(slug=str(r.slug),event_slug=str(r.event_slug),start=int(r.start),condition_id=str(r.condition_id),label_up=float(r.label_up),
                         fee_enabled=bool(r.fee_enabled),fee_type="" if pd.isna(r.fee_type) else str(r.fee_type),fee_rate=fnum(r.fee_rate),
                         fee_exponent=fnum(r.fee_exponent),fee_source="" if pd.isna(r.fee_source) else str(r.fee_source))


def ticket_for_equity(eq: float):
    if not (eq>0 and math.isfinite(eq)): return None
    exp=int(math.floor(math.log(eq/INITIAL_EQUITY,2.0)))
    ticket=BASE_TICKET*(2.0**exp)
    ticket=min(ticket,MAX_TICKET_USD)
    # Preserve prior symmetric downshift; stop only if the account is effectively exhausted.
    if ticket<0.0390625: return None
    return float(ticket)


def level_from_row(r, action: str, outcome: str):
    p=getattr(r,f"{'buy' if action=='BUY' else 'sell'}_{outcome}_px")
    s=getattr(r,f"{'buy' if action=='BUY' else 'sell'}_{outcome}_size")
    try: p=float(p); s=float(s)
    except Exception: return None
    return (p,s) if math.isfinite(p) and s>0 else None


def fair_from_row(r,outcome:str): return float(r.fair_up if outcome=="up" else r.fair_down)


def simulate_window(state: pd.DataFrame, inv: pd.DataFrame, start: str, end: str):
    s0=int(pd.Timestamp(start,tz="UTC").timestamp()); s1=int(pd.Timestamp(end,tz="UTC").timestamp())
    df=state[(state.start>=s0)&(state.start<s1)].sort_values(["start","sec"],kind="mergesort")
    iv=inv[(inv.start>=s0)&(inv.start<s1)].copy() if len(inv) else inv
    eq=INITIAL_EQUITY; peak=eq; maxdd=0.0
    entry_capital=0.0; gross_turnover=0.0; core_pnl=0.0; tail_pnl=0.0
    core_entries=core_conv=core_stops=core_settle=tail_entries=tail_wins=0; cap_rejects=0; max_ticket=0.0; max_open=0.0
    events=[]
    for start_ts,g in df.groupby("start",sort=True):
        g=g.sort_values("sec",kind="mergesort"); first=g.iloc[0]; m=market_from_row(first)
        core=None; tail=None; tail_blocks=False; reentry_after=m.start
        def open_cap(): return (float(core["cost"]) if core else 0.0)+(float(tail["cost"]) if tail else 0.0)
        for r in g.itertuples(index=False):
            sec=int(r.sec)
            # Existing CORE: convergence profit first, otherwise symmetric expected-profit stop.
            if core is not None:
                lv=level_from_row(r,"SELL",core["outcome"])
                if lv is not None:
                    bid,avail=lv; qty=float(core["qty"])
                    if avail+1e-12>=qty:
                        proceeds=qty*bid-v4.fee_total(m,bid,qty); pnl=proceeds-float(core["cost"])
                        fair=fair_from_row(r,core["outcome"])
                        reason=None
                        if bid>=fair-FAIR_BAND-1e-12 and pnl>1e-12: reason="convergence"
                        elif -pnl+1e-12>=float(core["target_profit"]): reason="symmetric_stop"
                        if reason:
                            eq+=pnl; core_pnl+=pnl; gross_turnover+=proceeds; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1.0)
                            if reason=="convergence": core_conv+=1
                            else: core_stops+=1
                            events.append({"time":sec,"equity":eq,"family":"core","event":reason,"pnl":pnl,"ticket":core["ticket"],"target_profit":core["target_profit"]})
                            core=None; reentry_after=sec+1
            # TAIL priority.
            if tail is None:
                ticket=ticket_for_equity(eq)
                if ticket is not None:
                    cand=[]
                    for outcome in ("up","down"):
                        lv=level_from_row(r,"BUY",outcome)
                        if lv is None: continue
                        ask,avail=lv; qty=v4.qty_for_budget(m,ask,ticket)
                        if qty<=0 or avail+1e-12<qty: continue
                        fair=fair_from_row(r,outcome); cost=qty*ask+v4.fee_total(m,ask,qty)
                        exp_profit=qty*fair-cost
                        if fair>=TAIL_FAVORITE_FAIR and exp_profit>0: cand.append((exp_profit,outcome,ask,qty,cost,fair))
                    if cand:
                        exp_profit,outcome,ask,qty,cost,fair=max(cand,key=lambda x:x[0])
                        if open_cap()+cost<=min(MAX_MARKET_OPEN_USD,eq)+1e-9:
                            tail={"outcome":outcome,"price":ask,"qty":qty,"cost":cost,"entry_time":sec,"ticket":ticket}
                            tail_entries+=1; entry_capital+=cost; gross_turnover+=cost; tail_blocks=True; max_ticket=max(max_ticket,ticket); max_open=max(max_open,open_cap())
                            events.append({"time":sec,"equity":eq,"family":"tail","event":"entry","pnl":0.0,"ticket":ticket})
                        else: cap_rejects+=1
            # Repeatable CORE.
            if core is None and not tail_blocks and sec>=reentry_after:
                ticket=ticket_for_equity(eq)
                if ticket is not None:
                    cand=[]
                    for outcome in ("up","down"):
                        lv=level_from_row(r,"BUY",outcome)
                        if lv is None: continue
                        ask,avail=lv; qty=v4.qty_for_budget(m,ask,ticket)
                        if qty<=0 or avail+1e-12<qty: continue
                        fair=fair_from_row(r,outcome); cost=qty*ask+v4.fee_total(m,ask,qty)
                        target_proceeds=qty*fair-v4.fee_total(m,min(max(fair,1e-6),1-1e-6),qty)
                        target_profit=target_proceeds-cost
                        if target_profit>1e-12: cand.append((target_profit,outcome,ask,qty,cost,fair))
                    if cand:
                        target_profit,outcome,ask,qty,cost,fair=max(cand,key=lambda x:x[0])
                        if open_cap()+cost<=min(MAX_MARKET_OPEN_USD,eq)+1e-9:
                            core={"outcome":outcome,"price":ask,"qty":qty,"cost":cost,"entry_time":sec,"ticket":ticket,"entry_fair":fair,"target_profit":target_profit}
                            core_entries+=1; entry_capital+=cost; gross_turnover+=cost; max_ticket=max(max_ticket,ticket); max_open=max(max_open,open_cap())
                            events.append({"time":sec,"equity":eq,"family":"core","event":"entry","pnl":0.0,"ticket":ticket,"target_profit":target_profit})
                        else: cap_rejects+=1
        # Settlement only for positions which never got an executable convergence/stop before close.
        won_up=m.label_up>=0.5
        if core is not None:
            won=won_up if core["outcome"]=="up" else (not won_up); payout=float(core["qty"]) if won else 0.0; pnl=payout-float(core["cost"])
            eq+=pnl; core_pnl+=pnl; core_settle+=1; peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1.0)
            events.append({"time":m.close,"equity":eq,"family":"core","event":"settlement_fallback","pnl":pnl,"ticket":core["ticket"],"target_profit":core["target_profit"]}); core=None
        if tail is not None:
            won=won_up if tail["outcome"]=="up" else (not won_up); payout=float(tail["qty"]) if won else 0.0; pnl=payout-float(tail["cost"])
            eq+=pnl; tail_pnl+=pnl; tail_wins+=int(won); peak=max(peak,eq); maxdd=min(maxdd,eq/peak-1.0)
            events.append({"time":m.close,"equity":eq,"family":"tail","event":"settlement","pnl":pnl,"ticket":tail["ticket"]}); tail=None
        if eq<=0: break
    days=(pd.Timestamp(end,tz="UTC")-pd.Timestamp(start,tz="UTC")).total_seconds()/86400.0
    ret=eq/INITIAL_EQUITY-1.0; cagr=(eq/INITIAL_EQUITY)**(365.0/days)-1.0 if eq>0 else math.nan
    pnl=eq-INITIAL_EQUITY
    mapped=int(iv.mapped.fillna(False).astype(bool).sum()) if len(iv) else 0; expected=int(len(iv))
    return {
        "period":[start,end],"days":days,"initial_equity":INITIAL_EQUITY,"final_equity":eq,"total_return":ret,"calendar_cagr":cagr,
        "realized_max_dd":maxdd,"total_pnl":pnl,"entry_capital":entry_capital,"pnl_over_entry_capital":pnl/entry_capital if entry_capital>0 else math.nan,
        "gross_turnover":gross_turnover,"annualized_entry_turnover_x_initial":(entry_capital/INITIAL_EQUITY)*(365.0/days),
        "core_pnl":core_pnl,"tail_pnl":tail_pnl,"core_entries":core_entries,"core_convergence_exits":core_conv,"core_symmetric_stops":core_stops,
        "core_settlement_fallbacks":core_settle,"tail_entries":tail_entries,"tail_wins":tail_wins,"cap_rejects":cap_rejects,"max_ticket_used":max_ticket,
        "max_simultaneous_open_capital":max_open,"markets_expected":expected,"markets_mapped":mapped,"market_mapping_coverage":mapped/expected if expected else math.nan,
        "state_markets":int(df.start.nunique()) if len(df) else 0,"events":events,
    }


def aggregate(root: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    sf=sorted(root.rglob("state_rows.csv")); inf=sorted(root.rglob("inventory.csv"))
    if not sf: raise RuntimeError("no state_rows.csv")
    state=pd.concat([pd.read_csv(p) for p in sf],ignore_index=True).drop_duplicates(["condition_id","sec"],keep="last")
    inv=pd.concat([pd.read_csv(p) for p in inf],ignore_index=True).drop_duplicates(["start"],keep="last") if inf else pd.DataFrame()
    state=state.sort_values(["start","sec"],kind="mergesort")
    results={}
    all_events=[]
    for name,(start,end) in WINDOWS.items():
        r=simulate_window(state,inv,start,end); all_events.extend([{**e,"window":name} for e in r.pop("events")]); results[name]=r
    summary={"protocol":PROTOCOL,"windows":results}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    pd.DataFrame(all_events).to_csv(out/"events.csv",index=False)
    state.to_csv(out/"all_state_rows.csv",index=False)
    inv.to_csv(out/"all_inventory.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("protocol"); p.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("score"); s.add_argument("--start",required=True); s.add_argument("--end",required=True); s.add_argument("--out",type=Path,required=True); s.add_argument("--workers",type=int,default=10)
    a=sub.add_parser("aggregate"); a.add_argument("--root",type=Path,required=True); a.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.cmd=="protocol":
        args.out.mkdir(parents=True,exist_ok=True); (args.out/"FROZEN_V5_PROTOCOL.json").write_text(json.dumps(PROTOCOL,indent=2)); print(json.dumps(PROTOCOL,indent=2))
    elif args.cmd=="score": score_range(args.start,args.end,args.out,args.workers)
    else: aggregate(args.root,args.out)

if __name__=="__main__": main()
