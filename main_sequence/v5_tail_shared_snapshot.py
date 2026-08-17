from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from main_sequence import final_recent_replay as base
from main_sequence import v5_hourly_symmetric_stop as v5
from main_sequence import v5_tail_only_fast as fast


def make_snapshot(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = v5.discover(start, end, workers=min(16, max(4, workers)))
    markets = sorted(markets, key=lambda m: m.start)
    inv.to_csv(out / "inventory.csv", index=False)
    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")
    payload = {"start": start, "end": end, "markets": markets, "inventory": inv, "bn": bn, "der": der, "spot": spot, "anchor_meta": anchor_meta}
    with (out / "snapshot.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"period": [start, end], "markets_expected": int(len(inv)), "markets_mapped": int(inv["mapped"].fillna(False).astype(bool).sum()), "markets": len(markets), "anchor_meta": anchor_meta, "invariant": "single shared snapshot for all scoring shards"}
    (out / "snapshot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


def score_snapshot(snapshot: Path, shard: int, nshards: int, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    with snapshot.open("rb") as f:
        p = pickle.load(f)
    markets = p["markets"]; inv = p["inventory"]; bn = p["bn"]; der = p["der"]; spot = p["spot"]
    s0 = int(pd.Timestamp(p["start"], tz="UTC").timestamp())
    selected = [m for m in markets if ((int(m.start) - s0) // 3600) % nshards == shard]
    rows=[]; failed=[]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut={ex.submit(fast.candidate_rows_for_market,m,spot,bn,der):m for m in selected}
        for i,f in enumerate(as_completed(fut),1):
            m=fut[f]
            try: rows.extend(f.result())
            except Exception as exc: failed.append((m,repr(exc)))
            if i%24==0 or i==len(selected): print("SHARED_SNAPSHOT",shard,i,"/",len(selected),"rows",len(rows),"fail",len(failed),flush=True)
    pending=failed; retry=[]
    for round_no,sleep_s in enumerate((5,10,20,40),1):
        if not pending: break
        time.sleep(sleep_s); nxt=[]
        for m,err in pending:
            try:
                rows.extend(fast.candidate_rows_for_market(m,spot,bn,der)); retry.append({"slug":m.slug,"round":round_no,"recovered":True,"prior_error":err})
            except Exception as exc:
                nxt.append((m,repr(exc))); retry.append({"slug":m.slug,"round":round_no,"recovered":False,"error":repr(exc)})
        pending=nxt
    if pending:
        (out/"failures.json").write_text(json.dumps([{"slug":m.slug,"condition_id":m.condition_id,"error":e} for m,e in pending],indent=2),encoding="utf-8")
        raise RuntimeError(f"unresolved failures={len(pending)}")
    df=pd.DataFrame(rows,columns=fast.CANDIDATE_COLS)
    if len(df): df=df.drop_duplicates(["condition_id","sec","outcome"],keep="last").sort_values(["start","sec","outcome"],kind="mergesort")
    df.to_csv(out/"tail_candidates.csv",index=False)
    (out/"retry_history.json").write_text(json.dumps(retry,indent=2),encoding="utf-8")
    meta={"shard":shard,"nshards":nshards,"selected_markets":len(selected),"candidate_rows":len(df),"anchor_meta":p["anchor_meta"],"snapshot_period":[p["start"],p["end"]]}
    (out/"summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("snapshot"); s.add_argument("--start",required=True); s.add_argument("--end",required=True); s.add_argument("--out",type=Path,required=True); s.add_argument("--workers",type=int,default=12)
    q=sub.add_parser("score"); q.add_argument("--snapshot",type=Path,required=True); q.add_argument("--shard",type=int,required=True); q.add_argument("--nshards",type=int,required=True); q.add_argument("--out",type=Path,required=True); q.add_argument("--workers",type=int,default=6)
    a=ap.parse_args()
    if a.cmd=="snapshot": make_snapshot(a.start,a.end,a.out,a.workers)
    else: score_snapshot(a.snapshot,a.shard,a.nshards,a.out,a.workers)

if __name__=="__main__": main()
