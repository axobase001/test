from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from main_sequence import final_recent_replay as base
from main_sequence import v4_hourly_core_tail as v4
from main_sequence import v5_hourly_symmetric_stop as v5

STATE_COLS = v5.STATE_COLS
TRADE_PAGE = 10_000
BATCH_MARKETS = 8


def query_trade_rows_10k(sess: requests.Session, markets, start: int, end: int) -> list[dict]:
    if not markets or end < start:
        return []
    q = {
        "market": ",".join(m.condition_id for m in markets),
        "start": int(start), "end": int(end), "limit": TRADE_PAGE,
        "offset": 0, "takerOnly": "true",
    }
    rows = base.get_json(sess, base.DATA_API + "/trades", params=q, timeout=60)
    if len(rows) < TRADE_PAGE:
        return rows
    if len(markets) > 1:
        mid = len(markets) // 2
        return query_trade_rows_10k(sess, markets[:mid], start, end) + query_trade_rows_10k(sess, markets[mid:], start, end)
    if start < end:
        mid_t = (start + end) // 2
        return query_trade_rows_10k(sess, markets, start, mid_t) + query_trade_rows_10k(sess, markets, mid_t + 1, end)
    q2 = dict(q); q2["offset"] = TRADE_PAGE
    rr = base.get_json(sess, base.DATA_API + "/trades", params=q2, timeout=60)
    out = list(rows) + list(rr)
    if len(rr) < TRADE_PAGE:
        return out
    raise RuntimeError(f"Data API >20k trades in one market-second {markets[0].slug} {start}")


def score_market_from_tape(m, g: pd.DataFrame, spot, bn, der):
    if g is None or g.empty:
        return []
    rows = []
    for sec in sorted(int(x) for x in g["timestamp"].dropna().unique().tolist() if m.start <= int(x) < m.close):
        fb = v4.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        q = g[g["timestamp"] == sec]
        bu = v5.pack_level(q, "BUY", "up"); su = v5.pack_level(q, "SELL", "up")
        bd = v5.pack_level(q, "BUY", "down"); sd = v5.pack_level(q, "SELL", "down")
        if not any(math.isfinite(x) for x in (bu[0], su[0], bd[0], sd[0])):
            continue
        rows.append({
            "start":m.start,"close":m.close,"slug":m.slug,"event_slug":m.event_slug,"condition_id":m.condition_id,"label_up":m.label_up,
            "fee_enabled":m.fee_enabled,"fee_type":m.fee_type,"fee_rate":m.fee_rate,"fee_exponent":m.fee_exponent,"fee_source":m.fee_source,
            "sec":sec,"fair_up":fb["up"],"fair_down":fb["down"],"p_rv":fb["p_rv"],"p_iv":fb["p_iv"],"rv":fb["rv"],"iv":fb["iv"],"spot":fb["spot"],"open_spot":fb["open"],
            "buy_up_px":bu[0],"buy_up_size":bu[1],"sell_up_px":su[0],"sell_up_size":su[1],
            "buy_down_px":bd[0],"buy_down_size":bd[1],"sell_down_px":sd[0],"sell_down_size":sd[1],
        })
    return rows


def fetch_batch(markets, spot, bn, der):
    if not markets:
        return [], 0
    sess = requests.Session(); sess.headers.update({"User-Agent":"main-sequence-v5-hourly-state-batched/1.0"})
    start = min(int(m.start) for m in markets)
    end = max(int(m.close) for m in markets) - 1
    raw = query_trade_rows_10k(sess, list(markets), start, end)
    tm = base.normalize_trades(raw)
    rows = []
    for m in markets:
        rows.extend(score_market_from_tape(m, tm.get(m.condition_id, pd.DataFrame()), spot, bn, der))
    return rows, len(raw)


def score_range(start: str, end: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    markets, inv = v5.discover(start, end, workers=min(24, max(4, workers * 2)))
    inv.to_csv(out / "inventory.csv", index=False)
    if not markets:
        pd.DataFrame(columns=STATE_COLS).to_csv(out / "state_rows.csv", index=False)
        return
    bn, der, anchor_meta = base.build_anchors(start, end, out)
    spot = base.load_binance_1s(start, end, out / "binance_1s_cache")
    batches = [markets[i:i+BATCH_MARKETS] for i in range(0, len(markets), BATCH_MARKETS)]
    rows = []; raw_rows = 0; failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(fetch_batch, b, spot, bn, der): (i, b) for i, b in enumerate(batches)}
        for k, f in enumerate(as_completed(fut), 1):
            i, b = fut[f]
            try:
                rr, nr = f.result(); rows.extend(rr); raw_rows += int(nr)
            except Exception as exc:
                failures.append({"batch":i,"start":int(b[0].start),"end":int(b[-1].close),"error":repr(exc)})
            if k % 24 == 0 or k == len(batches):
                print("STATE_BATCH", k, "/", len(batches), "rows", len(rows), "raw", raw_rows, "fail", len(failures), flush=True)
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"state batch failures {len(failures)}")
    sdf = pd.DataFrame(rows, columns=STATE_COLS)
    if len(sdf):
        sdf = sdf.drop_duplicates(["condition_id","sec"], keep="last").sort_values(["start","sec"], kind="mergesort")
    sdf.to_csv(out / "state_rows.csv", index=False)
    summary = {
        "period":[start,end],"markets_expected":int(len(inv)),"markets_mapped":int(inv["mapped"].sum()),
        "markets_scored":int(sdf["start"].nunique()) if len(sdf) else 0,"state_rows":int(len(sdf)),
        "raw_trade_rows":int(raw_rows),"trade_page_limit":TRADE_PAGE,"batch_markets":BATCH_MARKETS,
        "anchor_meta":anchor_meta,"protocol":v5.PROTOCOL,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "protocol"}, indent=2), flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",required=True); ap.add_argument("--end",required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--workers",type=int,default=4)
    a=ap.parse_args(); score_range(a.start,a.end,a.out,a.workers)

if __name__ == "__main__": main()
