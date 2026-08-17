from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd

from main_sequence import final_recent_no_lookahead as causal
from main_sequence import original_93d_anchor_freeze as af
from main_sequence import v5_15m_tail_only_fast as fast


def fast_fetch_deribit(instruments: list[str], start: date, end: date, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_d = end + timedelta(days=1)
    end_ms = int(datetime(end_d.year, end_d.month, end_d.day, tzinfo=timezone.utc).timestamp() * 1000) - 1
    by_name = {}
    # Transport-only acceleration. _fetch_one and all filtering/pagination semantics
    # are the frozen implementation; only independent instruments run wider.
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(af._fetch_one, name, start_ms, end_ms): name for name in instruments}
        for k, f in enumerate(as_completed(futs), 1):
            name = futs[f]
            by_name[name] = f.result()
            if k % 50 == 0 or k == len(futs):
                print('DERIBIT_FAST24', k, '/', len(futs), 'rows', sum(len(x) for x in by_name.values()), flush=True)
    all_rows = [r for name in instruments for r in by_name.get(name, [])]
    if not all_rows:
        raise RuntimeError('No Deribit option trades returned')
    df = pd.DataFrame(all_rows)
    for c in ['block_trade_id','block_rfq_id','combo_id','combo_trade_id']:
        if c in df.columns:
            df = df[df[c].isna()]
    for c in ['timestamp','iv','index_price','price','trade_seq']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['timestamp','iv','index_price','instrument_name'])
    df = df[(df['iv'] > 0) & (df['index_price'] > 0)].copy()
    df['timestamp'] = df['timestamp'].astype('int64')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df

# build_anchors_causal resolves this global dynamically.
causal.fetch_deribit_trades_parallel = fast_fetch_deribit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    ap.add_argument('--shard', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    fast.score_range(a.start, a.end, a.shard, a.out, a.workers)

if __name__ == '__main__':
    main()
