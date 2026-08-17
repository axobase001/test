from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Importing this module installs the causal Deribit-universe builder and paced
# Polymarket transport into final_recent_replay.
from main_sequence import final_recent_no_lookahead as causal
from main_sequence import v5_15m_tail_only_fast as fast

base = causal.base
engine = fast.engine


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def freeze(start: str, end: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    bn, der, meta = base.build_anchors(start, end, out)
    p = out / 'anchors.pkl'
    with p.open('wb') as f:
        pickle.dump({'start': start, 'end': end, 'bn': bn, 'der': der, 'meta': meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        'period': [start, end],
        'anchor_meta': meta,
        'binance_1m_points': int(len(bn.open_times)),
        'deribit_anchor_points': int(len(der.ts)),
        'anchors_pickle_sha256': sha256(p),
        'policy': {
            'fair_floor': fast.TAIL_FAIR_FLOOR,
            'post_fee_edge': '0 < edge < 0.03',
            'execution': 'same-second exact first-level full-size',
            'stake_tiers': list(fast.STAKE_TIERS),
        },
    }
    (out / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


def score(snapshot: Path, start: str, end: str, shard: str, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    with snapshot.open('rb') as f:
        z = pickle.load(f)
    bn, der = z['bn'], z['der']

    hours, by_hour, inventory = base.discover(start, end, workers=min(12, max(4, workers * 2)))
    inventory.to_csv(out / 'market_inventory.csv', index=False)
    theoretical = int(len(inventory))
    exists = inventory['exists_in_gamma'].fillna(False).astype(bool) if theoretical else pd.Series(dtype=bool)
    mapped_mask = inventory['mapped'].fillna(False).astype(bool) if theoretical else pd.Series(dtype=bool)
    tradable_expected = int(exists.sum()) if theoretical else 0
    mapped = int(mapped_mask.sum()) if theoretical else 0
    bad_existing = inventory[exists & ~mapped_mask] if theoretical else inventory
    if len(bad_existing):
        bad_existing.to_csv(out / 'mapping_failures.csv', index=False)
        raise RuntimeError(f'shared 15m fail-closed existing-market mapping {mapped}/{tradable_expected}; bad={len(bad_existing)}')

    markets = []
    for h in sorted(hours):
        markets.extend(sorted(by_hour.get(h, []), key=lambda m: int(m.start)))
    markets = sorted(markets, key=lambda m: int(m.start))
    if len(markets) != mapped:
        raise RuntimeError(f'market object count mismatch {len(markets)} vs mapped {mapped}')

    spot = base.load_binance_1s(start, end, out / 'binance_1s_cache')
    batches = [markets[i:i + fast.BATCH_MARKETS] for i in range(0, len(markets), fast.BATCH_MARKETS)]
    records = []
    raw_rows = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(fast.fetch_batch, b, spot, bn, der): (i, b) for i, b in enumerate(batches)}
        for k, f in enumerate(as_completed(fut), 1):
            i, b = fut[f]
            try:
                rr, nr = f.result()
                records.extend(rr); raw_rows += int(nr)
            except Exception as exc:
                failures.append({'batch': i, 'start': int(b[0].start) if b else None,
                                 'end': int(b[-1].close) if b else None, 'error': repr(exc)})
            if k % 24 == 0 or k == len(batches):
                print('TAIL15_SHARED', shard, k, '/', len(batches), 'records', len(records), 'raw', raw_rows, 'fail', len(failures), flush=True)
    if failures:
        (out / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
        raise RuntimeError(f'shared 15m batch failures {len(failures)}')

    df = pd.DataFrame(records, columns=engine.RECORD_COLUMNS)
    if len(df):
        df = df.drop_duplicates(['condition_id', 'stake_budget', 'signal_family'], keep='last')
        df = df.sort_values(['start', 'stake_budget', 'decision'], kind='mergesort')
    df.to_csv(out / 'tail_tiered_records.csv', index=False)
    summary = {
        'shard': shard, 'period': [start, end],
        'snapshot_period': [z['start'], z['end']],
        'snapshot_anchor_meta': z['meta'],
        'theoretical_quarter_hours': theoretical,
        'gamma_contracts_existing': tradable_expected,
        'markets_mapped': mapped,
        'existing_market_mapping_coverage': mapped / tradable_expected if tradable_expected else math.nan,
        'nonexistent_quarter_hours': theoretical - tradable_expected,
        'tail_records': int(len(df)),
        'tail_markets': int(df['start'].nunique()) if len(df) else 0,
        'raw_trade_rows': int(raw_rows),
        'trade_page_limit': fast.TRADE_PAGE,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--start', required=True); f.add_argument('--end', required=True); f.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('score'); s.add_argument('--snapshot', type=Path, required=True); s.add_argument('--start', required=True); s.add_argument('--end', required=True); s.add_argument('--shard', required=True); s.add_argument('--out', type=Path, required=True); s.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.cmd == 'freeze': freeze(a.start, a.end, a.out)
    else: score(a.snapshot, a.start, a.end, a.shard, a.out, a.workers)


if __name__ == '__main__':
    main()
