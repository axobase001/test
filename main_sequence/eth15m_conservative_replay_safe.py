from __future__ import annotations

import math
import pandas as pd
import numpy as np
import requests

from main_sequence import eth15m_conservative_replay as r
from main_sequence import final_recent_replay as base


def exact_level(qsec, action, outcome, market, need_mult=1.0, fixed_qty=None):
    z = qsec[(qsec.side_u == action) & (qsec.outcome_l == outcome) & (qsec['size'] > 0)].copy()
    if z.empty:
        return None
    prices = sorted(z.price.astype(float).unique(), reverse=(action == 'SELL'))
    for p in prices:
        if action == 'BUY' and p < r.ASK_FLOOR:
            continue
        q = fixed_qty if fixed_qty is not None else r.qty_for_budget(market, p)
        size = float(z[np.isclose(z.price.astype(float), p, rtol=0, atol=1e-12)]['size'].sum())
        if size + 1e-12 >= need_mult * q:
            return float(p), size, float(q)
    return None


def score_market(m, g, spot, bn, der):
    if g is None or g.empty:
        return None, None
    med_rv = []
    secs = sorted(int(x) for x in g.timestamp.dropna().unique() if m.start <= int(x) < m.close)
    entry = None
    for sec in secs:
        fb = r.fair_boundary(m, sec, spot, bn, der)
        if fb:
            med_rv.append(fb['rv'])
        if sec < m.close-r.MAX_S2C or sec > m.close-r.MIN_S2C or fb is None:
            continue
        qsec = g[g.timestamp == sec]
        cand = []
        for o in ('up', 'down'):
            lv = exact_level(qsec, 'BUY', o, m, r.VOLUME_MULT)
            if lv is None:
                continue
            p, size, q = lv
            cost = q*p + r.fee_total(m, p, q)
            fair = float(fb[o])
            raw = fair-p
            edge = (q*fair-r.fee_total(m, min(max(fair,1e-6),1-1e-6), q)-cost)/q
            if raw >= r.RAW_GAP_FLOOR and edge >= r.EDGE_FLOOR:
                cand.append((edge,o,p,q,cost,fair,raw))
        if cand:
            entry = max(cand, key=lambda x:x[0])
            entry_sec = sec
            break
    market_rv = float(np.median(med_rv)) if med_rv else math.nan
    if entry is None:
        return None, market_rv
    edge,o,p,q,cost,fair,raw = entry
    won = ((m.label_up >= .5) == (o == 'up'))
    exit_kind = 'settlement_fallback'
    exit_sec = m.close
    proceeds = q if won else 0.0
    for sec in [x for x in secs if x > entry_sec]:
        fb = r.fair_boundary(m, sec, spot, bn, der)
        if fb is None:
            continue
        lv = exact_level(g[g.timestamp == sec], 'SELL', o, m, r.VOLUME_MULT, fixed_qty=q)
        if lv is None:
            continue
        bid,size,_ = lv
        if bid >= float(fb[o])-r.EXIT_BAND-1e-12:
            net = q*bid-r.fee_total(m,bid,q)
            if net > cost:
                proceeds = net
                exit_sec = sec
                exit_kind = 'convergence'
                break
    return {
        'slug':m.slug,'condition_id':m.condition_id,'start':m.start,
        'decision':entry_sec,'exit':exit_sec,'exit_kind':exit_kind,'side':o,
        'entry_px':p,'qty':q,'cost':cost,'pnl':proceeds-cost,
        'raw_gap':raw,'edge_share':edge,
        'entry_rv':r.fair_boundary(m,entry_sec,spot,bn,der)['rv']
    }, market_rv


def score_hour(hour, markets, spot, bn, der):
    if not markets:
        return [], []
    sess = requests.Session()
    bmarkets = [base.Market(m.slug,m.start,m.condition_id,m.label_up,m.fee_enabled,m.fee_type,m.fee_rate,m.fee_exponent,m.fee_source) for m in markets]
    raw = base.query_trade_rows(sess,bmarkets,hour,hour+3599)
    tm = base.normalize_trades(raw)
    rec=[]; rv=[]
    for m in markets:
        z,mrv = score_market(m,tm.get(m.condition_id,pd.DataFrame()),spot,bn,der)
        rv.append({'start':m.start,'market_rv':mrv})
        if z:
            rec.append(z)
    return rec,rv

r.score_hour = score_hour

if __name__ == '__main__':
    r.main()
