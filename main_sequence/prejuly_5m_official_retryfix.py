from __future__ import annotations

import time
import requests

import prejuly_5m_official as core


def fetch_hour_fixed(hour_start: int):
    """Same data contract as core.fetch_hour; only hardens Data API batching.

    Empirical June probe shows 24-condition BTC+ETH requests can return a
    deterministic HTTP 500 while smaller batches succeed. On 500 we bisect
    the exact same condition-id set; no dates, rows, labels, features, policy
    constants, or model code are changed.
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-sealed-research/1.0"})
    wanted = []
    for a in core.ASSETS:
        for t0 in range(hour_start, hour_start + 3600, 300):
            wanted.append((f"{a.lower()}-updown-5m-{t0}", a, t0))

    params = [("slug", x[0]) for x in wanted] + [("closed", "true"), ("limit", 100)]
    js = core.get_json(sess, core.GAMMA + "/markets", params=params)
    byslug = {str(x.get("slug")): x for x in js}
    markets = []
    for slug, asset, t0 in wanted:
        m = core.parse_market(byslug.get(slug, {}), asset, t0)
        if m is not None:
            markets.append(m)
    if not markets:
        return [], []

    def pull_group(group):
        conds = ",".join(m.condition_id for m in group)
        q = {
            "market": conds,
            "start": hour_start,
            "end": hour_start + 3600,
            "limit": 10000,
            "offset": 0,
            "takerOnly": "true",
        }
        # One direct attempt lets us detect the deterministic batch-size 500
        # immediately instead of burning core.get_json's full retry schedule.
        try:
            r = sess.get(core.DATA_API + "/trades", params=q, timeout=60)
        except requests.RequestException:
            r = None

        if r is not None and r.status_code == 200:
            rows = r.json()
        elif r is not None and r.status_code == 500 and len(group) > 1:
            mid = len(group) // 2
            return pull_group(group[:mid]) + pull_group(group[mid:])
        else:
            # Preserve the original retry semantics for transient failures,
            # throttling, and irreducible single-market server errors.
            rows = core.get_json(sess, core.DATA_API + "/trades", params=q, timeout=60)

        if len(rows) >= 10000 and len(group) > 1:
            mid = len(group) // 2
            return pull_group(group[:mid]) + pull_group(group[mid:])
        if len(rows) >= 10000:
            raise RuntimeError(f"trade cap hit for {group[0].slug}")
        return rows

    trades = []
    for i in range(0, len(markets), 24):
        trades.extend(pull_group(markets[i:i + 24]))
    return markets, trades


core.fetch_hour = fetch_hour_fixed

if __name__ == "__main__":
    core.main()
