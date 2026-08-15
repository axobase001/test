from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
HOURS = [1781726400, 1781974800]  # 2026-06-17 20:00Z, 2026-06-20 17:00Z


def get_json(path: str, params):
    r = requests.get(
        GAMMA + path,
        params=params,
        timeout=30,
        headers={"User-Agent": "main-sequence-june-crosscheck/1.0"},
    )
    return r.status_code, r.url, (r.json() if r.status_code == 200 else None)


def site_status(slug: str):
    urls = [
        f"https://polymarket.com/event/{slug}",
        f"https://polymarket.com/market/{slug}",
    ]
    out = []
    for url in urls:
        try:
            r = requests.get(url, timeout=30, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            out.append({"url": url, "status": r.status_code, "final_url": r.url, "bytes": len(r.content)})
        except Exception as e:
            out.append({"url": url, "error": repr(e)})
    return out


def iso(ts: int):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def main():
    for h in HOURS:
        wanted = [(f"{a.lower()}-updown-5m-{t0}", a, t0)
                  for a in ASSETS for t0 in range(h, h + 3600, 300)]
        params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
        st, url, js = get_json("/markets", params)
        assert st == 200
        byslug = {str(x.get("slug")): x for x in js}
        missing = [x for x in wanted if x[0] not in byslug]
        present = [x for x in wanted if x[0] in byslug]
        print("HOUR", json.dumps({
            "hour": h,
            "hour_iso": iso(h),
            "batch_count": len(js),
            "present": len(present),
            "missing": [x[0] for x in missing],
        }, ensure_ascii=False), flush=True)

        # Deterministic sample: first 4 present plus all missing.
        samples = [("present", x) for x in present[:4]] + [("missing", x) for x in missing]
        for kind, (slug, asset, t0) in samples:
            mst, murl, markets = get_json("/markets", [("slug", slug), ("closed", "true"), ("limit", 20)])
            est, eurl, events = get_json("/events", [("slug", slug), ("closed", "true"), ("limit", 20)])
            # Also query events without closed filter because historical indexing can differ.
            east, eaurl, events_any = get_json("/events", [("slug", slug), ("limit", 20)])
            nested = []
            if isinstance(events_any, list):
                for ev in events_any:
                    for m in (ev.get("markets") or []):
                        if str(m.get("slug")) == slug:
                            nested.append({
                                "slug": m.get("slug"),
                                "conditionId": m.get("conditionId"),
                                "closed": m.get("closed"),
                                "active": m.get("active"),
                            })
            batch_obj = byslug.get(slug)
            record = {
                "kind": kind,
                "slug": slug,
                "asset": asset,
                "start_iso": iso(t0),
                "batch_conditionId": None if batch_obj is None else batch_obj.get("conditionId"),
                "market_single": {"status": mst, "count": len(markets) if isinstance(markets, list) else None, "url": murl},
                "event_closed": {"status": est, "count": len(events) if isinstance(events, list) else None, "url": eurl},
                "event_any": {"status": east, "count": len(events_any) if isinstance(events_any, list) else None, "url": eaurl},
                "nested_exact_markets": nested,
                "site": site_status(slug),
            }
            print("CROSSCHECK", json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
