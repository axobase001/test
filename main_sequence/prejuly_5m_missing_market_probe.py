from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
START = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())


def get(sess: requests.Session, path: str, params):
    r = sess.get(GAMMA + path, params=params, timeout=30)
    print("HTTP", r.status_code, r.url, flush=True) if r.status_code != 200 else None
    r.raise_for_status()
    return r.json()


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def main():
    sess = requests.Session()
    sess.headers.update({"User-Agent": "main-sequence-june-missing-audit/1.0"})

    expected = []
    batch_seen = set()
    batch_missing_by_hour = []

    for i, h in enumerate(range(START, END, 3600), 1):
        wanted = [(f"{a.lower()}-updown-5m-{t0}", a, t0)
                  for a in ASSETS for t0 in range(h, h + 3600, 300)]
        expected.extend(wanted)
        params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
        js = get(sess, "/markets", params)
        slugs = {str(x.get("slug")) for x in js}
        batch_seen |= slugs
        miss = [(slug, a, t0) for slug, a, t0 in wanted if slug not in slugs]
        if miss:
            batch_missing_by_hour.append({
                "hour": h,
                "hour_iso": iso(h),
                "missing": [x[0] for x in miss],
                "returned": len(js),
            })
        if i % 120 == 0:
            print("AUDIT_HOURS", i, "/", (END - START) // 3600, "batch_missing", sum(len(x["missing"]) for x in batch_missing_by_hour), flush=True)

    missing = [(slug, a, t0) for slug, a, t0 in expected if slug not in batch_seen]
    print("BATCH_SUMMARY", json.dumps({
        "expected": len(expected),
        "seen": len(batch_seen),
        "missing": len(missing),
        "missing_by_hour": batch_missing_by_hour,
    }, ensure_ascii=False, indent=2), flush=True)

    details = []
    for slug, asset, t0 in missing:
        q_any = get(sess, "/markets", [("slug", slug), ("limit", 20)])
        q_closed = get(sess, "/markets", [("slug", slug), ("closed", "true"), ("limit", 20)])
        q_open = get(sess, "/markets", [("slug", slug), ("closed", "false"), ("limit", 20)])
        # Check nearest canonical 5m slugs for the same asset to distinguish a genuine listing gap
        neighbors = []
        for dt in (-600, -300, 300, 600):
            nslug = f"{asset.lower()}-updown-5m-{t0 + dt}"
            qn = get(sess, "/markets", [("slug", nslug), ("limit", 5)])
            neighbors.append({"dt": dt, "slug": nslug, "count": len(qn)})
        d = {
            "slug": slug,
            "asset": asset,
            "start": t0,
            "start_iso": iso(t0),
            "single_any_count": len(q_any),
            "single_closed_count": len(q_closed),
            "single_open_count": len(q_open),
            "single_any": [{
                "slug": x.get("slug"),
                "conditionId": x.get("conditionId"),
                "closed": x.get("closed"),
                "active": x.get("active"),
                "archived": x.get("archived"),
                "question": x.get("question"),
            } for x in q_any],
            "neighbors": neighbors,
        }
        details.append(d)
        print("MISSING_DETAIL", json.dumps(d, ensure_ascii=False), flush=True)
        time.sleep(0.03)

    recovered = [d for d in details if d["single_any_count"] > 0]
    genuine = [d for d in details if d["single_any_count"] == 0]
    print("FINAL", json.dumps({
        "missing_total": len(details),
        "recovered_by_single_query": len(recovered),
        "genuine_absent_by_slug": len(genuine),
        "recovered_slugs": [d["slug"] for d in recovered],
        "genuine_absent_slugs": [d["slug"] for d in genuine],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
