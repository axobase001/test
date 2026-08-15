from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
# Prior complete-run counters localize all 19 missing theoretical listings to this window.
START = int(datetime(2026, 6, 17, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp())


def get(path: str, params):
    r = requests.get(
        GAMMA + path,
        params=params,
        timeout=30,
        headers={"User-Agent": "main-sequence-june-missing-audit/2.0"},
    )
    r.raise_for_status()
    return r.json()


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def audit_hour(h: int):
    wanted = [(f"{a.lower()}-updown-5m-{t0}", a, t0)
              for a in ASSETS for t0 in range(h, h + 3600, 300)]
    params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
    js = get("/markets", params)
    slugs = {str(x.get("slug")) for x in js}
    miss = [(slug, a, t0) for slug, a, t0 in wanted if slug not in slugs]
    return h, wanted, slugs, miss, len(js)


def audit_missing(item):
    slug, asset, t0 = item
    q_any = get("/markets", [("slug", slug), ("limit", 20)])
    q_closed = get("/markets", [("slug", slug), ("closed", "true"), ("limit", 20)])
    q_open = get("/markets", [("slug", slug), ("closed", "false"), ("limit", 20)])
    neighbors = []
    for dt in (-600, -300, 300, 600):
        nslug = f"{asset.lower()}-updown-5m-{t0 + dt}"
        qn = get("/markets", [("slug", nslug), ("limit", 5)])
        neighbors.append({"dt": dt, "slug": nslug, "count": len(qn)})
    return {
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


def main():
    hours = list(range(START, END, 3600))
    expected = []
    batch_seen = set()
    missing_by_hour = []
    missing = []

    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(audit_hour, h) for h in hours]
        for f in as_completed(futs):
            h, wanted, slugs, miss, returned = f.result()
            expected.extend(wanted)
            batch_seen |= slugs
            missing.extend(miss)
            if miss:
                missing_by_hour.append({
                    "hour": h,
                    "hour_iso": iso(h),
                    "missing": [x[0] for x in miss],
                    "returned": returned,
                })

    missing_by_hour.sort(key=lambda x: x["hour"])
    missing = sorted(set(missing), key=lambda x: (x[2], x[1]))
    print("BATCH_SUMMARY", json.dumps({
        "window": [iso(START), iso(END)],
        "expected": len(expected),
        "seen": len(batch_seen),
        "missing": len(missing),
        "missing_by_hour": missing_by_hour,
    }, ensure_ascii=False, indent=2), flush=True)

    details = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(audit_missing, x): x for x in missing}
        for f in as_completed(futs):
            d = f.result()
            details.append(d)
            print("MISSING_DETAIL", json.dumps(d, ensure_ascii=False), flush=True)

    details.sort(key=lambda d: (d["start"], d["asset"]))
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
