from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
HOURS = [1781726400, 1781974800]  # two anomalous June hours


def req(path: str, params):
    r = requests.get(
        GAMMA + path,
        params=params,
        timeout=20,
        headers={"User-Agent": "main-sequence-june-index-crosscheck/2.0"},
    )
    r.raise_for_status()
    return r.json()


def iso(ts: int):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def check_one(kind: str, slug: str, asset: str, t0: int, batch_obj):
    m_any = req("/markets", [("slug", slug), ("limit", 20)])
    e_any = req("/events", [("slug", slug), ("limit", 20)])
    nested = []
    for ev in e_any if isinstance(e_any, list) else []:
        for m in (ev.get("markets") or []):
            nested.append({
                "slug": m.get("slug"),
                "conditionId": m.get("conditionId"),
                "closed": m.get("closed"),
                "active": m.get("active"),
            })
    exact_nested = [m for m in nested if str(m.get("slug")) == slug]
    return {
        "kind": kind,
        "slug": slug,
        "asset": asset,
        "start_iso": iso(t0),
        "batch_conditionId": None if batch_obj is None else batch_obj.get("conditionId"),
        "market_single_count": len(m_any) if isinstance(m_any, list) else None,
        "event_single_count": len(e_any) if isinstance(e_any, list) else None,
        "event_exact_nested_count": len(exact_nested),
        "event_exact_nested": exact_nested,
    }


def main():
    records = []
    for h in HOURS:
        wanted = [(f"{a.lower()}-updown-5m-{t0}", a, t0)
                  for a in ASSETS for t0 in range(h, h + 3600, 300)]
        params = [("slug", slug) for slug, _, _ in wanted] + [("closed", "true"), ("limit", 100)]
        js = req("/markets", params)
        byslug = {str(x.get("slug")): x for x in js}
        present = [x for x in wanted if x[0] in byslug]
        missing = [x for x in wanted if x[0] not in byslug]
        print("HOUR", json.dumps({
            "hour_iso": iso(h),
            "batch_count": len(js),
            "present": len(present),
            "missing": [x[0] for x in missing],
        }, ensure_ascii=False), flush=True)

        # Check every missing slug plus eight deterministic present controls.
        controls = present[:4] + present[-4:]
        samples = [("present", x) for x in controls] + [("missing", x) for x in missing]
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {
                ex.submit(check_one, kind, slug, asset, t0, byslug.get(slug)): (kind, slug)
                for kind, (slug, asset, t0) in samples
            }
            for f in as_completed(futs):
                rec = f.result()
                records.append(rec)
                print("CROSSCHECK", json.dumps(rec, ensure_ascii=False), flush=True)

    def agg(kind: str):
        xs = [r for r in records if r["kind"] == kind]
        return {
            "n": len(xs),
            "market_single_hits": sum((r["market_single_count"] or 0) > 0 for r in xs),
            "event_single_hits": sum((r["event_single_count"] or 0) > 0 for r in xs),
            "event_exact_nested_hits": sum((r["event_exact_nested_count"] or 0) > 0 for r in xs),
        }

    print("FINAL", json.dumps({
        "present_controls": agg("present"),
        "missing_candidates": agg("missing"),
        "missing_slugs": [r["slug"] for r in sorted(records, key=lambda x: x["slug"]) if r["kind"] == "missing"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
