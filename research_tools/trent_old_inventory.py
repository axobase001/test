from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

REPO = "trentmkelly/polymarket_crypto_derivatives_old"
PAT = re.compile(r"(?:^|/)btc15m_market\d+_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(?:_all)?\.(?:ndjson|parquet)$", re.I)


def main() -> None:
    out = Path("trent_old_inventory"); out.mkdir(exist_ok=True)
    api = HfApi(); info = api.dataset_info(REPO); sha = str(info.sha)
    files = api.list_repo_files(REPO, repo_type="dataset", revision=sha)
    rows = []
    for f in files:
        m = PAT.search(f)
        if not m: continue
        day, hh, mm, ss = m.groups()
        t = pd.Timestamp(f"{day}T{hh}:{mm}:{ss}Z")
        rows.append((f, t))
    rows.sort(key=lambda z: z[1])
    ts = pd.DatetimeIndex([x[1] for x in rows])
    unique = pd.DatetimeIndex(sorted(set(ts)))
    diffs = unique.to_series().diff().dropna().dt.total_seconds() / 60.0 if len(unique) else pd.Series(dtype=float)
    expected = None
    coverage = None
    missing_slots = []
    if len(unique):
        full = pd.date_range(unique.min(), unique.max(), freq="15min", tz="UTC")
        expected = len(full)
        have = set(unique)
        missing_slots = [x.isoformat() for x in full if x not in have]
        coverage = len(unique) / expected if expected else None
    report = {
        "repo": REPO, "revision": sha, "repo_file_count": len(files),
        "btc15m_file_count": len(rows), "unique_slot_count": int(len(unique)),
        "first_slot": unique.min().isoformat() if len(unique) else None,
        "last_slot": unique.max().isoformat() if len(unique) else None,
        "calendar_days_inclusive": int((unique.max().normalize() - unique.min().normalize()).days + 1) if len(unique) else 0,
        "expected_15m_slots_between_endpoints": expected,
        "slot_coverage_ratio": coverage,
        "gap_minutes": {
            "median": float(diffs.median()) if len(diffs) else None,
            "p90": float(diffs.quantile(.9)) if len(diffs) else None,
            "p99": float(diffs.quantile(.99)) if len(diffs) else None,
            "max": float(diffs.max()) if len(diffs) else None,
            "gaps_gt_15m": int((diffs > 15.0).sum()) if len(diffs) else 0,
            "gaps_gt_60m": int((diffs > 60.0).sum()) if len(diffs) else 0,
        },
        "missing_slot_count": len(missing_slots),
        "first_missing_slots": missing_slots[:30],
        "first_files": [x[0] for x in rows[:10]], "last_files": [x[0] for x in rows[-10:]],
    }
    (out / "inventory.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__": main()
