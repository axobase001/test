from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/Vinayak19112003/Polymarket-orderbooks/280a29b9b818cc742c096af81c62e4bf899d44dc/csv_data"
DAYS = ("2026-04-24", "2026-04-25", "2026-04-26")
QTY = 5.0


def load_books(cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in DAYS:
        p = cache / f"orderbook_{day}.csv"
        if not p.exists() or p.stat().st_size < 1000:
            url = f"{RAW_BASE}/orderbook_{day}.csv"
            r = requests.get(url, timeout=180, headers={"User-Agent": "main-sequence-exec-audit/1.0"})
            r.raise_for_status()
            p.write_bytes(r.content)
        z = pd.read_csv(p, usecols=["ts_ms","window_slug","yes_ask","yes_ask_size","no_ask","no_ask_size"])
        frames.append(z)
        print("BOOK", day, len(z), flush=True)
    b = pd.concat(frames, ignore_index=True)
    b["ts_ms"] = pd.to_numeric(b["ts_ms"], errors="coerce")
    return b.dropna(subset=["ts_ms","window_slug"]).sort_values(["window_slug","ts_ms"], kind="mergesort")


def roi(g: pd.DataFrame, reward_col: str, cost_col: str) -> float:
    c = pd.to_numeric(g[cost_col], errors="coerce").sum()
    r = pd.to_numeric(g[reward_col], errors="coerce").sum()
    return float(r / c) if c > 0 else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("orderbook_cache"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.signals)
    lo = int(pd.Timestamp("2026-04-24", tz="UTC").timestamp())
    hi = int(pd.Timestamp("2026-04-27", tz="UTC").timestamp())
    x = x[(x["start"] >= lo) & (x["start"] < hi)].copy()
    # Restrict to settlement-held signals so entry selection is isolated from any
    # later convergence-exit timing difference. This is also the recently failing CORE family.
    x = x[x["witness_exit_kind"].eq("settlement")].copy()
    x["old_strict_fill"] = x["strict_cost"].notna()
    books = load_books(args.cache)

    by_slug = {s: g.sort_values("ts_ms", kind="mergesort") for s, g in books.groupby("window_slug", sort=False)}
    rows = []
    for r in x.itertuples(index=False):
        g = by_slug.get(str(r.slug))
        rec = {
            "slug": r.slug, "decision": int(r.decision), "side": r.side, "limit": float(r.limit),
            "won": bool(r.won), "witness_cost": float(r.witness_cost), "witness_reward": float(r.witness_reward),
            "old_strict_fill": bool(r.old_strict_fill), "old_strict_cost": r.strict_cost, "old_strict_reward": r.strict_reward,
        }
        if g is None or g.empty:
            for d in range(1,6):
                rec[f"d{d}_snapshot"] = False; rec[f"d{d}_fill"] = False
            rows.append(rec); continue
        ts = g["ts_ms"].to_numpy(np.int64)
        for d in range(1,6):
            target = (int(r.decision) + d) * 1000
            i = int(np.searchsorted(ts, target, side="left"))
            ok = i < len(g) and int(ts[i]) < target + 2000
            rec[f"d{d}_snapshot"] = bool(ok)
            if not ok:
                rec[f"d{d}_fill"] = False; rec[f"d{d}_ask"] = math.nan; rec[f"d{d}_ask_size"] = math.nan
                continue
            q = g.iloc[i]
            if str(r.side).lower() == "up":
                ask = float(q["yes_ask"]); size = float(q["yes_ask_size"])
            else:
                ask = float(q["no_ask"]); size = float(q["no_ask_size"])
            fill = math.isfinite(ask) and math.isfinite(size) and ask <= float(r.limit) + 1e-10 and size + 1e-10 >= QTY
            rec[f"d{d}_fill"] = bool(fill); rec[f"d{d}_ask"] = ask; rec[f"d{d}_ask_size"] = size
        rows.append(rec)

    a = pd.DataFrame(rows)
    a.to_csv(args.out / "aligned_signals.csv", index=False)
    summary = {"period": ["2026-04-24","2026-04-27"], "qty_shares": QTY, "signals": int(len(a)), "groups": {}}
    groups = {
        "all_settlement": np.ones(len(a), dtype=bool),
        "low_price_lt50c": a["limit"].to_numpy(float) < .5,
        "high_price_ge50c": a["limit"].to_numpy(float) >= .5,
    }
    for name, mask in groups.items():
        z = a.loc[mask].copy()
        old = z[z["old_strict_fill"]].copy()
        info = {
            "signals": int(len(z)),
            "old_future_tape": {
                "fills": int(len(old)),
                "fill_rate": float(len(old)/len(z)) if len(z) else math.nan,
                "roi": roi(old, "old_strict_reward", "old_strict_cost"),
            },
            "real_orderbook_frozen_limit": {},
        }
        for d in range(1,6):
            have = z[z[f"d{d}_snapshot"]].copy()
            f = have[have[f"d{d}_fill"]].copy()
            info["real_orderbook_frozen_limit"][f"delay_{d}s"] = {
                "snapshot_coverage": int(len(have)),
                "fills": int(len(f)),
                "fill_rate_among_snapshots": float(len(f)/len(have)) if len(have) else math.nan,
                # Use frozen decision-limit witness cost/reward: conservative if ask improved,
                # and exact for settlement payout apart from any price improvement we ignore.
                "conservative_roi_at_frozen_limit": roi(f, "witness_reward", "witness_cost"),
                "wins": int(f["won"].sum()) if len(f) else 0,
            }
        summary["groups"][name] = info

    # Confusion table at 1s between old future-tape proxy and actual best-ask depth.
    h = a[a["d1_snapshot"]].copy()
    summary["d1_proxy_confusion"] = {
        "both_fill": int((h.old_strict_fill & h.d1_fill).sum()),
        "book_fill_only": int((~h.old_strict_fill & h.d1_fill).sum()),
        "future_tape_only": int((h.old_strict_fill & ~h.d1_fill).sum()),
        "neither": int((~h.old_strict_fill & ~h.d1_fill).sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
