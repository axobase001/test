from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

OUT = Path("original_93d_source_audit")
OUT.mkdir(exist_ok=True)
AUDIT_PATH = OUT / "audit.json"
api = HfApi()
fs = HfFileSystem()

TARGET_START = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
TARGET_END = int(datetime(2026, 6, 2, tzinfo=timezone.utc).timestamp())  # exclusive = 93 days


def ts_iso(v, unit="s"):
    if v is None:
        return None
    try:
        x = int(v)
        if unit == "ms":
            x = x / 1000
        return datetime.fromtimestamp(x, tz=timezone.utc).isoformat()
    except Exception:
        return str(v)


def persist(result: dict) -> None:
    AUDIT_PATH.write_text(json.dumps(result, indent=2, default=str))


def safe(name: str, fn, result: dict):
    print("AUDIT_CANDIDATE", name, flush=True)
    try:
        result[name] = fn()
    except Exception as exc:
        result[name] = {"error": repr(exc)}
        print("AUDIT_CANDIDATE_UNAVAILABLE", name, repr(exc), flush=True)
    persist(result)


def parquet_footer(repo: str, rev: str, path: str, ts_col: str, unit: str):
    full = f"datasets/{repo}@{rev}/{path}"
    with fs.open(full, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        names = pf.schema_arrow.names
        if ts_col not in names:
            return {"path": path, "rows": md.num_rows, "row_groups": md.num_row_groups,
                    "columns": names, "error": f"missing {ts_col}"}
        ci = names.index(ts_col)
        lo = None
        hi = None
        stats_groups = 0
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(ci).statistics
            if st is not None and st.has_min_max:
                a, b = st.min, st.max
                try:
                    a = int(a); b = int(b)
                except Exception:
                    continue
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
                stats_groups += 1
        return {
            "path": path,
            "rows": md.num_rows,
            "row_groups": md.num_row_groups,
            "columns": names,
            "stats_groups": stats_groups,
            "min_raw": lo,
            "max_raw": hi,
            "min_utc": ts_iso(lo, unit),
            "max_utc": ts_iso(hi, unit),
        }


def audit_aliplayer():
    repo = "aliplayer1/polymarket-crypto-updown"
    info = api.dataset_info(repo)
    rev = info.sha
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    # Layout can drift. Capture matching paths first; then inspect plausible one-file subsets.
    prefixes = {
        "orderbook": "data/orderbook/crypto=BTC/timeframe=15-minute/",
        "ticks": "data/ticks/crypto=BTC/timeframe=15-minute/",
        "prices": "data/prices/crypto=BTC/timeframe=15-minute/",
    }
    out = {"repo": repo, "revision": rev, "total_files": len(files), "objects": {}}
    col_guesses = {
        "orderbook": [("ts_ms", "ms"), ("timestamp_ms", "ms"), ("timestamp", "s")],
        "ticks": [("timestamp_ms", "ms"), ("ts_ms", "ms"), ("timestamp", "s")],
        "prices": [("timestamp", "s"), ("timestamp_ms", "ms"), ("ts_ms", "ms")],
    }
    for kind, prefix in prefixes.items():
        matches = sorted(x for x in files if x.startswith(prefix) and x.endswith(".parquet"))
        rec = {"matches": matches[:100], "match_count": len(matches), "footers": []}
        # Inspect every shard footer when the subset is reasonably sharded. Footer-only access.
        for path in matches:
            done = None
            for col, unit in col_guesses[kind]:
                try:
                    z = parquet_footer(repo, rev, path, col, unit)
                    if not z.get("error"):
                        done = z
                        break
                    if done is None:
                        done = z
                except Exception as exc:
                    done = {"path": path, "error": repr(exc)}
            rec["footers"].append(done)
        los = [x.get("min_raw") for x in rec["footers"] if x and x.get("min_raw") is not None]
        his = [x.get("max_raw") for x in rec["footers"] if x and x.get("max_raw") is not None]
        # Unit is preserved by footer; use UTC strings already computed instead of recomputing mixed units.
        mins = sorted(x.get("min_utc") for x in rec["footers"] if x and x.get("min_utc"))
        maxs = sorted(x.get("max_utc") for x in rec["footers"] if x and x.get("max_utc"))
        rec["min_utc"] = mins[0] if mins else None
        rec["max_utc"] = maxs[-1] if maxs else None
        rec["rows"] = sum(int(x.get("rows", 0)) for x in rec["footers"] if x)
        out["objects"][kind] = rec
    market_matches = sorted(x for x in files if "market" in x.lower() and x.endswith(".parquet"))
    out["market_files"] = market_matches[:100]
    return out


def audit_kaboom():
    repo = "kaboomfox/15btc_eth"
    info = api.dataset_info(repo)
    rev = info.sha
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    shards = sorted(x for x in files if x.endswith(".parquet"))
    out = {"repo": repo, "revision": rev, "total_files": len(files), "parquet_files": len(shards)}
    stats = []
    for i, path in enumerate(shards):
        try:
            z = parquet_footer(repo, rev, path, "ts", "ms")
            stats.append(z)
        except Exception as exc:
            stats.append({"path": path, "error": repr(exc)})
        if (i + 1) % 25 == 0:
            print("KABOOM_FOOTERS", i + 1, "/", len(shards), flush=True)
    mins = sorted(x.get("min_utc") for x in stats if x.get("min_utc"))
    maxs = sorted(x.get("max_utc") for x in stats if x.get("max_utc"))
    out["min_utc"] = mins[0] if mins else None
    out["max_utc"] = maxs[-1] if maxs else None
    out["rows"] = sum(int(x.get("rows", 0)) for x in stats)
    out["errors"] = [x for x in stats if x.get("error")]
    out["sample_stats"] = stats[:2] + stats[-2:] if len(stats) > 4 else stats
    return out


def audit_brock_old():
    repo = "BrockMisner/polymarket_crypto_derivatives"
    rev = "ec37184fcc76a92045bb0a72afad0be6c626487c"
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    # Accept both bare and nested BTC15m episode names and any newline-json suffix.
    pat = re.compile(r"(?:^|/)btc15m_market\d+_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}).*\.(?:ndjson|jsonl)(?:\.gz)?$", re.I)
    dts = []
    for x in files:
        m = pat.search(x)
        if m:
            dts.append(datetime.strptime(m.group(1) + "_" + m.group(2), "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc))
    dts.sort()
    days = sorted({x.date().isoformat() for x in dts})
    return {
        "repo": repo,
        "revision": rev,
        "total_files": len(files),
        "btc15m_files": len(dts),
        "min_utc": dts[0].isoformat() if dts else None,
        "max_utc": dts[-1].isoformat() if dts else None,
        "calendar_days": len(days),
        "first_days": days[:5],
        "last_days": days[-5:],
        "sample_btc_paths": [x for x in files if "btc15m" in x.lower()][:20],
    }


def audit_unified_onchain_trades():
    """Execution-ground-truth candidate; metadata only, never strategy selection."""
    repo = "yamalalaxman/polymarket-btc-trades"
    info = api.dataset_info(repo)
    rev = info.sha
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    # Preserve the full path inventory shape; date parsing is deliberately permissive.
    dates = []
    for path in files:
        for m in re.finditer(r"(2026[-_/]\d{2}[-_/]\d{2})", path):
            s = m.group(1).replace("_", "-").replace("/", "-")
            try:
                dates.append(datetime.strptime(s, "%Y-%m-%d").date().isoformat())
            except Exception:
                pass
    dates = sorted(set(dates))
    return {
        "repo": repo,
        "revision": rev,
        "total_files": len(files),
        "min_date_from_paths": dates[0] if dates else None,
        "max_date_from_paths": dates[-1] if dates else None,
        "calendar_dates_from_paths": len(dates),
        "first_paths": files[:40],
        "last_paths": files[-40:],
    }


def audit_krish_v1():
    repo = "krish301/polymarket-crypto-trades-v1"
    info = api.dataset_info(repo)
    rev = info.sha
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    btc = [x for x in files if "/btc/15m/" in x.lower() or "btc/15m" in x.lower()]
    return {"repo": repo, "revision": rev, "total_files": len(files), "btc15m_files": len(btc),
            "first_btc_paths": btc[:20], "last_btc_paths": btc[-20:]}


def main():
    result = {
        "target": {"start": "2026-03-01T00:00:00Z", "end_exclusive": "2026-06-02T00:00:00Z", "days": 93},
        "audit_policy": "source coverage/schema only; no strategy score or PnL is computed in this workflow",
    }
    persist(result)
    safe("aliplayer", audit_aliplayer, result)
    safe("kaboomfox", audit_kaboom, result)
    safe("brock_old", audit_brock_old, result)
    safe("unified_onchain_trades", audit_unified_onchain_trades, result)
    safe("krish_v1", audit_krish_v1, result)
    persist(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
