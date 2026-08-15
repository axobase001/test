from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

OUT = Path("original_93d_source_audit")
OUT.mkdir(exist_ok=True)
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


def parquet_footer(repo: str, rev: str, path: str, ts_col: str, unit: str):
    full = f"datasets/{repo}@{rev}/{path}"
    # HfFileSystem supports random access, so ParquetFile normally reads only the footer
    # and row-group metadata rather than materializing the whole object.
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
    wanted = {
        "orderbook": ("data/orderbook/crypto=BTC/timeframe=15-minute/part-0.parquet", "ts_ms", "ms"),
        "ticks": ("data/ticks/crypto=BTC/timeframe=15-minute/part-0.parquet", "timestamp_ms", "ms"),
        "prices": ("data/prices/crypto=BTC/timeframe=15-minute/part-0.parquet", "timestamp", "s"),
        "markets": ("data/markets.parquet", "end_ts", "s"),
    }
    out = {"repo": repo, "revision": rev, "total_files": len(files), "objects": {}}
    for k, (path, col, unit) in wanted.items():
        if path not in files:
            # Preserve actual matching paths to catch naming/layout drift.
            stem = path.rsplit("/", 1)[0]
            out["objects"][k] = {"missing_expected_path": path,
                                 "matches": [x for x in files if x.startswith(stem)][:20]}
        else:
            try:
                out["objects"][k] = parquet_footer(repo, rev, path, col, unit)
            except Exception as exc:
                out["objects"][k] = {"path": path, "error": repr(exc)}
    return out


def audit_kaboom():
    repo = "kaboomfox/15btc_eth"
    info = api.dataset_info(repo)
    rev = info.sha
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    shards = sorted(x for x in files if x.endswith(".parquet"))
    out = {"repo": repo, "revision": rev, "total_files": len(files), "parquet_files": len(shards)}
    # Read footer stats for every shard; stop at metadata only. This is the only way to know
    # whether the stated February collection actually spans the full requested quarter.
    stats = []
    for i, path in enumerate(shards):
        try:
            z = parquet_footer(repo, rev, path, "ts", "ms")
            stats.append(z)
        except Exception as exc:
            stats.append({"path": path, "error": repr(exc)})
        if (i + 1) % 25 == 0:
            print("KABOOM_FOOTERS", i + 1, "/", len(shards), flush=True)
    los = [x.get("min_raw") for x in stats if x.get("min_raw") is not None]
    his = [x.get("max_raw") for x in stats if x.get("max_raw") is not None]
    out["min_utc"] = ts_iso(min(los), "ms") if los else None
    out["max_utc"] = ts_iso(max(his), "ms") if his else None
    out["rows"] = sum(int(x.get("rows", 0)) for x in stats)
    out["errors"] = [x for x in stats if x.get("error")]
    out["sample_stats"] = stats[:2] + stats[-2:] if len(stats) > 4 else stats
    return out


def audit_brock_old():
    repo = "BrockMisner/polymarket_crypto_derivatives"
    rev = "ec37184fcc76a92045bb0a72afad0be6c626487c"
    files = api.list_repo_files(repo, revision=rev, repo_type="dataset")
    pat = re.compile(r"^btc15m_market\d+_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.ndjson$")
    dts = []
    for x in files:
        m = pat.match(x)
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
    }


def main():
    result = {
        "target": {"start": "2026-03-01T00:00:00Z", "end_exclusive": "2026-06-02T00:00:00Z", "days": 93},
        "aliplayer": audit_aliplayer(),
        "kaboomfox": audit_kaboom(),
        "brock_old": audit_brock_old(),
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
