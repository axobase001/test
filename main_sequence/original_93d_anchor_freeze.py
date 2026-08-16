from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd

from pm_structural import recalc as original
from main_sequence import original_93d_tape_replay as replay


def _fetch_one(name: str, start_ms: int, end_ms: int) -> list[dict]:
    first_obj = original.fetch_json(
        f"{original.DERIBIT_BASE}/get_last_trades_by_instrument",
        {"instrument_name": name, "start_timestamp": start_ms, "end_timestamp": end_ms, "count": 1, "sorting": "asc"},
    )
    first_rows = first_obj.get("result", {}).get("trades", [])
    if not first_rows:
        return []
    last_obj = original.fetch_json(
        f"{original.DERIBIT_BASE}/get_last_trades_by_instrument",
        {"instrument_name": name, "start_timestamp": start_ms, "end_timestamp": end_ms, "count": 1, "sorting": "desc"},
    )
    last_rows = last_obj.get("result", {}).get("trades", [])
    if not last_rows:
        return []
    start_seq = int(first_rows[0]["trade_seq"])
    end_seq = int(last_rows[0]["trade_seq"])
    rows_out: list[dict] = []
    while start_seq <= end_seq:
        obj = original.fetch_json(
            f"{original.DERIBIT_BASE}/get_last_trades_by_instrument",
            {"instrument_name": name, "start_seq": start_seq, "end_seq": end_seq, "count": 1000, "sorting": "asc"},
        )
        result = obj.get("result", {})
        rows = result.get("trades", [])
        if not rows:
            break
        rows_out.extend(r for r in rows if start_ms <= int(r.get("timestamp", -1)) <= end_ms)
        seqs = [int(r["trade_seq"]) for r in rows if r.get("trade_seq") is not None]
        if not seqs:
            break
        new_start = max(seqs) + 1
        if new_start <= start_seq:
            raise RuntimeError(f"Deribit sequence pagination stalled for {name}")
        start_seq = new_start
        if not result.get("has_more"):
            break
    return rows_out


def fetch_deribit_trades_parallel(instruments: list[str], start: date, end: date, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_d = end + timedelta(days=1)
    end_ms = int(datetime(end_d.year, end_d.month, end_d.day, tzinfo=timezone.utc).timestamp() * 1000) - 1
    by_name: dict[str, list[dict]] = {}
    # Eight workers is intentionally conservative: enough to remove the serial bottleneck
    # without turning Deribit's historical endpoint into a rate-limit experiment.
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_one, name, start_ms, end_ms): name for name in instruments}
        for k, f in enumerate(as_completed(futs), 1):
            name = futs[f]
            by_name[name] = f.result()
            if k % 20 == 0 or k == len(futs):
                print("DERIBIT_PARALLEL", k, "/", len(futs), "rows", sum(len(x) for x in by_name.values()), flush=True)
    # Reassemble in the original selected-instrument order. Downstream DeribitAnchor sorts by timestamp,
    # so this is semantically identical even before that final canonical sort.
    all_rows = [r for name in instruments for r in by_name.get(name, [])]
    if not all_rows:
        raise RuntimeError("No Deribit option trades returned for selected instruments/date range")
    df = pd.DataFrame(all_rows)
    for c in ["block_trade_id", "block_rfq_id", "combo_id", "combo_trade_id"]:
        if c in df.columns:
            df = df[df[c].isna()]
    for c in ["timestamp", "iv", "index_price", "price", "trade_seq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "iv", "index_price", "instrument_name"])
    df = df[(df["iv"] > 0) & (df["index_price"] > 0)].copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def main() -> None:
    original.fetch_deribit_trades = fetch_deribit_trades_parallel
    replay.save_anchors(Path("original93_anchors"))


if __name__ == "__main__":
    main()
