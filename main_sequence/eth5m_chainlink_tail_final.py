from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from main_sequence import eth5m_chainlink_tail_qualified as qualified


def load_chainlink_no_future_source(paths):
    """Use only Chainlink rows whose oracle/source clock is not after capture clock."""
    audited = qualified.audited
    ps = audited.load_chainlink_dual_clock(paths)
    keep = ps.capture_ts >= ps.source_ts
    total_rows = int(audited._DUAL_CLOCK_AUDIT.get("chainlink_rows") or 0)
    kept = int(np.sum(keep))
    audited._DUAL_CLOCK_AUDIT["source_clock_rows_nonfuture_vs_capture"] = kept
    audited._DUAL_CLOCK_AUDIT["future_source_rows_rejected"] = int(np.sum(~keep))
    audited._DUAL_CLOCK_AUDIT["source_clock_valid_rate"] = (kept / total_rows) if total_rows else 0.0
    audited._DUAL_CLOCK_AUDIT["future_source_policy"] = "require source_ts <= capture_ts; collector clock skew never creates admissible future oracle state"
    if kept < 100:
        raise RuntimeError(f"insufficient nonfuture dual-clock Chainlink ETH observations: {kept}")
    return audited.DualClockPriceSeries(ps.capture_ts[keep], ps.source_ts[keep], ps.px[keep])


def one_market_rows_common_t90(book: pd.DataFrame, ps, resolutions: dict[str, bool]) -> pd.DataFrame:
    """One common T-90 information set per market.

    Up/Down books may have different last snapshot timestamps, but side/fair/barrier
    selection occurs only once, at the common T-90 clock. Historical opportunities
    at different book timestamps are never compared ex post.
    """
    runner = qualified.audited.runner
    rows = []
    for slug, g in book.groupby("slug", sort=False):
        first = g.iloc[0]
        cond = str(first.cond)
        if cond not in resolutions:
            continue

        win_start = int(first.win_start)
        end_ts = int(first.end_ts)
        open_target_ms = win_start * 1000
        end_target_ms = end_ts * 1000
        decision_ms = end_target_ms - runner.DECISION_S2C * 1000
        settle_up = bool(resolutions[cond])

        # All fair inputs are drawn from the information set available at T-90.
        open_px, open_src, open_cap = ps.at_source_time(
            open_target_ms, decision_ms, runner.MAX_CHAINLINK_LAG_MS
        )
        spot, spot_src, spot_cap = ps.latest_available(
            decision_ms, runner.MAX_CHAINLINK_LAG_MS, runner.MAX_CHAINLINK_LAG_MS
        )
        rv60 = ps.rv(decision_ms, source_target_ms=decision_ms)
        if not (
            open_px > 0 and spot > 0
            and 0 <= open_target_ms - open_src <= runner.MAX_CHAINLINK_LAG_MS
            and open_cap <= decision_ms
            and 0 <= decision_ms - spot_cap <= runner.MAX_CHAINLINK_LAG_MS
            and 0 <= decision_ms - spot_src <= runner.MAX_CHAINLINK_LAG_MS
            and math.isfinite(rv60) and 0.02 < rv60 < 5.0
        ):
            continue

        tau = float(runner.DECISION_S2C)
        p_up = runner.digital_prob_up(spot / open_px, tau, rv60)
        if not math.isfinite(p_up):
            continue
        side = "up" if p_up >= 0.5 else "down"
        fair = float(p_up if side == "up" else 1.0 - p_up)

        side_rows = g[g["outcome"].astype(str).str.lower().str.startswith(side)]
        if side_rows.empty:
            # The latest snapshot for the actual T-90 favorite failed the
            # price/depth validity gate; never substitute the opposite side.
            continue
        rr = side_rows.iloc[-1]
        book_ts = int(rr.ts_ms)
        book_staleness_ms = int(decision_ms - book_ts)
        if book_staleness_ms < 0 or book_staleness_ms > runner.MAX_BOOK_STALENESS_MS:
            continue

        ask = float(rr.best_ask)
        ask_sz = float(rr.ask_sz)
        qty = runner.qty_for_budget(ask)
        cost = qty * (ask + runner.fee_ps(ask))
        barrier_bps = abs(spot / open_px - 1.0) * 10000.0
        sigma_distance = abs(math.log(spot / open_px)) / max(
            rv60 * math.sqrt(tau / runner.YEAR_SECONDS), 1e-12
        )
        won = settle_up if side == "up" else not settle_up

        # Close reconstruction is diagnostic only; Gamma terminal outcome owns PnL.
        close_px, close_src, close_cap = ps.at_source_time(
            end_target_ms, end_target_ms + 10_000, runner.MAX_CHAINLINK_LAG_MS
        )
        captured_chainlink_up = bool(close_px >= open_px) if close_px > 0 else None
        mismatch = (
            captured_chainlink_up != settle_up
            if captured_chainlink_up is not None else None
        )

        rows.append({
            "slug": slug,
            "cond": cond,
            "win_start": win_start,
            "end_ts": end_ts,
            "decision_ts_ms": decision_ms,
            "decision_s2c_s": tau,
            "book_capture_ts_ms": book_ts,
            "book_staleness_ms": book_staleness_ms,
            "side": side,
            "fair": fair,
            "p_up": float(p_up),
            "ask": ask,
            "ask_sz": ask_sz,
            "qty_fixed5": qty,
            "cost": cost,
            "depth_headroom_x": ask_sz / max(qty, 1e-12),
            "net_settlement_edge_ps": fair - ask - runner.fee_ps(ask),
            "barrier_bps": barrier_bps,
            "sigma_distance": sigma_distance,
            "rv60": rv60,
            "threshold_chainlink": open_px,
            "spot_chainlink": spot,
            "close_chainlink_diagnostic": close_px,
            "gamma_settle_up": settle_up,
            "captured_chainlink_reconstructed_up": captured_chainlink_up,
            "gamma_vs_captured_chainlink_mismatch": mismatch,
            "won": bool(won),
            "pnl_fixed5": (qty if won else 0.0) - cost,
            "threshold_source_ts": open_src,
            "threshold_capture_ts": open_cap,
            "spot_source_ts": spot_src,
            "spot_capture_ts": spot_cap,
            "close_source_ts": close_src,
            "close_capture_ts": close_cap,
            "threshold_source_lag_ms": int(open_target_ms - open_src),
            "threshold_capture_minus_source_ms": int(open_cap - open_src),
            "threshold_known_before_decision_ms": int(decision_ms - open_cap),
            "spot_source_lag_ms": int(decision_ms - spot_src),
            "spot_capture_lag_ms": int(decision_ms - spot_cap),
            "spot_capture_minus_source_ms": int(spot_cap - spot_src),
            "common_decision_clock": "T-90",
        })

    return (
        pd.DataFrame(rows).sort_values("decision_ts_ms", kind="mergesort")
        if rows else pd.DataFrame()
    )


# Freeze one common bootstrap unit across all three lanes: market-start UTC day.
_base_daily_pnl_stats = qualified.daily_pnl_stats


def market_start_daily_pnl_stats(df, time_col, pnl_col, *, unit="s", seed_offset=0):
    if df is not None and hasattr(df, "columns") and "win_start" in df.columns:
        return _base_daily_pnl_stats(df, "win_start", pnl_col, unit="s", seed_offset=seed_offset)
    return _base_daily_pnl_stats(df, time_col, pnl_col, unit=unit, seed_offset=seed_offset)


qualified.audited.runner.load_chainlink = load_chainlink_no_future_source
qualified.audited.runner.one_market_rows = one_market_rows_common_t90
qualified.daily_pnl_stats = market_start_daily_pnl_stats


if __name__ == "__main__":
    qualified.audited.runner.main()
    qualified.postprocess()
    manifest_path = Path("eth5m_tail_out/manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        qa = manifest.setdefault("qualification_audit", {})
        qa["bootstrap_day_key"] = "market start UTC day"
        qa["decision_clock"] = "single common T-90 clock for fair, barrier and favorite-side selection"
        qa["book_state"] = "latest valid snapshot per outcome at/before common T-90; chosen favorite snapshot must be <=4s stale"
        qa["cross_timestamp_candidate_selection"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2))
