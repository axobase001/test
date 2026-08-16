from __future__ import annotations

import math

from main_sequence import v4_hourly_core_tail_v2 as v2

h = v2.h

FORCE_EXIT_S2C = 60

h.PROTOCOL["name"] = "Main Sequence V4 hourly CORE mark-to-market + TAIL / 2026-08-16 freeze v3"
h.PROTOCOL["core"]["fallback"] = "NO settlement conversion: thesis invalidation exits at first executable top bid; any CORE still open from T-60 is force-liquidated at first executable top bid; if no such SELL witness exists before close, conservatively write off entry cost."
h.PROTOCOL["core"]["thesis_invalidation"] = "If current causal fair, net of estimated exit fee at fair, no longer covers original entry cost, the positive round-trip thesis is invalid and CORE exits immediately at current executable top bid even if loss-making."
h.PROTOCOL["core"]["force_exit_s2c"] = FORCE_EXIT_S2C
h.PROTOCOL["core"]["new_entry_cutoff"] = "No new CORE at or inside T-60 seconds."
h.PROTOCOL["anti_lookahead"].extend([
    "CORE is never converted into a directional binary settlement bet.",
    "Thesis invalidation is evaluated from the current causal fair and original frozen entry cost only.",
    "At T-60, an open CORE enters force-liquidation mode; the first subsequent same-second executable top SELL level is used regardless of PnL.",
    "If no public SELL witness is observed after force-liquidation begins, the unresolved CORE is conservatively written off rather than settled using the final outcome.",
])


def simulate_market_ticket_v3(m, g, spot, bn, der, ticket: float):
    if g is None or g.empty:
        return {"ticket": ticket, "pnl": 0.0, "turnover": 0.0, "core_pnl": 0.0, "tail_pnl": 0.0,
                "core_entries": 0, "core_round_trips": 0, "tail_entries": 0, "cap_rejects": 0,
                "max_open_capital": 0.0, "events": []}

    secs = sorted(int(x) for x in g["timestamp"].dropna().unique().tolist() if m.start <= int(x) < m.close)
    core = None
    tail = None
    tail_blocks_core = False
    core_reentry_after = m.start
    core_pnl = tail_pnl = turnover = 0.0
    core_entries = core_round_trips = tail_entries = cap_rejects = 0
    max_open_cap = 0.0
    events = []
    fair_cache = {}

    def open_capital():
        return (float(core["cost"]) if core else 0.0) + (float(tail["cost"]) if tail else 0.0)

    for sec in secs:
        fb = fair_cache.get(sec)
        if fb is None:
            fb = h.fair_boundary(m, sec, spot, bn, der); fair_cache[sec] = fb
        if fb is None:
            continue
        qsec = g[g["timestamp"] == sec]

        # 1) Existing CORE: profitable convergence, thesis stop, then time stop.
        if core is not None:
            lv = h.top_level(qsec, "SELL", core["outcome"])
            if lv is not None:
                bid, avail = lv
                qty = float(core["qty"])
                if avail + 1e-12 >= qty:
                    fair = float(fb[core["outcome"]])
                    proceeds = qty * bid - h.fee_total(m, bid, qty)
                    fair_proceeds = qty * fair - h.fee_total(m, min(max(fair, 1e-6), 1-1e-6), qty)
                    event = None
                    if bid >= fair - h.FAIR_BAND - 1e-12 and proceeds > float(core["cost"]) + 1e-12:
                        event = "convergence_exit"
                    elif fair_proceeds <= float(core["cost"]) + 1e-12:
                        event = "thesis_stop"
                    elif m.close - sec <= FORCE_EXIT_S2C:
                        event = "time_stop"
                    if event is not None:
                        pnl = proceeds - float(core["cost"])
                        core_pnl += pnl; turnover += proceeds; core_round_trips += 1
                        events.append({"time": sec, "family": "core", "event": event, "outcome": core["outcome"],
                                       "pnl": pnl, "entry": core["price"], "exit": bid, "qty": qty,
                                       "fair": fair, "entry_time": core["entry_time"]})
                        core = None; core_reentry_after = sec + 1

        # 2) TAIL has priority over simultaneous NEW CORE and blocks later new CORE.
        if tail is None:
            tail_candidates = []
            for outcome in ("up", "down"):
                lv = h.top_level(qsec, "BUY", outcome)
                if lv is None: continue
                ask, avail = lv
                fair = float(fb[outcome])
                qty = h.qty_for_budget(m, ask, ticket)
                if qty <= 0 or avail + 1e-12 < qty: continue
                cost = qty * ask + h.fee_total(m, ask, qty)
                edge = fair - ask - h.fee_ps(m, ask, qty)
                if fair >= h.TAIL_FAVORITE_FAIR and edge > 0:
                    tail_candidates.append((edge, outcome, ask, qty, cost, fair))
            if tail_candidates:
                edge, outcome, ask, qty, cost, fair = max(tail_candidates, key=lambda x: x[0])
                if open_capital() + cost <= h.MARKET_CAP_USD + 1e-9:
                    tail = {"outcome": outcome, "price": ask, "qty": qty, "cost": cost, "entry_time": sec, "fair": fair}
                    tail_entries += 1; turnover += cost; tail_blocks_core = True
                    max_open_cap = max(max_open_cap, open_capital())
                    events.append({"time": sec, "family": "tail", "event": "entry", "outcome": outcome,
                                   "pnl": 0.0, "entry": ask, "qty": qty, "fair": fair, "edge": edge})
                else:
                    cap_rejects += 1

        # 3) Repeatable CORE, but never start a new convergence trade at/inside T-60.
        if core is None and not tail_blocks_core and sec >= core_reentry_after and m.close - sec > FORCE_EXIT_S2C:
            core_candidates = []
            for outcome in ("up", "down"):
                lv = h.top_level(qsec, "BUY", outcome)
                if lv is None: continue
                ask, avail = lv
                fair = float(fb[outcome])
                qty = h.qty_for_budget(m, ask, ticket)
                if qty <= 0 or avail + 1e-12 < qty: continue
                buy_fee = h.fee_ps(m, ask, qty)
                exit_fee = h.fee_ps(m, min(max(fair, 1e-6), 1-1e-6), qty)
                rt_edge = fair - ask - buy_fee - exit_fee
                if rt_edge > 0:
                    cost = qty * ask + h.fee_total(m, ask, qty)
                    core_candidates.append((rt_edge, outcome, ask, qty, cost, fair))
            if core_candidates:
                edge, outcome, ask, qty, cost, fair = max(core_candidates, key=lambda x: x[0])
                if open_capital() + cost <= h.MARKET_CAP_USD + 1e-9:
                    core = {"outcome": outcome, "price": ask, "qty": qty, "cost": cost, "entry_time": sec, "fair": fair}
                    core_entries += 1; turnover += cost
                    max_open_cap = max(max_open_cap, open_capital())
                    events.append({"time": sec, "family": "core", "event": "entry", "outcome": outcome,
                                   "pnl": 0.0, "entry": ask, "qty": qty, "fair": fair, "edge": edge})
                else:
                    cap_rejects += 1

    # CORE must never use final resolution. If force liquidation could not be witnessed,
    # conservatively write off the still-open capital.
    if core is not None:
        pnl = -float(core["cost"])
        core_pnl += pnl; core_round_trips += 1
        events.append({"time": m.close, "family": "core", "event": "unwitnessed_force_exit_writeoff",
                       "outcome": core["outcome"], "pnl": pnl, "entry": core["price"], "exit": 0.0,
                       "qty": core["qty"], "entry_time": core["entry_time"]})
        core = None

    # TAIL intentionally remains a settlement trade.
    won_up = m.label_up >= 0.5
    if tail is not None:
        won = won_up if tail["outcome"] == "up" else (not won_up)
        payout = float(tail["qty"]) if won else 0.0
        pnl = payout - float(tail["cost"])
        tail_pnl += pnl
        events.append({"time": m.close, "family": "tail", "event": "settlement", "outcome": tail["outcome"],
                       "pnl": pnl, "entry": tail["price"], "exit": 1.0 if won else 0.0, "qty": tail["qty"]})
        tail = None

    return {
        "ticket": float(ticket), "pnl": float(core_pnl + tail_pnl), "turnover": float(turnover),
        "core_pnl": float(core_pnl), "tail_pnl": float(tail_pnl), "core_entries": int(core_entries),
        "core_round_trips": int(core_round_trips), "tail_entries": int(tail_entries),
        "cap_rejects": int(cap_rejects), "max_open_capital": float(max_open_cap), "events": events,
    }


h.simulate_market_ticket = simulate_market_ticket_v3

if __name__ == "__main__":
    h.main()
