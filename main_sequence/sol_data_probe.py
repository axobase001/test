from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

S = requests.Session()
S.headers.update({"User-Agent": "main-sequence-sol-probe/1.0"})


def get(url, **kwargs):
    try:
        r = S.get(url, timeout=30, **kwargs)
        return {"status": r.status_code, "url": r.url, "text": r.text[:20000], "headers": dict(r.headers)}
    except Exception as e:
        return {"error": repr(e), "url": url}


def js(url, params=None):
    x = get(url, params=params)
    try:
        x["json"] = json.loads(x.get("text", ""))
    except Exception:
        pass
    return x


def main():
    out = {"probe_version": 2}
    out["bybit_instruments"] = js("https://api.bybit.com/v5/market/instruments-info", params={"category":"option","baseCoin":"SOL","limit":1000})
    out["bybit_recent_trades"] = js("https://api.bybit.com/v5/market/recent-trade", params={"category":"option","baseCoin":"SOL","limit":20})
    out["bybit_hv"] = js("https://api.bybit.com/v5/market/historical-volatility", params={"category":"option","baseCoin":"SOL","period":7})

    dirs = [
        "https://public.bybit.com/option/",
        "https://public.bybit.com/options/",
        "https://public.bybit.com/option/SOL/",
        "https://public.bybit.com/options/SOL/",
        "https://public.bybit.com/trading/SOL/",
        "https://public.bybit.com/trading/SOLUSDC/",
    ]
    out["public_dirs"] = {u: get(u) for u in dirs}

    pages = [
        "https://www.bybit.com/derivatives/en/history-data",
        "https://www.bybitglobal.com/derivatives/en/history-data",
    ]
    out["history_pages"] = {}
    for u in pages:
        x = get(u)
        text = x.get("text", "")
        x["urls_found"] = sorted(set(re.findall(r'https?://[^\"\'<> ]+', text)))[:100]
        x["csv_found"] = sorted(set(re.findall(r'[^\"\'<> ]+\.csv(?:\.gz)?', text)))[:100]
        out["history_pages"][u] = x

    gamma = "https://gamma-api.polymarket.com/markets"
    starts = [1778803200, 1780272000, 1782864000, 1785542400]
    out["pm15"] = {}
    for t in starts:
        slug = f"sol-updown-15m-{t}"
        out["pm15"][slug] = js(gamma, params={"slug":slug,"closed":"true","limit":5})

    out["pm1h"] = {}
    for t in starts:
        d = datetime.fromtimestamp(t, tz=timezone.utc)
        for slug in [f"sol-updown-1h-{t}", f"solana-up-or-down-{d.strftime('%B').lower()}-{d.day}-{d.year}-{d.hour or 12}am-et"]:
            out["pm1h"][slug] = js("https://gamma-api.polymarket.com/events", params={"slug":slug,"closed":"true","limit":5})

    bi = out["bybit_instruments"].get("json", {})
    inst = bi.get("result", {}).get("list", []) if isinstance(bi, dict) else []
    bt = out["bybit_recent_trades"].get("json", {})
    tr = bt.get("result", {}).get("list", []) if isinstance(bt, dict) else []
    print("SOL_PROBE_INSTRUMENTS", len(inst), flush=True)
    print("SOL_PROBE_INSTRUMENT_SAMPLE", json.dumps(inst[:3]), flush=True)
    print("SOL_PROBE_TRADES", len(tr), flush=True)
    print("SOL_PROBE_TRADE_SAMPLE", json.dumps(tr[:3]), flush=True)
    for slug,x in out["pm15"].items():
        j=x.get("json",[]); print("SOL_PM15",slug,"count",len(j) if isinstance(j,list) else -1, flush=True)
    for u,x in out["public_dirs"].items(): print("SOL_PUBLIC_DIR",u,x.get("status"),x.get("url"),flush=True)
    open("sol_probe.json","w",encoding="utf-8").write(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
