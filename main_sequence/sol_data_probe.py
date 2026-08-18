from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

S = requests.Session()
S.headers.update({"User-Agent": "main-sequence-sol-probe/1.1"})


def get(url, **kwargs):
    try:
        r = S.get(url, timeout=30, **kwargs)
        return {"status": r.status_code, "url": r.url, "text": r.text[:50000], "headers": dict(r.headers)}
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
    out = {"probe_version": 3}

    # Bybit global + regional API domains. SOL options are active on Bybit, but
    # the global API can geo-block US hosted runners.
    bybit_hosts = [
        "https://api.bybit.com", "https://api.bytick.com", "https://api.bybit.nl",
        "https://api.bybit.tr", "https://api.bybit.kz", "https://api.bybitgeorgia.ge",
        "https://api.bybit.ae", "https://api.bybit.eu", "https://api.bybit.id",
        "https://api.manepa.jp",
    ]
    out["bybit_hosts"] = {}
    for host in bybit_hosts:
        out["bybit_hosts"][host] = {
            "instruments": js(host+"/v5/market/instruments-info", params={"category":"option","baseCoin":"SOL","limit":20}),
            "trades": js(host+"/v5/market/recent-trade", params={"category":"option","baseCoin":"SOL","limit":20}),
        }

    # OKX supports paginated public history-trades for the prior 3 months.
    okx_hosts = ["https://www.okx.com", "https://app.okx.com", "https://my.okx.com", "https://tr.okx.com"]
    out["okx"] = {}
    for host in okx_hosts:
        out["okx"][host] = {
            "inst_family": js(host+"/api/v5/public/instruments", params={"instType":"OPTION","instFamily":"SOL-USD"}),
            "uly": js(host+"/api/v5/public/instruments", params={"instType":"OPTION","uly":"SOL-USD"}),
            "family_trades": js(host+"/api/v5/market/option/instrument-family-trades", params={"instFamily":"SOL-USD"}),
            "option_trades": js(host+"/api/v5/public/option-trades", params={"instFamily":"SOL-USD"}),
        }

    # Polymarket SOL market mapping sanity checks across the target window.
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

    for host,x in out["bybit_hosts"].items():
        ji=x["instruments"].get("json",{}); jt=x["trades"].get("json",{})
        il=ji.get("result",{}).get("list",[]) if isinstance(ji,dict) else []
        tl=jt.get("result",{}).get("list",[]) if isinstance(jt,dict) else []
        print("BYBIT_HOST",host,"inst_status",x["instruments"].get("status"),"inst",len(il),"trade_status",x["trades"].get("status"),"trades",len(tl),flush=True)
        if il: print("BYBIT_INST_SAMPLE",host,json.dumps(il[:1]),flush=True)
        if tl: print("BYBIT_TRADE_SAMPLE",host,json.dumps(tl[:1]),flush=True)

    for host,x in out["okx"].items():
        for key in ["inst_family","uly","family_trades","option_trades"]:
            jj=x[key].get("json",{}); data=jj.get("data",[]) if isinstance(jj,dict) else []
            print("OKX",host,key,"status",x[key].get("status"),"code",jj.get("code") if isinstance(jj,dict) else None,"n",len(data),flush=True)
            if data: print("OKX_SAMPLE",host,key,json.dumps(data[:2]),flush=True)

    for slug,x in out["pm15"].items():
        j=x.get("json",[]); print("SOL_PM15",slug,"count",len(j) if isinstance(j,list) else -1,flush=True)

    open("sol_probe.json","w",encoding="utf-8").write(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
