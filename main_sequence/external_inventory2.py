from huggingface_hub import HfApi
import json
from pathlib import Path

repos=["gude/july-2026-market-streams","aliplayer1/polymarket-crypto-updown","mingossx/polymarket-crypto-updown"]
out={}
api=HfApi()
for repo in repos:
    print("INVENTORY",repo,flush=True)
    try:
        info=api.dataset_info(repo,files_metadata=True)
    except Exception as e:
        print("ERROR",repr(e),flush=True); continue
    rows=[{"path":s.rfilename,"size":getattr(s,"size",None),"blob_id":getattr(s,"blob_id",None)} for s in info.siblings]
    out[repo]={"sha":info.sha,"files":rows}
    print("sha",info.sha,"files",len(rows),"size",sum((x['size'] or 0) for x in rows),flush=True)
    for x in rows:
        p=x['path'].lower()
        if any(k in p for k in ['reconstructed','orderbook','markets.parquet','ticks/crypto=btc/timeframe=15','spot_prices']) and (x['size'] or 0)<500_000_000:
            print(x,flush=True)
Path('external_inventory2').mkdir(exist_ok=True)
Path('external_inventory2/inventory2.json').write_text(json.dumps(out,indent=2))
