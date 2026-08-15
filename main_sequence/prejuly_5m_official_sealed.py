from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import prejuly_5m_official as core
import prejuly_5m_official_retryfix as transport

# Pure transport hardening from the previous commit.
core.fetch_hour = transport.fetch_hour_fixed

FINANCE_I = core.FEATURES.index("finance_p")


def finance_anchor(examples):
    return torch.tensor([float(e.x[FINANCE_I]) for e in examples], dtype=torch.float32)


def train_one_finance_anchor(train, val, sc, seed):
    """Exact frozen MLP/hyperparameters; restore the predeclared finance hard anchor."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    xt = torch.from_numpy(sc.transform(train))
    yt = torch.tensor([e.label for e in train], dtype=torch.float32)
    at = finance_anchor(train)
    xv = torch.from_numpy(sc.transform(val))
    yv = torch.tensor([e.label for e in val], dtype=torch.float32)
    av = finance_anchor(val)

    ds = torch.utils.data.TensorDataset(xt, yt, at)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    m = core.ResidualMLP(len(core.FEATURES))
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-3)
    best = None
    bestv = 1e9
    stale = 0
    bestep = 0
    for ep in range(80):
        m.train()
        for xb, yb, ab in dl:
            opt.zero_grad()
            d = m(xb)
            p = core.apply_residual(ab, d)
            loss = F.binary_cross_entropy(p, yb) + 0.02 * (d * d).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            d = m(xv)
            p = core.apply_residual(av, d)
            v = float(F.binary_cross_entropy(p, yv) + 0.02 * (d * d).mean())
        if v < bestv - 1e-5:
            bestv = v
            best = {k: z.detach().cpu().clone() for k, z in m.state_dict().items()}
            stale = 0
            bestep = ep + 1
        else:
            stale += 1
            if stale >= 10:
                break
    if best is None:
        raise RuntimeError("no checkpoint produced")
    m.load_state_dict(best)
    return m, {"seed": seed, "best_val_loss": bestv, "best_epoch": bestep}


def predict_finance_anchor(models, xs, sc):
    x = torch.from_numpy(sc.transform(xs))
    a = finance_anchor(xs)
    ps = []
    with torch.no_grad():
        for m in models:
            m.eval()
            ps.append(core.apply_residual(a, m(x)).numpy())
    return np.mean(ps, axis=0)


core.train_one = train_one_finance_anchor
core.predict = predict_finance_anchor


def arg_after(name: str) -> Path | None:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1])
    except (ValueError, IndexError):
        return None


def correct_frozen_metadata(out: Path) -> None:
    cp = out / "FROZEN_CONTRACT.json"
    sp = out / "train_summary.json"
    if not cp.exists():
        return
    contract = json.loads(cp.read_text())
    contract["model"] = "causal Binance RV finance hard anchor + bounded +/-0.50 logit residual MLP; PM trade microstructure and Binance state are correction features"
    contract["hard_anchor"] = "finance_p: causal Binance 1m spot/open + trailing RV digital probability"
    contract["pm_last_role"] = "microstructure feature and execution-price reference; never the hard probability anchor"
    cp.write_text(json.dumps(contract, indent=2))
    if sp.exists():
        summary = json.loads(sp.read_text())
        summary["contract"] = contract
        summary["implementation_attestation"] = "Finance hard anchor restored before any July data was requested; change corrects code-to-predeclared-protocol drift, not validation/test performance."
        sp.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    core.main()
    if phase == "train":
        out = arg_after("--out")
        if out is not None:
            correct_frozen_metadata(out)
