from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from honest_backtest import Decision, grade_taker
from honest_backtest.adapters.parquet_pm import load_corpus

from pm_structural.recalc import BinanceAnchor, download_binance_1m

YEAR_SECONDS = 365.0 * 24 * 3600
EPS = 1e-9
SEEDS = (7, 19, 42, 73, 101)
SEQ_LEN = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bn_return(anchor: BinanceAnchor, ts_ms: int, seconds: int) -> float:
    hi = int(np.searchsorted(anchor.close_times, int(ts_ms), side="right") - 1)
    lo = int(np.searchsorted(anchor.close_times, int(ts_ms) - seconds * 1000, side="right") - 1)
    if hi < 0 or lo < 0 or hi <= lo:
        return 0.0
    a, b = float(anchor.closes[lo]), float(anchor.closes[hi])
    if a <= 0 or b <= 0:
        return 0.0
    return math.log(b / a)


def bn_rv(anchor: BinanceAnchor, ts_ms: int, minutes: int, min_obs: int) -> float:
    hi = int(np.searchsorted(anchor.close_times, int(ts_ms), side="right"))
    lo = int(np.searchsorted(anchor.close_times, int(ts_ms) - minutes * 60_000, side="left"))
    vals = anchor.log_returns[lo:hi]
    vals = vals[np.isfinite(vals)]
    if vals.size < min_obs:
        return float("nan")
    sig_1m = float(np.std(vals, ddof=1))
    return sig_1m * math.sqrt(365 * 24 * 60)


def safe_mid(bid: float, ask: float) -> float:
    if 0 < bid < ask < 1:
        return 0.5 * (bid + ask)
    return float("nan")


def reward_per_share(ctx, yes: bool, ask: float) -> float:
    won = 1.0 if ((ctx.meta.resolved_side == "Yes") == yes) else 0.0
    fr = float(ctx.meta.fee_rate or 0.0)
    fee = fr * ask * (1.0 - ask)
    return won - ask - fee


@dataclass
class Example:
    ts_ms: int
    close_ts: int
    cid: str
    seq: np.ndarray
    static: np.ndarray
    settle_yes: float
    fade_fill: float
    follow_fill: float
    fade_reward: float
    follow_reward: float
    fade_cost: float
    follow_cost: float
    fade_yes: bool


STATIC_NAMES = [
    "edge_signal", "p_rv", "p_deribit", "anchor_mean", "anchor_gap",
    "rv60", "deribit_iv", "iv_minus_rv", "rel_spot", "log_rel_spot",
    "s2c_norm", "yes_bid", "yes_ask", "no_bid", "no_ask",
    "yes_spread", "no_spread", "yes_mid", "no_mid",
    "log_yes_bid_sz", "log_yes_ask_sz", "log_no_bid_sz", "log_no_ask_sz",
    "yes_book_imb", "no_book_imb", "pm_yes_residual",
    "ret_1m", "ret_5m", "ret_15m", "rv15", "vol_expansion", "shock_z_5m",
]

SEQ_NAMES = [
    "yes_bid", "yes_ask", "no_bid", "no_ask",
    "log_yes_bid_sz", "log_yes_ask_sz", "log_no_bid_sz", "log_no_ask_sz",
    "spot_log_rel_open", "s2c_norm", "yes_mid", "yes_spread",
]


def build_sequence(ctx, i: int) -> np.ndarray:
    # Right-pad so pack_padded_sequence sees all real observations first.
    out = np.full((SEQ_LEN, len(SEQ_NAMES)), np.nan, dtype=np.float32)
    lo = max(0, i - SEQ_LEN + 1)
    idxs = list(range(lo, i + 1))
    open_spot = float(ctx.meta.spot_at_open or 0.0)
    if open_spot <= 0 and idxs:
        open_spot = float(ctx.spot[idxs[0]])
    rows = []
    for j in idxs:
        yb, ya = float(ctx.yb[j]), float(ctx.ya[j])
        nb, na = float(ctx.nb[j]), float(ctx.na[j])
        ymid = safe_mid(yb, ya)
        ysp = ya - yb if 0 < yb < ya < 1 else float("nan")
        sp = float(ctx.spot[j])
        lr = math.log(sp / open_spot) if sp > 0 and open_spot > 0 else 0.0
        rows.append([
            yb, ya, nb, na,
            math.log1p(max(float(ctx.ybs[j]), 0.0)),
            math.log1p(max(float(ctx.yas[j]), 0.0)),
            math.log1p(max(float(ctx.nbs[j]), 0.0)),
            math.log1p(max(float(ctx.nas[j]), 0.0)),
            lr, float(ctx.s2c[j]) / 900.0, ymid, ysp,
        ])
    if rows:
        out[:len(rows)] = np.asarray(rows, dtype=np.float32)
    return out


def build_examples(pm_dir: Path, records_path: Path, bn: BinanceAnchor) -> list[Example]:
    ctxs = list(load_corpus(str(pm_dir), coins=("btc",), durations=("15m",)))
    by_cid = {str(c.meta.condition_id): c for c in ctxs}
    recs = pd.read_csv(records_path)
    examples: list[Example] = []

    for r in recs.itertuples(index=False):
        cid = str(r.cid)
        ctx = by_cid.get(cid)
        if ctx is None:
            continue
        ts = int(r.ts_ms)
        i = int(np.searchsorted(ctx.ts, ts, side="right") - 1)
        if i < 0 or abs(int(ctx.ts[i]) - ts) > 1000:
            continue

        fade_yes = bool(r.yes)
        fade_ask = float(ctx.ya[i] if fade_yes else ctx.na[i])
        follow_yes = not fade_yes
        follow_ask = float(ctx.ya[i] if follow_yes else ctx.na[i])
        if not (0 < fade_ask < 1 and 0 < follow_ask < 1):
            continue

        fade_d = Decision(i=i, ts_ms=ts, token_yes=fade_yes, action="taker", target_px=fade_ask, size=5.0)
        foll_d = Decision(i=i, ts_ms=ts, token_yes=follow_yes, action="taker", target_px=follow_ask, size=5.0)
        gf = grade_taker(ctx, fade_d, latency_ms=1000, tape_window_ms=1500)
        gg = grade_taker(ctx, foll_d, latency_ms=1000, tape_window_ms=1500)
        fade_fill = float(bool(gf.get("has_tape")) and bool(gf.get("fillable")))
        follow_fill = float(bool(gg.get("has_tape")) and bool(gg.get("fillable")))

        p_rv = float(r.p_rv)
        p_d = float(r.p_deribit)
        rv60 = float(r.rv)
        div = float(r.deribit_iv)
        rel_spot = float(r.rel_spot)
        s2c = float(r.s2c)
        yb, ya = float(ctx.yb[i]), float(ctx.ya[i])
        nb, na = float(ctx.nb[i]), float(ctx.na[i])
        ymid, nmid = safe_mid(yb, ya), safe_mid(nb, na)
        ysp = ya - yb if 0 < yb < ya < 1 else 0.0
        nsp = na - nb if 0 < nb < na < 1 else 0.0
        ybs, yas = max(float(ctx.ybs[i]), 0.0), max(float(ctx.yas[i]), 0.0)
        nbs, nas = max(float(ctx.nbs[i]), 0.0), max(float(ctx.nas[i]), 0.0)
        yimb = (ybs - yas) / (ybs + yas + EPS)
        nimb = (nbs - nas) / (nbs + nas + EPS)
        anchor_mean = 0.5 * (p_rv + p_d)

        ret1 = bn_return(bn, ts, 60)
        ret5 = bn_return(bn, ts, 300)
        ret15 = bn_return(bn, ts, 900)
        rv15 = bn_rv(bn, ts, 15, min_obs=8)
        if not math.isfinite(rv15):
            rv15 = rv60
        vol_exp = rv15 / max(rv60, 1e-6)
        sigma5 = rv60 * math.sqrt(300.0 / YEAR_SECONDS)
        shockz = ret5 / max(sigma5, 1e-6)

        static = np.asarray([
            float(r.edge_signal), p_rv, p_d, anchor_mean, abs(p_rv - p_d),
            rv60, div, div - rv60, rel_spot, math.log(max(rel_spot, 1e-8)),
            s2c / 900.0, yb, ya, nb, na,
            ysp, nsp, ymid, nmid,
            math.log1p(ybs), math.log1p(yas), math.log1p(nbs), math.log1p(nas),
            yimb, nimb, ymid - anchor_mean if math.isfinite(ymid) else 0.0,
            ret1, ret5, ret15, rv15, vol_exp, shockz,
        ], dtype=np.float32)
        static[~np.isfinite(static)] = 0.0
        seq = build_sequence(ctx, i)
        settle_yes = 1.0 if ctx.meta.resolved_side == "Yes" else 0.0
        examples.append(Example(
            ts_ms=ts,
            close_ts=int(ctx.meta.close_ts),
            cid=cid,
            seq=seq,
            static=static,
            settle_yes=settle_yes,
            fade_fill=fade_fill,
            follow_fill=follow_fill,
            fade_reward=reward_per_share(ctx, fade_yes, fade_ask),
            follow_reward=reward_per_share(ctx, follow_yes, follow_ask),
            fade_cost=fade_ask + float(ctx.meta.fee_rate or 0.0) * fade_ask * (1.0 - fade_ask),
            follow_cost=follow_ask + float(ctx.meta.fee_rate or 0.0) * follow_ask * (1.0 - follow_ask),
            fade_yes=fade_yes,
        ))
    examples.sort(key=lambda x: (x.close_ts, x.ts_ms))
    return examples


@dataclass
class Scaler:
    static_mean: np.ndarray
    static_std: np.ndarray
    seq_mean: np.ndarray
    seq_std: np.ndarray

    @classmethod
    def fit(cls, exs: list[Example]):
        s = np.stack([e.static for e in exs])
        sm = s.mean(0)
        ss = s.std(0)
        ss[ss < 1e-6] = 1.0
        q = np.concatenate([e.seq for e in exs], axis=0)
        qm = np.nanmean(q, axis=0)
        qs = np.nanstd(q, axis=0)
        qm[~np.isfinite(qm)] = 0.0
        qs[(~np.isfinite(qs)) | (qs < 1e-6)] = 1.0
        return cls(sm, ss, qm, qs)

    def transform(self, exs: list[Example]):
        stat = np.stack([(e.static - self.static_mean) / self.static_std for e in exs]).astype(np.float32)
        seqs = []
        masks = []
        for e in exs:
            valid = np.isfinite(e.seq).all(axis=1).astype(np.float32)
            q = (e.seq - self.seq_mean) / self.seq_std
            q[~np.isfinite(q)] = 0.0
            seqs.append(q.astype(np.float32))
            masks.append(valid)
        return stat, np.stack(seqs), np.stack(masks)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, exs: list[Example], scaler: Scaler):
        self.exs = exs
        self.static, self.seq, self.mask = scaler.transform(exs)

    def __len__(self):
        return len(self.exs)

    def __getitem__(self, i):
        e = self.exs[i]
        return (
            torch.from_numpy(self.seq[i]), torch.from_numpy(self.mask[i]), torch.from_numpy(self.static[i]),
            torch.tensor([e.fade_reward, e.follow_reward], dtype=torch.float32),
            torch.tensor([e.fade_fill, e.follow_fill], dtype=torch.float32),
            torch.tensor(e.settle_yes, dtype=torch.float32),
        )


class MainSequenceNet(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, 32, batch_first=True)
        self.static = nn.Sequential(nn.Linear(static_dim, 32), nn.SiLU(), nn.LayerNorm(32))
        self.fuse = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Dropout(0.10), nn.Linear(64, 48), nn.SiLU())
        self.q = nn.Linear(48, 2)
        self.fill = nn.Linear(48, 2)
        self.settle = nn.Linear(48, 1)

    def forward(self, seq, mask, static):
        lengths = mask.sum(1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        zq = h[-1]
        zs = self.static(static)
        z = self.fuse(torch.cat([zq, zs], dim=1))
        return self.q(z), self.fill(z), self.settle(z).squeeze(1)


def loss_fn(q, fill_logit, settle_logit, reward, fill, settle):
    m = fill > 0.5
    if m.any():
        qloss = F.smooth_l1_loss(q[m], reward[m])
    else:
        qloss = q.sum() * 0.0
    floss = F.binary_cross_entropy_with_logits(fill_logit, fill)
    sloss = F.binary_cross_entropy_with_logits(settle_logit, settle)
    return qloss + 0.45 * floss + 0.20 * sloss


def train_one(seed: int, train_ex: list[Example], val_ex: list[Example], scaler: Scaler):
    set_seed(seed)
    tr = Dataset(train_ex, scaler)
    va = Dataset(val_ex, scaler)
    model = MainSequenceNet(len(SEQ_NAMES), len(STATIC_NAMES))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
    best = None
    best_val = float("inf")
    patience = 12
    stale = 0
    history = []
    for epoch in range(120):
        model.train()
        losses = []
        for seq, mask, stat, reward, fill, settle in loader:
            opt.zero_grad(set_to_none=True)
            q, fl, sl = model(seq, mask, stat)
            loss = loss_fn(q, fl, sl, reward, fill, settle)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            seq, mask, stat, reward, fill, settle = next(iter(torch.utils.data.DataLoader(va, batch_size=len(va))))
            q, fl, sl = model(seq, mask, stat)
            vl = loss_fn(q, fl, sl, reward, fill, settle)
            v = float(vl)
        history.append({"epoch": epoch + 1, "train": float(np.mean(losses)), "val": v})
        if v < best_val - 1e-5:
            best_val = v
            best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best)
    return model, history, best_val


def predict(model: nn.Module, exs: list[Example], scaler: Scaler):
    ds = Dataset(exs, scaler)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    qs, fps, sps = [], [], []
    model.eval()
    with torch.no_grad():
        for seq, mask, stat, _, _, _ in loader:
            q, fl, sl = model(seq, mask, stat)
            qs.append(q.numpy())
            fps.append(torch.sigmoid(fl).numpy())
            sps.append(torch.sigmoid(sl).numpy())
    return np.concatenate(qs), np.concatenate(fps), np.concatenate(sps)


def cluster_ci(rows: pd.DataFrame, iters: int = 20000, seed: int = 17):
    if rows.empty:
        return [None, None]
    x = rows.copy()
    x["day"] = pd.to_datetime(x.ts_ms, unit="ms", utc=True).dt.date.astype(str)
    days = sorted(x.day.unique())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        sampled = rng.choice(days, size=len(days), replace=True)
        r = pd.concat([x[x.day == d] for d in sampled], ignore_index=True)
        if len(r):
            vals.append(float(r.real_reward.mean()))
    if not vals:
        return [None, None]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def evaluate_policy(exs: list[Example], q: np.ndarray, fp: np.ndarray, name: str, mode: str = "model"):
    rows = []
    for i, e in enumerate(exs):
        if mode == "baseline_fade":
            action = 0
        else:
            ev = fp[i] * q[i]
            action = int(np.argmax(ev))
            if float(ev[action]) <= 0.0:
                continue
        if action == 0:
            fill, rew, cost = e.fade_fill, e.fade_reward, e.fade_cost
        else:
            fill, rew, cost = e.follow_fill, e.follow_reward, e.follow_cost
        if fill < 0.5:
            continue
        rows.append({
            "ts_ms": e.ts_ms, "cid": e.cid, "action": "fade" if action == 0 else "follow",
            "real_reward": rew, "cost": cost,
            "pred_q": float(q[i, action]) if mode != "baseline_fade" else None,
            "pred_fill": float(fp[i, action]) if mode != "baseline_fade" else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"name": name, "fills": 0, "edge_share": None, "roi": None, "ci95": [None, None], "actions": {}}, df
    pnl = float(df.real_reward.sum())
    cost = float(df.cost.sum())
    return {
        "name": name,
        "fills": int(len(df)),
        "edge_share": float(df.real_reward.mean()),
        "roi": pnl / cost if cost > 0 else None,
        "ci95": cluster_ci(df),
        "actions": {k: int(v) for k, v in df.action.value_counts().to_dict().items()},
    }, df


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-dir", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--prior-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--start", default="2026-05-27")
    ap.add_argument("--end", default="2026-06-24")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    prior = json.loads(args.prior_summary.read_text())
    split_ts = int(prior["split_close_ts"])
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    print("Downloading official Binance 1m history...", flush=True)
    bn_df = download_binance_1m(start, end, args.cache / "binance")
    bn = BinanceAnchor.from_df(bn_df)
    print("Building frozen 3c neural examples...", flush=True)
    examples = build_examples(args.pm_dir, args.records, bn)
    pre = [e for e in examples if e.close_ts <= split_ts]
    hold = [e for e in examples if e.close_ts > split_ts]
    nsub = max(1, int(len(pre) * 0.80))
    subtrain, val = pre[:nsub], pre[nsub:]
    scaler = Scaler.fit(subtrain)
    print(f"examples={len(examples)} pre={len(pre)} train={len(subtrain)} val={len(val)} holdout={len(hold)}", flush=True)
    print(f"train tape labels: fade={sum(e.fade_fill for e in subtrain):.0f} follow={sum(e.follow_fill for e in subtrain):.0f}", flush=True)

    qh, fph, sph = [], [], []
    seed_meta = []
    state_dicts = {}
    for seed in SEEDS:
        print(f"training seed={seed}", flush=True)
        model, hist, bv = train_one(seed, subtrain, val, scaler)
        q, fp, sp = predict(model, hold, scaler)
        qh.append(q); fph.append(fp); sph.append(sp)
        seed_meta.append({"seed": seed, "best_val_loss": bv, "epochs": len(hist)})
        state_dicts[str(seed)] = model.state_dict()
    qh = np.mean(qh, axis=0)
    fph = np.mean(fph, axis=0)
    sph = np.mean(sph, axis=0)

    model_metrics, model_rows = evaluate_policy(hold, qh, fph, "main_sequence_v0")
    base_metrics, base_rows = evaluate_policy(hold, qh, fph, "frozen_3c_fade", mode="baseline_fade")
    settle_y = [e.settle_yes for e in hold]
    anchor_p = [float(e.static[3]) for e in hold]
    summary = {
        "name": "Main Sequence v0",
        "architecture": "GRU(32) PM-book sequence + static cross-market features; heads: conditional Q(fade/follow), fill probability, settlement probability",
        "threshold_frozen": "3 cents from prior replay; not retuned",
        "split_close_utc": datetime.fromtimestamp(split_ts, tz=timezone.utc).isoformat(),
        "data": {"examples": len(examples), "pre_holdout": len(pre), "subtrain": len(subtrain), "validation": len(val), "holdout": len(hold)},
        "seeds": seed_meta,
        "holdout": {
            "model_policy": model_metrics,
            "baseline_fade": base_metrics,
            "settlement_brier_model": brier(settle_y, sph),
            "settlement_brier_anchor_mean": brier(settle_y, anchor_p),
        },
        "notes": [
            "No holdout timestamps are used for fitting, scaling, early stopping, or threshold selection.",
            "Q heads are supervised only when the corresponding action has strict real-tape corroboration within 1500ms.",
            "Policy abstains when max predicted fill_probability * conditional_Q <= 0.",
            "This is a discovery-stage offline value network, not deployment-authorized RL or a market-impact simulator.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    model_rows.to_csv(args.out / "holdout_model_fills.csv", index=False)
    base_rows.to_csv(args.out / "holdout_baseline_fills.csv", index=False)
    np.savez(
        args.out / "scaler.npz",
        static_mean=scaler.static_mean, static_std=scaler.static_std,
        seq_mean=scaler.seq_mean, seq_std=scaler.seq_std,
        static_names=np.asarray(STATIC_NAMES), seq_names=np.asarray(SEQ_NAMES),
    )
    torch.save({"states": state_dicts, "static_names": STATIC_NAMES, "seq_names": SEQ_NAMES, "seeds": SEEDS}, args.out / "main_sequence_v0.pt")
    lines = [
        "# Main Sequence v0", "",
        f"Frozen 3c signal set; split {summary['split_close_utc']}; holdout={len(hold)}.", "",
        "| policy | strict tape fills | edge/share | fee ROI | day-cluster 95% CI | action mix |",
        "|---|---:|---:|---:|---|---|",
    ]
    for x in (base_metrics, model_metrics):
        lines.append(f"| {x['name']} | {x['fills']} | {x['edge_share']} | {x['roi']} | {x['ci95']} | {x['actions']} |")
    lines += ["", f"Settlement Brier: model={summary['holdout']['settlement_brier_model']:.6f}; anchor mean={summary['holdout']['settlement_brier_anchor_mean']:.6f}."]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((args.out / "SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
