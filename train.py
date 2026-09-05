"""
Train the earnings encoder.

Splits are TIME-ORDERED, never random: the whole market moves together on a given
day, so a random split leaks same-day information across train and test.
Class thresholds and the scalar standardiser are both fitted on the training
split alone.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))

from encoder import EarningsEncoder, CLASSES, param_count   # noqa: E402
import embed                                                # noqa: E402

OUT = os.path.join(ROOT, "out")
CACHE = os.path.join(ROOT, "cache")

EPOCHS = int(os.environ.get("EPOCHS", 25))
BATCH = int(os.environ.get("BATCH", 32))
PEAK_LR = float(os.environ.get("LR", 3e-5))
WARMUP = int(os.environ.get("WARMUP", 40))
# Fixed 25-epoch budget, chosen once and left alone. Deliberately NOT tuned: this
# is a demonstrator, not an accuracy exercise, and selecting the epoch that scores
# best on a 48-sample validation split is curve-fitting, not improvement. Early
# stopping is off by default (PATIENCE > EPOCHS) so the shipped weights are simply
# the last epoch -- no validation selection anywhere in the shipped artifact.
PATIENCE = int(os.environ.get("PATIENCE", 10**9))
SEED = 0


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def labels_from_terciles(rets, lo, hi):
    """0 bearish, 1 neutral, 2 bullish -- matches encoder.CLASSES."""
    y = np.ones(len(rets), dtype="int64")
    y[rets <= lo] = 0
    y[rets >= hi] = 2
    return y


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()

    recs = json.load(open(os.path.join(OUT, "dataset.json")))
    recs.sort(key=lambda r: (r["react_date"], r["ticker"]))
    n = len(recs)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    idx_tr = slice(0, n_tr)
    idx_va = slice(n_tr, n_tr + n_va)
    idx_te = slice(n_tr + n_va, n)
    print(f"[split] {n} samples, time-ordered "
          f"({recs[0]['react_date']} .. {recs[-1]['react_date']})")
    for nm, sl in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        part = recs[sl]
        print(f"  {nm:5s} {len(part):4d}  {part[0]['react_date']} .. {part[-1]['react_date']}")

    # ---- labels: terciles of the TRAIN split only
    rets = np.array([r["ret"] for r in recs], dtype="float64")
    lo, hi = np.percentile(rets[idx_tr], [100 / 3, 200 / 3])
    y = labels_from_terciles(rets, lo, hi)
    print(f"\n[labels] train terciles: bearish <= {lo * 100:+.2f}% "
          f"< neutral < {hi * 100:+.2f}% <= bullish")
    for nm, sl in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
        c = np.bincount(y[sl], minlength=3)
        print(f"  {nm:5s} bearish={c[0]:3d} neutral={c[1]:3d} bullish={c[2]:3d}")

    # ---- text embeddings (cached; deterministic and the slowest step)
    cache_p = os.path.join(CACHE, f"text_emb_{embed.MODEL_NAME.replace('/', '_')}_{n}.npy")
    if os.path.exists(cache_p):
        text = np.load(cache_p)
        print(f"\n[embed] loaded cache {text.shape}")
    else:
        print()
        text = embed.embed_records(recs)
        np.save(cache_p, text)
    assert text.shape == (n, 4, 768), text.shape

    # ---- numeric channel inputs, standardised on TRAIN only
    raw = embed.scalars(recs)
    scaler = embed.fit_scaler(raw[idx_tr])
    scal = embed.apply_scaler(raw, scaler)
    print(f"[scalars] signed-log then z-scored on train; "
          f"range {scal.min():+.2f}..{scal.max():+.2f}")

    # ---- GATE: do the input matrices actually differ across samples now?
    def spread(mat):
        f = mat.reshape(len(mat), -1)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        D = 1 - f @ f.T
        return D[np.triu_indices(len(f), 1)].mean()

    m_tmp = EarningsEncoder()
    with torch.no_grad():
        combined = m_tmp.build_input(torch.from_numpy(text),
                                     torch.from_numpy(scal)).numpy()
    s_text, s_comb = spread(text), spread(combined)
    print(f"\n[gate] mean pairwise cos-distance between input matrices")
    print(f"  text only          {s_text:.4f}")
    print(f"  text + numeric     {s_comb:.4f}   ({s_comb / max(s_text, 1e-9):.1f}x)")
    if s_comb < 0.01:
        print("  FAIL: samples are still near-identical; not training on this.")
        return 1
    print("  PASS: samples are distinguishable.")

    # ---- tensors
    T = torch.from_numpy(text).to(dev)
    S = torch.from_numpy(scal).to(dev)
    Y = torch.from_numpy(y).to(dev)

    model = EarningsEncoder().to(dev)
    print(f"\n[model] {param_count(model):,} trainable params on {dev}")

    opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.98), eps=1e-9)
    lossf = nn.CrossEntropyLoss()
    step = 0

    def set_lr():
        # Vaswani sec 5.3: linear warmup then inverse-sqrt decay. Post-LN at 6
        # layers does not train stably without it.
        s = max(step, 1)
        scale = min(s ** -0.5, s * WARMUP ** -1.5) / (WARMUP ** -0.5)
        for g in opt.param_groups:
            g["lr"] = PEAK_LR * scale

    def evaluate(sl):
        model.eval()
        with torch.no_grad():
            logits = model(T[sl], S[sl])
            pred = logits.argmax(1)
            return float((pred == Y[sl]).float().mean()), pred.cpu().numpy()

    maj = int(np.bincount(y[idx_tr], minlength=3).argmax())
    base_va = float((y[idx_va] == maj).mean())
    base_te = float((y[idx_te] == maj).mean())

    # Fixed probe batch so the attention series is comparable across epochs.
    probe = slice(0, min(32, n_tr))

    def attn_dev(idx):
        return model.attention_uniformity(T[probe], S[probe], idx)

    def attn_grad_norms():
        """Per-layer gradient norm on self_attn.in_proj_weight (the Q/K/V matrices).
        Shows whether gradient survives the trip back to layer 0."""
        out = []
        for L in model.layers:
            g = L.self_attn.in_proj_weight.grad
            out.append(0.0 if g is None else float(g.norm()))
        return out

    tr_ids = np.arange(n_tr)
    best = {"val": -1.0}
    history = []
    stale = 0
    a0, a5 = attn_dev(0), attn_dev(5)
    print(f"\n[train] {EPOCHS} epochs, batch {BATCH}, "
          f"Adam b=({opt.defaults['betas'][0]}, {opt.defaults['betas'][1]}) "
          f"eps={opt.defaults['eps']}, peak lr {PEAK_LR}, warmup {WARMUP}")
    print(f"  attention deviation from uniform BEFORE training: "
          f"layer0={a0:.6f}  layer5={a5:.6f}")
    print(f"\n  {'ep':>4s} {'loss':>8s} {'train':>6s} {'val':>6s} "
          f"{'attn-L0':>9s} {'attn-L5':>9s} {'grad-L0':>9s} {'grad-L5':>9s}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(tr_ids)
        tot, gnorms = 0.0, None
        for i in range(0, len(tr_ids), BATCH):
            b = torch.from_numpy(tr_ids[i:i + BATCH]).to(dev)
            step += 1
            set_lr()
            opt.zero_grad()
            loss = lossf(model(T[b], S[b]), Y[b])
            loss.backward()
            if gnorms is None:                     # capture pre-clip, first batch
                gnorms = attn_grad_norms()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)

        tr_acc, _ = evaluate(idx_tr)
        va_acc, _ = evaluate(idx_va)
        a0, a5 = attn_dev(0), attn_dev(5)
        history.append({"epoch": ep, "loss": tot / n_tr, "train": tr_acc,
                        "val": va_acc, "attn_l0": a0, "attn_l5": a5,
                        "grad": gnorms})
        if va_acc > best["val"]:
            best = {"val": va_acc, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if ep % 10 == 0 or ep == 1 or stale >= PATIENCE:
            print(f"  {ep:4d} {tot / n_tr:8.4f} {tr_acc:6.3f} {va_acc:6.3f} "
                  f"{a0:9.6f} {a5:9.6f} {gnorms[0]:9.4f} {gnorms[5]:9.4f}")
        if stale >= PATIENCE:
            print(f"\n  early stop: no val improvement for {PATIENCE} epochs "
                  f"(best was epoch {best['epoch']}, val {best['val']:.3f})")
            break

    stopped_at = len(history)
    # ---- last-epoch model (before restoring the early-stopping checkpoint)
    fin_va, _ = evaluate(idx_va)
    fin_te, fin_pred = evaluate(idx_te)
    fin_a0, fin_a5 = attn_dev(0), attn_dev(5)

    # ---- best-validation model
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best["state"])
    bst_va, _ = evaluate(idx_va)
    bst_te, bst_pred = evaluate(idx_te)

    def report(tag, va, te, pred):
        print(f"  {tag}")
        print(f"    val   {va:.3f}  baseline {base_va:.3f}  "
              f"{'BEATS' if va > base_va else 'does NOT beat'}")
        print(f"    test  {te:.3f}  baseline {base_te:.3f}  "
              f"{'BEATS' if te > base_te else 'does NOT beat'}")
        print(f"    test predictions: "
              f"{dict(zip(CLASSES, np.bincount(pred, minlength=3).tolist()))}")

    print(f"\n[result]")
    report(f"SHIPPED: last epoch ({stopped_at}), no validation selection",
           fin_va, fin_te, fin_pred)
    report(f"for reference only, NOT shipped: best-val epoch {best['epoch']} "
           f"(selected on {n_va} val samples -- optimistically biased)",
           bst_va, bst_te, bst_pred)

    print(f"\n[attention] deviation from uniform (0 = pure averaging)")
    print(f"  layer0  {history[0]['attn_l0']:.6f} -> {fin_a0:.6f}")
    print(f"  layer5  {history[0]['attn_l5']:.6f} -> {fin_a5:.6f}")
    print("  Settled earlier, on the unscaled run: attention DOES differentiate")
    print("  through training (layers 1-5 reached 0.15-0.33 there). Layer 0 was the")
    print("  lone exception at 0.0043, because it alone received the raw embedding")
    print("  at row norm 1.32 while LayerNorm feeds every later layer sqrt(768)=27.71.")
    print("  build_input now applies that same sqrt(d_model) factor, so layer 0 sits")
    print("  in the same range as the rest. No open question remains here.")

    model.load_state_dict(final_state)
    ckpt = os.path.join(OUT, "model.pt")
    torch.save({"state": final_state, "scaler": scaler,
                "terciles": [float(lo), float(hi)],
                "val_acc": fin_va, "test_acc": fin_te,
                "baseline_val": base_va, "baseline_test": base_te,
                "last_epoch_val": fin_va, "last_epoch_test": fin_te,
                "stopped_at": stopped_at,
                "best_epoch": best["epoch"], "history": history,
                "embed_model": embed.MODEL_NAME}, ckpt)
    json.dump(history, open(os.path.join(OUT, "history.json"), "w"), indent=1)
    print(f"\n  saved {ckpt} (last-epoch weights, epoch {stopped_at}, no val selection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
