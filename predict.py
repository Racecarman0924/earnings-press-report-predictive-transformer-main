"""
End-to-end demo: ticker -> SEC XBRL figures -> 4 x 768 -> encoder -> prediction.

    python predict.py --ticker AAPL
    python predict.py --ticker AAPL MSFT NVDA --attention
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "data"))

from encoder import EarningsEncoder, CLASSES        # noqa: E402
import embed                                        # noqa: E402

OUT = os.path.join(ROOT, "out")
CACHE = os.path.join(ROOT, "cache")
FIELD_LABEL = {
    "revenue": "GAAP total revenue",
    "net_income": "GAAP net income",
    "eps_basic": "GAAP basic EPS",
    "eps_diluted": "GAAP diluted EPS",
}


def load_figures(ticker):
    """The four GAAP figures for one ticker, from the cached SEC XBRL frames."""
    import fetch
    frames = json.load(open(os.path.join(CACHE, f"frames_{fetch.FRAME}.json")))
    t2c = fetch.ticker_to_cik()
    cik = t2c.get(ticker.upper())
    if cik is None:
        return None, f"{ticker}: no CIK in the SEC ticker map"
    rec = {}
    for field, _, _ in fetch.CONCEPTS:
        v = frames.get(field, {}).get(str(cik))
        if v is None:
            return None, f"{ticker}: SEC has no {field} tagged for {fetch.FRAME}"
        rec[field] = float(v)
    rec["ticker"] = ticker.upper()
    rec["cik"] = cik
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", nargs="+", required=True)
    ap.add_argument("--attention", action="store_true",
                    help="show the 4x4 attention matrix from encoder layer 0")
    a = ap.parse_args()

    ck = torch.load(os.path.join(OUT, "model.pt"), map_location="cpu",
                    weights_only=False)
    model = EarningsEncoder()
    model.load_state_dict(ck["state"])
    model.eval()

    lo, hi = ck["terciles"]
    print(f"model: val {ck['val_acc']:.3f} (baseline {ck['baseline_val']:.3f}) | "
          f"test {ck['test_acc']:.3f} (baseline {ck['baseline_test']:.3f})")
    print(f"classes: bearish <= {lo*100:+.2f}% < neutral < {hi*100:+.2f}% <= bullish "
          f"(open->close, first session after the print)\n")

    recs, bad = [], []
    for t in a.ticker:
        r, err = load_figures(t)
        (recs.append(r) if r else bad.append(err))
    for e in bad:
        print(f"  skipped -- {e}")
    if not recs:
        return 1

    text = embed.embed_records(recs, verbose=False)
    scal = embed.apply_scaler(embed.scalars(recs), ck["scaler"])
    T, S = torch.from_numpy(text), torch.from_numpy(scal)
    with torch.no_grad():
        probs = torch.softmax(model(T, S), dim=1).numpy()
    attn = model.first_layer_attention(T, S).numpy() if a.attention else None

    for i, r in enumerate(recs):
        print(f"── {r['ticker']}  (CIK {r['cik']})")
        for f in embed.FIELDS:
            v = r[f]
            shown = f"{v/1e6:,.1f} M" if "eps" not in f else f"{v:,.2f}"
            print(f"     {FIELD_LABEL[f]:22s} {shown:>16s}")
        k = int(probs[i].argmax())
        print(f"     {'->  PREDICTION':22s} {CLASSES[k].upper():>16s}"
              f"   ({probs[i][k]*100:.1f}% confidence)")
        print("     " + "  ".join(f"{c}={p*100:.1f}%"
                                  for c, p in zip(CLASSES, probs[i])))
        if attn is not None:
            short = ["rev", "ni", "epsB", "epsD"]
            print(f"\n     layer-0 attention (row attends to column), 12-head mean:")
            print("            " + "".join(f"{s:>7s}" for s in short))
            for j, s in enumerate(short):
                print(f"       {s:>5s} " + "".join(f"{attn[i][j][k]:7.3f}"
                                                   for k in range(4)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
