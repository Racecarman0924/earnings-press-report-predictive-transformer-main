"""
Turn the four GAAP figures into the 4 x 768 input matrix.

Per the plan: one fixed template string per metric, identical across every company,
units normalised, mean-pooled sentence embeddings from an English 768-d financial
encoder. No positional encoding is added -- metric identity lives in the text itself.
"""

import os
import numpy as np

MODEL_NAME = os.environ.get("EMBED_MODEL", "FinLang/finance-embeddings-investopedia")
DIM = 768
FIELDS = ["revenue", "net_income", "eps_basic", "eps_diluted"]

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def templates(rec):
    """Fixed wording, normalised units: currency in millions, EPS in dollars."""
    return [
        f"GAAP total revenue was {rec['revenue'] / 1e6:.1f} million dollars",
        f"GAAP net income was {rec['net_income'] / 1e6:.1f} million dollars",
        f"GAAP basic earnings per share was {rec['eps_basic']:.2f} dollars",
        f"GAAP diluted earnings per share was {rec['eps_diluted']:.2f} dollars",
    ]


def signed_log(x):
    """
    Heavy-tailed magnitudes that can be negative (net income routinely is).
    log1p compresses the five orders of magnitude in revenue; carrying the sign
    keeps a loss distinguishable from a profit of the same size.
    """
    x = np.asarray(x, dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def scalars(records):
    """-> float64 array (N, 4), raw units, column order == FIELDS."""
    return np.array([[r[f] for f in FIELDS] for r in records], dtype="float64")


def fit_scaler(raw_train):
    """Standardise the signed-log scalars. Train-split statistics ONLY."""
    z = signed_log(raw_train)
    mu = z.mean(axis=0)
    sd = z.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return {"mu": mu.tolist(), "sd": sd.tolist()}


def apply_scaler(raw, scaler):
    z = signed_log(raw)
    return ((z - np.array(scaler["mu"])) / np.array(scaler["sd"])).astype("float32")


def embed_records(records, batch=256, verbose=True):
    """-> float32 array (N, 4, 768)."""
    flat = [s for r in records for s in templates(r)]
    if verbose:
        print(f"  embedding {len(flat)} strings ({len(records)} records x 4) "
              f"with {MODEL_NAME}")
    vecs = model().encode(flat, batch_size=batch, show_progress_bar=verbose,
                          convert_to_numpy=True, normalize_embeddings=False)
    assert vecs.shape[1] == DIM, f"expected {DIM}-d, got {vecs.shape[1]}"
    out = vecs.reshape(len(records), 4, DIM).astype("float32")
    assert out.shape[1:] == (4, DIM), out.shape
    return out


# ------------------------------------------------------------------- probe

def monotonicity_probe():
    """
    Gate from Q4. A sentence encoder can be blind to numeric magnitude: if it is,
    every record collapses to nearly the same vector and the model cannot learn
    anything from the numbers. Embed a ladder of revenue values and check that
    cosine distance from the smallest rung grows with magnitude.
    """
    ladder = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
    strs = [f"GAAP total revenue was {v:.1f} million dollars" for v in ladder]
    v = model().encode(strs, convert_to_numpy=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    dist = 1.0 - v @ v[0]

    print(f"\n  magnitude probe ({MODEL_NAME})")
    print(f"  {'revenue ($M)':>14}  {'cos-dist from rung 1':>22}")
    for val, d in zip(ladder, dist):
        print(f"  {val:>14,.1f}  {d:>22.4f}")

    steps = np.diff(dist)
    mono = bool(np.all(steps >= -1e-6))
    spread = float(dist.max())
    print(f"\n  monotonic: {mono}   max cos-distance: {spread:.4f}")
    if not mono:
        print("  WARNING: distance is NOT monotonic in magnitude.")
    if spread < 0.02:
        print("  WARNING: near-zero spread -- the encoder is effectively blind to "
              "magnitude, so the 4 numbers carry almost no signal.")
    if mono and spread >= 0.02:
        print("  PASS: magnitude is represented and ordered.")
    return mono, spread


if __name__ == "__main__":
    monotonicity_probe()
