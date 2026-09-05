"""
Streamlit demo: SEC EDGAR -> 4 GAAP figures -> 4x768 -> transformer encoder
-> bullish / bearish / neutral, with the layer-0 attention matrix.

Run locally:  streamlit run app.py
"""
import json
import os
import sys

import numpy as np
import requests
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "model"))
from encoder import EarningsEncoder, CLASSES          # noqa: E402
import embed                                          # noqa: E402

UA = {"User-Agent": "earnings-encoder-demo research contact@example.com"}
FIELDS = ["revenue", "net_income", "eps_basic", "eps_diluted"]
LABEL = {"revenue": "GAAP total revenue", "net_income": "GAAP net income",
         "eps_basic": "GAAP basic EPS", "eps_diluted": "GAAP diluted EPS"}
SHORT = ["revenue", "net income", "basic EPS", "diluted EPS"]
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
UNITS = {"revenue": "USD", "net_income": "USD",
         "eps_basic": "USD/shares", "eps_diluted": "USD/shares"}

st.set_page_config(page_title="Earnings Press Report Predictive Transformer", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading the encoder…")
def load_model():
    ck = torch.load(os.path.join(ROOT, "model_deploy.pt"), map_location="cpu",
                    weights_only=False)
    m = EarningsEncoder()
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    m.eval()
    return m, ck


@st.cache_resource(show_spinner="Loading the sentence embedder (first run downloads ~440 MB)…")
def load_embedder():
    return embed.model()


@st.cache_data(show_spinner=False)
def ticker_map():
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=30)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}


@st.cache_data(show_spinner=False)
def fallback_dataset():
    p = os.path.join(ROOT, "out", "dataset.json")
    if not os.path.exists(p):
        return {}
    return {r["ticker"]: r for r in json.load(open(p))}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live(ticker):
    """Latest quarterly value of each GAAP concept, straight from SEC XBRL."""
    cik = ticker_map().get(ticker.upper())
    if cik is None:
        return None, f"{ticker} is not in SEC's ticker registry."
    rec = {}
    for field, tags in TAGS.items():
        got = None
        for tag in tags:
            url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
                   f"CIK{cik:010d}/us-gaap/{tag}.json")
            try:
                r = requests.get(url, headers=UA, timeout=20)
            except requests.RequestException:
                continue
            if r.status_code != 200:
                continue
            rows = r.json().get("units", {}).get(UNITS[field], [])
            quarterly = []
            for e in rows:
                if not e.get("start") or not e.get("end"):
                    continue
                days = (np.datetime64(e["end"]) - np.datetime64(e["start"])).astype(int)
                if 80 <= days <= 100:                       # a fiscal quarter
                    quarterly.append(e)
            if quarterly:
                got = max(quarterly, key=lambda e: e["end"])
                break
        if got is None:
            return None, f"SEC has no quarterly **{LABEL[field]}** tagged for {ticker}."
        rec[field] = float(got["val"])
        rec["period_end"] = got["end"]
    rec["ticker"], rec["cik"] = ticker.upper(), cik
    return rec, None


def predict(model, ck, rec):
    load_embedder()
    text = embed.embed_records([rec], verbose=False)
    scal = embed.apply_scaler(embed.scalars([rec]), ck["scaler"])
    T, S = torch.from_numpy(text), torch.from_numpy(scal)
    with torch.no_grad():
        probs = torch.softmax(model(T, S), dim=1).numpy()[0]
    attn = model.first_layer_attention(T, S).numpy()[0]
    return probs, attn


def attention_figure(attn):
    cmap = LinearSegmentedColormap.from_list("a", ["#0e1117", "#1f6feb", "#58a6ff"])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    fig.patch.set_alpha(0)
    im = ax.imshow(attn, cmap=cmap, vmin=attn.min(), vmax=attn.max())
    ax.set_xticks(range(4), SHORT, rotation=35, ha="right", fontsize=8, color="#c9d1d9")
    ax.set_yticks(range(4), SHORT, fontsize=8, color="#c9d1d9")
    ax.set_xlabel("attends to", fontsize=8, color="#8b949e")
    ax.set_ylabel("from", fontsize=8, color="#8b949e")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{attn[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.045).ax.tick_params(labelsize=7, colors="#8b949e")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- interface
model, ck = load_model()

st.title("📊 Earnings Press Report Predictive Transformer")
st.caption(
    "An attention-based transformer encoder that reads the four GAAP figures every "
    "public company must reconcile to in its earnings release — total revenue, net "
    "income, basic EPS, diluted EPS — and predicts the next session's move."
)

with st.sidebar:
    st.subheader("Architecture")
    st.markdown(
        "- **6** encoder layers, independently initialised\n"
        "- **12** attention heads, d_k = 64\n"
        "- d_model **768**, d_ff **3072**, ReLU\n"
        "- post-LN, ε = 1e-6\n"
        "- **42,535,683** trainable parameters\n"
        "- no decoder, no positional encoding"
    )
    st.subheader("Input")
    st.markdown(
        "Each metric becomes a **1×768** vector from a financial sentence embedder, "
        "summed with a per-metric learned projection of the scalar itself, scaled by "
        "√d_model. Four metrics → a **4×768** matrix."
    )
    st.subheader("Training")
    st.markdown(
        f"- Adam, β=(0.9, 0.98), ε=1e-9\n"
        f"- peak lr 3e-5, warmup + inverse-sqrt decay\n"
        f"- 3-class cross-entropy, {ck.get('epochs', 25)} epochs\n"
        f"- 323 S&P 500 earnings prints, time-ordered split"
    )

col_in, _ = st.columns([1, 2])
ticker = col_in.text_input("Ticker", value="AAPL",
                           help="Any SEC filer. Fetched live from EDGAR.").strip().upper()
go = col_in.button("Predict", type="primary", use_container_width=True)

if go or ticker:
    with st.spinner(f"Fetching {ticker} from SEC EDGAR…"):
        rec, err = fetch_live(ticker)
    source = "live SEC EDGAR XBRL"
    if rec is None:
        cached = fallback_dataset().get(ticker)
        if cached:
            rec, source = cached, "cached July-2026 dataset (live fetch unavailable)"
            st.info(f"{err}  Using the cached record instead.")
        else:
            st.error(err)
            st.stop()

    probs, attn = predict(model, ck, rec)
    k = int(probs.argmax())

    st.markdown(f"### {rec['ticker']} &nbsp;·&nbsp; <span style='font-size:0.55em;"
                f"color:#8b949e'>CIK {rec['cik']} · {source}"
                + (f" · period ending {rec['period_end']}" if rec.get("period_end") else "")
                + "</span>", unsafe_allow_html=True)

    c = st.columns(4)
    for i, f in enumerate(FIELDS):
        v = rec[f]
        c[i].metric(LABEL[f],
                    f"${v/1e6:,.0f}M" if "eps" not in f else f"${v:,.2f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prediction")
        colour = {"bullish": "#3fb950", "bearish": "#f85149", "neutral": "#8b949e"}[CLASSES[k]]
        st.markdown(
            f"<div style='font-size:2.6em;font-weight:700;color:{colour}'>"
            f"{CLASSES[k].upper()}</div>", unsafe_allow_html=True)
        for cl, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{cl}  {p*100:.1f}%")
        lo, hi = ck["terciles"]
        st.caption(
            f"Classes are terciles of the training split's open→close return: "
            f"bearish ≤ {lo*100:+.2f}% < neutral < {hi*100:+.2f}% ≤ bullish, measured "
            f"from the first regular-session open after the release to that day's close."
        )
    with right:
        st.subheader("Layer-0 attention")
        st.pyplot(attention_figure(attn), use_container_width=False)
        st.caption(
            "How each metric weights the others when forming its representation, "
            "averaged over the 12 heads. Uniform 0.250 would mean the attention is "
            "doing nothing; the spread here is learned."
        )
