"""
Build the MVP dataset: S&P 500 8-K Item 2.02 earnings prints in a target window,
joined to the four GAAP figures from SEC XBRL, labelled by the open->close return
of the first regular session after the release.

Per the approved plan:
  - numbers source  : SEC XBRL (look-ahead accepted for the MVP)
  - label window    : first RTH open after release -> that session's close, raw return
  - mid-session releases are SKIPPED
  - no silent truncation: every drop reason is counted and reported
"""

import json
import time
import sys
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "out")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# SEC fair-access policy requires a contact address in the User-Agent.
UA = {"User-Agent": "rohansudarshan0924@gmail.com earnings-encoder-research"}

# July 2026 season -> companies report the calendar Q2 2026 period.
WINDOW_START = datetime(2026, 7, 1).date()
WINDOW_END = datetime(2026, 8, 15).date()
FRAME = "CY2026Q2"

# Revenue is tagged inconsistently across filers; try in priority order.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
CONCEPTS = [
    ("revenue", REVENUE_TAGS, "USD"),
    ("net_income", ["NetIncomeLoss"], "USD"),
    ("eps_basic", ["EarningsPerShareBasic"], "USD-per-shares"),
    ("eps_diluted", ["EarningsPerShareDiluted"], "USD-per-shares"),
]

_last_call = [0.0]


def get(url, tries=3):
    """GET with SEC's 10 req/s cap respected, and a small retry."""
    for attempt in range(tries):
        gap = time.monotonic() - _last_call[0]
        if gap < 0.11:
            time.sleep(0.11 - gap)
        _last_call[0] = time.monotonic()
        try:
            r = requests.get(url, headers=UA, timeout=30)
        except requests.RequestException as e:
            if attempt == tries - 1:
                return None
            time.sleep(1.0 + attempt)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(1.0 + attempt)
    return None


def cached(name, fn):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    v = fn()
    with open(p, "w") as f:
        json.dump(v, f)
    return v


# ---------------------------------------------------------------- 1. universe

def sp500_tickers():
    """S&P 500 constituents. Wikipedia first, static fallback so we never hard-fail."""
    try:
        import io
        import pandas as pd
        # Fetch with requests (bundles certifi); pandas' own reader goes through
        # urllib, which fails on this framework Python for want of root certs.
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=UA, timeout=30,
        ).text
        tbl = pd.read_html(io.StringIO(html))[0]
        col = "Symbol" if "Symbol" in tbl.columns else tbl.columns[0]
        tics = [str(t).replace(".", "-").strip().upper() for t in tbl[col].tolist()]
        tics = [t for t in tics if t and t != "NAN"]
        if len(tics) > 400:
            return tics
        print(f"  wikipedia returned only {len(tics)} tickers, using fallback")
    except Exception as e:
        print(f"  wikipedia fetch failed ({type(e).__name__}), using fallback")
    from sp500_fallback import TICKERS
    return TICKERS


def ticker_to_cik():
    d = cached("company_tickers.json",
               lambda: get("https://www.sec.gov/files/company_tickers.json").json())
    return {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}


# ------------------------------------------------- 2. the four GAAP figures

def fetch_frames():
    """One request per concept returns every filer's value for that period."""
    vals = {}
    for field, tags, unit in CONCEPTS:
        per_cik = {}
        for tag in tags:
            url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/{FRAME}.json"
            r = get(url)
            if r is None:
                print(f"    {field:12s} {tag:52s} -> unavailable")
                continue
            rows = r.json().get("data", [])
            added = 0
            for row in rows:
                cik = int(row["cik"])
                if cik not in per_cik:          # first tag in priority order wins
                    per_cik[cik] = float(row["val"])
                    added += 1
            print(f"    {field:12s} {tag:52s} -> {len(rows):5d} rows, {added:5d} new")
        vals[field] = per_cik
    return vals


# ------------------------------------------------------ 3. 8-K release times

def earnings_8ks(cik):
    """8-K filings carrying Item 2.02 inside the window, with acceptance timestamps."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    r = get(url)
    if r is None:
        return []
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = recent.get("items", [""] * len(forms))[i] or ""
        if "2.02" not in items:
            continue
        fdate = recent.get("filingDate", [])[i]
        try:
            d = datetime.strptime(fdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if not (WINDOW_START <= d <= WINDOW_END):
            continue
        out.append({
            "filing_date": fdate,
            "accepted": recent.get("acceptanceDateTime", [""] * len(forms))[i],
            "accession": recent.get("accessionNumber", [""] * len(forms))[i],
        })
    return out


def reaction_session(accepted_iso, filing_date):
    """
    Map release timestamp -> the session we measure, per the locked label rule.
      before 09:30 ET -> same day        (BMO)
      at/after 16:00  -> next trading day (AMC)
      during RTH      -> skip
    acceptanceDateTime is published in US/Eastern.
    """
    if not accepted_iso:
        return None, "no_acceptance_time"
    try:
        ts = datetime.fromisoformat(accepted_iso.replace("Z", "+00:00"))
    except ValueError:
        return None, "bad_acceptance_time"
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone(timedelta(hours=-4))).replace(tzinfo=None)
    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60 + 30:
        return ts.date(), "bmo"
    if minutes >= 16 * 60:
        return ts.date() + timedelta(days=1), "amc"
    return None, "mid_session"


# ------------------------------------------------------------------ 4. build

def main():
    drops = defaultdict(int)

    print("[1/5] universe")
    tickers = sp500_tickers()
    t2c = ticker_to_cik()
    pairs = []
    for t in tickers:
        if t in t2c:
            pairs.append((t, t2c[t]))
        else:
            drops["ticker_not_in_sec_map"] += 1
    print(f"  {len(tickers)} S&P 500 tickers, {len(pairs)} mapped to a CIK")

    print(f"[2/5] XBRL frames for {FRAME}")
    frames = cached(f"frames_{FRAME}.json", fetch_frames)
    frames = {k: {int(kk): vv for kk, vv in v.items()} for k, v in frames.items()}

    print(f"[3/5] 8-K Item 2.02 filings {WINDOW_START}..{WINDOW_END}")

    def pull_all():
        acc = {}
        for n, (tic, cik) in enumerate(pairs, 1):
            acc[str(cik)] = earnings_8ks(cik)
            if n % 50 == 0:
                print(f"    {n}/{len(pairs)}")
        return acc

    filings = cached(f"filings_{WINDOW_START}_{WINDOW_END}.json", pull_all)

    rows = []
    for tic, cik in pairs:
        fs = filings.get(str(cik), [])
        if not fs:
            drops["no_8k_2.02_in_window"] += 1
            continue
        f = sorted(fs, key=lambda x: x["filing_date"])[0]
        sess, kind = reaction_session(f["accepted"], f["filing_date"])
        if sess is None:
            drops[kind] += 1
            continue
        rec = {"ticker": tic, "cik": cik, "accepted": f["accepted"],
               "session": sess.isoformat(), "timing": kind}
        missing = False
        for field, _, _ in CONCEPTS:
            v = frames[field].get(cik)
            if v is None:
                missing = True
                break
            rec[field] = v
        if missing:
            drops["missing_gaap_figure"] += 1
            continue
        rows.append(rec)
    print(f"  {len(rows)} prints with all four GAAP figures and a usable session")

    print("[4/5] prices")
    import yfinance as yf
    tics = sorted({r["ticker"] for r in rows})
    px = yf.download(tics, start="2026-06-25", end="2026-08-25",
                     progress=False, auto_adjust=False, group_by="ticker")
    labelled = []
    for r in rows:
        try:
            sub = px[r["ticker"]].dropna(subset=["Open", "Close"])
        except Exception:
            drops["no_price_data"] += 1
            continue
        want = datetime.fromisoformat(r["session"]).date()
        idx = [d.date() for d in sub.index]
        nxt = [d for d in idx if d >= want]
        if not nxt:
            drops["no_session_on_or_after"] += 1
            continue
        day = nxt[0]
        if (day - want).days > 4:
            drops["session_too_far"] += 1
            continue
        o = float(sub.loc[str(day), "Open"])
        c = float(sub.loc[str(day), "Close"])
        if not (o > 0):
            drops["bad_open"] += 1
            continue
        r["react_date"] = day.isoformat()
        r["ret"] = (c - o) / o
        labelled.append(r)

    print(f"[5/5] {len(labelled)} labelled samples")
    print("\n  drop reasons:")
    for k, v in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v}")

    path = os.path.join(OUT, "dataset.json")
    with open(path, "w") as f:
        json.dump(labelled, f, indent=1)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
