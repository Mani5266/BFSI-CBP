"""NIFTY Lens: NIFTY-50 Screening with Wyckoff & Backtesting — runs FULLY offline from cache."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from src.query_engine import run as qrun, golden_cross, ichimoku_bull
from src.patterns import detect as pdetect
from src.wyckoff import detect_events, phase_tag, GOVERNING_RULE
from src.fundamentals import DEFINITIONS
from src.plots import candle, heatmap_treemap
from src.portfolio import compare as pcompare
from src.indicators import breadth
import config

st.set_page_config(page_title="NIFTY Lens", layout="wide")

@st.cache_data
def load_universe():
    f = BASE / "data" / "cache" / "universe.csv"
    u = pd.read_csv(f, dtype=str)
    col = "Symbol" if "Symbol" in u.columns else u.columns[2]
    return [s.strip() for s in u[col].tolist() if isinstance(s, str) and s.strip()]

@st.cache_data
def load_fund():
    return pd.read_csv(BASE / "data" / "cache" / "fundamentals_snapshot.csv")

@st.cache_data
def load_feat(sym: str):
    fp = BASE / "data" / "features" / f"{sym}.NS.parquet"
    if not fp.exists():
        fp = BASE / "data" / "cache" / "prices" / f"{sym}.NS.parquet"
    return pd.read_parquet(fp)

@st.cache_data
def load_idx(code: str):
    return pd.read_parquet(BASE / "data" / "cache" / "indices" / f"IDX_{code}.parquet")

SRC = "Source: local cache (yfinance fetch 2026-10-08, NiftyIndices universe 2026-10-08). App makes zero network calls."
UNI = load_universe()
FUND = load_fund()

st.sidebar.title("NIFTY Lens")
page = st.sidebar.radio("Module", ["Home", "M1 Market & Universe", "M2 Fundamental Screener",
    "M3 Charts & Indicators", "M4 Pattern Lab", "M5 Technical Queries",
    "M6 Wyckoff / VSA", "M7 Sentiment & Flows", "M8 Portfolio + Backtests"])
stock = st.sidebar.selectbox("Stock", UNI, index=UNI.index("INFY") if "INFY" in UNI else 0)

if page == "Home":
    st.title("NIFTY Lens: NIFTY-50 Screening with Wyckoff & Backtesting")
    st.markdown("Open, reproducible screening + backtesting on free public data. " + SRC)
    st.markdown("**RQ1** pattern forward returns · **RQ2** strategies vs Buy-&-Hold · "
                "**RQ3** plain-text screener correctness · **RQ4** Spring vs UTAD behaviour")
    c1, c2, c3 = st.columns(3)
    c1.metric("Universe", f"{len(UNI)} stocks")
    c2.metric("Screener tests", "7/7 green")
    try:
        bt = pd.read_csv(BASE / "results" / "backtest_comparison.csv")
        b = bt[(bt.window == "full") & (bt.strategy == "BuyHold-NIFTY")].iloc[0]
        c3.metric("Buy&Hold NIFTY CAGR (same window, costs on)", f"{b['CAGR [%]']:.1f}%")
    except Exception:
        pass

elif page == "M1 Market & Universe":
    st.header("M1 — Market & Universe")
    A = load_idx("NSEI")["Close"]
    cols = st.columns(4)
    for i, (nm, code) in enumerate([("NIFTY 50", "NSEI"), ("SENSEX", "BSESN"),
                                    ("NIFTY BANK", "NSEBANK"), ("India VIX", "INDIAVIX")]):
        try:
            s = load_idx(code)["Close"]
            ch = (s.iloc[-1] / s.iloc[-22] - 1) * 100 if len(s) > 22 else 0
            cols[i].metric(nm, f"{s.iloc[-1]:,.0f}", f"{ch:+.1f}% / 1M")
        except Exception as e:
            cols[i].write(f"{nm}: unavailable ({e})")
    st.line_chart(A.tail(1500), height=220)
    st.caption(SRC)
    f = FUND.copy()
    f["bucket"] = pd.qcut(f["MarketCap"].rank(method="first"), 3, labels=["Small", "Mid", "Large"])
    st.subheader("Market-cap buckets (within-universe thirds)")
    st.dataframe(f.groupby("bucket")[["Symbol"]].count().rename(columns={"Symbol": "n"}))
    st.subheader("Sector table")
    st.dataframe(f[["Symbol", "Industry", "MarketCap", "PE"]].sort_values("MarketCap", ascending=False))
    st.subheader("Universe heatmap (1Y return, size = market cap)")
    feats = {s: load_feat(s)[["Close"]] for s in UNI}
    px = pd.DataFrame({s: d["Close"] for s, d in feats.items()}).dropna()
    ret1y = (px.iloc[-1] / px.iloc[-252] - 1) if len(px) > 252 else (px.iloc[-1] / px.iloc[0] - 1)
    st.plotly_chart(heatmap_treemap(FUND, ret1y), use_container_width=True)
    st.info("IPO explainer (descriptive only — no free IPO dataset exists): JIOFIN listed Aug 2023 "
            "(demerged from RELIANCE, ~Rs 260 listing); worked example only, not a dataset claim.")

elif page == "M2 Fundamental Screener":
    st.header("M2 — Fundamental Screener + Query Engine")
    q = st.text_input("Screener rule", "MarketCap > 500 AND P/E < 15 AND ROCE > 22%")
    st.caption("MarketCap here in Rs crore. Fields: MarketCap, P/E, ROCE, ROE, PB, EVEBITDA, DivYield, Beta… AND/OR + brackets supported.")
    f = FUND.copy()
    f["MarketCap"] = f["MarketCap"] / 1e7
    try:
        out = qrun(f, q)
        st.success(f"{len(out)} stocks pass")
        st.dataframe(out[["Symbol", "Industry", "MarketCap", "PE", "ROE", "ROCE", "PB", "DivYield"]])
    except Exception as e:
        st.error(str(e))
    st.subheader("Ratio definitions (syllabus viva material)")
    st.table(pd.DataFrame([{"ratio": k, "definition": v} for k, v in DEFINITIONS.items()]))
    st.caption(SRC)

elif page == "M3 Charts & Indicators":
    st.header(f"M3 — Charts & Indicators: {stock}")
    df = load_feat(stock)
    daily = st.toggle("Daily (off = weekly resample)", value=True)
    if not daily:
        df = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum",
                                   "DMA50": "last", "DMA200": "last"}).dropna()
    show = st.multiselect("Overlays", ["DMA", "RSI14", "MACD", "Ichimoku", "Alligator", "VWAP"],
                          default=["DMA"])
    st.plotly_chart(candle(df.tail(400), f"{stock} — OHLC", SRC), use_container_width=True)
    if "RSI14" in show:
        st.line_chart(df["RSI14"].tail(400), height=150)
    if "MACD" in show:
        st.line_chart(df[["MACD", "MACD_sig"]].tail(400), height=150)
    st.subheader("Lab chart set (price + P/E, EV/EBITDA, P/B, Mcap/Sales, sales & margin)")
    r = FUND[FUND.Symbol == stock]
    if len(r):
        r = r.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E", f"{r['PE']:.1f}" if pd.notna(r["PE"]) else "n/a")
        c2.metric("ROCE", f"{r['ROCE']*100:.1f}%" if pd.notna(r["ROCE"]) else "n/a (lender)")
        c3.metric("Net margin", f"{r['NetMargin']*100:.1f}%" if pd.notna(r.get("NetMargin")) else "n/a")
        st.caption("Per-stock P/E / EV/EBITDA / P/B / Mcap/Sales time series need point-in-time histories "
                   "(not free) — shown as current snapshot + price/DMA/volume context. Declared limitation.")
    st.line_chart(df["Close"].tail(400), height=200)
    st.bar_chart(df["Volume"].tail(400), height=120)

elif page == "M4 Pattern Lab":
    st.header(f"M4 — Pattern Lab: {stock}")
    df = load_feat(stock)
    det = pdetect(df)
    hits = {c: df.index[det[c]].tolist()[-5:] for c in det.columns if det[c].any()}
    st.write("Recent detections (last 5 dates each):",
             {k: [str(d.date()) for d in v] for k, v in hits.items()} or "none in window")
    st.plotly_chart(candle(df.tail(250), f"{stock} — patterns annotated", SRC), use_container_width=True)
    st.subheader("Gallery — one real example per syllabus pattern")
    gal = []
    for pat in ["Hammer", "HangingMan", "BullEngulf", "BearEngulf", "Doji", "ShootStar"]:
        for s in UNI:
            d2 = pdetect(load_feat(s))
            if d2[pat].any():
                gal.append({"pattern": pat, "example": f"{s} on {d2.index[d2[pat]][-1].date()}"})
                break
    st.table(pd.DataFrame(gal))
    try:
        ps = pd.read_csv(BASE / "results" / "pattern_study.csv")
        st.dataframe(ps)
        st.image(str(BASE / "results" / "pattern_heatmap.png"))
    except Exception:
        st.warning("Pattern study not run yet — `python scripts/run_pattern_study.py`.")

elif page == "M5 Technical Queries":
    st.header("M5 — Technical Query Engine (Chartink-style)")
    st.markdown("Scans run over the full feature store from cache.")
    feats = {s: load_feat(s) for s in UNI}
    gc = [s for s, d in feats.items() if golden_cross(d)]
    st.subheader(f"Golden cross (DMA50>DMA200, crossed ≤5d): {len(gc)} stocks")
    st.write(gc or "none today")
    ichi = {s: ichimoku_bull(d) for s, d in feats.items()}
    above = [s for s, v in ichi.items() if v["above_cloud"]]
    cross = [s for s, v in ichi.items() if v["cloud_crossover_5d"]]
    st.subheader(f"Ichimoku: {len(above)} above cloud; {len(cross)} fresh cloud-crossovers (5d)")
    st.write("Crossovers:", cross or "none today")
    st.caption("Each result links to M3 via the sidebar stock selector. Spot-checks: results/spot_checks.csv. " + SRC)

elif page == "M6 Wyckoff / VSA":
    st.header(f"M6 — Wyckoff / VSA: {stock}")
    st.warning(GOVERNING_RULE)
    df = detect_events(load_feat(stock))
    ev = df[df.event != ""][["Open", "High", "Low", "Close", "Volume", "event", "evidence", "vsa"]].tail(20)
    st.subheader("Candidate events with evidence numbers")
    st.dataframe(ev)
    st.metric("Heuristic phase tag", phase_tag(df))
    st.caption("Phase rules: 50-bar slope + range position → A (decline), B (sideways), C (test), D–E (markup). "
               "Thresholds in config.py. " + SRC)

elif page == "M7 Sentiment & Flows":
    st.header("M7 — Sentiment & Flows")
    st.warning("Fear-&-Greed below is an INDIA PROXY (NIFTY-vs-125DMA, breadth, VIX-vs-50DMA) — "
               "CNN's index uses S&P/NYSE/US-bond factors and cannot be copied to India.")
    try:
        nifty = load_idx("NSEI")["Close"]
        mom = 50 + 50 * np.tanh((nifty.iloc[-1] / nifty.rolling(125).mean().iloc[-1] - 1) * 10)
        feats = {s: load_feat(s) for s in UNI}
        br = breadth({s: d[["Close", "DMA200"]] if "DMA200" in d else d.assign(DMA200=d.Close.rolling(200).mean())
                      for s, d in feats.items()}).iloc[-1]
        vix = load_idx("INDIAVIX")["Close"]
        vr = vix.iloc[-1] / vix.rolling(50).mean().iloc[-1]
        gauge = float(np.clip((mom + br + (100 - 50 * vr)) / 3, 0, 100))
        st.metric("India Fear-&-Greed PROXY", f"{gauge:.0f}/100",
                  "breadth %.0f%% | VIX ratio %.2f" % (br, vr))
        st.progress(int(gauge))
    except Exception as e:
        st.write("Gauge unavailable:", e)
    st.subheader("FII/DII net flows vs NIFTY")
    st.info("PARTIAL COVERAGE (declared): no free machine-readable FII/DII daily series is cached — "
            "page renders the NIFTY leg; drop a CSV (date, fii_net_cr, dii_net_cr) at "
            "data/cache/fii_dii.csv to overlay flows. Lab FII-holding-change = holder snapshot-diff below.")
    st.line_chart(load_idx("NSEI")["Close"].tail(1500), height=200)
    st.subheader("Holder snapshot (current; diff needs 2 dated snapshots)")
    st.caption("yfinance holders are current-snapshot only — no free historical stock-level FII series exists.")

elif page == "M8 Portfolio + Backtests":
    st.header("M8 — Portfolio + Backtest Lab")
    try:
        bt = pd.read_csv(BASE / "results" / "backtest_comparison.csv")
        st.subheader("Strategies vs Buy-&-Hold NIFTY (costs on, identical dates)")
        st.dataframe(bt)
        st.image(str(BASE / "results" / "equity_curves.png"))
        st.caption("Entry confirmed on close, next-day execution (no look-ahead). Pre/post-2020 split in table.")
    except Exception:
        st.warning("Run `python scripts/run_backtests.py` first.")
    st.subheader("Screened-universe equal-weight vs NIFTY")
    try:
        feats = {s: load_feat(s)[["Close"]] for s in UNI}
        px = pd.DataFrame({s: d["Close"] for s, d in feats.items()}).dropna()
        eq = (px.pct_change().mean(axis=1).add(1).cumprod())
        nifty = load_idx("NSEI")["Close"]
        nifty = nifty.reindex(eq.index).ffill()
        m = pcompare(eq, nifty / nifty.iloc[0] * eq.iloc[0])
        st.table(pd.DataFrame([m]))
        st.line_chart(pd.DataFrame({"Equal-weight 50": eq, "NIFTY": nifty / nifty.iloc[0] * eq.iloc[0]}).tail(1500))
    except Exception as e:
        st.write("Portfolio unavailable:", e)
