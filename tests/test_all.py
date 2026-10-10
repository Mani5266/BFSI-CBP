"""PRD Section 12 — pytest suite. All must pass."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

def synth_fund():
    return pd.DataFrame([
        {"Symbol": "A", "MarketCap": 600, "PE": 10, "ROCE": 0.25},
        {"Symbol": "B", "MarketCap": 400, "PE": 10, "ROCE": 0.25},
        {"Symbol": "C", "MarketCap": 600, "PE": 20, "ROCE": 0.25},
        {"Symbol": "D", "MarketCap": 600, "PE": 10, "ROCE": 0.10},
    ])

def test_lab_query_exact():
    from src.query_engine import run
    out = run(synth_fund(), "MarketCap > 500 AND P/E < 15 AND ROCE > 22%")
    assert out["Symbol"].tolist() == ["A"]

def test_query_or_brackets_aliases():
    from src.query_engine import run
    df = synth_fund()
    assert set(run(df, "PE < 15 OR ROCE < 15%")["Symbol"]) == {"A", "B", "D"}
    assert set(run(df, "(MarketCap < 500 OR PE > 15) AND ROCE > 22%")["Symbol"]) == {"B", "C"}
    assert set(run(df, "PriceToEarning < 15 AND MCAP > 500")["Symbol"]) == {"A", "D"}

def test_fundamentals_formulas():
    from src.fundamentals import ratios_from_statements
    inc = pd.DataFrame({0: [1000.0, 200.0, 100.0, 10.0]},
                       index=["Total Revenue", "Operating Income", "Net Income", "Diluted EPS"])
    bs = pd.DataFrame({0: [1000.0, 400.0, 300.0, 500.0]},
                      index=["Total Assets", "Current Liabilities", "Total Debt", "Stockholders Equity"])
    r = ratios_from_statements(inc, bs, price=200.0, shares=50.0, mcap=10000.0)
    assert r["ROE_derived"] == 0.2 and r["ROA_derived"] == 0.1
    assert abs(r["ROCE_derived"] - 200 / 600) < 1e-9 and r["DE_derived"] == 0.6
    assert r["BookValuePS_derived"] == 10.0 and r["EarningsYield_derived"] == 0.05

def test_indicators_golden_ichimoku():
    from src.indicators import add_all
    from src.query_engine import golden_cross
    n = 260
    px = list(np.linspace(100, 80, 200)) + list(np.linspace(80, 140, 60))
    df = pd.DataFrame({"Open": px, "High": [p * 1.01 for p in px], "Low": [p * 0.99 for p in px],
                       "Close": px, "Volume": [1e6] * n})
    df = add_all(df)
    assert abs(df["SMA50"].iloc[-1] - np.mean(px[-50:])) < 1e-6
    assert 0 <= df["RSI14"].iloc[-1] <= 100
    assert golden_cross(df, within=90)  # long decline then rally must cross within window

def test_patterns_fire_and_quiet():
    from src.patterns import detect
    bull = pd.DataFrame({"Open": [100, 98], "High": [101, 103], "Low": [97, 97.5],
                         "Close": [98.5, 102], "Volume": [1e6, 2e6]})
    d = detect(bull)
    assert bool(d["BullEngulf"].iloc[-1])
    flat = pd.DataFrame({"Open": [100.0] * 5, "High": [100.5] * 5, "Low": [99.5] * 5,
                         "Close": [100.1, 99.9, 100.0, 100.2, 99.8], "Volume": [1e6] * 5})
    d2 = detect(flat)
    assert not d2[["Hammer", "BullEngulf", "BearEngulf", "ShootStar"]].any().any()

def test_wyckoff_spring():
    from src.wyckoff import detect_events
    n = 40
    base = {"Open": [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n,
            "Close": [100.0] * n, "Volume": [1e6] * n}
    df = pd.DataFrame(base)
    df.loc[39, "Low"] = 95.0  # undercut support 99
    df.loc[39, "Close"] = 100.5  # close back above
    df.loc[39, "Volume"] = 3e6
    ev = detect_events(df)
    assert "Spring" in ev["event"].iloc[-1]
    assert "vr=" in ev["evidence"].iloc[-1]
    plain = pd.DataFrame({k: ([100.0] * 30 if k != "Volume" else [1e6] * 30)
                          for k in ("Open", "High", "Low", "Close", "Volume")})
    plain["High"] = 100.2; plain["Low"] = 99.8
    assert "Spring" not in detect_events(plain)["event"].values

def test_cache_integrity_and_device():
    import json
    from pathlib import Path
    meta = json.loads(Path("data/meta.json").read_text())
    assert meta["retrieval_date"] and meta["n_stocks"] == 50
    missing = [k for k, v in meta["price_status"].items() if v == "failed"]
    assert not missing, f"missing prices: {missing}"
    assert Path("logs/env_check.txt").exists()
