"""Cleaning: tz, adjustment note, missing/outlier checks. CPU work by design (PRD 3.3)."""
import pandas as pd
import numpy as np

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a yfinance OHLCV frame: tz-naive index, sorted, ffill ≤3, flag gaps."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # yfinance multi-ticker frames have MultiIndex columns — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            df[c] = df[c].ffill(limit=3)
    return df

def quality_report(df: pd.DataFrame) -> dict:
    """% missing bars, % missing closes, negative-price and spike flags."""
    if df is None or df.empty:
        return {"rows": 0, "missing_pct": 100.0, "note": "empty"}
    n = len(df)
    missing = float(df["Close"].isna().mean() * 100) if "Close" in df else 100.0
    neg = int(((df["Close"] <= 0).sum()) if "Close" in df else 0)
    # >25% daily jump flags (splits already adjusted with auto_adjust=True; leftovers get flagged)
    spike = 0
    if "Close" in df and n > 1:
        ret = df["Close"].pct_change().abs()
        spike = int((ret > 0.25).sum())
    # expected-trading-day gaps: business days missing
    expected = len(pd.bdate_range(df.index.min(), df.index.max())) if n else 0
    gap_pct = round(max(0, (expected - n)) / expected * 100, 2) if expected else 0.0
    return {"rows": n, "from": str(df.index.min().date()), "to": str(df.index.max().date()),
            "missing_close_pct": round(missing, 2), "neg_prices": neg,
            "spikes_gt25pct": spike, "missing_bizday_pct": gap_pct}
