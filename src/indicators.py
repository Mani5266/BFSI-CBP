"""Technicals per stock (PRD 6.2). pandas only; Alligator composed by hand (~10 lines)."""
import pandas as pd
import numpy as np

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def smma(s: pd.Series, n: int) -> pd.Series:  # Wilder smoothing — Alligator needs it
    return s.ewm(alpha=1 / n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series, fast=12, slow=26, sig=9):
    m = ema(close, fast) - ema(close, slow)
    return m, m.ewm(span=sig, adjust=False).mean(), m - m.ewm(span=sig, adjust=False).mean()

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    ll, hh = low.rolling(k).min(), high.rolling(k).max()
    kline = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return kline, kline.rolling(d).mean()

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             t=9, k=26, s=52, disp=26):
    """No-lookahead: spans shifted FORWARD only (+disp), Chikou shifted BACK (−disp)."""
    tenkan = (high.rolling(t).max() + low.rolling(t).min()) / 2
    kijun = (high.rolling(k).max() + low.rolling(k).min()) / 2
    a = ((tenkan + kijun) / 2).shift(disp)
    b = ((high.rolling(s).max() + low.rolling(s).min()) / 2).shift(disp)
    chikou = close.shift(-disp)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": a, "senkou_b": b, "chikou": chikou}

def alligator(high: pd.Series, low: pd.Series):
    """Composed by hand: SMMA of median price; Jaw 13/8, Teeth 8/5, Lips 5/3. In NO library."""
    med = (high + low) / 2
    return {"jaw": smma(med, 13).shift(8), "teeth": smma(med, 8).shift(5), "lips": smma(med, 5).shift(3)}

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)

def fib_levels(swing_lo: float, swing_hi: float) -> dict:
    return {r: swing_hi - (swing_hi - swing_lo) * r for r in (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)}

def pivots(df: pd.DataFrame, order: int = 5):
    """Local extrema indices via scipy; swing = last major low→high for Fib."""
    from scipy.signal import argrelextrema
    c = df["Close"].values
    hi = argrelextrema(c, np.greater, order=order)[0]
    lo = argrelextrema(c, np.less, order=order)[0]
    return {"high_idx": df.index[hi].tolist(), "low_idx": df.index[lo].tolist()}

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add every indicator column. Swing rule for Fib (in code + report): last pivot low→high."""
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA50"], df["DMA50"] = sma(c, 50), sma(c, 50)
    df["SMA200"], df["DMA200"] = sma(c, 200), sma(c, 200)
    df["EMA12"], df["EMA26"] = ema(c, 12), ema(c, 26)
    df["RSI14"] = rsi(c)
    m, s, hh = macd(c)
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = m, s, hh
    k, d = stochastic(h, l, c)
    df["StochK"], df["StochD"] = k, d
    for key, val in ichimoku(h, l, c).items():
        df[f"Ichi_{key}"] = val
    for key, val in alligator(h, l).items():
        df[f"Gator_{key}"] = val
    df["VWAP"] = vwap(df)
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RANGE20"] = (df["High"] - df["Low"]).rolling(20).mean()
    return df

def breadth(features: dict) -> pd.Series:
    """% of universe above 200-DMA (M7 gauge input)."""
    closes = pd.DataFrame({s: df["Close"] for s, df in features.items()})
    above = closes.gt(closes.rolling(200).mean())
    return above.mean(axis=1) * 100
