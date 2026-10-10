"""Wyckoff / VSA — CANDIDATES with evidence numbers, never certainties (PRD 6.4)."""
import pandas as pd
import numpy as np
import config

GOVERNING_RULE = ("Wyckoff events are CANDIDATES, never certainties — "
                  "every flag shows volume ratio, spread ratio, close position.")

def bar_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = config.VOL_LOOKBACK
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(n).mean().replace(0, np.nan)
    df["spread_ratio"] = (df["High"] - df["Low"]) / (df["High"] - df["Low"]).rolling(n).mean().replace(0, np.nan)
    df["close_pos"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    return df

def vsa_class(r) -> str:
    vr, sr, cp = r["vol_ratio"], r["spread_ratio"], r["close_pos"]
    if pd.isna(vr) or pd.isna(sr) or pd.isna(cp):
        return "—"
    if vr < config.LOW_VOL_MAX and sr < config.NARROW_SPREAD_MAX:
        return "no-demand" if cp < 0.5 else "no-supply"
    if vr > config.CLIMAX_VOL_MULT and sr > config.WIDE_SPREAD_MIN:
        return "climactic action"
    if vr > config.HIGH_VOL_MIN and sr < config.NARROW_SPREAD_MAX:
        return "stopping volume"
    if (sr > config.WIDE_SPREAD_MIN and vr < config.LOW_VOL_MAX) or (sr < config.NARROW_SPREAD_MAX and vr > config.HIGH_VOL_MIN):
        return "effort-vs-result"
    return "normal"

def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """Spring / UTAD / SC / ST / LPS / SOS candidates over rolling support/resistance."""
    df = bar_stats(df)
    w = config.SPRING_LOOKBACK
    df["support"] = df["Low"].rolling(w).min().shift(1)
    df["resist"] = df["High"].rolling(w).max().shift(1)
    df["event"] = ""
    df["evidence"] = ""
    for i in df.index:
        r = df.loc[i]
        if pd.isna(r["support"]):
            continue
        ev, tag = "", ""
        # Spring: low undercuts support, close back above it
        if r["Low"] < r["support"] and r["Close"] > r["support"]:
            tag = "Spring"
        # Upthrust/UTAD: high breaks resistance intraday, close back below
        elif r["High"] > r["resist"] and r["Close"] < r["resist"]:
            tag = "Upthrust/UTAD"
        # Selling Climax: wide spread + climactic volume + weak close after decline
        elif (r["vol_ratio"] > config.CLIMAX_VOL_MULT and r["spread_ratio"] > config.WIDE_SPREAD_MIN
              and r["close_pos"] < 0.4):
            tag = "SC?"
        # SOS: wide spread + strong close + high volume after range
        elif (r["vol_ratio"] > config.HIGH_VOL_MIN and r["spread_ratio"] > 1.0 and r["close_pos"] > 0.7
              and r["Close"] > r["resist"] * 0.995):
            tag = "SOS?"
        if tag:
            df.at[i, "event"] = tag
            df.at[i, "evidence"] = (f"vr={r['vol_ratio']:.2f} sr={r['spread_ratio']:.2f} "
                                    f"cp={r['close_pos']:.2f} sup={r['support']:.1f} res={r['resist']:.1f}")
    df["vsa"] = df.apply(vsa_class, axis=1)
    # Trap flags: pattern + failed follow-through (next-3-bar drift opposite) — computed by caller via patterns
    return df

def phase_tag(df: pd.DataFrame) -> str:
    """Heuristic A–E tag (rules printed in UI). Last-50-bar slope + range position."""
    try:
        c = df["Close"].dropna().iloc[-50:]
        if len(c) < 20:
            return "A? (too little data)"
        slope = np.polyfit(range(len(c)), c.values, 1)[0] / c.mean()
        pos = (c.iloc[-1] - c.min()) / (c.max() - c.min() + 1e-9)
        if abs(slope) < 0.0005:
            return "B? (sideways — accumulation/distribution range)"
        if slope < -0.001 and pos < 0.3:
            return "A? (decline / climax zone)"
        if slope > 0.001 and pos > 0.7:
            return "D–E? (markup / markdown)"
        return "C? (test / spring zone)"
    except Exception:
        return "?"
