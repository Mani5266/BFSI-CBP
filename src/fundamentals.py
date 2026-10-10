"""Fundamentals — EVERY syllabus ratio from .info snapshot + statements (PRD 6.1).

Derivations (printed in app + report):
  ROE  = Net Income / Stockholders' Equity            (derived when .info lacks it)
  ROA  = Net Income / Total Assets
  ROCE = EBIT / (Total Assets − Current Liabilities)  (yfinance has NO ROCE field)
  D/E  = Total Debt / Equity;  Earnings Yield = EPS / Price
"""
import pandas as pd
import numpy as np

DEFINITIONS = {
    "P/E": "Price / trailing EPS (trailingPE; forwardPE shown separately)",
    "Industry P/E": "Median trailing P/E of same-Industry peers in this universe",
    "P/B": "Price / Book Value per share",
    "Book Value/share": "Stockholders' Equity / shares outstanding",
    "EV/EBITDA": "Enterprise Value / EBITDA",
    "MarketCap/Sales": "Market Cap / trailing revenue (TTM)",
    "PEG": "P/E / earnings growth rate",
    "EPS": "Trailing EPS (diluted where available)",
    "EBITDA": "Earnings before Interest, Tax, Depreciation & Amortisation",
    "Dividend Yield": "Dividend rate / Price",
    "Earnings Yield": "EPS / Price (= 1/PE, comparable across stocks)",
    "Face Value": "Nominal share value (NSE master / info, flagged by source)",
    "ROE": "Net Income / Stockholders' Equity — derived from statements when .info lacks it",
    "ROA": "Net Income / Total Assets",
    "ROCE": "EBIT / (Total Assets − Current Liabilities) — required derivation",
    "D/E": "Total Debt / Equity",
    "Revenue growth": "YoY total-revenue growth (latest annual)",
    "Profit growth": "YoY net-income growth (latest annual)",
    "Revenue CAGR 3Y/5Y": "(End/Start)^(1/n) − 1 on annual total revenue",
    "Profit CAGR 3Y/5Y": "(End/Start)^(1/n) − 1 on annual net income",
    "Operating margin": "Operating Income / Total Revenue",
    "Net margin": "Net Income / Total Revenue",
}

def _first(df, names):
    """Exact (case-insensitive) match first, then word-boundary, then contains."""
    import re
    if df is None or df.empty:
        return None
    labels = list(df.index)
    low = [str(i).strip().lower() for i in labels]
    for n in names:
        nl = n.strip().lower()
        for i, lab in enumerate(low):
            if lab == nl:
                return labels[i]
    for n in names:
        rx = re.compile(r"\b" + re.escape(n.strip().lower()) + r"\b")
        for i, lab in enumerate(low):
            if rx.search(lab):
                return labels[i]
    for n in names:
        nl = n.strip().lower()
        for i, lab in enumerate(low):
            if nl in lab:
                return labels[i]
    return None

def _val(df, names, col=0):
    try:
        r = _first(df, names)
        if r is None:
            return np.nan
        v = df.loc[r].iloc[col] if hasattr(df.loc[r], "iloc") else df.loc[r]
        return float(v)
    except Exception:
        return np.nan

def ratios_from_statements(inc, bs, price=np.nan, shares=np.nan, mcap=np.nan) -> dict:
    """Derive ratios from annual statements (latest column = most recent year)."""
    out = {}
    ni = _val(inc, ["net income", "net earnings"])
    eq = _val(bs, ["total stockholders", "stockholders' equity", "stockholders equity", "total equity"])
    ta = _val(bs, ["total assets"])
    cl = _val(bs, ["total current liabilities", "current liabilities"])
    ebit = _val(inc, ["ebit"])  # exact 'EBIT' row; else Operating Income below
    if np.isnan(ebit):
        oi = _val(inc, ["operating income", "operating profit"])
        ebit = oi  # ponytail: EBIT≈Operating Income when EBIT row absent; flagged in report
    debt = _val(bs, ["total debt"])
    if np.isnan(debt):
        debt = _val(bs, ["long term debt", "short term debt", "borrowings"])
    rev = _val(inc, ["total revenue", "total revenues"])
    ebitda = _val(inc, ["ebitda"])
    eps = _val(inc, ["diluted eps", "basic eps", "eps"])
    out["ROE_derived"] = ni / eq if eq else np.nan
    out["ROA_derived"] = ni / ta if ta else np.nan
    cap_emp = ta - cl if (not np.isnan(ta) and not np.isnan(cl)) else np.nan
    out["ROCE_derived"] = ebit / cap_emp if (not np.isnan(ebit) and cap_emp) else np.nan
    out["DE_derived"] = debt / eq if eq else np.nan
    out["BookValuePS_derived"] = eq / shares if (not np.isnan(eq) and shares) else np.nan
    out["EarningsYield_derived"] = eps / price if (not np.isnan(eps) and price) else np.nan
    try:  # growth + CAGR from annual columns (newest first or oldest first — normalise)
        revs = pd.to_numeric(inc.loc[_first(inc, ["total revenue", "total revenues"])], errors="coerce").dropna().values
        nis = pd.to_numeric(inc.loc[_first(inc, ["net income", "net earnings"])], errors="coerce").dropna().values
        if len(revs) >= 2:
            a, b = revs[0], revs[1]
            out["RevenueGrowth"] = (a - b) / abs(b) if b else np.nan
        for n, key in ((3, "RevenueCAGR3Y"), (5, "RevenueCAGR5Y")):
            if len(revs) > n and revs[n] and revs[n] > 0 and revs[0] > 0:
                out[key] = (revs[0] / revs[n]) ** (1 / n) - 1
        if len(nis) >= 2:
            out["ProfitGrowth"] = (nis[0] - nis[1]) / abs(nis[1]) if nis[1] else np.nan
        for n, key in ((3, "ProfitCAGR3Y"), (5, "ProfitCAGR5Y")):
            if len(nis) > n and nis[n] and nis[n] > 0 and nis[0] > 0:
                out[key] = (nis[0] / nis[n]) ** (1 / n) - 1
        oi = _val(inc, ["operating income", "operating profit"])
        out["OpMargin"] = oi / rev if (not np.isnan(oi) and rev) else np.nan
        out["NetMargin"] = ni / rev if (not np.isnan(ni) and rev) else np.nan
        out["MarketCapSales"] = mcap / rev if (not np.isnan(mcap) and rev) else np.nan
    except Exception:
        pass
    return out

INFO_MAP = {  # snapshot field → (ratio col, transform)
    "trailingPE": "PE", "forwardPE": "PE_fwd", "priceToBook": "PB",
    "enterpriseToEbitda": "EVEBITDA", "pegRatio": "PEG", "marketCap": "MarketCap",
    "enterpriseValue": "EV", "returnOnEquity": "ROE_info", "returnOnAssets": "ROA_info",
    "revenueGrowth": "RevenueGrowth_info", "earningsGrowth": "EarningsGrowth_info",
    "dividendYield": "DivYield", "dividendRate": "DivRate", "beta": "Beta",
    "sector": "Sector", "industry": "Industry", "fiftyDayAverage": "DMA50_info",
    "twoHundredDayAverage": "DMA200_info", "trailingEps": "EPS_info",
    "ebitda": "EBITDA_info", "totalRevenue": "Revenue_info",
}

def snapshot_row(symbol: str, info: dict, stmt_ratios: dict, price: float) -> dict:
    row = {"Symbol": symbol, "Price": price, "fetch_date": pd.Timestamp.today().date().isoformat()}
    info = info or {}
    for k, col in INFO_MAP.items():
        row[col] = info.get(k)
    # Prefer derived ROE/ROCE (yfinance often lacks them for NSE); keep both, final = derived-first
    for final, derived, inf in (("ROE", "ROE_derived", "ROE_info"), ("ROA", "ROA_derived", "ROA_info")):
        d, i = stmt_ratios.get(derived), row.get(inf)
        row[final] = d if d is not None and not (isinstance(d, float) and np.isnan(d)) else i
        row[final + "_src"] = "derived" if row[final] == d and d == d else "info"
    row["ROCE"] = stmt_ratios.get("ROCE_derived")
    row["ROCE_src"] = "derived"
    row["DE"] = stmt_ratios.get("DE_derived")
    for k in ("BookValuePS_derived", "EarningsYield_derived", "RevenueGrowth", "ProfitGrowth",
              "RevenueCAGR3Y", "RevenueCAGR5Y", "ProfitCAGR3Y", "ProfitCAGR5Y",
              "OpMargin", "NetMargin", "MarketCapSales"):
        row[k] = stmt_ratios.get(k)
    return row

def add_industry_pe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Industry" in df.columns and "PE" in df.columns:
        df["IndustryPE"] = df.groupby("Industry")["PE"].transform("median")
    return df
