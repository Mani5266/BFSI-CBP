"""Plain-text rule engine: screener-style (M2) + Chartink-style scans (M5). One parser, both."""
import re
import pandas as pd
import numpy as np

ALIASES = {
    "PE": "PE", "PRICETOEARNING": "PE", "P/E": "PE", "P_E": "PE",
    "MARKETCAP": "MarketCap", "MCAP": "MarketCap",
    "ROCE": "ROCE", "ROE": "ROE", "ROA": "ROA", "DE": "DE", "D/E": "DE",
    "DIVYIELD": "DivYield", "DIV YIELD": "DivYield", "BETA": "Beta",
    "PB": "PB", "PRICETOBOOK": "PB", "EVEBITDA": "EVEBITDA",
    "RSI": "RSI14", "DMA50": "DMA50", "DMA200": "DMA200",
}

def _norm(field: str) -> str:
    key = re.sub(r"[^A-Z/]", "", field.upper().replace(" ", ""))
    key2 = field.upper().replace(" ", "").replace("_", "").replace("/", "")
    return ALIASES.get(key, ALIASES.get(key2, field))

TOKEN = re.compile(r"(\bAND\b|\bOR\b|\(|\)|>=|<=|!=|=|>|<)", re.IGNORECASE)

def parse(query: str):
    """Shunting-yard → RPN over (field, op, value) conditions. % tolerated: ROCE > 22% → 22."""
    toks = [t.strip() for t in TOKEN.split(query) if t and t.strip()]
    out, ops = [], []
    prec = {"OR": 1, "AND": 2}
    i = 0
    while i < len(toks):
        t = toks[i]
        u = t.upper()
        if u in ("AND", "OR"):
            while ops and ops[-1] != "(" and prec.get(ops[-1], 0) >= prec[u]:
                out.append(ops.pop())
            ops.append(u)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            ops.pop()
        else:  # field op value — a trailing % means percent: ROCE > 22% → 0.22
            if i + 2 < len(toks) and toks[i + 1] in (">", "<", ">=", "<=", "=", "!=", "=="):
                raw = toks[i + 2]
                val = float(raw.replace("%", "")) / (100.0 if "%" in raw else 1.0)
                out.append((_norm(t), toks[i + 1], val))
                i += 2
            else:
                raise ValueError(f"Bad condition near: {t}")
        i += 1
    while ops:
        out.append(ops.pop())
    return out

def run(df: pd.DataFrame, query: str) -> pd.DataFrame:
    rpn = parse(query)
    stack = []
    for tok in rpn:
        if isinstance(tok, str):
            b, a = stack.pop(), stack.pop()
            stack.append(a | b if tok == "OR" else a & b)
        else:
            f, op, v = tok
            if f not in df.columns:
                raise ValueError(f"Unknown field '{f}'. Available incl: {sorted(set(ALIASES.values()))}")
            s = pd.to_numeric(df[f], errors="coerce")
            stack.append({"==": s == v, "=": s == v, "!=": s != v, ">": s > v,
                          "<": s < v, ">=": s >= v, "<=": s <= v}[op].fillna(False))
    mask = stack[0] if stack else pd.Series(True, index=df.index)
    return df[mask].copy()

# ---- M5 Chartink-style scans over per-stock daily frames ----
def golden_cross(df: pd.DataFrame, within: int = 5) -> bool:
    """DMA50 > DMA200 now AND a cross happened within last `within` days."""
    if len(df) < 205:
        return False
    diff = df["DMA50"] - df["DMA200"]
    if not (diff.iloc[-1] > 0):
        return False
    return bool((diff.shift(1) < 0).iloc[-within:].any())

def ichimoku_bull(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    above = bool(last["Close"] > max(last["Ichi_senkou_a"], last["Ichi_senkou_b"]))
    cross = bool((df["Close"] > df[["Ichi_senkou_a", "Ichi_senkou_b"]].max(axis=1)).iloc[-5:].any()
                 and not (df["Close"] > df[["Ichi_senkou_a", "Ichi_senkou_b"]].max(axis=1)).iloc[-6])
    return {"above_cloud": above, "cloud_crossover_5d": cross}
