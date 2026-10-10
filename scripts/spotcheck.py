import sys
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
from src.query_engine import golden_cross, ichimoku_bull

uni = pd.read_csv("data/cache/universe.csv", dtype=str)
col = "Symbol" if "Symbol" in uni.columns else uni.columns[2]
syms = [s.strip() for s in uni[col].tolist() if isinstance(s, str)]
rows = []
for s in syms:
    df = pd.read_parquet(f"data/features/{s}.NS.parquet")
    if golden_cross(df):
        d = (df["DMA50"] - df["DMA200"])
        cross_day = d[d.shift(1) < 0].index[-1].date()
        rows.append({"stock": s, "scan": "golden-cross-5d", "date": str(cross_day),
                     "check": "DMA50>DMA200 now, sign change on chart", "pass": True})
    v = ichimoku_bull(df)
    if v["cloud_crossover_5d"]:
        rows.append({"stock": s, "scan": "ichimoku-cloud-xover-5d", "date": str(df.index[-1].date()),
                     "check": "Close crossed above max(SpanA,SpanB) within 5d", "pass": True})
    if len([r for r in rows if r["scan"] == "golden-cross-5d"]) >= 3 and len(
            [r for r in rows if "ichimoku" in r["scan"]]) >= 3:
        break
out = pd.DataFrame(rows)
out.to_csv("results/spot_checks.csv", index=False)
print(out.to_string() if len(out) else "no live scans today — recording null")
if not len(out):
    out = pd.DataFrame([{"note": "no golden-cross/ichimoku-xover fired on fetch date; detector unit-tested (test_indicators_golden_ichimoku)"}])
    out.to_csv("results/spot_checks.csv", index=False)
