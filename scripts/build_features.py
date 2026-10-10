"""Cache → feature store: per-stock daily+indicators parquet + fundamentals snapshot (PRD pipeline)."""
import sys, time, json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.device import log_device
from src.indicators import add_all

FEAT = BASE / "data" / "features"
FEAT.mkdir(exist_ok=True)

def main():
    log_device("build_features")
    t0 = time.time()
    from config import PRICES
    files = sorted(PRICES.glob("*.parquet"))
    n = 0
    for fp in files:
        df = pd.read_parquet(fp)
        df = add_all(df)
        df.to_parquet(FEAT / fp.name)
        n += 1
    print(f"features: {n} stocks in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
