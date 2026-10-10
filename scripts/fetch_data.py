"""ONLY network component → data/cache/. Everything else runs offline (PRD 4.5)."""
import json, sys, time, urllib.request
from datetime import date
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import config
from src.data_clean import clean_prices, quality_report
from src.fundamentals import ratios_from_statements, snapshot_row, add_industry_pe

def fetch_universe():
    """NiftyIndices CSV → data/cache/universe.csv (+ retrieval date). Fallback: config list."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
    out = config.CACHE / "universe.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        from io import StringIO
        df = pd.read_csv(StringIO(txt), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(out, index=False)
        print(f"universe: {len(df)} rows from NiftyIndices")
        return df, "niftyindices", date.today().isoformat()
    except Exception as e:
        print(f"universe fetch FAILED ({e}); fallback to config.NIFTY50")
        df = pd.DataFrame({"Symbol": config.NIFTY50})
        df.to_csv(out, index=False)
        return df, "config-fallback", date.today().isoformat()

def _retry(fn, tries=3, sleep=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"  retry {i+1}/{tries}: {e}")
            time.sleep(sleep * (i + 1))
    return None

def fetch_prices(symbols, period="max", interval="1d", subdir=None):
    dest = config.PRICES if subdir is None else config.CACHE / subdir
    dest.mkdir(parents=True, exist_ok=True)
    status = {}
    for s in symbols:
        fp = dest / f"{s.replace('^','IDX_')}.parquet"
        df = _retry(lambda: yf.download(s, period=period, interval=interval,
                                        auto_adjust=True, progress=False, threads=False))
        if df is None or len(df) == 0:
            print(f"  {s}: EMPTY/FAILED"); status[s] = "failed"; continue
        df = clean_prices(df)
        df.to_parquet(fp)
        status[s] = f"ok:{len(df)}"
        print(f"  {s}: {len(df)} bars -> {fp.name}")
        time.sleep(1)  # throttle-friendly
    return status

def fetch_fundamentals(symbols):
    rows, stmt_dir = [], config.STATEMENTS
    stmt_dir.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        try:
            t = yf.Ticker(s)
            info = _retry(lambda: t.info) or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice") or float("nan")
            shares = info.get("sharesOutstanding") or float("nan")
            mcap = info.get("marketCap") or float("nan")
            try:
                inc, bs = t.income_stmt, t.balance_sheet
                inc.to_csv(stmt_dir / f"{s.replace('.','_')}_income.csv")
                bs.to_csv(stmt_dir / f"{s.replace('.','_')}_balance.csv")
                sr = ratios_from_statements(inc, bs, price, shares, mcap)
            except Exception as e:
                print(f"  {s}: statements failed ({e})"); sr = {}
            rows.append(snapshot_row(s.replace(".NS", ""), info, sr, price))
        except Exception as e:
            print(f"  {s}: info FAILED ({e})")
            rows.append({"Symbol": s.replace(".NS", ""), "Price": float("nan")})
        time.sleep(1)
    fund = add_industry_pe(pd.DataFrame(rows))
    fund.to_csv(config.CACHE / "fundamentals_snapshot.csv", index=False)
    print(f"fundamentals: {len(fund)} rows")
    return fund

def main():
    t0 = time.time()
    uni, uni_src, uni_date = fetch_universe()
    symcol = "Symbol" if "Symbol" in uni.columns else uni.columns[0]
    stocks = [f"{s.strip()}.NS" for s in uni[symcol].tolist()
              if isinstance(s, str) and s.strip()][:50]
    print(f"stocks: {len(stocks)}")
    print("== indices =="); idx_status = fetch_prices(list(config.INDEX_SYMBOLS.values()), period="max", subdir="indices")
    print("== stocks daily =="); px_status = fetch_prices(stocks)
    print("== stocks weekly =="); fetch_prices(stocks, period="max", interval="1wk", subdir="weekly")
    print("== fundamentals =="); fetch_fundamentals(stocks)
    # data-quality v1 from cached dailies
    q = []
    for s in stocks:
        fp = config.PRICES / f"{s}.parquet"
        key = s.replace(".NS", "")
        if fp.exists():
            df = pd.read_parquet(fp)
            r = quality_report(df); r["symbol"] = key; q.append(r)
        else:
            q.append({"symbol": key, "rows": 0, "note": "missing"})
    pd.DataFrame(q).to_csv(config.RESULTS / "data_quality.csv" if False else BASE / "results" / "data_quality.csv", index=False)
    meta = {"universe_source": uni_src, "retrieval_date": uni_date, "n_stocks": len(stocks),
            "fetch_date": date.today().isoformat(), "auto_adjust": True,
            "index_status": idx_status, "price_status": px_status,
            "bhavcopy_note": "spot-check runs in Phase 1 exit (5 stock-days vs NSE bhavcopy if reachable)"}
    (config.DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"DONE in {time.time()-t0:.0f}s → data/meta.json")

if __name__ == "__main__":
    (BASE / "results").mkdir(exist_ok=True)
    main()
