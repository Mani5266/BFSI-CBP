"""RQ2: backtesting.py strategies vs Buy&Hold ^NSEI, costs on + GPU sweep (PRD 3.3.2 + 9.2)."""
import sys, time, json
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from backtesting import Backtest, Strategy
from src.device import DEVICE, log_device
import config

COST = config.COST_PER_SIDE

def idx_df():
    f = config.INDICES / "IDX_NSEI.parquet"
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

class GoldenCross(Strategy):
    fast, slow = config.DMA_SHORT, config.DMA_LONG
    def init(self):
        c = self.data.Close
        self.s = self.I(lambda x, n: pd.Series(x).rolling(n).mean().values, c, self.fast)
        self.l = self.I(lambda x, n: pd.Series(x).rolling(n).mean().values, c, self.slow)
    def next(self):
        if len(self.data) < 2:
            return
        if self.s[-1] > self.l[-1] and self.s[-2] <= self.l[-2] and not self.position:
            self.buy()
        elif self.s[-1] < self.l[-1] and self.s[-2] >= self.l[-2] and self.position:
            self.position.close()

class RsiMR(Strategy):
    lo, hi, n = config.RSI_LO, config.RSI_HI, config.RSI_PERIOD
    def init(self):
        from src.indicators import rsi
        self.r = self.I(lambda c: rsi(pd.Series(c), self.n).values, self.data.Close)
    def next(self):
        if self.r[-1] < self.lo and not self.position:
            self.buy()
        elif self.r[-1] > 55 and self.position:
            self.position.close()

class MacdX(Strategy):
    def init(self):
        from src.indicators import macd
        self.m = self.I(lambda c: macd(pd.Series(c))[0].values, self.data.Close)
        self.s = self.I(lambda c: macd(pd.Series(c))[1].values, self.data.Close)
    def next(self):
        if self.m[-1] > self.s[-1] and self.m[-2] <= self.s[-2] and not self.position:
            self.buy()
        elif self.m[-1] < self.s[-1] and self.m[-2] >= self.s[-2] and self.position:
            self.position.close()

class SpringEntry(Strategy):
    hold = 10
    def init(self):
        from src.wyckoff import detect_events
        df = pd.DataFrame({"Open": self.data.Open, "High": self.data.High,
                           "Low": self.data.Low, "Close": self.data.Close,
                           "Volume": self.data.Volume})
        ev = detect_events(df)
        self.sig = self.I(lambda: (ev["event"] == "Spring").values)
        self.entry_i = -10**9
    def next(self):
        if self.sig[-1] and not self.position:
            self.buy()
            self.entry_i = len(self.data)
        elif self.position and (len(self.data) - self.entry_i) >= self.hold:
            self.position.close()

class BuyHold(Strategy):
    def init(self):
        pass
    def next(self):
        if not self.position:
            self.buy()

KEYS = ["Return [%]", "Buy & Hold Return [%]", "Sharpe Ratio", "Sortino Ratio",
        "Calmar Ratio", "Max. Drawdown [%]", "Win Rate [%]", "# Trades", "Profit Factor"]

def run_all(df: pd.DataFrame, label: str):
    out = {}
    for name, cls in [("GoldenCross", GoldenCross), ("RSI-MR", RsiMR),
                      ("MACD-X", MacdX), ("Spring-10d", SpringEntry), ("BuyHold-NIFTY", BuyHold)]:
        bt = Backtest(df, cls, cash=1_000_000, commission=COST, exclusive_orders=True)
        st = bt.run()
        row = {"window": label, "strategy": name}
        for k in KEYS:
            row[k] = float(st[k]) if k in st else np.nan
        # CAGR from equity curve
        eq = st["_equity_curve"]["Equity"]
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        row["CAGR [%]"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan
        out[name] = (row, eq)
    return out

def gpu_sweep(df: pd.DataFrame):
    """Golden-cross (short,long) grid: SMA via cumsum on DEVICE, Sharpe per combo. GPU vs CPU-loop timing."""
    close = df["Close"].values.astype(np.float64)
    shorts = list(range(10, 61, 5))   # 11 values
    longs = list(range(100, 251, 10))  # 16 values → 176 combos
    t0 = time.time()
    c = torch.tensor(close, dtype=torch.float64, device=DEVICE)
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=DEVICE), c.cumsum(0)])
    ar = torch.arange(len(close), device=DEVICE)
    def sma_gpu(n):
        return (cs[n:] - cs[:-n]) / n  # len T-n+1, align right by padding
    rets = (c[1:] / c[:-1] - 1)
    best, rows = (-9e9, None), []
    for s_ in shorts:
        ms = torch.nn.functional.pad(sma_gpu(s_), (s_ - 1, 0))[:len(close)]
        for l_ in longs:
            if l_ <= s_:
                continue
            ml = torch.nn.functional.pad(sma_gpu(l_), (l_ - 1, 0))[:len(close)]
            sig = (ms > ml).float()
            pos = sig[:-1]  # enter next day (no look-ahead)
            turnover = torch.cat([pos[:1], (pos[1:] - pos[:-1]).abs()])
            pr = pos * rets - turnover * COST
            mu, sd = pr[200:].mean(), pr[200:].std()
            sh = float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan")
            rows.append((s_, l_, sh))
            if sh == sh and sh > best[0]:
                best = (sh, s_, l_)
    t_gpu = time.time() - t0
    t0 = time.time()  # honest CPU loop equivalent
    cpd = pd.Series(close)
    dr = close[1:] / close[:-1] - 1
    for s_ in shorts:
        ms = cpd.rolling(s_).mean().values
        for l_ in longs:
            if l_ <= s_:
                continue
            ml = cpd.rolling(l_).mean().values
            pos = (ms > ml).astype(float)[:-1]
            pr = pos * dr - np.r_[pos[:1], np.abs(np.diff(pos))] * COST
            _ = pr[200:].mean() / (pr[200:].std() + 1e-12)
    t_cpu = time.time() - t0
    return {"shorts": shorts, "longs": longs, "combos": len(rows),
            "best": {"sharpe": best[0], "short": best[1], "long": best[2]},
            "gpu_s": round(t_gpu, 2), "cpu_s": round(t_cpu, 2),
            "device": DEVICE, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}

def main():
    log_device("backtests")
    t0 = time.time()
    df = idx_df()
    print(f"index window: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars)")
    res = run_all(df, "full")
    pre = run_all(df[df.index < "2020-01-01"], "pre-2020")
    post = run_all(df[df.index >= "2020-01-01"], "post-2020")
    rows = [r for d in (res, pre, post) for r, _ in d.values()]
    pd.DataFrame(rows).to_csv("results/backtest_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (_, eq) in res.items():
        ax.plot(eq.index, eq.values / eq.values[0], label=name)
    ax.legend(fontsize=8); ax.set_title("Equity curves (rebased) — costs on, ^NSEI window")
    fig.tight_layout(); fig.savefig("results/equity_curves.png", dpi=120)
    sw = gpu_sweep(df)
    print(pd.DataFrame(rows).to_string())
    print("SWEEP:", sw)
    summ = json.loads(Path("results/run_summary.json").read_text())
    summ.update({"backtests": {"window": [str(df.index[0].date()), str(df.index[-1].date())],
                               "cost_per_side": COST, "elapsed_s": round(time.time() - t0, 1)},
                 "sweep": sw})
    Path("results/run_summary.json").write_text(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
