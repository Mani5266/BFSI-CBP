"""Portfolio metrics vs NIFTY (M8)."""
import pandas as pd
import numpy as np

def metricsrets(rets: pd.Series) -> dict:
    rets = rets.dropna()
    if len(rets) < 2:
        return {}
    ann = rets.mean() * 252
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol else np.nan
    eq = (1 + rets).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    downside = rets[rets < 0].std() * np.sqrt(252)
    return {"ann_return": ann, "ann_vol": vol, "sharpe": sharpe,
            "sortino": ann / downside if downside else np.nan,
            "calmar": ann / abs(dd) if dd else np.nan, "max_dd": dd,
            "win_rate": (rets > 0).mean()}

def compare(port_prices: pd.Series, bench_prices: pd.Series) -> dict:
    pr, br = port_prices.pct_change(fill_method=None), bench_prices.pct_change(fill_method=None)
    m = metricsrets(pr)
    b = metricsrets(br)
    aligned = pd.concat([pr, br], axis=1).dropna()
    m["beta"] = aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var() if len(aligned) > 2 else np.nan
    m["corr_nifty"] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 2 else np.nan
    m["bench_ann_return"] = b.get("ann_return")
    return m
