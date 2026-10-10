"""Plotly interactive + mplfinance static figure builders."""
import pandas as pd

def candle(df: pd.DataFrame, title: str, src_note: str):
    import plotly.graph_objects as go
    fig = go.Figure([go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                    low=df["Low"], close=df["Close"], name="OHLC")])
    for col, nm in (("DMA50", "DMA50"), ("DMA200", "DMA200")):
        if col in df.columns:
            fig.add_scatter(x=df.index, y=df[col], mode="lines", name=nm)
    fig.update_layout(title=f"{title}<br><sub>{src_note}</sub>", xaxis_rangeslider_visible=False, height=520)
    return fig

def heatmap_treemap(fund: pd.DataFrame, period_ret: pd.Series):
    import plotly.express as px
    d = fund[["Symbol", "Sector", "MarketCap"]].copy()
    d["ret"] = d["Symbol"].map(period_ret)
    d = d.dropna(subset=["ret"])
    return px.treemap(d, path=["Sector", "Symbol"], values="MarketCap", color="ret",
                      color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                      title="Universe heatmap — sector × market-cap, coloured by period return")

def static_candle(df: pd.DataFrame, path: str, title: str):
    import mplfinance as mpf
    mpf.plot(df.tail(250), type="candle", mav=(50, 200), volume=True,
             title=title, savefig=str(path), style="yahoo")
