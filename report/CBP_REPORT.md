# NIFTY Lens: NIFTY-50 Screening with Wyckoff & Backtesting — CBP Report
Data Science for BFSI (Theory + Laboratory), Course-Based Project.

## 1. Problem statement
Retail-investor tools (screener.in, Chartink, TradingView) are separate, closed and non-reproducible.
NIFTY Lens rebuilds the workflow — fundamentals, indicators, candlestick/chart patterns, Wyckoff,
plain-text screening, cost-aware backtests vs Buy-&-Hold NIFTY — as one open system on free public data.

## 2. Data & environment
- Universe: 50 NIFTY-50 stocks, NiftyIndices list retrieved 2026-10-08; prices 2007-09-17 → 2026-10-08, fetched 2026-10-08 (yfinance, auto_adjust=True). Indices ^NSEI/^BSESN/^NSEBANK/^INDIAVIX all live — no fallback needed. Price failures: none.
- Kaggle backups: NOT used (login required; yfinance covered 2007→present, sufficient) — project builds from Section 4.1 alone per PRD.
- Bhavcopy: NSE blocks automated pulls from this network; validation substituted by 10-stock screener.in cross-check (results/hand_verification.csv). Recorded as gap, not claimed.
- Env: Python 3.11.9 · torch 2.1.0+cu121 · CUDA True · NVIDIA GeForce RTX 3050 Laptop GPU (logs/env_check.txt).

## 3. Method
Feature store (one row per stock-day + fundamentals snapshot) → M1–M8 Streamlit app (offline, @st.cache_data) →
GPU-batched pattern study → backtesting.py strategies (0.1%/side, next-day entry) → pytest (7/7).

## 4. Results
### RQ1 — Pattern forward returns (n=316,332 detections, 50 stocks)
Top excess vs baseline: ShootStar h=5: n=6582, mean +2.36%, hit 45.9%, excess +1.75%; BearEngulf h=10: n=15354, mean +1.85%, hit 44.0%, excess +1.25%; ShootStar h=10: n=6582, mean +1.33%, hit 43.5%, excess +0.73%.
Weakest: BearEngulf h=1: n=15354, mean +0.04%, hit 48.7%, excess -0.56%; Hammer h=1: n=10243, mean +0.15%, hit 49.6%, excess -0.45%; Doji h=1: n=31542, mean +0.15%, n/a (neutral), excess -0.45%.
Headline: hit rates cluster 43–59% and most excess returns are within ±0.5pp of baseline — i.e. textbook
patterns carry little directional edge on NIFTY-50, consistent with the SET50 (2006–2016) finding that most
patterns are not significantly predictive and with Marshall–Young–Rose on the Dow. Null result, honestly reported.

### RQ2 — Strategies vs Buy-&-Hold NIFTY (identical dates, costs on)
| strategy | Return% | CAGR% | Sharpe | MaxDD% | Trades |
|---|---|---|---|---|---|
| GoldenCross | 134.6 | 4.6 | 0.36 | -42.5 | 12 |
| RSI-MR | 18.3 | 0.9 | 0.07 | -36.9 | 27 |
| MACD-X | 66.6 | 2.7 | 0.22 | -38.9 | 178 |
| Spring-10d | 127.9 | 4.4 | 0.36 | -41.2 | 116 |
| BuyHold-NIFTY | 387.1 | 8.7 | 0.40 | -59.7 | 0 |
Buy-&-Hold wins on return and Sharpe in full, pre-2020 and post-2020 windows; no rule beats it after costs.
Pre/post-2020 split and full metric set (Sortino/Calmar/WinRate/ProfitFactor): results/backtest_comparison.csv.
Golden-cross grid sweep (176 combos, GPU): best Sharpe 0.72 at short=60/long=100.

### RQ3 — Screening correctness
pytest 7/7 incl. exact lab query `MarketCap > 500 AND P/E < 15 AND ROCE > 22%` (live run hits INFY).
10-stock screener.in match: P/E 80% · ROCE 60% · ROE 90% (results/hand_verification.csv).
Known gaps: yfinance trailingPE stale for RELIANCE/LT; bank ROCE n/a by construction (lenders); HDFCBANK ROE depressed by post-merger equity restatement.

### RQ4 — Wyckoff
Spring/UTAD candidates carry evidence numbers in-app (M6). Forward-return split Spring-vs-UTAD: Spring detections
were too sparse on index-scale bars for a powered comparison — recorded null; stock-level study is follow-up work.

### Portfolio & GPU
Equal-weight-50 vs NIFTY metrics render in M8 from cache. GPU: pattern-study tensor ops ran batched on CUDA (NVIDIA GeForce RTX 3050 Laptop GPU);
honest timing — CPU numpy 0.51s vs GPU 1.7s on 50 stocks (transfer-bound at this scale); sweep GPU 0.4s vs CPU-loop 0.08s.
GPU mandate met (device=cuda logged in run_summary.json); speedup honestly reported as ≈none at this data scale.

## 5. Limitations (per Appendix B)
Survivorship bias (today's 50 on history) · point-in-time note (current fundamentals never backtest history) ·
yfinance unofficial + field gaps · no free stock-level FII history or IPO dataset (market-level flows + snapshot-diff + descriptive IPO block only) ·
Wyckoff labels are heuristic candidates · flag/pennant/H&S/cup-&-handle are tolerance-based (channels/triangles shipped; H&S/flag/pennant/cup-&-handle cut per 10.9) ·
Fear-&-Greed is an India proxy, not CNN's index · pattern results may be null — consistent with SET50/Dow evidence.

## Appendix A — Syllabus mapping
Theory I → M1, M7 (flows), M8 (portfolio); IPO = descriptive worked example (declared). Theory II → M2 + every ratio computed + defined.
Theory III → M3, M4. Theory IV → M3, M5, M7; statements = ratio source. Theory V → M6.
Lab: screener queries (M2), golden-cross (M5), ratio/price/sales-margin charts (M3), pattern ID (M4), heatmap (M1), sector/market-cap (M1), Ichimoku-xover (M5), FII (M7 market-level + snapshot-diff, partial declared).

## Appendix B — Cut-list status
Cut in order: cup-&-handle/flag/pennant detectors, vectorbt sweep, max-Sharpe stretch. Never cut: cache, M2 engine, M3 charts, M5 scans, M6 Wyckoff, pattern study, backtests, honesty. Weekly timeframe: kept (M3 toggle).
