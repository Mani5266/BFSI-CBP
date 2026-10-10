# VIVA CHEAT SHEET — NIFTY Lens (2-line answers)

## Data
- **Universe?** Live NiftyIndices NIFTY-50 list, retrieved 2026-10-08, recorded in data/meta.json (survivorship-bias control).
- **Price range?** 2007-09-17 → 2026-10-08, yfinance auto_adjust=True, cached once in parquet; app runs fully offline.
- **Why no Kaggle?** Login-gated; yfinance covered 2007→present, sufficient per PRD fallback rule.

## Ratios
- **ROCE formula?** EBIT / (Total Assets − Current Liabilities). yfinance has no ROCE field — derived in src/fundamentals.py.
- **Banks show ROCE blank — bug?** No: lenders have no meaningful EBIT/current-liab split; ROE + D/E carry the analysis. Declared limitation.
- **RELIANCE PE 21 vs screener 40?** yfinance trailingPE snapshot stale; flagged in hand_verification.csv (PE match 80%).

## Patterns (RQ1)
- **Result?** 316k detections, hit rates 43–59%, excess ±0.5pp — no reliable edge. Null result = PASS, matches SET50 + Dow literature.
- **Why trust detectors?** Hand-built geometric rules, unit-tested (fire on fixtures, quiet on noise); pandas-ta-classic attempted, not depended on (Windows-safe, no TA-Lib).

## Backtests (RQ2)
- **Result?** Buy-&-Hold wins everywhere after 0.1%/side costs: 387% / Sharpe 0.40 vs GoldenCross 135% / 0.36. Holds pre- and post-2020.
- **No look-ahead?** Signals confirm on close, entered next day; Ichimoku spans shifted forward only.

## GPU
- **Where used?** Pattern forward-returns batched on CUDA tensors; 176-combo sweep on GPU; device=cuda in run_summary.json.
- **Speedup?** None at this scale — CPU 0.51s vs GPU 1.7s (PCIe-transfer-bound). Reported honestly; GPU pays off at larger scale.

## Wyckoff (RQ4)
- **Spring vs UTAD?** Candidates with evidence numbers (vr/sr/cp), never certainties — detections too sparse for a powered split, recorded null.
- **Phase tag?** Heuristic from 50-bar slope + range position; thresholds in config.py.

## Screening (RQ3)
- **Lab query?** `MarketCap > 500 AND P/E < 15 AND ROCE > 22%` hits INFY; 7/7 pytest incl. AND/OR/brackets/aliases + % handling.
- **FII data?** No free stock-level history exists — market-level note + snapshot-diff only, declared partial in M7.
