# NIFTY Lens: NIFTY-50 Screening with Wyckoff & Backtesting

Open, reproducible NIFTY-50 screening + backtesting on free public data (CBP — Data Science for BFSI).

## Setup (fresh machine, Windows; torch 2.1.0+cu121 with CUDA pre-installed — never reinstall it)

```bat
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/env_check.py        :: must show CUDA True + RTX GPU
python scripts/fetch_data.py       :: ONLY network step → data/cache/
python scripts/build_features.py
python scripts/run_pattern_study.py :: GPU-batched RQ1/RQ4
python scripts/run_backtests.py    :: RQ2 vs Buy-&-Hold, costs on
python scripts/generate_report.py  :: results/ → report/CBP_REPORT.md
python -m pytest tests/ -q
streamlit run app.py               :: → localhost:8501, fully offline
```

## Repo map
`config.py` universe/paths/costs/thresholds · `src/` device, data_clean, fundamentals, indicators,
patterns, wyckoff, query_engine, portfolio, plots · `scripts/` pipeline · `data/cache/` parquet+snapshot+meta.json ·
`results/` CSVs/PNGs/run_summary.json · `report/CBP_REPORT.md`.

## Key findings (from real runs)
- Patterns: hit rates 43–59%, excess ±0.5pp — no reliable edge (null = PASS, matches SET50/Dow literature).
- Buy-&-Hold NIFTY beats all three rules after costs (full + pre/post-2020).
- Screener tests 7/7; 10-stock screener.in match PE 80% / ROCE 60% / ROE 90%.
- GPU (RTX 3050, device=cuda logged): transfer-bound at this scale — CPU 0.51s vs GPU 1.7s, reported honestly.
