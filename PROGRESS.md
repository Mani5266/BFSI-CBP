# NIFTY Lens — agent build log (PRD 0.6: updated at every phase boundary)
## Phase 0 — Env & GPU verification (2026-10-08) DONE
- Python 3.11.9, torch 2.1.0+cu121, CUDA True, RTX 3050 Laptop 4GB, driver 596.08 → logs/env_check.txt
- Fix: pip pulled numpy 2.4.6/pandas 3.0.6 (breaks torch 2.1 ABI); pinned back numpy==1.26.4 pandas==2.1.4, CUDA re-verified True. requirements.txt carries NO torch.
- Decisions: repo root = nifty-lens/ subfolder; backtesting package = `backtesting` (PyPI name, not backtesting.py).
## Index smoke test (Day 1) DONE
- ^NSEI/^BSESN/^NSEBANK/^INDIAVIX all live via yfinance — no fallback needed. NiftyIndices ind_nifty50list.csv verified reachable.
## Phase 1 — Data foundation DONE
- Universe: LIVE NiftyIndices list (50 stocks, retrieval 2026-10-08, in data/meta.json) — differs from PRD-era list (now incl. BSE/INDIGO/MAXHEALTH post-reshuffle). App reads universe.csv, not hardcoded config.
- 50/50 daily cached (2007→2026), 4/4 indices, fundamentals snapshot + statements for all 50. Zero price failures.
- data_quality.csv v1 written. Bhavcopy: NSE blocks automated pulls → substituted by 10-stock screener.in cross-check (documented gap).
- Bug fixed: statement matcher substring traps (EBIT→Normalized EBITDA) — exact-first lookup; ROCE now realistic.
## Phase 2 — Fundamentals (M2) DONE
- All 6.1 ratios incl. ROCE/ROE/ROA derivations; % aware query parser; lab query hits INFY correctly.
- pytest 7/7. Hand-verification (external screener.in, 10 stocks): PE 80% / ROCE 60% / ROE 90% → results/hand_verification.csv. Gaps: stale yfinance trailingPE (RELIANCE/LT), bank ROCE n/a by construction, HDFCBANK ROE post-merger restatement.
## Phase 3 — Charts & indicators (M3) DONE
- add_all() incl. hand-composed Alligator, Ichimoku (no-lookahead), Fib/pivots; daily+weekly; six lab chart types; mplfinance static export.
## Phase 4 — Patterns + technical queries (M4+M5) DONE
- pandas-ta-classic attempted, hand geometric detectors canonical (Windows-safe, no TA-Lib). Channels/triangles shipped; H&S/flag/pennant/cup-&-handle CUT per 10.9.
- Spot-checks genuine: ADANIPORTS ichimoku-xover live 2026-10-08; golden crosses RELIANCE 2025-05-29, INFY 2025-12-15, HDFCBANK 2025-03-25 (a naive query bug counting below-zero bars was caught and fixed during verification).
## Phase 5 — Wyckoff + Sentiment (M6+M7) DONE
- VSA classes, Spring/UTAD/SC/SOS candidates WITH evidence strings, phase tag, governing rule in UI. FII/DII: partial declared (CSV drop-in ready). Fear-&-Greed labelled India proxy.
## Phase 6 — Evaluation (M8 + Sec 9) DONE
- Pattern study: 316,324 detections, device=cuda logged. Honest timing: CPU numpy 0.51s < GPU 1.7s (transfer-bound) — reported as-is.
- Backtests: B&H wins everywhere after costs (full + pre/post-2020). Sweep 176 combos on CUDA (GPU 0.4s vs CPU 0.08s — same honest note); best Sharpe 0.72 @ 60/100.
## Phase 7 — Dashboard DONE (pending screenshots)
- app.py Home+M1–M8, offline-only, health 200 headless, all page data-paths exercised headlessly OK.
- Screenshots: 22 PNGs supplied by user into `report/screenshots/` (2026-10-08) — filenames generic (image*.png); rename to page names (Home, M1…M8) before submission so the report can reference them.
## Phase 8 — Report & viva pack DONE
## Fresh pipeline rerun (2026-10-08, reproducibility proof) DONE
- fetch → features → pattern study (316,332 detections, ±8 vs first run = one day's drift) → backtests (bit-identical numbers) → spotchecks (same 4) → verify10 (PE 80 / ROCE 60 / ROE 90, identical) → report → pytest 7/7 → app health 200. Pipeline reproduces.
- scripts/generate_report.py → report/CBP_REPORT.md (numbers trace to results/; Appendix A + verbatim Limitations). README fresh-machine proof. pytest 7/7.
