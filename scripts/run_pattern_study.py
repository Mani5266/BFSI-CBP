"""RQ1/RQ4: GPU-batched forward-return study (PRD 3.3.1 + 9.1)."""
import sys, time, json
sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from src.device import DEVICE, log_device
from src.patterns import detect, IMPLIED, SYLLABUS
import config

H = config.FORWARD_HORIZONS

def fwd_gpu(close: np.ndarray, idx: np.ndarray, h: int) -> np.ndarray:
    """Batched forward returns on DEVICE. Chunk-safe: one stock at a time."""
    c = torch.tensor(close, dtype=torch.float64, device=DEVICE)
    i = torch.tensor(idx, dtype=torch.long, device=DEVICE)
    valid = i + h < len(close)
    r = (c[i[valid] + h] / c[i[valid]] - 1).cpu().numpy()
    return r

def study(stocks: list, feat_dir: Path):
    pool, base_all = {}, []
    for s in stocks:
        df = pd.read_parquet(feat_dir / f"{s}.parquet")
        close = df["Close"].values.astype(float)
        det = detect(df)
        for pat in SYLLABUS:
            ii = np.flatnonzero(det[pat].values)
            ii = ii[ii + max(H) < len(close)]
            if len(ii) == 0:
                continue
            for h in H:
                pool.setdefault((pat, h), []).append(fwd_gpu(close, ii, h))
        b = np.concatenate([fwd_gpu(close, np.arange(len(close) - max(H)), h) for h in H])
        base_all.append(b)
    recs = []
    for (pat, h), parts in pool.items():
        r = np.concatenate(parts)
        d = IMPLIED[pat]
        hit = float(((r > 0) if d > 0 else (r < 0)).mean()) if d != 0 else np.nan
        recs.append({"pattern": pat, "horizon": h, "n": len(r),
                     "mean": float(r.mean()), "median": float(np.median(r)), "hit_rate": hit})
    return pd.DataFrame(recs), np.concatenate(base_all)

def main():
    log_device("pattern_study")
    t0 = time.time()
    feat = Path("data/features")
    files = sorted(feat.glob("*.parquet"))
    stocks = [f.stem for f in files]
    t1 = time.time()
    full, base = study(stocks, feat)
    t_gpu = time.time() - t1
    # CPU sample timing (pure numpy equivalent on 5 stocks)
    t2 = time.time()
    for s in stocks[:5]:
        df = pd.read_parquet(feat / f"{s}.parquet")
        close = df["Close"].values.astype(float)
        det = detect(df)
        for pat in SYLLABUS:
            ii = np.flatnonzero(det[pat].values)
            for h in H:
                jj = ii[ii + h < len(close)]
                _ = close[jj + h] / close[jj] - 1
    t_cpu5 = time.time() - t2
    # Aggregate: mean across stocks already pooled; add baseline + flags
    base_mean = float(np.mean(base))
    full["baseline"] = base_mean
    full["excess"] = full["mean"] - base_mean
    full["signif"] = (full["n"] >= 30) & (full["excess"].abs() > 2 * full["mean"].abs() / np.sqrt(full["n"].clip(lower=1)))
    full.to_csv("results/pattern_study.csv", index=False)
    # heatmap: pattern x horizon mean excess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = full.pivot(index="pattern", columns="horizon", values="excess")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.01, vmax=0.01)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("forward horizon (days)"); ax.set_title("Pattern excess forward return vs baseline")
    fig.colorbar(im, ax=ax, label="excess return")
    fig.tight_layout(); fig.savefig("results/pattern_heatmap.png", dpi=120)
    el = time.time() - t0
    summ = {"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "stocks": len(stocks), "detections": int(full["n"].sum()),
            "gpu_full_time_s": round(t_gpu, 1), "cpu_5stock_time_s": round(t_cpu5, 2),
            "total_s": round(el, 1), "gpu_warning": DEVICE != "cuda"}
    Path("results/run_summary.json").write_text(json.dumps(summ, indent=1))
    print(full.to_string()); print(summ)

if __name__ == "__main__":
    main()
