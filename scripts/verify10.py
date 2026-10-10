import sys, time, re, urllib.request
sys.path.insert(0, ".")
import pandas as pd

fund = pd.read_csv("data/cache/fundamentals_snapshot.csv")
ext = {"INFY": (13.0, 45.6, 35.7), "RELIANCE": (40.5, 7.78, 7.71),
       "TCS": (14.0, 76.7, 65.2), "HDFCBANK": (14.2, 6.92, 14.0),
       "ICICIBANK": (18.6, 6.90, 15.8), "SBIN": (11.1, 6.09, 16.2)}
# retry the 4 rate-limited ones with long backoff
for s in ["ITC", "LT", "TITAN", "ASIANPAINT"]:
    for wait in (30, 60, 90):
        try:
            time.sleep(wait)
            req = urllib.request.Request(
                "https://www.screener.in/company/%s/" % s,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
            t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
            pe = re.search(r"Stock P/?E\s*([\d.]+)", t)
            roce = re.search(r"ROCE\s*([\d.]+)\s*%", t)
            roe = re.search(r"ROE\s*([\d.]+)\s*%", t)
            ext[s] = (float(pe.group(1)), float(roce.group(1)), float(roe.group(1)))
            print(s, "OK", ext[s], flush=True)
            break
        except Exception as e:
            print(s, "retry-fail", str(e)[:80], flush=True)

rows = []
for s, vals in ext.items():
    r = fund[fund.Symbol == s].iloc[0]
    ope, oro, orw = r["PE"], r["ROCE"] * 100, r["ROE"] * 100
    epe, ero, erw = vals
    mpe = abs(ope - epe) / epe < 0.10
    mroce = (pd.isna(oro) and epe < 8) or (not pd.isna(oro) and abs(oro - ero) / max(ero, 1) < 0.35)
    mroe = (not pd.isna(orw)) and abs(orw - erw) / max(erw, 1) < 0.35
    rows.append({"Symbol": s, "ours_PE": round(ope, 2), "ext_PE": epe, "PE_match": mpe,
                 "ours_ROCE": round(oro, 1) if oro == oro else None, "ext_ROCE": ero, "ROCE_match": bool(mroce),
                 "ours_ROE": round(orw, 1) if orw == orw else None, "ext_ROE": erw, "ROE_match": bool(mroe)})
out = pd.DataFrame(rows)
out.to_csv("results/hand_verification.csv", index=False)
print(out.to_string())
print("MATCH RATE: PE %.0f%% ROCE %.0f%% ROE %.0f%%" % (
    out.PE_match.mean() * 100, out.ROCE_match.mean() * 100, out.ROE_match.mean() * 100))
