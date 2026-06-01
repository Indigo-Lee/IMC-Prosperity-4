"""Per-day stability check on the headline signals.

Re-derive:
  - IV-residual mean/std/ACF1 per voucher PER DAY
  - Underlying AR(1), OU half-life PER DAY
  - HYDROGEL fair value (running mean) PER DAY

If a signal flips sign or vanishes on any day, flag it.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"
DAY_OFFSET = 1_000_000

VOUCHERS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]


def acf1(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan")
    x = x - x.mean()
    den = (x ** 2).sum()
    if den == 0:
        return float("nan")
    return float((x[:-1] * x[1:]).sum() / den)


def ar1_b(r: np.ndarray) -> float:
    if len(r) < 5:
        return float("nan")
    y = r[1:]; x = r[:-1]
    sxx = ((x - x.mean()) ** 2).sum()
    if sxx == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).sum() / sxx)


def main() -> None:
    wm = pd.read_pickle(OUT / "wide_mid.pkl")
    iv = pd.read_pickle(OUT / "iv.pkl")
    resid = pd.read_pickle(OUT / "iv_residuals.pkl")

    abs_ts = wm.index.values
    day_idx = (abs_ts // DAY_OFFSET).astype(int)

    # ====== Underlying AR(1) and mid mean per day ======
    print("=" * 100)
    print("UNDERLYING — per-day AR(1) on Δmid and mid mean")
    print("=" * 100)
    rows = []
    for d in (0, 1, 2):
        mask = day_idx == d
        for prod in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            s = wm[prod].values[mask]
            r = np.diff(s)
            rows.append({
                "day": d, "product": prod,
                "n": int(len(s)),
                "mid_mean": float(s.mean()),
                "mid_min": float(s.min()),
                "mid_max": float(s.max()),
                "ret_std": float(r.std()),
                "ar1_b": ar1_b(r),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "stability_underlying.csv", index=False)

    # ====== IV residual mean/std/ACF1 per day per voucher ======
    print("\n" + "=" * 100)
    print("IV RESIDUAL stats per voucher per day (mean / std / acf1)")
    print("=" * 100)
    rows = []
    for d in (0, 1, 2):
        mask = day_idx == d
        for v in VOUCHERS:
            r = resid[v].values[mask]
            r = r[~np.isnan(r)]
            if len(r) < 50:
                rows.append({"day": d, "voucher": v, "n": len(r),
                             "mean": np.nan, "std": np.nan, "acf1": np.nan,
                             "share_z2": np.nan})
                continue
            mean = float(r.mean()); std = float(r.std())
            z = (r - mean) / (std + 1e-12)
            rows.append({
                "day": d, "voucher": v, "n": int(len(r)),
                "mean": mean, "std": std,
                "acf1": acf1(r),
                "share_z2": float((np.abs(z) > 2).mean()),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "stability_iv_residuals.csv", index=False)

    # Cross-day consistency: is the SIGN of the residual mean stable?
    print("\nResidual-mean SIGN per voucher per day (sign stability check):")
    pivot = df.pivot(index="voucher", columns="day", values="mean")
    pivot["sign_consistent"] = (
        (pivot[0] > 0) == (pivot[1] > 0)
    ) & ((pivot[1] > 0) == (pivot[2] > 0))
    print(pivot.to_string())

    # ====== Per-day IV mean per voucher ======
    print("\n" + "=" * 100)
    print("IV mean per voucher per day (raw, drift across days?)")
    print("=" * 100)
    rows = []
    for d in (0, 1, 2):
        mask = day_idx == d
        for v in VOUCHERS:
            ivv = iv[v].values[mask]
            ivv = ivv[~np.isnan(ivv)]
            rows.append({
                "day": d, "voucher": v,
                "n": len(ivv),
                "iv_mean": float(ivv.mean()) if len(ivv) else np.nan,
                "iv_std": float(ivv.std()) if len(ivv) > 1 else np.nan,
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="voucher", columns="day", values="iv_mean")
    pivot.columns = [f"day{c}_iv_mean" for c in pivot.columns]
    print(pivot.to_string())
    pivot.to_csv(OUT / "stability_iv_levels.csv")


if __name__ == "__main__":
    main()
