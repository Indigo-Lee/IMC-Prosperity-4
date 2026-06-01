"""Phase 1.2 — underlying behavior.

For HYDROGEL_PACK and VELVETFRUIT_EXTRACT separately:
  - return distribution at lag 1, 5, 25
  - ACF of returns and abs(returns)
  - AR(1) coefficient and t-stat
  - Ornstein-Uhlenbeck half-life
  - spread / book-depth distributions
  - cross-correlation between the two products

Outputs in out/:
  underlying_summary.csv, underlying_acf.png, underlying_xcorr.png,
  underlying_returns_hist.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"

PRODS = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = (x ** 2).sum()
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = (x[:-k] * x[k:]).sum() / var
    return out


def ar1_fit(r: np.ndarray) -> dict:
    """OLS r_t = a + b r_{t-1}. Return b, t-stat, half-life from rho."""
    r = np.asarray(r, dtype=float)
    y = r[1:]
    x = r[:-1]
    n = len(y)
    xm = x.mean(); ym = y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    b = sxy / sxx if sxx > 0 else np.nan
    a = ym - b * xm
    resid = y - (a + b * x)
    sigma2 = (resid ** 2).sum() / max(n - 2, 1)
    se_b = np.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
    t_b = b / se_b if se_b and se_b > 0 else np.nan
    half_life_ticks = np.log(2) / -np.log(abs(b)) if 0 < abs(b) < 1 else np.inf
    return {
        "ar1_b": b, "ar1_t_stat": t_b,
        "half_life_ticks": half_life_ticks,
        "n_obs": n,
    }


def ou_fit(p: np.ndarray) -> dict:
    """Discrete OU on price levels: p_t - p_{t-1} = kappa*(mu - p_{t-1}) + e.
    Half-life = ln(2) / kappa (in tick units)."""
    p = np.asarray(p, dtype=float)
    dp = np.diff(p)
    pm = p[:-1]
    pmm = pm.mean(); dpm = dp.mean()
    sxx = ((pm - pmm) ** 2).sum()
    sxy = ((pm - pmm) * (dp - dpm)).sum()
    slope = sxy / sxx if sxx > 0 else np.nan  # = -kappa
    intercept = dpm - slope * pmm
    kappa = -slope
    mu = intercept / kappa if kappa not in (0, np.nan) else np.nan
    half_life = np.log(2) / kappa if kappa > 0 else np.inf
    return {"ou_kappa": kappa, "ou_mu": mu, "ou_half_life_ticks": half_life}


def xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> dict:
    """Pearson correlation between a_t and b_{t+lag} for lag in [-max_lag, max_lag]."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x = a[:len(a) - k]; y = b[k:]
        else:
            x = a[-k:]; y = b[:len(b) + k]
        if len(x) > 1:
            mx, my = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            if sx > 0 and sy > 0:
                out[k] = float(((x - mx) * (y - my)).mean() / (sx * sy))
            else:
                out[k] = np.nan
    return out


def main() -> None:
    wm = pd.read_pickle(OUT / "wide_mid.pkl")
    prices = pd.read_pickle(OUT / "prices.pkl")

    summary = {}
    fig_acf, axes_acf = plt.subplots(2, 2, figsize=(12, 8))

    for i, prod in enumerate(PRODS):
        s = wm[prod].dropna().values
        r = np.diff(s)

        # Distribution stats — note returns here are PRICE differences (ticks)
        # since prices are integer/half-integer ticks.
        info = {
            "mid_first": float(s[0]),
            "mid_last": float(s[-1]),
            "mid_min": float(s.min()),
            "mid_max": float(s.max()),
            "mid_std": float(np.std(s)),
            "drift_ticks_per_step": float(r.mean()),
            "ret1_std": float(r.std()),
            "ret1_skew": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12)),
            "ret1_kurt": float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-12) - 3),
            "share_zero_returns": float((r == 0).mean()),
        }
        info.update(ar1_fit(r))
        info.update(ou_fit(s))

        sub = prices[prices["product"] == prod]
        info["spread_p50"] = float(sub["spread"].median())
        info["spread_p90"] = float(sub["spread"].quantile(0.9))
        info["spread_max"] = float(sub["spread"].max())
        info["depth_p50"] = float(sub["book_depth"].median())
        info["depth_p10"] = float(sub["book_depth"].quantile(0.1))

        summary[prod] = info

        # ACF plots
        ac_r = acf(r, 50)
        ac_abs = acf(np.abs(r), 50)
        ax = axes_acf[i, 0]
        ax.bar(range(len(ac_r)), ac_r, color="C0")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(2 / np.sqrt(len(r)), color="r", ls="--", lw=0.5)
        ax.axhline(-2 / np.sqrt(len(r)), color="r", ls="--", lw=0.5)
        ax.set_title(f"{prod} — ACF of 1-step return")
        ax.set_xlabel("lag")

        ax = axes_acf[i, 1]
        ax.bar(range(len(ac_abs)), ac_abs, color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(2 / np.sqrt(len(r)), color="r", ls="--", lw=0.5)
        ax.set_title(f"{prod} — ACF of |return| (vol clustering)")
        ax.set_xlabel("lag")

    fig_acf.tight_layout()
    fig_acf.savefig(OUT / "underlying_acf.png", dpi=110)
    plt.close(fig_acf)

    # Cross-correlation
    sa = wm["HYDROGEL_PACK"].dropna()
    sb = wm["VELVETFRUIT_EXTRACT"].dropna()
    common = sa.index.intersection(sb.index)
    ra = np.diff(sa.loc[common].values)
    rb = np.diff(sb.loc[common].values)

    xc = xcorr(ra, rb, 10)
    fig, ax = plt.subplots(figsize=(8, 4))
    lags = sorted(xc.keys())
    ax.bar(lags, [xc[k] for k in lags])
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(2 / np.sqrt(len(ra)), color="r", ls="--", lw=0.5)
    ax.axhline(-2 / np.sqrt(len(ra)), color="r", ls="--", lw=0.5)
    ax.set_title("xcorr(HYDROGEL return_t, VELVETFRUIT return_{t+lag})")
    ax.set_xlabel("lag (positive = HYDROGEL leads VELVETFRUIT)")
    fig.tight_layout()
    fig.savefig(OUT / "underlying_xcorr.png", dpi=110)
    plt.close(fig)

    # Return histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, prod in zip(axes, PRODS):
        s = wm[prod].dropna().values
        r = np.diff(s)
        ax.hist(r, bins=51, color="C0", alpha=0.8)
        ax.set_title(f"{prod} 1-step Δmid distribution")
        ax.set_xlabel("Δmid (ticks)")
    fig.tight_layout()
    fig.savefig(OUT / "underlying_returns_hist.png", dpi=110)
    plt.close(fig)

    df = pd.DataFrame(summary).T
    df.to_csv(OUT / "underlying_summary.csv")
    print("=" * 80)
    print("UNDERLYING SUMMARY")
    print("=" * 80)
    print(df.to_string())

    print("\nCross-correlation HYDROGEL vs VELVETFRUIT (lag in steps of 100ts):")
    for k in sorted(xc.keys()):
        bar = "#" * int(abs(xc[k]) * 50)
        sign = "+" if xc[k] >= 0 else "-"
        print(f"  lag={k:+3d}  {sign}{bar}  {xc[k]:+.4f}")

    print(f"\nWrote underlying_summary.csv, underlying_acf.png, underlying_xcorr.png, underlying_returns_hist.png → {OUT}")


if __name__ == "__main__":
    main()
