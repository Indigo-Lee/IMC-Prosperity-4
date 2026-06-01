"""Phase 1.4 — IV surface analysis.

Compute BS implied vol per (voucher, timestamp).
TTE decays linearly: day_0 starts at 8d, day_1 at 7d, day_2 at 6d. Within each day,
TTE drops linearly across 10k timestamps. (Round-3 round itself is 7-day starting
on its first live day; the historical data labels day_0/1/2 with TTE 8/7/6 in our
brief — we'll use that mapping verbatim.)

Plot:
  - IV vs strike at 6 snapshot timestamps (raw smile)
  - IV vs moneyness m=log(K/S)/sqrt(T) at same snapshots (normalised smile)
  - IV time series per voucher
  - Per-snapshot parabolic fit IV = a + b*m + c*m^2; residuals.
  - Residual time series and ACF per voucher (= alpha signal).

Outputs:
  iv_smile_snapshots.png, iv_moneyness_snapshots.png, iv_timeseries.png,
  iv_fit_params.csv, iv_residuals_summary.csv, iv_residual_acf.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"

VOUCHERS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
STRIKE = {v: int(v.split("_")[1]) for v in VOUCHERS}

# TTE mapping
DAY_OFFSET = 1_000_000
TS_PER_DAY = 1_000_000  # abs_ts increments by 100/step → 10000 steps × 100 = 1_000_000
DAY_TTE_AT_START = {0: 8.0, 1: 7.0, 2: 6.0}  # at abs_ts = day*1e6


def tte_years_for_abs_ts(abs_ts: np.ndarray, year_basis: float = 365.0) -> np.ndarray:
    """TTE in years given the abs_ts of each price tick."""
    day = (abs_ts // DAY_OFFSET).astype(int)
    intra = abs_ts - day * DAY_OFFSET
    intra_frac = intra / TS_PER_DAY  # in [0, 1) — fraction of day elapsed
    tte_days = np.array([DAY_TTE_AT_START[d] for d in day]) - intra_frac
    return tte_days / year_basis


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S, K, T, sigma, r=0.0):
    sigma = np.maximum(sigma, 1e-12)
    T = np.maximum(T, 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def bs_vega(S, K, T, sigma):
    sigma = np.maximum(sigma, 1e-12)
    T = np.maximum(T, 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * norm_pdf(d1) * sqrtT


def implied_vol_vec(price, S, K, T, max_iter=60, tol=1e-6):
    """Vectorised IV solver via bisection + Newton polish.
    Returns nan where price <= intrinsic or solve fails."""
    intrinsic = np.maximum(S - K, 0.0)
    upper_bound = S
    valid = (price > intrinsic + 1e-9) & (price < upper_bound - 1e-9) & (T > 0)
    sigma = np.full_like(price, np.nan, dtype=float)

    if not np.any(valid):
        return sigma

    # Bisection bracket
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 5.0)
    sig = np.where(valid, 0.5, np.nan)

    # Bisection — guaranteed monotone in sigma for European call
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c = bs_call(S, K, T, mid)
        too_low = (c < price) & valid
        lo = np.where(too_low, mid, lo)
        hi = np.where(~too_low & valid, mid, hi)
        if np.all(hi - lo < tol):
            break

    sig = 0.5 * (lo + hi)
    sigma = np.where(valid, sig, np.nan)
    return sigma


def main() -> None:
    wm = pd.read_pickle(OUT / "wide_mid.pkl")
    abs_ts = wm.index.values

    S = wm["VELVETFRUIT_EXTRACT"].values
    T = tte_years_for_abs_ts(abs_ts)

    # Compute IV for all vouchers
    iv_frame = pd.DataFrame(index=wm.index)
    moneyness_frame = pd.DataFrame(index=wm.index)
    for v in VOUCHERS:
        K = STRIKE[v]
        price = wm[v].values
        sigma = implied_vol_vec(price, S, np.full_like(S, K), T)
        iv_frame[v] = sigma
        # moneyness m = log(K/S) / sqrt(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log(K / S) / np.sqrt(T)
        moneyness_frame[v] = m

    iv_frame.to_pickle(OUT / "iv.pkl")
    moneyness_frame.to_pickle(OUT / "moneyness.pkl")

    # Per-voucher IV stats
    print("=" * 90)
    print("IV summary per voucher")
    print("=" * 90)
    iv_summary = pd.DataFrame({
        "n_valid": iv_frame.notna().sum(),
        "iv_p10": iv_frame.quantile(0.10),
        "iv_p50": iv_frame.median(),
        "iv_p90": iv_frame.quantile(0.90),
        "iv_std": iv_frame.std(),
    })
    print(iv_summary.to_string())
    iv_summary.to_csv(OUT / "iv_summary.csv")

    # Smile snapshots: pick 6 evenly-spaced timestamps that have many valid IVs
    n_ts = len(wm)
    snap_idx = np.linspace(0, n_ts - 1, 6, dtype=int)
    fig_smile, ax_smile = plt.subplots(figsize=(10, 6))
    fig_money, ax_money = plt.subplots(figsize=(10, 6))

    fit_params = []
    for i in snap_idx:
        ts = wm.index[i]
        s_now = S[i]; t_now = T[i]
        ivs = iv_frame.iloc[i]
        ms = moneyness_frame.iloc[i]
        # Plot raw smile (strike vs IV)
        strikes = np.array([STRIKE[v] for v in VOUCHERS], dtype=float)
        iv_vals = ivs.values
        mask = ~np.isnan(iv_vals)
        if mask.sum() >= 3:
            ax_smile.plot(strikes[mask], iv_vals[mask], "o-", label=f"ts={ts} S={s_now:.1f} T={t_now*365:.2f}d")
            ax_money.plot(ms.values[mask], iv_vals[mask], "o-", label=f"ts={ts}")

            # Parabolic fit to (m, iv): IV = a + b*m + c*m^2
            xs = ms.values[mask]
            ys = iv_vals[mask]
            coef = np.polyfit(xs, ys, 2)  # coef[0] = c, coef[1] = b, coef[2] = a
            c, b, a = coef
            fit_params.append({
                "abs_ts": int(ts), "n_used": int(mask.sum()),
                "S": s_now, "T_days": t_now * 365, "a": a, "b": b, "c": c,
            })

    ax_smile.set_xlabel("Strike K")
    ax_smile.set_ylabel("BS Implied Vol (annualised, /365)")
    ax_smile.set_title("Smile snapshots — IV vs strike")
    ax_smile.legend(fontsize=8)
    ax_smile.grid(alpha=0.3)
    fig_smile.tight_layout()
    fig_smile.savefig(OUT / "iv_smile_snapshots.png", dpi=110)
    plt.close(fig_smile)

    ax_money.set_xlabel("moneyness m = log(K/S)/sqrt(T)")
    ax_money.set_ylabel("IV")
    ax_money.set_title("Smile snapshots — IV vs moneyness")
    ax_money.legend(fontsize=8)
    ax_money.grid(alpha=0.3)
    fig_money.tight_layout()
    fig_money.savefig(OUT / "iv_moneyness_snapshots.png", dpi=110)
    plt.close(fig_money)

    # IV time series (one line per voucher)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for v in VOUCHERS:
        axes[0].plot(iv_frame.index, iv_frame[v], lw=0.4, label=v)
    axes[0].set_title("IV time series — all vouchers")
    axes[0].set_ylabel("IV")
    axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(wm.index, S, color="C0", lw=0.5, label="VELVETFRUIT_EXTRACT mid")
    axes[1].set_ylabel("Underlying mid")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "iv_timeseries.png", dpi=110)
    plt.close(fig)

    # ===== Per-tick parabolic fit and residuals =====
    print("\nFitting parabola IV = a + b*m + c*m^2 each tick...")
    a_arr = np.full(n_ts, np.nan)
    b_arr = np.full(n_ts, np.nan)
    c_arr = np.full(n_ts, np.nan)
    n_used = np.zeros(n_ts, dtype=int)

    iv_mat = iv_frame.values  # (n_ts, n_vouchers)
    m_mat = moneyness_frame.values

    for i in range(n_ts):
        iv_row = iv_mat[i]
        m_row = m_mat[i]
        mask = ~np.isnan(iv_row)
        if mask.sum() >= 3:
            try:
                coef = np.polyfit(m_row[mask], iv_row[mask], 2)
                c_arr[i], b_arr[i], a_arr[i] = coef[0], coef[1], coef[2]
                n_used[i] = int(mask.sum())
            except Exception:
                pass

    # Residuals
    resid = np.full_like(iv_mat, np.nan)
    for j, v in enumerate(VOUCHERS):
        ms = m_mat[:, j]
        fit = a_arr + b_arr * ms + c_arr * ms * ms
        resid[:, j] = iv_mat[:, j] - fit
    resid_df = pd.DataFrame(resid, index=wm.index, columns=VOUCHERS)
    resid_df.to_pickle(OUT / "iv_residuals.pkl")

    # Per-voucher residual stats and ACF (lag 1)
    print("\nResidual stats per voucher (mean / std / |z|>2 share / ACF1):")
    rs_rows = []
    for v in VOUCHERS:
        r = resid_df[v].dropna().values
        if len(r) < 20:
            rs_rows.append({"voucher": v, "n": len(r), "mean": np.nan, "std": np.nan, "share_z2": np.nan, "acf1": np.nan})
            continue
        mean = float(r.mean()); std = float(r.std())
        z = (r - mean) / (std + 1e-12)
        share_z2 = float((np.abs(z) > 2).mean())
        # ACF1
        x = r - mean
        acf1 = float((x[:-1] * x[1:]).sum() / (x ** 2).sum())
        rs_rows.append({"voucher": v, "n": int(len(r)), "mean": mean, "std": std, "share_z2": share_z2, "acf1": acf1})
    rs_df = pd.DataFrame(rs_rows)
    rs_df.to_csv(OUT / "iv_residuals_summary.csv", index=False)
    print(rs_df.to_string(index=False))

    # ACF plot per voucher
    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, v in zip(axes, VOUCHERS):
        r = resid_df[v].dropna().values
        if len(r) < 50:
            ax.set_title(f"{v} — too few residuals")
            continue
        rm = r - r.mean()
        var = (rm ** 2).sum()
        max_lag = 50
        ac = [1.0]
        for k in range(1, max_lag + 1):
            ac.append((rm[:-k] * rm[k:]).sum() / var)
        ax.bar(range(max_lag + 1), ac, color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(2 / np.sqrt(len(r)), color="r", ls="--", lw=0.5)
        ax.axhline(-2 / np.sqrt(len(r)), color="r", ls="--", lw=0.5)
        ax.set_title(f"{v} — ACF residual (acf1={ac[1]:.3f})")
    fig.tight_layout()
    fig.savefig(OUT / "iv_residual_acf.png", dpi=110)
    plt.close(fig)

    # Fit-params time series
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(wm.index, a_arr, lw=0.4, color="C0"); axes[0].set_title("a (ATM IV intercept)"); axes[0].grid(alpha=0.3)
    axes[1].plot(wm.index, b_arr, lw=0.4, color="C1"); axes[1].set_title("b (skew)"); axes[1].grid(alpha=0.3)
    axes[2].plot(wm.index, c_arr, lw=0.4, color="C2"); axes[2].set_title("c (curvature)"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "iv_fit_params.png", dpi=110)
    plt.close(fig)

    # Save fit-params table
    pd.DataFrame({
        "abs_ts": wm.index, "a": a_arr, "b": b_arr, "c": c_arr, "n_used": n_used,
    }).to_csv(OUT / "iv_fit_params.csv", index=False)

    print("\nFit-params summary:")
    print(pd.DataFrame({"a": a_arr, "b": b_arr, "c": c_arr}).describe().to_string())

    print(f"\nWrote iv_summary.csv, iv_residuals_summary.csv, iv_fit_params.csv, "
          f"iv_smile_snapshots.png, iv_moneyness_snapshots.png, iv_timeseries.png, "
          f"iv_residual_acf.png, iv_fit_params.png → {OUT}")


if __name__ == "__main__":
    main()
