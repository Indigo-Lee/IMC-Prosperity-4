"""Phase 1.5 — trade-flow predictability.

Classify each trade as a buy (price >= mid_at_ts) or sell (price < mid_at_ts).
For each product, regress mid_change[t, t+W] on signed_flow[t-W, t] for W in {25, 50, 100}.
Report R^2 and slope per product/window. Plot signed-flow vs forward-return.

Outputs:
  flow_regression.csv, flow_scatter.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"

PRODS = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
         "VEV_4000", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500")

WINDOWS = (25, 50, 100)  # in timestamps (each step = 100 game-ts)


def main() -> None:
    wm = pd.read_pickle(OUT / "wide_mid.pkl")
    trades = pd.read_pickle(OUT / "trades.pkl")

    # Build a per-(symbol, abs_ts) mid lookup
    mid_lookup = {prod: wm[prod] for prod in wm.columns}

    # Classify trades
    trades = trades.copy()
    trades["mid_at_ts"] = trades.apply(
        lambda r: mid_lookup.get(r["symbol"], pd.Series()).get(r["abs_ts"], np.nan),
        axis=1,
    )
    trades["side"] = np.where(
        trades["price"] > trades["mid_at_ts"], +1,
        np.where(trades["price"] < trades["mid_at_ts"], -1, 0),
    )
    trades["signed_qty"] = trades["side"] * trades["quantity"]

    # Per-product flow regression
    rows = []
    fig, axes = plt.subplots(len(PRODS), len(WINDOWS), figsize=(13, 2.6 * len(PRODS)),
                             sharex="col")
    if len(PRODS) == 1:
        axes = axes.reshape(1, -1)

    for i, prod in enumerate(PRODS):
        sub = trades[trades["symbol"] == prod]
        if len(sub) < 10:
            for j in range(len(WINDOWS)):
                axes[i, j].set_title(f"{prod} — {len(sub)} trades (skip)")
            rows.append({"product": prod, "n_trades": len(sub),
                         "note": "insufficient_trades"})
            continue

        # Build a time-indexed signed flow series, summing per timestamp first
        flow = sub.groupby("abs_ts")["signed_qty"].sum()
        # Reindex to match the wide_mid index, fill 0
        flow_full = flow.reindex(wm.index, fill_value=0).astype(float)
        mid = wm[prod]

        for j, W in enumerate(WINDOWS):
            # rolling-flow over [t-W+1 .. t] (sum)
            roll_flow = flow_full.rolling(W, min_periods=1).sum()
            # forward mid change [t, t+W]
            fwd_mid = mid.shift(-W) - mid

            x = roll_flow.values
            y = fwd_mid.values
            mask = ~(np.isnan(x) | np.isnan(y)) & (x != 0)
            if mask.sum() < 30:
                rows.append({"product": prod, "window": W, "n_obs": int(mask.sum()),
                             "slope": np.nan, "r2": np.nan})
                axes[i, j].set_title(f"{prod} W={W}: too few non-zero flow")
                continue

            xs = x[mask]; ys = y[mask]
            xm = xs.mean(); ym = ys.mean()
            sxx = ((xs - xm) ** 2).sum()
            sxy = ((xs - xm) * (ys - ym)).sum()
            slope = sxy / sxx if sxx > 0 else np.nan
            intercept = ym - slope * xm
            yhat = intercept + slope * xs
            ss_res = ((ys - yhat) ** 2).sum()
            ss_tot = ((ys - ym) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            corr = sxy / (np.sqrt(((xs - xm) ** 2).sum() * ((ys - ym) ** 2).sum()) + 1e-12)

            rows.append({
                "product": prod, "window": W, "n_obs": int(mask.sum()),
                "slope": float(slope), "intercept": float(intercept),
                "r2": float(r2), "corr": float(corr),
                "flow_std": float(xs.std()), "fwd_mid_std": float(ys.std()),
            })

            ax = axes[i, j]
            # subsample for plot if many points
            sample = np.random.choice(np.where(mask)[0], min(2000, mask.sum()), replace=False)
            ax.scatter(x[sample], y[sample], s=3, alpha=0.4)
            ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
            xline = np.linspace(xs.min(), xs.max(), 20)
            ax.plot(xline, intercept + slope * xline, "r-", lw=1)
            ax.set_title(f"{prod} W={W}  corr={corr:+.3f}  R²={r2:.4f}")
            if i == len(PRODS) - 1:
                ax.set_xlabel("rolling signed flow")
            if j == 0:
                ax.set_ylabel("forward Δmid")

    fig.tight_layout()
    fig.savefig(OUT / "flow_scatter.png", dpi=110)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "flow_regression.csv", index=False)
    print("=" * 90)
    print("FLOW REGRESSION SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print(f"\nWrote flow_regression.csv, flow_scatter.png → {OUT}")


if __name__ == "__main__":
    main()
