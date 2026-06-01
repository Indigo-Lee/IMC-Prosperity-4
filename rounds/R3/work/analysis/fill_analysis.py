"""Phase 1.6 — off-mid fill behavior.

For each trade, compute delta = trade_price − mid_at_ts. Plot histograms per
product. Look for:
  - fat tails (price-insensitive bots fillable at wide prices ⇒ wide-quote alpha)
  - cliffs (bots stop biting beyond a level ⇒ price-sensitive bots, shape gives
    the maximum off-mid edge)

Also report distribution stats and the share of trades > 1 tick away from mid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"

PRODS_TO_PLOT = (
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
)


def main() -> None:
    wm = pd.read_pickle(OUT / "wide_mid.pkl")
    trades = pd.read_pickle(OUT / "trades.pkl").copy()

    trades["mid_at_ts"] = trades.apply(
        lambda r: wm[r["symbol"]].get(r["abs_ts"], np.nan) if r["symbol"] in wm.columns else np.nan,
        axis=1,
    )
    trades["delta"] = trades["price"] - trades["mid_at_ts"]
    trades["abs_delta"] = trades["delta"].abs()

    rows = []
    for prod in trades["symbol"].unique():
        sub = trades[trades["symbol"] == prod]
        deltas = sub["delta"].dropna().values
        if len(deltas) == 0:
            continue
        weights = sub.loc[sub["delta"].notna(), "quantity"].values.astype(float)
        rows.append({
            "product": prod,
            "n_trades": int(len(sub)),
            "n_with_mid": int(len(deltas)),
            "p1": float(np.percentile(deltas, 1)),
            "p5": float(np.percentile(deltas, 5)),
            "p50": float(np.percentile(deltas, 50)),
            "p95": float(np.percentile(deltas, 95)),
            "p99": float(np.percentile(deltas, 99)),
            "mean": float(deltas.mean()),
            "std": float(deltas.std()),
            "share_abs_ge_1": float((np.abs(deltas) >= 1).mean()),
            "share_abs_ge_2": float((np.abs(deltas) >= 2).mean()),
            "share_abs_ge_3": float((np.abs(deltas) >= 3).mean()),
            "share_abs_ge_5": float((np.abs(deltas) >= 5).mean()),
            "max_abs": float(np.abs(deltas).max()),
            "qty_p50": float(np.percentile(sub["quantity"], 50)),
            "qty_p95": float(np.percentile(sub["quantity"], 95)),
            "qty_max": float(sub["quantity"].max()),
            "total_qty": int(sub["quantity"].sum()),
        })
    df = pd.DataFrame(rows).sort_values("n_trades", ascending=False)
    df.to_csv(OUT / "fill_summary.csv", index=False)

    print("=" * 110)
    print("FILL BEHAVIOR — trade_price − mid_at_ts, per product")
    print("=" * 110)
    print(df.to_string(index=False))

    # Histogram grid for the named products
    rows_p = 4; cols_p = 2
    fig, axes = plt.subplots(rows_p, cols_p, figsize=(12, 12))
    axes = axes.ravel()
    for ax, prod in zip(axes, PRODS_TO_PLOT):
        sub = trades[trades["symbol"] == prod]
        deltas = sub["delta"].dropna().values
        if len(deltas) == 0:
            ax.set_title(f"{prod} — no trades")
            continue
        bins = np.arange(np.floor(deltas.min()) - 0.5, np.ceil(deltas.max()) + 1.5, 1)
        ax.hist(deltas, bins=bins, color="C0", alpha=0.8)
        ax.axvline(0, color="r", lw=0.7)
        ax.set_title(f"{prod}  n={len(deltas)}  p5={np.percentile(deltas,5):.1f}  p95={np.percentile(deltas,95):.1f}")
        ax.set_xlabel("trade_price − mid")
    fig.tight_layout()
    fig.savefig(OUT / "fill_hist.png", dpi=110)
    plt.close(fig)

    # Most extreme off-mid trades — diagnostic for "do bots get filled at any price?"
    extreme = trades[trades["abs_delta"] >= 3].sort_values("abs_delta", ascending=False).head(40)
    if len(extreme):
        print("\nMost-extreme off-mid trades (|delta| >= 3 ticks), top 40:")
        print(extreme[["abs_ts", "symbol", "price", "mid_at_ts", "delta", "quantity"]].to_string(index=False))
    else:
        print("\nNo trades with |delta| >= 3.")

    print(f"\nWrote fill_summary.csv, fill_hist.png → {OUT}")


if __name__ == "__main__":
    main()
