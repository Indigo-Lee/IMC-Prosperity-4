"""Diagnose the IMC live submission 437594 vs the backtest expectation.

The submission JSON has top-level: round, status, profit, activitiesLog (CSV),
positions (final). The .log file is per-tick stdout (sandbox+lambda lines).

Output:
  - per-product final PnL
  - per-product max abs position, total qty traded
  - per-product time-series PnL plot
  - whether the live data shape (mids, spreads) matches the historical data
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop/imcprosperity4/rounds/R3/ROUND_3/437594"
JSN = ROOT / "437594.json"
LOG = ROOT / "437594.log"

OUT = Path(__file__).parent / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

with open(JSN) as f:
    blob = json.load(f)

print(f"Round:    {blob.get('round')}")
print(f"Status:   {blob.get('status')}")
print(f"Profit:   {blob.get('profit')}")
print(f"Final positions: {blob.get('positions')}")

# Parse activitiesLog (CSV with ; delim)
csv_text = blob["activitiesLog"]
df = pd.read_csv(io.StringIO(csv_text), sep=";")
print(f"\nactivity rows: {len(df)}")
print(f"days: {sorted(df['day'].unique())}")
print(f"timestamp range: {df['timestamp'].min()} … {df['timestamp'].max()}")
print(f"products: {sorted(df['product'].unique())}")

# Per-product final PnL
print("\nPer-product final PnL (last timestamp):")
last_ts = df["timestamp"].max()
last = df[df["timestamp"] == last_ts][["product", "mid_price", "profit_and_loss"]]
last = last.sort_values("profit_and_loss")
print(last.to_string(index=False))
total = last["profit_and_loss"].sum()
print(f"\nSum of final PnL across products: {total:,.2f}")

# Per-product PnL trajectory
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
products = sorted(df["product"].unique())
for prod in products:
    sub = df[df["product"] == prod].sort_values("timestamp")
    axes[0].plot(sub["timestamp"], sub["profit_and_loss"], lw=0.7, label=prod)

axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title(f"Live submission 437594 — per-product cumulative PnL (final total {total:,.2f})")
axes[0].set_ylabel("PnL")
axes[0].legend(fontsize=7, ncol=4, loc="upper left")
axes[0].grid(alpha=0.3)

# Total PnL
total_pnl = df.groupby("timestamp")["profit_and_loss"].sum().sort_index()
axes[1].plot(total_pnl.index, total_pnl.values, color="C3", lw=0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title("Total PnL across all products")
axes[1].set_xlabel("timestamp (this is round-3 LIVE day 2 — i.e. the actual submission day)")
axes[1].set_ylabel("PnL")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "live_437594_pnl.png", dpi=110)
plt.close(fig)

# Mid trajectories vs historical Phase 1 expectation
print("\nMid-price summary (live data, all timestamps):")
mid_stats = df.groupby("product")["mid_price"].agg(["first", "last", "min", "max", "mean", "std"])
print(mid_stats.to_string())

# How does the live underlying behave vs historical?
hyd = df[df["product"] == "HYDROGEL_PACK"].sort_values("timestamp")
vel = df[df["product"] == "VELVETFRUIT_EXTRACT"].sort_values("timestamp")
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(hyd["timestamp"], hyd["mid_price"], lw=0.5)
axes[0].axhline(10000, color="r", ls="--", lw=0.7, label="our fair=10000")
axes[0].set_title(f"HYDROGEL_PACK mid — live (mean {hyd['mid_price'].mean():.1f})")
axes[0].set_ylabel("mid"); axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(vel["timestamp"], vel["mid_price"], lw=0.5)
axes[1].axhline(5250, color="r", ls="--", lw=0.7, label="our fair=5250")
axes[1].set_title(f"VELVETFRUIT_EXTRACT mid — live (mean {vel['mid_price'].mean():.1f})")
axes[1].set_xlabel("timestamp"); axes[1].set_ylabel("mid"); axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "live_437594_mids.png", dpi=110)
plt.close(fig)

# Mid divergence vs our pinned fair
print("\nMid vs our pinned fair-value:")
print(f"  HYDROGEL_PACK:       mean mid = {hyd['mid_price'].mean():.2f}  (we used fair=10000, gap = {hyd['mid_price'].mean()-10000:+.2f})")
print(f"  VELVETFRUIT_EXTRACT: mean mid = {vel['mid_price'].mean():.2f}  (we used fair=5250,  gap = {vel['mid_price'].mean()-5250:+.2f})")

# Per-product: PnL trajectory for the 4 products that matter
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
axes = axes.ravel()
for ax, prod in zip(axes, ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_6000", "VEV_6500"]):
    sub = df[df["product"] == prod].sort_values("timestamp")
    if len(sub) == 0:
        ax.set_title(f"{prod} — no data")
        continue
    ax.plot(sub["timestamp"], sub["profit_and_loss"], color="C0", lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    final = sub["profit_and_loss"].iloc[-1]
    ax.set_title(f"{prod} — final PnL {final:+,.2f}")
    ax.set_ylabel("cum PnL")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "live_437594_4panel.png", dpi=110)
plt.close(fig)

print(f"\nSaved live_437594_pnl.png, live_437594_mids.png, live_437594_4panel.png → {OUT}")
