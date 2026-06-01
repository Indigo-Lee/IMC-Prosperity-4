"""
IMC Prosperity 4 — Round 3 Market Visualizer
Reads the 6 raw CSV files (prices + trades, days 0-2) and produces a
comprehensive market overview PNG. Run from this script's directory.
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(__file__), "..")

# ── load CSVs ──────────────────────────────────────────────────────────────────
price_frames, trade_frames = [], []
for day in [0, 1, 2]:
    p = pd.read_csv(f"{BASE}/prices_round_3_day_{day}.csv", sep=";")
    p["day"] = day
    price_frames.append(p)
    t = pd.read_csv(f"{BASE}/trades_round_3_day_{day}.csv", sep=";")
    t["day"] = day
    trade_frames.append(t)

prices = pd.concat(price_frames, ignore_index=True)
trades = pd.concat(trade_frames, ignore_index=True)

prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL   = "HYDROGEL_PACK"
OPTIONS    = [f"VEV_{s}" for s in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]]
STRIKES    = {p: int(p.split("_")[1]) for p in OPTIONS}
DAYS       = [0, 1, 2]

# resample: keep every 1000 timestamps per product per day for speed
prices_thin = (prices[prices["timestamp"] % 1000 == 0]
               .copy().reset_index(drop=True))

# ── palette ────────────────────────────────────────────────────────────────────
FIG_BG    = "#0d1117"
PANEL_BG  = "#161b22"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#e6edf3"
DAY_CLR   = {0: "#58a6ff", 1: "#ffa657", 2: "#3fb950"}
DAY_LBL   = {0: "Day 0", 1: "Day 1", 2: "Day 2"}

plt.style.use("dark_background")

# ── helpers ────────────────────────────────────────────────────────────────────
def style_ax(ax, title, xlabel="", ylabel="", fontsize=11):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, color=TEXT_CLR, fontsize=fontsize, fontweight="bold", pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, lw=0.5, ls="--", alpha=0.6)

def plot_price_series(ax, product, show_trades=True):
    for day in DAYS:
        sub = prices_thin[(prices_thin["product"] == product) &
                          (prices_thin["day"] == day)].sort_values("timestamp")
        if sub.empty:
            continue
        ax.plot(sub["timestamp"], sub["mid_price"],
                color=DAY_CLR[day], lw=1.2, label=DAY_LBL[day], alpha=0.9)
        rm = sub.set_index("timestamp")["mid_price"].rolling(30).mean()
        ax.plot(rm.index, rm.values, color=DAY_CLR[day], lw=0.6,
                ls="--", alpha=0.45)
        if show_trades:
            t_sub = trades[(trades["symbol"] == product) & (trades["day"] == day)]
            if not t_sub.empty:
                ax.scatter(t_sub["timestamp"], t_sub["price"],
                           color=DAY_CLR[day], s=18, zorder=5,
                           edgecolors="white", linewidths=0.4, alpha=0.85)

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 36), facecolor=FIG_BG)
fig.suptitle("IMC Prosperity 4 · Round 3 — Raw Market Data (Days 0–2)",
             fontsize=19, fontweight="bold", color=TEXT_CLR, y=0.995)

gs = gridspec.GridSpec(
    6, 10,
    figure=fig,
    hspace=0.55, wspace=0.55,
    top=0.982, bottom=0.03, left=0.05, right=0.97,
)

# ── Panel 1: VELVETFRUIT_EXTRACT ───────────────────────────────────────────────
ax_vfe = fig.add_subplot(gs[0, :])
style_ax(ax_vfe, "VELVETFRUIT_EXTRACT — Mid Price (Underlying)",
         "Timestamp", "Mid Price")
plot_price_series(ax_vfe, UNDERLYING)
ax_vfe.legend(loc="upper right", fontsize=9, framealpha=0.35)

# ── Panel 2: HYDROGEL_PACK ─────────────────────────────────────────────────────
ax_hyd = fig.add_subplot(gs[1, :])
style_ax(ax_hyd, "HYDROGEL_PACK — Mid Price",
         "Timestamp", "Mid Price")
plot_price_series(ax_hyd, HYDROGEL)
ax_hyd.legend(loc="upper right", fontsize=9, framealpha=0.35)

# ── Panel 3: VEV options grid (2 rows × 5 cols) ────────────────────────────────
opt_gs = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[2:4, :],
                                          hspace=0.55, wspace=0.4)
for idx, opt in enumerate(OPTIONS):
    r, c = divmod(idx, 5)
    ax_o = fig.add_subplot(opt_gs[r, c])
    strike = STRIKES[opt]
    style_ax(ax_o, f"Strike {strike:,}", "ts", "Price", fontsize=9)
    ax_o.tick_params(labelsize=7)
    for day in DAYS:
        sub = prices_thin[(prices_thin["product"] == opt) &
                          (prices_thin["day"] == day)].sort_values("timestamp")
        if sub.empty:
            continue
        ax_o.plot(sub["timestamp"], sub["mid_price"],
                  color=DAY_CLR[day], lw=1.0, label=DAY_LBL[day], alpha=0.9)
        t_sub = trades[(trades["symbol"] == opt) & (trades["day"] == day)]
        if not t_sub.empty:
            ax_o.scatter(t_sub["timestamp"], t_sub["price"],
                         color=DAY_CLR[day], s=14, zorder=5,
                         edgecolors="white", linewidths=0.3, alpha=0.8)
    if idx == 0:
        ax_o.legend(fontsize=6, framealpha=0.3, loc="upper right")

# shared legend label for the grid
fig.text(0.5, 0.507, "VEV Options Mid Price by Strike — All 3 Days  (dots = actual trades)",
         ha="center", color=TEXT_CLR, fontsize=11, fontweight="bold")

# ── Panel 4: Options chain snapshots ──────────────────────────────────────────
chain_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[4, :],
                                             hspace=0.0, wspace=0.35)
snap_labels = {0: "Open  (ts ≈ 0)", 500000: "Midday  (ts ≈ 500k)", 999000: "Close  (ts ≈ 999k)"}
for ci, target_ts in enumerate([0, 500000, 999000]):
    ax_c = fig.add_subplot(chain_gs[ci])
    style_ax(ax_c, snap_labels[target_ts], "Strike", "Mid Price", fontsize=10)
    ax_c.tick_params(labelsize=7)
    strikes_arr = np.array([STRIKES[o] for o in OPTIONS])
    for day in DAYS:
        p_day = prices[prices["day"] == day]
        window = p_day[p_day["timestamp"].between(target_ts, target_ts + 10000)]
        mids = []
        for opt in OPTIONS:
            w_opt = window[window["product"] == opt]["mid_price"]
            mids.append(w_opt.mean() if not w_opt.empty else np.nan)
        mids = np.array(mids)
        ax_c.plot(strikes_arr, mids, "o-", color=DAY_CLR[day],
                  lw=1.4, ms=5, label=DAY_LBL[day], zorder=3)
        # intrinsic value line
        spot_w = window[window["product"] == UNDERLYING]["mid_price"].mean()
        if not np.isnan(spot_w):
            intrinsic = np.maximum(spot_w - strikes_arr, 0)
            ax_c.plot(strikes_arr, intrinsic, "--",
                      color=DAY_CLR[day], lw=0.7, alpha=0.45)
    ax_c.set_xticks(strikes_arr)
    ax_c.set_xticklabels([f"{s//1000}k" if s >= 1000 else str(s)
                           for s in strikes_arr], rotation=40, fontsize=7)
    if ci == 0:
        ax_c.legend(fontsize=8, framealpha=0.3)

# ── Panel 5 left: Bid-ask spread by product × day ─────────────────────────────
ax_sp = fig.add_subplot(gs[5, :5])
style_ax(ax_sp, "Avg Bid-Ask Spread by Product & Day",
         "Product", "Avg Spread (abs)")

all_prods = [UNDERLYING, HYDROGEL] + OPTIONS
spread_data = {day: [] for day in DAYS}
for prod in all_prods:
    for day in DAYS:
        sub = prices[(prices["product"] == prod) & (prices["day"] == day)]
        spread_data[day].append(sub["spread"].mean())

x = np.arange(len(all_prods))
bar_w = 0.25
for i, day in enumerate(DAYS):
    ax_sp.bar(x + i * bar_w, spread_data[day], bar_w,
              color=DAY_CLR[day], alpha=0.82, label=DAY_LBL[day])
ax_sp.set_xticks(x + bar_w)
short_labels = [p.replace("VELVETFRUIT_EXTRACT", "VFE")
                 .replace("HYDROGEL_PACK", "HYD") for p in all_prods]
ax_sp.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=7)
ax_sp.legend(fontsize=8, framealpha=0.3)

# ── Panel 5 right: Trade activity heatmap ─────────────────────────────────────
ax_th = fig.add_subplot(gs[5, 5:])
style_ax(ax_th, "Trade Activity Heatmap — Volume by Symbol & Time",
         "Time Bucket", "Symbol")

N_BUCKETS = 40
bucket_size = 1_000_000 // N_BUCKETS
trades["bucket"] = trades["timestamp"] // bucket_size
all_symbols = sorted(trades["symbol"].unique())
heat = np.zeros((len(all_symbols), N_BUCKETS))
for si, sym in enumerate(all_symbols):
    t_sym = trades[trades["symbol"] == sym]
    for b in range(N_BUCKETS):
        heat[si, b] = t_sym[t_sym["bucket"] == b]["quantity"].abs().sum()

cmap_heat = LinearSegmentedColormap.from_list(
    "heat", ["#161b22", "#ffa657", "#ff4444"])
im = ax_th.imshow(heat, aspect="auto", cmap=cmap_heat,
                  interpolation="nearest", vmin=0)
ax_th.set_yticks(range(len(all_symbols)))
ax_th.set_yticklabels(
    [s.replace("VELVETFRUIT_EXTRACT", "VFE").replace("HYDROGEL_PACK", "HYD")
     for s in all_symbols], fontsize=7)
tick_step = max(1, N_BUCKETS // 8)
ax_th.set_xticks(range(0, N_BUCKETS, tick_step))
ax_th.set_xticklabels(
    [f"{(b * bucket_size) // 1000}k" for b in range(0, N_BUCKETS, tick_step)],
    fontsize=7, rotation=30)
cbar = fig.colorbar(im, ax=ax_th, pad=0.01)
cbar.set_label("Volume traded", color=TEXT_CLR, fontsize=8)
cbar.ax.yaxis.set_tick_params(labelcolor=TEXT_CLR)

# ── save ───────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "market_visualizer.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print(f"Saved → {out}")
