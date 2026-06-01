"""
Runs Strategy 9 and plots net worth over time across Round 1 days -2, -1, 0.

Three-panel figure:
  Top    — ASH_COATED_OSMIUM    mid + adaptive fair/edge reference + fills
  Middle — INTARIAN_PEPPER_ROOT mid + fair line ± OFFSET + fills
  Bottom — combined mark-to-market net worth

Produces:  strategy9_net_worth.png
Usage:     python3 plot_strategy9.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent              # ROUND1/
sys.path.insert(0, str(ROOT))                    # finds datamodel, backtester
sys.path.insert(0, str(Path(__file__).parent))   # finds strategy9 (this folder)

from backtester import load_prices, load_trades, simulate
from strategy9 import (
    Trader,
    ASH_PRODUCT, ASH_FAIR, ASH_DEFAULT_EDGE,
    PEPPER_PRODUCT,
    PEPPER_INITIAL_FAIR, PEPPER_SLOPE_PER_TICK, PEPPER_PASSIVE_OFFSET,
    PEPPER_FLOOR, PEPPER_MM_BAND,
)

DATA_DIR = ROOT
OUT = Path(__file__).parent / "strategy9_net_worth.png"

DAYS = {
    -2: (DATA_DIR / "prices_round_1_day_-2.csv",
         DATA_DIR / "trades_round_1_day_-2.csv"),
    -1: (DATA_DIR / "prices_round_1_day_-1.csv",
         DATA_DIR / "trades_round_1_day_-1.csv"),
     0: (DATA_DIR / "prices_round_1_day_0.csv",
         DATA_DIR / "trades_round_1_day_0.csv"),
}

DAY_COLORS = {-2: "#1f77b4", -1: "#ff7f0e", 0: "#2ca02c"}
TICKS_PER_DAY = 10_000
TS_STEP       = 100


def global_tick(day_idx: int, ts: int) -> int:
    return day_idx * TICKS_PER_DAY + ts // TS_STEP


def main() -> None:
    print("Loading data …")
    prices_list, trades_list = [], []
    for day, (pp, tp) in sorted(DAYS.items()):
        if not pp.exists():
            print(f"  [skip] {pp} not found")
            continue
        prices_list.append(load_prices(pp))
        if tp.exists():
            trades_list.append(load_trades(tp))
        print(f"  Day {day} loaded")

    prices_df = pd.concat(prices_list, ignore_index=True)
    trades_df  = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()

    plot_prices = prices_df.copy()
    plot_prices.loc[plot_prices["mid_price"] == 0, "mid_price"] = pd.NA

    print("\nRunning Strategy 9 backtest …")
    result = simulate(
        Trader(), prices_df, trades_df,
        reset_between_days=False,
        passive_fills=True,
    )
    pnl   = result.pnl_series
    fills = result.fill_log

    # ---- x-axis stitching ------------------------------------------------
    day_order  = sorted(plot_prices["day"].unique())
    day_idx    = {int(d): i for i, d in enumerate(day_order)}
    max_ts     = int(plot_prices["timestamp"].max())
    x_offset   = {int(d): i * (max_ts + 10_000) for i, d in enumerate(day_order)}

    def combined_x(day_series, ts_series):
        return [x_offset[int(d)] + int(ts) for d, ts in zip(day_series, ts_series)]

    # ---- fair / quote lines for PEPPER -----------------------------------
    pepper_prices = plot_prices[plot_prices["product"] == PEPPER_PRODUCT].sort_values(
        ["day", "timestamp"]
    )
    fair_x, fair_y, bid_y, ask_y = [], [], [], []
    for day in day_order:
        sub = pepper_prices[pepper_prices["day"] == day]
        ts_arr = sub["timestamp"].values
        cx = combined_x(sub["day"], ts_arr)
        gt = np.array([global_tick(day_idx[int(day)], int(t)) for t in ts_arr])
        fv = PEPPER_INITIAL_FAIR + PEPPER_SLOPE_PER_TICK * gt
        fair_x.extend(cx)
        fair_y.extend(fv.tolist())
        bid_y.extend((np.round(fv) - PEPPER_PASSIVE_OFFSET).tolist())
        ask_y.extend((np.round(fv) + PEPPER_PASSIVE_OFFSET).tolist())

    # ---- categorise fills ------------------------------------------------
    ash_buys   = [f for f in fills if f.symbol == ASH_PRODUCT    and f.quantity > 0]
    ash_sells  = [f for f in fills if f.symbol == ASH_PRODUCT    and f.quantity < 0]
    pep_buys   = [f for f in fills if f.symbol == PEPPER_PRODUCT and f.quantity > 0]
    pep_sells  = [f for f in fills if f.symbol == PEPPER_PRODUCT and f.quantity < 0]

    ash_fills_total = len([f for f in fills if f.symbol == ASH_PRODUCT])
    pep_fills_total = len([f for f in fills if f.symbol == PEPPER_PRODUCT])

    # ---- figure ----------------------------------------------------------
    fig, (ax_ash, ax_pep, ax_pnl) = plt.subplots(
        3, 1, figsize=(15, 12),
        gridspec_kw={"height_ratios": [1, 1, 1.4]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"Strategy 9 — John's adaptive ASH MM + PEPPER drift MM "
        f"(offset=±{PEPPER_PASSIVE_OFFSET}, floor={PEPPER_FLOOR}, "
        f"mm_band={PEPPER_MM_BAND})",
        fontsize=13, fontweight="bold",
    )

    # --- ASH panel --------------------------------------------------------
    ash_prices = plot_prices[plot_prices["product"] == ASH_PRODUCT].sort_values(
        ["day", "timestamp"]
    )
    for day in day_order:
        sub = ash_prices[ash_prices["day"] == day]
        cx  = combined_x(sub["day"], sub["timestamp"])
        ax_ash.plot(cx, sub["mid_price"].values,
                    color=DAY_COLORS.get(int(day), "gray"),
                    linewidth=0.7, alpha=0.6, label=f"Mid (Day {int(day)})")

    # Reference lines: fair value and default-edge band
    ax_ash.axhline(ASH_FAIR, color="black", linewidth=1.0,
                   linestyle=":", label=f"Fair value {ASH_FAIR:,}")
    ax_ash.axhline(ASH_FAIR - ASH_DEFAULT_EDGE, color="#2ca02c", linewidth=1.0,
                   linestyle="--", alpha=0.8,
                   label=f"Default bid (fair − {ASH_DEFAULT_EDGE})")
    ax_ash.axhline(ASH_FAIR + ASH_DEFAULT_EDGE, color="#d62728", linewidth=1.0,
                   linestyle="--", alpha=0.8,
                   label=f"Default ask (fair + {ASH_DEFAULT_EDGE})")

    if ash_buys:
        ax_ash.scatter(
            [x_offset[f.day] + f.timestamp for f in ash_buys],
            [f.price for f in ash_buys],
            marker="^", color="green", s=14, zorder=5, alpha=0.6,
            label=f"Buys ({len(ash_buys)})")
    if ash_sells:
        ax_ash.scatter(
            [x_offset[f.day] + f.timestamp for f in ash_sells],
            [f.price for f in ash_sells],
            marker="v", color="red", s=14, zorder=5, alpha=0.6,
            label=f"Sells ({len(ash_sells)})")

    ax_ash.set_ylabel(f"{ASH_PRODUCT} Price")
    ax_ash.set_title(
        f"{ASH_PRODUCT} — adaptive MM (join/undercut, soft-skew @ ±{10}) "
        f"fills: {ash_fills_total}", fontsize=10)
    ax_ash.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax_ash.legend(fontsize=8, loc="upper left", ncol=3)
    ax_ash.grid(True, alpha=0.25)
    ax_ash.set_xticks([])

    # --- PEPPER panel -----------------------------------------------------
    for day in day_order:
        sub = pepper_prices[pepper_prices["day"] == day]
        cx  = combined_x(sub["day"], sub["timestamp"])
        ax_pep.plot(cx, sub["mid_price"].values,
                    color=DAY_COLORS.get(int(day), "gray"),
                    linewidth=0.7, alpha=0.6, label=f"Mid (Day {int(day)})")

    ax_pep.plot(fair_x, fair_y, color="black", linewidth=1.1,
                linestyle=":", label="Fair value line")
    ax_pep.plot(fair_x, bid_y, color="#2ca02c", linewidth=1.0,
                linestyle="--", alpha=0.9,
                label=f"Passive bid (fair − {PEPPER_PASSIVE_OFFSET})")
    ax_pep.plot(fair_x, ask_y, color="#d62728", linewidth=1.0,
                linestyle="--", alpha=0.9,
                label=f"Passive ask (fair + {PEPPER_PASSIVE_OFFSET})")

    if pep_buys:
        ax_pep.scatter(
            [x_offset[f.day] + f.timestamp for f in pep_buys],
            [f.price for f in pep_buys],
            marker="^", color="green", s=14, zorder=5, alpha=0.6,
            label=f"Buys ({len(pep_buys)})")
    if pep_sells:
        ax_pep.scatter(
            [x_offset[f.day] + f.timestamp for f in pep_sells],
            [f.price for f in pep_sells],
            marker="v", color="red", s=14, zorder=5, alpha=0.6,
            label=f"Sells ({len(pep_sells)})")

    ax_pep.set_ylabel(f"{PEPPER_PRODUCT} Price")
    ax_pep.set_title(
        f"{PEPPER_PRODUCT} — fair ± {PEPPER_PASSIVE_OFFSET}, "
        f"floor={PEPPER_FLOOR}  fills: {pep_fills_total}",
        fontsize=10)
    ax_pep.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax_pep.legend(fontsize=8, loc="upper left", ncol=3)
    ax_pep.grid(True, alpha=0.25)
    ax_pep.set_xticks([])

    # --- PnL panel --------------------------------------------------------
    for day in sorted(pnl["day"].unique()):
        sub = pnl[pnl["day"] == day].sort_values("timestamp")
        cx  = combined_x(sub["day"], sub["timestamp"])
        ax_pnl.plot(cx, sub["pnl"].values,
                    color=DAY_COLORS.get(int(day), "gray"),
                    linewidth=1.1, label=f"Day {int(day)}")
        ax_pnl.fill_between(cx, 0, sub["pnl"].values,
                             where=[v >= 0 for v in sub["pnl"].values],
                             alpha=0.15, color="green", interpolate=True)
        ax_pnl.fill_between(cx, 0, sub["pnl"].values,
                             where=[v < 0 for v in sub["pnl"].values],
                             alpha=0.15, color="red", interpolate=True)

    ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--")
    final_val = result.final_pnl
    ax_pnl.annotate(
        f"Final net worth: {final_val:+,.0f} seashells",
        xy=(1.0, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        color="green" if final_val >= 0 else "red",
        fontweight="bold",
    )

    for day in day_order[1:]:
        ax_pnl.axvline(x_offset[int(day)], color="gray",
                       linewidth=0.8, linestyle=":", alpha=0.6)

    tick_positions, tick_labels = [], []
    for day in day_order:
        day_sub = pnl[pnl["day"] == day]
        mid_ts  = (day_sub["timestamp"].min() + day_sub["timestamp"].max()) / 2
        tick_positions.append(x_offset[int(day)] + mid_ts)
        tick_labels.append(f"Day {int(day)}")

    ax_pnl.set_xticks(tick_positions)
    ax_pnl.set_xticklabels(tick_labels, fontsize=10)
    ax_pnl.set_ylabel("Net Worth (seashells)")
    ax_pnl.set_title("Combined Net Worth over Time (mark-to-market PnL)", fontsize=10)
    ax_pnl.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:+,.0f}"))
    ax_pnl.legend(fontsize=8, loc="upper left")
    ax_pnl.grid(True, alpha=0.25)

    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {OUT}")
    print(result.summary())


if __name__ == "__main__":
    main()
