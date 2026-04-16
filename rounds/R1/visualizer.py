#!/usr/bin/env python3
"""
Visualizer for the Round 1 backtest.

Runs backtest across all 3 days and produces a 3×2 figure:
  Row 1 – Mid price over time
  Row 2 – Cumulative P&L over time (mark-to-market, accumulated across days)
  Row 3 – Position over time

Usage:
    python3 visualizer.py            # saves visualizer.png and opens it
    python3 visualizer.py --no-show  # saves only (headless / CI)
"""

import sys
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Patch limits before runner imports ────────────────────────────────────
import prosperity2bt.data as _bt_data
_bt_data.LIMITS["INTARIAN_PEPPER_ROOT"] = 80
_bt_data.LIMITS["ASH_COATED_OSMIUM"] = 80

from prosperity2bt.runner import run_backtest
from prosperity2bt.file_reader import FileReader

sys.path.insert(0, str(Path(__file__).parent))
from trader import Trader

DATA_DIR  = Path(__file__).parent / "DataCapsule_R1"
DAYS      = [-2, -1, 0]
PRODUCTS  = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
LABELS    = {
    "INTARIAN_PEPPER_ROOT": "Intarian Pepper Root (IPR)",
    "ASH_COATED_OSMIUM":    "Ash-Coated Osmium (ACO)",
}
COLORS    = {"INTARIAN_PEPPER_ROOT": "#e05c2a", "ASH_COATED_OSMIUM": "#2a7ae0"}
DAY_TINTS = ["#e8f0fe", "#fce8e6", "#e6f4ea"]


# ── File reader ───────────────────────────────────────────────────────────

class FlatFileReader(FileReader):
    def __init__(self, root: Path):
        self._root = root

    @contextmanager
    def file(self, path_parts: list[str]):
        full = self._root / path_parts[-1]
        yield full if full.is_file() else None


# ── Data collection ───────────────────────────────────────────────────────

def collect(days: list[int]) -> tuple[dict, list[int]]:
    """
    Returns
    -------
    series : dict[product] -> {xs, mid, pnl, position}
        xs       – global tick counter (continuous across days)
        mid      – mid price (None when book is one-sided)
        pnl      – cumulative MTM P&L (accumulated across days, spike-free)
        position – net position at each tick
    day_breaks : list[int]
        xs values where each day starts (length = len(days))
    """
    series: dict[str, dict] = {
        p: {"xs": [], "mid": [], "pnl": [], "position": []}
        for p in PRODUCTS
    }
    day_breaks: list[int] = []
    tick = 0
    trader = Trader()
    pnl_carry: dict[str, float] = defaultdict(float)   # accumulated P&L from prior days

    for day in days:
        result = run_backtest(
            trader=trader,
            file_reader=FlatFileReader(DATA_DIR),
            round_num=1,
            day_num=day,
            print_output=False,
            disable_trades_matching=False,
            no_names=False,
            show_progress_bar=False,
        )

        # ── Build position timeline from SUBMISSION trades ────────────────
        # Each day is a fresh simulation so position resets to 0 at day start.
        running_pos: dict[str, int] = defaultdict(int)
        pos_snap: dict[str, dict[int, int]] = defaultdict(dict)  # ts → position after trades at ts

        for trow in result.trades:
            t = trow.trade
            if t.buyer == "SUBMISSION":
                running_pos[t.symbol] += t.quantity
            elif t.seller == "SUBMISSION":
                running_pos[t.symbol] -= t.quantity
            pos_snap[t.symbol][t.timestamp] = running_pos[t.symbol]

        # ── Group activity logs by timestamp ──────────────────────────────
        by_ts: dict[int, dict[str, list]] = defaultdict(dict)
        for row in result.activity_logs:
            ts   = row.columns[1]
            prod = row.columns[2]
            by_ts[ts][prod] = row.columns

        # ── Walk timestamps in order ──────────────────────────────────────
        day_breaks.append(tick)
        cur_pos: dict[str, int] = defaultdict(int)    # position cursor, resets each day
        last_valid_pnl: dict[str, float] = {}         # forward-fill for mid=0 spikes

        for ts in sorted(by_ts.keys()):
            # Update position cursor from trades at this timestamp
            for prod in PRODUCTS:
                if ts in pos_snap[prod]:
                    cur_pos[prod] = pos_snap[prod][ts]

            for prod in PRODUCTS:
                if prod not in by_ts[ts]:
                    continue

                cols      = by_ts[ts][prod]
                mid_raw   = cols[15]   # "" or float; 0 when book is one-sided
                pnl_raw   = cols[16]   # MTM P&L for THIS day only

                # Mid price – treat 0 / empty as missing
                try:
                    mid_val = float(mid_raw)
                    mid_val = mid_val if mid_val > 0 else None
                except (TypeError, ValueError):
                    mid_val = None

                # P&L – skip spike when mid is 0 (position * 0 tanks the value)
                if mid_val is not None:
                    try:
                        pnl_today = float(pnl_raw)
                    except (TypeError, ValueError):
                        pnl_today = last_valid_pnl.get(prod, 0.0)
                    last_valid_pnl[prod] = pnl_today
                else:
                    # forward-fill last clean value
                    pnl_today = last_valid_pnl.get(prod, 0.0)

                pnl_cumulative = pnl_carry[prod] + pnl_today

                series[prod]["xs"].append(tick)
                series[prod]["mid"].append(mid_val)
                series[prod]["pnl"].append(pnl_cumulative)
                series[prod]["position"].append(cur_pos[prod])

            tick += 1

        # Carry forward the day-end P&L into next day's offset
        for prod in PRODUCTS:
            pnl_carry[prod] += last_valid_pnl.get(prod, 0.0)

    return series, day_breaks


# ── Plot ──────────────────────────────────────────────────────────────────

def plot(series: dict, day_breaks: list[int], show: bool = True) -> None:
    fig, axes = plt.subplots(
        3, 2,
        figsize=(17, 10),
        sharex="col",
        gridspec_kw={"hspace": 0.07, "wspace": 0.20},
    )
    fig.patch.set_facecolor("#f5f5f5")
    fig.subplots_adjust(top=0.91)  # leave room for suptitle inside the figure

    # x-axis extent
    all_xs = [x for p in PRODUCTS for x in series[p]["xs"]]
    x_max  = max(all_xs) if all_xs else 1

    # Day ranges for shading
    day_ranges = []
    for i, start in enumerate(day_breaks):
        end = day_breaks[i + 1] - 1 if i + 1 < len(day_breaks) else x_max
        day_ranges.append((start, end))

    for col, prod in enumerate(PRODUCTS):
        color = COLORS[prod]
        s     = series[prod]
        xs    = s["xs"]

        for row in range(3):
            ax = axes[row][col]
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=8)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            # Day background + dividers
            for d_idx, (d_start, d_end) in enumerate(day_ranges):
                ax.axvspan(d_start, d_end, color=DAY_TINTS[d_idx],
                           alpha=0.5, zorder=0)
            for divider in day_breaks[1:]:
                ax.axvline(divider, color="#aaa", linewidth=0.8,
                           linestyle="--", zorder=1)

            # Day labels on top row
            if row == 0:
                for d_idx, (d_start, d_end) in enumerate(day_ranges):
                    ax.text(
                        (d_start + d_end) / 2, 1.03,
                        f"Day {DAYS[d_idx]}",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8.5, color="#444",
                    )

            # ── Row 0: Mid price ─────────────────────────────────────────
            if row == 0:
                mid_pts = [(x, v) for x, v in zip(xs, s["mid"]) if v is not None]
                if mid_pts:
                    mx, mv = zip(*mid_pts)
                    ax.plot(mx, mv, color=color, linewidth=0.9, zorder=2)
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.set_ylabel("Mid Price", fontsize=9)

            # ── Row 1: Cumulative P&L ────────────────────────────────────
            elif row == 1:
                pnl = s["pnl"]
                ax.plot(xs, pnl, color=color, linewidth=1.1, zorder=2)
                ax.axhline(0, color="#bbb", linewidth=0.7, zorder=1)
                ax.fill_between(xs, pnl, 0,
                                where=[v >= 0 for v in pnl],
                                color=color, alpha=0.12, zorder=1)
                ax.fill_between(xs, pnl, 0,
                                where=[v < 0 for v in pnl],
                                color="#cc3333", alpha=0.14, zorder=1)
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.set_ylabel("Cum. P&L (XIRECs)", fontsize=9)

                # Final value annotation
                final = pnl[-1] if pnl else 0
                ax.annotate(
                    f"  {final:+,.0f}",
                    xy=(xs[-1], final), fontsize=8.5,
                    color=color, fontweight="bold", va="center",
                )

            # ── Row 2: Position ──────────────────────────────────────────
            else:
                pos = s["position"]
                ax.step(xs, pos, color=color, linewidth=0.9,
                        where="post", zorder=3)
                ax.fill_between(xs, pos, 0, step="post",
                                where=[v >= 0 for v in pos],
                                color=color, alpha=0.25, zorder=2)
                ax.fill_between(xs, pos, 0, step="post",
                                where=[v < 0 for v in pos],
                                color="#cc3333", alpha=0.25, zorder=2)
                ax.axhline(0,   color="#bbb", linewidth=0.7, zorder=1)
                ax.axhline(80,  color="#888", linewidth=0.5, linestyle=":",
                           zorder=1, label="+80 limit")
                ax.axhline(-80, color="#888", linewidth=0.5, linestyle=":",
                           zorder=1)
                ax.set_ylim(-92, 92)
                ax.set_yticks([-80, -40, 0, 40, 80])
                ax.set_ylabel("Position", fontsize=9)

        # Column title
        axes[0][col].set_title(
            LABELS[prod], fontsize=11, fontweight="bold",
            color=color, pad=22,
        )

    # Grand total in suptitle
    total = sum(s["pnl"][-1] for s in series.values() if s["pnl"])
    fig.suptitle(
        f"Round 1 Backtest  ·  Days −2 → 0  ·  "
        f"Grand Total P&L: {total:,.0f} XIRECs",
        fontsize=13, fontweight="bold", y=0.97,
    )

    out = Path(__file__).parent / "visualizer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    if show:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    show = "--no-show" not in sys.argv
    print("Running backtest…")
    series, day_breaks = collect(DAYS)
    print("Rendering…")
    plot(series, day_breaks, show=show)
