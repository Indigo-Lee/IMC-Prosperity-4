#!/usr/bin/env python3
"""
Visualize the first tenth (ticks 0-9999) of the 242009 submission.

Reads 242009.log (IMC-format JSON with activitiesLog + tradeHistory) and
produces a 3x2 figure:
  Row 0 - Mid price + best bid/ask band + our fills overlaid
  Row 1 - Position over time
  Row 2 - Cumulative mark-to-market P&L

Usage:
    python3 visualize_first_tenth.py            # saves PNG and opens it
    python3 visualize_first_tenth.py --no-show  # save only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).parent
LOG_FILE = HERE / "242009.log"
OUT_PNG = HERE / "visualize_first_tenth.png"

TICK_CUTOFF = 99_000  # render ticks [0, TICK_CUTOFF) (exclusive)

PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
LABELS = {
    "ASH_COATED_OSMIUM":    "Ash-Coated Osmium (ACO)",
    "INTARIAN_PEPPER_ROOT": "Intarian Pepper Root (IPR)",
}
COLORS = {"ASH_COATED_OSMIUM": "#2a7ae0", "INTARIAN_PEPPER_ROOT": "#e05c2a"}
POS_LIMIT = 50


def parse_activities(csv_text: str):
    """Returns {product: {ts: {mid, best_bid, best_ask}}}."""
    lines = csv_text.strip().split("\n")
    header = lines[0].split(";")
    idx = {col: i for i, col in enumerate(header)}

    out: dict[str, dict[int, dict]] = {p: {} for p in PRODUCTS}
    for row in lines[1:]:
        parts = row.split(";")
        ts = int(parts[idx["timestamp"]])
        if ts >= TICK_CUTOFF:
            continue
        prod = parts[idx["product"]]
        if prod not in out:
            continue

        def num(col):
            v = parts[idx[col]]
            return float(v) if v else None

        out[prod][ts] = {
            "mid":      num("mid_price"),
            "best_bid": num("bid_price_1"),
            "best_ask": num("ask_price_1"),
            "pnl":      num("profit_and_loss"),
        }
    return out


def parse_trades(trade_list: list[dict]):
    """Returns {product: [{ts, side, price, qty}, ...]} for SUBMISSION only."""
    out: dict[str, list[dict]] = {p: [] for p in PRODUCTS}
    for t in trade_list:
        if t["timestamp"] >= TICK_CUTOFF:
            continue
        prod = t["symbol"]
        if prod not in out:
            continue
        if t["buyer"] == "SUBMISSION":
            side = "buy"
        elif t["seller"] == "SUBMISSION":
            side = "sell"
        else:
            continue
        out[prod].append({
            "ts":    t["timestamp"],
            "side":  side,
            "price": float(t["price"]),
            "qty":   int(t["quantity"]),
        })
    return out


def build_series(book: dict, fills: dict):
    """Build aligned xs + mid/bid/ask + positions + realized+MTM PnL series."""
    series = {}
    for prod in PRODUCTS:
        ts_sorted = sorted(book[prod].keys())
        mids, bids, asks, pnls = [], [], [], []
        for ts in ts_sorted:
            b = book[prod][ts]
            mids.append(b["mid"])
            bids.append(b["best_bid"])
            asks.append(b["best_ask"])
            pnls.append(b["pnl"])

        # position timeline by walking fills in order
        pos_ts, pos_val = [0], [0]
        cur = 0
        for f in sorted(fills[prod], key=lambda x: x["ts"]):
            cur += f["qty"] if f["side"] == "buy" else -f["qty"]
            pos_ts.append(f["ts"])
            pos_val.append(cur)
        pos_ts.append(ts_sorted[-1] if ts_sorted else 0)
        pos_val.append(cur)

        series[prod] = {
            "xs": ts_sorted,
            "mid": mids,
            "bid": bids,
            "ask": asks,
            "pnl": pnls,
            "pos_ts": pos_ts,
            "pos_val": pos_val,
            "fills": fills[prod],
        }
    return series


def plot(series: dict, show: bool = True):
    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 10),
        sharex="col",
        gridspec_kw={"hspace": 0.08, "wspace": 0.18},
    )
    fig.patch.set_facecolor("#f5f5f5")
    fig.subplots_adjust(top=0.91)

    for col, prod in enumerate(PRODUCTS):
        color = COLORS[prod]
        s = series[prod]
        xs = s["xs"]

        # ── Row 0: Mid / spread band / our fills ────────────────────────
        ax = axes[0][col]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

        # Bid-ask band — only where BOTH sides exist (drop one-sided / 0 ticks)
        def _ok(v): return v is not None and v > 0
        band_bid, band_ask = [], []
        for b, a in zip(s["bid"], s["ask"]):
            if _ok(b) and _ok(a):
                band_bid.append(b); band_ask.append(a)
            else:
                band_bid.append(float("nan")); band_ask.append(float("nan"))
        ax.fill_between(xs, band_bid, band_ask, color=color, alpha=0.12,
                        linewidth=0, zorder=1, label="best bid/ask")

        mid_pts = [(x, v) for x, v in zip(xs, s["mid"]) if v is not None]
        if mid_pts:
            mx, mv = zip(*mid_pts)
            ax.plot(mx, mv, color=color, linewidth=0.9, zorder=2, label="mid")
            # Clamp y-range to valid mids (prevents one-sided-book spikes from crushing scale)
            m_lo, m_hi = min(mv), max(mv)
            pad = max(5.0, (m_hi - m_lo) * 0.08)
            ax.set_ylim(m_lo - pad, m_hi + pad)

        # Our fills
        buys  = [(f["ts"], f["price"], f["qty"]) for f in s["fills"] if f["side"] == "buy"]
        sells = [(f["ts"], f["price"], f["qty"]) for f in s["fills"] if f["side"] == "sell"]
        if buys:
            bx, by, bq = zip(*buys)
            ax.scatter(bx, by, s=[8 + 4*q for q in bq], marker="^",
                       color="#1a8f3c", edgecolors="white", linewidths=0.4,
                       zorder=5, label=f"buys ({len(buys)})")
        if sells:
            sx, sy, sq = zip(*sells)
            ax.scatter(sx, sy, s=[8 + 4*q for q in sq], marker="v",
                       color="#c23030", edgecolors="white", linewidths=0.4,
                       zorder=5, label=f"sells ({len(sells)})")

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylabel("Price", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
        ax.set_title(LABELS[prod], fontsize=11, fontweight="bold",
                     color=color, pad=10)

        # ── Row 1: Position ─────────────────────────────────────────────
        ax = axes[1][col]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.step(s["pos_ts"], s["pos_val"], where="post",
                color=color, linewidth=0.9, zorder=3)
        ax.fill_between(s["pos_ts"], s["pos_val"], 0, step="post",
                        where=[v >= 0 for v in s["pos_val"]],
                        color=color, alpha=0.25, zorder=2)
        ax.fill_between(s["pos_ts"], s["pos_val"], 0, step="post",
                        where=[v < 0 for v in s["pos_val"]],
                        color="#cc3333", alpha=0.25, zorder=2)
        ax.axhline(0, color="#bbb", linewidth=0.7)
        ax.axhline(POS_LIMIT, color="#888", linewidth=0.5, linestyle=":")
        ax.axhline(-POS_LIMIT, color="#888", linewidth=0.5, linestyle=":")
        ax.set_ylim(-POS_LIMIT - 5, POS_LIMIT + 5)
        ax.set_yticks([-POS_LIMIT, -POS_LIMIT // 2, 0, POS_LIMIT // 2, POS_LIMIT])
        ax.set_ylabel("Position", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # ── Row 2: Cumulative P&L (from activitiesLog MTM) ──────────────
        ax = axes[2][col]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        pnl = [p if p is not None else 0.0 for p in s["pnl"]]
        ax.plot(xs, pnl, color=color, linewidth=1.1, zorder=2)
        ax.axhline(0, color="#bbb", linewidth=0.7)
        ax.fill_between(xs, pnl, 0, where=[v >= 0 for v in pnl],
                        color=color, alpha=0.12)
        ax.fill_between(xs, pnl, 0, where=[v < 0 for v in pnl],
                        color="#cc3333", alpha=0.14)
        final = pnl[-1] if pnl else 0.0
        ax.annotate(f"  {final:+,.1f}",
                    xy=(xs[-1], final), fontsize=9,
                    color=color, fontweight="bold", va="center")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_ylabel("Cum. P&L (XIRECs)", fontsize=9)
        ax.set_xlabel("Timestamp (ticks)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    totals = {p: (series[p]["pnl"][-1] if series[p]["pnl"] and series[p]["pnl"][-1] is not None else 0.0)
              for p in PRODUCTS}
    fig.suptitle(
        f"Strategy 9 · Submission 242009 · Ticks 0–{TICK_CUTOFF-1:,}   "
        f"ACO: {totals['ASH_COATED_OSMIUM']:+,.0f}   "
        f"IPR: {totals['INTARIAN_PEPPER_ROOT']:+,.0f}   "
        f"Total: {sum(totals.values()):+,.0f} XIRECs",
        fontsize=12, fontweight="bold", y=0.97,
    )

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {OUT_PNG}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    show = "--no-show" not in sys.argv
    with open(LOG_FILE) as f:
        log = json.load(f)

    book = parse_activities(log["activitiesLog"])
    fills = parse_trades(log["tradeHistory"])

    for prod in PRODUCTS:
        n_ts = len(book[prod])
        n_fills = len(fills[prod])
        print(f"{prod}: {n_ts} book snapshots, {n_fills} SUBMISSION fills "
              f"in ticks [0, {TICK_CUTOFF:,})")

    series = build_series(book, fills)
    plot(series, show=show)


if __name__ == "__main__":
    main()
