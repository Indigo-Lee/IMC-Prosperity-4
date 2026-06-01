"""
Round 2 price visualization.

Concatenates mid-price series across four segments:
  Day -1, Day 0, Day 1  (from DataCapsule_R2 CSVs)
  R1 live              (from R1.log — the actual submission on day 1)
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = Path(__file__).parent / "DataCapsule_R2"
PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
COLORS   = {"INTARIAN_PEPPER_ROOT": "#e05c2a", "ASH_COATED_OSMIUM": "#2a7ae0"}
LABELS   = {
    "INTARIAN_PEPPER_ROOT": "Intarian Pepper Root",
    "ASH_COATED_OSMIUM":    "Ash-Coated Osmium",
}
SEGMENTS = [
    ("Day -1 (historical)", DATA_DIR / "prices_round_2_day_-1.csv", None),
    ("Day  0 (historical)", DATA_DIR / "prices_round_2_day_0.csv",  None),
    ("Day  1 (historical)", DATA_DIR / "prices_round_2_day_1.csv",  None),
    ("R1 live (submission)", None,                                   DATA_DIR / "R1.log"),
]
TINTS = ["#e8f0fe", "#fce8e6", "#e6f4ea", "#fef7e0"]


def load_prices_csv(path: Path) -> dict[str, list[tuple[int, float]]]:
    out = {p: [] for p in PRODUCTS}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            prod = row["product"]
            if prod not in out:
                continue
            mid = row.get("mid_price", "")
            if not mid:
                continue
            try:
                m = float(mid)
            except ValueError:
                continue
            if m <= 0:
                continue
            out[prod].append((int(row["timestamp"]), m))
    for p in out:
        out[p].sort()
    return out


def load_r1_log(path: Path) -> dict[str, list[tuple[int, float]]]:
    """R1.log is a single-line JSON with an activitiesLog CSV string inside."""
    data = json.loads(path.read_text().splitlines()[0])
    csv_text = data["activitiesLog"]
    out = {p: [] for p in PRODUCTS}
    reader = csv.DictReader(io.StringIO(csv_text), delimiter=";")
    for row in reader:
        prod = row["product"]
        if prod not in out:
            continue
        mid = row.get("mid_price", "")
        if not mid:
            continue
        try:
            m = float(mid)
        except ValueError:
            continue
        if m <= 0:
            continue
        out[prod].append((int(row["timestamp"]), m))
    for p in out:
        out[p].sort()
    return out


def build_series():
    """Return per-product list of (xs, mids) and segment break x-positions."""
    per_product = {p: {"xs": [], "mid": []} for p in PRODUCTS}
    breaks: list[int] = []
    tick = 0

    for label, csv_path, log_path in SEGMENTS:
        breaks.append(tick)
        data = load_prices_csv(csv_path) if csv_path else load_r1_log(log_path)
        # Align all products to the same tick axis using the union of timestamps
        all_ts = sorted({ts for p in PRODUCTS for ts, _ in data[p]})
        ts_to_x = {ts: tick + i for i, ts in enumerate(all_ts)}
        for p in PRODUCTS:
            for ts, m in data[p]:
                per_product[p]["xs"].append(ts_to_x[ts])
                per_product[p]["mid"].append(m)
        tick += len(all_ts)

    breaks.append(tick)  # sentinel for last segment end
    return per_product, breaks


def plot(series, breaks, show=True):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"hspace": 0.15})
    fig.patch.set_facecolor("#f5f5f5")

    segment_ranges = list(zip(breaks[:-1], breaks[1:]))

    for ax, prod in zip(axes, PRODUCTS):
        s = series[prod]
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)

        for i, (start, end) in enumerate(segment_ranges):
            ax.axvspan(start, end, color=TINTS[i], alpha=0.6, zorder=0)
        for b in breaks[1:-1]:
            ax.axvline(b, color="#999", linewidth=0.8, linestyle="--", zorder=1)

        ax.plot(s["xs"], s["mid"], color=COLORS[prod], linewidth=0.9, zorder=2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_ylabel(f"{LABELS[prod]} mid", fontsize=10)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if prod == PRODUCTS[0]:
            for i, (start, end) in enumerate(segment_ranges):
                ax.text((start + end) / 2, 1.03, SEGMENTS[i][0],
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=9.5, color="#333")

    fig.suptitle("Round 2 — Price movement across four segments",
                 fontsize=13, fontweight="bold", y=0.98)
    out = Path(__file__).parent / "r2_price_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved -> {out}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    show = "--no-show" not in sys.argv
    series, breaks = build_series()
    plot(series, breaks, show=show)
