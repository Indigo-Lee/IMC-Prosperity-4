"""
Round 1 → Round 2 generalization validation.

Runs R1Algo.py and R2.1.py against the full day 1 market data
(DataCapsule_R2/prices_round_2_day_1.csv + trades CSV), then:

  * reports total PnL, per-tick PnL stats, Sharpe-like ratio,
    and capture vs theoretical-max
  * splits at the 10% boundary (in-sample vs OOS for R1Algo) and
    computes generalization efficiency
  * saves validation_report.png (4 panels: mids, cum PnL,
    per-tick PnL hist, rolling Sharpe)
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------
# prosperity2bt setup: patch position limits and make datamodel importable
# ---------------------------------------------------------------
import prosperity2bt
sys.path.insert(0, os.path.dirname(prosperity2bt.__file__))  # expose datamodel

import prosperity2bt.data as _bt_data
_bt_data.LIMITS["INTARIAN_PEPPER_ROOT"] = 80
_bt_data.LIMITS["ASH_COATED_OSMIUM"] = 80

from prosperity2bt.file_reader import FileReader
from prosperity2bt.runner import run_backtest

R2_DIR = Path(__file__).parent
DATA_DIR = R2_DIR / "DataCapsule_R2"

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
SPLIT_PCT = 0.10  # first tenth = in-sample for R1Algo


# ---------------------------------------------------------------
# File reader: DataCapsule_R2 layout is flat, use filename only
# ---------------------------------------------------------------
class FlatFileReader(FileReader):
    """prosperity2bt asks for trades files as `trades_round_N_day_D_nn.csv`
    (`_nn` / `_wn` suffix). DataCapsule_R2 uses the plain name without the
    suffix, so we strip it here when falling back."""

    def __init__(self, root: Path):
        self._root = root

    @contextmanager
    def file(self, path_parts: list[str]):
        name = path_parts[-1]
        full = self._root / name
        if not full.is_file():
            # Try stripping _nn or _wn before .csv
            stem, _, ext = name.rpartition(".")
            if stem.endswith(("_nn", "_wn")):
                alt = f"{stem[:-3]}.{ext}"
                full = self._root / alt
        yield full if full.is_file() else None


# ---------------------------------------------------------------
# Dynamic Trader loader (handles filenames with dots e.g. R2.1.py)
# ---------------------------------------------------------------
def load_trader(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    # Both algos use a Logger helper that assumes state.listings contains
    # Listing objects (as on the real IMC exchange). prosperity2bt hands us
    # dicts, which crashes compress_listings. Disable the logger for
    # backtest runs; trading logic is unaffected.
    if hasattr(mod, "logger") and hasattr(mod.logger, "flush"):
        mod.logger.flush = lambda *a, **kw: None
    return mod.Trader()


# ---------------------------------------------------------------
# Backtest + per-tick series extraction
# ---------------------------------------------------------------
@dataclass
class Series:
    ts: np.ndarray                       # timestamps, shape (T,)
    pnl: dict[str, np.ndarray]           # per-product cum PnL shape (T,)
    tick_pnl: dict[str, np.ndarray]      # per-product per-tick PnL shape (T,)
    position: dict[str, np.ndarray]      # per-product position shape (T,)
    mid: dict[str, np.ndarray]           # per-product mid shape (T,)

    @property
    def total_cum(self) -> np.ndarray:
        return sum(self.pnl.values())

    @property
    def total_tick(self) -> np.ndarray:
        return sum(self.tick_pnl.values())


def run_one(trader, label: str) -> Series:
    """Run one backtest against DataCapsule_R2 day 1 data."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_backtest(
            trader=trader,
            file_reader=FlatFileReader(DATA_DIR),
            round_num=2,
            day_num=1,
            print_output=False,
            disable_trades_matching=False,
            no_names=False,
            show_progress_bar=False,
        )

    # Reconstruct position timeline from own trades
    running_pos: dict[str, int] = defaultdict(int)
    pos_at_ts: dict[str, dict[int, int]] = {p: {} for p in PRODUCTS}
    for tr in result.trades:
        t = tr.trade
        if t.buyer == "SUBMISSION":
            running_pos[t.symbol] += t.quantity
        elif t.seller == "SUBMISSION":
            running_pos[t.symbol] -= t.quantity
        pos_at_ts[t.symbol][t.timestamp] = running_pos[t.symbol]

    # Group activity rows by timestamp
    by_ts: dict[int, dict[str, list]] = defaultdict(dict)
    for row in result.activity_logs:
        ts = row.columns[1]
        prod = row.columns[2]
        by_ts[ts][prod] = row.columns

    timestamps = sorted(by_ts.keys())
    T = len(timestamps)

    pnl = {p: np.zeros(T) for p in PRODUCTS}
    mid = {p: np.full(T, np.nan) for p in PRODUCTS}
    pos = {p: np.zeros(T, dtype=int) for p in PRODUCTS}

    last_valid_pnl: dict[str, float] = {p: 0.0 for p in PRODUCTS}
    last_pos: dict[str, int] = {p: 0 for p in PRODUCTS}

    for i, ts in enumerate(timestamps):
        for p in PRODUCTS:
            if ts in pos_at_ts[p]:
                last_pos[p] = pos_at_ts[p][ts]
            pos[p][i] = last_pos[p]

            cols = by_ts[ts].get(p)
            if cols is None:
                pnl[p][i] = last_valid_pnl[p]
                continue

            mid_raw = cols[15]
            pnl_raw = cols[16]

            try:
                m = float(mid_raw)
                if m > 0:
                    mid[p][i] = m
            except (TypeError, ValueError):
                pass

            try:
                v = float(pnl_raw)
                # Guard against zero-mid spikes (position * 0 tanks value)
                if not np.isnan(mid[p][i]):
                    last_valid_pnl[p] = v
            except (TypeError, ValueError):
                pass
            pnl[p][i] = last_valid_pnl[p]

    tick_pnl = {p: np.diff(pnl[p], prepend=0.0) for p in PRODUCTS}

    return Series(
        ts=np.array(timestamps),
        pnl=pnl,
        tick_pnl=tick_pnl,
        position=pos,
        mid=mid,
    )


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def sharpe_like(tick_pnl: np.ndarray) -> float:
    std = tick_pnl.std()
    return float(tick_pnl.mean() / std) if std > 0 else 0.0


def capture_vs_max(series: Series, prod: str, pos_limit: int = 80) -> float:
    """Theoretical max = pos_limit × (end_mid - start_mid)."""
    m = series.mid[prod]
    valid = m[~np.isnan(m)]
    if len(valid) < 2:
        return float("nan")
    drift = float(valid[-1] - valid[0])
    theoretical = pos_limit * abs(drift)
    if theoretical == 0:
        return float("nan")
    return float(series.pnl[prod][-1] / theoretical)


def summarize(series: Series, label: str) -> dict:
    out = {"label": label, "products": {}}
    for p in PRODUCTS:
        tp = series.tick_pnl[p]
        out["products"][p] = {
            "total_pnl": float(series.pnl[p][-1]),
            "tick_mean": float(tp.mean()),
            "tick_std": float(tp.std()),
            "sharpe": sharpe_like(tp),
            "capture": capture_vs_max(series, p),
        }
    total = series.total_tick
    out["overall"] = {
        "total_pnl": float(series.total_cum[-1]),
        "tick_mean": float(total.mean()),
        "tick_std": float(total.std()),
        "sharpe": sharpe_like(total),
    }
    return out


# ---------------------------------------------------------------
# In-sample vs out-of-sample split
# ---------------------------------------------------------------
def split_metrics(series: Series, split_pct: float) -> tuple[dict, dict]:
    T = len(series.ts)
    k = max(1, int(T * split_pct))

    def slice_series(lo, hi) -> Series:
        return Series(
            ts=series.ts[lo:hi],
            pnl={p: series.pnl[p][lo:hi] - (series.pnl[p][lo - 1] if lo > 0 else 0)
                 for p in PRODUCTS},
            tick_pnl={p: series.tick_pnl[p][lo:hi] for p in PRODUCTS},
            position={p: series.position[p][lo:hi] for p in PRODUCTS},
            mid={p: series.mid[p][lo:hi] for p in PRODUCTS},
        )

    in_sample = slice_series(0, k)
    oos = slice_series(k, T)
    return in_sample, oos


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------
def print_metrics_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("PER-ALGO METRICS — full day 1 backtest")
    print("=" * 90)
    header = f"{'Algo':<10} {'Product':<25} {'Total PnL':>14} {'μ tick':>10} {'σ tick':>10} {'Sharpe':>8} {'Capture':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        for p, m in s["products"].items():
            print(f"{s['label']:<10} {p:<25} "
                  f"{m['total_pnl']:>14,.2f} {m['tick_mean']:>10.3f} "
                  f"{m['tick_std']:>10.3f} {m['sharpe']:>8.4f} "
                  f"{m['capture']:>9.3f}")
        o = s["overall"]
        print(f"{s['label']:<10} {'OVERALL':<25} "
              f"{o['total_pnl']:>14,.2f} {o['tick_mean']:>10.3f} "
              f"{o['tick_std']:>10.3f} {o['sharpe']:>8.4f} {'':>9}")
        print("-" * len(header))


def print_generalization_table(splits: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"GENERALIZATION — in-sample (first {SPLIT_PCT:.0%}) vs out-of-sample (remaining)")
    print("=" * 90)
    header = f"{'Algo':<10} {'In-sample PnL':>15} {'OOS PnL':>15} {'OOS/IS ratio':>14} {'Flag':>8}"
    print(header)
    print("-" * len(header))
    for s in splits:
        is_total = s["in"]["overall"]["total_pnl"]
        oos_total = s["oos"]["overall"]["total_pnl"]
        # Normalize to per-tick to avoid the 1:9 length bias, then compare
        is_per_tick = s["in"]["overall"]["tick_mean"]
        oos_per_tick = s["oos"]["overall"]["tick_mean"]
        ratio = (oos_per_tick / is_per_tick) if is_per_tick != 0 else float("nan")
        flag = "GOOD" if (ratio >= 0.7) else "WEAK"
        print(f"{s['label']:<10} {is_total:>15,.2f} {oos_total:>15,.2f} "
              f"{ratio:>14.3f} {flag:>8}")
    print("(ratio uses per-tick PnL to correct for the 1:9 length imbalance)")


# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
def read_prices_csv(path: Path) -> dict[str, list[tuple[int, float]]]:
    out = {p: [] for p in PRODUCTS}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=";"):
            prod = row["product"]
            if prod not in out:
                continue
            try:
                m = float(row["mid_price"])
            except (TypeError, ValueError):
                continue
            if m <= 0:
                continue
            out[prod].append((int(row["timestamp"]), m))
    for p in out:
        out[p].sort()
    return out


def build_mid_segments() -> tuple[dict, list[int], list[str]]:
    """Concatenate mid series across 4 segments for the top panel."""
    segments = [
        ("Day -1",         DATA_DIR / "prices_round_2_day_-1.csv"),
        ("Day  0",         DATA_DIR / "prices_round_2_day_0.csv"),
        ("Day  1 partial", DATA_DIR / "prices_round_1_day_1_partial.csv"),
        ("Day  1 full",    DATA_DIR / "prices_round_1_day_1_full.csv"),
    ]
    per_prod = {p: {"xs": [], "ys": []} for p in PRODUCTS}
    breaks: list[int] = []
    labels: list[str] = []
    tick = 0
    for label, path in segments:
        labels.append(label)
        breaks.append(tick)
        data = read_prices_csv(path)
        all_ts = sorted({ts for p in PRODUCTS for ts, _ in data[p]})
        ts_to_x = {ts: tick + i for i, ts in enumerate(all_ts)}
        for p in PRODUCTS:
            for ts, m in data[p]:
                per_prod[p]["xs"].append(ts_to_x[ts])
                per_prod[p]["ys"].append(m)
        tick += len(all_ts)
    breaks.append(tick)
    return per_prod, breaks, labels


def plot_report(series_by_algo: dict[str, Series], out_path: Path) -> None:
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor("#f5f5f5")
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.22)

    # -- Panel 1: mid price across 4 segments, both products stacked ------
    ax1 = fig.add_subplot(gs[0, :])
    segs, breaks, seg_labels = build_mid_segments()
    tints = ["#e8f0fe", "#fce8e6", "#e6f4ea", "#fef7e0"]
    for i, (s, e) in enumerate(zip(breaks[:-1], breaks[1:])):
        ax1.axvspan(s, e, color=tints[i], alpha=0.55, zorder=0)
        ax1.text((s + e) / 2, 1.02, seg_labels[i],
                 transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=9, color="#333")
    for b in breaks[1:-1]:
        ax1.axvline(b, color="#999", linewidth=0.7, linestyle="--", zorder=1)

    # Two products share the axis via twinx
    colors = {"INTARIAN_PEPPER_ROOT": "#e05c2a", "ASH_COATED_OSMIUM": "#2a7ae0"}
    ax1b = ax1.twinx()
    ax1.plot(segs["INTARIAN_PEPPER_ROOT"]["xs"],
             segs["INTARIAN_PEPPER_ROOT"]["ys"],
             color=colors["INTARIAN_PEPPER_ROOT"], linewidth=0.8, label="IPR (left)")
    ax1b.plot(segs["ASH_COATED_OSMIUM"]["xs"],
              segs["ASH_COATED_OSMIUM"]["ys"],
              color=colors["ASH_COATED_OSMIUM"], linewidth=0.7, alpha=0.75, label="ACO (right)")
    ax1.set_ylabel("IPR mid", color=colors["INTARIAN_PEPPER_ROOT"], fontsize=10)
    ax1b.set_ylabel("ACO mid", color=colors["ASH_COATED_OSMIUM"], fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax1.set_title("1) Mid price across 4 segments",
                  fontsize=11, fontweight="bold", pad=22, loc="left")
    ax1.spines[["top"]].set_visible(False)
    ax1b.spines[["top"]].set_visible(False)

    # -- Panel 2: cumulative PnL of both algos on day 1 full --------------
    ax2 = fig.add_subplot(gs[1, :])
    algo_colors = {"R1Algo": "#6b46c1", "R2.1": "#2e7d32"}
    for label, s in series_by_algo.items():
        ax2.plot(s.ts, s.total_cum, color=algo_colors[label],
                 linewidth=1.2, label=f"{label} (final {s.total_cum[-1]:,.0f})")
    ax2.axhline(0, color="#999", linewidth=0.6)
    ax2.set_ylabel("Cumulative PnL (XIRECs)", fontsize=10)
    ax2.set_xlabel("Timestamp (day 1)", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_title("2) Cumulative PnL — full day 1",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    # Mark the 10% split
    split_ts = series_by_algo["R1Algo"].ts[int(len(series_by_algo["R1Algo"].ts) * SPLIT_PCT)]
    ax2.axvline(split_ts, color="#c62828", linewidth=0.8, linestyle=":")
    ax2.text(split_ts, ax2.get_ylim()[1] * 0.95, "  10% (in-sample boundary)",
             color="#c62828", fontsize=8, va="top")

    # -- Panel 3: per-tick PnL histogram ---------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(-200, 200, 81)
    for label, s in series_by_algo.items():
        ax3.hist(np.clip(s.total_tick, bins[0], bins[-1]),
                 bins=bins, color=algo_colors[label], alpha=0.45, label=label,
                 edgecolor="none")
    ax3.set_xlabel("Per-tick PnL (clipped ±200)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.legend(frameon=False, fontsize=9)
    ax3.set_title("3) Per-tick PnL distribution",
                  fontsize=11, fontweight="bold", loc="left")
    ax3.spines[["top", "right"]].set_visible(False)

    # -- Panel 4: rolling Sharpe ------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    window = 500

    def rolling_sharpe(x: np.ndarray, w: int) -> np.ndarray:
        if len(x) < w:
            return np.array([])
        c1 = np.cumsum(np.insert(x, 0, 0.0))
        c2 = np.cumsum(np.insert(x ** 2, 0, 0.0))
        mean = (c1[w:] - c1[:-w]) / w
        mean_sq = (c2[w:] - c2[:-w]) / w
        var = np.maximum(mean_sq - mean ** 2, 0.0)
        std = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(std > 0, mean / std, 0.0)
        return r

    for label, s in series_by_algo.items():
        r = rolling_sharpe(s.total_tick, window)
        xs = s.ts[window - 1:window - 1 + len(r)]
        ax4.plot(xs, r, color=algo_colors[label], linewidth=1.0, label=label)
    ax4.axhline(0, color="#999", linewidth=0.6)
    ax4.set_xlabel("Timestamp (day 1)", fontsize=10)
    ax4.set_ylabel("Rolling Sharpe (μ/σ)", fontsize=10)
    ax4.legend(frameon=False, fontsize=9)
    ax4.set_title(f"4) Rolling Sharpe (window={window})",
                  fontsize=11, fontweight="bold", loc="left")
    ax4.spines[["top", "right"]].set_visible(False)

    # -- Summary strip at bottom -----------------------------------------
    ax_info = fig.add_subplot(gs[3, :])
    ax_info.axis("off")
    lines = []
    for label, s in series_by_algo.items():
        lines.append(
            f"{label}: total PnL {s.total_cum[-1]:,.0f}   |   "
            f"IPR {s.pnl['INTARIAN_PEPPER_ROOT'][-1]:,.0f}   "
            f"ACO {s.pnl['ASH_COATED_OSMIUM'][-1]:,.0f}"
        )
    ax_info.text(
        0.0, 0.6, "\n".join(lines), ha="left", va="top",
        fontsize=10, family="monospace",
    )

    fig.suptitle("Round 1 → Round 2 generalization — full day 1 backtest",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> None:
    algos = {
        "R1Algo": DATA_DIR / "R1Algo.py",
        "R2.1":   R2_DIR / "R2.1.py",
    }

    series_by_algo: dict[str, Series] = {}
    summaries: list[dict] = []
    splits: list[dict] = []

    for label, path in algos.items():
        print(f"Running backtest for {label} ...")
        trader = load_trader(path, f"trader_{label.replace('.','_')}")
        s = run_one(trader, label)
        series_by_algo[label] = s

        summary = summarize(s, label)
        summaries.append(summary)

        in_s, oos_s = split_metrics(s, SPLIT_PCT)
        splits.append({
            "label": label,
            "in":  summarize(in_s,  f"{label}/IS"),
            "oos": summarize(oos_s, f"{label}/OOS"),
        })

    print_metrics_table(summaries)
    print_generalization_table(splits)

    plot_report(series_by_algo, R2_DIR / "validation_report.png")


if __name__ == "__main__":
    main()
