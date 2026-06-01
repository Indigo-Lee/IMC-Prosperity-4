"""Walk-forward backtest with the LIVE submission day appended as day 3.

Days 0/1/2 → historical (TTE 8/7/6 days)
Day 3     → live submission day (TTE = 5 days, the actual round-3 live setting)

This lets us sanity-check that the backtester reproduces the live result for
v1.1 before we evaluate v1.2.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import prosperity2bt
sys.path.insert(0, os.path.dirname(prosperity2bt.__file__))

import prosperity2bt.data as _bt_data
_bt_data.LIMITS["HYDROGEL_PACK"] = 200
_bt_data.LIMITS["VELVETFRUIT_EXTRACT"] = 200
for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500):
    _bt_data.LIMITS[f"VEV_{K}"] = 300

from prosperity2bt.file_reader import FileReader
from prosperity2bt.runner import run_backtest

R3_WORK = Path(__file__).parent
HISTORICAL_DIR = R3_WORK.parent / "ROUND_3"
LIVE_DIR = R3_WORK / "live_data"

PRODUCTS = [
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
DELTA1 = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")

# day_num → (data_dir, TTE in days, label)
DAY_CONFIG = {
    0: (HISTORICAL_DIR, 8.0, "historical day_0"),
    1: (HISTORICAL_DIR, 7.0, "historical day_1"),
    2: (HISTORICAL_DIR, 6.0, "historical day_2"),
    3: (LIVE_DIR,       5.0, "LIVE submission (437594)"),
}


class FlatFileReader(FileReader):
    def __init__(self, root: Path):
        self._root = root

    @contextmanager
    def file(self, path_parts: list[str]):
        name = path_parts[-1]
        full = self._root / name
        if not full.is_file():
            stem, _, ext = name.rpartition(".")
            if stem.endswith(("_nn", "_wn")):
                alt = f"{stem[:-3]}.{ext}"
                full = self._root / alt
        yield full if full.is_file() else None


def load_trader(path: Path, module_name: str, tte_initial_days: float):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "logger") and hasattr(mod.logger, "flush"):
        mod.logger.flush = lambda *a, **kw: None
    mod.Trader.TTE_INITIAL_DAYS = tte_initial_days
    return mod.Trader()


def run_day(trader_path: Path, day: int) -> dict:
    data_dir, tte, label = DAY_CONFIG[day]
    trader = load_trader(trader_path, f"trader_d{day}_{trader_path.stem}", tte)

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_backtest(
            trader=trader, file_reader=FlatFileReader(data_dir),
            round_num=3, day_num=day,
            print_output=False, disable_trades_matching=False,
            no_names=False, show_progress_bar=False,
        )

    last_pnl: dict[str, float] = {p: 0.0 for p in PRODUCTS}
    for row in result.activity_logs:
        prod = row.columns[2]
        if prod not in last_pnl:
            continue
        try:
            last_pnl[prod] = float(row.columns[16])
        except (TypeError, ValueError):
            pass

    qty_traded: dict[str, int] = defaultdict(int)
    n_fills: dict[str, int] = defaultdict(int)
    running_pos: dict[str, int] = defaultdict(int)
    max_pos: dict[str, int] = defaultdict(int)
    for tr in result.trades:
        t = tr.trade
        if t.buyer == "SUBMISSION":
            running_pos[t.symbol] += t.quantity
            n_fills[t.symbol] += 1; qty_traded[t.symbol] += t.quantity
        elif t.seller == "SUBMISSION":
            running_pos[t.symbol] -= t.quantity
            n_fills[t.symbol] += 1; qty_traded[t.symbol] += t.quantity
        max_pos[t.symbol] = max(max_pos[t.symbol], abs(running_pos[t.symbol]))

    return {
        "day": day, "tte": tte, "label": label,
        "last_pnl": last_pnl,
        "qty_traded": dict(qty_traded), "n_fills": dict(n_fills),
        "max_pos": dict(max_pos), "final_pos": dict(running_pos),
    }


def fmt(v: float) -> str:
    return f"{v:>11,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader", default="trader_r3_v1_1.py")
    parser.add_argument("--days", default="0,1,2,3",
                        help="comma-separated day numbers")
    args = parser.parse_args()
    trader_path = R3_WORK / args.trader
    days = [int(x) for x in args.days.split(",")]
    print(f"=== Backtesting {trader_path.name} on days {days} ===\n")

    rep_per_day = []
    for day in days:
        rep = run_day(trader_path, day)
        rep_per_day.append(rep)

    # Compact table
    header_cols = [r['label'][:18] for r in rep_per_day]
    print(f"{'Product':<22}" + "".join(f"{c:>14}" for c in header_cols) + "  Total")
    print("-" * (22 + 14 * len(rep_per_day) + 8))
    grand = 0.0
    for prod in PRODUCTS:
        row_total = 0.0
        cells = []
        for rep in rep_per_day:
            v = rep["last_pnl"][prod]
            row_total += v
            cells.append(fmt(v))
        grand += row_total
        if abs(row_total) > 0.5:
            print(f"{prod:<22}" + "".join(c.rjust(14) for c in cells) + f"  {fmt(row_total)}")
    print("-" * (22 + 14 * len(rep_per_day) + 8))
    totals = [sum(r["last_pnl"].values()) for r in rep_per_day]
    print(f"{'DAY TOTAL':<22}" + "".join(fmt(t).rjust(14) for t in totals) + f"  {fmt(sum(totals))}")

    # Per-product turnover (qty traded) summary on the LIVE day specifically
    live = next((r for r in rep_per_day if r["day"] == 3), None)
    if live is not None:
        print(f"\n--- LIVE day breakdown ---")
        print(f"{'product':<22}{'qty_traded':>11}{'fills':>8}{'maxpos':>8}{'final':>8}")
        for prod in PRODUCTS:
            q = live["qty_traded"].get(prod, 0)
            n = live["n_fills"].get(prod, 0)
            mp = live["max_pos"].get(prod, 0)
            fp = live["final_pos"].get(prod, 0)
            if q or n or fp:
                print(f"{prod:<22}{q:>11,}{n:>8,}{mp:>8}{fp:>8}")


if __name__ == "__main__":
    main()
