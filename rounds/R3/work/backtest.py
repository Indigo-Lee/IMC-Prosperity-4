"""Walk-forward backtest of trader_r3_v1 against prices_round_3_day_{0,1,2}.csv.

Day 0 → fit/sanity (8d TTE), day 1 → validation (7d TTE), day 2 → final (6d TTE).
Each day is run with TTE_INITIAL_DAYS configured appropriately and a fresh
trader instance (so traderData state doesn't leak across the artificial
day-boundary jumps that the historical data has).

Outputs:
  - per-day per-product PnL
  - per-day total PnL with breakdown by component (HYDROGEL / VELVET / vouchers)
  - aggregate metrics: max position, total turnover
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# prosperity2bt setup — patch position limits BEFORE importing the runner
# ---------------------------------------------------------------
import prosperity2bt
sys.path.insert(0, os.path.dirname(prosperity2bt.__file__))  # expose datamodel

import prosperity2bt.data as _bt_data
_bt_data.LIMITS["HYDROGEL_PACK"] = 200
_bt_data.LIMITS["VELVETFRUIT_EXTRACT"] = 200
for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500):
    _bt_data.LIMITS[f"VEV_{K}"] = 300

from prosperity2bt.file_reader import FileReader
from prosperity2bt.runner import run_backtest

R3_WORK = Path(__file__).parent
R3_DATA = R3_WORK.parent / "ROUND_3"

# Option settlement at expiry: max(VEV_final - K, 0)
VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
DAY_PERIOD = 1_000_000

_prices_df: pd.DataFrame | None = None

def _get_prices() -> pd.DataFrame:
    global _prices_df
    if _prices_df is None:
        pkl = R3_WORK / "analysis" / "out" / "prices.pkl"
        _prices_df = pd.read_pickle(pkl)
    return _prices_df


def _apply_settlement(rep: dict) -> None:
    """Add option settlement correction to last_pnl.

    prosperity2bt marks positions at the last book mid (not intrinsic value).
    At end of each day options settle at max(VEV_final - K, 0).
    Correction = (settlement - last_book_mid) × final_position.
    """
    day = rep["day"]
    prices = _get_prices()
    lo, hi = day * DAY_PERIOD, (day + 1) * DAY_PERIOD

    vev = prices[(prices["product"] == "VELVETFRUIT_EXTRACT") &
                 (prices["abs_ts"] >= lo) & (prices["abs_ts"] < hi)]
    if len(vev) == 0:
        return
    vev_final = vev.sort_values("abs_ts")["mid"].iloc[-1]

    for sym, K in VOUCHER_STRIKES.items():
        final_pos = rep["final_pos"].get(sym, 0)
        if final_pos == 0:
            continue
        opt = prices[(prices["product"] == sym) &
                     (prices["abs_ts"] >= lo) & (prices["abs_ts"] < hi)]
        if len(opt) == 0:
            continue
        last_mid = opt.sort_values("abs_ts")["mid"].iloc[-1]
        settlement = max(vev_final - K, 0.0)
        correction = (settlement - last_mid) * final_pos
        rep["last_pnl"][sym] += correction
        rep["settlement_corrections"] = rep.get("settlement_corrections", {})
        rep["settlement_corrections"][sym] = correction

PRODUCTS = [
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
DELTA1 = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")
DEEP_ITM = ("VEV_4000", "VEV_4500")
VOUCHERS_TRADED = tuple(p for p in PRODUCTS if p.startswith("VEV_"))

# Day → TTE in days mapping (per problem brief)
TTE_FOR_DAY = {0: 8.0, 1: 7.0, 2: 6.0}


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
    """Run one day of historical R3 data with the appropriate TTE setting."""
    trader = load_trader(
        trader_path,
        f"trader_r3_v1_day{day}",
        tte_initial_days=TTE_FOR_DAY[day],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_backtest(
            trader=trader,
            file_reader=FlatFileReader(R3_DATA),
            round_num=3,
            day_num=day,
            print_output=False,
            disable_trades_matching=False,
            no_names=False,
            show_progress_bar=False,
        )

    # Aggregate per-product metrics
    last_pnl: dict[str, float] = {p: 0.0 for p in PRODUCTS}
    by_ts_pnl: dict[str, list[tuple[int, float, float]]] = {p: [] for p in PRODUCTS}
    for row in result.activity_logs:
        ts = row.columns[1]
        prod = row.columns[2]
        if prod not in last_pnl:
            continue
        try:
            mid = float(row.columns[15])
        except (TypeError, ValueError):
            mid = float("nan")
        try:
            pnl = float(row.columns[16])
            last_pnl[prod] = pnl
            by_ts_pnl[prod].append((ts, mid, pnl))
        except (TypeError, ValueError):
            pass

    # Reconstruct positions from trades
    running_pos: dict[str, int] = defaultdict(int)
    max_abs_pos: dict[str, int] = defaultdict(int)
    n_own_fills: dict[str, int] = defaultdict(int)
    qty_traded: dict[str, int] = defaultdict(int)
    for tr in result.trades:
        t = tr.trade
        if t.buyer == "SUBMISSION":
            running_pos[t.symbol] += t.quantity
            n_own_fills[t.symbol] += 1
            qty_traded[t.symbol] += t.quantity
        elif t.seller == "SUBMISSION":
            running_pos[t.symbol] -= t.quantity
            n_own_fills[t.symbol] += 1
            qty_traded[t.symbol] += t.quantity
        max_abs_pos[t.symbol] = max(max_abs_pos[t.symbol], abs(running_pos[t.symbol]))

    rep = {
        "day": day,
        "tte": TTE_FOR_DAY[day],
        "last_pnl": last_pnl,
        "by_ts_pnl": by_ts_pnl,
        "max_abs_pos": dict(max_abs_pos),
        "n_own_fills": dict(n_own_fills),
        "qty_traded": dict(qty_traded),
        "final_pos": dict(running_pos),
    }
    _apply_settlement(rep)
    return rep


def print_day_report(rep: dict) -> None:
    print(f"\n=== DAY {rep['day']}  (TTE={rep['tte']}d) ===")
    total = 0.0
    print(f"{'product':<24} {'PnL':>14} {'fills':>7} {'qty':>7} {'maxpos':>7} {'finalpos':>8}")
    print("-" * 78)
    bucket = {"DELTA1": 0.0, "DEEP_ITM": 0.0, "VOUCHERS": 0.0}
    for p in PRODUCTS:
        v = rep["last_pnl"][p]
        total += v
        if p in DELTA1:
            bucket["DELTA1"] += v
        elif p in DEEP_ITM:
            bucket["DEEP_ITM"] += v
        else:
            bucket["VOUCHERS"] += v
        print(f"{p:<24} {v:>14,.2f} {rep['n_own_fills'].get(p, 0):>7} "
              f"{rep['qty_traded'].get(p, 0):>7} {rep['max_abs_pos'].get(p, 0):>7} "
              f"{rep['final_pos'].get(p, 0):>8}")
    settle_total = sum(rep.get("settlement_corrections", {}).values())
    print("-" * 78)
    print(f"{'DELTA1 (HYDRO+VELVET)':<24} {bucket['DELTA1']:>14,.2f}")
    print(f"{'DEEP_ITM (VEV_4000/4500)':<24} {bucket['DEEP_ITM']:>14,.2f}")
    print(f"{'VOUCHERS (VEV_5000+)':<24} {bucket['VOUCHERS']:>14,.2f}")
    if settle_total != 0:
        print(f"  (settle corrections applied: {settle_total:+,.0f})")
    print("=" * 78)
    print(f"{'DAY TOTAL':<24} {total:>14,.2f}")
    return total


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader", default="trader_r3_v1.py")
    args = parser.parse_args()
    trader_path = R3_WORK / args.trader
    print(f"Backtesting {trader_path.name} on R3 days 0/1/2 ...")

    grand = 0.0
    grand_bucket = {"DELTA1": 0.0, "DEEP_ITM": 0.0, "VOUCHERS": 0.0}
    daily = []
    for day in (0, 1, 2):
        rep = run_day(trader_path, day)
        day_total = sum(rep["last_pnl"].values())
        grand += day_total
        for p, v in rep["last_pnl"].items():
            if p in DELTA1:
                grand_bucket["DELTA1"] += v
            elif p in DEEP_ITM:
                grand_bucket["DEEP_ITM"] += v
            else:
                grand_bucket["VOUCHERS"] += v
        print_day_report(rep)
        daily.append((day, day_total))

    print("\n" + "=" * 78)
    print("WALK-FORWARD SUMMARY")
    print("=" * 78)
    for day, t in daily:
        print(f"  Day {day}: {t:>14,.2f}")
    print(f"  {'GRAND TOTAL':<7}: {grand:>14,.2f}")
    print(f"  Per-component:")
    print(f"    DELTA1   (HYDRO+VELVET):  {grand_bucket['DELTA1']:>12,.2f}")
    print(f"    DEEP_ITM (VEV_4000/4500): {grand_bucket['DEEP_ITM']:>12,.2f}")
    print(f"    VOUCHERS (VEV_5000+):     {grand_bucket['VOUCHERS']:>12,.2f}")
    print(f"\n  Acceptance check: ALL 3 days positive? "
          f"{'YES ✓' if all(t > 0 for _, t in daily) else 'NO ✗'}")


if __name__ == "__main__":
    main()
