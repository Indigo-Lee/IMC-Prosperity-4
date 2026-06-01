"""Sweep HYDRO_DEFAULT_EDGE and HYDRO_SOFT_POSITION on v1.1.

Wider edge → less aggressive quoting → less one-sided inventory build-up
on drifting days. Smaller soft_position → quotes skew sooner when long/short.
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
HISTORICAL = R3_WORK.parent / "ROUND_3"
LIVE = R3_WORK / "live_data"

DAY_CFG = {
    0: (HISTORICAL, 8.0),
    1: (HISTORICAL, 7.0),
    2: (HISTORICAL, 6.0),
    3: (LIVE, 5.0),
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
                full = self._root / f"{stem[:-3]}.{ext}"
        yield full if full.is_file() else None


TRADER_PATH = R3_WORK / "trader_r3_v1_1.py"


def load_trader(default_edge: int, soft_pos: int, tte: float):
    spec = importlib.util.spec_from_file_location(
        f"trader_e{default_edge}_s{soft_pos}", TRADER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "logger"):
        mod.logger.flush = lambda *a, **kw: None
    mod.HYDRO_DEFAULT_EDGE = default_edge
    mod.HYDRO_SOFT_POSITION = soft_pos
    mod.Trader.TTE_INITIAL_DAYS = tte
    return mod.Trader()


def run_day(day: int, default_edge: int, soft_pos: int) -> dict:
    data_dir, tte = DAY_CFG[day]
    trader = load_trader(default_edge, soft_pos, tte)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_backtest(
            trader=trader, file_reader=FlatFileReader(data_dir),
            round_num=3, day_num=day, print_output=False,
            disable_trades_matching=False, no_names=False,
            show_progress_bar=False,
        )
    last_pnl = defaultdict(float)
    for row in result.activity_logs:
        try:
            last_pnl[row.columns[2]] = float(row.columns[16])
        except Exception:
            pass
    return sum(last_pnl.values())


def main() -> None:
    edges = [2, 3, 4, 5, 6, 8]
    soft_pos_vals = [10, 20, 30, 50, 100]
    print(f"{'edge':<6}{'soft':<6}" + "".join(f"  day{d:>3}" for d in (0,1,2,3)) + "      total")
    rows = []
    for edge in edges:
        for sp in soft_pos_vals:
            pnls = [run_day(d, edge, sp) for d in (0, 1, 2, 3)]
            total = sum(pnls)
            rows.append((edge, sp, pnls, total))
            print(f"{edge:<6}{sp:<6}" + "".join(f"{v:>9,.0f}" for v in pnls) + f" {total:>11,.0f}")
    rows.sort(key=lambda r: -r[3])
    print(f"\nTop-5 by total PnL:")
    for edge, sp, pnls, total in rows[:5]:
        print(f"  edge={edge}, soft={sp}: total={total:,.0f}, days={pnls}")


if __name__ == "__main__":
    main()
