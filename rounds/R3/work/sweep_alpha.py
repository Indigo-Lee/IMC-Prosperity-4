"""Parameter sweep over FAIR_EMA_ALPHA in trader_r3_v1_2.

For each alpha, run backtest_extended on all 4 days and report per-day PnL.
"""

from __future__ import annotations

import contextlib
import importlib
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


TRADER_PATH = R3_WORK / "trader_r3_v1_2.py"


def load_trader(alpha: float, tte: float):
    spec = importlib.util.spec_from_file_location(
        f"trader_alpha_{alpha}", TRADER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "logger"):
        mod.logger.flush = lambda *a, **kw: None
    mod.FAIR_EMA_ALPHA = alpha
    mod.Trader.TTE_INITIAL_DAYS = tte
    return mod.Trader()


def run_day_with_alpha(day: int, alpha: float) -> float:
    data_dir, tte = DAY_CFG[day]
    trader = load_trader(alpha, tte)
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
    alphas = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    days = (0, 1, 2, 3)
    print(f"{'α':<10}" + "".join(f"  day{d:>3}" for d in days) + "      total")
    for alpha in alphas:
        pnls = [run_day_with_alpha(d, alpha) for d in days]
        total = sum(pnls)
        print(f"{alpha:<10.4f}" + "".join(f"{v:>9,.0f}" for v in pnls) + f" {total:>11,.0f}")


if __name__ == "__main__":
    main()
