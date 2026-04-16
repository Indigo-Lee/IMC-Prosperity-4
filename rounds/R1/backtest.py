#!/usr/bin/env python3
"""Run prosperity2bt against DataCapsule_R1 for round 1 days -2, -1, 0."""

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Patch position limits before importing the runner
import prosperity2bt.data as _bt_data
_bt_data.LIMITS["INTARIAN_PEPPER_ROOT"] = 80
_bt_data.LIMITS["ASH_COATED_OSMIUM"] = 80

from prosperity2bt.runner import run_backtest
from prosperity2bt.file_reader import FileReader

DATA_DIR = Path(__file__).parent / "DataCapsule_R1"
ROUND = 1
DAYS = [-2, -1, 0]


class FlatFileReader(FileReader):
    """Reads files directly from DATA_DIR, ignoring the round subdirectory."""
    def __init__(self, root: Path):
        self._root = root

    @contextmanager
    def file(self, path_parts: list[str]):
        # path_parts = ["round1", "prices_round_1_day_0.csv"]
        # We just want the filename (last part)
        filename = path_parts[-1]
        full_path = self._root / filename
        if full_path.is_file():
            yield full_path
        else:
            yield None


def main():
    # Import trader
    sys.path.insert(0, str(Path(__file__).parent))
    from trader import Trader

    file_reader = FlatFileReader(DATA_DIR)
    trader = Trader()

    grand_total = 0.0

    for day in DAYS:
        result = run_backtest(
            trader=trader,
            file_reader=file_reader,
            round_num=ROUND,
            day_num=day,
            print_output=False,
            disable_trades_matching=False,
            no_names=False,
            show_progress_bar=False,
        )

        # Collect final P&L per product from last timestamp activity log
        if not result.activity_logs:
            print(f"Day {day}: no activity logs")
            continue

        last_ts = result.activity_logs[-1].timestamp
        products_seen = set()
        day_total = 0.0

        print(f"\n--- Day {day} (last timestamp {last_ts}) ---")
        for row in reversed(result.activity_logs):
            if row.timestamp != last_ts:
                break
            # columns: [day, timestamp, product, ..., mid_price, profit_and_loss]
            product = row.columns[2]
            if product in products_seen:
                continue
            products_seen.add(product)
            pnl = row.columns[-1]
            day_total += pnl
            print(f"  {product:<30} P&L: {pnl:>12,.2f}")

        print(f"  {'DAY TOTAL':<30} P&L: {day_total:>12,.2f}")
        grand_total += day_total

        # Reinitialise trader state between days (traderData persists in state,
        # but the Trader object is reused which is fine since state is in traderData)

    print(f"\n{'='*50}")
    print(f"  GRAND TOTAL P&L: {grand_total:>12,.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
