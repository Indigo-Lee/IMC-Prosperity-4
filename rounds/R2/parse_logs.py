"""
Parse R1 submission logs into CSVs matching the DataCapsule_R2 schema.

Inputs:
  DataCapsule_R2/full fourth day data.log      (full day 1 submission)
  DataCapsule_R2/first tenth of 4th day/203633.log  (first-tenth tuning run)

Outputs in DataCapsule_R2/:
  prices_round_1_day_1_full.csv
  trades_round_1_day_1_full.csv
  prices_round_1_day_1_partial.csv
  trades_round_1_day_1_partial.csv

Schemas match existing files:
  prices: day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;...
          ...ask_price_3;ask_volume_3;mid_price;profit_and_loss
  trades: timestamp;buyer;seller;symbol;currency;price;quantity
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "DataCapsule_R2"

SOURCES = [
    ("full",    DATA_DIR / "full fourth day data.log"),
    ("partial", DATA_DIR / "first tenth of 4th day" / "203633.log"),
]

TRADE_FIELDS = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]


def write_prices_csv(activities_log: str, out_path: Path) -> int:
    rows = 0
    with out_path.open("w", newline="") as f:
        f.write(activities_log if activities_log.endswith("\n") else activities_log + "\n")
        # count data rows (exclude header + trailing blank)
        rows = sum(1 for line in activities_log.splitlines()[1:] if line.strip())
    return rows


def write_trades_csv(trade_history: list[dict], out_path: Path) -> int:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(TRADE_FIELDS)
        rows = 0
        for t in trade_history:
            writer.writerow([t.get(k, "") for k in TRADE_FIELDS])
            rows += 1
    return rows


def main() -> None:
    for label, path in SOURCES:
        if not path.is_file():
            print(f"[skip] {path} missing")
            continue

        payload = json.loads(path.read_text())
        prices_out = DATA_DIR / f"prices_round_1_day_1_{label}.csv"
        trades_out = DATA_DIR / f"trades_round_1_day_1_{label}.csv"

        p_rows = write_prices_csv(payload["activitiesLog"], prices_out)
        t_rows = write_trades_csv(payload.get("tradeHistory", []), trades_out)

        print(f"[{label}] {prices_out.name}: {p_rows} rows")
        print(f"[{label}] {trades_out.name}: {t_rows} rows")


if __name__ == "__main__":
    main()
