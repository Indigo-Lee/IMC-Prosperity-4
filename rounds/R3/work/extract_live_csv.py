"""Extract the live submission activity log into a prices CSV that
prosperity2bt can replay. Saves under the live_data/ directory with day=3 (so
it slots in after the historical 0/1/2)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

JSN = Path.home() / "Desktop/imcprosperity4/rounds/R3/ROUND_3/437594/437594.json"
OUT_DIR = Path.home() / "Desktop/imcprosperity4/rounds/R3/work/live_data"
OUT_DIR.mkdir(exist_ok=True)

with open(JSN) as f:
    blob = json.load(f)

df = pd.read_csv(io.StringIO(blob["activitiesLog"]), sep=";")
print(f"raw rows: {len(df)}, day(s): {sorted(df['day'].unique())}, "
      f"ts range {df['timestamp'].min()}..{df['timestamp'].max()}")

# Relabel as day 3.
df["day"] = 3

# Bid/ask price+volume columns get promoted to float by pandas when NaN is
# present in the column. prosperity2bt expects ints. Cast back, leaving NaN as
# empty strings (matches the historical CSV format).
INT_COLS = [f"{side}_{kind}_{lvl}"
            for side in ("bid", "ask")
            for kind in ("price", "volume")
            for lvl in (1, 2, 3)]
for col in INT_COLS:
    if col in df.columns:
        df[col] = df[col].apply(lambda v: "" if pd.isna(v) else str(int(v)))

out_path = OUT_DIR / "prices_round_3_day_3.csv"
df.to_csv(out_path, sep=";", index=False)
print(f"Wrote {out_path}  ({len(df)} rows)")

# Empty trades file (live submission JSON doesn't include market trades).
trades_cols = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
empty = pd.DataFrame(columns=trades_cols)
trades_path = OUT_DIR / "trades_round_3_day_3.csv"
empty.to_csv(trades_path, sep=";", index=False)
print(f"Wrote {trades_path}  (empty trades)")
