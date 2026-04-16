"""Parse the simulated day 3 .log file (JSON: activitiesLog + tradeHistory) into the
DataCapsule_R1 CSV format. Output:
  - prices_round_1_day_3_partial.csv  (relabel day column 0 -> 3)
  - trades_round_1_day_3_partial.csv  (preserve SUBMISSION labels for Step 2 flow analysis)
"""
from pathlib import Path
import json

SRC = Path("/Users/indigolee/Desktop/imcprosperity4/Simulated day3_t0_to_t9999/203633.log")
DST = Path("/Users/indigolee/Desktop/imcprosperity4/rounds/R1/DataCapsule_R1")

obj = json.loads(SRC.read_text())

# --- activitiesLog: pipe-delimited orderbook block ---
activities = obj["activitiesLog"].strip()
lines = activities.splitlines()
header = lines[0]
out_prices = [header]
for ln in lines[1:]:
    if not ln.strip():
        continue
    parts = ln.split(";")
    parts[0] = "3"  # relabel day 0 -> 3
    out_prices.append(";".join(parts))

(DST / "prices_round_1_day_3_partial.csv").write_text("\n".join(out_prices) + "\n")
print(f"prices: wrote {len(out_prices)-1} rows")

# --- tradeHistory: list of trade dicts ---
trades = obj["tradeHistory"]
trade_lines = ["timestamp;buyer;seller;symbol;currency;price;quantity"]
for t in trades:
    trade_lines.append(
        f"{t['timestamp']};{t.get('buyer','') or ''};{t.get('seller','') or ''};"
        f"{t['symbol']};{t.get('currency','XIRECS')};{float(t['price'])};{int(t['quantity'])}"
    )

(DST / "trades_round_1_day_3_partial.csv").write_text("\n".join(trade_lines) + "\n")
print(f"trades: wrote {len(trade_lines)-1} rows")
