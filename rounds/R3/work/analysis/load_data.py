"""Phase 1.1 — data loader for IMC Prosperity Round 3.

Loads prices_round_3_day_{0,1,2}.csv and trades_round_3_day_{0,1,2}.csv,
concatenates with a day-aware monotonic timestamp, and computes mid /
microprice / depth-weighted-mid per (product, timestamp).

Run directly to see a head() of the cleaned wide-mid frame plus
per-product timestamp coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("~/Desktop/imcprosperity4/rounds/R3/ROUND_3").expanduser()
DAYS = (0, 1, 2)
DAY_OFFSET = 1_000_000  # so day 1 starts at ts >= 1_000_000

PRODUCTS_DELTA1 = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")
VOUCHERS = (
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
)
ALL_PRODUCTS = PRODUCTS_DELTA1 + VOUCHERS

VOUCHER_STRIKE = {v: int(v.split("_")[1]) for v in VOUCHERS}


def _read_prices_one_day(day: int) -> pd.DataFrame:
    path = DATA_DIR / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df["day_index"] = day
    df["abs_ts"] = df["timestamp"] + day * DAY_OFFSET
    return df


def _read_trades_one_day(day: int) -> pd.DataFrame:
    path = DATA_DIR / f"trades_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df["day_index"] = day
    df["abs_ts"] = df["timestamp"] + day * DAY_OFFSET
    return df


def _best_bid(row) -> Optional[float]:
    bp1 = row.get("bid_price_1")
    return None if pd.isna(bp1) else float(bp1)


def _best_ask(row) -> Optional[float]:
    ap1 = row.get("ask_price_1")
    return None if pd.isna(ap1) else float(ap1)


def _microprice(row) -> Optional[float]:
    bp = row.get("bid_price_1"); ap = row.get("ask_price_1")
    bv = row.get("bid_volume_1"); av = row.get("ask_volume_1")
    if pd.isna(bp) or pd.isna(ap) or pd.isna(bv) or pd.isna(av):
        return None
    denom = bv + av
    if denom <= 0:
        return None
    return (bp * av + ap * bv) / denom


def _weighted_mid(row) -> Optional[float]:
    # Volume-weighted mid using up to 3 levels each side.
    bid_num = 0.0; bid_den = 0.0
    ask_num = 0.0; ask_den = 0.0
    for k in (1, 2, 3):
        bp = row.get(f"bid_price_{k}"); bv = row.get(f"bid_volume_{k}")
        if not (pd.isna(bp) or pd.isna(bv)):
            bid_num += bp * bv; bid_den += bv
        ap = row.get(f"ask_price_{k}"); av = row.get(f"ask_volume_{k}")
        if not (pd.isna(ap) or pd.isna(av)):
            ask_num += ap * av; ask_den += av
    if bid_den <= 0 or ask_den <= 0:
        return None
    bid_w = bid_num / bid_den
    ask_w = ask_num / ask_den
    return 0.5 * (bid_w + ask_w)


def _book_depth(row) -> int:
    total = 0
    for side in ("bid_volume", "ask_volume"):
        for k in (1, 2, 3):
            v = row.get(f"{side}_{k}")
            if not pd.isna(v):
                total += int(v)
    return total


def load_prices(days: tuple = DAYS) -> pd.DataFrame:
    frames = [_read_prices_one_day(d) for d in days]
    df = pd.concat(frames, ignore_index=True)

    df["best_bid"] = df.apply(_best_bid, axis=1)
    df["best_ask"] = df.apply(_best_ask, axis=1)
    df["spread"] = df["best_ask"] - df["best_bid"]
    df["mid"] = df.apply(
        lambda r: 0.5 * (r["best_bid"] + r["best_ask"])
        if pd.notna(r["best_bid"]) and pd.notna(r["best_ask"]) else np.nan,
        axis=1,
    )
    df["microprice"] = df.apply(_microprice, axis=1)
    df["weighted_mid"] = df.apply(_weighted_mid, axis=1)
    df["book_depth"] = df.apply(_book_depth, axis=1)

    return df


def load_trades(days: tuple = DAYS) -> pd.DataFrame:
    frames = [_read_trades_one_day(d) for d in days]
    return pd.concat(frames, ignore_index=True)


def wide_mid(prices_df: pd.DataFrame, col: str = "mid") -> pd.DataFrame:
    """Return a wide DataFrame: index=abs_ts, columns=product, values=col."""
    return prices_df.pivot_table(
        index="abs_ts", columns="product", values=col, aggfunc="last"
    ).sort_index()


def coverage_report(prices_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prod in ALL_PRODUCTS:
        sub = prices_df[prices_df["product"] == prod]
        n = len(sub)
        n_mid = sub["mid"].notna().sum()
        n_micro = sub["microprice"].notna().sum()
        med_spread = sub["spread"].median()
        med_depth = sub["book_depth"].median()
        rows.append({
            "product": prod,
            "n_rows": n,
            "n_mid": n_mid,
            "nan_mid_rate": 1 - n_mid / max(n, 1),
            "median_spread": med_spread,
            "median_book_depth": med_depth,
            "first_ts": sub["abs_ts"].min(),
            "last_ts": sub["abs_ts"].max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 80)
    print("LOADING PRICES (3 days × 12 products)")
    print("=" * 80)
    prices = load_prices()
    print(f"Total price rows: {len(prices):,}")
    print(f"Distinct products: {prices['product'].nunique()}")
    print(f"abs_ts range: {prices['abs_ts'].min()} … {prices['abs_ts'].max()}")

    print("\nRaw price head() (first product encountered):")
    print(prices.head(3).to_string())

    print("\nWide mid frame head():")
    wm = wide_mid(prices, "mid")
    print(wm.head().to_string())

    print("\nWide mid frame tail():")
    print(wm.tail().to_string())

    print("\nCoverage report:")
    cov = coverage_report(prices)
    print(cov.to_string(index=False))

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    cov.to_csv(out_dir / "coverage.csv", index=False)
    wm.to_pickle(out_dir / "wide_mid.pkl")
    prices.to_pickle(out_dir / "prices.pkl")

    print("\nLOADING TRADES")
    trades = load_trades()
    print(f"Total trade rows: {len(trades):,}")
    print("\nTrades head():")
    print(trades.head().to_string())
    print("\nTrades per product:")
    print(trades["symbol"].value_counts().to_string())

    trades.to_pickle(out_dir / "trades.pkl")
    print(f"\nSaved coverage.csv, wide_mid.pkl, prices.pkl, trades.pkl → {out_dir}")


if __name__ == "__main__":
    main()
