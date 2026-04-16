"""Walk-forward analysis across 3 train/test splits for both products.

Windows:
  W1: train=[-2]            test=[-1]
  W2: train=[-2,-1]         test=[0]
  W3: train=[-2,-1,0]       test=[3_partial]

Per (window, product) compute:
  - mid-price slope (linear regression on tick index)
  - spread mean/std (best_ask - best_bid)
  - order book imbalance (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1), mean
  - return autocorrelation lag-1 (mid-to-mid log returns)
  - VWAP-mid prediction skill: corr(vwap_t - mid_t,  mid_{t+1} - mid_t)
       vwap_t = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (bid_vol_1 + ask_vol_1)
       (size-weighted, "lean" — heavier opposite size pulls fair value toward that side)
  - trade flow: buy_init_frac, sell_init_frac
       buy-init  = trade price >= mid_at_or_before_t  (aggressor lifted ask)
       sell-init = trade price <= mid_at_or_before_t  (aggressor hit bid)
     and predictive value: corr(net_flow_per_tick, next-tick mid change)

Outputs printed: per-product table; train vs test comparison; stable/inconsistent flags.
"""
from pathlib import Path
import csv
import math
import statistics

DC = Path("/Users/indigolee/Desktop/imcprosperity4/rounds/R1/DataCapsule_R1")

PRICES = {
    -2: DC / "prices_round_1_day_-2.csv",
    -1: DC / "prices_round_1_day_-1.csv",
     0: DC / "prices_round_1_day_0.csv",
    "3p": DC / "prices_round_1_day_3_partial.csv",
}
TRADES = {
    -2: DC / "trades_round_1_day_-2.csv",
    -1: DC / "trades_round_1_day_-1.csv",
     0: DC / "trades_round_1_day_0.csv",
    "3p": DC / "trades_round_1_day_3_partial.csv",
}

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]


def load_prices(paths):
    """Return dict[product] -> list of dict rows sorted by (day, timestamp)."""
    out = {p: [] for p in PRODUCTS}
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for r in reader:
                p = r["product"]
                if p not in out:
                    continue
                def f1(k):
                    v = r.get(k, "")
                    return float(v) if v not in ("", None) else None
                row = {
                    "day": int(r["day"]),
                    "ts": int(r["timestamp"]),
                    "bp1": f1("bid_price_1"), "bv1": f1("bid_volume_1"),
                    "ap1": f1("ask_price_1"), "av1": f1("ask_volume_1"),
                    "mid": f1("mid_price"),
                }
                out[p].append(row)
    # Sort by day then timestamp
    for p in out:
        out[p].sort(key=lambda r: (r["day"], r["ts"]))
    return out


def load_trades(paths):
    out = {p: [] for p in PRODUCTS}
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for r in reader:
                p = r["symbol"]
                if p not in out:
                    continue
                out[p].append({
                    "ts": int(r["timestamp"]),
                    "buyer": r.get("buyer", "") or "",
                    "seller": r.get("seller", "") or "",
                    "price": float(r["price"]),
                    "qty": int(r["quantity"]),
                })
    for p in out:
        out[p].sort(key=lambda r: r["ts"])
    return out


def linreg_slope(ys):
    n = len(ys)
    if n < 2: return float("nan")
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    return num/den if den else float("nan")


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))
             and not (isinstance(y, float) and math.isnan(y))]
    n = len(pairs)
    if n < 3: return float("nan")
    mx = sum(p[0] for p in pairs)/n
    my = sum(p[1] for p in pairs)/n
    sx = math.sqrt(sum((p[0]-mx)**2 for p in pairs))
    sy = math.sqrt(sum((p[1]-my)**2 for p in pairs))
    if sx == 0 or sy == 0: return float("nan")
    cov = sum((p[0]-mx)*(p[1]-my) for p in pairs)
    return cov/(sx*sy)


def metrics_for(prices_rows, trades_rows):
    """Compute per-product metrics dict given a sorted list of price rows (single product)."""
    if len(prices_rows) < 5:
        return None

    mids = [r["mid"] for r in prices_rows if r["mid"] is not None and r["mid"] > 0]
    spreads = [r["ap1"] - r["bp1"] for r in prices_rows
               if r["ap1"] is not None and r["bp1"] is not None]
    imbs = []
    leans = []        # vwap - mid (size-weighted lean)
    next_dmids = []   # mid_{t+1} - mid_t aligned to leans
    for i, r in enumerate(prices_rows):
        bv, av = r["bv1"], r["av1"]
        bp, ap, mid = r["bp1"], r["ap1"], r["mid"]
        if bv is not None and av is not None and (bv + av) > 0:
            imbs.append((bv - av) / (bv + av))
        if (bv is not None and av is not None and bp is not None and ap is not None
                and (bv + av) > 0 and mid is not None and mid > 0):
            vwap = (bp * av + ap * bv) / (bv + av)  # heavier opposite size pulls toward that side
            lean = vwap - mid
            # next available mid in the same product list
            nxt = None
            for j in range(i+1, len(prices_rows)):
                if prices_rows[j]["mid"] is not None and prices_rows[j]["mid"] > 0:
                    nxt = prices_rows[j]["mid"]; break
            if nxt is not None:
                leans.append(lean)
                next_dmids.append(nxt - mid)

    # log returns + autocorr lag-1
    rets = []
    for a, b in zip(mids[:-1], mids[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b/a))
    autocorr1 = corr(rets[:-1], rets[1:]) if len(rets) > 3 else float("nan")

    # Trade flow: classify each trade against last available mid before its timestamp
    # Build a sorted (ts, mid) list (use rows where mid valid)
    ts_mid = [(r["ts"] + r["day"]*10**7, r["mid"]) for r in prices_rows
              if r["mid"] is not None and r["mid"] > 0]
    # We need a flat ts (day*1e7 + ts) for trades too, but trades come per-day. To keep this
    # simple: classify only trades whose ts has a same-day mid before. Since walk windows pass
    # full days, we just iterate trades + a per-day pointer.
    buy_init = sell_init = neutral = 0
    flow_per_tick = {}   # (day, ts) -> net signed qty (buy_init - sell_init)
    # Build per-day mid lookup keyed by ts
    per_day_ts_mid = {}  # day -> sorted list of (ts, mid)
    for r in prices_rows:
        if r["mid"] is not None and r["mid"] > 0:
            per_day_ts_mid.setdefault(r["day"], []).append((r["ts"], r["mid"]))
    for d in per_day_ts_mid:
        per_day_ts_mid[d].sort()

    # Trades CSVs lack a 'day' column; we partition by ts ranges per day. Each day's ts range is
    # 0..99900. When walk windows include multiple days, the trades CSVs are concatenated
    # without any day field, so trades from different days share overlapping ts. To avoid
    # mis-classification we instead match by (day index in iteration, ts).
    # Solution: caller passes trades_by_day not trades_rows. See below — we adapt by interleaving.

    # If trades_rows already stamped with 'day' field, use that.
    flow_xs, flow_ys = [], []
    for t in trades_rows:
        d = t.get("day")
        if d is None or d not in per_day_ts_mid: continue
        # binary search the last (ts, mid) with ts <= trade ts
        lst = per_day_ts_mid[d]
        lo, hi = 0, len(lst)
        while lo < hi:
            mi = (lo+hi)//2
            if lst[mi][0] <= t["ts"]: lo = mi + 1
            else: hi = mi
        if lo == 0: continue
        ref_mid = lst[lo-1][1]
        side = 0
        if t["price"] > ref_mid + 1e-9: side = +1; buy_init += t["qty"]
        elif t["price"] < ref_mid - 1e-9: side = -1; sell_init += t["qty"]
        else: neutral += t["qty"]
        if side != 0:
            key = (d, t["ts"])
            flow_per_tick[key] = flow_per_tick.get(key, 0) + side * t["qty"]

    # Predictive value of trade flow: align flow_per_tick to next-tick mid change
    for (d, ts), net in flow_per_tick.items():
        lst = per_day_ts_mid[d]
        # find mid AT this ts (or last <=) and next mid >
        lo, hi = 0, len(lst)
        while lo < hi:
            mi = (lo+hi)//2
            if lst[mi][0] <= ts: lo = mi + 1
            else: hi = mi
        if lo == 0 or lo >= len(lst): continue
        cur_mid = lst[lo-1][1]; nxt_mid = lst[lo][1]
        flow_xs.append(net); flow_ys.append(nxt_mid - cur_mid)

    total_classified = buy_init + sell_init + neutral
    return {
        "n_ticks": len(mids),
        "mid_slope_per_tick": linreg_slope(mids),
        "mid_drift_total": (mids[-1] - mids[0]) if mids else float("nan"),
        "spread_mean": statistics.fmean(spreads) if spreads else float("nan"),
        "spread_std": statistics.pstdev(spreads) if len(spreads) > 1 else float("nan"),
        "imbalance_mean": statistics.fmean(imbs) if imbs else float("nan"),
        "imbalance_std": statistics.pstdev(imbs) if len(imbs) > 1 else float("nan"),
        "ret_autocorr1": autocorr1,
        "vwap_lean_corr_next_dmid": corr(leans, next_dmids),
        "buy_init_frac": buy_init / total_classified if total_classified else float("nan"),
        "sell_init_frac": sell_init / total_classified if total_classified else float("nan"),
        "neutral_frac": neutral / total_classified if total_classified else float("nan"),
        "flow_corr_next_dmid": corr(flow_xs, flow_ys),
        "n_trades": len(trades_rows),
    }


def stamp_trades_with_day(trades_rows, day_label):
    """Tag trades with their day so cross-day windows still classify correctly."""
    for t in trades_rows:
        t["day"] = day_label
    return trades_rows


def load_window(days):
    prices = {p: [] for p in PRODUCTS}
    trades = {p: [] for p in PRODUCTS}
    for d in days:
        # day_label used to bucket trades; for prices we already preserve r["day"] from the CSV
        # but for our 3_partial day we relabel = 3.
        day_label = 3 if d == "3p" else d
        # load prices for this single day, set the day field in-loaded rows is already correct
        per = load_prices([PRICES[d]])
        for p in PRODUCTS:
            for r in per[p]:
                # ensure day field reflects desired label
                r["day"] = day_label
                prices[p].append(r)
        per_t = load_trades([TRADES[d]])
        for p in PRODUCTS:
            stamp_trades_with_day(per_t[p], day_label)
            trades[p].extend(per_t[p])
    # Sort
    for p in PRODUCTS:
        prices[p].sort(key=lambda r: (r["day"], r["ts"]))
        trades[p].sort(key=lambda r: (r["day"], r["ts"]))
    return prices, trades


WINDOWS = [
    ("W1", [-2],          [-1]),
    ("W2", [-2, -1],      [0]),
    ("W3", [-2, -1, 0],   ["3p"]),
]

def fmt(x):
    if x is None: return "  -   "
    if isinstance(x, float):
        if math.isnan(x): return "  nan "
        if abs(x) >= 1000: return f"{x:8.1f}"
        if abs(x) >= 1:    return f"{x:8.3f}"
        return f"{x:8.4f}"
    return str(x)

def print_table(label, rows, products, keys):
    print(f"\n=== {label} ===")
    headers = ["metric"] + products
    widths = [max(len(h), 12) for h in headers]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-"*w for w in widths))
    for k in keys:
        line = [k.ljust(widths[0])]
        for p, w in zip(products, widths[1:]):
            v = rows.get(p, {}).get(k)
            line.append(fmt(v).ljust(w))
        print("  " + "  ".join(line))


def main():
    KEYS = [
        "n_ticks", "n_trades",
        "mid_slope_per_tick", "mid_drift_total",
        "spread_mean", "spread_std",
        "imbalance_mean", "imbalance_std",
        "ret_autocorr1",
        "vwap_lean_corr_next_dmid",
        "buy_init_frac", "sell_init_frac", "neutral_frac",
        "flow_corr_next_dmid",
    ]
    # Compute and store
    all_results = {}  # window_name -> {"train": {p: metrics}, "test": {p: metrics}}
    for name, train_days, test_days in WINDOWS:
        train_prices, train_trades = load_window(train_days)
        test_prices, test_trades   = load_window(test_days)
        train_m = {p: metrics_for(train_prices[p], train_trades[p]) for p in PRODUCTS}
        test_m  = {p: metrics_for(test_prices[p],  test_trades[p])  for p in PRODUCTS}
        all_results[name] = {"train": train_m, "test": test_m,
                             "train_days": train_days, "test_days": test_days}

    for name in ["W1", "W2", "W3"]:
        td = all_results[name]["train_days"]; sd = all_results[name]["test_days"]
        print_table(f"{name} TRAIN  days={td}", all_results[name]["train"], PRODUCTS, KEYS)
        print_table(f"{name} TEST   days={sd}", all_results[name]["test"],  PRODUCTS, KEYS)

    # Stable vs inconsistent across the 3 TEST sets
    print("\n\n##### CROSS-WINDOW STABILITY (TEST sets) #####")
    for p in PRODUCTS:
        print(f"\n--- {p} ---")
        for k in KEYS:
            vals = []
            for name in ["W1", "W2", "W3"]:
                m = all_results[name]["test"].get(p)
                v = m.get(k) if m else None
                vals.append(v)
            # stability: report values + flag
            num_vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
            flag = "n/a"
            if len(num_vals) == 3:
                mu = statistics.fmean(num_vals)
                sd = statistics.pstdev(num_vals)
                # Use a sign-stability flag: STABLE if all same sign AND coeff of variation modest
                same_sign = all(v >= 0 for v in num_vals) or all(v <= 0 for v in num_vals)
                cv = sd / abs(mu) if abs(mu) > 1e-9 else float("inf")
                if same_sign and cv < 0.5:    flag = "STABLE"
                elif same_sign and cv < 1.0:  flag = "moderate"
                else:                         flag = "INCONSISTENT"
            print(f"  {k:32s}  W1={fmt(vals[0])}  W2={fmt(vals[1])}  W3={fmt(vals[2])}   -> {flag}")


if __name__ == "__main__":
    main()
