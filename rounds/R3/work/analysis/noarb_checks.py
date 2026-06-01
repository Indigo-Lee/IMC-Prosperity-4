"""Phase 1.3 — no-arbitrage bound checks (HIGHEST-PRIORITY SIGNAL).

For every timestamp, check:
  - Upper bound: voucher <= S
  - Lower bound: voucher >= max(S - K, 0)
  - Monotonicity: C(K1) >= C(K2) for K1 < K2
  - Butterfly convexity: C(K1) - 2*C(K2) + C(K3) >= 0  for equally-spaced strikes

For each, we report:
  (a) MID-based violation rate (informational): tells us if the bot model itself
      is mispricing.
  (b) EXECUTABLE arb (using bid/ask prices): is there a free trade we can take?
      e.g. lower bound: buy voucher_ask, sell underlying_bid, payoff at expiry
      pays at least max(S-K,0), so if voucher_ask + K < S_bid we have an
      instant edge.
  (c) Persistence: distribution of run lengths of consecutive violation ts.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "out"

VOUCHERS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
STRIKE = {v: int(v.split("_")[1]) for v in VOUCHERS}


def best_book(prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide best_bid / best_ask / mid frames per product."""
    wb = prices.pivot_table(index="abs_ts", columns="product", values="best_bid", aggfunc="last").sort_index()
    wa = prices.pivot_table(index="abs_ts", columns="product", values="best_ask", aggfunc="last").sort_index()
    wm = prices.pivot_table(index="abs_ts", columns="product", values="mid", aggfunc="last").sort_index()
    return wb, wa, wm


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return the lengths of consecutive True runs in a boolean array."""
    if not mask.any():
        return np.array([], dtype=int)
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def summarize_violation(name: str, mask: np.ndarray, mag: np.ndarray | None = None) -> dict:
    n = len(mask)
    n_viol = int(mask.sum())
    rls = run_lengths(mask)
    info = {
        "check": name,
        "n_ts": n,
        "n_violation_ts": n_viol,
        "violation_rate": n_viol / n if n else 0.0,
        "n_runs": len(rls),
        "max_run_len": int(rls.max()) if len(rls) else 0,
        "p90_run_len": int(np.percentile(rls, 90)) if len(rls) else 0,
        "median_run_len": float(np.median(rls)) if len(rls) else 0.0,
    }
    if mag is not None and n_viol > 0:
        m = mag[mask]
        info["mag_p50"] = float(np.median(m))
        info["mag_p90"] = float(np.percentile(m, 90))
        info["mag_max"] = float(m.max())
        info["mag_mean"] = float(m.mean())
    return info


def main() -> None:
    prices = pd.read_pickle(OUT / "prices.pkl")
    wb, wa, wm = best_book(prices)

    s_mid = wm["VELVETFRUIT_EXTRACT"].values
    s_bid = wb["VELVETFRUIT_EXTRACT"].values
    s_ask = wa["VELVETFRUIT_EXTRACT"].values
    ts = wm.index.values

    rows = []
    examples_rows = []

    # =========================
    # 1) Upper bound: C <= S
    # =========================
    for v in VOUCHERS:
        c_mid = wm[v].values
        c_bid = wb[v].values
        # MID-based: C_mid > S_mid
        mask_mid = c_mid > s_mid
        mag_mid = np.where(mask_mid, c_mid - s_mid, 0.0)
        rows.append({**summarize_violation(f"upper_mid:{v}", mask_mid, mag_mid)})

        # EXECUTABLE: sell voucher_bid, buy underlying_ask. Profit if c_bid > s_ask.
        mask_exe = c_bid > s_ask
        mag_exe = np.where(mask_exe, c_bid - s_ask, 0.0)
        rows.append({**summarize_violation(f"upper_exe:{v}", mask_exe, mag_exe)})

    # =========================
    # 2) Lower bound: C >= max(S - K, 0)
    # =========================
    for v in VOUCHERS:
        K = STRIKE[v]
        c_mid = wm[v].values
        c_ask = wa[v].values
        intrinsic_mid = np.maximum(s_mid - K, 0.0)
        # MID-based: C_mid < max(S_mid - K, 0)
        mask_mid = c_mid < intrinsic_mid - 1e-9
        mag_mid = np.where(mask_mid, intrinsic_mid - c_mid, 0.0)
        rows.append({**summarize_violation(f"lower_mid:{v}", mask_mid, mag_mid)})

        # EXECUTABLE: buy voucher_ask + (sell underlying_bid). Profit if c_ask + K < s_bid (when S>K).
        # General version: c_ask < max(s_bid - K, 0)
        intrinsic_exe = np.maximum(s_bid - K, 0.0)
        mask_exe = c_ask < intrinsic_exe - 1e-9
        mag_exe = np.where(mask_exe, intrinsic_exe - c_ask, 0.0)
        rows.append({**summarize_violation(f"lower_exe:{v}", mask_exe, mag_exe)})

    # =========================
    # 3) Monotonicity: C(K1) >= C(K2) for K1 < K2
    # =========================
    sorted_vouchers = sorted(VOUCHERS, key=lambda v: STRIKE[v])
    for i in range(len(sorted_vouchers) - 1):
        v1 = sorted_vouchers[i]; v2 = sorted_vouchers[i + 1]
        c1m, c2m = wm[v1].values, wm[v2].values
        c1b, c1a = wb[v1].values, wa[v1].values
        c2b, c2a = wb[v2].values, wa[v2].values

        # MID-based: C_mid(K1) < C_mid(K2)  (lower-strike call cheaper than higher-strike — wrong)
        mask_mid = c1m < c2m - 1e-9
        mag_mid = np.where(mask_mid, c2m - c1m, 0.0)
        rows.append({**summarize_violation(f"mono_mid:{v1}<{v2}", mask_mid, mag_mid)})

        # EXECUTABLE: buy v1 at ask, sell v2 at bid; profit if c2_bid > c1_ask
        mask_exe = c2b > c1a + 1e-9
        mag_exe = np.where(mask_exe, c2b - c1a, 0.0)
        rows.append({**summarize_violation(f"mono_exe:{v1}<{v2}", mask_exe, mag_exe)})

    # =========================
    # 4) Butterfly convexity over equally-spaced triplets
    # =========================
    # Build triplets (K1<K2<K3, K2-K1 == K3-K2)
    strikes = sorted({STRIKE[v] for v in VOUCHERS})
    inv_strike = {STRIKE[v]: v for v in VOUCHERS}
    triplets = []
    for k1, k2, k3 in combinations(strikes, 3):
        if k2 - k1 == k3 - k2:
            triplets.append((inv_strike[k1], inv_strike[k2], inv_strike[k3]))

    for v1, v2, v3 in triplets:
        c1m, c2m, c3m = wm[v1].values, wm[v2].values, wm[v3].values
        c1b, c2b, c3b = wb[v1].values, wb[v2].values, wb[v3].values
        c1a, c2a, c3a = wa[v1].values, wa[v2].values, wa[v3].values

        # MID: convex if C1 + C3 - 2*C2 >= 0
        butterfly_mid = c1m + c3m - 2 * c2m
        mask_mid = butterfly_mid < -1e-9
        mag_mid = np.where(mask_mid, -butterfly_mid, 0.0)
        rows.append({**summarize_violation(f"bfly_mid:{v1}/{v2}/{v3}", mask_mid, mag_mid)})

        # EXECUTABLE: buy 1×K1 at ask, buy 1×K3 at ask, sell 2×K2 at bid.
        #   Net cash at trade: -c1a - c3a + 2*c2b
        #   Payoff at expiry is convex butterfly >= 0, so if net cash > 0 → arb.
        net_cash = -c1a - c3a + 2 * c2b
        mask_exe = net_cash > 1e-9
        mag_exe = np.where(mask_exe, net_cash, 0.0)
        rows.append({**summarize_violation(f"bfly_exe:{v1}/{v2}/{v3}", mask_exe, mag_exe)})

    df = pd.DataFrame(rows)
    df["family"] = df["check"].str.split(":").str[0]
    df.to_csv(OUT / "noarb_summary.csv", index=False)

    # ===== Summary print =====
    print("=" * 100)
    print("NO-ARB CHECKS — summary (only checks with at least one violation shown)")
    print("=" * 100)
    nz = df[df["n_violation_ts"] > 0].copy()
    nz_sorted = nz.sort_values("violation_rate", ascending=False)
    if len(nz_sorted) == 0:
        print("\n(No violations found on any check.)")
    else:
        # show top by violation rate
        print(f"\nTotal checks: {len(df)}; with at least one violation: {len(nz)}")
        cols = ["check", "n_violation_ts", "violation_rate", "n_runs",
                "median_run_len", "max_run_len",
                "mag_p50", "mag_p90", "mag_max", "mag_mean"]
        cols = [c for c in cols if c in nz_sorted.columns]
        print(nz_sorted[cols].head(30).to_string(index=False))

    # Group rates by check-family
    print("\nViolation rate by check family:")
    fam = df.groupby("family").agg(
        total_checks=("check", "count"),
        avg_violation_rate=("violation_rate", "mean"),
        max_violation_rate=("violation_rate", "max"),
        sum_violation_ts=("n_violation_ts", "sum"),
    )
    print(fam.to_string())

    # Timeline plot (sum of executable violation indicators per 1000-ts bin, by family)
    fams_to_plot = ["upper_exe", "lower_exe", "mono_exe", "bfly_exe"]
    fig, axes = plt.subplots(len(fams_to_plot), 1, figsize=(12, 8), sharex=True)
    n_ts = len(ts)
    bin_size = 1000
    n_bins = (n_ts + bin_size - 1) // bin_size

    for ax, family in zip(axes, fams_to_plot):
        members = df[df["family"] == family]["check"].tolist()
        # accumulate per-bin counts
        bin_counts = np.zeros(n_bins)
        for chk in members:
            # Re-derive the boolean from the raw frames (only for plot — coarse)
            pass
        # Quick aggregate: total violation_ts as a constant horizontal — but we want timeline.
        # Recompute timeline per family by re-looping through checks and summing booleans.
        running = np.zeros(n_ts, dtype=int)
        if family == "upper_exe":
            for v in VOUCHERS:
                running = running + (wb[v].values > wa["VELVETFRUIT_EXTRACT"].values).astype(int)
        elif family == "lower_exe":
            for v in VOUCHERS:
                K = STRIKE[v]
                running = running + (wa[v].values < np.maximum(wb["VELVETFRUIT_EXTRACT"].values - K, 0.0) - 1e-9).astype(int)
        elif family == "mono_exe":
            for i in range(len(sorted_vouchers) - 1):
                v1 = sorted_vouchers[i]; v2 = sorted_vouchers[i + 1]
                running = running + (wb[v2].values > wa[v1].values + 1e-9).astype(int)
        elif family == "bfly_exe":
            for v1, v2, v3 in triplets:
                running = running + (-wa[v1].values - wa[v3].values + 2 * wb[v2].values > 1e-9).astype(int)

        # Bin
        binned = np.add.reduceat(running, np.arange(0, n_ts, bin_size))
        ax.bar(np.arange(len(binned)) * bin_size, binned, width=bin_size * 0.9, color="C3")
        ax.set_title(f"executable violations per {bin_size}-ts bin — {family}")
        ax.set_ylabel("count (sum across pairs)")
    axes[-1].set_xlabel("abs_ts")
    fig.tight_layout()
    fig.savefig(OUT / "noarb_timeline.png", dpi=110)
    plt.close(fig)

    # Top-magnitude examples for each executable family
    print("\nTop-magnitude executable violations (by family):")
    for family in fams_to_plot:
        members = nz_sorted[nz_sorted["family"] == family].head(5)
        if len(members):
            print(f"\n  [{family}] highest violation_rate (top 5):")
            print(members[["check", "n_violation_ts", "violation_rate", "mag_p50", "mag_p90", "mag_max"]].to_string(index=False))

    print(f"\nWrote noarb_summary.csv, noarb_timeline.png → {OUT}")


if __name__ == "__main__":
    main()
