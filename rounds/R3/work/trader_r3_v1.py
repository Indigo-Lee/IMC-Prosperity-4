"""IMC Prosperity Round 3 — trader_r3_v1.

Strategy composition is dictated by Phase 1 findings (see work/findings.md):

  1. HYDROGEL_PACK market making at fair=10000 (R1 ASH playbook).
     Reason: OU half-life ~300 ticks, AR(1) Δmid b≈-0.13 (t≈-23) every day,
     spread=16, fair-value pinned at 10000.
  2. VELVETFRUIT_EXTRACT market making with EMA fair value.
     Reason: same OU/AR(1) profile, but mid drifts slowly (5246→5255 across days).
  3. Voucher IV-residual mean-reversion vs per-tick parabolic smile fit.
     Reason: residual sign is consistent across all 3 days for 9/10 vouchers,
     ACF(1) is high (0.4–0.9) so mispricing persists.
  4. Free-call lottery on VEV_6000/VEV_6500.
     Reason: bots dump worthless options at price 0; absorbing them costs nothing
     and has nonzero (tiny) expected value.

Excluded from voucher trading:
  - VEV_4000, VEV_4500: priced at intrinsic, BS IV undefined (skip the smile fit).
  - VEV_5100: residual sign flips between days — not directionally consistent.

Confirmed dead-ends (no code wasted on these):
  - No-arb arbitrage: 0 executable violations across 41 checks × 30k ts.
  - HYDROGEL leads VELVETFRUIT: max |xcorr| = 0.012 (no signal).
  - Trade-flow → forward mid: max R² = 0.005 (no signal).
  - Wide-quote pickoff: bots fill at touch only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState

# =============================================================================
# Products
# =============================================================================
HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

VOUCHERS = (
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
)
STRIKE = {v: int(v.split("_")[1]) for v in VOUCHERS}

# Vouchers used for the smile FIT (exclude pinned-at-intrinsic deep ITMs).
SMILE_FIT_VOUCHERS = tuple(v for v in VOUCHERS if v not in ("VEV_4000", "VEV_4500"))

# Vouchers we TRADE on residual signal (exclude pinned + sign-flippy VEV_5100).
RESIDUAL_TRADE_VOUCHERS = tuple(
    v for v in VOUCHERS if v not in ("VEV_4000", "VEV_4500", "VEV_5100")
)

# Vouchers eligible for free-call lottery (mid at floor 0.5).
LOTTERY_VOUCHERS = ("VEV_6000", "VEV_6500")

# Position limits (problem statement)
LIMITS = {HYDROGEL: 200, VELVET: 200}
for v in VOUCHERS:
    LIMITS[v] = 300

# =============================================================================
# Time-to-expiry config — set externally by backtester per day.
# Live default: 5 days (start of round 3). For backtesting historical data:
#   day_0 → 8.0, day_1 → 7.0, day_2 → 6.0
# =============================================================================
TTE_INITIAL_DAYS = 5.0
DAYS_TO_YEARS = 365.0  # mapping used in IV solver

# =============================================================================
# HYDROGEL_PACK params (R2.1 ASH-style)
# =============================================================================
HYDRO_FAIR = 10_000
HYDRO_TAKE_WIDTH = 1
HYDRO_CLEAR_WIDTH = 0
HYDRO_DISREGARD_EDGE = 1
HYDRO_JOIN_EDGE = 2
HYDRO_DEFAULT_EDGE = 4
HYDRO_SOFT_POSITION = 30  # in [0, 200]; nudge quotes when |pos| > this

# =============================================================================
# VELVETFRUIT_EXTRACT params
# =============================================================================
VELVET_EMA_ALPHA = 0.005   # EMA on mid → tracks slow drift (half-life ~140 ticks)
VELVET_TAKE_WIDTH = 1
VELVET_CLEAR_WIDTH = 0
VELVET_DISREGARD_EDGE = 1
VELVET_JOIN_EDGE = 1
VELVET_DEFAULT_EDGE = 2
VELVET_SOFT_POSITION = 30

# =============================================================================
# Voucher residual-MR params
# =============================================================================
RESID_BUFFER_SIZE = 200      # rolling window for residual mean/std (ring buffer)
RESID_MIN_FOR_TRADE = 30     # need this many obs before trading
RESID_Z_THRESHOLD = 1.0      # |z| above which we start scaling in
RESID_Z_FULL_SIZE = 3.0      # |z| at which we hit full position
RESID_VOUCHER_CAP_FRACTION = 0.7  # max fraction of LIMITS used for residual trades
                                  # (leaves room for delta hedge / lottery)

# Per-voucher conviction multiplier (1.0 = baseline).
# VEV_5400 had ACF(1)=0.93 day0 and the strongest negative residual every day.
# VEV_5200, VEV_5300 long-side conviction also high.
RESID_CONVICTION = {
    "VEV_5000": 0.7,
    "VEV_5200": 1.0,
    "VEV_5300": 1.0,
    "VEV_5400": 1.3,
    "VEV_5500": 1.0,
    "VEV_6000": 0.8,
    "VEV_6500": 0.8,
}

# =============================================================================
# Free-call lottery params
# =============================================================================
LOTTERY_BID_PRICE = 0        # bid at 0 — bots dump VEV_6000/6500 at this price
LOTTERY_TARGET_QTY = 300     # full long up to position limit (cost = 0)

# =============================================================================
# Delta governor
# =============================================================================
DELTA_CAP = 280              # absolute cap on net total delta in underlying-equivalent units
HEDGE_PRODUCT = VELVET       # we hedge using VELVETFRUIT_EXTRACT

# =============================================================================
# Black-Scholes utilities (pure stdlib; uses math.erf for normal CDF)
# =============================================================================
SQRT2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * norm_pdf(d1) * sqrtT


def implied_vol(price: float, S: float, K: float, T: float,
                seed: float = 0.5, max_iter: int = 50, tol: float = 1e-5) -> float | None:
    """Bisection IV solver. Return None if undefined / out of bracket."""
    intrinsic = max(S - K, 0.0)
    if not (intrinsic + 1e-9 < price < S - 1e-9):
        return None
    if T <= 0:
        return None
    lo, hi = 1e-4, 5.0
    # Confirm bracket
    if bs_call(S, K, T, lo) > price or bs_call(S, K, T, hi) < price:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c = bs_call(S, K, T, mid)
        if abs(c - price) < tol:
            return mid
        if c < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# Helpers
# =============================================================================

def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None, int, int]:
    """Return (best_bid, best_ask, bid_vol, ask_vol). None on empty side."""
    bb = max(depth.buy_orders) if depth.buy_orders else None
    ba = min(depth.sell_orders) if depth.sell_orders else None
    bv = depth.buy_orders[bb] if bb is not None else 0
    av = -depth.sell_orders[ba] if ba is not None else 0
    return bb, ba, bv, av


def safe_mid(depth: OrderDepth) -> float | None:
    bb, ba, _, _ = best_bid_ask(depth)
    if bb is None or ba is None or ba <= bb:
        return None
    return 0.5 * (bb + ba)


# =============================================================================
# Market-making block (port of R2.1 ASH helpers, parameterised per product)
# =============================================================================

def mm_block(
    symbol: str,
    depth: OrderDepth,
    position: int,
    fair: float,
    take_width: int,
    clear_width: int,
    disregard_edge: int,
    join_edge: int,
    default_edge: int,
    soft_position: int,
    limit: int,
    out_orders: List[Order],
) -> int:
    """Three-step MM: (1) take aggressive fills under fair, (2) clear stale
    inventory at the touch, (3) post passive bid/ask around fair."""
    buy_vol = 0
    sell_vol = 0
    # ----- (1) take favourable orders -----
    if depth.sell_orders:
        best_ask = min(depth.sell_orders)
        best_ask_amount = -depth.sell_orders[best_ask]
        if best_ask <= fair - take_width:
            qty = min(best_ask_amount, limit - position)
            if qty > 0:
                out_orders.append(Order(symbol, best_ask, qty))
                buy_vol += qty
    if depth.buy_orders:
        best_bid = max(depth.buy_orders)
        best_bid_amount = depth.buy_orders[best_bid]
        if best_bid >= fair + take_width:
            qty = min(best_bid_amount, limit + position)
            if qty > 0:
                out_orders.append(Order(symbol, best_bid, -qty))
                sell_vol += qty
    # ----- (2) clear inventory at fair-side touch -----
    pos_after_take = position + buy_vol - sell_vol
    fair_for_bid = round(fair - clear_width)
    fair_for_ask = round(fair + clear_width)
    if pos_after_take > 0:
        clear_qty = sum(v for p, v in depth.buy_orders.items() if p >= fair_for_ask)
        send = min(min(limit + (position - sell_vol), clear_qty), pos_after_take)
        if send > 0:
            out_orders.append(Order(symbol, fair_for_ask, -send))
            sell_vol += send
    if pos_after_take < 0:
        clear_qty = sum(abs(v) for p, v in depth.sell_orders.items() if p <= fair_for_bid)
        send = min(min(limit - (position + buy_vol), clear_qty), abs(pos_after_take))
        if send > 0:
            out_orders.append(Order(symbol, fair_for_bid, send))
            buy_vol += send
    # ----- (3) passive make -----
    asks_above = [p for p in depth.sell_orders if p > fair + disregard_edge]
    bids_below = [p for p in depth.buy_orders if p < fair - disregard_edge]
    best_ask_above = min(asks_above) if asks_above else None
    best_bid_below = max(bids_below) if bids_below else None
    ask = round(fair + default_edge)
    if best_ask_above is not None:
        ask = best_ask_above if abs(best_ask_above - fair) <= join_edge else best_ask_above - 1
    bid = round(fair - default_edge)
    if best_bid_below is not None:
        bid = best_bid_below if abs(fair - best_bid_below) <= join_edge else best_bid_below + 1
    if position > soft_position:
        ask -= 1
    elif position < -soft_position:
        bid += 1
    buy_q = limit - (position + buy_vol)
    if buy_q > 0:
        out_orders.append(Order(symbol, int(bid), buy_q))
    sell_q = limit + (position - sell_vol)
    if sell_q > 0:
        out_orders.append(Order(symbol, int(ask), -sell_q))
    return position + buy_vol - sell_vol


# =============================================================================
# Smile fit — closed-form parabolic LS over (m_i, iv_i)
# =============================================================================

def fit_parabola(xs: List[float], ys: List[float]) -> Tuple[float, float, float] | None:
    """Return (a, b, c) where iv = a + b*m + c*m^2, or None if singular."""
    n = len(xs)
    if n < 3:
        return None
    sx = sum(xs); sx2 = sum(x * x for x in xs); sx3 = sum(x ** 3 for x in xs); sx4 = sum(x ** 4 for x in xs)
    sy = sum(ys); sxy = sum(x * y for x, y in zip(xs, ys)); sx2y = sum(x * x * y for x, y in zip(xs, ys))
    # Normal eqs:
    # n a + sx b + sx2 c = sy
    # sx a + sx2 b + sx3 c = sxy
    # sx2 a + sx3 b + sx4 c = sx2y
    M = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    rhs = [sy, sxy, sx2y]
    # 3x3 solve (Cramer)
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(M)
    if abs(D) < 1e-9:
        return None
    Ma = [[rhs[i] if c == 0 else M[i][c] for c in range(3)] for i in range(3)]
    Mb = [[rhs[i] if c == 1 else M[i][c] for c in range(3)] for i in range(3)]
    Mc = [[rhs[i] if c == 2 else M[i][c] for c in range(3)] for i in range(3)]
    return (det3(Ma) / D, det3(Mb) / D, det3(Mc) / D)


# =============================================================================
# Welford-style ring buffer for rolling mean/std (kept in traderData)
# =============================================================================

def buffer_push(buf: List[float], x: float, cap: int) -> None:
    buf.append(x)
    if len(buf) > cap:
        del buf[0]


def buffer_mean_std(buf: List[float]) -> Tuple[float, float]:
    n = len(buf)
    if n == 0:
        return 0.0, 0.0
    m = sum(buf) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in buf) / (n - 1)
    return m, math.sqrt(max(var, 1e-12))


# =============================================================================
# Trader
# =============================================================================
class Trader:

    # Settable by backtester before each historical day:
    #   Trader.TTE_INITIAL_DAYS = 8.0  (or 7.0, 6.0, …, 5.0 live)
    TTE_INITIAL_DAYS = TTE_INITIAL_DAYS

    def _init_data(self) -> Dict[str, Any]:
        return {
            "velvet_ema": None,
            "iv_last": {v: None for v in VOUCHERS},
            "resid_buf": {v: [] for v in RESIDUAL_TRADE_VOUCHERS},
            "smile": {"a": None, "b": None, "c": None},
        }

    def _tte_years(self, state: TradingState) -> float:
        days = self.TTE_INITIAL_DAYS - state.timestamp / 1_000_000.0
        return max(days, 0.5 / 24) / DAYS_TO_YEARS  # floor at half an hour

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        try:
            data = json.loads(state.traderData) if state.traderData else self._init_data()
        except Exception:
            data = self._init_data()

        # backfill any missing keys (forward-compatibility with older traderData)
        for k, v in self._init_data().items():
            if k not in data:
                data[k] = v
        for v in RESIDUAL_TRADE_VOUCHERS:
            if v not in data["resid_buf"]:
                data["resid_buf"][v] = []
        for v in VOUCHERS:
            if v not in data["iv_last"]:
                data["iv_last"][v] = None

        orders: Dict[Symbol, List[Order]] = {}
        T = self._tte_years(state)

        # =========================================================
        # Block 1 — HYDROGEL_PACK MM at fair=10000
        # H: OU half-life 300 ticks, fair pinned at 10000 every day.
        # =========================================================
        if HYDROGEL in state.order_depths:
            depth = state.order_depths[HYDROGEL]
            pos = state.position.get(HYDROGEL, 0)
            hydra_orders: List[Order] = []
            mm_block(
                HYDROGEL, depth, pos, HYDRO_FAIR,
                HYDRO_TAKE_WIDTH, HYDRO_CLEAR_WIDTH,
                HYDRO_DISREGARD_EDGE, HYDRO_JOIN_EDGE, HYDRO_DEFAULT_EDGE,
                HYDRO_SOFT_POSITION, LIMITS[HYDROGEL], hydra_orders,
            )
            if hydra_orders:
                orders[HYDROGEL] = hydra_orders

        # =========================================================
        # Block 2 — VELVETFRUIT_EXTRACT MM with EMA fair
        # H: OU around ~5250, slow drift (5246→5255 across 3 days).
        # =========================================================
        velvet_mid = None
        if VELVET in state.order_depths:
            depth = state.order_depths[VELVET]
            mid = safe_mid(depth)
            if mid is not None:
                if data["velvet_ema"] is None:
                    data["velvet_ema"] = mid
                else:
                    data["velvet_ema"] = (1 - VELVET_EMA_ALPHA) * data["velvet_ema"] + VELVET_EMA_ALPHA * mid
                velvet_mid = mid
                fair = data["velvet_ema"]
                pos = state.position.get(VELVET, 0)
                v_orders: List[Order] = []
                mm_block(
                    VELVET, depth, pos, fair,
                    VELVET_TAKE_WIDTH, VELVET_CLEAR_WIDTH,
                    VELVET_DISREGARD_EDGE, VELVET_JOIN_EDGE, VELVET_DEFAULT_EDGE,
                    VELVET_SOFT_POSITION, LIMITS[VELVET], v_orders,
                )
                if v_orders:
                    orders[VELVET] = v_orders

        # =========================================================
        # Block 3 — Voucher IV-residual mean reversion
        # H: 9/10 vouchers have sign-stable residual every day,
        # ACF(1) 0.4–0.9 → mispricings persist.
        # =========================================================
        if velvet_mid is None:
            # Without underlying mid we can't compute IV — skip voucher block.
            return self._finalise(orders, data, state)
        S = velvet_mid

        # Solve IV for each voucher we have a book on.
        iv_now: Dict[str, float] = {}
        money_now: Dict[str, float] = {}
        for v in VOUCHERS:
            if v not in state.order_depths:
                continue
            mid_v = safe_mid(state.order_depths[v])
            if mid_v is None:
                continue
            K = STRIKE[v]
            iv = implied_vol(mid_v, S, K, T)
            if iv is None and data["iv_last"].get(v) is not None:
                # Fallback: use last-good IV (only for delta hedging,
                # not for residual computation since price < intrinsic
                # means the voucher is non-BS-pricable right now).
                pass
            else:
                if iv is not None:
                    iv_now[v] = iv
                    data["iv_last"][v] = iv
                    if T > 0:
                        money_now[v] = math.log(K / S) / math.sqrt(T)

        # Fit smile across SMILE_FIT_VOUCHERS that have a current IV.
        xs: List[float] = []
        ys: List[float] = []
        for v in SMILE_FIT_VOUCHERS:
            if v in iv_now and v in money_now:
                xs.append(money_now[v])
                ys.append(iv_now[v])
        coefs = fit_parabola(xs, ys) if len(xs) >= 4 else None
        if coefs is not None:
            data["smile"]["a"], data["smile"]["b"], data["smile"]["c"] = coefs

        # Compute residual & target position for each tradable voucher.
        a, b, c = data["smile"]["a"], data["smile"]["b"], data["smile"]["c"]
        targets: Dict[str, int] = {}
        if a is not None:
            for v in RESIDUAL_TRADE_VOUCHERS:
                if v not in iv_now or v not in money_now:
                    continue
                m = money_now[v]
                fit_iv = a + b * m + c * m * m
                resid = iv_now[v] - fit_iv
                buf = data["resid_buf"][v]
                buffer_push(buf, resid, RESID_BUFFER_SIZE)
                if len(buf) < RESID_MIN_FOR_TRADE:
                    continue
                mu, sd = buffer_mean_std(buf)
                if sd <= 0:
                    continue
                z = (resid - mu) / sd
                # Trade in the reversion direction:
                # residual > 0 → IV high → call overpriced → SHORT (negative position)
                # residual < 0 → IV low → call underpriced → LONG (positive position)
                if abs(z) < RESID_Z_THRESHOLD:
                    targets[v] = 0
                    continue
                # Linear ramp from 0 at z_threshold to full at z_full
                scale = (abs(z) - RESID_Z_THRESHOLD) / max(RESID_Z_FULL_SIZE - RESID_Z_THRESHOLD, 1e-9)
                scale = max(0.0, min(1.0, scale))
                conviction = RESID_CONVICTION.get(v, 1.0)
                cap = int(LIMITS[v] * RESID_VOUCHER_CAP_FRACTION)
                size = int(scale * conviction * cap)
                size = max(0, min(cap, size))
                target_pos = -size if z > 0 else size
                targets[v] = target_pos

        # Convert targets → orders (ride the touch; passive at touch on opposite side).
        for v, target in targets.items():
            if v not in state.order_depths:
                continue
            depth = state.order_depths[v]
            pos = state.position.get(v, 0)
            delta = target - pos
            if delta == 0:
                continue
            v_orders = orders.setdefault(v, [])
            if delta > 0:
                # Need to buy. Take the ask up to delta, then post a passive bid.
                if depth.sell_orders:
                    best_ask = min(depth.sell_orders)
                    avail = -depth.sell_orders[best_ask]
                    take = min(delta, avail, LIMITS[v] - pos)
                    if take > 0:
                        v_orders.append(Order(v, best_ask, take))
                        delta -= take; pos += take
                if delta > 0 and depth.buy_orders:
                    best_bid = max(depth.buy_orders)
                    join = min(delta, LIMITS[v] - pos)
                    if join > 0:
                        v_orders.append(Order(v, best_bid, join))
            else:
                need = -delta  # positive number to sell
                if depth.buy_orders:
                    best_bid = max(depth.buy_orders)
                    avail = depth.buy_orders[best_bid]
                    take = min(need, avail, LIMITS[v] + pos)
                    if take > 0:
                        v_orders.append(Order(v, best_bid, -take))
                        need -= take; pos -= take
                if need > 0 and depth.sell_orders:
                    best_ask = min(depth.sell_orders)
                    join = min(need, LIMITS[v] + pos)
                    if join > 0:
                        v_orders.append(Order(v, best_ask, -join))

        # =========================================================
        # Block 4 — Free-call lottery (VEV_6000 / VEV_6500)
        # H: bots dump worthless options at price 0; absorbing them
        # has zero cost and tiny positive expected payoff.
        # =========================================================
        for v in LOTTERY_VOUCHERS:
            if v not in state.order_depths:
                continue
            depth = state.order_depths[v]
            pos = state.position.get(v, 0)
            current_targets_pos = targets.get(v)
            # Don't double-trade if residual block already wants this voucher.
            if current_targets_pos is not None and current_targets_pos != 0:
                continue
            # If a seller exists at 0 or 1, take it; else place a passive bid at 0.
            free_orders = orders.setdefault(v, [])
            if depth.sell_orders:
                best_ask = min(depth.sell_orders)
                if best_ask <= 1:
                    avail = -depth.sell_orders[best_ask]
                    headroom = LOTTERY_TARGET_QTY - pos
                    take = min(avail, max(headroom, 0))
                    if take > 0:
                        free_orders.append(Order(v, best_ask, take))
                        pos += take
            if pos < LOTTERY_TARGET_QTY:
                # Already-existing orders for this voucher might include higher-priced bids.
                free_orders.append(Order(v, LOTTERY_BID_PRICE, LOTTERY_TARGET_QTY - pos))

        # =========================================================
        # Block 5 — Delta governor
        # Hedge net portfolio delta (in underlying-equivalent units)
        # in VELVETFRUIT_EXTRACT.
        # =========================================================
        # Build *projected* positions = current position + sum of orders we just placed.
        projected_pos: Dict[str, int] = dict(state.position)
        for sym, ord_list in orders.items():
            for o in ord_list:
                projected_pos[sym] = projected_pos.get(sym, 0) + o.quantity

        # Net delta:
        #   - VELVETFRUIT_EXTRACT: pos*1
        #   - VEV_4000, VEV_4500: pos*1 (forwards in disguise — full delta)
        #   - VEV_5000..6500: pos*BS_delta (using current iv or fallback)
        net_delta = projected_pos.get(VELVET, 0)
        net_delta += projected_pos.get("VEV_4000", 0)
        net_delta += projected_pos.get("VEV_4500", 0)
        for v in SMILE_FIT_VOUCHERS:
            p = projected_pos.get(v, 0)
            if p == 0:
                continue
            iv_v = iv_now.get(v) or data["iv_last"].get(v)
            if iv_v is None:
                continue
            d = bs_delta(S, STRIKE[v], T, iv_v)
            net_delta += int(round(p * d))

        if abs(net_delta) > DELTA_CAP and HEDGE_PRODUCT in state.order_depths:
            depth = state.order_depths[HEDGE_PRODUCT]
            pos = projected_pos.get(HEDGE_PRODUCT, 0)
            limit = LIMITS[HEDGE_PRODUCT]
            v_orders = orders.setdefault(HEDGE_PRODUCT, [])
            if net_delta > DELTA_CAP:
                # Long delta → sell underlying.
                need = min(net_delta - DELTA_CAP, limit + pos)
                if need > 0 and depth.buy_orders:
                    best_bid = max(depth.buy_orders)
                    avail = depth.buy_orders[best_bid]
                    qty = min(need, avail)
                    if qty > 0:
                        v_orders.append(Order(HEDGE_PRODUCT, best_bid, -qty))
            elif net_delta < -DELTA_CAP:
                need = min(-net_delta - DELTA_CAP, limit - pos)
                if need > 0 and depth.sell_orders:
                    best_ask = min(depth.sell_orders)
                    avail = -depth.sell_orders[best_ask]
                    qty = min(need, avail)
                    if qty > 0:
                        v_orders.append(Order(HEDGE_PRODUCT, best_ask, qty))

        return self._finalise(orders, data, state)

    def _finalise(self, orders: Dict[Symbol, List[Order]], data: Dict[str, Any],
                  state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        # Drop empty product entries.
        orders = {sym: olist for sym, olist in orders.items() if olist}
        td = json.dumps(data, separators=(",", ":"))
        # Cap traderData size — drop oldest residual buffer entries first.
        if len(td) > 40000:
            for v in data["resid_buf"]:
                if len(data["resid_buf"][v]) > 50:
                    data["resid_buf"][v] = data["resid_buf"][v][-50:]
            td = json.dumps(data, separators=(",", ":"))
        return orders, 0, td
