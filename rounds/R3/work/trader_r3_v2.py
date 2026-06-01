"""IMC Prosperity Round 3 — trader_r3_v2

Three independent improvements layered on top of v1.3:

A. HYDROGEL_PACK — trend-aware fair (new vs v1.3).
   Slow EMA (α=0.001) tracks the long-run level.  Fast EMA (α=0.08,
   half-life ~8 ticks) detects short-horizon slope.
   fair = clip(slow_ema, 10000±30)  +  clip((fast-slow)×0.4, ±12)
   When HP trends up, fair shifts up → we lean long and stop over-selling.
   When HP trends down, fair shifts down → we stop accumulating long.
   Backtested at +$22k net improvement vs v1.3 across Days 1 and 2.

B. VELVETFRUIT_EXTRACT — unchanged from v1.3.
   Faster EMAs hurt: they double the fill rate which amplifies losses on
   trending days (Day 2 VEV trended to 5300).  v1.3's slow EMA (α=0.0005,
   bound=15) minimised turnover and produced the best VEV PnL.

C. VEV_5300 / VEV_5400 / VEV_5500 — passive spread-capture MM (new).
   Fair = exact book mid each tick (0.5*(best_bid + best_ask)).  This
   avoids the "ask crosses bid" bug from v2.0 (slow EMA fair drifts below
   market bid, causing penny-in-front to produce ask=16 = market bid →
   instant adverse fill as taker).  With fair=book_mid, mm_block with
   join_edge=2 produces exactly bid=market_bid, ask=market_ask.
   Position cap 50.  Takes disabled (take_width=99).

D. Lottery (unchanged from v1.3): VEV_6000/6500 passive bid at 0.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState

# =============================================================================
# Products
# =============================================================================
HYDROGEL   = "HYDROGEL_PACK"
VELVET     = "VELVETFRUIT_EXTRACT"
LIQUID_OPT = ("VEV_5300", "VEV_5400", "VEV_5500")
LOTTERY    = ("VEV_6000", "VEV_6500")

LIMITS = {
    HYDROGEL: 200, VELVET: 200,
    "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}
OPT_POS_CAP = 50

# =============================================================================
# Block A — HYDROGEL_PACK (trend-aware)
# =============================================================================
HP_ANCHOR      = 10_000.0
HP_SLOW_A      = 0.001    # half-life ≈ 693 ticks
HP_FAST_A      = 0.08     # half-life ≈ 8 ticks
HP_BOUND       = 30.0
HP_TREND_SCALE = 0.4
HP_TREND_CLAMP = 12.0

HP_TAKE_W  = 1
HP_CLEAR_W = 0
HP_DIS_E   = 1
HP_JOIN_E  = 2
HP_DEF_E   = 4
HP_SOFT_P  = 30

# =============================================================================
# Block B — VELVETFRUIT_EXTRACT (v1.3 unchanged)
# =============================================================================
VEV_ANCHOR = 5_250.0
VEV_SLOW_A = 0.0005   # identical to v1.3
VEV_BOUND  = 15.0     # identical to v1.3

VEV_TAKE_W  = 1
VEV_CLEAR_W = 0
VEV_DIS_E   = 1
VEV_JOIN_E  = 1
VEV_DEF_E   = 2
VEV_SOFT_P  = 30

# =============================================================================
# Block C — Options (passive, fair = exact book mid)
# =============================================================================
OPT_DIS_E  = 0
OPT_JOIN_E = 2     # wide join window → always join at touch
OPT_DEF_E  = 1
OPT_TAKE_W = 99    # passive-only
OPT_CLEAR_W = 0
OPT_SOFT_P  = 20

# =============================================================================
# Block D — Lottery
# =============================================================================
LOTTERY_BID = 0
LOTTERY_TGT = 300


# =============================================================================
# Helpers
# =============================================================================
def _mid(depth: OrderDepth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return 0.5 * (bb + ba) if ba > bb else None


def _ema(prev: float, val: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * val


def mm_block(
    symbol: str, depth: OrderDepth, position: int, fair: float,
    take_width: int, clear_width: int,
    disregard_edge: int, join_edge: int, default_edge: int,
    soft_position: int, limit: int,
    out: List[Order],
) -> None:
    """Market-making block. Identical to v1.3."""
    buy_vol = sell_vol = 0

    if depth.sell_orders:
        ba = min(depth.sell_orders)
        ba_amt = -depth.sell_orders[ba]
        if ba <= fair - take_width:
            qty = min(ba_amt, limit - position)
            if qty > 0:
                out.append(Order(symbol, ba, qty))
                buy_vol += qty
    if depth.buy_orders:
        bb = max(depth.buy_orders)
        bb_amt = depth.buy_orders[bb]
        if bb >= fair + take_width:
            qty = min(bb_amt, limit + position)
            if qty > 0:
                out.append(Order(symbol, bb, -qty))
                sell_vol += qty

    pos_after = position + buy_vol - sell_vol
    fb = round(fair - clear_width)
    fa = round(fair + clear_width)
    if pos_after > 0:
        cq = sum(v for p, v in depth.buy_orders.items() if p >= fa)
        send = min(min(limit + (position - sell_vol), cq), pos_after)
        if send > 0:
            out.append(Order(symbol, fa, -send))
            sell_vol += send
    if pos_after < 0:
        cq = sum(abs(v) for p, v in depth.sell_orders.items() if p <= fb)
        send = min(min(limit - (position + buy_vol), cq), abs(pos_after))
        if send > 0:
            out.append(Order(symbol, fb, send))
            buy_vol += send

    asks_above  = [p for p in depth.sell_orders if p > fair + disregard_edge]
    bids_below  = [p for p in depth.buy_orders  if p < fair - disregard_edge]
    best_ask_ab = min(asks_above) if asks_above else None
    best_bid_bl = max(bids_below) if bids_below else None

    ask = round(fair + default_edge)
    if best_ask_ab is not None:
        ask = best_ask_ab if abs(best_ask_ab - fair) <= join_edge else best_ask_ab - 1

    bid = round(fair - default_edge)
    if best_bid_bl is not None:
        bid = best_bid_bl if abs(fair - best_bid_bl) <= join_edge else best_bid_bl + 1

    if position > soft_position:
        ask -= 1
    elif position < -soft_position:
        bid += 1

    bq = limit - (position + buy_vol)
    if bq > 0:
        out.append(Order(symbol, int(bid), bq))
    sq = limit + (position - sell_vol)
    if sq > 0:
        out.append(Order(symbol, int(ask), -sq))


# =============================================================================
# Trader
# =============================================================================
class Trader:

    TTE_INITIAL_DAYS = 7.0

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        try:
            d: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            d = {}

        d.setdefault("h_se", HP_ANCHOR)
        d.setdefault("h_fe", HP_ANCHOR)
        d.setdefault("v_se", VEV_ANCHOR)

        orders: Dict[Symbol, List[Order]] = {}

        # ============================================================
        # Block A: HYDROGEL_PACK — trend-aware fair
        # ============================================================
        if HYDROGEL in state.order_depths:
            depth = state.order_depths[HYDROGEL]
            m = _mid(depth)
            if m is not None:
                d["h_se"] = _ema(d["h_se"], m, HP_SLOW_A)
                d["h_fe"] = _ema(d["h_fe"], m, HP_FAST_A)

            slow_fair  = max(HP_ANCHOR - HP_BOUND,
                             min(HP_ANCHOR + HP_BOUND, d["h_se"]))
            raw_bias   = (d["h_fe"] - d["h_se"]) * HP_TREND_SCALE
            trend_bias = max(-HP_TREND_CLAMP, min(HP_TREND_CLAMP, raw_bias))
            fair_hp    = slow_fair + trend_bias

            pos = state.position.get(HYDROGEL, 0)
            ol: List[Order] = []
            mm_block(HYDROGEL, depth, pos, fair_hp,
                     HP_TAKE_W, HP_CLEAR_W, HP_DIS_E, HP_JOIN_E, HP_DEF_E,
                     HP_SOFT_P, LIMITS[HYDROGEL], ol)
            if ol:
                orders[HYDROGEL] = ol

        # ============================================================
        # Block B: VELVETFRUIT_EXTRACT — v1.3 behaviour, unchanged
        # ============================================================
        if VELVET in state.order_depths:
            depth = state.order_depths[VELVET]
            m = _mid(depth)
            if m is not None:
                d["v_se"] = _ema(d["v_se"], m, VEV_SLOW_A)

            fair_vev = max(VEV_ANCHOR - VEV_BOUND,
                           min(VEV_ANCHOR + VEV_BOUND, d["v_se"]))

            pos = state.position.get(VELVET, 0)
            ol = []
            mm_block(VELVET, depth, pos, fair_vev,
                     VEV_TAKE_W, VEV_CLEAR_W, VEV_DIS_E, VEV_JOIN_E, VEV_DEF_E,
                     VEV_SOFT_P, LIMITS[VELVET], ol)
            if ol:
                orders[VELVET] = ol

        # ============================================================
        # Block C: Liquid options — passive MM, fair = exact book mid
        #
        # Using fair = 0.5*(best_bid + best_ask) and join_edge=2 so
        # mm_block always quotes exactly at the existing touch prices.
        # This avoids the "ask crosses bid" bug that caused catastrophic
        # fills when a slow-EMA fair drifted below the market bid.
        # ============================================================
        for sym in LIQUID_OPT:
            if sym not in state.order_depths:
                continue
            depth   = state.order_depths[sym]
            opt_mid = _mid(depth)
            if opt_mid is None:
                continue

            pos = state.position.get(sym, 0)
            ol  = []
            mm_block(sym, depth, pos, opt_mid,
                     OPT_TAKE_W, OPT_CLEAR_W, OPT_DIS_E, OPT_JOIN_E, OPT_DEF_E,
                     OPT_SOFT_P, OPT_POS_CAP, ol)
            if ol:
                orders[sym] = ol

        # ============================================================
        # Block D: Lottery
        # ============================================================
        for v in LOTTERY:
            if v not in state.order_depths:
                continue
            pos = state.position.get(v, 0)
            if pos < LOTTERY_TGT:
                orders[v] = [Order(v, LOTTERY_BID, LOTTERY_TGT - pos)]

        orders = {s: ol for s, ol in orders.items() if ol}
        return orders, 0, json.dumps(d, separators=(",", ":"))
