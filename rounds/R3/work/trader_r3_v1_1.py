"""IMC Prosperity Round 3 — trader_r3_v1_1 (minimal, post-diagnosis).

After v1's backtest revealed that the IV-residual edge (~0.1 ticks/contract) is
strictly smaller than half-spread on every voucher (>=0.5 ticks), the residual
block was a guaranteed money-loser via spread crossing. Stripped down to:

  1. HYDROGEL_PACK market-make at fair=10000 (worked: +28k/day average in v1).
  2. VELVETFRUIT_EXTRACT market-make at fair=5250 (constant; fix EMA lag).
  3. Free-call lottery: BUY-ONLY VEV_6000 and VEV_6500 at price 0.
     Bots dump worthless options at 0; we accumulate at zero cost. Never sell
     (selling at 0 with mid=0.5 is half-spread leakage, exactly the v1 bug).

Explicitly NOT trading:
  - VEV_4000, VEV_4500: at intrinsic, no edge.
  - VEV_5000, VEV_5100: no bot flow (1 trade/3 days).
  - VEV_5200..5500: residual edge < spread, no passive-fill story works.
  - Delta hedge: voucher exposure is small if we're not active there.

If this baseline lands cleanly positive on all 3 days, layer voucher passive
quotes back in v1.2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState

# =============================================================================
HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
LOTTERY_VOUCHERS = ("VEV_6000", "VEV_6500")

LIMITS = {HYDROGEL: 200, VELVET: 200,
          "VEV_6000": 300, "VEV_6500": 300}

# Fair values are pinned (Phase 1 OU μ estimates).
HYDRO_FAIR = 10_000
VELVET_FAIR = 5_250

# MM params (R1 ASH-style)
HYDRO_TAKE_WIDTH = 1
HYDRO_CLEAR_WIDTH = 0
HYDRO_DISREGARD_EDGE = 1
HYDRO_JOIN_EDGE = 2
HYDRO_DEFAULT_EDGE = 4
HYDRO_SOFT_POSITION = 30

VELVET_TAKE_WIDTH = 1
VELVET_CLEAR_WIDTH = 0
VELVET_DISREGARD_EDGE = 1
VELVET_JOIN_EDGE = 1
VELVET_DEFAULT_EDGE = 2
VELVET_SOFT_POSITION = 30

# Lottery: buy at price 0, never sell. Cap small to limit MTM noise / risk.
LOTTERY_BID_PRICE = 0
LOTTERY_TARGET = 300


def mm_block(
    symbol: str, depth: OrderDepth, position: int, fair: float,
    take_width: int, clear_width: int,
    disregard_edge: int, join_edge: int, default_edge: int,
    soft_position: int, limit: int,
    out_orders: List[Order],
) -> None:
    buy_vol = 0; sell_vol = 0
    if depth.sell_orders:
        best_ask = min(depth.sell_orders)
        best_ask_amount = -depth.sell_orders[best_ask]
        if best_ask <= fair - take_width:
            qty = min(best_ask_amount, limit - position)
            if qty > 0:
                out_orders.append(Order(symbol, best_ask, qty)); buy_vol += qty
    if depth.buy_orders:
        best_bid = max(depth.buy_orders)
        best_bid_amount = depth.buy_orders[best_bid]
        if best_bid >= fair + take_width:
            qty = min(best_bid_amount, limit + position)
            if qty > 0:
                out_orders.append(Order(symbol, best_bid, -qty)); sell_vol += qty
    pos_after = position + buy_vol - sell_vol
    fair_for_bid = round(fair - clear_width)
    fair_for_ask = round(fair + clear_width)
    if pos_after > 0:
        clear_qty = sum(v for p, v in depth.buy_orders.items() if p >= fair_for_ask)
        send = min(min(limit + (position - sell_vol), clear_qty), pos_after)
        if send > 0:
            out_orders.append(Order(symbol, fair_for_ask, -send)); sell_vol += send
    if pos_after < 0:
        clear_qty = sum(abs(v) for p, v in depth.sell_orders.items() if p <= fair_for_bid)
        send = min(min(limit - (position + buy_vol), clear_qty), abs(pos_after))
        if send > 0:
            out_orders.append(Order(symbol, fair_for_bid, send)); buy_vol += send
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


class Trader:

    # Kept for backtest compatibility; unused by v1.1 since no IV math here.
    TTE_INITIAL_DAYS = 5.0

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        orders: Dict[Symbol, List[Order]] = {}

        # ---- HYDROGEL_PACK MM ----
        if HYDROGEL in state.order_depths:
            depth = state.order_depths[HYDROGEL]
            pos = state.position.get(HYDROGEL, 0)
            ol: List[Order] = []
            mm_block(HYDROGEL, depth, pos, HYDRO_FAIR,
                     HYDRO_TAKE_WIDTH, HYDRO_CLEAR_WIDTH,
                     HYDRO_DISREGARD_EDGE, HYDRO_JOIN_EDGE, HYDRO_DEFAULT_EDGE,
                     HYDRO_SOFT_POSITION, LIMITS[HYDROGEL], ol)
            if ol: orders[HYDROGEL] = ol

        # ---- VELVETFRUIT_EXTRACT MM ----
        if VELVET in state.order_depths:
            depth = state.order_depths[VELVET]
            pos = state.position.get(VELVET, 0)
            ol: List[Order] = []
            mm_block(VELVET, depth, pos, VELVET_FAIR,
                     VELVET_TAKE_WIDTH, VELVET_CLEAR_WIDTH,
                     VELVET_DISREGARD_EDGE, VELVET_JOIN_EDGE, VELVET_DEFAULT_EDGE,
                     VELVET_SOFT_POSITION, LIMITS[VELVET], ol)
            if ol: orders[VELVET] = ol

        # ---- Lottery (BUY-ONLY, strictly passive at price 0) ----
        # Bots dump VEV_6000/6500 at price 0 (Phase 1.6: 100% of fills at the bid
        # with mid=0.5). Posting a passive bid at 0 captures their flow at zero
        # cost. Crossing the spread to take at price 1 would cost half-spread per
        # contract — which was the v1.1 v0 bleed (~$150/voucher/day).
        for v in LOTTERY_VOUCHERS:
            if v not in state.order_depths:
                continue
            pos = state.position.get(v, 0)
            if pos < LOTTERY_TARGET:
                orders[v] = [Order(v, LOTTERY_BID_PRICE, LOTTERY_TARGET - pos)]

        orders = {sym: ol for sym, ol in orders.items() if ol}
        return orders, 0, json.dumps(data, separators=(",", ":"))
