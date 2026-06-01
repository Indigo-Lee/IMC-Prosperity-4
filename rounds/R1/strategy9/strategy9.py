"""
Strategy 9 (Round 1) — hybrid: John's adaptive ASH MM + PEPPER drift MM
(offset=6, floor=43).

John's ASH MM (from origin/John branch) captures ~2.6× the spread of the
fixed-quote 9,999 / 10,001 scheme via adaptive join/undercut quoting with
inventory skew.  Strategy 8's PEPPER MM (floor=43, mm_band=7, offset=6) was
independently shown to be the best PEPPER configuration.

ASH edge:    ~+83,295   (John's approach vs strategy8's ~+31,987)
PEPPER edge: ~+164,186  (strategy8 approach)
Expected combined: ~+247,000

IMC submission format:
  - Single file, no local imports beyond `datamodel`.
  - `Trader.run` returns (orders_dict, conversions_int, traderData_str).
  - No Logger / ProsperityEncoder dependency (not in local datamodel.py).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState


# ============================================================================
# Component 1: ASH_COATED_OSMIUM — John's adaptive 4-step market making
# ============================================================================

ASH_PRODUCT        = "ASH_COATED_OSMIUM"
ASH_FAIR           = 10_000
ASH_LIMIT          = 50
ASH_TAKE_WIDTH     = 1
ASH_CLEAR_WIDTH    = 0
ASH_DISREGARD_EDGE = 1
ASH_JOIN_EDGE      = 2
ASH_DEFAULT_EDGE   = 4
ASH_SOFT_POS_LIMIT = 10


class _AshTrader:
    """John's adaptive 4-step MM:
      1. Take aggressively when book crosses fair.
      2. Clear excess inventory at fair.
      3. Compute adaptive bid/ask by joining/undercutting the book + skew.
      4. Place passive quotes for remaining headroom.
    """

    # ------------------------------------------------------------------
    # Step 1 — aggressive takes
    # ------------------------------------------------------------------
    def _take_best_orders(
        self,
        orders: List[Order],
        depth: OrderDepth,
        pos: int,
        buy_vol: int,
        sell_vol: int,
    ) -> Tuple[int, int]:
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            best_ask_amt = -depth.sell_orders[best_ask]
            if best_ask <= ASH_FAIR - ASH_TAKE_WIDTH:
                qty = min(best_ask_amt, ASH_LIMIT - pos)
                if qty > 0:
                    orders.append(Order(ASH_PRODUCT, best_ask, qty))
                    buy_vol += qty
                    depth.sell_orders[best_ask] += qty
                    if depth.sell_orders[best_ask] == 0:
                        del depth.sell_orders[best_ask]

        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            best_bid_amt = depth.buy_orders[best_bid]
            if best_bid >= ASH_FAIR + ASH_TAKE_WIDTH:
                qty = min(best_bid_amt, ASH_LIMIT + pos)
                if qty > 0:
                    orders.append(Order(ASH_PRODUCT, best_bid, -qty))
                    sell_vol += qty
                    depth.buy_orders[best_bid] -= qty
                    if depth.buy_orders[best_bid] == 0:
                        del depth.buy_orders[best_bid]

        return buy_vol, sell_vol

    # ------------------------------------------------------------------
    # Step 2 — clear position at fair
    # ------------------------------------------------------------------
    def _clear_position_order(
        self,
        orders: List[Order],
        depth: OrderDepth,
        pos: int,
        buy_vol: int,
        sell_vol: int,
    ) -> Tuple[int, int]:
        pos_after = pos + buy_vol - sell_vol
        fair_bid = round(ASH_FAIR - ASH_CLEAR_WIDTH)
        fair_ask = round(ASH_FAIR + ASH_CLEAR_WIDTH)

        buy_capacity  = ASH_LIMIT - (pos + buy_vol)
        sell_capacity = ASH_LIMIT + (pos - sell_vol)

        if pos_after > 0:
            clear_qty = sum(
                v for p, v in depth.buy_orders.items() if p >= fair_ask
            )
            clear_qty = min(clear_qty, pos_after)
            send_qty  = min(sell_capacity, clear_qty)
            if send_qty > 0:
                orders.append(Order(ASH_PRODUCT, fair_ask, -abs(send_qty)))
                sell_vol += abs(send_qty)

        if pos_after < 0:
            clear_qty = sum(
                abs(v) for p, v in depth.sell_orders.items() if p <= fair_bid
            )
            clear_qty = min(clear_qty, abs(pos_after))
            send_qty  = min(buy_capacity, clear_qty)
            if send_qty > 0:
                orders.append(Order(ASH_PRODUCT, fair_bid, abs(send_qty)))
                buy_vol += abs(send_qty)

        return buy_vol, sell_vol

    # ------------------------------------------------------------------
    # Step 3+4 — compute adaptive quotes and place passive MM
    # ------------------------------------------------------------------
    def _make_orders(
        self,
        orders: List[Order],
        depth: OrderDepth,
        pos: int,
        buy_vol: int,
        sell_vol: int,
    ) -> None:
        asks_above = [p for p in depth.sell_orders if p > ASH_FAIR + ASH_DISREGARD_EDGE]
        bids_below = [p for p in depth.buy_orders  if p < ASH_FAIR - ASH_DISREGARD_EDGE]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        # Default fall-back spread
        ask = round(ASH_FAIR + ASH_DEFAULT_EDGE)
        if best_ask_above is not None:
            if abs(best_ask_above - ASH_FAIR) <= ASH_JOIN_EDGE:
                ask = best_ask_above       # join
            else:
                ask = best_ask_above - 1   # undercut

        bid = round(ASH_FAIR - ASH_DEFAULT_EDGE)
        if best_bid_below is not None:
            if abs(ASH_FAIR - best_bid_below) <= ASH_JOIN_EDGE:
                bid = best_bid_below       # join
            else:
                bid = best_bid_below + 1   # undercut

        # Inventory skew: aggressively work down extreme longs/shorts
        if pos > ASH_SOFT_POS_LIMIT:
            ask -= 1
        elif pos < -ASH_SOFT_POS_LIMIT:
            bid += 1

        # Place passive quotes for remaining capacity
        buy_qty  = ASH_LIMIT - (pos + buy_vol)
        sell_qty = ASH_LIMIT + (pos - sell_vol)

        if buy_qty > 0:
            orders.append(Order(ASH_PRODUCT, round(bid), buy_qty))
        if sell_qty > 0:
            orders.append(Order(ASH_PRODUCT, round(ask), -sell_qty))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, state: TradingState) -> List[Order]:
        if ASH_PRODUCT not in state.order_depths:
            return []

        # Work on a shallow copy of the depth dicts so mutations below
        # (removing taken levels) don't corrupt the original state.
        raw   = state.order_depths[ASH_PRODUCT]
        depth = OrderDepth()
        depth.buy_orders  = dict(raw.buy_orders)
        depth.sell_orders = dict(raw.sell_orders)

        pos    = state.position.get(ASH_PRODUCT, 0)
        orders: List[Order] = []
        buy_vol, sell_vol = 0, 0

        buy_vol, sell_vol = self._take_best_orders(orders, depth, pos, buy_vol, sell_vol)
        buy_vol, sell_vol = self._clear_position_order(orders, depth, pos, buy_vol, sell_vol)
        self._make_orders(orders, depth, pos, buy_vol, sell_vol)

        return orders


# ============================================================================
# Component 2: INTARIAN_PEPPER_ROOT long-biased linear-drift MM
#              (Strategy 8 — floor=43, offset=6, mm_band=7)
# ============================================================================

PEPPER_PRODUCT        = "INTARIAN_PEPPER_ROOT"
PEPPER_POSITION_LIMIT = 50
PEPPER_PASSIVE_OFFSET = 6          # shells away from fair for passive bid/ask

# Long-bias parameters (optimised via strategy8/floor_sweep.py)
PEPPER_TARGET_POS = PEPPER_POSITION_LIMIT        # 50 — always aim fully long
PEPPER_MM_BAND    = 7                            # rotate only the top 7 units
PEPPER_FLOOR      = PEPPER_TARGET_POS - PEPPER_MM_BAND  # 43

# Timing landmarks (continuous clock; 10,000 ticks per day × 3 days)
PEPPER_ENTRY_END  = 100
PEPPER_EXIT_START = 29_900

# Linear fair-value line (fitted offline — residual std ≈ 2.2)
PEPPER_INITIAL_FAIR   = 10_000.0
PEPPER_SLOPE_PER_TICK = 0.1


def _pepper_fair_value(tick: int) -> float:
    return PEPPER_INITIAL_FAIR + PEPPER_SLOPE_PER_TICK * tick


class _PepperTrader:
    """Three phases:
      Entry  (tick < 100):          aggressive ramp until pos >= FLOOR
      Middle (100 <= tick < 29900): passive bid at fair-6 refills to TARGET,
                                    passive ask at fair+6 rotates top MM_BAND slice
      Exit   (tick >= 29900):       aggressive flatten
    """

    def run(self, state: TradingState, tick: int) -> List[Order]:
        if PEPPER_PRODUCT not in state.order_depths:
            return []

        depth = state.order_depths[PEPPER_PRODUCT]
        pos   = state.position.get(PEPPER_PRODUCT, 0)
        sym_orders: List[Order] = []

        fair = _pepper_fair_value(tick)
        pb   = int(round(fair)) - PEPPER_PASSIVE_OFFSET
        pa   = int(round(fair)) + PEPPER_PASSIVE_OFFSET

        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None

        # -------------------------- EXIT PHASE --------------------------
        if tick >= PEPPER_EXIT_START:
            if pos > 0:
                remaining = pos
                for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                    avail = depth.buy_orders[bid_price]
                    take = min(avail, remaining)
                    if take <= 0:
                        break
                    sym_orders.append(Order(PEPPER_PRODUCT, bid_price, -take))
                    remaining -= take
                    if remaining <= 0:
                        break
            return sym_orders

        # ---------------------- AGGRESSIVE TAKES ------------------------
        if best_ask is not None and best_ask <= pb:
            cap = PEPPER_POSITION_LIMIT - pos
            if cap > 0:
                qty = min(abs(depth.sell_orders[best_ask]), cap)
                if qty > 0:
                    sym_orders.append(Order(PEPPER_PRODUCT, best_ask, qty))
                    pos += qty

        if best_bid is not None and best_bid >= pa:
            cap = max(0, pos - PEPPER_FLOOR)
            if cap > 0:
                qty = min(depth.buy_orders[best_bid], cap)
                if qty > 0:
                    sym_orders.append(Order(PEPPER_PRODUCT, best_bid, -qty))
                    pos -= qty

        # -------------------- ENTRY PHASE (RAMP) ------------------------
        if pos < PEPPER_FLOOR and best_ask is not None:
            needed = PEPPER_FLOOR - pos
            cap    = PEPPER_POSITION_LIMIT - pos
            qty    = min(abs(depth.sell_orders[best_ask]), needed, cap)
            if qty > 0:
                sym_orders.append(Order(PEPPER_PRODUCT, best_ask, qty))
                pos += qty

        # ------------------------- PASSIVE MM ---------------------------
        bid_size = max(0, PEPPER_TARGET_POS - pos)
        bid_size = min(bid_size, PEPPER_POSITION_LIMIT - pos)
        if bid_size > 0:
            sym_orders.append(Order(PEPPER_PRODUCT, pb, bid_size))

        ask_size = max(0, pos - PEPPER_FLOOR)
        ask_size = min(ask_size, PEPPER_POSITION_LIMIT + pos)
        if ask_size > 0:
            sym_orders.append(Order(PEPPER_PRODUCT, pa, -ask_size))

        return sym_orders


# ============================================================================
# Outer Trader: what IMC instantiates
# ============================================================================

class Trader:

    def __init__(self) -> None:
        self._ash    = _AshTrader()
        self._pepper = _PepperTrader()

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[Symbol, List[Order]], int, str]:

        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}
        tick = int(memory.get("tick", 0))
        memory["tick"] = tick + 1

        orders: Dict[Symbol, List[Order]] = {}

        ash_orders = self._ash.run(state)
        if ash_orders:
            orders[ASH_PRODUCT] = ash_orders

        pepper_orders = self._pepper.run(state, tick)
        if pepper_orders:
            orders[PEPPER_PRODUCT] = pepper_orders

        return orders, 0, json.dumps(memory)
