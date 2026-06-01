"""
Round 4 Conservative Trader, revision 2.

Purpose of this revision:
  - Stop overfitting to the three historical days.
  - Remove the voucher residual strategy that lost heavily in live test output.
  - Trade only robust, repeated delta-1 structure with small internal risk caps.
  - Use online product loss stops so a wrong fair cannot bleed for the full day.

The only active products here are HYDROGEL_PACK and VELVETFRUIT_EXTRACT.
All vouchers are deliberately left flat. That is not because they never have
edge; it is because the submitted logs show the fitted option signals were not
robust out of sample.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, Trade, TradingState


# =============================================================================
# CONSTANTS: retune here
# =============================================================================

SUBMISSION = "SUBMISSION"
DAY_PERIOD = 100_000

HG = "HYDROGEL_PACK"
VE = "VELVETFRUIT_EXTRACT"

VOUCHERS = (
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
)

EXCHANGE_LIMITS = {
    HG: 200,
    VE: 200,
    **{symbol: 300 for symbol in VOUCHERS},
}

# Internal caps are intentionally far below exchange limits. This is the main
# anti-overfit control: no signal can use the full limit just because it did so
# well on a small sample.
RISK_LIMITS = {
    HG: 80,
    VE: 55,
    **{symbol: 0 for symbol in VOUCHERS},
}

# Product-level circuit breakers based on our own marked-to-mid accounting.
PRODUCT_LOSS_LIMITS = {
    HG: -2_200.0,
    VE: -2_000.0,
}

# HYDROGEL: stable anchor, wide spread, mild mean reversion.
HG_ANCHOR = 10_000.0
HG_SLOW_ALPHA = 0.010
HG_FAST_ALPHA = 0.200
HG_ANCHOR_WEIGHT = 0.70
HG_BOUND = 90.0
HG_TREND_CLAMP = 4.0
HG_TAKE_EDGE = 7
HG_PASSIVE_EDGE = 8
HG_JOIN_EDGE = 1
HG_TAKE_SIZE = 10
HG_QUOTE_SIZE = 12
HG_TREND_GUARD = 4.0

# VELVETFRUIT: center is stable across days, but short-run drift matters. We
# blend anchor and EMA; pure EMA overreacted in the bad submission.
VE_ANCHOR = 5_250.0
VE_SLOW_ALPHA = 0.012
VE_FAST_ALPHA = 0.180
VE_ANCHOR_WEIGHT = 0.55
VE_BOUND = 75.0
VE_TREND_CLAMP = 3.0
VE_TAKE_EDGE = 4
VE_PASSIVE_EDGE = 4
VE_JOIN_EDGE = 1
VE_TAKE_SIZE = 8
VE_QUOTE_SIZE = 10
VE_TREND_GUARD = 3.0

# If our fair and the market disagree too violently, assume regime shift and
# only quote inventory-reducing orders until the live EMA catches up.
MAX_FAIR_GAP_TO_ADD = {
    HG: 45.0,
    VE: 28.0,
}


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(str(obj) for obj in objects) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        flat_orders = [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]
        payload = [
            state.timestamp,
            state.position,
            flat_orders,
            conversions,
            self.truncate(trader_data, 900),
            self.truncate(self.logs, 1800),
        ]
        print(json.dumps(payload, separators=(",", ":")))
        self.logs = ""

    @staticmethod
    def truncate(value: str, max_len: int) -> str:
        if len(value) <= max_len:
            return value
        return value[: max_len - 3] + "..."


logger = Logger()


# =============================================================================
# Trader
# =============================================================================

class Trader:
    def default_data(self) -> Dict[str, Any]:
        return {
            "fair": {HG: {}, VE: {}},
            "acct": {"cash": {}, "turnover": {}},
            "disabled": {},
        }

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data or trader_data == "SAMPLE":
            return self.default_data()
        try:
            data = json.loads(trader_data)
            base = self.default_data()
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key].update(value)
                else:
                    base[key] = value
            return base
        except Exception:
            return self.default_data()

    @staticmethod
    def top_of_book(depth: OrderDepth) -> Optional[Tuple[int, int, int, int]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders)
        ask = min(depth.sell_orders)
        return bid, depth.buy_orders[bid], ask, -depth.sell_orders[ask]

    def mid(self, state: TradingState, symbol: Symbol) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if depth is None:
            return None
        top = self.top_of_book(depth)
        if top is None:
            return None
        bid, _, ask, _ = top
        return 0.5 * (bid + ask)

    def add_order(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        symbol: Symbol,
        price: int,
        quantity: int,
    ) -> int:
        """Respect exchange limits and stricter internal risk limits."""
        if quantity == 0:
            return 0

        exchange_limit = EXCHANGE_LIMITS.get(symbol, 0)
        risk_limit = min(exchange_limit, RISK_LIMITS.get(symbol, exchange_limit))
        current = state.position.get(symbol, 0)
        existing = orders.get(symbol, [])
        planned_buys = sum(max(0, order.quantity) for order in existing)
        planned_sells = sum(max(0, -order.quantity) for order in existing)

        if quantity > 0:
            # Buying is allowed up to the internal long cap; if already too
            # short, this naturally permits flattening back inside the cap.
            allowed = risk_limit - current - planned_buys
            send = min(quantity, max(0, allowed))
        else:
            allowed = risk_limit + current - planned_sells
            send = -min(-quantity, max(0, allowed))

        if send:
            orders.setdefault(symbol, []).append(Order(symbol, int(price), int(send)))
        return send

    def mark_to_mid_pnl(self, state: TradingState, data: Dict[str, Any], symbol: Symbol) -> float:
        cash = float(data.get("acct", {}).get("cash", {}).get(symbol, 0.0))
        pos = state.position.get(symbol, 0)
        mid = self.mid(state, symbol)
        return cash + (pos * mid if mid is not None else 0.0)

    def update_accounting(self, state: TradingState, data: Dict[str, Any]) -> None:
        acct = data.setdefault("acct", {"cash": {}, "turnover": {}})
        cash = acct.setdefault("cash", {})
        turnover = acct.setdefault("turnover", {})
        for symbol, trades in state.own_trades.items():
            for trade in trades:
                qty = int(trade.quantity)
                px = float(trade.price)
                if trade.buyer == SUBMISSION:
                    cash[symbol] = cash.get(symbol, 0.0) - qty * px
                    turnover[symbol] = turnover.get(symbol, 0) + qty
                elif trade.seller == SUBMISSION:
                    cash[symbol] = cash.get(symbol, 0.0) + qty * px
                    turnover[symbol] = turnover.get(symbol, 0) + qty

    def update_loss_stops(self, state: TradingState, data: Dict[str, Any]) -> None:
        disabled = data.setdefault("disabled", {})
        for symbol, limit in PRODUCT_LOSS_LIMITS.items():
            if symbol in disabled:
                continue
            pnl = self.mark_to_mid_pnl(state, data, symbol)
            if pnl < limit:
                disabled[symbol] = {"timestamp": state.timestamp, "pnl": pnl}
                logger.print("LOSS_STOP", symbol, f"pnl={pnl:.1f}")

    def fair_value(self, data: Dict[str, Any], symbol: Symbol, mid: float) -> Tuple[int, float]:
        store = data.setdefault("fair", {}).setdefault(symbol, {})
        slow = float(store.get("slow", mid))
        fast = float(store.get("fast", mid))

        if symbol == HG:
            anchor = HG_ANCHOR
            slow_alpha = HG_SLOW_ALPHA
            fast_alpha = HG_FAST_ALPHA
            anchor_weight = HG_ANCHOR_WEIGHT
            bound = HG_BOUND
            trend_clamp = HG_TREND_CLAMP
        else:
            anchor = VE_ANCHOR
            slow_alpha = VE_SLOW_ALPHA
            fast_alpha = VE_FAST_ALPHA
            anchor_weight = VE_ANCHOR_WEIGHT
            bound = VE_BOUND
            trend_clamp = VE_TREND_CLAMP

        slow += slow_alpha * (mid - slow)
        fast += fast_alpha * (mid - fast)
        store["slow"] = slow
        store["fast"] = fast

        core = anchor_weight * anchor + (1.0 - anchor_weight) * slow
        core = max(anchor - bound, min(anchor + bound, core))
        trend = max(-trend_clamp, min(trend_clamp, fast - slow))
        return int(round(core + trend)), fast - slow

    def flatten_only(self, state: TradingState, orders: Dict[Symbol, List[Order]], symbol: Symbol, max_qty: int) -> None:
        position = state.position.get(symbol, 0)
        if position == 0:
            return
        depth = state.order_depths.get(symbol)
        top = self.top_of_book(depth) if depth is not None else None
        if top is None:
            return
        bid, bid_qty, ask, ask_qty = top
        if position > 0:
            qty = min(max_qty, position, bid_qty)
            if qty > 0:
                # Flattening can ignore the internal cap by manually using a
                # direct order that still moves toward zero.
                orders.setdefault(symbol, []).append(Order(symbol, bid, -qty))
        elif position < 0:
            qty = min(max_qty, -position, ask_qty)
            if qty > 0:
                orders.setdefault(symbol, []).append(Order(symbol, ask, qty))

    def run_delta_one(self, state: TradingState, data: Dict[str, Any], orders: Dict[Symbol, List[Order]], symbol: Symbol) -> None:
        depth = state.order_depths.get(symbol)
        top = self.top_of_book(depth) if depth is not None else None
        mid = self.mid(state, symbol)
        if top is None or mid is None:
            return

        if symbol in data.get("disabled", {}):
            self.flatten_only(state, orders, symbol, max_qty=10)
            return

        fair, trend = self.fair_value(data, symbol, mid)
        position = state.position.get(symbol, 0)
        bid, bid_qty, ask, ask_qty = top

        if symbol == HG:
            take_edge = HG_TAKE_EDGE
            passive_edge = HG_PASSIVE_EDGE
            join_edge = HG_JOIN_EDGE
            take_size = HG_TAKE_SIZE
            quote_size = HG_QUOTE_SIZE
            trend_guard = HG_TREND_GUARD
        else:
            take_edge = VE_TAKE_EDGE
            passive_edge = VE_PASSIVE_EDGE
            join_edge = VE_JOIN_EDGE
            take_size = VE_TAKE_SIZE
            quote_size = VE_QUOTE_SIZE
            trend_guard = VE_TREND_GUARD

        fair_gap = abs(mid - fair)
        buy_allowed = trend > -trend_guard and fair_gap <= MAX_FAIR_GAP_TO_ADD[symbol]
        sell_allowed = trend < trend_guard and fair_gap <= MAX_FAIR_GAP_TO_ADD[symbol]

        # Taking is stricter than making. It fires only for clear misprices and
        # is disabled in the direction of a live adverse trend.
        if buy_allowed and ask <= fair - take_edge:
            self.add_order(state, orders, symbol, ask, min(ask_qty, take_size))
        if sell_allowed and bid >= fair + take_edge:
            self.add_order(state, orders, symbol, bid, -min(bid_qty, take_size))

        # Passive quotes are small, and we do not quote the side that adds to an
        # already one-sided inventory.
        passive_bid = min(fair - passive_edge, bid + join_edge)
        passive_ask = max(fair + passive_edge, ask - join_edge)
        if passive_bid >= passive_ask:
            passive_bid = fair - passive_edge
            passive_ask = fair + passive_edge

        if buy_allowed and position <= RISK_LIMITS[symbol] * 0.70:
            self.add_order(state, orders, symbol, int(passive_bid), quote_size)
        if sell_allowed and position >= -RISK_LIMITS[symbol] * 0.70:
            self.add_order(state, orders, symbol, int(passive_ask), -quote_size)

        # Gentle inventory repair every tick.
        if position > RISK_LIMITS[symbol] * 0.75:
            self.flatten_only(state, orders, symbol, max_qty=4)
        elif position < -RISK_LIMITS[symbol] * 0.75:
            self.flatten_only(state, orders, symbol, max_qty=4)

        pnl = self.mark_to_mid_pnl(state, data, symbol)
        logger.print(symbol, f"fair={fair}", f"mid={mid:.1f}", f"trend={trend:.1f}", f"pos={position}", f"pnl={pnl:.1f}")

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        data = self.load_data(state.traderData)
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        self.update_accounting(state, data)
        self.update_loss_stops(state, data)

        self.run_delta_one(state, data, orders, HG)
        self.run_delta_one(state, data, orders, VE)

        # Explicitly keep every voucher flat. If a previous upload left a
        # position, try to reduce it; otherwise send no voucher orders.
        for symbol in VOUCHERS:
            self.flatten_only(state, orders, symbol, max_qty=12)
            orders.setdefault(symbol, [])

        orders.setdefault(HG, [])
        orders.setdefault(VE, [])

        trader_data = json.dumps(data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


"""
Plain-English summary:
  This algorithm trades only HYDROGEL_PACK and VELVETFRUIT_EXTRACT with small,
  anchor-blended market making. It avoids fitted voucher residuals entirely.
  It has internal caps of 80 HG and 55 VE, and it disables a product for the
  rest of the day if marked-to-mid PnL breaches its loss limit.

Key risks:
  The strategy may under-trade and leave money on the table. The fair anchors
  can still be wrong on a genuinely shifted day, but loss stops and small caps
  are meant to keep that mistake survivable.

Most sensitive constants:
  RISK_LIMITS, PRODUCT_LOSS_LIMITS, VE_ANCHOR_WEIGHT, HG_ANCHOR_WEIGHT,
  VE_TAKE_EDGE, HG_TAKE_EDGE, and MAX_FAIR_GAP_TO_ADD.
"""
