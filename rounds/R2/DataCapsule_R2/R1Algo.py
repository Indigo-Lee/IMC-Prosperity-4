"""
Combined Round 1 Strategy:
  - INTARIAN_PEPPER_ROOT: Aggressive trend rider (buy to max long, hold)
  - ASH_COATED_OSMIUM: Resin-style market making around 10,000
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]

            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


# --- INTARIAN_PEPPER_ROOT params ---
IPR = "INTARIAN_PEPPER_ROOT"
IPR_LIMIT = 80
IPR_WARMUP = 3
IPR_SLOPE = 0.108

# --- ASH_COATED_OSMIUM params ---
ACO = "ASH_COATED_OSMIUM"
ACO_LIMIT = 80
ACO_FAIR = 10_000
ACO_TAKE_WIDTH = 1
ACO_CLEAR_WIDTH = 0
ACO_DISREGARD_EDGE = 1
ACO_JOIN_EDGE = 2
ACO_DEFAULT_EDGE = 4
ACO_SOFT_POSITION_LIMIT = 10


class Trader:
    def aco_take_best_orders(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]

            if best_ask <= ACO_FAIR - ACO_TAKE_WIDTH:
                quantity = min(best_ask_amount, ACO_LIMIT - position)
                if quantity > 0:
                    orders.append(Order(ACO, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= ACO_FAIR + ACO_TAKE_WIDTH:
                quantity = min(best_bid_amount, ACO_LIMIT + position)
                if quantity > 0:
                    orders.append(Order(ACO, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def aco_clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(ACO_FAIR - ACO_CLEAR_WIDTH)
        fair_for_ask = round(ACO_FAIR + ACO_CLEAR_WIDTH)

        buy_quantity = ACO_LIMIT - (position + buy_order_volume)
        sell_quantity = ACO_LIMIT + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)

            if sent_quantity > 0:
                orders.append(Order(ACO, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)

            if sent_quantity > 0:
                orders.append(Order(ACO, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def aco_market_make(
        self,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        buy_quantity = ACO_LIMIT - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(ACO, round(bid), buy_quantity))

        sell_quantity = ACO_LIMIT + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(ACO, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def aco_make_orders(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > ACO_FAIR + ACO_DISREGARD_EDGE
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < ACO_FAIR - ACO_DISREGARD_EDGE
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(ACO_FAIR + ACO_DEFAULT_EDGE)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - ACO_FAIR) <= ACO_JOIN_EDGE:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(ACO_FAIR - ACO_DEFAULT_EDGE)
        if best_bid_below_fair is not None:
            if abs(ACO_FAIR - best_bid_below_fair) <= ACO_JOIN_EDGE:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if position > ACO_SOFT_POSITION_LIMIT:
            ask -= 1
        elif position < -ACO_SOFT_POSITION_LIMIT:
            bid += 1

        return self.aco_market_make(
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        if state.traderData:
            data = json.loads(state.traderData)
        else:
            data = {
                "n": 0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "sum_xy": 0.0,
                "sum_xx": 0.0,
                "first_mid": None,
            }

        # ============================================================
        # INTARIAN_PEPPER_ROOT — aggressive trend rider
        # ============================================================
        if IPR in state.order_depths:
            depth = state.order_depths[IPR]
            best_bid = max(depth.buy_orders) if depth.buy_orders else None
            best_ask = min(depth.sell_orders) if depth.sell_orders else None

            mid = None
            if best_bid is not None and best_ask is not None and best_ask > best_bid:
                mid = (best_bid + best_ask) / 2.0

            if mid is not None:
                if data["first_mid"] is None:
                    data["first_mid"] = mid

                x = data["n"]
                data["n"] += 1
                data["sum_x"] += x
                data["sum_y"] += mid
                data["sum_xy"] += x * mid
                data["sum_xx"] += x * x

            n = data["n"]
            fair_value = None

            if n >= IPR_WARMUP:
                denom = n * data["sum_xx"] - data["sum_x"] ** 2
                if denom != 0:
                    slope = (n * data["sum_xy"] - data["sum_x"] * data["sum_y"]) / denom
                    intercept = (data["sum_y"] - slope * data["sum_x"]) / n
                    fair_value = intercept + slope * (n - 1)
            elif data["first_mid"] is not None:
                fair_value = data["first_mid"] + IPR_SLOPE * (n - 1)

            pos = state.position.get(IPR, 0)
            capacity = IPR_LIMIT - pos

            if capacity > 0:
                ipr_orders: List[Order] = []

                if best_ask is not None:
                    ask_vol = abs(depth.sell_orders.get(best_ask, 0))
                    aggressive_qty = min(capacity, ask_vol)
                    if aggressive_qty > 0:
                        ipr_orders.append(Order(IPR, best_ask, aggressive_qty))
                        capacity -= aggressive_qty

                if capacity > 0 and fair_value is not None:
                    ipr_orders.append(Order(IPR, int(fair_value), capacity))

                if ipr_orders:
                    orders[IPR] = ipr_orders

        # ============================================================
        # ASH_COATED_OSMIUM — Resin-style market making around 10,000
        # ============================================================
        if ACO in state.order_depths:
            depth = state.order_depths[ACO]
            pos = state.position.get(ACO, 0)
            aco_orders: List[Order] = []

            buy_order_volume = 0
            sell_order_volume = 0

            buy_order_volume, sell_order_volume = self.aco_take_best_orders(
                aco_orders,
                depth,
                pos,
                buy_order_volume,
                sell_order_volume,
            )

            buy_order_volume, sell_order_volume = self.aco_clear_position_order(
                aco_orders,
                depth,
                pos,
                buy_order_volume,
                sell_order_volume,
            )

            self.aco_make_orders(
                aco_orders,
                depth,
                pos,
                buy_order_volume,
                sell_order_volume,
            )

            if aco_orders:
                orders[ACO] = aco_orders

        trader_data = json.dumps(data)

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data