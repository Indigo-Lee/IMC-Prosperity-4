"""
Round 4 Strategy:
  - HYDROGEL_PACK: Adaptive market making around a stable 10,000 anchor
  - VELVETFRUIT_EXTRACT: Standalone fair-value trader
  - VEV vouchers: Restore prior profitable option-pricing logic, no hedging
"""

from __future__ import annotations

import json
import math
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


# --- HYDROGEL_PACK params ---
HG = "HYDROGEL_PACK"
HG_LIMIT = 200
HG_ANCHOR_FAIR = 10_000
HG_HISTORY_WINDOW = 40
HG_TAKE_EDGE = 2
HG_JOIN_EDGE = 1
HG_DEFAULT_EDGE = 3
HG_SOFT_POSITION_LIMIT = 40

# --- VELVETFRUIT_EXTRACT params ---
VE = "VELVETFRUIT_EXTRACT"
VE_LIMIT = 200
VE_FAIR = 5_253
VE_TAKE_WIDTH = 1
VE_CLEAR_WIDTH = 0
VE_DISREGARD_EDGE = 1
VE_JOIN_EDGE = 2
VE_DEFAULT_EDGE = 4
VE_SOFT_POS_LIMIT = 10
VEV_LIMIT = 300
VEV_SYMBOLS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
VE_ACTIVE_OPTION_SYMBOLS = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400"]
VELVET_PRODUCTS = [VE, *VEV_SYMBOLS]
VEV_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
VE_YEAR_DAYS = 365.0
VE_TTE_CANDIDATES_DAYS = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
VE_FALLBACK_VOL = 0.20
VE_MIN_VOL = 0.05
VE_MAX_VOL = 1.00
VE_SIGMA_SMOOTH = 0.35
VE_OPTION_SOFT_LIMIT = 70
VE_OPTION_TAKE_SIZE = 35
VE_OPTION_QUOTE_SIZE = 12
VE_UNDERLYING_HISTORY_WINDOW = 60
VE_UNDERLYING_IMBALANCE_WEIGHT = 6.0
VE_UNDERLYING_TAKE_SIZE = 30
VE_UNDERLYING_QUOTE_SIZE = 20
VE_UNDERLYING_SOFT_LIMIT = 120
VE_UNDERLYING_BASE_EDGE = 2.0
VE_POSITION_BIAS_TICKS = 3.0


class Trader:
    def get_mid_price(self, state: TradingState, symbol: Symbol) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        popular_buy_price = max(order_depth.buy_orders.items(), key=lambda level: level[1])[0]
        popular_sell_price = min(order_depth.sell_orders.items(), key=lambda level: level[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def hg_estimate_fair(self, history: List[float]) -> int:
        recent_history = history[-HG_HISTORY_WINDOW:]
        blended_fair = (sum(recent_history) / len(recent_history) + HG_ANCHOR_FAIR) / 2
        return round(blended_fair)

    def hg_run(
        self,
        state: TradingState,
        hg_data: Dict[str, Any],
        orders: Dict[Symbol, List[Order]],
    ) -> None:
        if HG not in state.order_depths:
            return

        order_depth = state.order_depths[HG]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        history = hg_data.setdefault("history", [])
        mid_price = self.get_mid_price(state, HG)
        if mid_price is None:
            return

        history.append(mid_price)
        if len(history) > HG_HISTORY_WINDOW:
            history.pop(0)

        fair_value = self.hg_estimate_fair(history)
        position = state.position.get(HG, 0)
        buy_capacity = HG_LIMIT - position
        sell_capacity = HG_LIMIT + position

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        hg_orders: List[Order] = []

        for price, volume in sell_orders:
            if buy_capacity <= 0 or price > fair_value - HG_TAKE_EDGE:
                break
            quantity = min(buy_capacity, -volume)
            if quantity > 0:
                hg_orders.append(Order(HG, price, quantity))
                buy_capacity -= quantity

        for price, volume in buy_orders:
            if sell_capacity <= 0 or price < fair_value + HG_TAKE_EDGE:
                break
            quantity = min(sell_capacity, volume)
            if quantity > 0:
                hg_orders.append(Order(HG, price, -quantity))
                sell_capacity -= quantity

        if buy_capacity > 0:
            best_bid = buy_orders[0][0]
            passive_bid = min(fair_value - HG_DEFAULT_EDGE, best_bid + HG_JOIN_EDGE)
            if position < -HG_SOFT_POSITION_LIMIT:
                passive_bid += 1
            hg_orders.append(Order(HG, passive_bid, buy_capacity))

        if sell_capacity > 0:
            best_ask = sell_orders[0][0]
            passive_ask = max(fair_value + HG_DEFAULT_EDGE, best_ask - HG_JOIN_EDGE)
            if position > HG_SOFT_POSITION_LIMIT:
                passive_ask -= 1
            hg_orders.append(Order(HG, passive_ask, -sell_capacity))

        if hg_orders:
            orders[HG] = hg_orders

    def normal_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call_price(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if tte <= 0:
            return max(spot - strike, 0.0)
        if sigma <= 0:
            return max(spot - strike, 0.0)

        vol_term = sigma * math.sqrt(tte)
        if vol_term == 0:
            return max(spot - strike, 0.0)

        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / vol_term
        d2 = d1 - vol_term
        return spot * self.normal_cdf(d1) - strike * self.normal_cdf(d2)

    def bs_call_delta(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if tte <= 0:
            return 1.0 if spot > strike else 0.0
        if sigma <= 0:
            return 1.0 if spot > strike else 0.0

        vol_term = sigma * math.sqrt(tte)
        if vol_term == 0:
            return 1.0 if spot > strike else 0.0

        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / vol_term
        return self.normal_cdf(d1)

    def implied_volatility(self, option_price: float, spot: float, strike: float, tte: float) -> float | None:
        intrinsic = max(spot - strike, 0.0)
        if option_price <= intrinsic + 0.01:
            return None

        low = VE_MIN_VOL
        high = VE_MAX_VOL
        low_price = self.bs_call_price(spot, strike, tte, low)
        high_price = self.bs_call_price(spot, strike, tte, high)
        if option_price < low_price or option_price > high_price:
            return None

        for _ in range(35):
            mid = (low + high) / 2.0
            mid_price = self.bs_call_price(spot, strike, tte, mid)
            if mid_price < option_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

    def ve_edge(self, fair_value: float, best_bid: int, best_ask: int) -> float:
        spread = max(1, best_ask - best_bid)
        return max(1.0, 0.35 * spread, 0.012 * fair_value)

    def book_imbalance(self, order_depth: OrderDepth) -> float:
        top_buy = sum(volume for _, volume in sorted(order_depth.buy_orders.items(), reverse=True)[:2])
        top_sell = sum(-volume for _, volume in sorted(order_depth.sell_orders.items())[:2])
        total = top_buy + top_sell
        if total == 0:
            return 0.0
        return (top_buy - top_sell) / total

    def ve_run_suhas_style(self, state: TradingState, orders: Dict[Symbol, List[Order]]) -> None:
        if VE not in state.order_depths:
            return

        raw = state.order_depths[VE]
        if not raw.buy_orders and not raw.sell_orders:
            return

        depth = OrderDepth()
        depth.buy_orders = dict(raw.buy_orders)
        depth.sell_orders = dict(raw.sell_orders)

        fair = VE_FAIR
        pos = state.position.get(VE, 0)
        ve_orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0

        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            best_ask_amt = -depth.sell_orders[best_ask]
            if best_ask <= fair - VE_TAKE_WIDTH:
                qty = min(best_ask_amt, VE_LIMIT - pos)
                if qty > 0:
                    ve_orders.append(Order(VE, best_ask, qty))
                    buy_vol += qty
                    depth.sell_orders[best_ask] += qty
                    if depth.sell_orders[best_ask] == 0:
                        del depth.sell_orders[best_ask]

        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            best_bid_amt = depth.buy_orders[best_bid]
            if best_bid >= fair + VE_TAKE_WIDTH:
                qty = min(best_bid_amt, VE_LIMIT + pos)
                if qty > 0:
                    ve_orders.append(Order(VE, best_bid, -qty))
                    sell_vol += qty
                    depth.buy_orders[best_bid] -= qty
                    if depth.buy_orders[best_bid] == 0:
                        del depth.buy_orders[best_bid]

        pos_after = pos + buy_vol - sell_vol
        fair_bid = round(fair - VE_CLEAR_WIDTH)
        fair_ask = round(fair + VE_CLEAR_WIDTH)
        buy_capacity = VE_LIMIT - (pos + buy_vol)
        sell_capacity = VE_LIMIT + (pos - sell_vol)

        if pos_after > 0:
            clear_qty = sum(v for p, v in depth.buy_orders.items() if p >= fair_ask)
            clear_qty = min(clear_qty, pos_after)
            send_qty = min(sell_capacity, clear_qty)
            if send_qty > 0:
                ve_orders.append(Order(VE, fair_ask, -abs(send_qty)))
                sell_vol += abs(send_qty)

        if pos_after < 0:
            clear_qty = sum(abs(v) for p, v in depth.sell_orders.items() if p <= fair_bid)
            clear_qty = min(clear_qty, abs(pos_after))
            send_qty = min(buy_capacity, clear_qty)
            if send_qty > 0:
                ve_orders.append(Order(VE, fair_bid, abs(send_qty)))
                buy_vol += abs(send_qty)

        asks_above = [p for p in depth.sell_orders if p > fair + VE_DISREGARD_EDGE]
        bids_below = [p for p in depth.buy_orders if p < fair - VE_DISREGARD_EDGE]
        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        ask = round(fair + VE_DEFAULT_EDGE)
        if best_ask_above is not None:
            if abs(best_ask_above - fair) <= VE_JOIN_EDGE:
                ask = best_ask_above
            else:
                ask = best_ask_above - 1

        bid = round(fair - VE_DEFAULT_EDGE)
        if best_bid_below is not None:
            if abs(fair - best_bid_below) <= VE_JOIN_EDGE:
                bid = best_bid_below
            else:
                bid = best_bid_below + 1

        if pos > VE_SOFT_POS_LIMIT:
            ask -= 1
        elif pos < -VE_SOFT_POS_LIMIT:
            bid += 1

        if bid >= ask:
            ask = bid + 1

        buy_qty = VE_LIMIT - (pos + buy_vol)
        sell_qty = VE_LIMIT + (pos - sell_vol)
        if buy_qty > 0:
            ve_orders.append(Order(VE, round(bid), buy_qty))
        if sell_qty > 0:
            ve_orders.append(Order(VE, round(ask), -sell_qty))

        if ve_orders:
            orders[VE] = ve_orders

    def ve_estimate_underlying_fair(self, state: TradingState, velvet_data: Dict[str, Any]) -> float | None:
        order_depth = state.order_depths.get(VE)
        ve_mid = self.get_mid_price(state, VE)
        if order_depth is None or ve_mid is None:
            return None

        history = velvet_data.setdefault("underlying_history", [])
        history.append(ve_mid)
        if len(history) > VE_UNDERLYING_HISTORY_WINDOW:
            history.pop(0)

        history_mean = sum(history) / len(history)
        imbalance = self.book_imbalance(order_depth)
        spread = max(1, min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys()))
        imbalance_adjustment = imbalance * VE_UNDERLYING_IMBALANCE_WEIGHT * spread
        return 0.65 * ve_mid + 0.35 * history_mean + imbalance_adjustment

    def ve_estimate_sigma_and_tte(self, state: TradingState, ve_mid: float, velvet_data: Dict[str, Any]) -> Tuple[float, float]:
        previous_sigma = velvet_data.get("sigma", VE_FALLBACK_VOL)
        best_sigma = previous_sigma
        best_tte_days = velvet_data.get("tte_days", 5.0)
        best_score = float("inf")

        for tte_days in VE_TTE_CANDIDATES_DAYS:
            tte = tte_days / VE_YEAR_DAYS
            implied_vols: List[float] = []

            for symbol, strike in VEV_STRIKES.items():
                option_mid = self.get_mid_price(state, symbol)
                if option_mid is None:
                    continue

                implied_vol = self.implied_volatility(option_mid, ve_mid, strike, tte)
                if implied_vol is not None and VE_MIN_VOL <= implied_vol <= VE_MAX_VOL:
                    implied_vols.append(implied_vol)

            if len(implied_vols) < 3:
                continue

            implied_vols.sort()
            observed_sigma = implied_vols[len(implied_vols) // 2]
            mean_abs_deviation = sum(abs(vol - observed_sigma) for vol in implied_vols) / len(implied_vols)
            sigma_jump_penalty = 0.35 * abs(observed_sigma - previous_sigma)
            score = mean_abs_deviation + sigma_jump_penalty

            if score < best_score:
                best_score = score
                best_sigma = observed_sigma
                best_tte_days = tte_days

        sigma = (1.0 - VE_SIGMA_SMOOTH) * previous_sigma + VE_SIGMA_SMOOTH * best_sigma
        velvet_data["sigma"] = sigma
        velvet_data["tte_days"] = best_tte_days
        return sigma, best_tte_days / VE_YEAR_DAYS

    def ve_trade_underlying_alpha(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        fair_value: float,
    ) -> None:
        order_depth = state.order_depths.get(VE)
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = -order_depth.sell_orders[best_ask]
        position = state.position.get(VE, 0)

        inventory_bias = (position / max(1, VE_LIMIT)) * VE_POSITION_BIAS_TICKS
        adjusted_fair = fair_value - inventory_bias
        edge = self.ve_edge(adjusted_fair, best_bid, best_ask) * VE_UNDERLYING_BASE_EDGE

        buy_capacity = VE_LIMIT - position
        sell_capacity = VE_LIMIT + position
        ve_orders: List[Order] = []

        buy_edge = adjusted_fair - best_ask
        sell_edge = best_bid - adjusted_fair

        if buy_edge > edge and buy_capacity > 0:
            quantity = min(buy_capacity, best_ask_volume, VE_UNDERLYING_TAKE_SIZE)
            if quantity > 0:
                ve_orders.append(Order(VE, best_ask, quantity))
                buy_capacity -= quantity

        if sell_edge > edge and sell_capacity > 0:
            quantity = min(sell_capacity, best_bid_volume, VE_UNDERLYING_TAKE_SIZE)
            if quantity > 0:
                ve_orders.append(Order(VE, best_bid, -quantity))
                sell_capacity -= quantity

        if buy_edge > 0.6 * edge and buy_capacity > 0:
            bid_price = min(best_bid + 1, math.floor(adjusted_fair - edge))
            if bid_price > 0 and bid_price < best_ask:
                ve_orders.append(Order(VE, bid_price, min(buy_capacity, VE_UNDERLYING_QUOTE_SIZE)))

        if sell_edge > 0.6 * edge and sell_capacity > 0:
            ask_price = max(best_ask - 1, math.ceil(adjusted_fair + edge))
            if ask_price > best_bid:
                ve_orders.append(Order(VE, ask_price, -min(sell_capacity, VE_UNDERLYING_QUOTE_SIZE)))

        if ve_orders:
            orders[VE] = ve_orders

    def ve_trade_options(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        ve_mid: float,
        sigma: float,
        tte: float,
    ) -> None:
        for symbol, strike in VEV_STRIKES.items():
            if symbol not in VE_ACTIVE_OPTION_SYMBOLS:
                continue
            order_depth = state.order_depths.get(symbol)
            if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            position = state.position.get(symbol, 0)
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            best_ask_volume = -order_depth.sell_orders[best_ask]

            fair_value = self.bs_call_price(ve_mid, strike, tte, sigma)
            edge = self.ve_edge(fair_value, best_bid, best_ask)
            buy_capacity = VEV_LIMIT - position
            sell_capacity = VEV_LIMIT + position
            option_orders: List[Order] = []
            buy_edge = fair_value - best_ask
            sell_edge = best_bid - fair_value

            if buy_edge > edge and buy_capacity > 0:
                quantity = min(buy_capacity, best_ask_volume, VE_OPTION_TAKE_SIZE)
                if quantity > 0:
                    option_orders.append(Order(symbol, best_ask, quantity))
                    buy_capacity -= quantity

            if sell_edge > edge and sell_capacity > 0:
                quantity = min(sell_capacity, best_bid_volume, VE_OPTION_TAKE_SIZE)
                if quantity > 0:
                    option_orders.append(Order(symbol, best_bid, -quantity))
                    sell_capacity -= quantity

            passive_buy_quantity = min(buy_capacity, VE_OPTION_QUOTE_SIZE)
            if passive_buy_quantity > 0 and position < VE_OPTION_SOFT_LIMIT and buy_edge > 0.45 * edge and position <= 0:
                bid_price = min(best_bid + 1, math.floor(fair_value - edge))
                if bid_price > 0 and bid_price < best_ask:
                    option_orders.append(Order(symbol, bid_price, passive_buy_quantity))

            passive_sell_quantity = min(sell_capacity, VE_OPTION_QUOTE_SIZE)
            if passive_sell_quantity > 0 and position > -VE_OPTION_SOFT_LIMIT and sell_edge > 0.45 * edge and position >= 0:
                ask_price = max(best_ask - 1, math.ceil(fair_value + edge))
                if ask_price > best_bid:
                    option_orders.append(Order(symbol, ask_price, -passive_sell_quantity))

            if option_orders:
                orders[symbol] = option_orders

    def run_velvet(self, state: TradingState, velvet_data: Dict[str, Any], orders: Dict[Symbol, List[Order]]) -> None:
        ve_mid = self.get_mid_price(state, VE)
        if ve_mid is None:
            return

        self.ve_run_suhas_style(state, orders)

        sigma, tte = self.ve_estimate_sigma_and_tte(state, ve_mid, velvet_data)
        self.ve_trade_options(state, orders, ve_mid, sigma, tte)

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        if state.traderData:
            data = json.loads(state.traderData)
        else:
            data = {
                "hydrogel": {"history": []},
                "velvet": {},
            }

        if "hydrogel" not in data:
            data = {
                "hydrogel": data.get("hydrogel", {"history": []}),
                "velvet": data.get("velvet", {}),
            }
        else:
            hydrogel = data["hydrogel"]
            if "history" not in hydrogel:
                hydrogel["history"] = hydrogel.pop("prices", [])
            hydrogel.pop("prev_price", None)

        hydrogel_data = data["hydrogel"]
        velvet_data = data.setdefault("velvet", {})

        # ============================================================
        # HYDROGEL_PACK — adaptive market making around a stable fair
        # ============================================================
        self.hg_run(state, hydrogel_data, orders)

        # ============================================================
        # VELVETFRUIT_EXTRACT + VEV vouchers — voucher-first pricing with light emergency hedge
        # ============================================================
        self.run_velvet(state, velvet_data, orders)

        trader_data = json.dumps(data)

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data