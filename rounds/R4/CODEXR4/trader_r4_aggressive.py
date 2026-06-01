"""
Round 4 Aggressive Trader, revision 2.

This keeps the conservative anti-overfit base and adds a live-only counterparty
copy layer. No trader ID is trusted from historical data. A counterparty must
prove product-specific edge inside the current run before we copy it.

This is intentionally less exciting than the previous Mark-seeded version. The
bad results showed that historical Mark IDs are too easy to overfit.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, Trade, TradingState


# =============================================================================
# CONSTANTS: retune here
# =============================================================================

SUBMISSION = "SUBMISSION"
DAY_PERIOD = 100_000
TICK_SIZE_TS = 100

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

RISK_LIMITS = {
    HG: 90,
    VE: 65,
    **{symbol: 0 for symbol in VOUCHERS},
}

PRODUCT_LOSS_LIMITS = {
    HG: -2_500.0,
    VE: -2_400.0,
}

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

MAX_FAIR_GAP_TO_ADD = {
    HG: 45.0,
    VE: 28.0,
}

# Live counterparty validation. Product-specific stats are required; global hit
# rate alone is not enough.
COPY_PRODUCTS = (HG, VE)
CP_FORWARD_TICKS = 20
CP_MIN_PRODUCT_TRADES = 20
CP_HIT_RATE = 0.68
CP_RECENT_WINDOW = 30
CP_RECENT_KILL_RATE = 0.55
CP_PENDING_CAP = 800
CP_MAX_SLIPPAGE = {
    HG: 10,
    VE: 5,
}
CP_SUBLIMIT = {
    HG: 45,
    VE: 35,
}
CP_MAX_ORDER = {
    HG: 8,
    VE: 12,
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
            "cp": {"pending": [], "stats": {}, "blocked": {}},
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
        if quantity == 0:
            return 0

        exchange_limit = EXCHANGE_LIMITS.get(symbol, 0)
        risk_limit = min(exchange_limit, RISK_LIMITS.get(symbol, exchange_limit))
        current = state.position.get(symbol, 0)
        existing = orders.get(symbol, [])
        planned_buys = sum(max(0, order.quantity) for order in existing)
        planned_sells = sum(max(0, -order.quantity) for order in existing)

        if quantity > 0:
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
        fair = int(round(core + trend))
        store["last_fair"] = fair
        store["last_trend"] = fast - slow
        return fair, fast - slow

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

        if buy_allowed and ask <= fair - take_edge:
            self.add_order(state, orders, symbol, ask, min(ask_qty, take_size))
        if sell_allowed and bid >= fair + take_edge:
            self.add_order(state, orders, symbol, bid, -min(bid_qty, take_size))

        passive_bid = min(fair - passive_edge, bid + join_edge)
        passive_ask = max(fair + passive_edge, ask - join_edge)
        if passive_bid >= passive_ask:
            passive_bid = fair - passive_edge
            passive_ask = fair + passive_edge

        if buy_allowed and position <= RISK_LIMITS[symbol] * 0.70:
            self.add_order(state, orders, symbol, int(passive_bid), quote_size)
        if sell_allowed and position >= -RISK_LIMITS[symbol] * 0.70:
            self.add_order(state, orders, symbol, int(passive_ask), -quote_size)

        if position > RISK_LIMITS[symbol] * 0.75:
            self.flatten_only(state, orders, symbol, max_qty=4)
        elif position < -RISK_LIMITS[symbol] * 0.75:
            self.flatten_only(state, orders, symbol, max_qty=4)

        pnl = self.mark_to_mid_pnl(state, data, symbol)
        logger.print(symbol, f"fair={fair}", f"mid={mid:.1f}", f"trend={trend:.1f}", f"pos={position}", f"pnl={pnl:.1f}")

    # ------------------------------------------------------------------
    # Live-only counterparty validation
    # ------------------------------------------------------------------

    def cp_stat(self, data: Dict[str, Any], trader: str, symbol: str) -> Dict[str, Any]:
        cp = data.setdefault("cp", {"pending": [], "stats": {}, "blocked": {}})
        stats = cp.setdefault("stats", {})
        trader_stat = stats.setdefault(trader, {"products": {}})
        return trader_stat.setdefault("products", {}).setdefault(symbol, {"n": 0, "hits": 0, "recent": []})

    def record_cp_result(self, data: Dict[str, Any], trader: str, symbol: str, hit: bool) -> None:
        if not trader or trader == SUBMISSION:
            return
        stat = self.cp_stat(data, trader, symbol)
        stat["n"] = int(stat.get("n", 0)) + 1
        stat["hits"] = int(stat.get("hits", 0)) + (1 if hit else 0)
        recent = stat.setdefault("recent", [])
        recent.append(1 if hit else 0)
        if len(recent) > CP_RECENT_WINDOW:
            del recent[:-CP_RECENT_WINDOW]

    def update_counterparty_scores(self, state: TradingState, data: Dict[str, Any]) -> None:
        cp = data.setdefault("cp", {"pending": [], "stats": {}, "blocked": {}})
        pending = cp.setdefault("pending", [])
        keep = []

        for item in pending:
            if state.timestamp < item["eval_ts"]:
                keep.append(item)
                continue
            now_mid = self.mid(state, item["symbol"])
            if now_mid is None:
                keep.append(item)
                continue
            move = now_mid - float(item["entry_mid"])
            if abs(move) < 1e-9:
                continue
            hit = (item["side"] > 0 and move > 0) or (item["side"] < 0 and move < 0)
            self.record_cp_result(data, item["trader"], item["symbol"], hit)

        for symbol in COPY_PRODUCTS:
            entry_mid = self.mid(state, symbol)
            if entry_mid is None:
                continue
            for trade in state.market_trades.get(symbol, []):
                if trade.buyer and trade.buyer != SUBMISSION:
                    keep.append({
                        "eval_ts": state.timestamp + CP_FORWARD_TICKS * TICK_SIZE_TS,
                        "symbol": symbol,
                        "trader": trade.buyer,
                        "side": 1,
                        "entry_mid": entry_mid,
                    })
                if trade.seller and trade.seller != SUBMISSION:
                    keep.append({
                        "eval_ts": state.timestamp + CP_FORWARD_TICKS * TICK_SIZE_TS,
                        "symbol": symbol,
                        "trader": trade.seller,
                        "side": -1,
                        "entry_mid": entry_mid,
                    })

        if len(keep) > CP_PENDING_CAP:
            keep = keep[-CP_PENDING_CAP:]
        cp["pending"] = keep

    def trader_product_is_good(self, data: Dict[str, Any], trader: str, symbol: str) -> bool:
        cp = data.get("cp", {})
        if trader in cp.get("blocked", {}).get(symbol, []):
            return False
        stat = cp.get("stats", {}).get(trader, {}).get("products", {}).get(symbol)
        if not stat:
            return False
        n = int(stat.get("n", 0))
        if n < CP_MIN_PRODUCT_TRADES:
            return False
        rate = int(stat.get("hits", 0)) / max(1, n)
        recent = stat.get("recent", [])
        recent_rate = sum(recent) / len(recent) if recent else rate
        if len(recent) >= CP_RECENT_WINDOW and recent_rate < CP_RECENT_KILL_RATE:
            return False
        return rate >= CP_HIT_RATE and recent_rate >= CP_RECENT_KILL_RATE

    def copy_live_counterparties(self, state: TradingState, data: Dict[str, Any], orders: Dict[Symbol, List[Order]]) -> None:
        for symbol in COPY_PRODUCTS:
            if symbol in data.get("disabled", {}):
                continue
            depth = state.order_depths.get(symbol)
            top = self.top_of_book(depth) if depth is not None else None
            if top is None:
                continue
            fair_store = data.get("fair", {}).get(symbol, {})
            fair = fair_store.get("last_fair")
            trend = float(fair_store.get("last_trend", 0.0))
            if fair is None:
                continue
            bid, bid_qty, ask, ask_qty = top
            position = state.position.get(symbol, 0)
            buy_signal = 0
            sell_signal = 0
            sources = []

            for trade in state.market_trades.get(symbol, []):
                if self.trader_product_is_good(data, trade.buyer, symbol):
                    buy_signal += int(trade.quantity)
                    sources.append(f"{trade.buyer}:B")
                if self.trader_product_is_good(data, trade.seller, symbol):
                    sell_signal += int(trade.quantity)
                    sources.append(f"{trade.seller}:S")

            # Copy only when the live fair does not say the fill is wildly bad.
            if buy_signal > 0 and ask <= fair + CP_MAX_SLIPPAGE[symbol] and trend > -MAX_FAIR_GAP_TO_ADD[symbol]:
                qty = min(buy_signal, CP_MAX_ORDER[symbol], ask_qty, max(0, CP_SUBLIMIT[symbol] - position))
                sent = self.add_order(state, orders, symbol, ask, qty)
                if sent:
                    logger.print("COPY_BUY", symbol, sent, ",".join(sources))

            if sell_signal > 0 and bid >= fair - CP_MAX_SLIPPAGE[symbol] and trend < MAX_FAIR_GAP_TO_ADD[symbol]:
                qty = min(sell_signal, CP_MAX_ORDER[symbol], bid_qty, max(0, CP_SUBLIMIT[symbol] + position))
                sent = self.add_order(state, orders, symbol, bid, -qty)
                if sent:
                    logger.print("COPY_SELL", symbol, -sent, ",".join(sources))

    def log_counterparties(self, data: Dict[str, Any]) -> None:
        stats = data.get("cp", {}).get("stats", {})
        pieces = []
        for trader, tstat in stats.items():
            for symbol, stat in tstat.get("products", {}).items():
                n = int(stat.get("n", 0))
                if n >= CP_MIN_PRODUCT_TRADES:
                    rate = int(stat.get("hits", 0)) / max(1, n)
                    if rate >= CP_HIT_RATE - 0.05:
                        pieces.append(f"{trader}/{symbol}:{rate:.2f}/{n}")
        if pieces:
            logger.print("CP", " | ".join(pieces[:6]))

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        data = self.load_data(state.traderData)
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        self.update_accounting(state, data)
        self.update_loss_stops(state, data)
        self.update_counterparty_scores(state, data)

        self.run_delta_one(state, data, orders, HG)
        self.run_delta_one(state, data, orders, VE)
        self.copy_live_counterparties(state, data, orders)

        for symbol in VOUCHERS:
            self.flatten_only(state, orders, symbol, max_qty=12)
            orders.setdefault(symbol, [])
        orders.setdefault(HG, [])
        orders.setdefault(VE, [])

        self.log_counterparties(data)
        trader_data = json.dumps(data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


"""
Plain-English summary:
  Aggressive v2 is conservative v2 plus live-only counterparty following.
  It never starts by trusting Mark 14, Mark 67, or any other historical ID.
  A trader must show product-specific forward hit rate of at least 68% over
  20 evaluated trades before we copy. The copy layer only trades HG/VE, uses
  smaller sublimits, and refuses fills far outside the current robust fair.

Key risks:
  The detector may activate late or never activate. That is intentional: missed
  upside is better than replaying an overfit Mark-ID story into a new day.

Most sensitive constants:
  CP_MIN_PRODUCT_TRADES, CP_HIT_RATE, CP_FORWARD_TICKS, CP_SUBLIMIT,
  PRODUCT_LOSS_LIMITS, RISK_LIMITS, and anchor weights.
"""
