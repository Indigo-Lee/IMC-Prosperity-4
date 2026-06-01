"""
Round 4 Aggressive Trader

This file uses the conservative Round 4 base and adds a generic counterparty
detector. Historical Round 4 trades show Mark 67 as a strong VE buyer
(about 82% good over 10 ticks, about 69% over 20 ticks), so Mark 67 is seeded
as a suspect while the live rolling tracker can confirm or kill the signal.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Symbol, TradingState


# =============================================================================
# CONSTANTS: retune here
# =============================================================================

DAY_PERIOD = 100_000
TICK_SIZE_TS = 100
SUBMISSION = "SUBMISSION"

HG = "HYDROGEL_PACK"
VE = "VELVETFRUIT_EXTRACT"

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

POSITION_LIMITS = {
    HG: 200,
    VE: 200,
    **{symbol: 300 for symbol in VEV_STRIKES},
}

# Conservative base.
HG_ANCHOR = 10_000.0
HG_SLOW_ALPHA = 0.010
HG_FAST_ALPHA = 0.300
HG_BOUND = 200.0
HG_TREND_CLAMP = 6.0
HG_TAKE_EDGE = 3
HG_PASSIVE_EDGE = 4
HG_JOIN_EDGE = 1
HG_TAKE_SIZE = 30
HG_QUOTE_SIZE = 42
HG_SOFT_LIMIT = 80

VE_ANCHOR = 5_250.0
VE_SLOW_ALPHA = 0.010
VE_FAST_ALPHA = 0.220
VE_BOUND = 120.0
VE_TREND_CLAMP = 4.0
VE_TAKE_EDGE = 2
VE_PASSIVE_EDGE = 3
VE_JOIN_EDGE = 1
VE_TAKE_SIZE = 35
VE_QUOTE_SIZE = 48
VE_SOFT_LIMIT = 70

DEEP_INTRINSIC_VOUCHERS = ("VEV_4000", "VEV_4500")
SMILE_FIT_VOUCHERS = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")
TRADE_VOUCHERS = ("VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")
FREE_CALL_VOUCHERS = ("VEV_6000", "VEV_6500")

YEAR_DAYS = 365.0
TTE_CANDIDATES_DAYS = (7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
VOL_FALLBACK = 0.25
VOL_MIN = 0.05
VOL_MAX = 1.25
VOL_SMOOTH = 0.35

IV_HISTORY = 160
IV_MIN_HISTORY = 28
IV_CHEAP_Z = -1.35
IV_EXIT_Z = -0.10
IV_LONG_SUBLIMIT = 90
IV_TAKE_SIZE = 18
IV_PASSIVE_SIZE = 6

FREE_CALL_BID = 0
FREE_CALL_LIMIT = 120
FREE_CALL_SIZE = 24

DELTA_HEDGE_TRIGGER = 95
DELTA_HEDGE_LIMIT = 120
DELTA_HEDGE_SIZE = 22

# Insider detector.
INSIDER_FORWARD_TICKS = 20
INSIDER_MIN_TRADES = 20
INSIDER_HIT_THRESHOLD = 0.65
INSIDER_KILL_WINDOW = 30
INSIDER_KILL_RATE = 0.55
INSIDER_PENDING_CAP = 700
INSIDER_MAX_COPY_MULT = 2.0

INSIDER_SUBLIMITS = {
    HG: 55,
    VE: 75,
    "VEV_5200": 45,
    "VEV_5300": 45,
    "VEV_5400": 45,
    "VEV_5500": 35,
}
INSIDER_MAX_ORDER = {
    HG: 18,
    VE: 36,
    "VEV_5200": 12,
    "VEV_5300": 12,
    "VEV_5400": 12,
    "VEV_5500": 8,
}

# Historical R4 seed from the data read for this build.
SEEDED_INSIDERS = {
    "Mark 67": {
        "n": 156,
        "hits": 107,
        "products": {"VELVETFRUIT_EXTRACT": {"n": 156, "hits": 107}},
    }
}


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(str(o) for o in objects) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        compact_orders = [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]
        payload = [
            state.timestamp,
            dict(sorted(state.position.items())),
            compact_orders,
            conversions,
            self._truncate(trader_data, 900),
            self._truncate(self.logs, 1800),
        ]
        print(json.dumps(payload, separators=(",", ":")))
        self.logs = ""

    @staticmethod
    def _truncate(value: str, max_len: int) -> str:
        if len(value) <= max_len:
            return value
        return value[: max(0, max_len - 3)] + "..."


logger = Logger()


# =============================================================================
# Trader
# =============================================================================

class Trader:
    # ------------------------------------------------------------------
    # State and market helpers
    # ------------------------------------------------------------------

    def default_data(self) -> Dict[str, Any]:
        stats = {}
        suspects = {}
        for trader, seed in SEEDED_INSIDERS.items():
            stats[trader] = {
                "n": seed["n"],
                "hits": seed["hits"],
                "recent": [],
                "products": seed["products"],
                "seeded": True,
            }
            suspects[trader] = {
                "rate": seed["hits"] / max(1, seed["n"]),
                "since": 0,
                "source": "historical_seed",
            }
        return {
            "hg": {},
            "ve": {},
            "iv": {"sigma": VOL_FALLBACK, "tte_days": 4.0, "resid": {}},
            "acct": {"cash": {}, "traded": {}},
            "insider": {"pending": [], "stats": stats, "suspects": suspects, "disabled": {}},
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
    def best_bid_ask(depth: OrderDepth) -> Optional[Tuple[int, int, int, int]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders)
        ask = min(depth.sell_orders)
        return bid, depth.buy_orders[bid], ask, -depth.sell_orders[ask]

    def mid(self, state: TradingState, symbol: Symbol) -> Optional[float]:
        depth = state.order_depths.get(symbol)
        if depth is None:
            return None
        top = self.best_bid_ask(depth)
        if top is None:
            return None
        bid, _, ask, _ = top
        return 0.5 * (bid + ask)

    def add_order(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        symbol: Symbol,
        price: float,
        quantity: int,
    ) -> int:
        if quantity == 0:
            return 0
        limit = POSITION_LIMITS.get(symbol)
        if limit is None:
            return 0

        current = state.position.get(symbol, 0)
        existing = orders.get(symbol, [])
        planned_buys = sum(max(0, order.quantity) for order in existing)
        planned_sells = sum(max(0, -order.quantity) for order in existing)

        if quantity > 0:
            allowed = limit - current - planned_buys
            send = min(quantity, max(0, allowed))
        else:
            allowed = limit + current - planned_sells
            send = -min(-quantity, max(0, allowed))

        if send != 0:
            orders.setdefault(symbol, []).append(Order(symbol, int(round(price)), int(send)))
        return send

    @staticmethod
    def scaled_size(direction: int, position: int, limit: int, soft_limit: int, base_size: int) -> int:
        if direction > 0 and position > soft_limit:
            scale = max(0.10, (limit - position) / max(1, limit - soft_limit))
        elif direction < 0 and position < -soft_limit:
            scale = max(0.10, (limit + position) / max(1, limit - soft_limit))
        else:
            scale = 1.0
        return max(0, int(base_size * scale))

    def update_accounting(self, state: TradingState, data: Dict[str, Any]) -> None:
        acct = data.setdefault("acct", {"cash": {}, "traded": {}})
        cash = acct.setdefault("cash", {})
        traded = acct.setdefault("traded", {})
        for symbol, trades in state.own_trades.items():
            for trade in trades:
                qty = int(trade.quantity)
                px = float(trade.price)
                if trade.buyer == SUBMISSION:
                    cash[symbol] = cash.get(symbol, 0.0) - px * qty
                    traded[symbol] = traded.get(symbol, 0) + qty
                elif trade.seller == SUBMISSION:
                    cash[symbol] = cash.get(symbol, 0.0) + px * qty
                    traded[symbol] = traded.get(symbol, 0) + qty

    def log_attribution(self, state: TradingState, data: Dict[str, Any]) -> None:
        cash = data.get("acct", {}).get("cash", {})
        pieces = []
        for symbol in (HG, VE, "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"):
            pos = state.position.get(symbol, 0)
            mark = self.mid(state, symbol)
            pnl = cash.get(symbol, 0.0) + (pos * mark if mark is not None else 0.0)
            if pos or abs(cash.get(symbol, 0.0)) > 1e-9:
                pieces.append(f"{symbol}:pos={pos},mtm={pnl:.1f}")
        if pieces:
            logger.print("ATTR", " | ".join(pieces))

    # ------------------------------------------------------------------
    # Conservative base
    # ------------------------------------------------------------------

    def update_ema_fair(
        self,
        store: Dict[str, Any],
        mid: float,
        anchor: float,
        slow_alpha: float,
        fast_alpha: float,
        bound: float,
        trend_clamp: float,
        anchored: bool,
    ) -> int:
        slow = float(store.get("slow", mid))
        fast = float(store.get("fast", mid))
        slow += slow_alpha * (mid - slow)
        fast += fast_alpha * (mid - fast)
        store["slow"] = slow
        store["fast"] = fast
        core = 0.50 * slow + 0.50 * anchor if anchored else slow
        core = max(anchor - bound, min(anchor + bound, core))
        trend = max(-trend_clamp, min(trend_clamp, fast - slow))
        return int(round(core + trend))

    def run_delta_mm(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        data: Dict[str, Any],
        symbol: Symbol,
    ) -> None:
        depth = state.order_depths.get(symbol)
        if depth is None:
            return
        top = self.best_bid_ask(depth)
        mid = self.mid(state, symbol)
        if top is None or mid is None:
            return

        if symbol == HG:
            fair = self.update_ema_fair(
                data.setdefault("hg", {}), mid, HG_ANCHOR, HG_SLOW_ALPHA, HG_FAST_ALPHA,
                HG_BOUND, HG_TREND_CLAMP, anchored=True
            )
            take_edge, passive_edge, join_edge = HG_TAKE_EDGE, HG_PASSIVE_EDGE, HG_JOIN_EDGE
            take_size, quote_size, soft = HG_TAKE_SIZE, HG_QUOTE_SIZE, HG_SOFT_LIMIT
        else:
            fair = self.update_ema_fair(
                data.setdefault("ve", {}), mid, VE_ANCHOR, VE_SLOW_ALPHA, VE_FAST_ALPHA,
                VE_BOUND, VE_TREND_CLAMP, anchored=False
            )
            take_edge, passive_edge, join_edge = VE_TAKE_EDGE, VE_PASSIVE_EDGE, VE_JOIN_EDGE
            take_size, quote_size, soft = VE_TAKE_SIZE, VE_QUOTE_SIZE, VE_SOFT_LIMIT

        position = state.position.get(symbol, 0)
        for ask in sorted(depth.sell_orders):
            if ask > fair - take_edge:
                break
            qty = min(-depth.sell_orders[ask], self.scaled_size(1, position, POSITION_LIMITS[symbol], soft, take_size))
            if qty > 0:
                sent = self.add_order(state, orders, symbol, ask, qty)
                position += sent

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < fair + take_edge:
                break
            qty = min(depth.buy_orders[bid], self.scaled_size(-1, position, POSITION_LIMITS[symbol], soft, take_size))
            if qty > 0:
                sent = self.add_order(state, orders, symbol, bid, -qty)
                position += sent

        bid, _, ask, _ = top
        passive_bid = min(fair - passive_edge, bid + join_edge)
        passive_ask = max(fair + passive_edge, ask - join_edge)
        if passive_bid >= passive_ask:
            passive_bid = fair - passive_edge
            passive_ask = fair + passive_edge

        buy_size = self.scaled_size(1, state.position.get(symbol, 0), POSITION_LIMITS[symbol], soft, quote_size)
        sell_size = self.scaled_size(-1, state.position.get(symbol, 0), POSITION_LIMITS[symbol], soft, quote_size)
        self.add_order(state, orders, symbol, passive_bid, buy_size)
        self.add_order(state, orders, symbol, passive_ask, -sell_size)
        logger.print(symbol, f"fair={fair}", f"mid={mid:.1f}", f"pos={state.position.get(symbol, 0)}")

    @staticmethod
    def normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if spot <= 0 or strike <= 0 or tte <= 0 or sigma <= 0:
            return max(spot - strike, 0.0)
        vol = sigma * math.sqrt(tte)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / vol
        d2 = d1 - vol
        return spot * self.normal_cdf(d1) - strike * self.normal_cdf(d2)

    def bs_delta(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if spot <= 0 or strike <= 0 or tte <= 0 or sigma <= 0:
            return 1.0 if spot > strike else 0.0
        vol = sigma * math.sqrt(tte)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / vol
        return self.normal_cdf(d1)

    def implied_vol(self, price: float, spot: float, strike: float, tte: float) -> Optional[float]:
        intrinsic = max(spot - strike, 0.0)
        if price <= intrinsic + 0.01:
            return None
        lo, hi = VOL_MIN, VOL_MAX
        if price < self.bs_call(spot, strike, tte, lo) or price > self.bs_call(spot, strike, tte, hi):
            return None
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if self.bs_call(spot, strike, tte, mid) < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def estimate_sigma_tte(self, state: TradingState, spot: float, iv_data: Dict[str, Any]) -> Tuple[float, float]:
        prev_sigma = float(iv_data.get("sigma", VOL_FALLBACK))
        prev_tte = float(iv_data.get("tte_days", 4.0))
        best_sigma = prev_sigma
        best_tte_days = prev_tte
        best_score = float("inf")
        for tte_days in TTE_CANDIDATES_DAYS:
            tte = tte_days / YEAR_DAYS
            vols = []
            for symbol in SMILE_FIT_VOUCHERS:
                mid = self.mid(state, symbol)
                if mid is None:
                    continue
                vol = self.implied_vol(mid, spot, VEV_STRIKES[symbol], tte)
                if vol is not None:
                    vols.append(vol)
            if len(vols) < 3:
                continue
            vols.sort()
            median = vols[len(vols) // 2]
            mad = sum(abs(v - median) for v in vols) / len(vols)
            score = mad + 0.35 * abs(median - prev_sigma)
            if score < best_score:
                best_score = score
                best_sigma = median
                best_tte_days = tte_days
        sigma = (1.0 - VOL_SMOOTH) * prev_sigma + VOL_SMOOTH * best_sigma
        iv_data["sigma"] = sigma
        iv_data["tte_days"] = best_tte_days
        return sigma, best_tte_days / YEAR_DAYS

    @staticmethod
    def fit_parabola(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        n = len(points)
        if n < 3:
            return None
        sx = sum(x for x, _ in points)
        sx2 = sum(x * x for x, _ in points)
        sx3 = sum(x * x * x for x, _ in points)
        sx4 = sum(x * x * x * x for x, _ in points)
        sy = sum(y for _, y in points)
        sxy = sum(x * y for x, y in points)
        sx2y = sum(x * x * y for x, y in points)
        det = n * (sx2 * sx4 - sx3 * sx3) - sx * (sx * sx4 - sx2 * sx3) + sx2 * (sx * sx3 - sx2 * sx2)
        if abs(det) < 1e-12:
            return None
        a = (sy * (sx2 * sx4 - sx3 * sx3) - sx * (sxy * sx4 - sx3 * sx2y) + sx2 * (sxy * sx3 - sx2 * sx2y)) / det
        b = (n * (sxy * sx4 - sx3 * sx2y) - sy * (sx * sx4 - sx2 * sx3) + sx2 * (sx * sx2y - sxy * sx2)) / det
        c = (n * (sx2 * sx2y - sxy * sx3) - sx * (sx * sx2y - sxy * sx2) + sy * (sx * sx3 - sx2 * sx2)) / det
        return a, b, c

    def residual_zscores(self, state: TradingState, data: Dict[str, Any], spot: float, tte: float) -> Dict[str, float]:
        observations: Dict[str, Tuple[float, float]] = {}
        fit_points = []
        for symbol in SMILE_FIT_VOUCHERS:
            mid = self.mid(state, symbol)
            if mid is None:
                continue
            vol = self.implied_vol(mid, spot, VEV_STRIKES[symbol], tte)
            if vol is None:
                continue
            moneyness = math.log(VEV_STRIKES[symbol] / spot)
            observations[symbol] = (moneyness, vol)
            fit_points.append((moneyness, vol))
        fit = self.fit_parabola(fit_points)
        if fit is None:
            return {}
        a, b, c = fit
        iv_data = data.setdefault("iv", {})
        history = iv_data.setdefault("resid", {})
        zscores: Dict[str, float] = {}
        for symbol, (moneyness, vol) in observations.items():
            fitted = a + b * moneyness + c * moneyness * moneyness
            resid = vol - fitted
            arr = history.setdefault(symbol, [])
            arr.append(resid)
            if len(arr) > IV_HISTORY:
                del arr[:-IV_HISTORY]
            if len(arr) >= IV_MIN_HISTORY:
                mean = sum(arr) / len(arr)
                var = sum((x - mean) * (x - mean) for x in arr) / len(arr)
                zscores[symbol] = (resid - mean) / math.sqrt(max(var, 1e-12))
        return zscores

    def run_vouchers(self, state: TradingState, orders: Dict[Symbol, List[Order]], data: Dict[str, Any]) -> None:
        spot = self.mid(state, VE)
        if spot is None:
            return
        iv_data = data.setdefault("iv", {"sigma": VOL_FALLBACK, "tte_days": 4.0, "resid": {}})
        sigma, tte = self.estimate_sigma_tte(state, spot, iv_data)
        zscores = self.residual_zscores(state, data, spot, tte)

        for symbol in TRADE_VOUCHERS:
            depth = state.order_depths.get(symbol)
            top = self.best_bid_ask(depth) if depth is not None else None
            if top is None:
                continue
            bid, bid_qty, ask, ask_qty = top
            pos = state.position.get(symbol, 0)
            z = zscores.get(symbol)
            if z is None:
                continue
            if z <= IV_CHEAP_Z and pos < IV_LONG_SUBLIMIT:
                self.add_order(state, orders, symbol, ask, min(ask_qty, IV_TAKE_SIZE, IV_LONG_SUBLIMIT - pos))
            elif z >= IV_EXIT_Z and pos > 0:
                self.add_order(state, orders, symbol, bid, -min(bid_qty, IV_TAKE_SIZE, pos))
            if z <= 0.25 and pos < IV_LONG_SUBLIMIT:
                self.add_order(state, orders, symbol, bid, min(IV_PASSIVE_SIZE, IV_LONG_SUBLIMIT - pos))
            logger.print(symbol, f"z={z:.2f}", f"pos={pos}")

        for symbol in FREE_CALL_VOUCHERS:
            pos = state.position.get(symbol, 0)
            if pos < FREE_CALL_LIMIT:
                self.add_order(state, orders, symbol, FREE_CALL_BID, min(FREE_CALL_SIZE, FREE_CALL_LIMIT - pos))

        for symbol in DEEP_INTRINSIC_VOUCHERS:
            if symbol in state.order_depths:
                logger.print(symbol, "skip=intrinsic_like")

        self.delta_governor(state, orders, sigma, tte)

    def delta_governor(self, state: TradingState, orders: Dict[Symbol, List[Order]], sigma: float, tte: float) -> None:
        spot = self.mid(state, VE)
        depth = state.order_depths.get(VE)
        top = self.best_bid_ask(depth) if depth is not None else None
        if spot is None or top is None:
            return
        option_delta = 0.0
        for symbol, strike in VEV_STRIKES.items():
            if symbol in DEEP_INTRINSIC_VOUCHERS:
                continue
            pos = state.position.get(symbol, 0)
            if pos:
                option_delta += pos * self.bs_delta(spot, strike, tte, sigma)
        ve_pos = state.position.get(VE, 0)
        net_delta = ve_pos + option_delta
        if abs(net_delta) < DELTA_HEDGE_TRIGGER:
            return
        target_ve = int(max(-DELTA_HEDGE_LIMIT, min(DELTA_HEDGE_LIMIT, -option_delta)))
        need = target_ve - ve_pos
        bid, bid_qty, ask, ask_qty = top
        if need > 0:
            self.add_order(state, orders, VE, ask, min(ask_qty, DELTA_HEDGE_SIZE, need))
        elif need < 0:
            self.add_order(state, orders, VE, bid, -min(bid_qty, DELTA_HEDGE_SIZE, -need))
        logger.print("DELTA", f"net={net_delta:.1f}", f"target_ve={target_ve}")

    # ------------------------------------------------------------------
    # Insider detection and copy trading
    # ------------------------------------------------------------------

    def ensure_trader_stat(self, insider: Dict[str, Any], trader: str) -> Dict[str, Any]:
        stats = insider.setdefault("stats", {})
        if trader not in stats:
            stats[trader] = {"n": 0, "hits": 0, "recent": [], "products": {}}
        return stats[trader]

    def record_good_trade(self, insider: Dict[str, Any], trader: str, symbol: str, hit: bool) -> None:
        if not trader or trader == SUBMISSION:
            return
        stat = self.ensure_trader_stat(insider, trader)
        stat["n"] = int(stat.get("n", 0)) + 1
        stat["hits"] = int(stat.get("hits", 0)) + (1 if hit else 0)
        recent = stat.setdefault("recent", [])
        recent.append(1 if hit else 0)
        if len(recent) > INSIDER_KILL_WINDOW:
            del recent[:-INSIDER_KILL_WINDOW]
        prod = stat.setdefault("products", {}).setdefault(symbol, {"n": 0, "hits": 0})
        prod["n"] = int(prod.get("n", 0)) + 1
        prod["hits"] = int(prod.get("hits", 0)) + (1 if hit else 0)

    def refresh_suspects(self, state: TradingState, insider: Dict[str, Any]) -> None:
        suspects = insider.setdefault("suspects", {})
        disabled = insider.setdefault("disabled", {})
        for trader, stat in insider.setdefault("stats", {}).items():
            n = int(stat.get("n", 0))
            hits = int(stat.get("hits", 0))
            if n <= 0:
                continue
            rate = hits / n
            recent = stat.get("recent", [])
            if trader in suspects and len(recent) >= INSIDER_KILL_WINDOW:
                recent_rate = sum(recent[-INSIDER_KILL_WINDOW:]) / INSIDER_KILL_WINDOW
                if recent_rate < INSIDER_KILL_RATE:
                    suspects.pop(trader, None)
                    disabled[trader] = state.timestamp
                    logger.print("INSIDER_KILL", trader, f"recent={recent_rate:.2f}")
                    continue
            if trader in disabled:
                continue
            if n >= INSIDER_MIN_TRADES and rate >= INSIDER_HIT_THRESHOLD:
                suspects[trader] = {"rate": rate, "since": suspects.get(trader, {}).get("since", state.timestamp)}

    def update_insider_tracker(self, state: TradingState, data: Dict[str, Any]) -> None:
        insider = data.setdefault("insider", {"pending": [], "stats": {}, "suspects": {}, "disabled": {}})
        pending = insider.setdefault("pending", [])
        still_pending = []
        for item in pending:
            if state.timestamp < item["eval_ts"]:
                still_pending.append(item)
                continue
            future_mid = self.mid(state, item["symbol"])
            if future_mid is None:
                if state.timestamp <= item["eval_ts"] + 5 * TICK_SIZE_TS:
                    still_pending.append(item)
                continue
            move = future_mid - float(item["entry_mid"])
            if abs(move) < 1e-9:
                continue
            hit = (move > 0 and item["side"] > 0) or (move < 0 and item["side"] < 0)
            self.record_good_trade(insider, item["trader"], item["symbol"], hit)

        for symbol, trades in state.market_trades.items():
            entry_mid = self.mid(state, symbol)
            if entry_mid is None:
                continue
            for trade in trades:
                if trade.buyer and trade.buyer != SUBMISSION:
                    still_pending.append({
                        "eval_ts": state.timestamp + INSIDER_FORWARD_TICKS * TICK_SIZE_TS,
                        "symbol": symbol,
                        "trader": trade.buyer,
                        "side": 1,
                        "entry_mid": entry_mid,
                    })
                if trade.seller and trade.seller != SUBMISSION:
                    still_pending.append({
                        "eval_ts": state.timestamp + INSIDER_FORWARD_TICKS * TICK_SIZE_TS,
                        "symbol": symbol,
                        "trader": trade.seller,
                        "side": -1,
                        "entry_mid": entry_mid,
                    })

        if len(still_pending) > INSIDER_PENDING_CAP:
            still_pending = still_pending[-INSIDER_PENDING_CAP:]
        insider["pending"] = still_pending
        self.refresh_suspects(state, insider)

    def trader_hit_rate(self, insider: Dict[str, Any], trader: str, symbol: str) -> float:
        stat = insider.get("stats", {}).get(trader, {})
        prod = stat.get("products", {}).get(symbol)
        if prod and prod.get("n", 0) >= 10:
            return prod.get("hits", 0) / max(1, prod.get("n", 0))
        return stat.get("hits", 0) / max(1, stat.get("n", 1))

    def copy_insiders(self, state: TradingState, orders: Dict[Symbol, List[Order]], data: Dict[str, Any]) -> None:
        insider = data.setdefault("insider", {"pending": [], "stats": {}, "suspects": {}, "disabled": {}})
        suspects = insider.get("suspects", {})
        if not suspects:
            return

        for symbol, trades in state.market_trades.items():
            if symbol not in INSIDER_SUBLIMITS or symbol not in state.order_depths:
                continue
            top = self.best_bid_ask(state.order_depths[symbol])
            if top is None:
                continue
            bid, bid_qty, ask, ask_qty = top
            position = state.position.get(symbol, 0)
            sublimit = INSIDER_SUBLIMITS[symbol]

            buy_signal = 0
            sell_signal = 0
            source_bits = []
            for trade in trades:
                if trade.buyer in suspects:
                    rate = self.trader_hit_rate(insider, trade.buyer, symbol)
                    mult = 1.0 + min(INSIDER_MAX_COPY_MULT - 1.0, max(0.0, rate - INSIDER_HIT_THRESHOLD) / (1.0 - INSIDER_HIT_THRESHOLD))
                    buy_signal += int(max(1, round(trade.quantity * mult)))
                    source_bits.append(f"{trade.buyer}:B:{rate:.2f}")
                if trade.seller in suspects:
                    rate = self.trader_hit_rate(insider, trade.seller, symbol)
                    mult = 1.0 + min(INSIDER_MAX_COPY_MULT - 1.0, max(0.0, rate - INSIDER_HIT_THRESHOLD) / (1.0 - INSIDER_HIT_THRESHOLD))
                    sell_signal += int(max(1, round(trade.quantity * mult)))
                    source_bits.append(f"{trade.seller}:S:{rate:.2f}")

            max_order = INSIDER_MAX_ORDER.get(symbol, 10)
            if buy_signal > 0 and position < sublimit:
                qty = min(buy_signal, max_order, ask_qty, sublimit - position)
                sent = self.add_order(state, orders, symbol, ask, qty)
                if sent:
                    logger.print("COPY_BUY", symbol, sent, ",".join(source_bits))
            if sell_signal > 0 and position > -sublimit:
                qty = min(sell_signal, max_order, bid_qty, sublimit + position)
                sent = self.add_order(state, orders, symbol, bid, -qty)
                if sent:
                    logger.print("COPY_SELL", symbol, -sent, ",".join(source_bits))

    def log_insiders(self, data: Dict[str, Any]) -> None:
        insider = data.get("insider", {})
        suspects = insider.get("suspects", {})
        if not suspects:
            return
        pieces = []
        for trader, meta in sorted(suspects.items()):
            stat = insider.get("stats", {}).get(trader, {})
            rate = stat.get("hits", 0) / max(1, stat.get("n", 1))
            recent = stat.get("recent", [])
            recent_rate = sum(recent) / len(recent) if recent else None
            if recent_rate is None:
                pieces.append(f"{trader}:rate={rate:.2f},n={stat.get('n', 0)}")
            else:
                pieces.append(f"{trader}:rate={rate:.2f},recent={recent_rate:.2f},n={stat.get('n', 0)}")
        logger.print("INSIDERS", " | ".join(pieces))

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        data = self.load_data(state.traderData)
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        self.update_accounting(state, data)
        self.update_insider_tracker(state, data)

        # Conservative base first.
        self.run_delta_mm(state, orders, data, HG)
        self.run_delta_mm(state, orders, data, VE)
        self.run_vouchers(state, orders, data)

        # Additive copy-trading layer, bounded by smaller sub-limits.
        self.copy_insiders(state, orders, data)

        for symbol in state.order_depths:
            orders.setdefault(symbol, [])

        self.log_attribution(state, data)
        self.log_insiders(data)
        trader_data = json.dumps(data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


"""
Summary:
  This aggressive trader starts from the conservative base and adds rolling
  counterparty scoring. Each market trade creates a pending evaluation; after
  INSIDER_FORWARD_TICKS, the trader gets a hit if their side matches the price
  move. Traders above 65% hit rate over 20+ evaluations are followed. Mark 67 is
  seeded from historical Round 4 evidence, but the live kill switch stops
  following any suspect whose trailing 30 evaluated trades fall below 55%.

Key risks:
  The biggest risk is false insider identification or a seeded Mark 67 regime
  change. Copy orders are therefore capped by INSIDER_SUBLIMITS and added after
  the base strategy, so they cannot replace the market-making logic.

Most sensitive constants:
  INSIDER_HIT_THRESHOLD, INSIDER_FORWARD_TICKS, INSIDER_SUBLIMITS,
  INSIDER_KILL_RATE, VE_SLOW_ALPHA, HG_SLOW_ALPHA, and IV_CHEAP_Z.
"""