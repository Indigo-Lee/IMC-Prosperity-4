"""
Round 4 Conservative Trader

Research basis:
  - HYDROGEL_PACK: wide-spread mean reversion around 10000.
  - VELVETFRUIT_EXTRACT: tighter mean reversion with real intraday drift.
  - VEV_4000 / VEV_4500: intrinsic-like delta-1 vouchers; skipped here.
  - VEV_5200..VEV_5500: option-smile residuals and dump absorption.
  - VEV_6000 / VEV_6500: flat 0/1 markets, useful only as zero-cost lottery.

The Round 3 bug is explicitly avoided: all EMAs below are scaled for the
1,000-tick competition day, and DAY_PERIOD is 100_000 timestamps.
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

# HYDROGEL_PACK: anchored fair-value market making.
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

# VELVETFRUIT_EXTRACT: EMA fair, not a fixed anchor.
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

# Vouchers.
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


# =============================================================================
# Lightweight logger
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
        return {
            "hg": {},
            "ve": {},
            "iv": {"sigma": VOL_FALLBACK, "tte_days": 4.0, "resid": {}},
            "acct": {"cash": {}, "traded": {}},
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
        """Append an order while respecting exchange gross buy/sell limits."""
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
        """Scale down only when adding to already-heavy inventory."""
        if direction > 0 and position > soft_limit:
            scale = max(0.10, (limit - position) / max(1, limit - soft_limit))
        elif direction < 0 and position < -soft_limit:
            scale = max(0.10, (limit + position) / max(1, limit - soft_limit))
        else:
            scale = 1.0
        return max(0, int(base_size * scale))

    def update_accounting(self, state: TradingState, data: Dict[str, Any]) -> None:
        """Approximate product PnL attribution from own trade cash plus current mid."""
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
    # Delta-1 products
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

        if anchored:
            core = 0.50 * slow + 0.50 * anchor
        else:
            core = slow
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

        # Take only when book is meaningfully through fair.
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

    # ------------------------------------------------------------------
    # Option model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if spot <= 0 or strike <= 0 or tte <= 0 or sigma <= 0:
            return max(spot - strike, 0.0)
        vol = sigma * math.sqrt(tte)
        if vol <= 0:
            return max(spot - strike, 0.0)
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
                std = math.sqrt(max(var, 1e-12))
                zscores[symbol] = (resid - mean) / std
        return zscores

    # ------------------------------------------------------------------
    # Voucher blocks
    # ------------------------------------------------------------------

    def run_vouchers(self, state: TradingState, orders: Dict[Symbol, List[Order]], data: Dict[str, Any]) -> None:
        spot = self.mid(state, VE)
        if spot is None:
            return
        iv_data = data.setdefault("iv", {"sigma": VOL_FALLBACK, "tte_days": 4.0, "resid": {}})
        sigma, tte = self.estimate_sigma_tte(state, spot, iv_data)
        zscores = self.residual_zscores(state, data, spot, tte)

        # Conservative residual trade: buy cheap vouchers, sell only to reduce longs.
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
                qty = min(ask_qty, IV_TAKE_SIZE, IV_LONG_SUBLIMIT - pos)
                self.add_order(state, orders, symbol, ask, qty)
            elif z >= IV_EXIT_Z and pos > 0:
                qty = min(bid_qty, IV_TAKE_SIZE, pos)
                self.add_order(state, orders, symbol, bid, -qty)

            # Passive dump absorption, only when the smile says the option is not rich.
            if z <= 0.25 and pos < IV_LONG_SUBLIMIT:
                qty = min(IV_PASSIVE_SIZE, IV_LONG_SUBLIMIT - pos)
                self.add_order(state, orders, symbol, bid, qty)

            logger.print(symbol, f"z={z:.2f}", f"pos={pos}")

        # Zero-cost lottery: only bid 0, never pay 1 in the conservative file.
        for symbol in FREE_CALL_VOUCHERS:
            pos = state.position.get(symbol, 0)
            if pos < FREE_CALL_LIMIT:
                self.add_order(state, orders, symbol, FREE_CALL_BID, min(FREE_CALL_SIZE, FREE_CALL_LIMIT - pos))

        # The deep ITM vouchers behave like delta-1 intrinsic products. We skip them
        # rather than create extra unhedged VE exposure with a very wide spread.
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
    # Main
    # ------------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        data = self.load_data(state.traderData)
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        self.update_accounting(state, data)

        self.run_delta_mm(state, orders, data, HG)
        self.run_delta_mm(state, orders, data, VE)
        self.run_vouchers(state, orders, data)

        for symbol in state.order_depths:
            orders.setdefault(symbol, [])

        self.log_attribution(state, data)
        trader_data = json.dumps(data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


"""
Summary:
  This conservative trader is mostly a market maker. HYDROGEL uses a 10k anchor
  plus fast/slow EMAs; VELVETFRUIT uses a faster EMA fair that can follow the
  Round 4 day-scale drift. Vouchers use only long-side residual trades, passive
  dump absorption, and zero-cost bids on VEV_6000/6500.

Key risks:
  EMA fair can still lag a violent one-way move. Voucher residuals can be noisy
  late in expiry, and passive bids can accumulate inventory faster than expected.
  The delta governor is intentionally small and may not fully neutralize options.

Most sensitive constants:
  HG_SLOW_ALPHA, VE_SLOW_ALPHA, HG_BOUND, VE_BOUND, IV_CHEAP_Z,
  IV_LONG_SUBLIMIT, FREE_CALL_LIMIT, and DELTA_HEDGE_TRIGGER.
"""