"""
Round 4 Conservative Trader

Validated signals only:
  - HYDROGEL_PACK MM around 10,000 anchor (mean-reverting, AR(1) b ~ -0.13)
  - VELVETFRUIT_EXTRACT MM around drifting EMA fair (random-walk-friendly)
  - VEV active strikes (5100-5400): IV-residual mean reversion via parabolic smile
  - VEV out-of-money (6000, 6500): free-call lottery via passive bid at 1
  - Deep ITM strikes (4000, 4500): SKIP (pinned at intrinsic, no edge)

R3 calibration fixes applied:
  - All EMA alphas scaled ~10x for 1000-tick competition days (was 10000-tick local)
  - DAY_PERIOD = 100,000 timestamps per day
  - No daily-expiry assumption; vouchers expire at round end only

Position management: every block tapers order size as inventory approaches limits.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# =============================================================================
# Logger (verbatim from r4_algo_sample.py)
# =============================================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [d.buy_orders, d.sell_orders] for s, d in order_depths.items()}

    def compress_trades(self, trades):
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, observations):
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [obs.bidPrice, obs.askPrice, obs.transportFees, obs.exportTariff,
                           obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = value[:mid]
            if len(cand) < len(value):
                cand += "..."
            if len(json.dumps(cand)) <= max_length:
                out = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# =============================================================================
# CONSTANTS (retune block)
# =============================================================================

# Competition tick scale — 1000 ticks/day, 100 timestamp units per tick
DAY_PERIOD = 100_000

# --- HYDROGEL_PACK ---
HG = "HYDROGEL_PACK"
HG_LIMIT = 200
HG_FAIR_ANCHOR = 10_000
HG_SLOW_A = 0.01           # was 0.001 in R3 — 10x scale fix
HG_FAST_A = 0.30           # was 0.08
HG_BOUND  = 200            # max EMA deviation from anchor (was 30)
HG_TREND_CLAMP = 6.0       # max trend term magnitude
HG_TAKE_W = 3              # take edge in ticks
HG_JOIN_EDGE = 1
HG_DEFAULT_EDGE = 3
HG_SOFT = 40               # inventory taper threshold

# --- VELVETFRUIT_EXTRACT ---
VE = "VELVETFRUIT_EXTRACT"
VE_LIMIT = 200
VE_FAIR_INIT = 5253        # observed center of mass across days 1-3
VE_SLOW_A = 0.005          # was 0.0005 — 10x scale fix
VE_BOUND  = 60             # was 15 — wider band so EMA can drift across the round
VE_TAKE_W = 1
VE_JOIN_EDGE = 2
VE_DEFAULT_EDGE = 4
VE_SOFT = 30

# --- VEV vouchers ---
VEV_LIMIT = 300
VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
VEV_ACTIVE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400"]   # IV-residual MR universe
VEV_FREE   = ["VEV_6000", "VEV_6500"]                            # free-call lottery universe

# Black-Scholes / vol surface
VE_YEAR_DAYS = 365.0
VE_TTE_CANDIDATES_DAYS = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
VE_FALLBACK_VOL = 0.25
VE_MIN_VOL = 0.05
VE_MAX_VOL = 1.00
VE_SIGMA_SMOOTH = 0.35

# IV-residual mean reversion
IV_RES_HISTORY_LEN = 200
IV_RES_Z_ENTRY = 1.5
IV_RES_Z_EXIT  = 0.3
IV_RES_SUBLIMIT = 80       # per voucher position cap from this signal alone
IV_RES_TAKE_SIZE = 20

# Free-call lottery (zero-cost OTM)
VEV_FREE_LIMIT = 50        # smaller than findings' 300 to be safe
VEV_FREE_BID = 1

# =============================================================================
# Trader
# =============================================================================
class Trader:

    # ----- generic helpers ------------------------------------------------

    def get_mid(self, state: TradingState, sym: Symbol) -> float | None:
        d = state.order_depths.get(sym)
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2

    @staticmethod
    def size_with_taper(want: int, position: int, limit: int, soft: int) -> int:
        """Reduce desired qty as |position| approaches limit; never exceed limit."""
        if want > 0:
            cap = limit - position
        else:
            cap = limit + position
        if cap <= 0:
            return 0
        # soft taper: above |soft|, scale down linearly
        excess = max(0, abs(position) - soft)
        scale = max(0.0, 1.0 - excess / max(1, limit - soft))
        return int(min(cap, abs(want) * scale)) * (1 if want > 0 else -1)

    # ----- HYDROGEL_PACK MM -----------------------------------------------

    def hg_run(self, state: TradingState, hg_data: Dict[str, Any], orders: Dict[Symbol, List[Order]]) -> None:
        if HG not in state.order_depths:
            return
        depth = state.order_depths[HG]
        if not depth.buy_orders or not depth.sell_orders:
            return
        mid = self.get_mid(state, HG)
        if mid is None:
            return

        # EMA fair: slow EMA toward the 10000 anchor, plus trend term clamped
        slow = hg_data.get("slow", mid)
        fast = hg_data.get("fast", mid)
        slow = slow + HG_SLOW_A * (mid - slow)
        fast = fast + HG_FAST_A * (mid - fast)
        hg_data["slow"], hg_data["fast"] = slow, fast

        # Anchored fair: average of slow EMA and 10k anchor, bounded
        anchored = (slow + HG_FAIR_ANCHOR) / 2.0
        anchored = max(HG_FAIR_ANCHOR - HG_BOUND, min(HG_FAIR_ANCHOR + HG_BOUND, anchored))
        # Trend term, clamped
        trend = max(-HG_TREND_CLAMP, min(HG_TREND_CLAMP, fast - slow))
        fair = round(anchored + trend)

        position = state.position.get(HG, 0)
        buy_cap = HG_LIMIT - position
        sell_cap = HG_LIMIT + position
        hg_orders: List[Order] = []

        # TAKE: cross book when prices are well inside fair
        for price in sorted(depth.sell_orders):
            if buy_cap <= 0 or price > fair - HG_TAKE_W:
                break
            qty = min(buy_cap, -depth.sell_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, qty))
                buy_cap -= qty
        for price in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0 or price < fair + HG_TAKE_W:
                break
            qty = min(sell_cap, depth.buy_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, -qty))
                sell_cap -= qty

        # MAKE: passive quotes, taper near limits, bias if inventory heavy
        if buy_cap > 0:
            best_bid = max(depth.buy_orders.keys())
            bid = min(fair - HG_DEFAULT_EDGE, best_bid + HG_JOIN_EDGE)
            if position < -HG_SOFT:
                bid += 1   # be more aggressive buying when short
            tapered = self.size_with_taper(buy_cap, position, HG_LIMIT, HG_SOFT)
            if tapered > 0:
                hg_orders.append(Order(HG, bid, tapered))
        if sell_cap > 0:
            best_ask = min(depth.sell_orders.keys())
            ask = max(fair + HG_DEFAULT_EDGE, best_ask - HG_JOIN_EDGE)
            if position > HG_SOFT:
                ask -= 1
            tapered = self.size_with_taper(-sell_cap, position, HG_LIMIT, HG_SOFT)
            if tapered < 0:
                hg_orders.append(Order(HG, ask, tapered))

        if hg_orders:
            existing = orders.get(HG, [])
            orders[HG] = existing + hg_orders

    # ----- VELVETFRUIT_EXTRACT MM -----------------------------------------

    def ve_run(self, state: TradingState, ve_data: Dict[str, Any], orders: Dict[Symbol, List[Order]]) -> None:
        if VE not in state.order_depths:
            return
        depth = state.order_depths[VE]
        if not depth.buy_orders or not depth.sell_orders:
            return
        mid = self.get_mid(state, VE)
        if mid is None:
            return

        slow = ve_data.get("slow", VE_FAIR_INIT)
        slow = slow + VE_SLOW_A * (mid - slow)
        ve_data["slow"] = slow
        fair = round(max(VE_FAIR_INIT - VE_BOUND, min(VE_FAIR_INIT + VE_BOUND, slow)))

        position = state.position.get(VE, 0)
        buy_cap = VE_LIMIT - position
        sell_cap = VE_LIMIT + position
        ve_orders: List[Order] = []

        # TAKE
        best_ask = min(depth.sell_orders.keys())
        if best_ask <= fair - VE_TAKE_W and buy_cap > 0:
            qty = min(buy_cap, -depth.sell_orders[best_ask])
            if qty > 0:
                ve_orders.append(Order(VE, best_ask, qty))
                buy_cap -= qty
        best_bid = max(depth.buy_orders.keys())
        if best_bid >= fair + VE_TAKE_W and sell_cap > 0:
            qty = min(sell_cap, depth.buy_orders[best_bid])
            if qty > 0:
                ve_orders.append(Order(VE, best_bid, -qty))
                sell_cap -= qty

        # MAKE
        bid = min(fair - VE_DEFAULT_EDGE, best_bid + VE_JOIN_EDGE)
        ask = max(fair + VE_DEFAULT_EDGE, best_ask - VE_JOIN_EDGE)
        if position > VE_SOFT:
            ask -= 1
        elif position < -VE_SOFT:
            bid += 1
        if bid >= ask:
            ask = bid + 1

        tapered_buy = self.size_with_taper(buy_cap, position, VE_LIMIT, VE_SOFT)
        if tapered_buy > 0:
            ve_orders.append(Order(VE, bid, tapered_buy))
        tapered_sell = self.size_with_taper(-sell_cap, position, VE_LIMIT, VE_SOFT)
        if tapered_sell < 0:
            ve_orders.append(Order(VE, ask, tapered_sell))

        if ve_orders:
            existing = orders.get(VE, [])
            orders[VE] = existing + ve_orders

    # ----- Black-Scholes helpers ------------------------------------------

    @staticmethod
    def normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot: float, strike: float, tte: float, sigma: float) -> float:
        if tte <= 0 or sigma <= 0:
            return max(spot - strike, 0.0)
        v = sigma * math.sqrt(tte)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / v
        d2 = d1 - v
        return spot * self.normal_cdf(d1) - strike * self.normal_cdf(d2)

    def implied_vol(self, price: float, spot: float, strike: float, tte: float) -> float | None:
        if price <= max(spot - strike, 0.0) + 0.01:
            return None
        lo, hi = VE_MIN_VOL, VE_MAX_VOL
        if price < self.bs_call(spot, strike, tte, lo) or price > self.bs_call(spot, strike, tte, hi):
            return None
        for _ in range(35):
            mid = (lo + hi) / 2
            if self.bs_call(spot, strike, tte, mid) < price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def estimate_sigma_tte(self, state: TradingState, spot: float, ve_data: Dict[str, Any]) -> Tuple[float, float]:
        """Pick the (sigma, TTE) pair that makes the smile flattest. Smoothed across ticks."""
        prev_sigma = ve_data.get("sigma", VE_FALLBACK_VOL)
        best_sigma, best_tte_d, best_score = prev_sigma, ve_data.get("tte_days", 6.0), float("inf")

        for tte_d in VE_TTE_CANDIDATES_DAYS:
            tte = tte_d / VE_YEAR_DAYS
            ivs: List[float] = []
            for sym, k in VEV_STRIKES.items():
                m = self.get_mid(state, sym)
                if m is None:
                    continue
                iv = self.implied_vol(m, spot, k, tte)
                if iv is not None and VE_MIN_VOL <= iv <= VE_MAX_VOL:
                    ivs.append(iv)
            if len(ivs) < 3:
                continue
            ivs.sort()
            med = ivs[len(ivs) // 2]
            mad = sum(abs(v - med) for v in ivs) / len(ivs)
            score = mad + 0.35 * abs(med - prev_sigma)
            if score < best_score:
                best_score, best_sigma, best_tte_d = score, med, tte_d

        sigma = (1.0 - VE_SIGMA_SMOOTH) * prev_sigma + VE_SIGMA_SMOOTH * best_sigma
        ve_data["sigma"], ve_data["tte_days"] = sigma, best_tte_d
        return sigma, best_tte_d / VE_YEAR_DAYS

    # ----- Voucher IV-residual mean reversion -----------------------------

    def voucher_iv_residuals(self, state: TradingState, ve_data: Dict[str, Any], orders: Dict[Symbol, List[Order]]) -> None:
        spot = self.get_mid(state, VE)
        if spot is None:
            return

        sigma, tte = self.estimate_sigma_tte(state, spot, ve_data)
        if tte <= 0 or sigma <= 0:
            return

        # Collect (moneyness, iv) for active strikes
        obs: List[Tuple[str, float, float]] = []  # (sym, m, iv)
        for sym in VEV_ACTIVE:
            mid = self.get_mid(state, sym)
            if mid is None:
                continue
            iv = self.implied_vol(mid, spot, VEV_STRIKES[sym], tte)
            if iv is None:
                continue
            m = math.log(VEV_STRIKES[sym] / spot)
            obs.append((sym, m, iv))

        if len(obs) < 3:
            return

        # Fit parabola iv = a + b*m + c*m^2 by least squares (manual normal eqs)
        n = len(obs)
        sm = sum(o[1] for o in obs)
        sm2 = sum(o[1] ** 2 for o in obs)
        sm3 = sum(o[1] ** 3 for o in obs)
        sm4 = sum(o[1] ** 4 for o in obs)
        sy = sum(o[2] for o in obs)
        smy = sum(o[1] * o[2] for o in obs)
        sm2y = sum(o[1] ** 2 * o[2] for o in obs)

        # Solve 3x3:
        # [n   sm  sm2 ] [a]   [sy ]
        # [sm  sm2 sm3 ] [b] = [smy]
        # [sm2 sm3 sm4 ] [c]   [sm2y]
        det = (n * (sm2 * sm4 - sm3 * sm3)
               - sm * (sm * sm4 - sm3 * sm2)
               + sm2 * (sm * sm3 - sm2 * sm2))
        if abs(det) < 1e-12:
            return
        a = (sy * (sm2 * sm4 - sm3 * sm3)
             - sm * (smy * sm4 - sm3 * sm2y)
             + sm2 * (smy * sm3 - sm2 * sm2y)) / det
        b = (n * (smy * sm4 - sm3 * sm2y)
             - sy * (sm * sm4 - sm3 * sm2)
             + sm2 * (sm * sm2y - smy * sm2)) / det
        c = (n * (sm2 * sm2y - smy * sm3)
             - sm * (sm * sm2y - smy * sm2)
             + sy * (sm * sm3 - sm2 * sm2)) / det

        # For each active strike, compute residual and z-score
        hist = ve_data.setdefault("iv_resid_hist", {sym: [] for sym in VEV_ACTIVE})
        for sym, m, iv in obs:
            fitted = a + b * m + c * m * m
            resid = iv - fitted
            arr = hist.setdefault(sym, [])
            arr.append(resid)
            if len(arr) > IV_RES_HISTORY_LEN:
                arr.pop(0)
            if len(arr) < 30:
                continue

            mean = sum(arr) / len(arr)
            var = sum((r - mean) ** 2 for r in arr) / len(arr)
            std = math.sqrt(max(var, 1e-12))
            z = (resid - mean) / std

            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            position = state.position.get(sym, 0)
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            opt_orders: List[Order] = []

            # iv too high -> voucher rich -> short
            if z > IV_RES_Z_ENTRY and position > -IV_RES_SUBLIMIT:
                qty = min(IV_RES_TAKE_SIZE,
                          IV_RES_SUBLIMIT + position,
                          depth.buy_orders[best_bid])
                if qty > 0:
                    opt_orders.append(Order(sym, best_bid, -qty))
            # iv too low -> voucher cheap -> long
            elif z < -IV_RES_Z_ENTRY and position < IV_RES_SUBLIMIT:
                qty = min(IV_RES_TAKE_SIZE,
                          IV_RES_SUBLIMIT - position,
                          -depth.sell_orders[best_ask])
                if qty > 0:
                    opt_orders.append(Order(sym, best_ask, qty))
            # exit when residual normalizes
            elif abs(z) < IV_RES_Z_EXIT and position != 0:
                if position > 0:
                    qty = min(position, depth.buy_orders[best_bid])
                    if qty > 0:
                        opt_orders.append(Order(sym, best_bid, -qty))
                else:
                    qty = min(-position, -depth.sell_orders[best_ask])
                    if qty > 0:
                        opt_orders.append(Order(sym, best_ask, qty))

            if opt_orders:
                existing = orders.get(sym, [])
                orders[sym] = existing + opt_orders

    # ----- Free-call lottery (VEV_6000, VEV_6500) -------------------------

    def free_call_lottery(self, state: TradingState, orders: Dict[Symbol, List[Order]]) -> None:
        for sym in VEV_FREE:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            position = state.position.get(sym, 0)
            if position >= VEV_FREE_LIMIT:
                continue
            # Passive bid at price 1; if bots dump (0.5 mid implies fills at 0/1 occasionally), we collect.
            qty = VEV_FREE_LIMIT - position
            qty = min(qty, 5)   # small per-tick to avoid stacking the queue too far
            if qty > 0:
                existing = orders.get(sym, [])
                orders[sym] = existing + [Order(sym, VEV_FREE_BID, qty)]

    # ----- main entry ------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}
        else:
            data = {}

        hg_data = data.setdefault("hg", {})
        ve_data = data.setdefault("ve", {})

        # Block A: HYDROGEL_PACK MM (mean-reverting around 10k anchor)
        self.hg_run(state, hg_data, orders)
        # Block B: VELVETFRUIT_EXTRACT MM (drifting EMA fair)
        self.ve_run(state, ve_data, orders)
        # Block C: voucher IV-residual mean reversion (active strikes only)
        self.voucher_iv_residuals(state, ve_data, orders)
        # Block D: free-call lottery on OTM strikes
        self.free_call_lottery(state, orders)

        trader_data = json.dumps(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


# =============================================================================
# Plain-English summary
# =============================================================================
"""
WHAT THIS ALGO DOES
  - Block A (HG MM): keeps a slow + fast EMA of HYDROGEL mid; fair = avg(slow EMA, 10k)
    bounded ±200, plus a clamped trend term. Takes when book is 3+ ticks inside fair,
    quotes passively otherwise. Inventory-tapered sizing.
  - Block B (VE MM): slow EMA fair anchored at 5253, bound ±60. Same take/make shape,
    narrower edges (1 take, 4 default).
  - Block C (Voucher IV-residual MR): every tick, fits parabolic smile across active
    strikes (5100-5400), computes residual vs fit, z-scores against rolling 200-tick
    history, enters at |z|>=1.5, exits at |z|<=0.3. Sub-limit 80 per voucher.
  - Block D (Free-call lottery): passive bids at price 1 on VEV_6000/6500 up to 50 each.

KEY RISKS
  - VE EMA could pick up a long downtrend and drift below VE_FAIR_INIT-VE_BOUND,
    causing late MM. Mitigation: VE_BOUND is generous (60) given empirical drift ~10/day.
  - IV-residual fit relies on >=3 mid-pricing strikes; on illiquid ticks block C silently
    skips. Acceptable.
  - Free-call lottery captures negligible $ but could snowball position over many days;
    cap of 50 per strike limits loss to ~50.

MOST SENSITIVE CONSTANTS (in order of priority)
  1. HG_SLOW_A / VE_SLOW_A   — increase (0.02 / 0.01) if EMA lags real moves; decrease if noisy.
  2. IV_RES_Z_ENTRY          — start 1.5; lower to 1.0 if vouchers under-trade, raise to 2.0 if over.
  3. HG_BOUND / VE_BOUND     — bound on EMA deviation from anchor; widen if live drift exceeds.
  4. IV_RES_SUBLIMIT          — cap on voucher position from this signal alone.
  5. VEV_FREE_LIMIT          — free-call accumulation cap.
"""