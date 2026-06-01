"""
Round 4 Conservative Trader — v2 (post-OOS-loss redesign)

ANTI-OVERFIT POSTURE
  v1 lost -$19.9k on the 10%-sample backtest. Per-product attribution showed:
    - HG MM:                      -$448   (~flat, structurally sound)
    - VE MM:                      +$2,091 (positive)
    - voucher IV-residual MR:     -$21,419 (overfit parabolic smile)
    - free-call lottery:          -$100
  The MM blocks worked. The voucher engine fit a 4-strike parabola to noisy
  IV data and the residuals were noise — z-score >=1.5 trades were random.
  v2 deletes the entire voucher pricing/IV/sigma/TTE stack.

  Surviving signals are STRUCTURAL, not pattern-fit:
    - HG MM around 10,000 anchor       (anchor is fixed across rounds)
    - VE MM around drifting EMA fair    (EMA tracks regime, not parameters)
    - Free-call lottery passive bid     (asymmetric payoff, not statistical)

  No counterparty IDs. No fitted vol surface. No statistical priors.

POSITION RISK
  HG_HARD_CAP and VE_HARD_CAP cut inventory build below the IMC limits, so
  we never sit at full ±200 exposure into a trend.
"""

from __future__ import annotations

import json
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

    def compress_state(self, state, trader_data):
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, depths):
        return {s: [d.buy_orders, d.sell_orders] for s, d in depths.items()}

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

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = value[:mid]
            if len(cand) < len(value):
                cand += "..."
            if len(json.dumps(cand)) <= max_length:
                out = cand; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# =============================================================================
# CONSTANTS (retune block)
# =============================================================================
DAY_PERIOD = 100_000

# --- HYDROGEL_PACK MM ---
HG = "HYDROGEL_PACK"
HG_LIMIT       = 200
HG_HARD_CAP    = 120         # NEW: stop accumulating beyond ±120 even though limit is ±200
HG_FAIR_ANCHOR = 10_000
HG_SLOW_A      = 0.01
HG_FAST_A      = 0.30
HG_BOUND       = 80          # was 200 — tighter so EMA can't drift far from anchor
HG_TREND_CLAMP = 4.0         # was 6 — tighter trend term
HG_TAKE_W      = 3
HG_JOIN_EDGE   = 1
HG_DEFAULT_EDGE = 3
HG_SOFT        = 30          # taper kicks in earlier

# --- VELVETFRUIT_EXTRACT MM ---
VE = "VELVETFRUIT_EXTRACT"
VE_LIMIT       = 200
VE_HARD_CAP    = 120
VE_FAIR_INIT   = 5253
VE_SLOW_A      = 0.005
VE_BOUND       = 60
VE_TAKE_W      = 1
VE_JOIN_EDGE   = 2
VE_DEFAULT_EDGE = 4
VE_SOFT        = 25

# --- Free-call lottery (asymmetric, structural) ---
VEV_FREE       = ["VEV_6000", "VEV_6500"]
VEV_FREE_LIMIT = 30          # was 50; smaller so a -$1 mark on each costs <=$30
VEV_FREE_BID   = 1


class Trader:

    # ----- helpers --------------------------------------------------------
    def get_mid(self, state, sym):
        d = state.order_depths.get(sym)
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2

    @staticmethod
    def size_with_taper(want, position, hard_cap, soft):
        """Return signed qty respecting hard inventory cap and soft taper."""
        if want > 0:
            cap = hard_cap - position
        else:
            cap = hard_cap + position
        if cap <= 0:
            return 0
        excess = max(0, abs(position) - soft)
        scale = max(0.0, 1.0 - excess / max(1, hard_cap - soft))
        return int(min(cap, abs(want) * scale)) * (1 if want > 0 else -1)

    # ----- HG MM (mean-reverting around anchor) ---------------------------
    def hg_run(self, state, hg_data, orders):
        if HG not in state.order_depths:
            return
        depth = state.order_depths[HG]
        if not depth.buy_orders or not depth.sell_orders:
            return
        mid = self.get_mid(state, HG)
        if mid is None:
            return

        slow = hg_data.get("slow", mid) + HG_SLOW_A * (mid - hg_data.get("slow", mid))
        fast = hg_data.get("fast", mid) + HG_FAST_A * (mid - hg_data.get("fast", mid))
        hg_data["slow"], hg_data["fast"] = slow, fast

        anchored = (slow + HG_FAIR_ANCHOR) / 2.0
        anchored = max(HG_FAIR_ANCHOR - HG_BOUND, min(HG_FAIR_ANCHOR + HG_BOUND, anchored))
        trend = max(-HG_TREND_CLAMP, min(HG_TREND_CLAMP, fast - slow))
        fair = round(anchored + trend)

        position = state.position.get(HG, 0)
        hg_orders: List[Order] = []

        # take when book is well inside fair, capped by HG_HARD_CAP
        buy_room = max(0, HG_HARD_CAP - position)
        sell_room = max(0, HG_HARD_CAP + position)
        for price in sorted(depth.sell_orders):
            if buy_room <= 0 or price > fair - HG_TAKE_W:
                break
            qty = min(buy_room, -depth.sell_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, qty)); buy_room -= qty
        for price in sorted(depth.buy_orders, reverse=True):
            if sell_room <= 0 or price < fair + HG_TAKE_W:
                break
            qty = min(sell_room, depth.buy_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, -qty)); sell_room -= qty

        # passive quotes — tapered size
        if buy_room > 0:
            best_bid = max(depth.buy_orders.keys())
            bid = min(fair - HG_DEFAULT_EDGE, best_bid + HG_JOIN_EDGE)
            if position < -HG_SOFT:
                bid += 1
            tb = self.size_with_taper(buy_room, position, HG_HARD_CAP, HG_SOFT)
            if tb > 0:
                hg_orders.append(Order(HG, bid, tb))
        if sell_room > 0:
            best_ask = min(depth.sell_orders.keys())
            ask = max(fair + HG_DEFAULT_EDGE, best_ask - HG_JOIN_EDGE)
            if position > HG_SOFT:
                ask -= 1
            ts = self.size_with_taper(-sell_room, position, HG_HARD_CAP, HG_SOFT)
            if ts < 0:
                hg_orders.append(Order(HG, ask, ts))

        if hg_orders:
            orders.setdefault(HG, []).extend(hg_orders)

    # ----- VE MM (drifting EMA fair) --------------------------------------
    def ve_run(self, state, ve_data, orders):
        if VE not in state.order_depths:
            return
        depth = state.order_depths[VE]
        if not depth.buy_orders or not depth.sell_orders:
            return
        mid = self.get_mid(state, VE)
        if mid is None:
            return

        slow = ve_data.get("slow", VE_FAIR_INIT) + VE_SLOW_A * (mid - ve_data.get("slow", VE_FAIR_INIT))
        ve_data["slow"] = slow
        fair = round(max(VE_FAIR_INIT - VE_BOUND, min(VE_FAIR_INIT + VE_BOUND, slow)))

        position = state.position.get(VE, 0)
        ve_orders: List[Order] = []
        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())

        buy_room = max(0, VE_HARD_CAP - position)
        sell_room = max(0, VE_HARD_CAP + position)

        if best_ask <= fair - VE_TAKE_W and buy_room > 0:
            qty = min(buy_room, -depth.sell_orders[best_ask])
            if qty > 0:
                ve_orders.append(Order(VE, best_ask, qty)); buy_room -= qty
        if best_bid >= fair + VE_TAKE_W and sell_room > 0:
            qty = min(sell_room, depth.buy_orders[best_bid])
            if qty > 0:
                ve_orders.append(Order(VE, best_bid, -qty)); sell_room -= qty

        bid = min(fair - VE_DEFAULT_EDGE, best_bid + VE_JOIN_EDGE)
        ask = max(fair + VE_DEFAULT_EDGE, best_ask - VE_JOIN_EDGE)
        if position > VE_SOFT:
            ask -= 1
        elif position < -VE_SOFT:
            bid += 1
        if bid >= ask:
            ask = bid + 1

        tb = self.size_with_taper(buy_room, position, VE_HARD_CAP, VE_SOFT)
        if tb > 0:
            ve_orders.append(Order(VE, bid, tb))
        ts = self.size_with_taper(-sell_room, position, VE_HARD_CAP, VE_SOFT)
        if ts < 0:
            ve_orders.append(Order(VE, ask, ts))

        if ve_orders:
            orders.setdefault(VE, []).extend(ve_orders)

    # ----- Free-call lottery (structural asymmetric bet) ------------------
    def free_call_lottery(self, state, orders):
        for sym in VEV_FREE:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            position = state.position.get(sym, 0)
            if position >= VEV_FREE_LIMIT:
                continue
            qty = min(VEV_FREE_LIMIT - position, 5)
            if qty > 0:
                orders.setdefault(sym, []).append(Order(sym, VEV_FREE_BID, qty))

    # ----- Main entry -----------------------------------------------------
    def run(self, state: TradingState):
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

        self.hg_run(state, hg_data, orders)
        self.ve_run(state, ve_data, orders)
        self.free_call_lottery(state, orders)

        trader_data = json.dumps(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


# =============================================================================
# Plain-English summary
# =============================================================================
"""
WHAT THIS ALGO DOES (v2)
  - HG MM: passive quotes around (slow EMA + 10000 anchor)/2, bound ±80, with
    inventory taper at ±30 and HARD CAP at ±120. Takes when book is 3+ ticks
    inside fair. Hard cap means we can't accumulate to full ±200 — preserves
    capacity to flatten if a trend persists.
  - VE MM: passive quotes around drifting EMA fair (anchor 5253 ± 60), narrow
    edges (1 take, 4 default), HARD CAP ±120.
  - Free-call lottery: passive bid at price 1 on VEV_6000/6500, capped at 30.

ANTI-OVERFIT GUARANTEES
  - No fitted vol surface. No IV-residual mean reversion. No counterparty
    pattern matching. No statistical priors.
  - All thresholds are anchor-based, not historical-percentile-based.
  - Hard inventory caps below position limit prevent trend wipeouts.

KEY RISKS
  - If HG or VE trends >80 ticks from anchor for an extended period, the EMA
    fair clips at the boundary and we keep selling/buying into the trend.
    Mitigation: HG_BOUND=80 is wider than the day-3 sample's HG range (range
    was ~70 ticks). If this proves too tight live, raise to 120.
  - VE drifted from 5295 → 5253 in the day-3 sample (-42 ticks). VE_BOUND=60
    is enough. If VE drifts beyond ±60 in production, bid-take side will sit
    at the boundary; raise to 100 if needed.

MOST SENSITIVE CONSTANTS
  1. HG_BOUND / VE_BOUND     — widen if the price drifts beyond the band live.
  2. HG_HARD_CAP / VE_HARD_CAP — tighten (e.g. 80) if trend wipeouts persist;
     loosen (e.g. 160) if we're missing fills near limits in calm regimes.
  3. HG_SLOW_A / VE_SLOW_A   — already 10x scale-fixed for 1000-tick days.
  4. VEV_FREE_LIMIT          — cap on lottery accumulation; 30 limits worst
     case to ~$30 if all marked at 0.
"""