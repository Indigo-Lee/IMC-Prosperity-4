"""
Round 4 Aggressive Trader

Conservative MM/IV stack + Block F counterparty copy/fade layer.

Block F detects insiders and "anti-insiders" by tracking each counterparty's hit
rate vs forward mid (horizon = 1000 timestamp units = 10 ticks). Traders with
hit rate >=0.65 over >=20 trades are FOLLOWED; <=0.35 are FADED (mirror their
losing trades). A trailing-30 window kill switch demotes degrading followers.

Counterparty orders use a per-symbol sub-limit smaller than the MM position
caps so the MM blocks still have working capital. After Block F submits, the
MM blocks see effective_position = state.position + block_F_pending_qty so
they don't push the combined position past the IMC limit.

Seeded priors (from offline analysis on r4_hist/) bootstrap the tracker so we
don't trade blind for the first ~50 trades. If the priors are wrong the
runtime tracker overwrites them quickly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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


# =============================================================================
# CONSTANTS — conservative base
# =============================================================================
DAY_PERIOD = 100_000

HG = "HYDROGEL_PACK"
HG_LIMIT = 200
HG_FAIR_ANCHOR = 10_000
HG_SLOW_A = 0.01
HG_FAST_A = 0.30
HG_BOUND  = 200
HG_TREND_CLAMP = 6.0
HG_TAKE_W = 3
HG_JOIN_EDGE = 1
HG_DEFAULT_EDGE = 3
HG_SOFT = 40

VE = "VELVETFRUIT_EXTRACT"
VE_LIMIT = 200
VE_FAIR_INIT = 5253
VE_SLOW_A = 0.005
VE_BOUND  = 60
VE_TAKE_W = 1
VE_JOIN_EDGE = 2
VE_DEFAULT_EDGE = 4
VE_SOFT = 30

VEV_LIMIT = 300
VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
VEV_ACTIVE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400"]
VEV_FREE   = ["VEV_6000", "VEV_6500"]

VE_YEAR_DAYS = 365.0
VE_TTE_CANDIDATES_DAYS = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
VE_FALLBACK_VOL = 0.25
VE_MIN_VOL = 0.05
VE_MAX_VOL = 1.00
VE_SIGMA_SMOOTH = 0.35

IV_RES_HISTORY_LEN = 200
IV_RES_Z_ENTRY = 1.5
IV_RES_Z_EXIT  = 0.3
IV_RES_SUBLIMIT = 80
IV_RES_TAKE_SIZE = 20

VEV_FREE_LIMIT = 50
VEV_FREE_BID = 1

# =============================================================================
# CONSTANTS — Block F (counterparty layer)
# =============================================================================
COPY_HIT_THRESHOLD     = 0.65   # min hit rate to qualify as FOLLOW
COPY_MIN_TRADES        = 20
COPY_HORIZON_TS        = 1000   # forward-mid horizon = 10 ticks
COPY_KILL_THRESHOLD    = 0.55   # demote if trailing-30 hit rate falls below
COPY_KILL_WINDOW       = 30

FADE_HIT_THRESHOLD     = 0.35   # max hit rate to qualify as FADE
FADE_MIN_TRADES        = 20

PENDING_MAX_AGE        = 5_000  # drop unscorable pending obs older than this

# Sub-limits per symbol (separate from MM caps)
SUBLIMIT: Dict[str, int] = {
    HG: 80,
    VE: 80,
    "VEV_4000": 100, "VEV_4500": 100, "VEV_5000": 100,
    "VEV_5100": 100, "VEV_5200": 100, "VEV_5300": 100,
    "VEV_5400": 100, "VEV_5500": 100,
    "VEV_6000": 100, "VEV_6500": 100,
}

POSITION_LIMITS: Dict[str, int] = {
    HG: HG_LIMIT, VE: VE_LIMIT,
    **{sym: VEV_LIMIT for sym in VEV_STRIKES},
}

# Seeded priors from offline hit-rate study on r4_hist/ days 1-3.
# (hits, n) — runtime updates these every tick.
SEED_PRIORS: Dict[str, Tuple[int, int]] = {
    "Mark 14": (87, 100),   # 86.9% hit, +$49k paper PnL
    "Mark 01": (89, 100),   # 88.7% hit, +$11k paper PnL
    "Mark 38": (7,  100),   # 6.5% hit, -$41k paper PnL — strong fade
    "Mark 22": (10, 100),   # 9.5% hit, -$3k paper — fade
    "Mark 55": (20, 100),   # 20% hit, -$16k paper — fade
}


# =============================================================================
# Trader
# =============================================================================
class Trader:

    # ----- generic helpers ------------------------------------------------

    def get_mid(self, state, sym):
        d = state.order_depths.get(sym)
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2

    @staticmethod
    def size_with_taper(want, position, limit, soft):
        if want > 0:
            cap = limit - position
        else:
            cap = limit + position
        if cap <= 0:
            return 0
        excess = max(0, abs(position) - soft)
        scale = max(0.0, 1.0 - excess / max(1, limit - soft))
        return int(min(cap, abs(want) * scale)) * (1 if want > 0 else -1)

    # ======================================================================
    # BLOCK F — counterparty copy/fade
    # ======================================================================

    def _bootstrap_tracker(self, cp_data: Dict[str, Any]) -> None:
        """First-tick init: load seeded priors."""
        if cp_data.get("seeded"):
            return
        tracker: Dict[str, Dict[str, Any]] = {}
        for name, (hits, n) in SEED_PRIORS.items():
            tracker[name] = {"hits": hits, "n": n, "recent": [1] * hits + [0] * (n - hits)}
            tracker[name]["recent"] = tracker[name]["recent"][-COPY_KILL_WINDOW:]
        cp_data["tracker"] = tracker
        cp_data["pending"] = []
        cp_data["sublimit_pos"] = {}
        cp_data["seeded"] = True

    def _update_tracker(self, cp_data: Dict[str, Any], name: str, won: bool) -> None:
        tracker = cp_data.setdefault("tracker", {})
        e = tracker.setdefault(name, {"hits": 0, "n": 0, "recent": []})
        e["n"] += 1
        if won:
            e["hits"] += 1
        e["recent"].append(1 if won else 0)
        if len(e["recent"]) > COPY_KILL_WINDOW:
            e["recent"].pop(0)

    def _classify_traders(self, cp_data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns (follow_dict, fade_dict) mapping name -> hit_rate."""
        follow: Dict[str, float] = {}
        fade: Dict[str, float] = {}
        for name, e in cp_data.get("tracker", {}).items():
            n = e["n"]
            if n < COPY_MIN_TRADES:
                continue
            rate = e["hits"] / n
            recent = e.get("recent", [])
            recent_rate = (sum(recent) / len(recent)) if recent else rate

            if rate >= COPY_HIT_THRESHOLD and recent_rate >= COPY_KILL_THRESHOLD:
                follow[name] = rate
            elif rate <= FADE_HIT_THRESHOLD and recent_rate <= (1.0 - COPY_KILL_THRESHOLD):
                fade[name] = rate
        return follow, fade

    def counterparty_layer(self, state: TradingState, cp_data: Dict[str, Any],
                           orders: Dict[Symbol, List[Order]]) -> Dict[str, int]:
        """Returns {sym: net_qty_added} so MM blocks can adjust their effective position."""
        self._bootstrap_tracker(cp_data)
        net_pos: Dict[str, int] = defaultdict(int)
        now = state.timestamp

        # 1. Score expired pending observations
        pending = cp_data.get("pending", [])
        keep: List[Dict[str, Any]] = []
        for obs in pending:
            age = now - obs["ts"]
            if age >= COPY_HORIZON_TS:
                cur_mid = self.get_mid(state, obs["sym"])
                if cur_mid is None:
                    if age >= PENDING_MAX_AGE:
                        continue   # drop stale unscorable
                    keep.append(obs)
                    continue
                won = (cur_mid > obs["price"]) if obs["side"] == "buy" else (cur_mid < obs["price"])
                self._update_tracker(cp_data, obs["name"], won)
            elif age >= PENDING_MAX_AGE:
                continue
            else:
                keep.append(obs)
        cp_data["pending"] = keep

        # 2. Reclassify
        follow, fade = self._classify_traders(cp_data)

        # 3. For each market trade this tick, queue an observation and maybe trade
        sublimit_pos: Dict[str, int] = cp_data.setdefault("sublimit_pos", {})

        for sym, trades in state.market_trades.items():
            for t in trades:
                for role, name in (("buyer", t.buyer), ("seller", t.seller)):
                    if not name:
                        continue
                    # queue observation for future scoring
                    cp_data["pending"].append({
                        "ts": t.timestamp, "sym": sym, "name": name,
                        "side": "buy" if role == "buyer" else "sell",
                        "price": t.price,
                    })

                    # decide action
                    rate = None
                    side = None
                    if name in follow:
                        rate = follow[name]
                        side = "buy" if role == "buyer" else "sell"
                    elif name in fade:
                        rate = fade[name]
                        # mirror: if fader bought, we sell
                        side = "sell" if role == "buyer" else "buy"
                    if side is None:
                        continue

                    conviction = min(1.0, abs(rate - 0.5) / 0.35)
                    if conviction <= 0:
                        continue

                    sublim = SUBLIMIT.get(sym, 50)
                    cur_sublim = sublimit_pos.get(sym, 0)
                    pos = state.position.get(sym, 0) + net_pos[sym]
                    plimit = POSITION_LIMITS.get(sym, 50)
                    base_qty = max(1, int(round(t.quantity * conviction)))

                    depth = state.order_depths.get(sym)
                    if depth is None:
                        continue

                    if side == "buy" and depth.sell_orders:
                        cap = min(sublim - cur_sublim, plimit - pos)
                        qty = max(0, min(base_qty, cap))
                        if qty > 0:
                            best_ask = min(depth.sell_orders.keys())
                            orders.setdefault(sym, []).append(Order(sym, best_ask, qty))
                            sublimit_pos[sym] = cur_sublim + qty
                            net_pos[sym] += qty
                    elif side == "sell" and depth.buy_orders:
                        cap = min(sublim + cur_sublim, plimit + pos)
                        qty = max(0, min(base_qty, cap))
                        if qty > 0:
                            best_bid = max(depth.buy_orders.keys())
                            orders.setdefault(sym, []).append(Order(sym, best_bid, -qty))
                            sublimit_pos[sym] = cur_sublim - qty
                            net_pos[sym] -= qty

        return dict(net_pos)

    # ======================================================================
    # Conservative blocks (verbatim from trader_r4_conservative)
    # ======================================================================

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
        buy_cap = HG_LIMIT - position
        sell_cap = HG_LIMIT + position
        hg_orders: List[Order] = []

        for price in sorted(depth.sell_orders):
            if buy_cap <= 0 or price > fair - HG_TAKE_W:
                break
            qty = min(buy_cap, -depth.sell_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, qty)); buy_cap -= qty
        for price in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0 or price < fair + HG_TAKE_W:
                break
            qty = min(sell_cap, depth.buy_orders[price])
            if qty > 0:
                hg_orders.append(Order(HG, price, -qty)); sell_cap -= qty

        if buy_cap > 0:
            best_bid = max(depth.buy_orders.keys())
            bid = min(fair - HG_DEFAULT_EDGE, best_bid + HG_JOIN_EDGE)
            if position < -HG_SOFT:
                bid += 1
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
            orders.setdefault(HG, []).extend(hg_orders)

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
        buy_cap = VE_LIMIT - position
        sell_cap = VE_LIMIT + position
        ve_orders: List[Order] = []

        best_ask = min(depth.sell_orders.keys())
        if best_ask <= fair - VE_TAKE_W and buy_cap > 0:
            qty = min(buy_cap, -depth.sell_orders[best_ask])
            if qty > 0:
                ve_orders.append(Order(VE, best_ask, qty)); buy_cap -= qty
        best_bid = max(depth.buy_orders.keys())
        if best_bid >= fair + VE_TAKE_W and sell_cap > 0:
            qty = min(sell_cap, depth.buy_orders[best_bid])
            if qty > 0:
                ve_orders.append(Order(VE, best_bid, -qty)); sell_cap -= qty

        bid = min(fair - VE_DEFAULT_EDGE, best_bid + VE_JOIN_EDGE)
        ask = max(fair + VE_DEFAULT_EDGE, best_ask - VE_JOIN_EDGE)
        if position > VE_SOFT:
            ask -= 1
        elif position < -VE_SOFT:
            bid += 1
        if bid >= ask:
            ask = bid + 1

        tb = self.size_with_taper(buy_cap, position, VE_LIMIT, VE_SOFT)
        if tb > 0:
            ve_orders.append(Order(VE, bid, tb))
        ts = self.size_with_taper(-sell_cap, position, VE_LIMIT, VE_SOFT)
        if ts < 0:
            ve_orders.append(Order(VE, ask, ts))
        if ve_orders:
            orders.setdefault(VE, []).extend(ve_orders)

    @staticmethod
    def normal_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, spot, strike, tte, sigma):
        if tte <= 0 or sigma <= 0:
            return max(spot - strike, 0.0)
        v = sigma * math.sqrt(tte)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / v
        d2 = d1 - v
        return spot * self.normal_cdf(d1) - strike * self.normal_cdf(d2)

    def implied_vol(self, price, spot, strike, tte):
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

    def estimate_sigma_tte(self, state, spot, ve_data):
        prev = ve_data.get("sigma", VE_FALLBACK_VOL)
        best_sigma, best_tte_d, best_score = prev, ve_data.get("tte_days", 6.0), float("inf")
        for tte_d in VE_TTE_CANDIDATES_DAYS:
            tte = tte_d / VE_YEAR_DAYS
            ivs = []
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
            score = mad + 0.35 * abs(med - prev)
            if score < best_score:
                best_score, best_sigma, best_tte_d = score, med, tte_d
        sigma = (1 - VE_SIGMA_SMOOTH) * prev + VE_SIGMA_SMOOTH * best_sigma
        ve_data["sigma"], ve_data["tte_days"] = sigma, best_tte_d
        return sigma, best_tte_d / VE_YEAR_DAYS

    def voucher_iv_residuals(self, state, ve_data, orders):
        spot = self.get_mid(state, VE)
        if spot is None:
            return
        sigma, tte = self.estimate_sigma_tte(state, spot, ve_data)
        if tte <= 0 or sigma <= 0:
            return

        obs = []
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

        n = len(obs)
        sm = sum(o[1] for o in obs); sm2 = sum(o[1]**2 for o in obs)
        sm3 = sum(o[1]**3 for o in obs); sm4 = sum(o[1]**4 for o in obs)
        sy = sum(o[2] for o in obs); smy = sum(o[1]*o[2] for o in obs)
        sm2y = sum(o[1]**2*o[2] for o in obs)
        det = (n*(sm2*sm4 - sm3*sm3) - sm*(sm*sm4 - sm3*sm2) + sm2*(sm*sm3 - sm2*sm2))
        if abs(det) < 1e-12:
            return
        a = (sy*(sm2*sm4 - sm3*sm3) - sm*(smy*sm4 - sm3*sm2y) + sm2*(smy*sm3 - sm2*sm2y)) / det
        b = (n*(smy*sm4 - sm3*sm2y) - sy*(sm*sm4 - sm3*sm2) + sm2*(sm*sm2y - smy*sm2)) / det
        c = (n*(sm2*sm2y - smy*sm3) - sm*(sm*sm2y - smy*sm2) + sy*(sm*sm3 - sm2*sm2)) / det

        hist = ve_data.setdefault("iv_resid_hist", {})
        for sym, m, iv in obs:
            fitted = a + b*m + c*m*m
            resid = iv - fitted
            arr = hist.setdefault(sym, [])
            arr.append(resid)
            if len(arr) > IV_RES_HISTORY_LEN:
                arr.pop(0)
            if len(arr) < 30:
                continue
            mean = sum(arr) / len(arr)
            var = sum((r-mean)**2 for r in arr) / len(arr)
            std = math.sqrt(max(var, 1e-12))
            z = (resid - mean) / std

            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            position = state.position.get(sym, 0)
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            opt_orders = []
            if z > IV_RES_Z_ENTRY and position > -IV_RES_SUBLIMIT:
                qty = min(IV_RES_TAKE_SIZE, IV_RES_SUBLIMIT + position, depth.buy_orders[best_bid])
                if qty > 0:
                    opt_orders.append(Order(sym, best_bid, -qty))
            elif z < -IV_RES_Z_ENTRY and position < IV_RES_SUBLIMIT:
                qty = min(IV_RES_TAKE_SIZE, IV_RES_SUBLIMIT - position, -depth.sell_orders[best_ask])
                if qty > 0:
                    opt_orders.append(Order(sym, best_ask, qty))
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
                orders.setdefault(sym, []).extend(opt_orders)

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

    # ======================================================================
    # Main entry
    # ======================================================================
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

        cp_data = data.setdefault("cp", {})
        hg_data = data.setdefault("hg", {})
        ve_data = data.setdefault("ve", {})

        # Block F: counterparty layer (runs first; updates state.position copy
        # so later MM blocks see effective inventory and don't breach limits).
        net_pos = self.counterparty_layer(state, cp_data, orders)
        if net_pos:
            for sym, dq in net_pos.items():
                state.position[sym] = state.position.get(sym, 0) + dq

        # Conservative blocks
        self.hg_run(state, hg_data, orders)
        self.ve_run(state, ve_data, orders)
        self.voucher_iv_residuals(state, ve_data, orders)
        self.free_call_lottery(state, orders)

        trader_data = json.dumps(data)
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data


# =============================================================================
# Plain-English summary
# =============================================================================
"""
WHAT THIS ALGO DOES
  Block F (NEW): counterparty layer
    - Tracks per-trader hit rate vs 10-tick forward mid.
    - Bootstraps from SEED_PRIORS so it isn't blind for the first ~50 trades.
    - FOLLOW: hit_rate >= 0.65 and trailing-30 >= 0.55 (kill switch).
    - FADE  : hit_rate <= 0.35 and trailing-30 <= 0.45 (mirror their losses).
    - Per-symbol SUBLIMIT smaller than MM caps so MM keeps working capital.
    - Conviction-scaled size: |rate-0.5|/0.35.
  Block A-D (same as conservative): HG MM, VE MM, IV-residual MR, free-call lottery.
  Block F runs first and updates the copy of state.position so MM blocks treat
  the freshly-acquired position as already in inventory.

KEY RISKS
  - Trader IDs in production may differ from r4_hist/. Seeded priors then
    contribute nothing; the runtime tracker still classifies after ~20 trades
    per name. Worst case: Block F is silent on tick 0 and warms up gradually.
  - If every classified trader is on the same side of the book at once, the
    sub-limit caps (and POSITION_LIMITS recheck in the cap math) prevent breach.
  - FADE side mirrors a counterparty's bad trade. If the fade target's bad
    streak ends, the trailing-30 kill window flips them to IGNORE within 30
    trades, after which we revert to MM-only on that name.

MOST SENSITIVE CONSTANTS
  1. COPY_HIT_THRESHOLD / FADE_HIT_THRESHOLD — start 0.65/0.35; tighten if
     too many false-positives appear (raise to 0.70/0.30).
  2. SUBLIMIT[sym] — halve any product whose Block F PnL goes negative across
     two consecutive 1k-timestamp windows in backtest.
  3. COPY_HORIZON_TS — 1000 timestamps = 10 ticks, matches the offline study.
     Shorten if signal decays faster live, lengthen if longer signal needed.
  4. Conservative-block constants (HG_SLOW_A, VE_SLOW_A, IV_RES_Z_ENTRY) —
     same as in the conservative trader.
"""