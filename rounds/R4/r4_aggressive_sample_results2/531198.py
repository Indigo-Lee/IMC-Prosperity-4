"""
Round 4 Aggressive Trader — v2 (post-OOS-loss redesign)

ANTI-OVERFIT POSTURE
  v1 lost -$28.3k on the 10%-sample backtest. The voucher IV-residual MR
  block (-$21.9k) and seeded counterparty priors fading VEV_4000 (-$5.9k)
  were the bleeders. The MM blocks were ~flat, free-call neutral.

  v2 = lean MM base (HG, VE, free-call lottery) + a counterparty layer
  with explicit anti-overfit guards:

    - NO seeded priors. The tracker must classify a counterparty purely
      from observed live trades. Day-of-data names will not be hardcoded.

    - SPLIT-SAMPLE VALIDATION. A trader is FOLLOW-classified only if BOTH
      the full-history hit rate AND the trailing-30 hit rate clear the
      threshold. Same for FADE. This filters out lucky streaks that fail
      to repeat.

    - HIGH N. Need >=100 scored observations before classifying (was 20).
      A counterparty's profile must be statistically robust before we
      commit capital.

    - TIGHT SUB-LIMITS. Per-product Block F sub-limits 30 (was 80-100).
      So even a wrong classification caps loss at ~30 * adverse-move.

    - FAST KILL SWITCH. Trailing-15 hit rate <0.55 → demote.

    - REGIME-FLIP RESET. If a trader's classification flips (FOLLOW → FADE
      or vice versa), erase their tracker and start over. Their behavior
      changed; old data is poison.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# =============================================================================
# Logger
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
# CONSTANTS — MM base (mirrors trader_r4_conservative v2)
# =============================================================================
DAY_PERIOD = 100_000

HG = "HYDROGEL_PACK"
HG_LIMIT       = 200
HG_HARD_CAP    = 120
HG_FAIR_ANCHOR = 10_000
HG_SLOW_A      = 0.01
HG_FAST_A      = 0.30
HG_BOUND       = 80
HG_TREND_CLAMP = 4.0
HG_TAKE_W      = 3
HG_JOIN_EDGE   = 1
HG_DEFAULT_EDGE = 3
HG_SOFT        = 30

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

VEV_FREE       = ["VEV_6000", "VEV_6500"]
VEV_FREE_LIMIT = 30
VEV_FREE_BID   = 1


# =============================================================================
# CONSTANTS — Block F (counterparty layer, OOS-validated)
# =============================================================================
COPY_HORIZON_TS        = 1000   # forward-mid score horizon
COPY_MIN_TRADES        = 100    # was 20 — need much more data before trusting
COPY_HIT_THRESHOLD     = 0.70   # was 0.65 — tighter
FADE_HIT_THRESHOLD     = 0.30   # was 0.35 — tighter

# Split-sample validation: trailing-30 hit rate must agree with full-history
COPY_RECENT_WINDOW     = 30
COPY_RECENT_TOL        = 0.10   # |full_rate - recent_rate| must be ≤ this
COPY_KILL_RECENT_HIT   = 0.55   # demote if trailing-15 < this
COPY_KILL_WINDOW       = 15

PENDING_MAX_AGE        = 5_000

# Per-product sub-limits (HALVED from v1)
SUBLIMIT: Dict[str, int] = {
    HG: 30, VE: 30,
    "VEV_4000": 40, "VEV_4500": 40, "VEV_5000": 40,
    "VEV_5100": 40, "VEV_5200": 40, "VEV_5300": 40,
    "VEV_5400": 40, "VEV_5500": 40,
    "VEV_6000": 40, "VEV_6500": 40,
}

POSITION_LIMITS: Dict[str, int] = {
    HG: HG_LIMIT, VE: VE_LIMIT,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
    "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
    "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}


class Trader:

    # ----- helpers --------------------------------------------------------
    def get_mid(self, state, sym):
        d = state.order_depths.get(sym)
        if d is None or not d.buy_orders or not d.sell_orders:
            return None
        return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2

    @staticmethod
    def size_with_taper(want, position, hard_cap, soft):
        if want > 0:
            cap = hard_cap - position
        else:
            cap = hard_cap + position
        if cap <= 0:
            return 0
        excess = max(0, abs(position) - soft)
        scale = max(0.0, 1.0 - excess / max(1, hard_cap - soft))
        return int(min(cap, abs(want) * scale)) * (1 if want > 0 else -1)

    # ======================================================================
    # BLOCK F — counterparty layer with OOS guards
    # ======================================================================

    def _ensure_cp(self, cp_data: Dict[str, Any]) -> None:
        cp_data.setdefault("tracker", {})       # name -> {hits, n, recent[..]}
        cp_data.setdefault("pending", [])       # list of pending observations
        cp_data.setdefault("sublimit_pos", {})  # tracked position from this layer
        cp_data.setdefault("class", {})         # name -> "follow"/"fade" — to detect flips

    def _update(self, cp_data: Dict[str, Any], name: str, won: bool) -> None:
        e = cp_data["tracker"].setdefault(name, {"hits": 0, "n": 0, "recent": []})
        e["n"] += 1
        if won:
            e["hits"] += 1
        e["recent"].append(1 if won else 0)
        # cap recent to the longer of our two windows
        max_keep = max(COPY_RECENT_WINDOW, COPY_KILL_WINDOW)
        if len(e["recent"]) > max_keep:
            e["recent"] = e["recent"][-max_keep:]

    def _classify(self, cp_data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns (follow_dict, fade_dict). Applies all anti-overfit guards."""
        follow: Dict[str, float] = {}
        fade: Dict[str, float] = {}
        for name, e in cp_data["tracker"].items():
            n = e["n"]
            if n < COPY_MIN_TRADES:
                continue
            full_rate = e["hits"] / n

            recent = e["recent"][-COPY_RECENT_WINDOW:]
            if len(recent) < COPY_RECENT_WINDOW:
                continue
            recent_rate = sum(recent) / len(recent)

            kill = e["recent"][-COPY_KILL_WINDOW:]
            kill_rate = sum(kill) / max(1, len(kill))

            # Split-sample validation: full and recent must agree
            if abs(full_rate - recent_rate) > COPY_RECENT_TOL:
                continue

            prev_class = cp_data["class"].get(name)

            if full_rate >= COPY_HIT_THRESHOLD and recent_rate >= COPY_HIT_THRESHOLD - COPY_RECENT_TOL \
                    and kill_rate >= COPY_KILL_RECENT_HIT:
                if prev_class == "fade":
                    # regime flip — wipe and require re-qualification
                    cp_data["tracker"][name] = {"hits": 0, "n": 0, "recent": []}
                    cp_data["class"].pop(name, None)
                    continue
                follow[name] = full_rate
                cp_data["class"][name] = "follow"
            elif full_rate <= FADE_HIT_THRESHOLD and recent_rate <= FADE_HIT_THRESHOLD + COPY_RECENT_TOL \
                    and kill_rate <= 1.0 - COPY_KILL_RECENT_HIT:
                if prev_class == "follow":
                    cp_data["tracker"][name] = {"hits": 0, "n": 0, "recent": []}
                    cp_data["class"].pop(name, None)
                    continue
                fade[name] = full_rate
                cp_data["class"][name] = "fade"
            else:
                # demoted — classification doesn't hold this tick
                cp_data["class"].pop(name, None)
        return follow, fade

    def counterparty_layer(self, state: TradingState, cp_data: Dict[str, Any],
                           orders: Dict[Symbol, List[Order]]) -> Dict[str, int]:
        self._ensure_cp(cp_data)
        net_pos: Dict[str, int] = defaultdict(int)
        now = state.timestamp

        # 1. Score expired pending observations
        keep: List[Dict[str, Any]] = []
        for obs in cp_data["pending"]:
            age = now - obs["ts"]
            if age >= COPY_HORIZON_TS:
                cur_mid = self.get_mid(state, obs["sym"])
                if cur_mid is None:
                    if age < PENDING_MAX_AGE:
                        keep.append(obs)
                    continue
                won = (cur_mid > obs["price"]) if obs["side"] == "buy" else (cur_mid < obs["price"])
                self._update(cp_data, obs["name"], won)
            elif age < PENDING_MAX_AGE:
                keep.append(obs)
        cp_data["pending"] = keep

        # 2. Classify
        follow, fade = self._classify(cp_data)

        # 3. Generate orders, queue new pending obs
        sublimit_pos: Dict[str, int] = cp_data["sublimit_pos"]
        for sym, trades in state.market_trades.items():
            for t in trades:
                for role, name in (("buyer", t.buyer), ("seller", t.seller)):
                    if not name:
                        continue
                    cp_data["pending"].append({
                        "ts": t.timestamp, "sym": sym, "name": name,
                        "side": "buy" if role == "buyer" else "sell",
                        "price": t.price,
                    })

                    rate = None; side = None
                    if name in follow:
                        rate = follow[name]
                        side = "buy" if role == "buyer" else "sell"
                    elif name in fade:
                        rate = fade[name]
                        side = "sell" if role == "buyer" else "buy"
                    if side is None:
                        continue

                    conviction = min(1.0, abs(rate - 0.5) / 0.30)
                    if conviction <= 0:
                        continue

                    sublim = SUBLIMIT.get(sym, 30)
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
    # MM blocks (same as conservative v2)
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
        hg_orders: List[Order] = []
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

        # Block F first; mutate state.position copy so MM blocks treat
        # freshly acquired exposure as inventory.
        net_pos = self.counterparty_layer(state, cp_data, orders)
        for sym, dq in net_pos.items():
            state.position[sym] = state.position.get(sym, 0) + dq

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
  Block F (counterparty layer with OOS guards):
    - Tracks per-trader hit rate vs 10-tick forward mid.
    - NO seeded priors. Tracker starts empty every round.
    - Need >=100 scored trades before classifying.
    - Split-sample test: full hit rate AND trailing-30 hit rate must both
      clear the threshold (full=0.70/0.30, recent=0.60/0.40 with tol).
    - Trailing-15 < 0.55 → demote.
    - Classification flip → wipe tracker for that name.
    - Per-product sub-limits: 30 (HG/VE) / 40 (vouchers).
  HG / VE / free-call-lottery: same as conservative v2 (tight EMA fair,
  hard caps at 120, free-call accumulation at 30 max).

ANTI-OVERFIT GUARANTEES
  - No hardcoded counterparty names. No fitted vol surface. No statistical
    priors imported from historical analysis.
  - Even if a 100% hit-rate trader appears in production, we wait for
    >=100 trades AND consistent recent hit rate before committing capital.
  - Sub-limit 30-40 means a wrong classification caps loss to ~30 contracts
    × adverse move.

KEY RISKS
  - Block F may not classify anyone within a 1000-tick day. That's fine —
    failing silent is better than failing loud.
  - If a counterparty's signal genuinely flips mid-round, the regime-flip
    reset wipes their tracker; they need 100 fresh trades to re-qualify.

MOST SENSITIVE CONSTANTS
  1. COPY_MIN_TRADES (100) — lower (50) if no one classifies before round
     end; raise (150) if false positives appear.
  2. COPY_HIT_THRESHOLD / FADE_HIT_THRESHOLD — primary edge gate.
  3. SUBLIMIT[sym] — already tight; halve again if Block F PnL goes
     negative for any product on next test sample.
  4. COPY_RECENT_TOL — split-sample agreement tolerance; tighten to 0.05
     if false positives slip through.
"""