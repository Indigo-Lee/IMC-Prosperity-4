"""
IMC Prosperity 4 – Round 1 Trader
Products: INTARIAN_PEPPER_ROOT, ASH_COATED_OSMIUM
Position limits (per IMC wiki): 80 for both products.

Strategy
─────────────────────────────────────────────────────────────────────
INTARIAN_PEPPER_ROOT (IPR)
  The wiki describes IPR as "quite steady", but the 3 sample days show
  a clean linear uptrend of +990 units per day with strong negative
  return autocorrelation (-0.51) — mean-reverting noise layered on top
  of the drift.

  Approach: trend-following take.
    • fast EMA (α=0.40) − slow EMA (α=0.01) quickly detects the upward
      momentum within the first few timestamps.
    • Fair value = fast_ema + 8 × (fast_ema − slow_ema) — projects the
      trend forward so that the fair value sits above the best ask.
    • Result: we cross the book immediately and load up to the +80
      position limit, then hold all day.
    • Passive quotes (±5 around fair) add marginal spread income.
    • Theoretical daily max: 80 × 990 = 79,200 XIRECs.

ASH_COATED_OSMIUM (ACO)
  Stationary near 10,000 with a ±16 book spread. Return autocorr
  ≈ −0.49 → pure mean-reverting oscillations.

  Approach: tight market-making + mean-reversion position targeting.
    • Passive quotes at mid ± 1 (inside the existing spread) to maximise
      fill rate against historical market trades.
    • Slow EMA deviation from the long-run mean (10,000) drives a
      target position: lean long when price is cheap, short when dear.
    • Aggressive inventory skew (2.3 × normalised inventory) cycles
      the position rapidly, earning the bid-ask bounce on each trip.

Optimised over days −2, −1, 0 from DataCapsule_R1.
Backtest: ~250,900 XIRECs across three days
  (~79,300/day from IPR drift + ~4,000/day from ACO market-making).

Import note: use `from datamodel import ...` when uploading to IMC.
For local backtesting with prosperity2bt, the import below works as-is.
"""

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

# ── Position limits (confirmed from IMC Round 1 wiki page) ────────────────
POSITION_LIMITS: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

# Long-run mean for mean-reversion signal (ACO only)
LONG_RUN_MEAN: dict[str, float] = {
    "ASH_COATED_OSMIUM": 10_000.0,
}

PARAMS: dict[str, dict[str, Any]] = {
    "INTARIAN_PEPPER_ROOT": {
        # EMA parameters
        "fast_alpha": 0.40,        # tracks intraday price very closely
        "slow_alpha": 0.01,        # lags to measure momentum
        "momentum_factor": 8.0,    # forward-project (fast−slow) N half-lives
        # Quoting
        "half_spread": 5,          # ±5 around fair value for passive quotes
        "skew_per_unit": 0.3,      # quote skew per (pos−target)/limit unit
        "max_order_size": 10,
        # Opportunistic takes from book when edge ≥ take_edge vs fair value
        "take_edge": 1,
        # Inventory target: lean +15 to ride uptrend, keep some headroom
        "target_position": 15,
        # No mean-reversion signal for IPR (trend dominates)
        "mr_factor": 0.0,
        "mr_clamp": 0,
    },
    "ASH_COATED_OSMIUM": {
        "fast_alpha": 0.15,
        "slow_alpha": 0.04,
        "momentum_factor": 0.0,    # no trend to project
        "half_spread": 1,          # very tight — maximise fill rate
        "skew_per_unit": 2.3,      # aggressive skew for fast inventory cycling
        "max_order_size": 10,
        "take_edge": 0,            # take whenever book crosses fair value
        "target_position": 0,      # overridden by mean-reversion signal
        "mr_factor": 20.0,         # target = −(slow_ema − 10000) × 20
        "mr_clamp": 80,            # cap at ±80 (= position limit)
    },
}


class Trader:
    def run(self, state: TradingState):
        try:
            pstate: dict[str, Any] = (
                json.loads(state.traderData) if state.traderData else {}
            )
        except Exception:
            pstate = {}

        result: dict[str, list[Order]] = {}

        for product, params in PARAMS.items():
            if product not in state.order_depths:
                continue

            ps = pstate.setdefault(product, {})
            position = state.position.get(product, 0)
            od = state.order_depths[product]

            orders = self._trade(product, params, od, position, ps)
            if orders:
                result[product] = orders

        return result, 0, json.dumps(pstate)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _mid(od: OrderDepth) -> float | None:
        """Best mid-price from order book. Returns None if one side is empty."""
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    def _trade(
        self,
        product: str,
        params: dict[str, Any],
        od: OrderDepth,
        position: int,
        ps: dict[str, Any],
    ) -> list[Order]:
        orders: list[Order] = []
        limit = POSITION_LIMITS[product]

        mid = self._mid(od)
        if mid is None:
            return orders

        # ── Update EMAs ──────────────────────────────────────────────────
        fast_a = params["fast_alpha"]
        slow_a = params["slow_alpha"]
        if "fast_ema" not in ps:
            ps["fast_ema"] = mid
            ps["slow_ema"] = mid
        else:
            ps["fast_ema"] = fast_a * mid + (1.0 - fast_a) * ps["fast_ema"]
            ps["slow_ema"] = slow_a * mid + (1.0 - slow_a) * ps["slow_ema"]

        # ── Fair value: EMA + forward-projected momentum ─────────────────
        momentum = ps["fast_ema"] - ps["slow_ema"]
        fair = ps["fast_ema"] + params["momentum_factor"] * momentum

        # ── Inventory target ─────────────────────────────────────────────
        if params["mr_factor"] > 0.0 and product in LONG_RUN_MEAN:
            deviation = ps["slow_ema"] - LONG_RUN_MEAN[product]
            target = -deviation * params["mr_factor"]
            target = max(-params["mr_clamp"], min(params["mr_clamp"], target))
        else:
            target = params["target_position"]

        # ── Quote skew based on inventory vs target ───────────────────────
        inv_ratio = (position - target) / limit
        skew = params["skew_per_unit"] * inv_ratio * params["half_spread"] * 2

        bid_price = round(fair - params["half_spread"] - skew)
        ask_price = round(fair + params["half_spread"] - skew)
        if ask_price <= bid_price:
            ask_price = bid_price + 1

        take_edge = params["take_edge"]
        max_sz = params["max_order_size"]

        # ── Opportunistic takes from order book ───────────────────────────
        # Buy at best ask if it is cheap vs fair value
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            if best_ask <= fair - take_edge:
                room = limit - position
                # sell_orders values are negative per IMC spec
                avail = abs(od.sell_orders[best_ask])
                size = min(avail, room, max_sz)
                if size > 0:
                    orders.append(Order(product, best_ask, size))

        # Sell at best bid if it is rich vs fair value
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            if best_bid >= fair + take_edge:
                room = limit + position
                avail = od.buy_orders[best_bid]
                size = min(avail, room, max_sz)
                if size > 0:
                    orders.append(Order(product, best_bid, -size))

        # ── Passive quotes ────────────────────────────────────────────────
        long_used = sum(o.quantity for o in orders if o.quantity > 0)
        short_used = sum(abs(o.quantity) for o in orders if o.quantity < 0)

        buy_size = min(max_sz, max(0, limit - position - long_used))
        sell_size = min(max_sz, max(0, limit + position - short_used))

        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))

        return orders