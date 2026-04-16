"""
IMC Prosperity 4 – Round 1 Trader (v2 — walk-forward tuned + microstructure)
Products: INTARIAN_PEPPER_ROOT, ASH_COATED_OSMIUM
Position limits (per IMC wiki): 80 for both products.

This file extends trader.py with parameter tuning and two microstructure
upgrades, all justified by walk-forward analysis. The Trader class
signature, datamodel imports, and traderData (de)serialisation remain
unchanged so it stays drop-in submittable to the IMC platform.

Tuning was driven by walk-forward analysis across three train/test splits:
  W1: train [-2]            → test [-1]
  W2: train [-2,-1]         → test [0]
  W3: train [-2,-1,0]       → test [3_partial]

Only signals stable in sign and magnitude across ALL THREE test windows
were used to justify a change. Each modified parameter — and each new
helper — has a comment citing the supporting metric and the windows that
confirmed it. Anything inconsistent across the three windows was left
untouched.

Two microstructure upgrades over trader.py:
  1. _mid() replaced by _vwap_mid() — opposite-side-weighted VWAP using top
     3 book levels. Walk-forward measured corr(vwap_lean, next-tick Δmid)
     ≈ +0.31 (IPR) and +0.32 (ACO), STABLE across W1/W2/W3, so depth-
     weighted micro-price is a better fair-value reference than raw mid.
  2. IPR-only trade flow signal — recent buyer-initiated minus seller-
     initiated volume nudges fair value. Walk-forward measured
     corr(flow, next-tick Δmid) ≈ +0.43 (IPR), STABLE across W1/W2/W3.
     For ACO the same metric was INCONSISTENT (+0.12 / −0.02 / +0.51) so
     the signal is gated by `flow_factor=0.0` for that product.
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
        # ── EMA parameters ───────────────────────────────────────────────
        # CHANGED 0.40 → 0.30. Walk-forward W1/W2/W3 all show ret_autocorr1
        # ≈ −0.50 (W1 −0.4950, W2 −0.5086, W3 −0.5034 — STABLE). At α=0.40 the
        # fast EMA whipsaws on the consistently mean-reverting tick noise that
        # rides on top of the +0.10/tick drift. Smoother fast α reduces
        # noise pickup while still capturing the very stable intraday trend.
        "fast_alpha": 0.30,
        # Unchanged. mid_slope_per_tick = +0.1002 STABLE across W1/W2/W3
        # (W1 0.1002, W2 0.1002, W3 0.1001) — α=0.01 lags far enough to
        # measure the persistent drift without being polluted by tick noise.
        "slow_alpha": 0.01,
        # Unchanged. mid_drift over ~1000 ticks ≈ +1000 across W1/W2; the
        # +0.1002/tick slope is stable, so projecting the (fast−slow) gap
        # forward is what loads us above best ask immediately.
        "momentum_factor": 8.0,
        # ── Quoting ──────────────────────────────────────────────────────
        # CHANGED 5 → 4. spread_mean STABLE across W1/W2/W3 at 13.012 / 14.129
        # / 13.710 (and spread_std ≈ 2.5 stable). Quoting at ±5 left ~3 ticks
        # of unused margin inside the market's consistently wide ~13-tick
        # spread. ±4 still sits well inside that spread (gives ~5 ticks of
        # buffer to the touch on each side) and yields more passive fills.
        "half_spread": 4,
        # Unchanged. ret_autocorr1 STABLE at ≈ −0.50 means inventory taken
        # against the immediate move tends to revert favourably, so heavy
        # skewing isn't needed; light 0.3 is consistent with this signal.
        "skew_per_unit": 0.3,
        "max_order_size": 10,
        # Unchanged. flow_corr_next_dmid = +0.43 STABLE across W1/W2/W3
        # (0.4623 / 0.4313 / 0.4384) confirms aggressive crossing is +EV.
        "take_edge": 1,
        # Unchanged. Stable +0.10/tick drift justifies a long lean.
        "target_position": 15,
        # No mean-reversion signal for IPR (trend dominates).
        "mr_factor": 0.0,
        "mr_clamp": 0,
        # ── NEW: trade-flow fair-value adjustment (IPR only) ─────────────
        # corr(net flow, next-tick Δmid) STABLE at +0.43 across W1/W2/W3
        # (W1 0.4623, W2 0.4313, W3 0.4384). Net signed trade volume over
        # the last `flow_window` ticks nudges fair value by `flow_factor`
        # per share. Window of 10 ticks holds ~3 trades on average; factor
        # of 0.03 keeps the flow contribution bounded (typical ±30 share
        # net → ±0.9 tick fair adjustment), small relative to the ~13-wide
        # market spread so it refines fair value without dominating EMA.
        "flow_factor": 0.03,
        "flow_window": 10,
    },
    "ASH_COATED_OSMIUM": {
        # ── EMA parameters ───────────────────────────────────────────────
        # CHANGED 0.15 → 0.10. ret_autocorr1 STABLE at ≈ −0.49 across W1/W2/W3
        # (W1 −0.4980, W2 −0.4872, W3 −0.4923). Tick-by-tick reversion is
        # extremely consistent, so a lower fast α dampens the noise the EMA
        # would otherwise chase. Note vwap_lean_corr_next_dmid ≈ +0.32 STABLE
        # across windows confirms microstructure (size-weighted) is the real
        # signal, not raw mid moves — slowing fast α reduces over-reaction
        # to those raw mid moves without losing reversion responsiveness.
        "fast_alpha": 0.10,
        # Unchanged. The slow EMA anchors the mean-reversion target; the
        # stable autocorr structure means there is no reason to alter it.
        "slow_alpha": 0.04,
        # Unchanged. mid_slope_per_tick is INCONSISTENT (W1 −0.0001,
        # W2 +0.0002, W3 −0.0017 — flips sign) and flow_corr_next_dmid is
        # INCONSISTENT (+0.12 / −0.02 / +0.51 — flips sign). No stable
        # directional signal → keep momentum projection off.
        "momentum_factor": 0.0,
        # Unchanged. spread_mean STABLE at ≈ 16.2 across W1/W2/W3. ±1 is
        # already maximally tight inside that consistent 16-wide spread;
        # tightening further would risk crossing our own quotes.
        "half_spread": 1,
        # Unchanged. Stable −0.49 autocorr supports continued aggressive
        # inventory cycling.
        "skew_per_unit": 2.3,
        "max_order_size": 10,
        "take_edge": 0,
        "target_position": 0,
        # Unchanged. The mean-reversion factor is driven by slow_ema
        # deviation from 10000, justified by the long-run stationarity that
        # autocorr ≈ −0.49 across all windows continues to confirm.
        "mr_factor": 20.0,
        "mr_clamp": 80,
        # Disabled. corr(net flow, next-tick Δmid) is INCONSISTENT for ACO
        # across windows (W1 +0.12, W2 −0.02, W3 +0.51 — flips sign), so
        # we do NOT incorporate trade flow into ACO fair value.
        "flow_factor": 0.0,
        "flow_window": 10,
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

            # New trades since the last call (own + market). The IMC platform
            # delivers only the trades that occurred during the previous
            # round, so this list is naturally non-overlapping per tick.
            new_trades = list(state.own_trades.get(product, [])) + list(
                state.market_trades.get(product, [])
            )

            orders = self._trade(product, params, od, position, ps, new_trades)
            if orders:
                result[product] = orders

        return result, 0, json.dumps(pstate)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _vwap_mid(od: OrderDepth, depth: int = 3) -> float | None:
        """Opposite-side-weighted micro-price using top `depth` book levels.

        Walk-forward analysis showed corr(vwap_lean, next-tick Δmid) STABLE at
        +0.31 (IPR) and +0.32 (ACO) across W1/W2/W3, so this depth-weighted
        micro-price is a strictly better fair-value reference than raw mid.

        Construction:
          • bid_vwap = Σ(bid_price_i × bid_vol_i) / Σ bid_vol_i  (top `depth`)
          • ask_vwap = Σ(ask_price_i × ask_vol_i) / Σ ask_vol_i  (top `depth`)
          • micro    = (bid_vwap × ask_vol_total + ask_vwap × bid_vol_total)
                       / (bid_vol_total + ask_vol_total)

        Heavier opposite-side size pulls the fair value toward that side, the
        standard "micro-price" formulation extended to depth. Returns None if
        either side of the book is empty.
        """
        if not od.buy_orders or not od.sell_orders:
            return None
        # Top `depth` bids by descending price, asks by ascending price
        bids = sorted(od.buy_orders.items(), key=lambda kv: -kv[0])[:depth]
        asks = sorted(od.sell_orders.items(), key=lambda kv: kv[0])[:depth]
        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(abs(v) for _, v in asks)  # IMC asks are negative-signed
        if bid_vol <= 0 or ask_vol <= 0:
            return None
        bid_vwap = sum(p * v for p, v in bids) / bid_vol
        ask_vwap = sum(p * abs(v) for p, v in asks) / ask_vol
        return (bid_vwap * ask_vol + ask_vwap * bid_vol) / (bid_vol + ask_vol)

    def _trade(
        self,
        product: str,
        params: dict[str, Any],
        od: OrderDepth,
        position: int,
        ps: dict[str, Any],
        new_trades: list,
    ) -> list[Order]:
        orders: list[Order] = []
        limit = POSITION_LIMITS[product]

        mid = self._vwap_mid(od)
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

        # ── Trade-flow signal (gated by flow_factor; IPR-only by config) ──
        # Classify each new trade against the PRIOR tick's vwap-mid (stored
        # below as ps["last_mid"]). Trades printed above the prior fair are
        # buyer-initiated (someone lifted the offer); below are seller-
        # initiated (someone hit the bid). Net signed quantity is appended
        # to a rolling `flow_window`-tick buffer; the sum nudges fair value.
        # Justification: corr(net flow, next-tick Δmid) = +0.43 STABLE across
        # W1/W2/W3 for IPR. For ACO the same metric is INCONSISTENT, so we
        # set flow_factor=0.0 above and skip the adjustment entirely.
        flow_signal = 0.0
        flow_factor = params.get("flow_factor", 0.0)
        if flow_factor > 0.0:
            ref = ps.get("last_mid")
            net_this_tick = 0
            if ref is not None:
                for tr in new_trades:
                    if tr.price > ref:
                        net_this_tick += int(tr.quantity)
                    elif tr.price < ref:
                        net_this_tick -= int(tr.quantity)
            hist = ps.setdefault("flow_hist", [])
            hist.append(net_this_tick)
            window = int(params.get("flow_window", 10))
            if len(hist) > window:
                del hist[: len(hist) - window]
            flow_signal = float(sum(hist))

        # Persist current vwap-mid for next tick's flow classification
        ps["last_mid"] = mid

        # ── Fair value: EMA + forward-projected momentum + flow nudge ─────
        momentum = ps["fast_ema"] - ps["slow_ema"]
        fair = ps["fast_ema"] + params["momentum_factor"] * momentum
        fair += flow_factor * flow_signal

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
