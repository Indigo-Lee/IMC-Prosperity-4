# Round 3 — Phase 1 Findings

## (a) Persistent signals across all 3 days

| # | Signal | Magnitude | Confidence | Days consistent |
|---|---|---|---|---|
| 1 | **IV-residual mean reversion vs fitted parabolic smile** (mid-OTM vouchers) | residual std ≈ 0.005–0.008 vol-pts; ACF(1) 0.4–0.9 | **High** — sign of residual mean is consistent across all 3 days for 9/10 vouchers | All 3 |
| 2 | **HYDROGEL_PACK fair-value mean reversion at ~10000** | OU half-life ≈ 300 ticks; AR(1) Δmid b ≈ −0.13 (t ≈ −23) every day | **High** | All 3 (9991 / 9992 / 9989 daily mean) |
| 3 | **VELVETFRUIT_EXTRACT fair-value mean reversion at ~5250 with slight drift** | OU half-life ≈ 280 ticks; AR(1) b ≈ −0.16 (t ≈ −28) every day | **High**, with caveat that fair drifts (5246 → 5248 → 5255 across days) | All 3 |
| 4 | **One-way bot selling pressure on deep-OTM vouchers** (VEV_6000/6500 always at 0; VEV_5300/5400/5500 mostly at the bid) | ~280 trades/day each, total qty ~1k–4k contracts/voucher across 3 days | **High** — direction identical all 3 days | All 3 |
| 5 | **VEV_4000 / VEV_4500 are pinned at intrinsic** with zero time value | Mid lower-bound violations VEV_4500 = 29% of ts (magnitude 0.5–1.5 ticks, lagged catch-up) | High but **not exploitable** — bid/ask straddles intrinsic | All 3 |

### Signals that turned out to be NOT real (rejected)
- **No-arbitrage executable arbitrage** (H3): 0 upper-bound, 0 monotonicity, 0 butterfly, 2 single-tick lower-bound noise events across 30k snapshots × 41 executable checks. The bot pricer is internally consistent. **Do not write code looking for free arb.**
- **HYDROGEL leads VELVETFRUIT** (H2): max |xcorr| across ±10 lags = 0.012, all within 2σ. The two underlyings are independent.
- **Trade-flow → forward mid prediction**: best corr in the entire grid is ±0.06, R² < 0.005. Flow doesn't predict price. **Do not build a flow-based directional signal.**
- **Wide-quote pickoff (H4)**: every product's fill histogram peaks at the best bid/ask. No fat tails — bots fill at touch or one tick deeper, not at arbitrary prices. The exception is VELVETFRUIT_EXTRACT where 13.5% of fills happen 1 tick past best (tiny secondary edge).

## (b) Edge size per signal (rough cents-per-tick estimate)

- **IV-residual MR on VEV_5400**: residual ≈ −0.014 vol-pts ⇔ price mispricing of vega × Δσ. With vega(VEV_5400, S=5250, T=8/365, σ=0.23) ≈ 8 per voucher, that's **~0.11 ticks of mispricing per contract** — small per-trade but persistent (ACF(1) = 0.93 on day 0, 0.56 day 1, 0.31 day 2). At full position 300, that's ~$33 carry. The PnL comes from doing this every tick × tens of vouchers, so ~$1k–$5k/day ballpark, not a six-figure edge.
- **HYDROGEL_PACK fair-value MM at 10000**: spread = 16 ticks. If we capture 4-tick edge per round-trip (R1 ASH playbook captures 1–4) at ~50 trades/day × 4–6 contracts ≈ **$1k–2k/day**.
- **VELVETFRUIT_EXTRACT MM at rolling fair**: spread = 5, half-spread = 2.5. Capture ~1.5 ticks per round-trip × ~100 trades/day × ~5 size ≈ **$500–1k/day**.
- **Free-call lottery on VEV_6000/VEV_6500 at price 0**: cost = 0 per contract. Up to 300 contracts each, expiry ITM probability tiny but nonzero. Expected value across the round = ~$0–500 (mostly nuisance, but it's literally free).
- **Bid-absorption on VEV_5300/5400/5500**: capture the spread (1–2 ticks) when bots dump. ~250 contracts/voucher/day × 1.5 ticks ≈ **$300–750/day per voucher**, but limited by the 300-position cap and how fast we can recycle.

Total back-of-envelope: ~**$3k–10k per day**. Single-product hero PnL is unlikely; this is a diversified-edges round.

## (c) Reinterpretation of the Magritte clue

The painting hint *"Ceci n'est pas une pipe"* lined up cleanly with one specific feature of the data:

> **The vouchers are not Black-Scholes calls.**

The IV-vs-moneyness chart shows it directly: the implied-vol curve is **flat at 0.24 for ATM/ITM and ramps linearly into 0.4–0.65 for OTM**. A real BS smile is a soft U with skew. What we see is a **floor + linear-in-moneyness markup** pricing rule with **zero time value at deep ITM** (VEV_4000 / VEV_4500 priced literally at max(S − K, 0)). The bot pricer is some heuristic, not Black-Scholes — that mismatch creates the IV-residual signal because we're fitting BS to non-BS prices and what's left over has structure.

**Operational consequence**: trade IV residuals against the *fitted* smile (which captures the bot's heuristic), don't trade IV residuals against a theoretical model. We are exploiting the deviations of the bot's heuristic from itself, tick-to-tick — not from "fair" BS.

A secondary reading: VELVETFRUIT_EXTRACT is sometimes called the "underlying" but functionally it's just another mean-reverting product with no special role for the vouchers (no leadership, no convexity hedging structure). Treat it like HYDROGEL — a standalone MM target. The "underlying" label is slightly misleading.

## (d) Strategy primitives the data supports, ranked

1. **HYDROGEL_PACK fair-value market making at 10000** — clone of R2.1 ASH_COATED_OSMIUM block, only the constant changes (10000, position limit 200 instead of 80). Highest expected PnL/risk ratio of any block, lowest implementation risk. Days-stable.

2. **VELVETFRUIT_EXTRACT fair-value market making with rolling fair** — same logic, but fair = exponentially-weighted mid (α ≈ 0.001 to track the slow drift), edge = 2 ticks since spread is only 5. Tight quotes, rapid turnover.

3. **Voucher IV-residual mean-reversion** — per-tick parabolic fit IV = a + b·m + c·m², residual z-score → direction. Trade VEV_5000–5500 + VEV_6000/6500. **Exclude VEV_4000, VEV_4500 from the fit** (they're pinned at intrinsic, IV undefined often). **Exclude VEV_5100 from directional bets** (sign-flipped between days). Hedge net delta in VELVETFRUIT_EXTRACT.

4. **VEV_5400 specialist short-IV trade** — strongest, most consistent negative residual (every day, growing magnitude). Worth a dedicated, tighter-threshold sub-strategy.

5. **VEV_5300 / VEV_5200 specialist long-IV trade** — strongest consistent positive residual. Buy when residual z is sufficiently positive (i.e., voucher is under-priced vs neighbors).

6. **Free-call lottery on VEV_6000 / VEV_6500** — passive bid at price 0 (or 1 tick below mid). Bots dump worthless options at 0 ~280 times/day; we accumulate inventory at zero cost. Position cap 300 each.

7. **Bid-absorption on VEV_5300 / VEV_5400 / VEV_5500** — passive bid at the touch, passive ask at touch + 1. Captures the spread on bot dumps.

**De-prioritized / drop**:
- Cross-product hedging between HYDROGEL and VELVETFRUIT (no signal).
- Trade-flow regression (no signal).
- No-arb arbitrage (no signal).
- Wide-quote pickoff (no signal).

## (e) Surprises not in the original brief

1. **Two vouchers (VEV_4000, VEV_4500) are forwards in disguise** — priced exactly at intrinsic, no time value. Could be used as a synthetic long-underlying instrument with zero theta, useful if VELVETFRUIT_EXTRACT capacity is exhausted but we want more long delta. (Position limit on VEV_4000 is 300 vs 200 for VELVETFRUIT — extra 50% delta capacity.)
2. **Three vouchers (VEV_4500, VEV_5000, VEV_5100) had 1 trade each across 3 days.** Bot flow doesn't reach those strikes. Edge has to come from *us* posting quotes; we'll be the entire counterparty. This means strategies for those strikes cannot rely on absorbing other-side flow — they have to be IV-residual or skip.
3. **VEV_6000 and VEV_6500 trade exclusively at price 0** when mid is 0.5. Bots literally give worthless options away for free. Free expected value at the position cap — zero downside, tiny upside.
4. **Residual ACF(1) on VEV_5400 is 0.93 on day 0 but only 0.31 on day 2.** Alpha decay is faster as TTE shrinks. By round-3-live (TTE = 5d, less than day_2's 6d), persistence may be even shorter. Build the strategy with a per-tick fit, not a slow EMA, and don't hold static positions for too long.
5. **The OTM-IV ramp magnitude grows with day** — VEV_6500 IV 0.55 / 0.60 / 0.64 across days 0/1/2. This is exactly time-decay-driven: as TTE shrinks, the bot's heuristic widens the wings to compensate. So in the live round we should expect even higher OTM IVs than day_2.

## v1 trader composition (for Phase 2 planning)

Single-tick run order:
1. **Fast path (skip if disabled)**: scan no-arb checks. (Disabled by default — Phase 1.3 found nothing — but leave 5 lines of code for completeness.)
2. **HYDROGEL_PACK MM** (R2.1-port).
3. **VELVETFRUIT_EXTRACT MM** with EMA fair.
4. **Compute IVs and fit smile** (excluding VEV_4000, VEV_4500). Cache (a, b, c) in `traderData`.
5. **For each voucher in {VEV_5000, 5200, 5300, 5400, 5500, 6000, 6500}**: residual z-score → desired voucher position (proportional to z, capped at limit). Send orders to move toward target.
6. **Free-call lottery**: VEV_6000 / VEV_6500 — passive buy at price 0 / 1 below mid up to 300.
7. **Net-delta governor**: hedge in VELVETFRUIT_EXTRACT to keep |Δ_total| < some cap.

## Critical files (reused / to be written in Phase 2)

- `~/Desktop/imcprosperity4/rounds/R2/R2.1.py` — port `aco_take_best_orders`, `aco_clear_position_order`, `aco_make_orders`, `Logger`, `compress_*`. The IPR trend-rider block is not used.
- `~/Desktop/imcprosperity4/rounds/R3/work/analysis/iv_analysis.py` — port the BS / IV / parabolic-fit code into the trader (translate to pure stdlib + math).
- `~/Desktop/imcprosperity4/rounds/R1/datamodel.py` — re-exports the IMC types we need.
- New: `~/Desktop/imcprosperity4/rounds/R3/work/trader_r3_v1.py`.
- New: `~/Desktop/imcprosperity4/rounds/R3/work/backtest.py` (FlatFileReader pattern from R1).
