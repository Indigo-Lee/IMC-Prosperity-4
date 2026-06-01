# IMC Prosperity 4

**#198 globally / #56 in the U.S.** out of 20,000+ teams — IMC Prosperity 4 algorithmic trading competition, April 2026.

Four rounds of live algo development on a simulated exchange, with new products, mechanics, and data released each round. All strategies written in Python under strict constraints: no external libraries, 900ms execution budget per tick, stateless Lambda runtime (state serialized through `traderData`).

---

## Results by Round

| Round | Products | Core Strategy | Key Mechanic |
|---|---|---|---|
| R1 | INTARIAN_PEPPER_ROOT, ASH_COATED_OSMIUM | Trend-following + market-making | EMA crossover drift detection |
| R2 | Same + Market Access Fee | Same + MAF blind auction | Bid sizing for flow allocation |
| R3 | HYDROGEL_PACK, VELVETFRUIT_EXTRACT, 10 vouchers | MM + options IV residual MR + free-call lottery | Parabolic smile fitting, non-BS bot discovery |
| R4 | Same + named counterparty IDs | Conservative MM + live counterparty scoring | Rolling hit-rate validator, circuit breakers |

---

## Strategy Details

### Round 1 — Trend + Mean Reversion
- **INTARIAN_PEPPER_ROOT**: detected a clean +990/tick/day uptrend via EMA crossover (fast α=0.40, slow α=0.01). Projected fair value above the ask, crossed the book immediately, and held max-long (+80) all day. ~79,200 XIRECs/day theoretical.
- **ASH_COATED_OSMIUM**: stationary at 10,000 with ±16 spread and strong negative return autocorrelation (−0.49). Tight passive quotes at mid±1, EMA deviation from long-run mean drives a target position to earn the bid-ask bounce.
- Backtest: **~250,900 XIRECs** over 3 days.

### Round 2 — MAF Auction
- Same trading logic as R1. Added `bid()` method to participate in the Market Access Fee blind auction (top 50% of bids get ~25% extra quote flow, fee subtracted from PnL).

### Round 3 — Options Microstructure
Products: mean-reverting HYDROGEL (≈10,000) and VELVETFRUIT_EXTRACT (≈5,250) delta-1, plus 10 call vouchers (VEV_4000–VEV_6500).

**Key discovery**: the vouchers are not Black-Scholes calls. The implied-vol curve shows a flat floor at 0.24 for ATM/ITM and a linear ramp to 0.4–0.65 OTM — a heuristic pricing rule, not a true smile. Deep-ITM vouchers (VEV_4000/4500) are priced at intrinsic with zero time value.

**Strategy blocks:**
1. **HG/VE market-making** — EMA fair-value quotes, lean position into mean-reversion
2. **IV residual mean-reversion** — per-tick parabolic smile fit (IV = a + b·m + c·m²), z-score residuals → directional voucher positions on VEV_5000–5500
3. **Free-call lottery** — passive bids at price 0 on VEV_6000/6500; bots dump worthless OTM calls for free, 300-contract limit cap

Analysis infrastructure: IV fitting ([`analysis/iv_analysis.py`](rounds/R3/work/analysis/iv_analysis.py)), no-arb checks (no executable arb found), flow regression (R² < 0.005, dropped), per-day stability testing. See [`findings.md`](rounds/R3/work/findings.md) for the full signal audit.

### Round 4 — Counterparty Intelligence
New mechanic: `Trade.buyer` / `Trade.seller` now expose named counterparty IDs. Historical data analysis identified candidate informed traders by computing rolling hit-rates per counterparty-product pair.

**Two submission variants:**
- **Conservative** ([`trader_r4_conservative.py`](rounds/R4/trader_r4_conservative.py)): HG + VE market-making with tightened internal risk caps and product-level circuit breakers.
- **Aggressive** ([`trader_r4_aggressive.py`](rounds/R4/trader_r4_aggressive.py)): conservative base + rolling counterparty scoring. Copy-trades only after live product-specific hit-rate clears a validation threshold (>55% over 30 trades), building trust from zero each run.

**Anti-overfit design:**
- Internal risk caps well below exchange limits
- Copy-trading requires live validation — no hard-coded counterparty IDs
- Circuit breakers halt new risk-opening if marked-to-mid loss exceeds per-product threshold

---

## Repo Structure

```
trader.py                          final consolidated submission
datamodel.py                       IMC exchange type definitions
rounds/
  R1/
    trader.py                      R1 final submission
    backtest.py / walkforward.py   backtesting framework
    strategy9/                     parameter sweep results
  R2/
    R2.1.py                        R2 final submission
    validate.py / visualize_r2.py  validation tooling
  R3/
    work/
      trader_r3_v3.py              R3 final submission
      analysis/                    IV fitting, flow analysis, no-arb checks
      exploits/                    microstructure exploit studies (10 hypotheses)
      findings.md                  full signal audit from phase 1
      backtest.py
  R4/
    trader_r4_conservative.py      R4 conservative submission
    trader_r4_aggressive.py        R4 aggressive submission
    CODEXR4/
      r4_research_summary.md       counterparty findings, data analysis
wiki/
  wiki_algorithm.md                IMC exchange mechanics reference
```

---

## Stack

Python 3.12 stdlib only (competition constraint). No external libraries in submitted traders — `math`, `json`, `typing` only. Analysis scripts use `pandas`, `numpy`, `matplotlib`, `scipy`.
