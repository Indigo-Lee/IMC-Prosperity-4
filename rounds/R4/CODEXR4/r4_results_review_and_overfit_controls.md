# R4 Results Review and Anti-Overfit Controls

## Submitted result diagnosis

### Conservative submission: `CODEX_r4_conservative/530955.json`

- Final PnL: `-27,559`
- Main losses:
  - `VELVETFRUIT_EXTRACT`: about `-18,092`
  - `VEV_5200`: about `-5,126`
  - `VEV_5300`, `VEV_5400`, `VEV_5500`: about `-3,941` combined
  - `HYDROGEL_PACK`: about `-400`
- Diagnosis: the VE EMA/fair logic and the fitted voucher residual logic were too eager. They traded a lot on a sample-specific shape and did not have a strong enough live-loss brake.

### Aggressive submission: `CODEX_r4_aggressive/530788.json`

- Final PnL: `-32,554`
- Main losses:
  - `VELVETFRUIT_EXTRACT`: about `-22,791`
  - `VEV_5200`: about `-3,435`
  - `HYDROGEL_PACK`: about `-2,434`
  - remaining voucher losses about `-3,894`
- Diagnosis: the aggressive copy layer inherited the bad base and seeded a historical ID. That is exactly the overfit failure mode we want to avoid.

### Exploit sample: `r4_exploit_sample_results/530555.json`

- Final PnL: `-15,990`
- Main losses:
  - `VEV_4000`: about `-9,762`
  - `VELVETFRUIT_EXTRACT`: about `-3,845`
  - `HYDROGEL_PACK`: about `-2,383`
- Diagnosis: fixed-name Mark mirroring plus `VEV_4000` was brittle. Historical counterparty behavior did not generalize.

## Changes made

- Removed fitted voucher residual trading from conservative and aggressive files.
- Removed all hard-coded/seeded Mark IDs from aggressive and exploit files.
- Added internal risk caps far below exchange limits.
- Added product-level marked-to-mid loss stops.
- Added trend/fair-gap guards so the algos do not keep adding into fast adverse moves.
- Aggressive now copies counterparties only after live product-specific validation.
- Exploit now does nothing until a live trader/product pair proves itself in the current run.

## Anti-overfit system

The revised files use three layers:

1. Capacity control: every signal has an internal cap smaller than the exchange limit.
2. Live validation: copy trading starts from zero trust and must pass current-run hit-rate thresholds.
3. Circuit breakers: if a product loses too much marked-to-mid, the trader stops opening new risk and only flattens.

The design accepts lower peak backtest PnL in exchange for much lower chance of a full-day blowup on a new random day.
