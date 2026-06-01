# CODEXR4 Research Summary

## Notion pages read

- `writing-an-algorithm-in-python`: `Trader.run` returns `(orders, conversions, traderData)`. The Lambda runtime is stateless, so persistent state must be serialized into `traderData`. Position limits are enforced on gross orders per iteration.
- `Round 3 - Gloves Off`: products are `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, and vouchers `VEV_4000` through `VEV_6500`. Limits are 200 for each delta-1 product and 300 per voucher. Vouchers expire at round end, not daily.
- `Round 4 - The More The Merrier`: same algorithmic products and limits as Round 3. New mechanic: `Trade.buyer` and `Trade.seller` now contain named counterparty IDs. Flavor clue: "Hello, I'm Mark"; historical logs use `Mark xx` IDs.

## Local postmortem internalized

- Round 3 live failure came mainly from calibrating EMAs on 10,000-tick local days while live runs use 1,000 ticks/day. Constants in the new traders use `DAY_PERIOD = 100_000` and roughly 10x faster EMAs.
- The daily option-expiry idea was false and is not used.
- Historical Round 3 analysis rejected no-arb, cross-product lead-lag, and generic trade-flow predictors.

## R4 historical data findings

- `HYDROGEL_PACK`: mean around 9995, average spread about 15.7, negative return autocorrelation around -0.12. Use anchored fair-value market making around 10000.
- `VELVETFRUIT_EXTRACT`: mean around 5248, average spread about 5.0, negative return autocorrelation around -0.16, but meaningful intraday drift. Use faster EMA fair-value market making.
- `VEV_4000` and `VEV_4500`: intrinsic-like, delta-1 behavior with wide spreads. Conservative file skips them to avoid unhedged extra VE exposure.
- `VEV_5200` through `VEV_5500`: best fit for residual and dump-absorption logic.
- `VEV_6000` and `VEV_6500`: constant 0/1 markets; conservative file bids only at 0 for a zero-cost lottery.

## Counterparty findings

- `Mark 67` is the strongest historical candidate: about 82% good-trade rate over a 10-tick window and about 69% over a 20-tick window, almost entirely as a `VELVETFRUIT_EXTRACT` buyer.
- The aggressive file seeds `Mark 67` as a suspected insider, but also runs a generic rolling detector over every trader ID and applies a kill switch below 55% over 30 recent evaluated trades.

## Deliverables

- `trader_r4_conservative.py`: validated market-making and long-only voucher residual/lottery logic.
- `trader_r4_aggressive.py`: conservative base plus rolling counterparty scoring and bounded copy trading.
- `notion_writing.json`, `notion_round3.json`, `notion_round4.json`: raw public Notion API resources fetched for this build.
