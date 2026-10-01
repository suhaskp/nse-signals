# NSE Pre-Open Quantitative Screening Pipeline

> **Disclaimer:** research/educational software, not investment advice. Paper trading only: this
> build never places broker orders. Model probabilities are estimates and validation results do not
> guarantee future performance. Consult a SEBI-registered adviser before investing.

Screens NSE equities during the **09:00–09:15 IST pre-open call auction**, scores momentum
candidates with an XGBoost classifier, and sizes trades in INR with ATR stops, before the
09:15 open.

## Quick start (Windows, no manual steps afterwards)

1. Install Python 3.10+ from python.org (tick **Add python.exe to PATH**). One time only.
2. Double-click **`install_autostart.bat`**. It sets everything up and starts the dashboard, which opens in your
   browser at http://localhost:8501. From then on it starts by itself whenever you log in to Windows.
   (Prefer to start it yourself? Double-click `start_dashboard.bat` instead. macOS/Linux: `./start_dashboard.sh`.)
3. That's it. Leave the small minimised window running. It:
   - installs and updates its own packages, and restarts itself if it ever stops;
   - refreshes in the background, even with the browser closed: breakouts every 5 minutes in market hours,
     the morning pre-open check from 09:00, the full research and model run once per session after 16:00;
   - downloads the Nifty 500 list and price history itself (the very first run takes a few minutes).

Optional, set once: copy `config/settings.example.yaml` to `config/settings.yaml` to change equity or
risk limits (picked up automatically), and save new Trendlyne Superstar exports into `data/superstar/`.
The computer must be on and online during market hours for intraday updates.

**What you see each morning** (Market intelligence page): the BUY and SELL ideas from the last close,
re-checked from 09:00 against the pre-open price and after 09:15 against the live price. Each idea shows
today's gap, a status (**OK**, **CAUTION**, or **SKIP** when the gap makes the plan invalid or too risky)
and entry, stop, target and quantity recalculated from today's price.

## Layout

```
config/            config.py (NSE market, INR thresholds, tick table, costs), logging, settings.example.yaml
data_ingestion/    data_fetcher.py (Yahoo .NS daily history, batch download, backoff, sanitize), cache.py, universe.py
                   preopen.py (NSE / Kite Connect pre-open IEP, pre-open quantity history)
                   synthetic.py (offline demo market)
screeners/         screener.py, breakout.py (Chartink rules), buy_quality.py (quality-filtered BUY)
models/            ml_engine.py, ranker.py (cross-sectional ranker, SHAP reasons), portfolio.py (construction + costs)
risk_engine/       risk_manager.py (ATR stops, 2R targets, sizing, NSE ticks, costs), guardrails.py (paper ledger)
monitoring/        performance_log.py, breakout_backtest.py, strategy_lab.py (breakout variants by year)
pipeline.py        DailyPipeline orchestrator + session timing (pre-open)
pipeline_eod.py    End-of-day 125-day breakout/breakdown pipeline
main.py            CLI
app.py             Dashboard router; opens on views/intelligence.py (also live.py, preopen.py)
preopen_engine.py  Automatic pre-open scan (09:08-09:15 IST, own background thread)
intelligence_engine.py  Research + predictive ranking + validation + BUY/SELL ideas (cached per session)
research/          market.py, optimizer.py (holdout-confirmed tuning), stress.py (drawdown stress test)
live_engine.py     Auto-refreshing signal engine (universe, cache, live/confirmed signals, backtest)
start_dashboard.bat  Self-installing, self-restarting launcher; install_autostart.bat adds it to Windows startup
serve.py           Starts the background refresher, then the web server (used by the launchers)
engines_core.py    Shared engines + background refresher (fresh data even with no browser open)
tests/             pytest suite (173 tests, offline) + a week-long end-to-end test (`RUN_E2E=1 pytest tests/test_e2e_week.py`)
```

## Setup

Run the full suite with `pytest`. Before trusting an update, also run the slow end-to-end check, which drives every
engine through six simulated trading days and verifies every record: `RUN_E2E=1 pytest tests/test_e2e_week.py`
(Windows PowerShell: `$env:RUN_E2E=1; pytest tests/test_e2e_week.py`).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Run

```bash
python main.py --demo                  # offline synthetic NSE data - try this first
streamlit run app.py                   # dashboard at http://localhost:8501
python main.py                         # live: Yahoo daily history + NSE website pre-open
python main.py --preopen-source kite   # live pre-open from Kite Connect (set KITE_API_KEY, KITE_ACCESS_TOKEN)
python main.py --record-paper-orders   # write LONG signals to outputs/paper_ledger.csv
python main.py --evaluate              # after the close: replay logged signals, net of costs
```

**When to run:** 09:08–09:15 IST, after price discovery fixes the IEP. The pipeline warns if run
earlier or after the open, and exits on weekends and on dates listed in `market.holidays`
(fill these from NSE's yearly holiday circular). Cron example, with the server clock on IST:
`9 9 * * 1-5 cd /path/to/premarket_pipeline && .venv/bin/python main.py >> logs/cron.log 2>&1`

## Safety and status

- **Ready for real money?** A fixed checklist on the home page (NSE) and crypto page: enough closed forward trades
  on one rule version (`real_money_min_trades`, 50), average above `real_money_min_avg_r` (+0.10R), live results
  consistent with the backtest, not fragile under execution stress, no sign of overfitting. Until every item is ✅,
  the answer is "paper only". Thresholds are set in advance so the goalposts cannot move.
- **Heartbeat:** the sidebar shows each engine's last refresh; a watchdog thread raises an alert if live signals
  have not refreshed for 2 hours in market hours (or crypto for 2 hours at any time).
- **Version footer:** app version and both rule versions in the sidebar.
- **This computer only:** the dashboard listens on 127.0.0.1, so other devices on your Wi-Fi cannot open it. Set
  `allow_network_access: true` to change that.
- **Simple view** (sidebar switch, on by default): the decision, best setup, readiness and alerts. Switch it off for
  every research tab.

## Crypto (Binance) page

The same rules and safeguards, applied to the 80 most-traded Binance USDT spot pairs (stablecoins and leveraged
tokens excluded), using Binance's **public** market data (no API key; `data-api.binance.vision`, falling back to
`api.binance.com`). Paper only: no exchange orders are placed.

- **Market regime = Bitcoin** above its 200-day average; relative strength is measured against BTC; breadth is the
  share of coins above their 200-day average.
- **Daily candles close at 00:00 UTC (05:30 IST);** signals use completed days only and enter at the next daily open,
  at most 3% above the signal close. Prices refresh every 15 minutes in their own background thread.
- **Same BUY checks** (crypto-tuned: at least $20M traded a day, stops up to 20% of price), the ranking model with
  walk-forward validation, backtest verdict with evidence strength, results by regime, stress test, a movers test
  (days up 8%+), a forward record with its own rule version, and alerts tagged [Crypto].
- **Sizing:** 1% risk of `crypto.account_equity_usdt` (10,000 USDT) per trade, fractional quantities rounded to
  Binance lot sizes, at most 4 positions and 3% total risk; coins moving almost in lockstep (correlation > 0.9) with
  one already chosen are skipped. Costs assumed at 0.3% per round trip; Indian crypto tax (30% + 1% TDS) is not modelled.
- **Market permission vs trade opportunity** are shown separately on both pages (a favourable market is not a
  trade signal). The crypto page adds a **crypto market score** (7 components), an **altcoin regime** (are coins
  beating BTC?), a **Strength vs BTC** tab (7 / 30 / 90 days), and an **illustrative after-tax view** (30% on each
  gain, losses not set off, 1% TDS). The home page's **capital allocation** panel weighs NSE, crypto and cash.
- Evidence labels: LOW < 30 trades, PRELIMINARY < 100, MEDIUM < 300, HIGH from 300.
- Settings live under `crypto:` in `config/settings.yaml`; set `crypto: enabled: false` to hide the page. Crypto
  settings have their own rule version, so they never affect the NSE rule freeze.

## Daily briefing, journal and research checks

- **Daily briefing** (top of the home page): today's decision first, then the market regime, a cash score
  (0-100, with its reasons), the best candidate with its blockers and next trigger, changes since the previous
  session, and what not to do today.
- **Journal & changes** tab: every state change since the previous session with the checks that changed, and
  an append-only **decision journal** (every candidate's decision per session, with rule version, price, blockers,
  what would make it a BUY or an AVOID). Written once, never rewritten.
- **Evidence strength**: historical rates show their sample size, a 95% range and LOW / MEDIUM / HIGH evidence.
- **Robustness** (Strategy lab, weekly): the BUY rules with the breakout lookback and volume rule nudged up and
  down (flags results that only work at one exact setting), costs x1.5 and x2, extra slippage, and entry one session
  late, each with a verdict.
- **Multiple-testing check**: a running count of every distinct experiment, and the Deflated Sharpe Ratio of the
  model portfolio's chosen setup given that count.
- **Strategy health**: the forward record compared with the backtest once 30+ live trades have closed; flags
  degradation only when it exceeds normal variation.

## Records, rule freeze, alerts

- **Records live outside the app folder**, in `C:\Users\<you>\NSE_Signals` (set `PIPELINE_HOME` to change it):
  forward record, paper portfolio, watch history, alerts, pre-open history, caches and models. Reinstalling or
  deleting an app copy never touches them. Older installs are migrated automatically on first start, and a
  daily zip backup is kept in `NSE_Signals\backups` (last 14 days).
- **Forward record:** every BUY signal is saved when issued, with a rule-version stamp, and followed (next-open
  entry inside the range, then stop, target or time exit). Nothing is recomputed when rules change; each version
  is reported separately.
- **Rule freeze:** 75 days from first start (`rules_freeze_days`). The model portfolio's plan is pinned, and a
  warning appears if settings change anyway. Delete `NSE_Signals\rules_freeze.json` to restart it deliberately.
- **Alerts:** small pop-up notifications in the dashboard plus a 🔔 list in the sidebar for BUY signals, opening-check
  OK / SKIP, intraday breakouts, stale data, market-regime changes and rule changes during the freeze.
- **Evidence summary** at the top of the Strategy lab explains in words which checks help and whether the
  market-regime check is supported.
- **The pre-open gap scan is retired** (its model is the least likely to have an edge). Set
  `preopen_scan_enabled: true` to bring it back as an experimental page. The opening check of your BUY ideas
  against the pre-open price is unaffected.

## Fully automatic

Nothing in the dashboard is run by hand. After `start_dashboard.bat` (or autostart) the server keeps every page current:

| Job | When |
|---|---|
| Live breakout signals, today's movers | every 5 minutes in market hours, 30 minutes otherwise |
| Morning opening check | from 09:00, then against live prices after 09:15 |
| Research, model, validation, strategy lab | once per session after 16:00 (weekly tuning) |
| Paper portfolio, live track records, movers follow-up | after each completed session |

All settings come from `config/settings.yaml` (copy `config/settings.example.yaml`); each page shows the settings in
use. The command-line options in `main.py` remain for developers, but the dashboard never needs them.

## Market intelligence (home page)

`start_dashboard.bat` (or `python serve.py`) opens on **Market intelligence**, which builds itself:

- **Research:** Nifty trend state (50/200-day), drawdown and volatility regime; market breadth
  (% above 200-day, new highs/lows, advancers/decliners); sectors ranked by 3-month strength
  (sectors come from the Nifty 500 list's Industry column).
- **Prediction:** a cross-sectional XGBoost ranker scores every stock on its likely 10-session return
  *relative to peers*, from ~20 features: momentum (1 week to 12-1 months), 52-week high/low distance,
  volatility and its trend, RSI, volume surge, liquidity, trend vs 50/200-day averages, strength vs the
  Nifty and vs its sector, a 125-day breakout flag, and market context (index trend, index volatility,
  breadth). Features are ranked within each date; labels run from the next open, so nothing uses
  information you would not have had.
- **Validation gate:** purged walk-forward testing (train on the past, test on unseen later periods,
  with an embargo). The model is marked validated only if out-of-sample mean rank IC >= 0.02, IC
  t-stat >= 2, and the top-minus-bottom decile spread is positive in >= 60% of years. Otherwise every
  idea is labelled **Unvalidated**. A planted-signal test confirms the gate accepts real edges and
  rejects noise.
- **Ideas:** top-ranked BUYs and bottom-ranked SELLs with conviction (only when validated), plain-language
  reasons from SHAP contributions, confirmations (200-day trend, breakout, top sector, Superstar), how
  stocks in the same score decile did historically, ATR stop, 2R target, position size and a review date.
- **Model evidence:** out-of-sample portfolio (top 10, rebalanced every 10 sessions, after costs) vs the
  Nifty 50 with and without a regime filter, per-year results and decile calibration.
- **Strategy lab:** Chartink breakout rules vs improved variants (regime filter, trailing stop, no RSI
  cap), with a Passes / Marginal / Fails verdict per variant.

The heavy work (validation, training, lab) runs once per session and is cached in `data/cache/intelligence`,
so reopening the page the same day is instant. CLI equivalent: `python main.py --research`.

## Quality-filtered BUY signals

Every BUY (home page and Live breakout page) must pass all of these checks (settings in `buy_quality`):

| Check | Rule |
|---|---|
| Breakout quality | close above the prior 55-day high; strong close (top 40% of the day's range, green candle); at most 1 ATR beyond the breakout level |
| Volume | at least 1.5× the prior 50-day average; 20-day turnover of at least ₹10 crore |
| Trend | close > 50-day > 200-day average, 200-day rising, 3-month return above the Nifty's |
| ATR stop | 0.5 ATR below the breakout level, at least 1.5 ATR and at most 2.5 ATR (and 10% of price) from the entry |
| Market regime | Nifty above its 200-day average (optional breadth floor) |
| Reward/risk | at least 2:1, with the target = measured move, capped at the 52-week high when it is overhead and at 6 ATR |
| Model | when the ranker is validated, model score of at least 50/100 |
| Next-day entry | only at the next open, only if it opens at or above the breakout level and at most 2% above the signal close; stop and reward/risk are recomputed from the actual open and must still pass |

Top-ranked stocks that fail a check appear on a watch list with the failed checks named. The same rules are
backtested (Strategy lab, with a funnel showing how many breakouts survive each check) and tracked forward
from the day you start the dashboard (Buy ideas tab: live track record).

## Decision engine

- **Market regime panel:** five states (Risk-on, Risk-on weakening, Transition, Risk-off improving, Risk-off) from
  eight components, with a confidence level, breadth path and exposure guidance. Descriptive; the trading rule is
  the regime check in `buy_quality`.
- **Decision card** for the best setup: hard checks (all must pass, so a high score cannot hide a failure),
  conviction factors (model rank, sector rank, relative strength, regime, days on watch), the historical conditional
  win rate of similar setups, "Why not BUY", "Becomes BUY if", "Becomes AVOID if", and the trade plan with the
  maximum rupee loss.
- **Stale data never produces BUY NOW;** BUYs wait for fresh prices.
- **Watch ageing:** days on watch and readiness trend; setups expire after `watch_expiry_sessions` (10) without a breakout.
- **Portfolio concentration:** at most `max_per_sector` (2) new BUYs per sector, and no BUY whose recent daily returns
  correlate above `max_correlation` (0.8) with one already chosen.

## Setup readiness, states and research studies

- **Best setup card** (top of Market intelligence): the most complete setup, with its full checklist, breakout
  distance, volume vs requirement, regime, entry zone / stop / target / reward-risk once it has broken out, and a
  decision: 🟢 BUY NOW, 🟡 WATCH CLOSELY, ⚪ WAIT or 🔴 IGNORE.
- **Readiness 0-100** on the watch list and trending list: how complete a setup is (market regime excluded).
  It measures completeness, not the probability of profit.
- **Breakout status**: distance to the breakout level (very close / approaching / early), or how far above it,
  with "extended, don't chase" beyond 1 ATR.
- **Breadth trend**: % of stocks above their 200-day now vs 5 and 20 sessions ago.
- **Live page**: relative strength vs the Nifty (3 months and today); SELL signals are shown as "EXIT warnings"
  while the SELL rule has not made money historically.
- **Strategy lab**: each BUY check added one at a time (does it improve results or only cut trades?), results by
  market regime at the signal, and a three-regime variant (full size risk-on, half size neutral, none risk-off).
  Switch it on with `buy_quality: regime_mode: three_state` only if it holds up there.

## Trending stocks and today's movers

- **Trending now** (Trade signals: BUY tab): stocks in their own uptrend (close above the 50-day and a rising
  200-day, within 5% of the 52-week high, beating the Nifty over 3 months, liquid). Each shows the BUY checks it
  still fails; those failing *only* the market regime are highlighted as closest to a BUY. A research list, not a signal.
- **Today's strongest movers** (Live breakout signals page, market hours): top gainers on at least normal volume pace.
- **Regime evidence:** the Strategy lab also backtests the BUY rules *without* the regime check
  (`quality_no_regime`). To switch the check off, set `buy_quality: regime_required: false` in
  `config/settings.yaml`, but only if that variant holds up in the lab and stress test.
- **Single instance:** starting the dashboard while it is already running just opens the browser to it.

## Drawdown stress test (Strategy lab)

For any strategy variant (default: the quality BUY rules), at your risk per trade and drawdown tolerance:
worst historical drawdown (in R and % of equity), when it happened and how long recovery took, longest losing
streak, longest stretch without a new high, every year's results, and 5,000 Monte Carlo reshuffles of the
trades showing typical and bad-year (1 in 20) drawdowns, losing streaks, the chance of a losing year, and the
largest risk per trade that keeps a bad year within your tolerance.

## Profitability improvements (tested, not assumed)

- **Tuning (weekly):** the ranker is re-tested walk-forward for 10- and 20-session holding periods, and 32
  portfolio setups are simulated with turnover-based costs: top 10/20, hold-buffer (keep a stock while it stays
  in the top 20%, which cuts trading costs), cash when the Nifty is below its 200-day average, equal vs
  volatility-scaled weights. The best setup is chosen on the earlier out-of-sample years and **adopted only if it
  also beats the current setup on the final, unseen year** (Sharpe and return). The Tuning tab also says plainly
  whether the result beat simply holding a Nifty index fund.
- **Paper portfolio (automatic):** follows the chosen plan every session. Orders fill at the next open with
  costs, holdings are valued at each close, and the equity curve is compared with the Nifty 50. The Buy ideas
  tab shows the day's plan ("Rebalance at the next open: BUY ... SELL ... HOLD ..."). State lives in
  `outputs/paper_portfolio.json`; delete it to restart.

## Hands-free breakout page

Double-click `start_dashboard.bat` (or run `streamlit run app.py`). The **Live signals** page loads by
itself and shows the BUY and SELL lists; nothing needs to be clicked or downloaded:

| Time (IST, trading days) | What the page shows |
|---|---|
| Before 09:15 | Last session's confirmed signals, for today |
| 09:15–16:00 | Live signals on today's forming bar, rechecked every 5 minutes. Provisional: a stock can drop off before the close |
| After 16:00 | Today's confirmed signals, for tomorrow |
| Weekends/holidays | Last confirmed list |

- **Universe:** NSE's Nifty 500 list, downloaded automatically and refreshed weekly (falls back to the
  cached copy, then to the configured tickers, if NSE blocks the download).
- **Data:** about 5 years of daily history is downloaded once (a few minutes on the first start) and
  cached in `data/cache`; each refresh then fetches only recent days. Yahoo prices may be delayed and are
  not tick-by-tick. Tick-level real time needs a broker feed such as Kite Connect.
- **Superstar list:** Trendlyne requires a login, so it can't be downloaded automatically. Save any
  *Buys by Superstar Investors* export into `data/superstar/`; the newest file is used and those stocks
  are scanned and tagged.
- **Rule health:** each list shows whether its rule made or lost money historically on this universe
  (after costs), or has too few trades to judge. A losing or unproven rule is flagged as watch-only.
- **Open at login:** press Win+R, type `shell:startup`, and put a shortcut to `start_dashboard.bat` there.
- The page refreshes only while it is open in a browser; closing it pauses updates, and the next open
  catches up automatically.

## Daily routine

| When (IST) | Command | What you get |
|---|---|---|
| After 16:00 | `python main.py --eod --universe nifty500.csv --superstar superstar_buys.csv` | Tomorrow's BUY (125-day breakout) and SELL (125-day breakdown) list, sized, with each rule's backtested win rate and average R |
| 09:08–09:15 | `python main.py` | Pre-open gap screen and model score; re-check yesterday's BUYs against the IEP before entering |
| After close | `python main.py --evaluate` | Replay of logged pre-open signals, net of costs |

## End-of-day breakout scan (Chartink guide rules)

- **BUY:** close > prior 125-day high, volume > 2× the prior 125-day average, RSI(14) < 70.
- **SELL:** close < prior 125-day low, RSI(14) > 30, volume below the 125-day average. That volume
  condition is the saved Chartink scan in the guide; the guide's text says "×2", so
  `breakout.bear_volume_rule: above_multiple` switches to a high-volume breakdown.
- Each run **backtests the same rules on the same history** (next-day open entry, 1.5× ATR stop, 2R
  target, 20-session time exit, ~0.25% delivery costs) and prints the expectancy. Rules with fewer than
  30 historical trades are reported as too few to judge. Scan a wide universe (the Nifty 500 list from
  niftyindices.com has a `Symbol` column) so the backtest has enough trades.
- `--superstar` adds the Trendlyne "Buys by Superstar Investors" stocks to the scan and tags them.
  Rows without an NSE code (BSE-only or SME) are skipped and listed.
- SELL means exit or avoid if held. Cash-segment shorts must be squared off the same day; overnight
  shorts need F&O.

## Pre-open data sources

- **NSE website** (`preopen_source: nse`, default): free JSON behind nseindia.com's pre-open page.
  It is unofficial: it needs browser-like headers and cookies, can block automated traffic or change
  without notice, and is subject to NSE's terms of use. Keep request volume low.
- **Zerodha Kite Connect** (`preopen_source: kite`): read-only quotes; needs an API key and a daily
  access token. Check in your own account that `last_price` reflects the IEP after 09:08.
- Other brokers (Upstox, Angel One SmartAPI, etc.) can be added by implementing `PreOpenSource.fetch`.

## Signal logic

1. **Screen** (all must pass): IEP gap vs. previous close in [1%, 8%]; RVOL ≥ 1.5; RSI(14) in [50, 80];
   IEP > SMA50; EMA20 > EMA50; 20-day average turnover ≥ ₹25 crore; price in [₹50, ₹50,000]; and the
   pre-open feed's previous close within 2% of history (catches stale feeds and corporate actions).
2. **RVOL:** today's pre-open matched quantity vs. its average over prior sessions. The pipeline stores
   each day's pre-open quantities in `data/preopen_history.csv`; until 10 sessions exist, RVOL falls
   back to the previous session's volume vs. its 20-day average (shown as "prior_session").
3. **Model:** P(close > open). `LONG` if ≥ 0.55, else `WATCH`. On NSE the IEP *is* the opening price,
   so the gap feature is exact at prediction time.
4. **Risk:** stop = IEP − 1.5 × ATR, target = IEP + 2R, prices rounded to NSE price-band ticks; size so
   a stop-out costs 1% of equity, capped at 20% of equity per position, 5 positions and 4% total risk.
   Each plan shows estimated round-trip costs (brokerage, STT, exchange charges, GST, stamp duty).

## Paper trading and SEBI

`--record-paper-orders` (or the dashboard checkbox) validates LONG plans against every risk limit
and writes them to `outputs/paper_ledger.csv`; `--evaluate` replays them after the close with costs
deducted. Nothing is sent to a broker. Automating real order placement for retail traders in India
falls under SEBI's framework for algorithmic trading through brokers; check the current rules and
your broker's requirements before adding it.

## Known limitations

- Yahoo's `.NS` data is unofficial; verify prices against NSE for anything important.
- The default universe is a fixed large-cap list: training history has survivorship bias.
- Evaluation uses daily bars (stop assumed first if both levels are hit); slippage is ignored.
- The tick table and cost estimate are configurable defaults, not live exchange data. Verify both.
