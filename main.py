"""CLI entry point for the NSE pre-open pipeline.

Examples:
    python main.py --demo                        # offline synthetic NSE data
    python main.py                               # Yahoo daily history + NSE pre-open (run 09:08-09:15 IST)
    python main.py --preopen-source kite         # pre-open from Zerodha Kite Connect
    python main.py --retrain --equity 500000
    python main.py --record-paper-orders         # write LONG signals to the paper ledger
    python main.py --evaluate                    # score past signals (after the close)
    python main.py --eod                         # after 16:00 IST: tomorrow's 125-day BUY/SELL list
    python main.py --eod --universe nifty500.csv --superstar Buys_by_Superstar_Investors.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

from config import ConfigError, load_config, setup_logging
from data_ingestion.data_fetcher import IST, MarketDataFetcher
from data_ingestion.preopen import PreOpenHistory
from data_ingestion.synthetic import SyntheticMarket, demo_now
from pipeline import DailyPipeline, session_status
from pipeline_eod import EodBreakoutPipeline
from risk_engine.guardrails import GuardrailViolation

logger = logging.getLogger("main")

DISCLAIMER = (
    "DISCLAIMER: Research and educational software only; not investment advice.\n"
    "Model probabilities are statistical estimates that can be wrong, and backtest/CV\n"
    "results do not guarantee future performance. Paper-trading only: no broker orders are\n"
    "placed. Trading involves substantial risk of loss. Consult a SEBI-registered adviser\n"
    "before investing."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE pre-open quantitative screening pipeline")
    p.add_argument("--config", help="Path to YAML settings file")
    p.add_argument("--tickers", help="Comma-separated NSE symbols, e.g. RELIANCE,TCS,INFY")
    p.add_argument("--equity", type=float, help="Account equity in INR for position sizing")
    p.add_argument("--preopen-source", choices=["nse", "kite"], help="Pre-open quote source")
    p.add_argument("--demo", action="store_true", help="Use offline synthetic data")
    p.add_argument("--retrain", action="store_true", help="Force model retraining")
    p.add_argument("--record-paper-orders", action="store_true",
                   help="Write validated LONG signals to outputs/paper_ledger.csv")
    p.add_argument("--evaluate", action="store_true", help="Evaluate previously logged signals")
    p.add_argument("--force", action="store_true", help="Run on weekends or configured holidays")
    p.add_argument("--research", action="store_true",
                   help="Market intelligence: regime, sectors, validated model ranking, BUY/SELL ideas")
    p.add_argument("--eod", action="store_true",
                   help="End-of-day 125-day breakout (BUY) / breakdown (SELL) scan with rule backtest")
    p.add_argument("--universe", help="CSV of symbols to scan (column 'Symbol' or 'NSE Code'), e.g. Nifty 500 list")
    p.add_argument("--superstar", help="Trendlyne 'Buys by Superstar Investors' CSV: scanned and tagged")
    return p.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    o: dict = {}
    if args.tickers:
        o["tickers"] = args.tickers.split(",")
    if args.equity:
        o.setdefault("risk", {})["account_equity"] = args.equity
    if args.demo:
        o.setdefault("data", {}).update(provider="synthetic", preopen_source="synthetic",
                                        preopen_history_path="data/demo/preopen_history.csv")
        o["output_dir"] = "outputs/demo"
        o.setdefault("model", {})["model_dir"] = "artifacts/demo"
    elif args.preopen_source:
        o.setdefault("data", {})["preopen_source"] = args.preopen_source
    if args.universe:
        o["universe_file"] = args.universe
    if args.superstar:
        o["superstar_file"] = args.superstar
    return o


MIN_TRADES_FOR_VERDICT = 30


def rule_verdict(trades: int, avg_r: float) -> str:
    """Plain-language read of a rule's backtest, refusing to judge tiny samples."""
    if trades < MIN_TRADES_FOR_VERDICT:
        return (f"only {trades} historical trades: too few to judge ({avg_r:+.2f}R is noise at this size). "
                "Widen the universe (e.g. --universe with the Nifty 500 list).")
    if avg_r > 0.05:
        return f"positive expectancy, {avg_r:+.2f}R per trade over {trades} trades"
    if avg_r < -0.05:
        return f"NEGATIVE expectancy, {avg_r:+.2f}R per trade over {trades} trades: signals have lost money on average"
    return f"roughly break-even ({avg_r:+.2f}R per trade over {trades} trades)"


def print_research(rep) -> None:
    """Console summary of the market-intelligence report."""
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 90):
        print(f"\nMARKET INTELLIGENCE - close of {rep.session:%d %b %Y} | {rep.universe_label}")
        if rep.regime:
            print(f"Nifty: {rep.regime['label']} (3M {100 * rep.regime['ret_3m']:+.1f}%, "
                  f"{100 * rep.regime['drawdown']:+.1f}% from 1-year high)")
        print(f"Breadth: {100 * rep.breadth['pct_above_200']:.0f}% of stocks above their 200-day average")
        print("\nStrongest sectors:", ", ".join(rep.sectors["Sector"].head(3)))
        v = rep.validation
        if v:
            status = "VALIDATED" if v["validated"] else "NOT VALIDATED (ideas are research only)"
            print(f"\nModel: {status} | mean IC {v['mean_ic']:.3f}, t-stat {v['ic_tstat']:.1f}, "
                  f"out-of-sample {v['oos_start']}..{v['oos_end']}")
            for k, ok in v["checks"].items():
                print(f"  [{'x' if ok else ' '}] {k}")
            print(f"  Top-ranked portfolio CAGR {100 * v['perf']['strategy']['cagr']:+.1f}% "
                  f"vs Nifty {100 * v['perf']['nifty']['cagr']:+.1f}% (after costs, same windows)")
        cols = ["Ticker", "Conviction", "Model score", "Price", "Stop Loss", "Target Price", "Qty", "Review by", "Why"]
        for action in ("BUY", "SELL"):
            part = rep.recommendations[rep.recommendations["Action"] == action]
            print(f"\n{action} ideas ({len(part)}):")
            print(part[cols].to_string(index=False) if not part.empty else "  none")
        if not rep.lab_summary.empty:
            print("\nStrategy lab (breakout variants):")
            print(rep.lab_summary.drop(columns=["Description"]).to_string(index=False))


def print_eod(result, cfg) -> None:
    """Console report for the end-of-day breakout scan."""
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(f"\nEOD BREAKOUT SIGNALS - session {result.session:%d %b %Y} (for the next session)")
        if result.skipped:
            print(f"Skipped (no NSE symbol): {', '.join(result.skipped)}")
        if result.rule_stats.empty:
            print("\nNo historical trades for these rules on this universe; no evidence either way.")
        else:
            print("\nHow these rules performed on this universe's history (after est. costs):")
            print(result.rule_stats[["Rule", "Trades", "Win rate %", "Avg R", "Profit factor", "Target %",
                                     "Stop %", "Avg hold (days)"]].to_string(index=False))
            for _, r in result.rule_stats.iterrows():
                print(f"  {r['Rule']}: {rule_verdict(r['Trades'], r['Avg R'])}")
        sig = result.signals
        if sig.empty:
            print("\nNo BUY or SELL signals today.")
        else:
            cols = ["Ticker", "Signal", "Close", "Beyond Range %", "Volume x Avg", "RSI", "Stop Loss",
                    "Target Price", "Max Position Units", "Superstar Buy", "Ticker Hist. Trades", "Ticker Hist. Avg R"]
            for label in ("BUY", "SELL"):
                part = sig[sig["Signal"] == label]
                if not part.empty:
                    print(f"\n{label} ({len(part)}):")
                    print(part[cols].to_string(index=False))
            print("\nBUY: " + "entry next session near the open; re-check the pre-open IEP.")
            print("SELL: exit/avoid if held. Cash-segment shorts are intraday only; overnight shorts need F&O.")
        print("\nBacktest caveat: using today's symbol list ignores delisted stocks (survivorship bias).")
        print(f"Saved: {cfg.output_dir / f'eod_signals_{result.session}.csv'}")


RULE_LABELS = [
    ("gap", "Gap outside range"), ("rvol", "Relative volume too low"), ("RSI", "RSI outside range"),
    ("price below", "Below SMA50"), ("EMA", "EMA20 not above EMA50"), ("avg turnover", "Turnover too low"),
    ("pre-open prev close", "Prev close mismatch"), ("price ", "Price outside range"),
    ("history", "Not enough history"),
]


def explain_scan(scan: pd.DataFrame) -> str:
    """Plain-text breakdown of why symbols failed, plus the closest near-misses."""
    if scan.empty:
        return "No symbols were screened (no pre-open quotes came back)."
    counts: Counter[str] = Counter()
    for failures in scan["failures"].fillna(""):
        for f in filter(None, failures.split("; ")):
            counts[next((label for prefix, label in RULE_LABELS if f.startswith(prefix)), f)] += 1
    lines = [f"Why symbols failed (out of {len(scan)}; one symbol can fail several rules):"]
    lines += [f"  {n:>3}  {label}" for label, n in counts.most_common()]
    view = scan.assign(rules_failed=scan["failures"].fillna("").map(lambda x: len([f for f in x.split("; ") if f])))
    view = view.sort_values(["rules_failed", "gap_pct"], ascending=[True, False]).head(8)
    lines.append("\nClosest near-misses:")
    for _, r in view.iterrows():
        lines.append(f"  {r['ticker']:<11} gap {r['gap_pct']:+5.2f}%  rvol {r['rvol']:4.2f} ({r['rvol_source']})  "
                     f"RSI {r['rsi']:5.1f}  ->  {r['failures']}")
    return "\n".join(lines)


def build_fetcher(cfg, demo: bool, now: datetime) -> MarketDataFetcher:
    """Real fetcher, or a synthetic one with seeded pre-open history for demos."""
    if not demo:
        return MarketDataFetcher(cfg.data)
    market = SyntheticMarket(now=now)
    market.seed_history(PreOpenHistory(cfg.data.preopen_history_path), list(cfg.tickers))
    return MarketDataFetcher(cfg.data, provider=market, preopen=market)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, build_overrides(args))
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    setup_logging(cfg.log_dir, cfg.log_level)
    print(DISCLAIMER + "\n")

    if args.research:
        from intelligence_engine import IntelligenceEngine
        now = demo_now("16:30") if args.demo else datetime.now(IST)
        fetcher = build_fetcher(cfg, args.demo, now)
        try:
            rep = IntelligenceEngine(cfg, fetcher, auto_universe=not args.demo).refresh(now, force=True)
        except Exception:  # noqa: BLE001
            logger.exception("Research run failed")
            return 1
        print_research(rep)
        return 0

    if args.eod:
        now = demo_now("16:30") if args.demo else datetime.now(IST)
        trading_day, note = session_status(cfg, now)
        if not trading_day and not args.force:
            logger.info("%s Use --force to scan the last completed session anyway.", note)
            return 0
        try:
            result = EodBreakoutPipeline(cfg, fetcher=build_fetcher(cfg, args.demo, now), now=now).run()
        except GuardrailViolation as exc:
            logger.critical("Guardrail halted the run: %s", exc)
            return 3
        except Exception:  # noqa: BLE001
            logger.exception("EOD scan failed")
            return 1
        print_eod(result, cfg)
        return 0

    now = demo_now() if args.demo else datetime.now(IST)
    trading_day, note = session_status(cfg, now)
    if not trading_day and not (args.force or args.evaluate):
        logger.info("%s Use --force to run anyway.", note)
        return 0

    try:
        pipeline = DailyPipeline(cfg, fetcher=build_fetcher(cfg, args.demo, now), now=now)
        if args.evaluate:
            results, summary = pipeline.tracker.evaluate(pipeline.load_histories())
            print(summary or "No evaluable signals yet.")
            return 0
        result = pipeline.run(retrain=args.retrain, record_paper_orders=args.record_paper_orders)
    except GuardrailViolation as exc:
        logger.critical("Guardrail halted the run: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Pipeline failed")
        return 1

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(f"\nNSE PRE-OPEN WATCH LIST - {result.run_time:%d %b %Y %H:%M} IST")
        print(result.timing_note)
        if result.summary.empty:
            print("No symbols passed today's screen.\n")
            print(explain_scan(result.scan))
        else:
            print(result.summary.to_string(index=False))
        print(f"\nFull scan saved to {cfg.output_dir / f'scan_{result.run_time:%Y-%m-%d}.csv'}")
    if result.cv_report:
        s = result.cv_report["summary"]
        print(f"\nModel CV: AUC {s['mean_auc']:.3f} | Brier {s['mean_brier']:.4f} | "
              f"Precision@thr {s['mean_precision_at_threshold']:.3f} vs base rate {s['mean_base_rate']:.3f}")
    for o in result.orders:
        print(f"Paper order {o['ticker']}: {o['status']} {o.get('reason', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
