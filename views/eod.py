"""Dashboard page: end-of-day 125-day breakout (BUY) / breakdown (SELL) scan.

Run the app with `streamlit run app.py`; this page appears in the sidebar.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ConfigError, load_config, setup_logging
from config.config import DEFAULT_TICKERS
from data_ingestion.data_fetcher import IST, DataFetchError, MarketDataFetcher
from data_ingestion.synthetic import SyntheticMarket, demo_now
from pipeline import session_status
from pipeline_eod import EodBreakoutPipeline
from screeners.breakout import breakout_frame

C = {"ink": "#1B2A41", "muted": "#5B6B82", "panel": "#FFFFFF", "grid": "#D8DEE7",
     "long": "#1F7A5C", "stop": "#B23A48", "watch": "#7A8699"}
MIN_TRADES = 30

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; font-variant-numeric: tabular-nums; }
.block-container { padding-top: 2rem; max-width: 1400px; }
.disclaimer { color: #5B6B82; font-size: 0.8rem; border-top: 1px solid #D8DEE7; padding-top: .75rem; margin-top: 2rem; max-width: 80ch; }
</style>""", unsafe_allow_html=True)


def _save_upload(upload) -> str | None:
    if upload is None:
        return None
    path = Path(tempfile.gettempdir()) / f"eod_{upload.name}"
    path.write_bytes(upload.getvalue())
    return str(path)


with st.sidebar:
    st.header("End-of-day scan")
    demo = st.toggle("Demo data (offline)", value=True)
    universe_up = st.file_uploader("Universe CSV (optional)", type="csv",
                                   help="Any CSV with a 'Symbol' or 'NSE Code' column, e.g. the Nifty 500 "
                                        "constituents list from niftyindices.com.")
    tickers = st.text_area("NSE symbols (used when no universe CSV)", ", ".join(DEFAULT_TICKERS), height=100)
    superstar_up = st.file_uploader("Superstar buys CSV (optional)", type="csv",
                                    help="Trendlyne 'Buys by Superstar Investors' export. These stocks are "
                                         "added to the scan and tagged.")
    st.subheader("Rules")
    lookback = st.number_input("Range lookback (sessions)", 20, 250, 125)
    vol_mult = st.number_input("BUY volume multiple", 1.0, 5.0, 2.0, 0.25)
    rsi_bounds = st.slider("RSI: SELL above / BUY below", 0, 100, (30, 70))
    bear_rule = st.radio("SELL volume rule", ["below_avg", "above_multiple"],
                         format_func=lambda r: "Below average (as saved in the guide)" if r == "below_avg"
                         else "High-volume breakdown (mirror of BUY)")
    equity = st.number_input("Account equity (₹)", 10_000.0, value=1_000_000.0, step=50_000.0)
    run = st.button("Run end-of-day scan", type="primary", width="stretch")
    st.caption("Run after 16:00 IST, once today's daily bar is available. Signals are for the next session.")

if run:
    overrides = {
        "tickers": [t for t in tickers.replace("\n", ",").split(",") if t.strip()],
        "universe_file": _save_upload(universe_up), "superstar_file": _save_upload(superstar_up),
        "breakout": {"lookback": int(lookback), "volume_multiple": vol_mult, "bear_rsi_min": float(rsi_bounds[0]),
                     "bull_rsi_max": float(rsi_bounds[1]), "bear_volume_rule": bear_rule},
        "risk": {"account_equity": equity},
    }
    if demo:
        overrides.update(output_dir="outputs/demo", data={"provider": "synthetic", "preopen_source": "synthetic"})
    try:
        cfg = load_config(overrides=overrides)
        setup_logging(cfg.log_dir, cfg.log_level)
        now = demo_now("16:30") if demo else datetime.now(IST)
        if demo:
            market = SyntheticMarket(now=now)
            fetcher = MarketDataFetcher(cfg.data, provider=market, preopen=market)
        else:
            fetcher = MarketDataFetcher(cfg.data)
        n = len(cfg.tickers) if not cfg.universe_file else "the universe file's"
        with st.spinner(f"Fetching history and backtesting the rules on {n} symbols. Large universes take a few minutes."):
            st.session_state["eod"] = (EodBreakoutPipeline(cfg, fetcher, now=now).run(), cfg)
            st.session_state["eod_error"] = None
    except (ConfigError, ValueError, DataFetchError) as exc:
        st.session_state["eod_error"] = str(exc)

st.title("End-of-day breakout signals")
if err := st.session_state.get("eod_error"):
    st.error(err)
if "eod" not in st.session_state:
    st.info("Configure the universe and rules in the sidebar, then select **Run end-of-day scan**. "
            "BUY = close above the prior 125-day high on 2× volume with RSI below 70. "
            "SELL = close below the prior 125-day low with RSI above 30.")
else:
    res, cfg = st.session_state["eod"]
    sig, stats = res.signals, res.rule_stats
    st.caption(f"Session {res.session:%A %d %B %Y}. Signals apply to the next session. "
               f"{len(res.histories)} symbols scanned." + (f" Skipped (no NSE symbol): {', '.join(res.skipped)}."
                                                          if res.skipped else ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BUY signals", int((sig["Signal"] == "BUY").sum()))
    c2.metric("SELL signals", int((sig["Signal"] == "SELL").sum()))
    for col, rule in ((c3, "BUY"), (c4, "SELL")):
        row = stats[stats["Rule"].str.startswith(rule)] if not stats.empty else stats
        if row.empty:
            col.metric(f"{rule} rule, avg R", "no trades")
        else:
            r = row.iloc[0]
            label = "too few trades" if r["Trades"] < MIN_TRADES else f"{r['Trades']} trades, {r['Win rate %']:.0f}% wins"
            col.metric(f"{rule} rule, avg R after costs", f"{r['Avg R']:+.2f}R", label, delta_color="off")

    tab_sig, tab_bt, tab_chart, tab_scan = st.tabs(["Signals", "Rule backtest", "Chart", "Full scan"])
    with tab_sig:
        if sig.empty:
            st.info("No BUY or SELL signals for this session.")
        else:
            st.dataframe(sig.style.map(lambda v: {"BUY": f"background-color:#DDEFE7;color:{C['long']};font-weight:600",
                                                  "SELL": f"background-color:#F6DDE0;color:{C['stop']};font-weight:600"}
                                       .get(v, ""), subset=["Signal"]),
                         hide_index=True, width="stretch", column_config={
                             "Close": st.column_config.NumberColumn(format="₹%.2f"),
                             "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
                             "Target Price": st.column_config.NumberColumn(format="₹%.2f"),
                             "Risk (INR)": st.column_config.NumberColumn("Risk", format="₹%.0f"),
                             "Max Position Units": st.column_config.NumberColumn("Quantity", format="%d"),
                             "Beyond Range %": st.column_config.NumberColumn(format="%.2f%%"),
                             "Volume x Avg": st.column_config.NumberColumn(format="%.2f×"),
                         })
            st.download_button("Download signals (CSV)", sig.to_csv(index=False).encode(),
                               f"eod_signals_{res.session}.csv", "text/csv")
            st.caption("BUY: entry next session near the open; re-check the pre-open IEP. SELL: exit or avoid "
                       "if held. Cash-segment shorts must be squared off the same day; overnight shorts need F&O.")
    with tab_bt:
        if stats.empty:
            st.info("These rules produced no historical trades on this universe.")
        else:
            st.dataframe(stats, hide_index=True, width="stretch")
            st.markdown("Each historical signal was entered at the next day's open, with a stop at "
                        f"{cfg.risk.atr_stop_multiplier}× ATR, a target at {cfg.risk.reward_risk_ratio}R and a "
                        f"{cfg.breakout.backtest_max_hold_days}-session time exit. Estimated delivery costs "
                        f"({100 * cfg.breakout.est_round_trip_cost_pct:.2f}% round trip) are deducted. "
                        "Avg R above zero means the rule made money on average; below zero means it lost.")
            st.warning("Survivorship bias: the backtest uses today's symbol list, so stocks that were delisted or "
                       "dropped from the index are missing. Real results would be somewhat worse.")
            if not res.trades.empty:
                t = res.trades.sort_values("exit_date")
                fig = go.Figure()
                for side, color in (("long", C["long"]), ("short", C["stop"])):
                    g = t[t["side"] == side]
                    if not g.empty:
                        fig.add_scatter(x=g["exit_date"], y=g["r_multiple"].cumsum(), mode="lines",
                                        name="BUY rule" if side == "long" else "SELL rule", line=dict(color=color))
                fig.add_hline(y=0, line_dash="dot", line_color=C["muted"])
                fig.update_layout(height=340, yaxis_title="Cumulative R", plot_bgcolor=C["panel"],
                                  paper_bgcolor=C["panel"], margin=dict(l=10, r=10, t=10, b=10),
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")
    with tab_chart:
        choices = sig["Ticker"].tolist() or sorted(res.histories)
        tk = st.selectbox("Ticker", choices)
        df = res.histories[tk].tail(250)
        f = breakout_frame(res.histories[tk], cfg.breakout).loc[df.index]
        fig = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                       increasing=dict(line=dict(color=C["long"]), fillcolor=C["long"]),
                                       decreasing=dict(line=dict(color=C["stop"]), fillcolor=C["stop"]), name=tk))
        fig.add_scatter(x=f.index, y=f["prior_high"], name=f"Prior {cfg.breakout.lookback}D high",
                        line=dict(color=C["ink"], width=1.2, dash="dot"))
        fig.add_scatter(x=f.index, y=f["prior_low"], name=f"Prior {cfg.breakout.lookback}D low",
                        line=dict(color=C["muted"], width=1.2, dash="dot"))
        for flag, sym, color, name in (("buy", "triangle-up", C["long"], "BUY signal"),
                                       ("sell", "triangle-down", C["stop"], "SELL signal")):
            pts = f[f[flag]]
            if not pts.empty:
                fig.add_scatter(x=pts.index, y=pts["close"], mode="markers", name=name,
                                marker=dict(symbol=sym, size=12, color=color, line=dict(color="white", width=1)))
        fig.update_layout(height=520, plot_bgcolor=C["panel"], paper_bgcolor=C["panel"], hovermode="x unified",
                          xaxis=dict(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])],
                                     gridcolor=C["grid"]),
                          yaxis=dict(tickprefix="₹", gridcolor=C["grid"]), margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.06))
        st.plotly_chart(fig, width="stretch")
    with tab_scan:
        st.dataframe(res.scan, hide_index=True, width="stretch")

st.markdown('<p class="disclaimer">Research and educational software, not investment advice. Backtests use '
            'historical data with survivorship bias and do not guarantee future results. No broker orders are '
            'placed. Consult a SEBI-registered adviser before investing.</p>', unsafe_allow_html=True)
