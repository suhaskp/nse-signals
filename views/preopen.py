"""Pre-open scan (automatic). Nothing to click or configure here.

The scan runs by itself every trading day from 09:08 IST, when NSE's pre-open price discovery fixes the IEP,
and re-checks every 2 minutes until the 09:15 open. Settings come from config/settings.yaml.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring.performance_log import PerformanceTracker
from screeners.screener import add_indicators
from views.engines import DEMO, background_running, get_engines, get_preopen, now_ist, progress_panel

C = {
    "ink": "#1B2A41", "muted": "#5B6B82", "paper": "#EEF2F6", "panel": "#FFFFFF",
    "grid": "#D8DEE7", "long": "#1F7A5C", "stop": "#B23A48", "skip": "#B7791F", "watch": "#7A8699",
}
SIGNAL_STYLE = {
    "LONG": f"background-color:#DDEFE7;color:{C['long']};font-weight:600",
    "SKIP_RISK_LIMIT": f"background-color:#F6EBD7;color:{C['skip']};font-weight:600",
    "WATCH": f"color:{C['watch']}",
}
def inr(value: float, decimals: int = 0) -> str:
    """Format rupees with Indian digit grouping: 1234567 -> ₹12,34,567."""
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    body = ",".join(groups + [tail]) if groups else tail
    return f"{sign}₹{body}" + (f".{frac}" if decimals else "")


def style_summary(df: pd.DataFrame) -> Any:
    return df.style.map(lambda v: SIGNAL_STYLE.get(v, ""), subset=["Signal"])


SUMMARY_COLUMN_CONFIG = {
    "Current Price": st.column_config.NumberColumn("Entry (IEP)", format="₹%.2f"),
    "Target Price": st.column_config.NumberColumn("Target", format="₹%.2f"),
    "Stop Loss": st.column_config.NumberColumn("Stop", format="₹%.2f"),
    "ATR": st.column_config.NumberColumn(format="%.2f"),
    "Model Probability Score": st.column_config.ProgressColumn(
        "P(close > open)", min_value=0.0, max_value=1.0, format="%.2f"),
    "Max Position Units": st.column_config.NumberColumn("Units", format="%d"),
    "Gap %": st.column_config.NumberColumn(format="%.2f%%"),
    "RVOL": st.column_config.NumberColumn(format="%.2f×"),
    "RSI": st.column_config.NumberColumn(format="%.1f"),
    "RVOL Source": st.column_config.TextColumn("RVOL from"),
    "Notional (INR)": st.column_config.NumberColumn("Value", format="₹%.0f"),
    "Risk (INR)": st.column_config.NumberColumn("Risk", format="₹%.0f"),
    "Est. Cost (INR)": st.column_config.NumberColumn("Est. costs", format="₹%.0f"),
    "R:R": st.column_config.NumberColumn(format="%.1f"),
    "Constraint": st.column_config.TextColumn("Sized by"),
}


def trade_chart(ticker: str, row: pd.Series, daily: pd.DataFrame, cfg: Any, run_time: datetime) -> go.Figure:
    """Candles for the last 90 sessions, moving averages, and the trade's risk and reward zones."""
    ind = add_indicators(daily, cfg.indicators).tail(90)
    entry, stop, target = row["Current Price"], row["Stop Loss"], row["Target Price"]
    today = pd.Timestamp(run_time.date())  # IEP plotted on today's session
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"], close=ind["Close"], name=ticker,
        increasing=dict(line=dict(color=C["long"]), fillcolor=C["long"]),
        decreasing=dict(line=dict(color=C["stop"]), fillcolor=C["stop"]),
    ))
    for col, color, dash in [(f"SMA_{cfg.indicators.sma_fast}", C["ink"], "solid"),
                             (f"SMA_{cfg.indicators.sma_slow}", C["muted"], "dot")]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=col.replace("_", " "),
                                 line=dict(color=color, width=1.4, dash=dash)))
    x0, x1 = ind.index[-15], today + pd.Timedelta(days=3)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=stop, y1=entry, fillcolor=C["stop"], opacity=0.12, line_width=0)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=entry, y1=target, fillcolor=C["long"], opacity=0.12, line_width=0)
    for y, label, color in [(target, "Target", C["long"]), (entry, "Entry", C["ink"]), (stop, "Stop", C["stop"])]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=color, width=1.5))
        fig.add_annotation(x=x1, y=y, text=f"{label} {inr(y, 2)}", showarrow=False, xanchor="left",
                           font=dict(color=color, size=12), xshift=6)
    fig.add_trace(go.Scatter(x=[today], y=[entry], mode="markers", name="Pre-open IEP",
                             marker=dict(size=11, color=C["ink"], symbol="diamond", line=dict(color="white", width=1.5))))
    fig.update_layout(
        height=520, margin=dict(l=10, r=110, t=10, b=10), plot_bgcolor=C["panel"], paper_bgcolor=C["panel"],
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=C["ink"]),
        xaxis=dict(rangeslider_visible=False, gridcolor=C["grid"], rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(gridcolor=C["grid"], tickprefix="₹", side="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), hovermode="x unified",
    )
    return fig



_, _, cfg = get_engines()
pre = get_preopen()
st.title("Pre-open scan")


@st.fragment(run_every=timedelta(seconds=30))
def view() -> None:
    if not background_running():
        try:
            pre.refresh(now_ist())
        except Exception as exc:  # noqa: BLE001
            st.error(f"The pre-open scan failed: {exc}. It will retry automatically.")
    result = pre.peek()
    st.caption("Runs automatically every trading day from 09:08 IST (when the pre-open price is final) and re-checks "
               "every 2 minutes until the 09:15 open. " + pre.status + (" · DEMO MODE: synthetic data" if DEMO else ""))
    if result is None:
        progress_panel(pre, "the first pre-open scan")
        return
    today = now_ist().date()
    age = "today" if result.run_time.date() == today else f"on {result.run_time:%A %d %b}"
    st.markdown(f'<p class="run-line">Last scan {age} at {result.run_time:%H:%M} IST. {result.timing_note}</p>',
                unsafe_allow_html=True)
    summary, scan = result.summary, result.scan
    longs = summary[summary["Signal"] == "LONG"]
    total_risk = float(longs["Risk (INR)"].sum()) if not longs.empty else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Passed the screen", f"{len(summary)} of {len(scan)}")
    m2.metric("LONG signals", len(longs))
    m3.metric("Open risk if all taken", inr(total_risk),
              f"{100 * total_risk / cfg.risk.account_equity:.2f}% of equity", delta_color="off", delta_arrow="off")
    if result.cv_report:
        m4.metric("Model AUC (walk-forward)", f"{result.cv_report['summary']['mean_auc']:.3f}", "0.500 is chance",
                  delta_color="off", delta_arrow="off")
    else:
        m4.metric("Model", "Loaded from disk", delta_color="off", delta_arrow="off")

    tab_list, tab_chart, tab_scan, tab_hist, tab_cfg = st.tabs(
        ["Watch list", "Trade chart", "Full scan", "History", "Settings in use"])
    with tab_list:
        if summary.empty:
            st.info("Nothing passed the pre-open screen.")
        else:
            st.dataframe(style_summary(summary), column_config=SUMMARY_COLUMN_CONFIG, hide_index=True, width="stretch")
            st.caption("LONG: passed the screen and the model threshold, sized within portfolio limits. These are the "
                       "same-day gap signals; the main trading decision is on the Market intelligence page.")
    with tab_chart:
        if summary.empty:
            st.info("Charts appear once a stock passes the screen.")
        else:
            tk = st.selectbox("Ticker", summary["Ticker"].tolist())
            row = summary.set_index("Ticker").loc[tk]
            st.plotly_chart(trade_chart(tk, row, result.histories[tk], cfg, result.run_time), width="stretch")
    with tab_scan:
        st.dataframe(scan.rename(columns={"failures": "Why it failed"}), hide_index=True, width="stretch")
    with tab_hist:
        tracker = PerformanceTracker(cfg.output_dir, cfg.risk.est_round_trip_cost_pct)
        results, stats = tracker.evaluate(result.histories)
        if results.empty:
            st.info("Past LONG signals are evaluated here automatically once their session has closed.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Trades", stats["n_trades"])
            b.metric("Target hit rate", f"{stats['target_hit_rate']:.0%}")
            c.metric("Average R", f"{stats['avg_r_multiple']:+.2f}")
            d.metric("Total R", f"{stats['total_r']:+.1f}")
            st.dataframe(results, hide_index=True, width="stretch")
    with tab_cfg:
        s, r = cfg.screener, cfg.risk
        st.markdown(
            f"- Stocks: {len(cfg.tickers)} ({', '.join(cfg.tickers[:8])}{' …' if len(cfg.tickers) > 8 else ''})\n"
            f"- Pre-open source: {cfg.data.preopen_source}\n"
            f"- Gap {s.min_gap_pct:g}–{s.max_gap_pct:g}%, relative volume ≥ {s.min_rvol:g}, RSI {s.rsi_min:g}–{s.rsi_max:g}, "
            f"turnover ≥ ₹{s.min_avg_turnover_cr:g} crore\n"
            f"- Equity ₹{r.account_equity:,.0f}, risk {100 * r.risk_per_trade_pct:.2f}% per trade, stop "
            f"{r.atr_stop_multiplier:g}× ATR, target {r.reward_risk_ratio:g}R, max {r.max_open_positions} positions\n"
            f"- Record LONG signals to the paper ledger: {'yes' if cfg.execution.record_paper_orders else 'no'}")
        st.caption("To change any of these, edit config/settings.yaml (copy config/settings.example.yaml) and restart "
                   "the dashboard.")


view()
