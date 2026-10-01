"""Live signals: opens straight onto today's BUY and SELL lists, refreshes itself.

No buttons are needed. The page loads data on open, then re-checks every
``live_refresh_seconds`` while the market is open (every 30 minutes otherwise).
Set PIPELINE_DEMO=1 to run on offline synthetic data.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from data_ingestion.data_fetcher import DataFetchError
from views.engines import DEMO, background_running, get_engines, now_ist, progress_panel
from research.movers import verdict as movers_verdict
from views.charts import C, range_chart

PHASE_BADGE = {"open": ("🟢", "Market open"), "settling": ("🟡", "Closing"), "after_close": ("⚪", "Closed"),
               "pre_market": ("🔵", "Pre-market"), "closed": ("⚫", "Market holiday")}


def signal_table(df: pd.DataFrame) -> None:
    cols = [c for c in ["Ticker", "Close", "Stop Loss", "Target Price", "Max Position Units", "Risk (INR)",
                        "RS vs Nifty 3M %", "RS today %", "Volume x Avg", "RSI", "Superstar Buy", "Note"] if c in df.columns]
    st.dataframe(df[cols], hide_index=True, width="stretch", column_config={
        "Close": st.column_config.NumberColumn("Price", format="₹%.2f"),
        "Stop Loss": st.column_config.NumberColumn("Stop", format="₹%.2f"),
        "Target Price": st.column_config.NumberColumn("Target", format="₹%.2f"),
        "Max Position Units": st.column_config.NumberColumn("Qty", format="%d"),
        "Risk (INR)": st.column_config.NumberColumn("Risk", format="₹%.0f"),
        "Volume x Avg": st.column_config.NumberColumn("Vol ×avg", format="%.1f×"),
        "RSI": st.column_config.NumberColumn(format="%.0f"),
        "Superstar Buy": st.column_config.TextColumn("Superstar"),
        "Ticker Hist. Trades": st.column_config.NumberColumn("Past signals", format="%d"),
        "Ticker Hist. Avg R": st.column_config.NumberColumn("Past avg R", format="%+.2f"),
        "RS vs Nifty 3M %": st.column_config.NumberColumn("RS vs Nifty 3M", format="%+.1f%%",
            help="3-month return minus the Nifty's. Positive = stronger than the market."),
        "RS today %": st.column_config.NumberColumn("RS today", format="%+.2f%%",
            help="Today's change minus the Nifty's change today."),
    })


def health_box(snap, rule: str) -> None:
    status, msg = snap.rule_health(rule)
    {"proven": st.success, "losing": st.error, "unproven": st.warning, "none": st.info}[status](msg)


intel_engine, engine, cfg = get_engines()
st.title("Buy and sell signals")


@st.fragment(run_every=timedelta(seconds=20))
def live_view() -> None:
    if background_running():  # the background does the work; the page only displays, never waits
        snap = engine.peek()
        if snap is None:
            progress_panel(engine, "the breakout signals")
            return
        render(snap)
        return
    first = engine._last is None
    try:
        if first:
            with st.spinner("Loading market data. The very first start downloads about 5 years of history for "
                            "the Nifty 500 and takes a few minutes; after that it opens in seconds."):
                snap = engine.refresh(now_ist())
        else:
            snap = engine.refresh(now_ist())
    except DataFetchError as exc:
        st.error(f"{exc} The page will retry automatically.")
        return
    render(snap)


def movers_evidence() -> None:
    """Does buying movers work? Historical proxy now, live follow-up as it accumulates."""
    rep = intel_engine.peek()
    st.subheader("Does buying today's movers actually work?")
    if rep is None or not rep.movers_proxy:
        st.caption("The historical test appears here once the daily research run has finished.")
        return
    proxy = rep.movers_proxy
    level, text = movers_verdict(proxy)
    if not proxy["summary"].empty:
        from research.evidence import label as evidence_label
        r5 = proxy["summary"].set_index("Hold").loc["5 sessions"]
        text += f" 5-session win rate: {evidence_label(r5['Win %'], int(r5['Mover trades']))}."
    {"ok": st.success, "bad": st.warning, "none": st.info}[level](text)
    if not proxy["summary"].empty:
        st.markdown(f"**Historical test** ({proxy['rule']}; daily bars, because intraday history is not free)")
        st.dataframe(proxy["summary"], hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%+.2f%%") for c in
            ("Avg return %", "Median %", "Avg vs Nifty %", "Typical stock avg %", "Movers minus typical %")})
        with st.expander("5-session result by year"):
            st.dataframe(proxy["by_year"], hide_index=True, width="stretch")
    fu = rep.movers_followup or {}
    st.markdown("**Live follow-up of the movers shown on this page**")
    st.caption("Each trading day the movers list is recorded at 12:00 and 15:15 IST with the price shown then, and "
               "followed: did the gain hold to that day's close, and where was it 1, 5 and 10 sessions later?")
    if fu.get("summary") is None or fu["summary"].empty:
        st.info(f"Recording started; {fu.get('days', 0)} day(s) logged so far. Results appear after the first close "
                "and fill in over the next two weeks.")
    else:
        st.dataframe(fu["summary"], hide_index=True, width="stretch", column_config={
            "Avg %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Avg vs Nifty %": st.column_config.NumberColumn(format="%+.2f%%")})
        with st.expander(f"Every recorded mover ({len(fu['rows'])})"):
            st.dataframe(fu["rows"], hide_index=True, width="stretch")
        st.caption("A few weeks of results are still a small sample; read them together with the historical test.")


def render(snap) -> None:
    icon, label = PHASE_BADGE.get(snap.phase, ("", snap.phase))
    next_in = cfg.live_refresh_seconds // 60 if snap.phase in ("open", "settling") else 30
    what = (f"live signals on today's bar (provisional)" if snap.mode == "live"
            else f"confirmed signals from {snap.session:%a %d %b}, for the next session")
    st.markdown(f"**{icon} {label}** · {what} · updated {snap.as_of:%H:%M} IST · rechecks every {next_in} min")
    st.caption(f"{snap.phase_note} Universe: {snap.universe_label}. Superstar: {snap.superstar_label}. "
               "Prices from Yahoo Finance may be delayed; they are not tick-by-tick."
               + (" DEMO MODE: synthetic data." if DEMO else ""))

    buys = snap.signals[snap.signals["Signal"] == "BUY"]
    sells = snap.signals[snap.signals["Signal"] == "SELL"]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(f"BUY ({len(buys)})")
        q = cfg.buy_quality
        st.caption(f"Quality breakouts: every check passed (breakout above the {q.breakout_lookback}-day high, "
                   f"volume ≥ {q.volume_multiple:g}× average, uptrend, ATR stop, Nifty above its 200-day average, "
                   f"reward/risk ≥ {q.min_reward_risk:g}:1). Buy only at the next open within the stated range. "
                   "Intraday, volume is still building, so fewer stocks pass before the close.")
        health_box(snap, "BUY")
        if not buys.empty:
            signal_table(buys)
        else:
            st.write("No BUY signals right now.")
    with right:
        sell_status, _ = snap.rule_health("SELL")
        if sell_status in ("losing", "unproven", "none"):
            st.subheader(f"EXIT warnings ({len(sells)})")
            st.caption("⚠️ UNVALIDATED: the SELL rule has not made money historically, so these are warnings to "
                       "review stocks you hold, not recommendations to short.")
        else:
            st.subheader(f"SELL ({len(sells)})")
        st.caption("Closed below the prior 125-day low, RSI above 30. Exit or avoid if held; "
                   "overnight shorts need F&O.")
        health_box(snap, "SELL")
        if not sells.empty:
            signal_table(sells)
        else:
            st.write("No SELL signals right now.")

    if snap.movers is not None:
        st.subheader("Today's strongest movers")
        st.caption("Top gainers today trading at least their normal volume pace for this time of day. Intraday and "
                   "provisional (Yahoo prices may be delayed). A mover is not a BUY signal: most big up-days are not "
                   "the start of a trend, and chasing them is how the 'not extended' and next-day-entry checks lose "
                   "money for people.")
        if snap.movers.empty:
            st.write("No stock is up today on above-normal volume.")
        else:
            st.dataframe(snap.movers, hide_index=True, width="stretch", column_config={
                "Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Today %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Volume pace": st.column_config.NumberColumn(format="%.1f×", help="Today's volume so far vs the "
                                                             "20-day average, adjusted for the time of day."),
                "Near day high": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

    movers_evidence()

    tickers = snap.signals["Ticker"].tolist()
    if tickers:
        st.subheader("Chart")
        tk = st.selectbox("Stock", tickers, format_func=lambda t: f"{t} ({snap.signals.set_index('Ticker').at[t, 'Signal']})")
        st.plotly_chart(range_chart(snap.histories[tk], cfg.breakout, snap.signals.set_index("Ticker").loc[tk]),
                        width="stretch")

    if snap.mode == "live":
        with st.expander(f"Confirmed signals from the last completed session ({snap.confirmed_session:%d %b})"):
            st.dataframe(snap.confirmed.drop(columns=["Note"]), hide_index=True, width="stretch")
    with st.expander("How these rules have performed (backtest on this universe)"):
        if snap.rule_stats.empty:
            st.write("No historical trades yet.")
        else:
            st.dataframe(snap.rule_stats, hide_index=True, width="stretch")
            st.caption("Entry at the next open, stop 1.5× ATR, target 2R, 20-session time exit, ~0.25% costs. "
                       "Uses today's stock list, so delisted stocks are missing (survivorship bias).")
    with st.expander("What runs automatically"):
        st.markdown(
            "- The Nifty 500 list is downloaded from NSE and refreshed weekly.\n"
            "- Price history is cached on disk; each refresh downloads only recent days.\n"
            f"- During market hours the page rechecks every {cfg.live_refresh_seconds // 60} minutes; "
            "otherwise every 30 minutes.\n"
            "- Superstar list: save any Trendlyne *Buys by Superstar Investors* export into "
            f"`{cfg.superstar_dir}`. The newest file is picked up automatically (Trendlyne needs a login, "
            "so it cannot be downloaded for you).\n"
            "- Every confirmed list is saved to `outputs/eod_signals_<date>.csv`.")


live_view()
st.markdown(f'<p style="color:{C["muted"]};font-size:0.8rem;border-top:1px solid {C["grid"]};padding-top:.75rem;'
            'margin-top:2rem;max-width:80ch">Signals are rule-based screens, not predictions or investment advice, '
            'and no system can guarantee profitable trades. Backtests do not guarantee future results. No broker '
            'orders are placed. Consult a SEBI-registered adviser before investing.</p>', unsafe_allow_html=True)
