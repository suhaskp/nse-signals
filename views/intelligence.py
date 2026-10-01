"""Market intelligence: research, predictive ranking, validated BUY/SELL ideas. Loads by itself.

Recomputed once per session (after 16:00 IST for today's close; otherwise from the last
completed session) and rechecked every 30 minutes. Set PIPELINE_DEMO=1 for offline demo data.
"""
from __future__ import annotations

from datetime import timedelta

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_ingestion.data_fetcher import DataFetchError
from intelligence_engine import morning_check
from views.engines import DEMO, background_running, get_engines, now_ist, progress_panel
from models.ranker import LABELS, explain
from research.market import stock_profile
from research.stress import stress_test, verdict
from research.evidence import label as evidence_label, strength as evidence_strength
import decisions
from views.charts import C, range_chart

STATE = {"bull": ("🟢", "Uptrend"), "neutral": ("🟡", "Mixed"), "bear": ("🔴", "Downtrend")}
CONV_STYLE = {"High": f"background-color:#DDEFE7;color:{C['long']};font-weight:600",
              "Medium": "background-color:#EEF4EA;color:#3F6B2F;font-weight:600",
              "Low": f"color:{C['muted']}", "Unvalidated": "background-color:#F6EBD7;color:#8A5A12;font-weight:600"}


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{digits}f}%"


def rec_table(df: pd.DataFrame) -> None:
    buy = (df["Action"] == "BUY").any()
    cols = (["Ticker", "Conviction", "Model score", "Price", "Entry rule", "Stop Loss", "Target Price", "R:R", "Qty",
             "Risk (INR)", "Allocation", "Review by", "Checks", "Why", "Sector", "Superstar"] if buy else
            ["Ticker", "Conviction", "Model score", "Price", "Stop Loss", "Target Price", "Qty", "Risk (INR)",
             "Allocation", "Hist. excess %", "Hist. beat %", "Review by", "Why", "Checks", "Sector", "Superstar"])
    st.dataframe(df[cols].style.map(lambda v: CONV_STYLE.get(v, ""), subset=["Conviction"]),
                 hide_index=True, width="stretch", column_config={
                     "Model score": st.column_config.ProgressColumn("Rank (percentile)", min_value=0, max_value=100,
                                                                    format="%.0f / 100", help="Percentile rank among all stocks today, not a probability or confidence. 100 = ranked above about 99% of stocks. See Hist. beat peers for how often stocks with this rank actually outperformed."),
                     "Price": st.column_config.NumberColumn(format="₹%.2f"),
                     "Stop Loss": st.column_config.NumberColumn("Stop", format="₹%.2f"),
                     "Target Price": st.column_config.NumberColumn("Target", format="₹%.2f"),
                     "Risk (INR)": st.column_config.NumberColumn("Risk", format="₹%.0f"),
                     "Hist. excess %": st.column_config.NumberColumn("Hist. 10d vs peers", format="%+.2f%%",
                         help="Average 10-session return versus the median stock, for stocks in this score "
                              "decile during out-of-sample testing."),
                     "Hist. beat %": st.column_config.NumberColumn("Hist. beat peers", format="%.0f%%"),
                     "Why": st.column_config.TextColumn(width="large"),
                     "R:R": st.column_config.NumberColumn("Reward/risk", format="%.1f : 1"),
                     "Entry rule": st.column_config.TextColumn(width="medium"),
                 })


def morning_table(check: pd.DataFrame | None, action: str, note: str) -> None:
    """Today's opening check for one side: recalculated levels and skip warnings."""
    if check is None:
        return
    part = check[check["Action"] == action]
    if part.empty:
        return
    ok = int((part["Status"] == "OK").sum())
    st.markdown(f"**Today's opening check** · {ok} of {len(part)} still OK · {note}")
    style = lambda v: (f"color:{C['long']};font-weight:600" if v == "OK" else  # noqa: E731
                       f"color:{C['stop']};font-weight:600" if str(v).startswith("SKIP") else
                       "color:#8A5A12;font-weight:600" if str(v).startswith("CAUTION") else "")
    st.dataframe(part.drop(columns=["Action"]).style.map(style, subset=["Status"]), hide_index=True,
                 width="stretch", column_config={
                     "Yesterday close": st.column_config.NumberColumn(format="₹%.2f"),
                     "Today price": st.column_config.NumberColumn(format="₹%.2f"),
                     "Gap %": st.column_config.NumberColumn(format="%+.2f%%"),
                     "Entry": st.column_config.NumberColumn(format="₹%.2f"),
                     "Stop Loss": st.column_config.NumberColumn("Stop", format="₹%.2f"),
                     "Target Price": st.column_config.NumberColumn("Target", format="₹%.2f"),
                     "Risk (INR)": st.column_config.NumberColumn("Risk", format="₹%.0f")})
    st.caption("Entry, stop, target and quantity are recalculated from today's price. The full idea list "
               "from the last close follows.")


VARIANT_NAMES = {"quality": "Quality BUY rules", "quality_no_regime": "Quality BUY rules, no regime check",
                 "quality_3regime": "Quality BUY rules, three regimes (half size neutral)", "original": "Chartink original", "regime": "Chartink + regime",
                 "trailing": "Chartink + regime + trailing", "no_rsi_cap": "No RSI cap + regime + trailing",
                 "sell_original": "Chartink SELL (short)"}


def evidence_summary(rep) -> None:
    """Plain-language reading of the filter and regime studies on this universe."""
    lines = []
    ab = rep.ablation
    if ab is not None and not ab.empty:
        body = ab.iloc[1:]
        good = body.loc[body["Effect"] == "improves results", "Stage"].str.lstrip("+ ").tolist()
        bad = body.loc[body["Effect"] == "hurts results", "Stage"].str.lstrip("+ ").tolist()
        cut = body.loc[body["Effect"] == "mostly just cuts trades", "Stage"].str.lstrip("+ ").tolist()
        if good:
            lines.append("✅ Checks that improve results: " + ", ".join(good) + ".")
        if cut:
            lines.append("➖ Checks that mostly just reduce the number of trades: " + ", ".join(cut) + ".")
        if bad:
            lines.append("❌ Checks that made results worse on this history: " + ", ".join(bad) + ".")
    rb = rep.regime_breakdown
    if rb is not None and not rb.empty:
        rbi = rb.set_index("Regime at signal")
        on = next((i for i in rbi.index if i.startswith("Risk-on")), None)
        off = next((i for i in rbi.index if i.startswith("Risk-off")), None)
        if on and off and rbi.at[on, "Trades"] >= 30 and rbi.at[off, "Trades"] >= 30:
            a, b = rbi.at[on, "Avg R"], rbi.at[off, "Avg R"]
            verdict = ("so the market-regime check is supported by the evidence" if a > b + 0.05 else
                       "so the regime check is NOT clearly helping on this history" if b >= a else
                       "a small difference; the regime check is at best mildly helpful")
            lines.append(f"{'✅' if a > b + 0.05 else '⚠️'} BUY rules returned {a:+.2f}R per trade in risk-on markets vs "
                         f"{b:+.2f}R in risk-off markets, {verdict}.")
    lab = rep.lab_summary
    if lab is not None and not lab.empty and "quality" in set(lab["Variant"]):
        qrow = lab.set_index("Variant").loc["quality"]
        lines.append(f"Overall, the quality BUY rules are rated **{qrow['Verdict']}** ({qrow['Trades']} trades, "
                     f"{qrow['Avg R']:+.2f}R per trade, positive in {qrow['Positive years %']:.0f}% of years).")
    if not lines:
        return
    with st.container(border=True):
        st.markdown("**What the evidence says (this universe, after costs)**")
        st.markdown("\n".join(f"- {x}" for x in lines))
        st.caption("Backtests use today's stock list (survivorship bias) and are optimistic. Change live rules only after "
                   "the rule freeze, and only if the forward record agrees.")


def filter_studies(rep) -> None:
    """Does each check earn its place, and does the market regime really matter?"""
    ab = rep.ablation
    if ab is not None and not ab.empty:
        st.markdown("### Each BUY check, added one at a time")
        st.caption("Every row adds one more check to the one above and re-runs the backtest. A check that only "
                   "cuts trades without improving results is adding caution, not edge. Effect compares average R "
                   "with the row above (±0.03R counts as a change).")
        st.dataframe(ab.style.map(lambda x: {"improves results": f"color:{C['long']};font-weight:600",
                                             "hurts results": f"color:{C['stop']};font-weight:600"}.get(x, ""),
                                  subset=["Effect"]), hide_index=True, width="stretch")
    rb = rep.regime_breakdown
    if rb is not None and not rb.empty:
        st.markdown("### Results by market regime")
        st.caption("The BUY rules run without the regime check, split by the market condition on the signal day. "
                   "This shows directly whether buying in a weak market has worked on this history.")
        st.dataframe(rb.assign(Evidence=rb["Trades"].map(lambda n: evidence_strength(int(n)))), hide_index=True,
                     width="stretch")
        st.caption("To trade a neutral market at half size, set buy_quality: regime_mode: three_state in "
                   "config/settings.yaml, but only if the 'three regimes' variant holds up in the table and "
                   "stress test above.")


def stress_section(rep) -> None:
    """Drawdown stress test for one strategy variant."""
    st.markdown("### Drawdown stress test")
    st.caption("How bad the bad stretches get, not just the average: the historical worst case, losing streaks, "
               "every year, and 5,000 reshuffles of the trades (Monte Carlo) to show the realistic range.")
    trades = rep.lab_trades
    if trades is None or trades.empty:
        st.info("No trades to stress-test yet.")
        return
    counts = trades["variant"].value_counts()
    options = [v for v in VARIANT_NAMES if counts.get(v, 0) >= 5]
    if not options:
        st.info("Every strategy has fewer than 5 historical trades, too few to stress-test.")
        return
    c1, c2, c3 = st.columns(3)
    variant = c1.selectbox("Strategy", options, format_func=lambda v: f"{VARIANT_NAMES[v]} ({counts[v]} trades)",
                           index=options.index("quality") if "quality" in options else 0, key="stress_variant")
    risk = c2.number_input("Risk per trade (% of equity)", 0.25, 5.0, 100 * cfg.risk.risk_per_trade_pct, 0.25,
                           key="stress_risk") / 100
    tol = c3.slider("Largest drawdown you would accept (%)", 5, 40, 15, key="stress_tol") / 100
    res = stress_test(trades[trades["variant"] == variant], risk, tol)
    if res is None:
        st.info("Too few trades for this strategy.")
        return
    level, text = verdict(res, risk, tol)
    {"ok": st.success, "caution": st.warning, "danger": st.error}[level](text)

    m1 = res.mc["one_year"]
    a, b, c, d = st.columns(4)
    a.metric("Worst historical drawdown", f"{100 * res.max_dd_pct:.1f}%", f"{res.max_dd_r:.1f}R", delta_color="off", delta_arrow="off")
    b.metric("Bad-year drawdown (1 in 20)", f"{100 * m1['dd_p95']:.1f}%",
             f"typical {100 * m1['dd_median']:.1f}%", delta_color="off", delta_arrow="off")
    c.metric("Longest losing streak", f"{res.longest_losing_streak} trades",
             f"1-in-20 year: {m1['streak_p95']:.0f}", delta_color="off", delta_arrow="off")
    d.metric("Chance of a losing year", f"{100 * m1['p_loss']:.0f}%",
             f"range {100 * m1['ret_p5']:+.0f}% to {100 * m1['ret_p95']:+.0f}%", delta_color="off", delta_arrow="off")
    st.caption(f"Average winner {res.avg_win_r:+.2f}R, average loser {res.avg_loss_r:+.2f}R, expectancy {res.avg_r:+.2f}R per "
               f"trade, about {res.trades_per_year:.0f} trades a year. Best year {res.best_year}, worst year {res.worst_year}.")
    st.caption(f"Worst drawdown ran from {res.dd_peak} to {res.dd_trough}; "
               + (f"recovered on {res.dd_recovered} ({res.dd_days_to_recover} days)."
                  if res.dd_recovered else "not yet recovered.")
               + f" Longest time without a new high: {res.longest_flat_days} days. Up to {res.max_concurrent} trades "
                 "overlapped at once; the test treats trades one after another, so real drawdowns can arrive faster.")

    left, right = st.columns([3, 2])
    with left:
        cv = res.curve
        fig = go.Figure()
        fig.add_scatter(x=cv["date"], y=100 * cv["equity"], name="Equity (₹100 start)", line=dict(color=C["long"], width=2))
        fig.add_scatter(x=cv["date"], y=cv["drawdown_pct"], name="Drawdown %", yaxis="y2", fill="tozeroy",
                        line=dict(color=C["stop"], width=1), fillcolor="rgba(178,58,72,0.18)")
        fig.update_layout(height=340, plot_bgcolor=C["panel"], paper_bgcolor=C["panel"],
                          margin=dict(l=10, r=10, t=30, b=10), title="Historical equity and drawdown",
                          yaxis=dict(title="Equity", gridcolor=C["grid"]),
                          yaxis2=dict(title="Drawdown %", overlaying="y", side="right", showgrid=False,
                                      range=[min(-5, 1.6 * cv["drawdown_pct"].min()), 0]),
                          legend=dict(orientation="h", y=-0.15), font=dict(family="IBM Plex Sans, sans-serif"))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Histogram(x=100 * res.mc_dd_pct, nbinsx=40, marker_color=C["watch"]))
        fig.add_vline(x=-100 * tol, line_color=C["stop"], line_dash="dash",
                      annotation_text=f"your limit {100 * tol:.0f}%", annotation_position="top left")
        fig.update_layout(height=340, plot_bgcolor=C["panel"], paper_bgcolor=C["panel"],
                          margin=dict(l=10, r=10, t=30, b=10), title="Worst drawdown in a year: 5,000 simulations",
                          xaxis_title="Drawdown %", yaxis_title="Simulations", font=dict(family="IBM Plex Sans, sans-serif"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{100 * m1['p_dd_beyond_tolerance']:.0f}% of simulated years went beyond your limit.")
    st.markdown("**Year by year**")
    st.dataframe(res.by_year, hide_index=True, width="stretch", column_config={
        "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Worst drawdown %": st.column_config.NumberColumn(format="%.1f%%")})


def trending_section(rep) -> None:
    """Leadership stocks in their own uptrends: research, not BUY signals."""
    lead = rep.leaders
    st.markdown("**Trending now: stocks in their own strong uptrend** (research list, not a BUY signal)")
    risk_off = bool(rep.regime and not rep.regime["risk_on"])
    st.caption("Close above a rising 200-day and the 50-day, within 5% of the 52-week high, and beating the Nifty over "
               "3 months: stocks holding up while the market falls. They are usually the first to produce real BUY "
               "signals when the market turns. 'Still needs' lists the BUY checks each one fails today"
               + ("; the market regime blocks all of them right now." if risk_off else "."))
    if lead is None or lead.empty:
        st.info("No stock currently meets the trend-leader criteria.")
        return
    close = lead[lead["_regime_only"]]
    if len(close) and risk_off:
        st.success(f"Closest to a BUY: {', '.join(close['Ticker'])} pass every other check and would be BUY "
                   "signals if the Nifty were above its 200-day average.")
    if rep.quality is not None and not rep.quality.empty:
        lead = lead.merge(rep.quality[["Ticker", "Readiness", "State", "Breakout status"]], on="Ticker", how="left")
        lead["State"] = lead["State"].map(lambda x: f"{STATE_ICON.get(x, '')} {x}" if isinstance(x, str) else x)
        front = ["Ticker", "State", "Readiness", "Breakout status"]
        lead = lead[front + [c for c in lead.columns if c not in front]]
    st.dataframe(lead.drop(columns=["_regime_only"]), hide_index=True, width="stretch", column_config={
        "Readiness": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Breakout level": st.column_config.NumberColumn(format="₹%.2f"),
        "1M %": st.column_config.NumberColumn(format="%+.1f%%"), "3M %": st.column_config.NumberColumn(format="%+.1f%%"),
        "vs Nifty 3M %": st.column_config.NumberColumn(format="%+.1f%%"),
        "From 52w high %": st.column_config.NumberColumn(format="%.1f%%"),
        "To breakout %": st.column_config.NumberColumn(format="%+.1f%%", help="How far price must rise to close above the "
                                                         "prior 55-day high (negative = already above)."),
        "Rank": st.column_config.NumberColumn("Rank /100", format="%.0f"),
        "Still needs": st.column_config.TextColumn(width="large")})
    lab = rep.lab_summary
    if lab is not None and not lab.empty and {"quality", "quality_no_regime"} <= set(lab["Variant"]):
        a = lab.set_index("Variant").loc["quality"]
        b = lab.set_index("Variant").loc["quality_no_regime"]
        better = b["Avg R"] > a["Avg R"]
        st.caption(f"Evidence on the regime check: with it, {a['Trades']} trades at {a['Avg R']:+.2f}R each; without it "
                   f"(buying in downtrends too), {b['Trades']} trades at {b['Avg R']:+.2f}R each. "
                   + ("On this history, skipping the regime check would have done better per trade; to switch it off, set "
                      "buy_quality: regime_required: false in config/settings.yaml. Judge it in the Strategy lab and "
                      "stress test first." if better else
                      "On this history the regime check improved results per trade, so keeping it on is supported."))


def quality_health(rep) -> None:
    """Verdict on the quality BUY rules from their backtest on this universe."""
    lab = rep.lab_summary
    row = lab[lab["Variant"] == "quality"] if lab is not None and not lab.empty else pd.DataFrame()
    if row.empty:
        st.info("The quality BUY rules have no historical trades on this universe yet, so there is no evidence either way.")
        return
    r = row.iloc[0]
    msg = (f"Quality BUY rules, backtested on this universe: {r['Trades']} trades (evidence "
           f"{evidence_strength(int(r['Trades']))}), {r['Win %']:.0f}% winners, "
           f"{r['Avg R']:+.2f}R per trade after costs, profit factor {r['Profit factor']:.2f}, positive in "
           f"{r['Positive years %']:.0f}% of years. Verdict: **{r['Verdict']}**.")
    {"Passes": st.success, "Marginal": st.warning, "Fails": st.error}.get(r["Verdict"], st.info)(msg)


def track_record(rep) -> None:
    """The forward record: every BUY signal saved when issued, with its rule version, then followed."""
    t = rep.tracking
    if not t:
        return
    fr = t.get("freeze", {})
    summ = t.get("summary")
    version = fr.get("current_version", "")
    if summ is None or summ.empty:
        st.caption(f"Forward record: started {t['start']}. Every BUY signal issued from now on is saved with its rule "
                   f"version ({version}) and followed automatically: entry only at the next open inside the range, "
                   "then stop, target or time exit.")
        return
    closed = t["closed"]
    line = f"Forward record since {t['start']}: {int(summ['Signals'].sum())} signal(s), {len(closed)} closed"
    if len(closed):
        line += f", {100 * t['win_rate']:.0f}% winners, {t['avg_r']:+.2f}R per closed trade after costs"
    st.info(line + f", {len(t['open'])} open.")
    st.dataframe(summ.assign(Evidence=summ["Closed"].map(lambda n: evidence_strength(int(n)))), hide_index=True,
                 width="stretch")
    with st.expander("Every recorded signal"):
        st.dataframe(t["results"], hide_index=True, width="stretch")
    st.caption("Signals are stored as issued and never recomputed. If the rules change, the new version is reported "
               "on its own row, so one version's results are never mixed with another's.")


def freeze_banner(rep) -> None:
    fr = (rep.tracking or {}).get("freeze") or {}
    if not fr or fr.get("disabled"):
        return
    until = pd.Timestamp(fr["until"]).strftime("%d %b %Y")
    if fr.get("active") and fr.get("drifted"):
        st.warning(f"🔒 Rules are frozen until {until}, but the settings changed (version {fr['version']} → "
                   f"{fr['current_version']}). The forward record reports the new version separately; to restart the "
                   f"freeze deliberately, delete {fr['path']}.")
    elif fr.get("active"):
        st.caption(f"🔒 Rules frozen until {until} (day {fr['day']} of {fr['days_total']}) · rule version "
                   f"{fr['version']}. Keep settings unchanged so the forward record measures one set of rules.")
    else:
        st.info(f"🔓 The rule freeze ended on {until}. Review the forward record and the Strategy lab before changing "
                "any rule; a change starts a new rule version.")


STATE_ICON = {"BUY NOW": "🟢", "WATCH CLOSELY": "🟡", "WAIT": "⚪", "IGNORE": "🔴", "EXPIRED": "⌛"}


def cash_score(rep) -> tuple[int, list[str]]:
    """How strongly today favours staying in cash, with the reasons (0-100)."""
    pts, why = 0, []
    d = rep.regime_detail or {}
    state = d.get("state", "")
    if state == "Risk-off":
        pts, why = pts + 40, why + ["market risk-off"]
    elif state in ("Risk-off, improving", "Transition"):
        pts, why = pts + 20, why + [f"market {state.lower()}"]
    recs = rep.recommendations
    if recs.empty or not ((recs["Action"] == "BUY") & (recs["Qty"] > 0)).any():
        pts, why = pts + 30, why + ["no valid BUY signal"]
    lab = rep.lab_summary
    if lab is not None and not lab.empty and "quality" in set(lab["Variant"]):
        v = lab.set_index("Variant").at["quality", "Verdict"]
        if v != "Passes":
            pts, why = pts + 15, why + [f"BUY rules only '{v}' historically"]
    if rep.data_warning:
        pts, why = pts + 15, why + ["price data stale"]
    return min(pts, 100), why


def daily_briefing(rep) -> None:
    """The decision first, evidence second."""
    q = cfg.buy_quality
    with st.container(border=True):
        st.markdown(f"#### 📋 Daily briefing · based on the close of {rep.session:%a %d %b %Y}")
        trading_decision(rep)
        capital_allocation(rep)
        c1, c2 = st.columns(2)
        with c1:
            d = rep.regime_detail
            if d:
                st.markdown(f"**Market:** {d['icon']} {d['state']} ({d['confidence']}% confidence) · "
                            f"exposure guidance {d['exposure']}")
            score, why = cash_score(rep)
            st.markdown(f"**Cash score: {score}/100**" + (" · staying in cash is the right call today" if score >= 70
                        else "") + (f" ({'; '.join(why)})" if why else ""))
            qd = rep.quality
            live = qd[qd["State"].isin(["BUY NOW", "WATCH CLOSELY", "WAIT"])].sort_values("Readiness", ascending=False) \
                if qd is not None and not qd.empty else pd.DataFrame()
            if not live.empty:
                r = live.iloc[0]
                stale = bool(rep.data_warning)
                blk = [decisions.short_blocker(b) for b in decisions.blockers(r, q, stale)]
                nxt = decisions.becomes_buy(r, q, stale)
                st.markdown(f"**Best candidate:** {r['Ticker']} · {STATE_ICON.get(r['State'], '')} {r['State']} · "
                            f"readiness {int(r['Readiness'])}/100")
                st.markdown(f"**{'Only blocker' if len(blk) == 1 else 'Blockers'}:** {', '.join(blk) or 'none'}")
                if nxt:
                    st.markdown(f"**Next trigger:** {nxt[0]}")
        with c2:
            ch = rep.changes
            if ch is not None and not ch.empty:
                st.markdown(f"**Changes since {pd.Timestamp(ch.attrs.get('previous')).strftime('%d %b')}:**")
                st.markdown("\n".join(f"- {r['Change']} {r['Ticker']}: {r['Was']} → {r['Now']}"
                                       for _, r in ch.head(4).iterrows()))
            dont = []
            if rep.regime and not rep.regime["risk_on"] and q.regime_required:
                dont.append("override the market-regime check")
            dont += ["chase today's strongest movers", "treat the model rank as a probability"]
            if rep.data_warning:
                dont.insert(0, "act on stale prices")
            st.markdown("**Don't:** " + "; ".join(dont) + ".")
            h = (rep.research or {}).get("health") or {}
            if h.get("level") == "bad":
                st.error(h["text"])


def journal_tab(rep) -> None:
    ch = rep.changes
    st.markdown("**What changed since the previous session**")
    if ch is None or ch.empty:
        st.info("No state changes yet. This fills in from the second session the dashboard records.")
    else:
        st.dataframe(ch, hide_index=True, width="stretch")
    st.markdown("**Decision journal** (every candidate's decision, written once and never rewritten)")
    j = rep.journal
    if j is None or j.empty:
        st.info("The journal starts with today's session.")
        return
    j = j.sort_values(["session", "Readiness"], ascending=[False, False])
    st.dataframe(j, hide_index=True, width="stretch")
    st.caption(f"Stored in {Path(cfg.output_dir) / 'decision_journal.csv'} (backed up daily). Each row keeps the rule "
               "version, the price and exactly what would have turned it into a BUY or an AVOID at the time.")


def research_section(rep) -> None:
    """Robustness, multiple-testing and live-health checks."""
    rs = rep.research or {}
    st.markdown("### Robustness")
    sens = rs.get("sensitivity")
    if isinstance(sens, pd.DataFrame) and not sens.empty:
        lvl, txt = rs.get("sensitivity_verdict", ("none", ""))
        {"ok": st.success, "caution": st.warning, "bad": st.error}.get(lvl, st.info)("Parameter sensitivity: " + txt)
        st.dataframe(sens, hide_index=True, width="stretch")
    stress = rs.get("stress")
    if isinstance(stress, pd.DataFrame) and not stress.empty:
        lvl, txt = rs.get("stress_verdict", ("none", ""))
        {"ok": st.success, "caution": st.warning, "bad": st.error}.get(lvl, st.info)("Execution stress: " + txt)
        st.dataframe(stress, hide_index=True, width="stretch")
    st.markdown("### Multiple-testing check")
    dsr = rs.get("dsr")
    n = rs.get("n_trials", 0)
    if dsr and dsr.get("dsr") == dsr.get("dsr"):
        st.markdown(f"Distinct experiments tried so far: **{n}**. Deflated Sharpe ratio of the model portfolio's chosen "
                    f"setup: **{dsr['dsr']:.0%}** (probability its true Sharpe ratio is above zero after allowing for "
                    f"{n} experiments and non-normal returns; raw per-period Sharpe {dsr['sr']:.2f} vs {dsr['sr0']:.2f} "
                    "expected from luck alone).")
        st.caption("Above 95% is strong evidence; below 50% means the result is compatible with luck from trying many "
                   "variants. Every new experiment raises the bar.")
    else:
        st.markdown(f"Distinct experiments tried so far: **{n}**. The deflated Sharpe ratio appears once the weekly tuning "
                    "has run.")
    st.markdown("### Strategy health (forward record vs backtest)")
    h = rs.get("health") or {}
    if h:
        {"ok": st.success, "caution": st.warning, "bad": st.error}.get(h.get("level"), st.info)(h["text"])


def readiness_panel(rep) -> None:
    t = rep.tracking or {}
    rs = rep.research or {}
    from status import readiness as _readiness, ready as _ready
    checks = _readiness(t.get("results"), rs.get("health"), rs, (t.get("freeze") or {}).get("current_version", ""),
                        cfg.real_money_min_trades, cfg.real_money_min_avg_r)
    ok_all = _ready(checks)
    icon = lambda v: "✅" if v is True else ("❌" if v is False else "⏳")  # noqa: E731
    with st.expander(f"{'✅' if ok_all else '🔒'} NSE strategy ready for real money? "
                     f"{'YES' if ok_all else 'NO: paper only'} ({sum(v is True for v, _ in checks)}/{len(checks)})",
                     expanded=False):
        st.markdown("\n".join(f"- {icon(v)} {t}" for v, t in checks))
        st.caption("These thresholds were fixed in advance (real_money_min_trades, real_money_min_avg_r) so the goalposts "
                   "cannot move once results arrive. ⏳ = not enough data yet. Even when every item is ✅, consult a "
                   "SEBI-registered adviser and account for tax before risking money.")


def regime_panel(rep) -> None:
    """Five-state market regime with confidence and early-recovery detection (descriptive)."""
    d = rep.regime_detail
    if not d:
        return
    path = " → ".join(f"{x}%" for x in d.get("breadth_path", []))
    with st.container(border=True):
        st.markdown(f"**Market regime: {d['icon']} {d['state'].upper()}** · confidence {d['confidence']}% · "
                    f"exposure guidance: {d['exposure']}" + (f" · breadth over 15 sessions: {path}" if path else ""))
        with st.expander("What the regime is based on"):
            st.markdown("\n".join(f"- {'✅' if ok else '❌'} {name}" for name, ok in d["components"].items()))
            st.caption("Descriptive: it tells you the market's condition and whether it is improving. The BUY rules "
                       "use the market-regime check set in buy_quality (currently "
                       f"{'on, ' + cfg.buy_quality.regime_mode if cfg.buy_quality.regime_required else 'off'}); change "
                       "that only if the Strategy lab supports it.")


def _similar_setups(rep, regime_state: str) -> dict | None:
    """Historical outcome of setups that passed every stock check, in the same market regime."""
    t = rep.lab_trades
    if t is None or t.empty or "regime_state" not in t:
        return None
    t = t[(t["variant"] == "quality_no_regime") & (t["regime_state"] == regime_state)]
    if len(t) < 10:
        return None
    r = t["r_multiple"]
    return {"n": len(t), "win": 100 * (r > 0).mean(), "avg_r": r.mean(),
            "target": 100 * (t["outcome"] == "target").mean(), "hold": t["hold_days"].median()}


def best_setup(rep) -> None:
    """The decision card: hard checks, conviction factors, history, and exactly what would change the decision."""
    qd = rep.quality
    if qd is None or qd.empty:
        return
    cands = qd[qd["State"].isin(["BUY NOW", "WATCH CLOSELY", "WAIT"])].sort_values("Readiness", ascending=False)
    if cands.empty:
        return
    r = cands.iloc[0]
    q = cfg.buy_quality
    state = r["State"]
    if state == "BUY NOW" and rep.data_warning:
        state = "WATCH CLOSELY"  # never BUY NOW on stale data
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    hard = decisions.hard_checks(r, q, bool(rep.data_warning))
    scores = rep.scores if rep.scores is not None else pd.DataFrame()
    soft = []
    if not scores.empty and r["Ticker"] in scores.index:
        srow = scores.loc[r["Ticker"]]
        soft.append(f"Model rank {100 * srow['score']:.0f}/100")
        sect = rep.sectors
        if sect is not None and not sect.empty and srow["sector"] in set(sect["Sector"]):
            srank = sect.set_index("Sector").loc[srow["sector"]]
            soft.append(f"Sector {srow['sector']}: rank {int(srank['Rank'])} of {len(sect)} "
                        f"({srank['3M median %']:+.1f}% median over 3 months)")
    if np.isfinite(r["RS vs Nifty 3M %"]):
        soft.append(f"Relative strength {r['RS vs Nifty 3M %']:+.1f}% vs the Nifty over 3 months")
    if rep.regime_detail:
        soft.append(f"Market: {rep.regime_detail['icon']} {rep.regime_detail['state']} "
                    f"({rep.regime_detail['confidence']}% confidence)")
    dw = r.get("Days on watch", 0)
    if dw:
        tr = r.get("Readiness trend", np.nan)
        soft.append(f"On watch {int(dw)} session(s)" + (f", readiness {tr:+.0f} over the last 3" if np.isfinite(tr) else "")
                    + f"; expires after {q.watch_expiry_sessions} without a breakout")
    becomes_buy = decisions.becomes_buy(r, q, bool(rep.data_warning))
    becomes_avoid = decisions.becomes_avoid(r)
    sim = _similar_setups(rep, str(r["Regime state"]))
    icon = STATE_ICON.get(state, "")
    with st.container(border=True):
        st.markdown(f"### {icon} {r['Ticker']}: {state}")
        st.caption(f"Best setup right now · readiness {int(r['Readiness'])}/100 (completeness, not a probability)")
        left, right = st.columns(2)
        with left:
            st.markdown("**Hard checks (all must pass)**")
            st.markdown("\n".join(f"- {ok(b)} {t}" for b, t in hard))
        with right:
            st.markdown("**Conviction factors**")
            st.markdown("\n".join(f"- {x}" for x in soft) or "-")
            if sim:
                st.markdown(f"**Similar setups in history** (every stock check passed, market "
                            f"{r['Regime state']}): won {evidence_label(sim['win'], sim['n'])} · "
                            f"{sim['avg_r']:+.2f}R per trade · hit target {sim['target']:.0f}% · median hold "
                            f"{sim['hold']:.0f} sessions")
                st.caption("Historical conditional win rate, not a forecast: conditions change.")
        failing = [t for b, t in hard if not b]
        if failing:
            st.markdown("**Why not BUY:** " + "; ".join(failing) + ".")
        st.markdown("**Becomes BUY if:** " + ("; ".join(becomes_buy) if becomes_buy else "it already qualifies") + ".")
        st.markdown("**Becomes AVOID if:** " + "; ".join(becomes_avoid) + ".")
        if r["g_breakout"] and np.isfinite(r["Stop"]):
            from risk_engine.risk_manager import RiskError, RiskManager
            try:
                plan = RiskManager(cfg.risk).plan_from_levels(r["Ticker"], r["Close"], r["Stop"], r["Target"], r["ATR"])
                lead = "Trade plan" if state == "BUY NOW" else "If it becomes a BUY"
                st.markdown(f"**{lead}:** entry only at the next open between ₹{r['Breakout level']:,.2f} "
                            f"and ₹{r['Close'] * (1 + q.entry_max_gap_pct / 100):,.2f} · stop ₹{plan.stop_loss:,.2f} · "
                            f"target ₹{plan.take_profit:,.2f} · {plan.units} shares · maximum loss "
                            f"₹{plan.risk_amount:,.0f} ({100 * plan.risk_amount / cfg.risk.account_equity:.2f}% of equity)")
            except RiskError:
                pass
        others = ", ".join(f"{t} ({int(v)})" for t, v in zip(cands["Ticker"].iloc[1:4], cands["Readiness"].iloc[1:4]))
        if others:
            st.caption(f"Next most complete setups: {others}.")


def _nse_status(rep) -> tuple[bool, list[str]]:
    """(market permits new BUYs, tickers with a sized BUY signal)."""
    recs = rep.recommendations
    buys = recs[(recs["Action"] == "BUY") & (recs["Qty"] > 0)] if not recs.empty else recs
    q = cfg.buy_quality
    permitted = not (rep.regime and not rep.regime["risk_on"] and q.regime_required)
    return permitted, list(buys["Ticker"]) if not buys.empty else []


def trading_decision(rep) -> None:
    """Market permission and trade opportunity are separate questions; the decision needs both."""
    permitted, buys = _nse_status(rep)
    env = ("🟢 **Market environment: favourable.** The Nifty's trend allows new positions." if permitted else
           "🔴 **Market environment: unfavourable.** The Nifty is below its 200-day average, so your market-regime "
           "rule blocks every new BUY.")
    opp = (f"🟢 **Trade opportunity: {len(buys)} stock(s) pass every check:** {', '.join(buys)}." if buys else
           "⚪ **Trade opportunity: none today.** No stock passes every BUY check"
           + (" (stocks that pass everything except the market regime are on the watch list)." if not permitted else "."))
    if buys:
        st.success(f"**Today's trading decision: BUY at the next open, within each stock's entry range.**  \n{env}  \n{opp}")
    elif not permitted:
        st.error(f"**Today's trading decision: no new BUY trades; WAIT.**  \n{env}  \n{opp}")
    else:
        st.warning(f"**Today's trading decision: no new BUY trades; WAIT.**  \n{env}  \n{opp}")


def capital_allocation(rep) -> None:
    """Where should new capital go: NSE, crypto, or cash?"""
    if not cfg.crypto.enabled:
        return
    from views.engines import get_crypto
    crep = get_crypto().peek()
    permitted, buys = _nse_status(rep)
    rows = [{"Market": "NSE", "Environment": "🟢 favourable" if permitted else "🔴 unfavourable",
             "Valid setups": len(buys), "Action": "BUY " + ", ".join(buys) if buys else "wait"}]
    c_buys = []
    if crep is not None:
        sig = crep.signals
        c_buys = list(sig.loc[(sig["Action"] == "BUY") & (sig["Qty"] > 0), "Ticker"]) if not sig.empty else []
        rows.append({"Market": "Crypto", "Environment": ("🟢 favourable" if crep.regime["risk_on"] else "🔴 unfavourable")
                     + f" (score {crep.regime.get('score', '?')}/100)", "Valid setups": len(c_buys),
                     "Action": "BUY " + ", ".join(c_buys) if c_buys else "wait"})
    else:
        rows.append({"Market": "Crypto", "Environment": "loading", "Valid setups": 0, "Action": "wait"})
    rows.append({"Market": "Cash", "Environment": "always available", "Valid setups": "",
                 "Action": "hold" if not (buys or c_buys) else "keep the rest"})
    if buys or c_buys:
        verdict = ("New capital goes only to the valid setups above, each within its own position and risk limits; "
                   "everything else stays in cash.")
    else:
        why = []
        why.append("NSE environment " + ("is favourable but has no valid setup" if permitted else "is unfavourable"))
        if crep is not None:
            why.append("crypto environment " + ("is favourable but has no valid setup" if crep.regime["risk_on"]
                                                else "is unfavourable"))
        verdict = "Hold new capital in cash: " + "; ".join(why) + ". Don't buy the least-bad candidate."
    with st.container(border=True):
        st.markdown("**Capital allocation for new money**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(verdict)


def action_plan(rep) -> None:
    """The model portfolio's own paper rebalance (a separate experiment, not a trade signal)."""
    pp = rep.paper
    if not pp or rep.settings is None:
        return
    st.markdown("This is a **separate, paper-only experiment**: a portfolio of the top-ranked stocks, rebalanced "
                "automatically, used to measure whether the ranking model adds value. **It is not a trade signal.** "
                "Trade decisions come from the Trade signals tabs and the decision line above.")
    st.caption(f"Experiment settings: {rep.settings.label()}.")
    if not rep.settings.regime and rep.regime and not rep.regime["risk_on"]:
        st.caption("Note: this experiment stays invested in all markets (a regime filter did not improve it in "
                   "testing), while the trade-signal rules sit out until the Nifty recovers.")
    pend = pp.get("pending")
    held = set(pp["holdings"]["Ticker"]) if not pp["holdings"].empty else set()
    if pend and pend["date"] == str(rep.session):
        targets = set(pend["targets"])
        buy, sell, keep = sorted(targets - held), sorted(held - targets), sorted(targets & held)
        if not targets:
            st.info("**Paper rebalance at the next open:** move the experiment to cash"
                    + (f" (remove {', '.join(sell)})" if sell else "") + ".")
        else:
            st.info("**Paper rebalance at the next open (experiment only).** "
                    + (f"Adds: {', '.join(buy)}. " if buy else "")
                    + (f"Removes: {', '.join(sell)}. " if sell else "")
                    + (f"Keeps: {', '.join(keep)}." if keep else ""))
    elif held:
        st.info(f"**No paper rebalance today.** The experiment holds {len(held)} stocks: {', '.join(sorted(held))}.")


def paper_tab(rep) -> None:
    pp = rep.paper
    if not pp or pp.get("equity") is None:
        st.info("The paper portfolio starts after the first completed session; it then runs by itself.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Paper equity", f"₹{pp['equity']:,.0f}", f"{pp['return_pct']:+.2f}% since {pp['start']}", delta_color="off", delta_arrow="off")
    nr = pp.get("nifty_return_pct")
    b.metric("Nifty 50, same period", f"{nr:+.2f}%" if nr is not None else "n/a")
    c.metric("Ahead of Nifty", f"{pp['return_pct'] - nr:+.2f} pts" if nr is not None else "n/a")
    d.metric("Cash", f"₹{pp['cash']:,.0f}")
    hist = pp["history"]
    if len(hist) > 1:
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=100 * hist["equity"] / hist["equity"].iloc[0], name="Paper portfolio",
                        line=dict(color=C["long"], width=2))
        if hist["nifty"].notna().sum() > 1:
            n = hist["nifty"]
            fig.add_scatter(x=hist["date"], y=100 * n / n.dropna().iloc[0], name="Nifty 50", line=dict(color=C["watch"]))
        fig.update_layout(height=320, yaxis_title="Growth of ₹100", plot_bgcolor=C["panel"], paper_bgcolor=C["panel"],
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Holdings**")
    if pp["holdings"].empty:
        st.write("In cash.")
    else:
        st.dataframe(pp["holdings"], hide_index=True, width="stretch", column_config={
            "Entry": st.column_config.NumberColumn(format="₹%.2f"),
            "Last close": st.column_config.NumberColumn(format="₹%.2f"),
            "Value": st.column_config.NumberColumn(format="₹%.0f"),
            "P&L %": st.column_config.NumberColumn(format="%+.2f%%")})
    with st.expander(f"Trade log ({len(pp['trades'])} trades)"):
        st.dataframe(pp["trades"], hide_index=True, width="stretch")
    st.caption("Runs automatically: orders are filled at the next session's open with estimated costs, valued at "
               "each close. This is simulated, not real trading. Delete outputs/paper_portfolio.json to restart it. "
               "Judge it over months, not days.")


def tuning_tab(rep) -> None:
    opt = rep.optimizer
    if not opt:
        st.info("Tuning is switched off (ranker.optimize = false).")
        return
    (st.success if opt["adopted"] else st.info)(opt["reason"])
    if opt.get("index_note"):
        (st.success if opt.get("beats_index") else st.warning)(opt["index_note"])
    st.markdown(f"Settings are chosen on the out-of-sample years before **{opt.get('holdout_start')}** and must then "
                "beat the current setup on the final, unseen year. Re-tuned weekly.")
    curves = opt.get("curves") or {}
    if curves:
        fig = go.Figure()
        for (name, sim), color in zip(curves.items(), (C["long"], C["watch"])):
            if not sim.empty:
                fig.add_scatter(x=sim.index, y=100 * (1 + sim["ret"]).cumprod(), name=name, line=dict(color=color, width=2))
        if opt.get("holdout_start"):
            fig.add_vline(x=pd.Timestamp(opt["holdout_start"]), line_dash="dot", line_color=C["muted"])
            fig.add_annotation(x=pd.Timestamp(opt["holdout_start"]), y=1, yref="paper", text="unseen final year →",
                               showarrow=False, xanchor="left", font=dict(color=C["muted"]))
        fig.update_layout(height=320, yaxis_title="Growth of ₹100 (after costs)", plot_bgcolor=C["panel"],
                          paper_bgcolor=C["panel"], margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    table = opt["table"]
    if not table.empty:
        cols = ["Setup", "validated", "sel_sharpe", "sel_cagr", "sel_excess_cagr", "sel_avg_turnover",
                "hold_sharpe", "hold_cagr", "hold_nifty_cagr", "hold_max_dd"]
        st.dataframe(table[cols].rename(columns={
            "sel_sharpe": "Sharpe (selection)", "sel_cagr": "CAGR % (selection)", "sel_excess_cagr": "vs Nifty % (sel.)",
            "sel_avg_turnover": "Turnover % per rebalance", "hold_sharpe": "Sharpe (final year)",
            "hold_cagr": "CAGR % (final year)", "hold_nifty_cagr": "Nifty % (final year)",
            "hold_max_dd": "Max DD % (final year)"}), hide_index=True, width="stretch")
        st.caption("Sorted by selection-period Sharpe. Many settings were tried, so the top row is flattered by luck; "
                   "that is why only the unseen final year decides.")


engine, _, cfg = get_engines()
st.title("Market intelligence")


@st.fragment(run_every=timedelta(seconds=30))
def view() -> None:
    if background_running():  # the background does the work; the page only displays, never waits
        rep = engine.peek()
        if rep is None:
            progress_panel(engine, "the market research and model")
            return
        render(rep)
        return
    try:
        if engine._last is None:
            with st.spinner("Researching the market: downloading history, validating the model walk-forward and "
                            "scoring every stock. The first run of the day takes a few minutes; later opens are instant."):
                rep = engine.refresh(now_ist())
        else:
            rep = engine.refresh(now_ist())
    except DataFetchError as exc:
        st.error(f"{exc} Retrying automatically.")
        return
    render(rep)


def render(rep) -> None:
    if rep.data_warning:
        st.warning("⚠️ " + rep.data_warning)
    st.caption(f"Based on the close of {rep.session:%A %d %B %Y} · updated {rep.as_of:%H:%M} IST · "
               f"{rep.universe_label} · Superstar: {rep.superstar_label}"
               + (" · DEMO MODE: synthetic data" if DEMO else "")
               + (f" · Not available on the data source (skipped): {', '.join(rep.skipped)}" if rep.skipped else ""))

    # ---------------------------------------------------------------- market
    m1, m2, m3, m4 = st.columns(4)
    if rep.regime:
        icon, label = STATE[rep.regime["state"]]
        m1.metric("Nifty trend", f"{icon} {label}", f"{pct(rep.regime['ret_3m'])} in 3 months", delta_color="normal")
        m3.metric("Nifty from 1-year high", pct(rep.regime["drawdown"]), f"1-year {pct(rep.regime['ret_1y'])}",
                  delta_color="off", delta_arrow="off")
        vp = rep.regime["vol_percentile"]
        m4.metric("Volatility", f"{100 * rep.regime['vol_21d']:.0f}% ann.",
                  f"higher than {100 * vp:.0f}% of the last 3 years" if np.isfinite(vp) else "", delta_color="off", delta_arrow="off")
    else:
        m1.metric("Nifty trend", "unavailable")
    b = rep.breadth
    d20 = 100 * (b["pct_above_200"] - b.get("pct_above_200_20d", np.nan))
    trend_txt = f"{d20:+.0f} pts in 20 days" if np.isfinite(d20) else ""
    m2.metric("Breadth", f"{100 * b['pct_above_200']:.0f}%", trend_txt or f"{b['advancers']} up / {b['decliners']} down",
              delta_color="normal" if np.isfinite(d20) else "off", delta_arrow="auto" if np.isfinite(d20) else "off",
              help=(f"Share of stocks above their 200-day average: {100 * b['pct_above_200']:.0f}% today, "
                    f"{100 * b.get('pct_above_200_5d', np.nan):.0f}% 5 sessions ago, "
                    f"{100 * b.get('pct_above_200_20d', np.nan):.0f}% 20 sessions ago. Above the 50-day: "
                    f"{100 * b['pct_above_50']:.0f}% (was {100 * b.get('pct_above_50_20d', np.nan):.0f}% 20 sessions ago). "
                    f"Today {b['advancers']} up / {b['decliners']} down; {b['new_highs']} new highs, {b['new_lows']} new lows."))
    if rep.regime:
        st.caption(rep.regime["label"] + ("" if rep.regime["risk_on"] else
                   ". Risk-off: BUY conviction is reduced while the Nifty is below its 200-day average."))

    # ----------------------------------------------------------------- trust
    v = rep.validation
    if rep.error:
        st.error(f"The prediction model could not run: {rep.error}")
    elif rep.validated:
        st.success(f"Model validated out of sample ({v['oos_start']} to {v['oos_end']}): mean rank IC "
                   f"{v['mean_ic']:.3f}, t-stat {v['ic_tstat']:.1f}, top-minus-bottom decile "
                   f"{100 * v['spread_mean']:+.2f}% per 10 sessions. Evidence of an edge is not a guarantee.")
    else:
        failed = [k for k, ok in v["checks"].items() if not ok]
        st.warning("Model NOT validated on this data, so every idea below is marked Unvalidated. Failed: "
                   + "; ".join(failed) + f". (Mean IC {v['mean_ic']:.3f}, t-stat {v['ic_tstat']:.1f}.) "
                   "Treat the lists as a research starting point, not trade signals.")

    recs = rep.recommendations
    check, check_note = morning_check(engine, rep, now_ist(), fetch=not background_running())
    if check_note and check is None:
        st.info(check_note)
    daily_briefing(rep)
    readiness_panel(rep)
    from views.engines import simple_view
    if simple_view(cfg.simple_view_default):
        best_setup(rep)
        st.caption("Simple view: the decision, the best setup and alerts. Switch off Simple view in the sidebar for the "
                   "market-regime detail, every signal table and all research tabs.")
        return
    regime_panel(rep)
    freeze_banner(rep)
    best_setup(rep)
    tabs = st.tabs(["Trade signals: BUY", "Trade signals: SELL / avoid", "Model portfolio (paper experiment)",
                    "Stock research", "Model evidence", "Tuning", "Sectors", "Strategy lab", "Journal & changes"])
    with tabs[0]:
        b_ = recs[recs["Action"] == "BUY"]
        q = cfg.buy_quality
        st.caption("Rank (percentile) orders stocks against each other today; it is not a probability. The model uses "
                   "price and volume only (momentum, trend, volatility, volume, strength vs the Nifty and sector); it does "
                   "not look at fundamentals such as earnings, debt or valuation.")
        st.caption(f"A BUY must pass every check: close above the prior {q.breakout_lookback}-day high with a strong "
                   f"close and not extended; volume ≥ {q.volume_multiple:g}× the {q.volume_window}-day average; trend "
                   "(price > 50-day > rising 200-day, beating the Nifty); ATR stop below the breakout; Nifty above its "
                   f"200-day average; reward/risk ≥ {q.min_reward_risk:g}:1 with the target capped at the 52-week high; "
                   "and, when the model is validated, a model score of at least "
                   f"{100 * q.min_model_score:.0f}. Entry only at the next open, within the stated range.")
        quality_health(rep)
        if rep.regime and not rep.regime["risk_on"] and q.regime_required:
            st.warning("Market-regime check: the Nifty is below its 200-day average, so no new BUYs are allowed today. "
                       "Stocks that pass everything else are on the watch list below.")
        morning_table(check, "BUY", check_note)
        if not b_.empty:
            sized = b_[b_["Qty"] > 0]
            st.caption(f"Sized together within your limits: {len(sized)} position(s), total risk "
                       f"₹{b_['Risk (INR)'].where(b_['Qty'] > 0, 0).sum():,.0f} of the "
                       f"₹{cfg.risk.account_equity * cfg.risk.max_portfolio_heat_pct:,.0f} allowed "
                       f"({100 * cfg.risk.max_portfolio_heat_pct:.0f}% of equity, max "
                       f"{cfg.risk.max_open_positions} positions). The rest stay on the watch list.")
            rec_table(b_)
        else:
            st.info("No stock passed every BUY check this session. That is normal: the filters are strict on purpose.")
        track_record(rep)
        trending_section(rep)
        w_ = recs[recs["Action"] == "WATCH"]
        if not w_.empty:
            st.markdown("**Watch list: top-ranked by the model, but failed a check**")
            if rep.quality is not None and not rep.quality.empty:
                extra = [c for c in ["Ticker", "Readiness", "State", "Breakout status", "Days on watch"]
                         if c in rep.quality.columns]
                w_ = w_.merge(rep.quality[extra], on="Ticker", how="left")
                w_["State"] = w_["State"].map(lambda x: f"{STATE_ICON.get(x, '')} {x}" if isinstance(x, str) else x)
            cols = [c for c in ["Ticker", "State", "Readiness", "Breakout status", "Days on watch", "Model score",
                                "Hist. beat %", "Price",
                                "R:R", "Checks", "Why", "Sector"] if c in w_.columns]
            st.dataframe(w_[cols].sort_values("Readiness", ascending=False) if "Readiness" in w_ else w_[cols],
                         hide_index=True,
                         width="stretch", column_config={
                             "Model score": st.column_config.ProgressColumn("Rank (percentile)", min_value=0,
                                                                            max_value=100, format="%.0f / 100", help="Percentile rank among all stocks today, not a probability or confidence. 100 = ranked above about 99% of stocks. See Hist. beat peers for how often stocks with this rank actually outperformed."),
                             "Hist. beat %": st.column_config.NumberColumn("Hist. beat peers", format="%.0f%%",
                                 help="How often stocks in this rank decile beat the median stock over the holding "
                                      "period, in out-of-sample testing. This is the closest thing to a probability."),
                             "Price": st.column_config.NumberColumn(format="₹%.2f"),
                             "R:R": st.column_config.NumberColumn("Reward/risk", format="%.1f : 1",
                                 help="Shown only after a breakout; without one there is no entry to measure from."),
                             "Checks": st.column_config.TextColumn("Failed checks", width="medium"),
                             "Readiness": st.column_config.ProgressColumn("Readiness", min_value=0, max_value=100,
                                 format="%.0f", help="How complete the BUY setup is (market regime excluded). Not a "
                                                     "probability of profit.")})
    with tabs[1]:
        s_ = recs[recs["Action"] == "SELL"]
        st.caption("Bottom-ranked stocks: exit or avoid if held. Cash-segment shorts are intraday only; "
                   "overnight shorts need F&O.")
        morning_table(check, "SELL", check_note)
        if not s_.empty:
            rec_table(s_)
        else:
            st.info("No SELL ideas this session.")

    with tabs[2]:
        action_plan(rep)
        paper_tab(rep)

    with tabs[5]:
        tuning_tab(rep)

    with tabs[3]:
        scores = rep.scores
        if scores.empty:
            st.info("No scores available.")
        else:
            default = recs["Ticker"].iloc[0] if not recs.empty else scores.index[0]
            options = list(scores.index)
            tk = st.selectbox("Stock", options, index=options.index(default) if default in options else 0)
            r = scores.loc[tk]
            prof = stock_profile(rep.histories[tk], rep.benchmark)
            a, b2, c2, d = st.columns(4)
            a.metric("Rank (percentile)", f"{100 * r['score']:.0f} / 100", f"#{options.index(tk) + 1} of {len(options)} today",
                     delta_color="off", delta_arrow="off")
            b2.metric("3 months", pct(prof["ret_3m"]), f"vs Nifty {pct(prof.get('vs_index_3m', np.nan))}",
                      delta_color="off", delta_arrow="off")
            c2.metric("From 52-week high", pct(prof["from_52w_high"]), f"1-year {pct(prof['ret_1y'])}", delta_color="off", delta_arrow="off")
            d.metric("Volatility", f"{100 * prof['vol_ann']:.0f}% ann.", f"max drawdown 1y {pct(prof['max_dd_1y'])}",
                     delta_color="off", delta_arrow="off")
            contrib = rep.contributions.loc[tk] if rep.contributions is not None else None
            left, right = st.columns([3, 2])
            with left:
                st.plotly_chart(range_chart(rep.histories[tk], cfg.breakout), width="stretch")
            with right:
                st.markdown("**Pushing the score up**")
                st.markdown("\n".join(f"- {x}" for x in explain(r, contrib, +1)))
                st.markdown("**Pushing the score down**")
                st.markdown("\n".join(f"- {x}" for x in explain(r, contrib, -1)))
                if contrib is not None:
                    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
                    names = [LABELS.get(f[2:] if f.startswith("r_") else f, f) for f in top.index]
                    fig = go.Figure(go.Bar(x=top.values, y=names, orientation="h",
                                           marker_color=[C["long"] if x > 0 else C["stop"] for x in top.values]))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor=C["panel"],
                                      paper_bgcolor=C["panel"], title="Feature contributions (SHAP)",
                                      font=dict(family="IBM Plex Sans, sans-serif", size=12))
                    st.plotly_chart(fig, width="stretch")
                st.caption(f"Sector: {r['sector']}. Above 50-day: {'yes' if prof['above_50dma'] else 'no'}; "
                           f"above 200-day: {'yes' if prof['above_200dma'] else 'no'}; RSI {prof['rsi']:.0f}.")

    with tabs[4]:
        if not v:
            st.info("No validation results.")
        else:
            st.markdown("Walk-forward test: the model was trained only on past data and scored on the following, "
                        "unseen period, repeatedly. Every 10 sessions the simulated portfolio holds the top "
                        f"{cfg.ranker.top_n} stocks, equal weight, after {100 * cfg.ranker.cost_pct:.2f}% costs.")
            port = v["portfolio"]
            fig = go.Figure()
            for col, name, color in (("strategy", "Top-ranked portfolio", C["long"]),
                                     ("strategy_regime", "Same, in cash when Nifty < 200-day", C["ink"]),
                                     ("nifty", "Nifty 50", C["watch"])):
                if col in port and port[col].notna().any():
                    fig.add_scatter(x=port.index, y=100 * (1 + port[col].fillna(0)).cumprod(), name=name,
                                    line=dict(color=color, width=2))
            fig.update_layout(height=360, yaxis_title="Growth of ₹100", plot_bgcolor=C["panel"], paper_bgcolor=C["panel"],
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
                              font=dict(family="IBM Plex Sans, sans-serif"))
            st.plotly_chart(fig, width="stretch")
            perf = pd.DataFrame(v["perf"]).T.rename(index={"strategy": "Top-ranked portfolio",
                "strategy_regime": "With regime filter", "nifty": "Nifty 50", "universe": "Equal-weight universe (no costs)"})
            st.dataframe((perf * 100).round(1).rename(columns={"cagr": "CAGR %", "vol": "Volatility %",
                         "sharpe": "Sharpe ×100", "max_dd": "Max drawdown %", "total": "Total %"}), width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**By year (out of sample)**")
                st.dataframe(v["by_year"], hide_index=True, width="stretch")
            with c2:
                cal = v["calibration"]
                fig = go.Figure(go.Bar(x=cal["decile"], y=cal["avg_excess_pct"],
                                       marker_color=[C["long"] if y > 0 else C["stop"] for y in cal["avg_excess_pct"]]))
                fig.update_layout(height=280, title="Avg 10-session return vs peers, by score decile",
                                  xaxis_title="Score decile (10 = best)", yaxis_title="%", plot_bgcolor=C["panel"],
                                  paper_bgcolor=C["panel"], margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("A useful model shows bars rising from left to right.")
            st.markdown("**Validation checks**")
            st.markdown("\n".join(f"- {'✅' if ok else '❌'} {k}" for k, ok in v["checks"].items()))
            st.caption("Survivorship bias: history uses today's stock list, so delisted stocks are missing and "
                       "results are somewhat optimistic.")

    with tabs[6]:
        sect = rep.sectors
        fig = go.Figure(go.Bar(x=sect["3M median %"], y=sect["Sector"], orientation="h",
                               marker_color=[C["long"] if x > 0 else C["stop"] for x in sect["3M median %"]]))
        fig.update_layout(height=max(260, 26 * len(sect)), yaxis=dict(autorange="reversed"), plot_bgcolor=C["panel"],
                          paper_bgcolor=C["panel"], margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Median 3-month return %")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sect, hide_index=True, width="stretch")

    with tabs[7]:
        evidence_summary(rep)
        st.markdown("The Chartink breakout rules against improved variants, backtested on this universe. "
                    "**Passes** requires 30+ trades, average above +0.1R, profit factor above 1.3 and positive "
                    "results in at least 60% of years.")
        if rep.lab_summary.empty:
            st.info("No breakout trades on this history.")
        else:
            st.dataframe(rep.lab_summary.assign(Evidence=rep.lab_summary["Trades"].map(
                lambda n: evidence_strength(int(n)))).style.map(
                lambda x: {"Passes": f"color:{C['long']};font-weight:600", "Fails": f"color:{C['stop']};font-weight:600"}
                .get(x, ""), subset=["Verdict"]), hide_index=True, width="stretch")
            st.markdown("**Average R by year**")
            st.dataframe(rep.lab_by_year, width="stretch")
        fun = rep.quality_funnel
        if fun is not None and not fun.empty:
            st.markdown("**Quality BUY funnel: how many historical breakouts survived each check**")
            fig = go.Figure(go.Funnel(y=fun["Stage"], x=fun["Signals remaining"], marker_color=C["long"],
                                      textinfo="value+percent initial"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=C["panel"],
                              paper_bgcolor=C["panel"], font=dict(family="IBM Plex Sans, sans-serif"))
            st.plotly_chart(fig, width="stretch")
        filter_studies(rep)
        research_section(rep)
        stress_section(rep)

    with tabs[8]:
        journal_tab(rep)


view()
st.markdown(f'<p style="color:{C["muted"]};font-size:0.8rem;border-top:1px solid {C["grid"]};padding-top:.75rem;'
            'margin-top:2rem;max-width:85ch">Model-generated research, not investment advice. Rankings estimate '
            'relative performance from historical patterns and can be wrong; validation on past data does not '
            'guarantee future results, and no system can ensure profitable trades. No broker orders are placed. '
            'Consult a SEBI-registered adviser before investing.</p>', unsafe_allow_html=True)
