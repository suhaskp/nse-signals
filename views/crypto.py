"""Crypto (Binance spot, paper only): the NSE rules applied to crypto. Updates itself; nothing to run."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from research.evidence import label as evidence_label, strength as evidence_strength
from research.movers import verdict as movers_verdict
from research.stress import stress_test, verdict as stress_verdict
from views.charts import range_chart
from views.engines import DEMO, background_running, get_crypto, get_engines, now_ist, progress_panel

_, _, cfg = get_engines()
eng = get_crypto()
c = cfg.crypto
st.title("Crypto (Binance)")
fmt = lambda x: f"{x:,.6g}" if x == x else "n/a"  # noqa: E731


STATE_ICON = {"BUY NOW": "🟢", "WATCH CLOSELY": "🟡", "WAIT": "⚪", "IGNORE": "🔴", "EXPIRED": "⌛"}


def closest_setup(rep) -> None:
    """If nothing qualifies: which coin is closest, and exactly what has to happen."""
    import decisions
    qd = rep.quality
    if qd is None or qd.empty:
        return
    cands = qd[qd["State"].isin(["BUY NOW", "WATCH CLOSELY", "WAIT"]) & (qd["Ticker"] != c.benchmark)]
    if cands.empty:
        return
    r = cands.sort_values("Readiness", ascending=False).iloc[0]
    q = eng.q
    hard = decisions.hard_checks(r, q, False, market="BTC")
    with st.container(border=True):
        st.markdown(f"**Closest setup: {STATE_ICON.get(r['State'], '')} {r['Ticker']}: {r['State']}** · readiness "
                    f"{int(r['Readiness'])}/100 (completeness, not a probability)")
        st.markdown("\n".join(f"- {'✅' if ok else '❌'} {t}" for ok, t in hard if "Data is fresh" not in t))
        st.markdown("**Becomes BUY if:** " + ("; ".join(decisions.becomes_buy(r, q, False, market="BTC")) or
                                              "it already qualifies") + ".")
        st.markdown("**Becomes AVOID if:** " + "; ".join(decisions.becomes_avoid(r, market="BTC"))
                    + f"; or it opens more than {c.entry_max_gap_pct:g}% above the signal close (don't chase).")
        others = ", ".join(f"{t} ({int(v)})" for t, v in zip(cands.sort_values("Readiness", ascending=False)["Ticker"].iloc[1:4],
                                                            cands.sort_values("Readiness", ascending=False)["Readiness"].iloc[1:4]))
        if others:
            st.caption(f"Next closest: {others}.")


def crypto_readiness(rep) -> None:
    from crypto_engine import crypto_rule_version
    f = rep.forward
    from status import readiness as _readiness, ready as _ready
    checks = _readiness(f.get("results"), f.get("health"), None, crypto_rule_version(cfg),
                        cfg.real_money_min_trades, cfg.real_money_min_avg_r)
    ok_all = _ready(checks)
    icon = lambda v: "✅" if v is True else ("❌" if v is False else "⏳")  # noqa: E731
    with st.expander(f"{'✅' if ok_all else '🔒'} Crypto strategy ready for real money? "
                     f"{'YES' if ok_all else 'NO: paper only'} ({sum(v is True for v, _ in checks)}/{len(checks)})",
                     expanded=False):
        st.markdown("\n".join(f"- {icon(v)} {t}" for v, t in checks))
        st.caption("These thresholds were fixed in advance (real_money_min_trades, real_money_min_avg_r) so the goalposts "
                   "cannot move once results arrive. ⏳ = not enough data yet. Even when every item is ✅, consult a "
                   "SEBI-registered adviser and account for tax before risking money.")


def exposure_panel(rep) -> None:
    """BTC vs best altcoin vs cash: which form of exposure, if any, is preferred today."""
    qd, sig, g = rep.quality, rep.signals, rep.regime
    buys = list(sig.loc[(sig["Action"] == "BUY") & (sig["Qty"] > 0), "Ticker"]) if sig is not None and not sig.empty else []
    btc = qd[qd["Ticker"] == c.benchmark].iloc[0] if qd is not None and (qd["Ticker"] == c.benchmark).any() else None
    btc_trend = ("trend intact (above its 50- and 200-day averages)" if btc is not None and g["risk_on"]
                 and btc["Close"] > btc["SMA50"] else "trend not confirmed")
    alts = qd[(qd["Ticker"] != c.benchmark) & qd["State"].isin(["BUY NOW", "WATCH CLOSELY", "WAIT"])] if qd is not None else qd
    best = alts.sort_values("Readiness", ascending=False).iloc[0] if alts is not None and not alts.empty else None
    rows = [{"Exposure": "BTC", "Status": f"WATCH: {btc_trend}",
             "Note": "BTC cannot be a BUY under the current rules (it cannot beat itself); shown for information"},
            {"Exposure": "Best altcoin", "Status": ("🟢 BUY " + ", ".join(buys)) if buys else
             (f"WAIT: {best['Ticker']} closest (readiness {int(best['Readiness'])})" if best is not None else "WAIT: none close"),
             "Note": g.get("alt_state", "")},
            {"Exposure": "Cash", "Status": "secondary" if buys else "✅ PREFERRED",
             "Note": "no coin passes every check" if not buys else "keep the rest of the capital"}]
    with st.container(border=True):
        st.markdown("**BTC vs altcoins vs cash**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def with_vs_btc(df: pd.DataFrame, rep) -> pd.DataFrame:
    vb = rep.vs_btc
    return df.merge(vb, on="Ticker", how="left") if vb is not None and not vb.empty and not df.empty else df


def after_tax(rep) -> None:
    """Illustrative only: Indian crypto tax takes 30% of each gain and losses cannot be set off."""
    t = rep.trades
    t = t[t["variant"] == "quality"] if t is not None and not t.empty else None
    if t is None or len(t) < 5:
        return
    r = t["r_multiple"]
    after = r.where(r <= 0, r * 0.7)
    risk_ps = (t["entry"] - t["stop"]).where(lambda x: x > 0)
    tds = (0.01 * t["exit"] / risk_ps).fillna(0)          # 1% TDS on each sale (recoverable later, but ties up cash)
    level = st.error if after.mean() <= 0 else st.warning
    level(f"**Strategy performance (before tax):** {r.mean():+.2f}R per trade after trading costs.  \n"
          f"**Illustrative tax impact:** if 30% were taken from every winning trade and losses could not be set off, that "
          f"would be about {after.mean():+.2f}R per trade; 1% TDS on each sale would also tie up about {tds.mean():.2f}R "
          "until refunded.  \n**Actual investor outcome:** depends on the order of your trades, your other income and "
          "current law; this is a rough illustration, not tax advice.")


@st.fragment(run_every=timedelta(seconds=60))
def view() -> None:
    if not background_running():
        try:
            eng.refresh(now_ist())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Crypto data could not be loaded: {exc}. Retrying automatically.")
    rep = eng.peek()
    if rep is None:
        progress_panel(eng, "the crypto research")
        return
    st.caption(f"Binance spot, public market data · {rep.symbols} most-traded USDT pairs · daily candles close 00:00 UTC "
               f"(05:30 IST); signals use the last completed day ({rep.session:%d %b %Y}) · updated {rep.as_of:%H:%M} IST · "
               "paper only, no orders are placed" + (" · DEMO MODE: synthetic coins" if DEMO else ""))
    cov = rep.coverage or {}
    if cov and cov.get("used", 0) < cov.get("requested", 0):
        why = []
        if cov.get("short_history"):
            why.append(f"{len(cov['short_history'])} have less than about a year of daily history (recent listings), "
                       "which the trend and breakout checks need")
        if cov.get("no_data"):
            why.append(f"{len(cov['no_data'])} returned no data")
        st.caption(f"Using {cov['used']} of the {cov['requested']} most-traded pairs: " + "; ".join(why) + ".")
    m1, m2, m3, m4 = st.columns(4)
    g = rep.regime
    m1.metric("Market (BTC)", ("🟢 " if g["risk_on"] else "🔴 ") + g["state"], f"{100 * g['btc_vs_200d']:+.1f}% vs 200-day")
    m2.metric("BTC, 3 months", f"{100 * g['btc_3m']:+.1f}%", f"{fmt(g['btc'])} USDT", delta_arrow="off", delta_color="off")
    m3.metric("Breadth", f"{100 * rep.breadth:.0f}%", "of coins above their 200-day", delta_arrow="off", delta_color="off")
    v = rep.validation
    m4.metric("Ranking model", "Validated" if v and v.get("validated") else "Not validated",
              f"IC {v['mean_ic']:.3f}, t {v['ic_tstat']:.1f}" if v else "not enough history", delta_arrow="off",
              delta_color="off")
    if v and not v.get("validated"):
        failed = [k for k, ok in v["checks"].items() if not ok]
        st.caption("Ranking model not validated. Failed: " + "; ".join(failed) + ". With few coins and a short history, "
                   "a high IC can be noise; the BUY rules still work without the model, it just adds no extra filter.")
    sig = rep.signals
    buys = sig[(sig["Action"] == "BUY") & (sig["Qty"] > 0)] if not sig.empty else sig
    env = ("🟢 **Market environment: favourable.** BTC's trend and breadth permit new positions." if g["risk_on"] else
           "🔴 **Market environment: unfavourable.** BTC is below its 200-day average, so the market-regime check "
           "blocks every new BUY.")
    opp = (f"🟢 **Trade opportunity: {len(buys)} coin(s) pass every check:** {', '.join(buys['Ticker'])}." if not buys.empty
           else "⚪ **Trade opportunity: none today.** No coin currently satisfies every entry requirement.")
    if not buys.empty:
        st.success(f"**Crypto decision: BUY at the next daily open (05:30 IST).**  \n{env}  \n{opp}")
    elif g["risk_on"]:
        st.warning(f"**Crypto decision: WAIT.**  \n{env}  \n{opp}  \nMarket permission is not a trade signal.")
    else:
        st.error(f"**Crypto decision: WAIT.**  \n{env}  \n{opp}")
    with st.container(border=True):
        st.markdown(f"**Market environment score: {g.get('score', '?')}/100** · {g.get('alt_state', '')} "
                    f"({100 * g.get('alt_share_beating_btc_30d', float('nan')):.0f}% of coins beat BTC over 30 days; "
                    f"median {100 * g.get('alt_median_vs_btc_30d', float('nan')):+.1f}%)")
        st.caption("This measures whether the environment permits trades. It does not indicate expected return or the "
                   "probability of profit.")
        with st.expander("What the score is based on"):
            st.markdown("\n".join(f"- {'✅' if ok else '⚠️'} {k}" for k, ok in g.get("components", {}).items()))
            st.caption("Descriptive. The BUY rule's market check is BTC above its 200-day average. 'Altcoins lagging BTC' "
                       "means extra altcoin risk has not been rewarded recently; any coin weaker than BTC over 3 months "
                       "already fails the trend check.")
    lab = rep.lab
    qrow = lab.set_index("Variant").loc["quality"] if lab is not None and not lab.empty and "quality" in set(lab["Variant"]) else None
    if qrow is not None:
        msg = (f"BUY rules backtested on crypto (rules carried over from NSE, awaiting crypto-specific validation): "
               f"{qrow['Trades']} trades (evidence {evidence_strength(int(qrow['Trades']))}), "
               f"{qrow['Win %']:.0f}% winners, {qrow['Avg R']:+.2f}R per trade after {100 * c.cost_pct:.1f}% costs, positive "
               f"in {qrow['Positive years %']:.0f}% of years. Verdict: **{qrow['Verdict']}**. Indian crypto tax (30% plus 1% "
               "TDS) is not included.")
        {"Passes": st.success, "Marginal": st.warning, "Fails": st.error}.get(qrow["Verdict"], st.info)(msg)

    exposure_panel(rep)
    crypto_readiness(rep)
    closest_setup(rep)
    from views.engines import simple_view
    if simple_view(cfg.simple_view_default):
        st.caption("Simple view. Switch off Simple view in the sidebar for the market score, evidence, tax illustration "
                   "and every tab.")
        return
    after_tax(rep)
    tabs = st.tabs(["BUY signals", "Watch list", "Strength vs BTC", "Evidence", "Movers", "Forward record"])
    money = {"Price": st.column_config.NumberColumn(format="%.6g"), "Stop Loss": st.column_config.NumberColumn("Stop", format="%.6g"),
             "Target Price": st.column_config.NumberColumn("Target", format="%.6g"),
             "Breakout level": st.column_config.NumberColumn(format="%.6g"),
             "Risk (INR)": st.column_config.NumberColumn("Risk (USDT)", format="%.2f"),
             "R:R": st.column_config.NumberColumn("Reward/risk", format="%.1f : 1"),
             "Readiness": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
             "Qty": st.column_config.NumberColumn(format="%.8g")}
    with tabs[0]:
        b = sig[sig["Action"] == "BUY"] if not sig.empty else sig
        st.caption(f"Same checks as NSE: breakout above the prior {cfg.crypto.breakout_lookback}-day high with a strong "
                   f"close, not extended, volume ≥ {c.volume_multiple:g}× average, at least ${10 * c.min_turnover:g}M a day "
                   "traded, uptrend beating BTC, ATR stop, BTC above its 200-day average, reward/risk ≥ 2:1. Entry only at "
                   f"the next daily open (05:30 IST), at most {c.entry_max_gap_pct:g}% above the signal close. Sized at "
                   f"{100 * c.risk_per_trade_pct:g}% risk of {c.account_equity_usdt:,.0f} USDT, max {c.max_open_positions} "
                   "positions; highly correlated coins are skipped.")
        if b.empty:
            st.info("No coin passed every BUY check.")
        else:
            st.dataframe(with_vs_btc(b.drop(columns=["Action"]), rep), hide_index=True, width="stretch", column_config=money)
    with tabs[1]:
        w = sig[sig["Action"] == "WATCH"].sort_values("Readiness", ascending=False) if not sig.empty else sig
        if w.empty:
            st.info("No coins are close to a BUY setup.")
        else:
            st.dataframe(with_vs_btc(w[["Ticker", "State", "Readiness", "Rank", "Price", "Breakout level", "R:R", "Checks"]],
                                     rep), hide_index=True, width="stretch", column_config=money)
            tk = st.selectbox("Chart", w["Ticker"].tolist())
            st.plotly_chart(range_chart(rep.histories[tk], cfg.breakout), width="stretch")
    with tabs[2]:
        vb = rep.vs_btc
        st.caption("Each coin's return minus Bitcoin's over the same period. A coin rising 20% while BTC rises 30% shows "
                   "-10%: relative weakness, extra risk without extra reward.")
        if vb is not None and not vb.empty:
            st.dataframe(vb, hide_index=True, width="stretch", column_config={
                c_: st.column_config.NumberColumn(format="%+.1f%%") for c_ in vb.columns if c_ != "Ticker"})
    with tabs[3]:
        if lab is not None and not lab.empty:
            st.dataframe(lab.assign(Evidence=lab["Trades"].map(lambda n: evidence_strength(int(n)))), hide_index=True,
                         width="stretch")
        if rep.regime_table is not None and not rep.regime_table.empty:
            st.markdown("**Results by market regime (BTC vs its 200-day average at the signal)**")
            st.dataframe(rep.regime_table, hide_index=True, width="stretch")
        q_trades = rep.trades[rep.trades["variant"] == "quality"] if rep.trades is not None and not rep.trades.empty else None
        res = stress_test(q_trades, c.risk_per_trade_pct, 0.20) if q_trades is not None and len(q_trades) >= 5 else None
        if res:
            level, text = stress_verdict(res, c.risk_per_trade_pct, 0.20)
            {"ok": st.success, "caution": st.warning, "danger": st.error}[level](text)
        if v:
            st.markdown(f"**Ranking model:** {'validated' if v['validated'] else 'not validated'} out of sample "
                        f"({v['oos_start']} to {v['oos_end']}); mean rank IC {v['mean_ic']:.3f}, t-stat {v['ic_tstat']:.1f}.")
        st.caption("Crypto history on Binance is shorter than NSE's and covers only coins still listed today "
                   "(survivorship bias), so treat these numbers as optimistic.")
    with tabs[4]:
        if rep.movers_now is not None and not rep.movers_now.empty:
            st.markdown("**Biggest 24-hour moves now**")
            st.dataframe(rep.movers_now.head(15), hide_index=True, width="stretch", column_config={
                "Price": st.column_config.NumberColumn(format="%.6g"), "24h %": st.column_config.NumberColumn(format="%+.2f%%"),
                "24h volume (USDT)": st.column_config.NumberColumn(format="%.3g")})
        level, text = movers_verdict(rep.movers_proxy)
        if not rep.movers_proxy.get("summary", pd.DataFrame()).empty:
            r5 = rep.movers_proxy["summary"].set_index("Hold").loc["5 sessions"]
            text += f" 5-day win rate: {evidence_label(r5['Win %'], int(r5['Mover trades']))}."
        {"ok": st.success, "bad": st.warning, "none": st.info}[level](text.replace("stock", "coin"))
        if not rep.movers_proxy.get("summary", pd.DataFrame()).empty:
            st.dataframe(rep.movers_proxy["summary"], hide_index=True, width="stretch")
    with tabs[5]:
        f = rep.forward
        h = f.get("health") or {}
        if h:
            {"ok": st.success, "caution": st.warning, "bad": st.error}.get(h.get("level"), st.info)(
                "Strategy health: " + h["text"])
        if f["summary"] is None or f["summary"].empty:
            st.info("Every crypto BUY signal from now on is saved with its rule version and followed automatically "
                    "(next-open entry, stop, target or time exit).")
        else:
            st.dataframe(f["summary"], hide_index=True, width="stretch")
            st.dataframe(f["results"], hide_index=True, width="stretch")


view()
st.caption("Research and paper trading only; not investment advice. Crypto is highly volatile and can fall sharply; "
           "gains are taxed in India at 30% plus 1% TDS. No exchange orders are placed.")
