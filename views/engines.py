"""Dashboard access to the shared engines (see engines_core.py), plus a shared progress panel."""
from datetime import datetime

import pandas as pd
import streamlit as st

from engines_core import DEMO, background_running, get_crypto, get_engines, get_preopen, now_ist, start_background
from data_ingestion.data_fetcher import IST

__all__ = ["DEMO", "background_running", "get_crypto", "get_engines", "get_preopen", "now_ist", "start_background", "progress_panel"]


def progress_panel(engine, what: str) -> None:
    """Shown instead of a blank page while the background prepares the first results."""
    elapsed = int((datetime.now(IST) - engine.status_since).total_seconds())
    mins, secs = divmod(elapsed, 60)
    st.info(f"**Preparing {what}.** {engine.status} ({mins}m {secs:02d}s so far). This page updates by itself; "
            "the very first start downloads about 5 years of prices for 500 stocks and takes several minutes. "
            "After that it opens instantly.")
    if engine.last_error:
        st.warning(f"Last attempt hit a problem: {engine.last_error}. Retrying automatically.")


def simple_view(default: bool) -> bool:
    """Whether the page shows the simple view (sidebar switch; PIPELINE_SIMPLE_VIEW overrides the default)."""
    import os
    import streamlit as st
    env = os.getenv("PIPELINE_SIMPLE_VIEW")
    base = default if env is None else env == "1"
    return bool(st.session_state.get("simple_view", base))


def closest_to_buy(qd: pd.DataFrame | None, q, market: str, data_stale: bool, exclude: str | None = None) -> None:
    """Top-of-page reassurance: the most complete setup and exactly what it still needs. No timing forecasts."""
    import decisions
    if qd is None or qd.empty:
        return
    live = qd[qd["State"].isin(["BUY NOW", "WATCH CLOSELY", "WAIT"])]
    if exclude:
        live = live[live["Ticker"] != exclude]
    if live.empty:
        st.caption("No setup is close to a BUY right now. The system is waiting for one to form.")
        return
    r = live.sort_values("Readiness", ascending=False).iloc[0]
    missing = decisions.needs(r, q, data_stale, market)
    with st.container(border=True):
        if not missing:
            st.markdown(f"**🟢 {r['Ticker']} passes every check** · readiness {int(r['Readiness'])}/100")
            return
        st.markdown(f"**Closest to a BUY: {'🟡' if r['State'] == 'WATCH CLOSELY' else '⚪'} {r['Ticker']}** · "
                    f"readiness {int(r['Readiness'])}/100")
        st.markdown("**Needs:** " + " · ".join(f"🔴 {m}" for m in missing) + "  \n**Everything else:** ✅ passes")
        st.caption("A BUY becomes possible only if all of these confirm on a completed daily candle. No forecast of "
                   "when: don't anticipate the signal. Buying before confirmation is a different strategy from the one "
                   "that was tested.")
