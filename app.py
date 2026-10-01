"""Dashboard entry point:  python serve.py  (or streamlit run app.py)

Opens on the Market intelligence page. Every page updates itself; there is nothing to run manually.
New alerts pop up as small notifications, and the sidebar keeps a list of recent ones.
"""
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="NSE signals", page_icon="📈", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; font-variant-numeric: tabular-nums; }
.block-container { padding-top: 2rem; max-width: 1400px; }
.alert-row { font-size: 0.82rem; line-height: 1.3; margin: 0 0 .45rem 0; }
.alert-time { color: #5B6B82; font-size: 0.72rem; }
</style>""", unsafe_allow_html=True)

from alerts import AlertCenter  # noqa: E402
from data_ingestion.data_fetcher import IST  # noqa: E402
from views.engines import get_engines, start_background  # noqa: E402

start_background()  # keeps data fresh even when no browser tab is open
_, _, CFG = get_engines()


@st.fragment(run_every=timedelta(seconds=20))
def notifications() -> None:
    """Pop up new alerts as small notifications and list the most recent ones."""
    items = AlertCenter(Path(CFG.output_dir) / "alerts.json").recent(15)
    seen = st.session_state.get("seen_alerts")
    if seen is None:  # first view in this browser tab: show what happened in the last 12 hours (up to 3)
        cutoff = datetime.now(IST) - timedelta(hours=12)
        for a in reversed([a for a in items if datetime.fromisoformat(a["time"]) >= cutoff][:3]):
            st.toast(a["text"], icon=a["icon"])
        st.session_state["seen_alerts"] = {a["id"] for a in items}
    else:
        new = [a for a in items if a["id"] not in seen]
        for a in reversed(new[:5]):
            st.toast(a["text"], icon=a["icon"])
        seen.update(a["id"] for a in new)
    st.markdown("**🔔 Alerts**")
    if not items:
        st.caption("No alerts yet. BUY signals, opening-check results, intraday breakouts, stale data and regime "
                   "changes will appear here and pop up on screen.")
        return
    for a in items[:8]:
        t = datetime.fromisoformat(a["time"])
        st.markdown(f'<p class="alert-row">{a["icon"]} {a["text"]}<br><span class="alert-time">'
                    f'{t:%d %b %H:%M}</span></p>', unsafe_allow_html=True)


def footer() -> None:
    """Version and heartbeat, so it is always clear what is running and whether it is up to date."""
    from crypto_engine import crypto_rule_version
    from engines_core import get_crypto
    from pipeline import session_status
    from status import APP_VERSION, heartbeat
    from storage import rule_version
    from views.engines import get_engines as _ge, now_ist
    intel, live, _ = _ge()
    now = now_ist()
    crypto = get_crypto() if CFG.crypto.enabled else None
    beats = heartbeat(intel.peek(), live.peek(), crypto.peek() if crypto else None, now, session_status(CFG, now)[0])
    st.divider()
    for b in beats:
        when = b["last"].strftime("%d %b %H:%M") if b["last"] else "not yet"
        st.caption(("⚠️ " if b["overdue"] else "") + f"{b['engine']}: last refresh {when}")
    st.caption(f"App {APP_VERSION} · NSE rules {rule_version(CFG)}"
               + (f" · crypto rules {crypto_rule_version(CFG)}" if CFG.crypto.enabled else "")
               + (" · reachable from other devices" if CFG.allow_network_access else " · this computer only"))


pages = [st.Page("views/intelligence.py", title="Market intelligence", icon="🧠", default=True),
         st.Page("views/live.py", title="Live breakout signals", icon="📡")]
if CFG.crypto.enabled:
    pages.append(st.Page("views/crypto.py", title="Crypto (Binance)", icon="🪙"))
if CFG.preopen_scan_enabled:
    pages.append(st.Page("views/preopen.py", title="Pre-open gap scan (experimental)", icon="🧪"))
page = st.navigation(pages)
with st.sidebar:
    from views.engines import simple_view
    st.toggle("Simple view", value=simple_view(CFG.simple_view_default), key="simple_view",
              help="Just the decision, the best setup and alerts. Switch off for every research tab.")
    notifications()
    footer()
page.run()
