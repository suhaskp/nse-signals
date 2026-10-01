"""Dashboard access to the shared engines (see engines_core.py), plus a shared progress panel."""
from datetime import datetime

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
