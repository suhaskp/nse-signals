"""Smoke test: the dashboard renders, runs a demo scan and shows every tab without errors."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_live_page_loads_by_itself(tmp_path, monkeypatch):
    """The live breakout page shows signals with no clicks (demo data)."""
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "live.py"), default_timeout=240).run()
    assert not at.exception, at.exception
    subheaders = [h.value for h in at.subheader]
    assert any(h.startswith("BUY (") for h in subheaders)
    assert any(h.startswith("SELL (") or h.startswith("EXIT warnings (") for h in subheaders)
    assert any("updated" in m.value for m in at.markdown)
    assert len(at.button) == 0  # nothing to click


def test_dashboard_opens_on_market_intelligence(tmp_path, monkeypatch):
    """app.py opens straight onto research + ranked ideas, with nothing to click."""
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=400).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Trade signals: BUY", "Trade signals: SELL / avoid",
                                          "Model portfolio (paper experiment)", "Stock research", "Model evidence",
                                          "Tuning", "Sectors", "Strategy lab", "Journal & changes"]
    labels = [m.label for m in at.metric]
    assert "Nifty trend" in labels and "Breadth" in labels
    assert at.success or at.warning  # the model-trust verdict is always shown
    assert len(at.get("plotly_chart")) >= 4
    assert len(at.button) == 0


def test_pages_never_print_python_objects(tmp_path, monkeypatch):
    """Guard against Streamlit 'magic' dumping DeltaGenerator docs onto the page."""
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=400).run()
    assert not at.exception
    assert not at.get("help")  # st.help / magic object renders
    assert all("DeltaGenerator" not in m.value for m in at.markdown)


def test_pages_show_progress_instead_of_blank_while_background_works(tmp_path, monkeypatch):
    """With the background refresher running and no results yet, pages render a progress panel at once."""
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    import engines_core
    monkeypatch.setattr(engines_core, "background_running", lambda: True)
    import views.engines as ve
    monkeypatch.setattr(ve, "background_running", lambda: True)
    intel, live, _ = engines_core.get_engines(True)
    monkeypatch.setattr(intel, "_last", None)
    monkeypatch.setattr(live, "_last", None)
    live.set_status("First run: downloading ~5 years of history: 200/500 stocks done")
    root = Path(__file__).resolve().parents[1]
    for page in ("views/live.py", "views/intelligence.py"):
        at = AppTest.from_file(str(root / page), default_timeout=30).run()
        assert not at.exception, at.exception
        assert any("Preparing" in i.value for i in at.info), page


def test_decision_line_is_first_and_model_portfolio_never_says_buy(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception
    boxes = [x.value for x in list(at.error) + list(at.warning) + list(at.success) + list(at.info)]
    assert any("Today's trading decision" in b for b in boxes)
    assert not any(b.startswith("**Rebalance at the next open") or "BUY: " in b for b in boxes if "Paper" in b or "Rebalance" in b)


def test_best_setup_card_and_exit_warning_label(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "app.py"), default_timeout=600).run()
    assert not at.exception
    text = " ".join(m.value for m in list(at.markdown) + list(at.caption))
    assert "Best setup right now" in text
    assert "Hard checks" in text and "Becomes BUY if" in text and "Becomes AVOID if" in text
    assert "Market regime:" in text
    live = AppTest.from_file(str(root / "views" / "live.py"), default_timeout=600).run()
    assert not live.exception
    subs = [h.value for h in live.subheader]
    assert any(s.startswith("EXIT warnings") or s.startswith("SELL") for s in subs)


def test_no_page_has_a_run_button(tmp_path, monkeypatch):
    """Everything is automatic: no page may ask the user to run a scan."""
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    for page in ("app.py", "views/live.py", "views/preopen.py"):
        at = AppTest.from_file(str(root / page), default_timeout=600).run()
        assert not at.exception, (page, at.exception)
        assert len(at.button) == 0, page
        assert not at.get("file_uploader"), page
    assert not (root / "views" / "eod.py").exists()


def test_preopen_page_runs_by_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "preopen.py"), default_timeout=600).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Watch list", "Trade chart", "Full scan", "History", "Settings in use"]
    assert any(m.label == "Passed the screen" for m in at.metric)


def test_metric_arrows_follow_the_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception
    breadth = next(m for m in at.metric if m.label == "Breadth")
    assert breadth.delta[0] in "+-" and "pts in 20 days" in breadth.delta   # signed, so the arrow is right
    assert "↑" not in breadth.delta and "↓" not in breadth.delta           # no hand-made arrows in the text


def test_notifications_pop_up_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    import engines_core
    from alerts import AlertCenter
    _, _, cfg = engines_core.get_engines(True)
    AlertCenter(Path(cfg.output_dir) / "alerts.json").add(f"test:{tmp_path.name}", "buy", "BUY signal: TESTCO for the next open.")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception
    assert any("TESTCO" in t.value for t in at.toast)                    # small pop-up notification
    assert any("TESTCO" in m.value for m in at.sidebar.markdown)        # and in the sidebar list


def test_daily_briefing_and_journal_tab(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Daily briefing" in text and "Cash score" in text and "Don't:" in text
    assert "Capital allocation for new money" in text
    assert "What changed since the previous session" in text and "Decision journal" in text
    assert "Multiple-testing check" in text and "Strategy health" in text


def test_crypto_page_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "crypto.py"), default_timeout=600).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["BUY signals", "Watch list", "Strength vs BTC", "Evidence", "Movers",
                                          "Forward record"]
    text = " ".join(m.value for m in at.markdown) + " ".join(x.value for x in list(at.warning) + list(at.error) + list(at.success))
    assert "Market environment" in text and "Trade opportunity" in text and "Market environment score" in text
    assert "BTC vs altcoins vs cash" in text
    assert any(m.label == "Market (BTC)" for m in at.metric)
    assert len(at.button) == 0


def test_simple_view_footer_and_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.setenv("PIPELINE_SIMPLE_VIEW", "1")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception
    assert len(at.tabs) == 0                                              # research tabs hidden
    assert any("ready for real money" in e.label for e in at.expander)
    side = " ".join(c.value for c in at.sidebar.caption)
    assert "App " in side and "NSE rules" in side and "last refresh" in side and "this computer only" in side
    assert any(t.label == "Simple view" for t in at.sidebar.toggle)


def test_no_table_needs_arrow_repair(tmp_path, monkeypatch):
    """Every table on every page must serialise cleanly (regression: a mixed int/text column logged a traceback
    on every refresh). Streamlit reports a repaired table through this logger."""
    import streamlit.dataframe_util as du
    repaired = []
    monkeypatch.setattr(du._LOGGER, "info", lambda msg, *a, **k: repaired.append(msg) if "Arrow" in str(msg) else None)
    monkeypatch.setenv("PIPELINE_DEMO", "1")
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    for page in ("app.py", "views/live.py", "views/crypto.py"):
        at = AppTest.from_file(str(root / page), default_timeout=600).run()
        assert not at.exception, (page, at.exception)
    assert not repaired, repaired


