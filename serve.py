"""Start the background refresher, then the dashboard web server, in one process.

    python serve.py            (what start_dashboard.bat / .sh run)

Single instance: a lock on 127.0.0.1:8599 ensures only one dashboard runs. A second start
opens the browser to the running one and exits with code 3 (the launchers then stop too).
The web server always uses port 8501, so it never drifts to 8502 and above.
"""
import socket
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOCK_PORT = 8599
URL = "http://localhost:8501"
ALREADY_RUNNING = 3


def acquire_lock(port: int = LOCK_PORT) -> socket.socket | None:
    """Hold a local port as a process-wide lock; None if another dashboard already holds it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: forbid sharing the port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        return s
    except OSError:
        s.close()
        return None


if __name__ == "__main__":
    lock = acquire_lock()
    if lock is None:
        print(f"The dashboard is already running at {URL}. Opening it in your browser; not starting a second copy.")
        webbrowser.open(URL)
        sys.exit(ALREADY_RUNNING)
    from config import load_config
    from storage import migrate_legacy
    home = load_config().data_home
    moved = migrate_legacy(ROOT, home)
    print(f"Records are kept in {home}" + (f" (moved from the app folder: {', '.join(moved)})" if moved else ""))
    from engines_core import start_background
    start_background()
    from streamlit.web import cli as stcli
    extra = sys.argv[1:]
    if not any(a.startswith("--server.port") for a in extra):
        extra = ["--server.port", "8501", *extra]
    if not load_config().allow_network_access and not any(a.startswith("--server.address") for a in extra):
        extra = ["--server.address", "127.0.0.1", *extra]  # only this computer can open the dashboard
    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), *extra]
    sys.exit(stcli.main())
