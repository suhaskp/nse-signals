"""Background refresher: keeps every engine current while the dashboard server runs.

Started once per server process (see ``views/engines.py``). Every minute it lets each
engine refresh; the engines throttle themselves (5 minutes in market hours, 30 minutes
otherwise; the heavy model work once per session), so this costs almost nothing when
there is nothing new. Because it runs inside the server, data stays fresh even when no
browser tab is open, and opening the page is instant.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class BackgroundRefresher:
    def __init__(self, jobs: list[tuple[str, Callable[[datetime], object]]], now_fn: Callable[[], datetime],
                 interval_seconds: int = 60) -> None:
        self.jobs, self.now_fn, self.interval = jobs, now_fn, interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="signal-refresher", daemon=True)
        self.last_run: datetime | None = None
        self.last_error: str | None = None

    def start(self) -> "BackgroundRefresher":
        self._thread.start()
        logger.info("Background refresher started (checks every %ds)", self.interval)
        return self

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> None:
        for name, job in self.jobs:
            try:
                job(self.now_fn())
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - keep running; the next cycle retries
                self.last_error = f"{name}: {exc}"
                logger.exception("Background %s refresh failed", name)
        self.last_run = self.now_fn()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval)
