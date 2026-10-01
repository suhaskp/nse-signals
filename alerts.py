"""In-app notifications. Engines add alerts; the dashboard shows new ones as small pop-ups and lists recent ones.

Each alert has a key, so the same event (e.g. "BUY X for 1 Oct") is raised only once however often the
engines refresh. Stored in ``outputs/alerts.json`` (last 300).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from data_ingestion.data_fetcher import IST

_LOCK = threading.Lock()
ICONS = {"buy": "🟢", "ok": "✅", "skip": "🔴", "watch": "🟡", "warn": "⚠️", "info": "ℹ️"}


class AlertCenter:
    def __init__(self, path: Path, keep: int = 300) -> None:
        self.path, self.keep = Path(path), keep

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add(self, key: str, kind: str, text: str, when: datetime | None = None) -> bool:
        """Add an alert unless one with the same key exists. Returns True if it was new."""
        when = (when or datetime.now(IST)).astimezone(IST)
        with _LOCK:
            items = self._load()
            if any(a["key"] == key for a in items):
                return False
            items.append({"id": f"{when:%Y%m%d%H%M%S%f}-{len(items)}", "key": key, "kind": kind,
                          "icon": ICONS.get(kind, "ℹ️"), "text": text, "time": when.isoformat(timespec="seconds")})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(items[-self.keep:], indent=0))
            tmp.replace(self.path)
            return True

    def recent(self, n: int = 20) -> list[dict]:
        return list(reversed(self._load()))[:n]
