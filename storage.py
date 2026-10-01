"""Where the system keeps its records, how they survive reinstalls, and which rules produced them.

* Records live OUTSIDE the application folder, in ``PIPELINE_HOME`` (default: ``~/NSE_Signals``, i.e.
  ``C:\\Users\\<you>\\NSE_Signals`` on Windows). Reinstalling, updating or deleting an app copy never touches them.
* ``migrate_legacy`` copies records from older versions (which kept them inside the app folder) once.
* ``backup_records`` writes a daily zip of every record file and keeps the last 14.
* ``rule_version`` is a short fingerprint of every setting that affects signals; each recorded signal carries it.
* The rule freeze keeps the rules fixed for a period so the forward record measures one set of rules.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
LEGACY_DIRS = ("outputs", "data", "artifacts", "logs")
RECORD_SUFFIXES = {".csv", ".json", ".txt"}


def default_home() -> Path:
    return Path(os.getenv("PIPELINE_HOME") or (Path.home() / "NSE_Signals"))


def migrate_legacy(app_dir: Path, home: Path) -> list[str]:
    """Copy record folders from an old in-app installation into ``home`` (only if not already there)."""
    moved = []
    home.mkdir(parents=True, exist_ok=True)
    marker = home / ".migrated_from"
    for name in LEGACY_DIRS:
        src, dst = app_dir / name, home / name
        if src.is_dir() and any(src.iterdir()) and not dst.exists():
            shutil.copytree(src, dst)
            moved.append(name)
    if moved:
        marker.write_text(f"{app_dir}\n{datetime.now().isoformat()}\n{', '.join(moved)}\n")
        logger.info("Moved records from %s to %s: %s", app_dir, home, ", ".join(moved))
    return moved


def backup_records(home: Path, keep: int = 14, today: date | None = None) -> Path | None:
    """Zip every record file (CSV/JSON/TXT under outputs and data) once per day."""
    today = today or date.today()
    bdir = home / "backups"
    target = bdir / f"records_{today.isoformat()}.zip"
    if target.exists():
        return None
    files = [p for root in ("outputs", "data") if (home / root).exists()
             for p in (home / root).rglob("*") if p.is_file() and p.suffix.lower() in RECORD_SUFFIXES
             and "cache" not in p.parts]
    if not files:
        return None
    bdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(home))
    for old in sorted(bdir.glob("records_*.zip"))[:-keep]:
        old.unlink(missing_ok=True)
    return target


def rule_version(cfg) -> str:
    """8-character fingerprint of every setting that changes which signals are produced."""
    parts = {k: asdict(getattr(cfg, k)) for k in ("buy_quality", "breakout", "screener", "indicators", "risk")}
    parts["ranker"] = {k: v for k, v in asdict(cfg.ranker).items()}
    parts["model_threshold"] = cfg.model.probability_threshold
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def freeze_status(cfg, home: Path, today: date | None = None) -> dict:
    """Load or start the rule freeze. Returns start, until, frozen version, active flag and whether rules drifted."""
    today = today or date.today()
    path = home / "rules_freeze.json"
    version = rule_version(cfg)
    days = cfg.rules_freeze_days
    if days <= 0:
        return {"active": False, "disabled": True, "version": version}
    try:
        state = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"version": version, "start": today.isoformat(),
                 "until": (today + timedelta(days=days - 1)).isoformat(), "settings": None}
        home.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1))
    until = date.fromisoformat(state["until"])
    return {**state, "active": today <= until, "day": (today - date.fromisoformat(state["start"])).days + 1,
            "days_total": (until - date.fromisoformat(state["start"])).days + 1, "current_version": version,
            "drifted": version != state["version"], "path": str(path), "disabled": False}


def pin_freeze_settings(home: Path, settings: dict) -> None:
    """Remember the model-portfolio settings chosen at the start of the freeze."""
    path = home / "rules_freeze.json"
    try:
        state = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if state.get("settings") is None:
        state["settings"] = settings
        path.write_text(json.dumps(state, indent=1))
