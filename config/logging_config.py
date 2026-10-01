"""Logging setup: console + rotating file handler, idempotent."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_HANDLER_TAG = "_premarket_pipeline_handler"


def setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    """Configure root logging for the pipeline.

    Args:
        log_dir: Directory for ``pipeline.log`` (created if missing).
        level: Logging level name.

    Returns:
        Path to the active log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    for noisy in ("urllib3", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # missing symbols are reported by the cache instead
    return log_file
