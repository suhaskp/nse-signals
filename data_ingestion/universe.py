"""Load a ticker universe or a Superstar-investor list from CSV exports.

Accepts any CSV with one of these columns: ``NSE Code`` (Trendlyne exports),
``Symbol`` (NSE index constituent lists, e.g. the Nifty 500 CSV from
niftyindices.com), ``NSE Symbol`` or ``Ticker``. Rows without an NSE symbol
(BSE-only or SME listings) are skipped and reported.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.config import normalize_symbol

logger = logging.getLogger(__name__)
SYMBOL_COLUMNS = ("NSE Code", "Symbol", "NSE Symbol", "Ticker", "SYMBOL")


def load_symbols(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(nse_symbols, skipped_names)`` from a CSV file.

    Raises:
        ValueError: If no recognised symbol column exists.
    """
    df = pd.read_csv(path)
    col = next((c for c in SYMBOL_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path.name}: needs one of the columns {', '.join(SYMBOL_COLUMNS)}")
    name_col = next((c for c in ("Stock", "Company Name", "Name") if c in df.columns), col)
    has = df[col].notna() & df[col].astype(str).str.strip().ne("")
    symbols = list(dict.fromkeys(normalize_symbol(str(s)) for s in df.loc[has, col]))
    placeholders = [s for s in symbols if s.startswith("DUMMY")]  # NSE uses these during demergers
    symbols = [s for s in symbols if not s.startswith("DUMMY")]
    if placeholders:
        logger.info("%s: ignoring NSE placeholder symbols %s", path.name, ", ".join(placeholders))
    skipped = df.loc[~has, name_col].astype(str).tolist()
    if skipped:
        logger.warning("%s: %d rows have no NSE symbol and were skipped: %s",
                       path.name, len(skipped), ", ".join(skipped))
    return symbols, skipped


NIFTY500_URLS = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def refresh_nifty500(dest: Path, max_age_days: int = 7, getter=None) -> bool:
    """Download NSE's Nifty 500 constituent CSV if the local copy is missing or stale.

    Returns True if ``dest`` holds a usable list afterwards (fresh or previously cached).
    """
    import time as _time
    if dest.exists() and (_time.time() - dest.stat().st_mtime) < max_age_days * 86400:
        return True
    if getter is None:
        import requests
        getter = lambda url: requests.get(url, headers=_HEADERS, timeout=20)  # noqa: E731
    for url in NIFTY500_URLS:
        try:
            resp = getter(url)
            if resp.status_code == 200 and b"Symbol" in resp.content[:500]:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                logger.info("Nifty 500 list refreshed from %s", url)
                return True
            logger.warning("Nifty 500 list: %s returned HTTP %s", url, resp.status_code)
        except Exception as exc:  # noqa: BLE001 - network problems must not stop the dashboard
            logger.warning("Nifty 500 list: %s failed (%s)", url, exc)
    return dest.exists()


def latest_csv(folder: Path) -> Path | None:
    """Newest ``*.csv`` in ``folder`` (by modification time), or None."""
    files = sorted(Path(folder).glob("*.csv"), key=lambda p: p.stat().st_mtime) if Path(folder).exists() else []
    return files[-1] if files else None


def load_sectors(path: Path) -> dict[str, str]:
    """Symbol -> sector/industry from a constituent CSV (``Industry`` or ``Sector`` column)."""
    df = pd.read_csv(path)
    col = next((c for c in SYMBOL_COLUMNS if c in df.columns), None)
    sec = next((c for c in ("Industry", "Sector", "Basic Industry") if c in df.columns), None)
    if col is None or sec is None:
        return {}
    df = df.dropna(subset=[col])
    return {normalize_symbol(str(s)): str(v) for s, v in zip(df[col], df[sec].fillna("Unknown"))}
