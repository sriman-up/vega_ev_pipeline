# utils/helpers.py
"""
Shared utility functions used across the EV pipeline.

Covers:
  - Logging setup
  - Date/string normalisation
  - Haversine distance
  - Safe type coercions
  - PDF filename → SCNo heuristic
  - Retry decorator
  - Simple in-memory cache for Places API responses
"""

import functools
import logging
import math
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logger with console + optional file handler."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=handlers,
        force=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Type coercions
# ─────────────────────────────────────────────────────────────────────────────

def to_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float, stripping commas."""
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def to_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def clean_str(val: Any) -> Optional[str]:
    """Strip whitespace; return None for empty strings."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def derive_station_name(
    consumer_name: Optional[str] = None,
    places_name: Optional[str] = None,
) -> Optional[str]:
    """
    Best available display name for a station: the TGSPDCL consumer_name
    (the billing-account holder, e.g. 'BALAGONI SWATHI') is preferred since
    it's always present from the seed PDFs; places_name (e.g. 'Ather Energy
    Charging Station') is the fallback when consumer_name is missing.
    """
    return clean_str(consumer_name) or clean_str(places_name)


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_ABBR = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def parse_ddmonyy(s: str) -> Optional[str]:
    """
    Parse dates like '30-MAY-23', '04-MAR-26', '14-JUL-2023'.
    Returns ISO 'YYYY-MM-DD' or None.
    """
    m = re.match(r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})", s.strip())
    if not m:
        return None
    day, mon_str, yr = m.groups()
    mon = _MONTH_ABBR.get(mon_str.lower())
    if not mon:
        return None
    yr = f"20{yr}" if len(yr) == 2 else yr
    return f"{yr}-{mon}-{int(day):02d}"


def bill_month_to_iso(month_year: str) -> Optional[str]:
    """
    'Apr/2026' or 'Apr/26' → '2026-04-01'.
    Returns None on parse failure.
    """
    m = re.match(r"([A-Za-z]{3})/(\d{2,4})", month_year.strip())
    if not m:
        return None
    mon = _MONTH_ABBR.get(m.group(1).lower())
    yr = m.group(2)
    yr = f"20{yr}" if len(yr) == 2 else yr
    if not mon:
        return None
    return f"{yr}-{mon}-01"


def iso_to_date(s: Any) -> Optional[date]:
    """Coerce an ISO string or date object to a date."""
    if isinstance(s, date):
        return s
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def current_bill_month() -> str:
    """Return the first day of the current month as 'YYYY-MM-01'."""
    now = datetime.utcnow()
    return f"{now.year}-{now.month:02d}-01"


# ─────────────────────────────────────────────────────────────────────────────
# Geospatial
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    # Coerce to float — psycopg2 returns Postgres `numeric` columns as
    # decimal.Decimal, which can't be mixed with float in arithmetic.
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# PDF filename heuristic
# ─────────────────────────────────────────────────────────────────────────────

def scno_from_filename(filename: str) -> Optional[str]:
    """
    Extract the Unique SCNo from a PDF filename.
    Expected patterns:
      114229478-Swathi-Kandujur.pdf  → '114229478'
      114313853-Tata-Srpt-History.pdf → '114313853'
    Returns the leading digit sequence if it looks like an SCNo (7–12 digits).
    """
    stem = Path(filename).stem
    m = re.match(r"(\d{7,12})", stem)
    return m.group(1) if m else None


def is_history_pdf(filename: str) -> bool:
    """Return True if the filename suggests a billing history PDF."""
    return bool(re.search(r"hist", filename, re.IGNORECASE))


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay_s: float = 2.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to max_attempts times with exponential backoff.

    Usage:
        @retry(max_attempts=3, delay_s=1.5, exceptions=(requests.RequestException,))
        def fetch_data(url):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay_s
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        log.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            func.__name__, attempt, max_attempts, e, wait,
                        )
                        time.sleep(wait)
                        wait *= backoff
            log.error("%s failed after %d attempts: %s", func.__name__, max_attempts, last_exc)
            raise last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Simple in-memory cache (for Places API — avoids duplicate calls in one run)
# ─────────────────────────────────────────────────────────────────────────────

_cache: Dict[str, Any] = {}


def cache_get(key: str) -> Optional[Any]:
    return _cache.get(key)


def cache_set(key: str, value: Any) -> None:
    _cache[key] = value


def cache_clear() -> None:
    _cache.clear()


def cached(key_fn: Callable[..., str]) -> Callable[[F], F]:
    """
    Decorator: cache return value in _cache using key_fn(*args, **kwargs) as key.

    Usage:
        @cached(lambda place_id: f"place_details:{place_id}")
        def _place_details(place_id):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            hit = cache_get(key)
            if hit is not None:
                log.debug("Cache hit: %s", key)
                return hit
            result = func(*args, **kwargs)
            if result is not None:
                cache_set(key, result)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def chunk(lst: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of at most `size`."""
    return [lst[i: i + size] for i in range(0, len(lst), size)]


def flatten(nested: List[List[Any]]) -> List[Any]:
    return [item for sublist in nested for item in sublist]


def dict_subset(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Return a new dict containing only the specified keys that exist in d."""
    return {k: d[k] for k in keys if k in d}


def remove_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of d with all None-valued keys removed."""
    return {k: v for k, v in d.items() if v is not None}