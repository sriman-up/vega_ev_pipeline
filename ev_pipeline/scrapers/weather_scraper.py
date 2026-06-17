# scrapers/weather_scraper.py
"""
Fetch monthly weather aggregates per station using the Open-Meteo
historical weather API — free, no API key required.

API docs: https://open-meteo.com/en/docs/historical-weather-api

Two important constraints:
  1. The archive endpoint only has data up to ~5 days ago. Requesting the
     current or future month returns a 400 Bad Request. fetch_monthly_weather()
     automatically clamps to the last fully-completed month when called for
     the current month.
  2. 'relative_humidity_2m_mean' is not a valid daily variable in Open-Meteo.
     We use 'precipitation_hours' as a proxy for rainy days and derive
     humidity from the hourly endpoint if needed. For now we omit humidity
     and use the reliable subset of daily variables only.

Valid daily variables used:
  temperature_2m_max, temperature_2m_min, temperature_2m_mean,
  precipitation_sum, wind_speed_10m_max, rain_sum, snowfall_sum
"""

import calendar
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from ev_pipeline.utils.helpers import retry

log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Only daily variables that Open-Meteo archive actually supports
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "precipitation_hours",   # hours of precipitation — proxy for rainy days
]


def _last_available_date() -> date:
    """Open-Meteo archive has a ~5-day lag. Return the latest safe date."""
    return datetime.utcnow().date() - timedelta(days=5)


def _clamp_to_available(year: int, month: int):
    """
    Returns (start_str, end_str) for the requested month, clamped so that
    end_date never exceeds the last available archive date.

    If the entire month is in the future / within the lag window, returns
    (None, None) — caller should skip the fetch.
    """
    first_day = date(year, month, 1)
    last_day  = date(year, month, calendar.monthrange(year, month)[1])
    cutoff    = _last_available_date()

    if first_day > cutoff:
        return None, None                          # month not available yet

    end = min(last_day, cutoff)
    return first_day.isoformat(), end.isoformat()


@retry(max_attempts=3, delay_s=2.0)
def _fetch_daily(lat: float, lon: float, start: str, end: str) -> Dict:
    params = {
        "latitude":   round(lat, 6),
        "longitude":  round(lon, 6),
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(_DAILY_VARS),
        "timezone":   "Asia/Kolkata",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
    # Surface the actual API error message before raising
    if not resp.ok:
        try:
            msg = resp.json().get("reason", resp.text[:200])
        except Exception:
            msg = resp.text[:200]
        log.error("Open-Meteo error %s: %s", resp.status_code, msg)
    resp.raise_for_status()
    return resp.json()


def fetch_monthly_weather(
    unique_scno: str,
    lat: float,
    lon: float,
    year: int,
    month: int,
) -> Optional[Dict[str, Any]]:
    """
    Fetch and aggregate weather for one station-month.

    Automatically handles:
      - Current/future months (clamps to last available archive date)
      - Partial months (aggregates whatever days are available)

    Returns a dict ready for the monthly_weather table, or None on failure.
    """
    if not lat or not lon:
        log.warning("No lat/lon for SCNo %s - skipping weather fetch", unique_scno)
        return None

    start, end = _clamp_to_available(year, month)
    if start is None:
        log.info(
            "Weather data for %04d-%02d not yet available (archive lag) - skipping SCNo %s",
            year, month, unique_scno,
        )
        return None

    log.info(
        "Fetching weather for SCNo %s at (%.4f, %.4f) for %s (available up to %s)",
        unique_scno, lat, lon, f"{year}-{month:02d}", end,
    )

    try:
        data = _fetch_daily(lat, lon, start, end)
    except Exception as e:
        log.error("Weather fetch failed for SCNo %s: %s", unique_scno, e)
        return None

    daily = data.get("daily", {})
    if not daily:
        log.warning("Empty daily data for SCNo %s %s", unique_scno, f"{year}-{month:02d}")
        return None

    def safe_mean(key) -> Optional[float]:
        vals = [v for v in daily.get(key, []) if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def safe_max(key) -> Optional[float]:
        vals = [v for v in daily.get(key, []) if v is not None]
        return round(max(vals), 2) if vals else None

    def safe_min(key) -> Optional[float]:
        vals = [v for v in daily.get(key, []) if v is not None]
        return round(min(vals), 2) if vals else None

    def safe_sum(key) -> Optional[float]:
        vals = [v for v in daily.get(key, []) if v is not None]
        return round(sum(vals), 2) if vals else None

    def count_above(key, threshold) -> int:
        return sum(1 for v in daily.get(key, []) if v is not None and v > threshold)

    # "rainfall_days" = days where precipitation_sum > 2.5mm
    # Fallback: count days where precipitation_hours > 0 if precip_sum unavailable
    precip_vals = daily.get("precipitation_sum", [])
    if any(v is not None for v in precip_vals):
        rainfall_days = count_above("precipitation_sum", 2.5)
    else:
        rainfall_days = count_above("precipitation_hours", 0)

    return {
        "unique_scno":        unique_scno,
        "weather_month":      f"{year}-{month:02d}-01",
        "avg_temp_c":         safe_mean("temperature_2m_mean"),
        "max_temp_c":         safe_max("temperature_2m_max"),
        "min_temp_c":         safe_min("temperature_2m_min"),
        "total_rainfall_mm":  safe_sum("precipitation_sum"),
        "avg_wind_speed_kmh": safe_mean("wind_speed_10m_max"),
        "heatwave_days":      count_above("temperature_2m_max", 40),
        "rainfall_days":      rainfall_days,
        # avg_humidity_pct intentionally omitted — not available as a daily
        # aggregate in Open-Meteo archive without the hourly endpoint.
    }


def fetch_weather_for_all_stations(
    stations: List[Dict],
    year: int,
    month: int,
) -> Dict[str, Optional[Dict]]:
    """Batch fetch. Returns {unique_scno: weather_dict or None}."""
    results = {}
    for i, s in enumerate(stations):
        scno = s["unique_scno"]
        results[scno] = fetch_monthly_weather(
            scno, s.get("latitude"), s.get("longitude"), year, month
        )
        if i < len(stations) - 1:
            time.sleep(0.3)
    return results