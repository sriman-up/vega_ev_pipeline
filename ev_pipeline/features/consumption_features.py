# features/consumption_features.py
"""
Compute consumption-based features from monthly billing history.

All features are computed using only data up to (and including) the
target month — no data leakage for the ML model.
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Keys computed here that belong only in monthly_bills (not in station_features)
_BILL_LEVEL_KEYS = {"kwh_mom_change", "kwh_yoy_change", "kwh_growth_rate_pct", "is_anomaly"}

# Max absolute value for kwh_growth_rate_pct to fit NUMERIC(8,4)
_GROWTH_RATE_CLAMP = 9999.9999


def _to_date(s) -> Optional[date]:
    if isinstance(s, date):
        return s
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _py_float(v) -> float:
    """Convert any numpy scalar to a plain Python float."""
    return float(v)


def _py_bool(v) -> bool:
    """Convert any numpy bool to a plain Python bool."""
    return bool(v)


def compute_consumption_features(
    history: List[Dict[str, Any]],
    up_to_month: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute station-level consumption features up to up_to_month.
    Returns a dict suitable for station_features (bill-level keys excluded).
    """
    if not history:
        return {}

    cutoff = _to_date(up_to_month) if up_to_month else None
    rows = sorted(
        [r for r in history if _to_date(r.get("bill_month"))],
        key=lambda r: _to_date(r["bill_month"]),
    )
    if cutoff:
        rows = [r for r in rows if _to_date(r["bill_month"]) <= cutoff]
    if not rows:
        return {}

    def kwh(r):
        return _py_float(r.get("kwh_units") or r.get("billed_units") or 0)

    units = [kwh(r) for r in rows]
    months_active = len(rows)
    arr = np.array(units, dtype=float)

    rolling_3 = _py_float(np.mean(arr[-3:])) if len(arr) >= 3 else _py_float(np.mean(arr))
    rolling_6 = _py_float(np.mean(arr[-6:])) if len(arr) >= 6 else _py_float(np.mean(arr))

    kwh_growth_rate_overall = 0.0
    if len(arr) >= 3:
        x = np.arange(len(arr), dtype=float)
        slope = _py_float(np.polyfit(x, arr, 1)[0])
        mean_kwh = _py_float(np.mean(arr)) or 1.0
        kwh_growth_rate_overall = slope / mean_kwh

    summer_months = {4, 5, 6}
    winter_months = {11, 12, 1}
    summer_vals = [kwh(r) for r in rows if _to_date(r["bill_month"]).month in summer_months]
    winter_vals = [kwh(r) for r in rows if _to_date(r["bill_month"]).month in winter_months]

    zero_count = int(np.sum(arr == 0))
    pct_zero = round(100 * zero_count / months_active, 2) if months_active else 0.0

    is_anomaly = False
    if len(arr) >= 6:
        z = (arr[-1] - np.mean(arr)) / (np.std(arr) + 1e-9)
        is_anomaly = _py_bool(abs(z) > 2.5)

    # CoV
    mean_v = _py_float(np.mean(arr))
    std_v  = _py_float(np.std(arr))
    cv = round(std_v / (mean_v + 1e-9), 4) if mean_v > 0 else 0.0

    return {
        "months_active":              months_active,
        "avg_kwh_units":              round(_py_float(np.mean(arr)), 2),
        "std_kwh_units":              round(std_v, 2),
        "max_kwh_units":              round(_py_float(np.max(arr)), 2),
        "min_kwh_units":              round(_py_float(np.min(arr)), 2),
        "rolling_avg_3m_kwh":         round(rolling_3, 2),
        "rolling_avg_6m_kwh":         round(rolling_6, 2),
        "kwh_growth_rate_overall":    round(kwh_growth_rate_overall, 6),
        "seasonal_summer_avg_kwh":    round(_py_float(np.mean(summer_vals)), 2) if summer_vals else None,
        "seasonal_winter_avg_kwh":    round(_py_float(np.mean(winter_vals)), 2) if winter_vals else None,
        "pct_months_zero_consumption": pct_zero,
        "kwh_coefficient_of_variation": cv,
        "is_anomaly":                 is_anomaly,
        # ── NOTE: kwh_mom_change, kwh_yoy_change, kwh_growth_rate_pct are bill-level
        # keys (backfill_rolling_features) and must NOT be passed to station_features.
    }


def backfill_rolling_features(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each month in history compute per-row rolling/growth features.
    Returns the same list with added fields for bulk monthly_bills upserts.

    All values are plain Python floats/bools — safe for psycopg2.
    """
    sorted_hist = sorted(
        [r for r in history if _to_date(r.get("bill_month"))],
        key=lambda r: _to_date(r["bill_month"]),
    )

    def kwh(r):
        return _py_float(r.get("kwh_units") or r.get("billed_units") or 0)

    all_units = [kwh(r) for r in sorted_hist]

    for i, row in enumerate(sorted_hist):
        arr = np.array(all_units[: i + 1], dtype=float)

        row["rolling_avg_3m_kwh"] = round(_py_float(np.mean(arr[-3:])), 2)
        row["rolling_avg_6m_kwh"] = round(_py_float(np.mean(arr[-6:])), 2)

        if len(arr) >= 2:
            row["kwh_mom_change"] = round(_py_float(arr[-1] - arr[-2]), 2)
        else:
            row["kwh_mom_change"] = 0.0

        if len(arr) >= 13:
            row["kwh_yoy_change"] = round(_py_float(arr[-1] - arr[-13]), 2)
        else:
            row["kwh_yoy_change"] = 0.0

        # Growth rate — guard against zero previous month and NUMERIC(8,4) overflow
        if len(arr) >= 2:
            prev = _py_float(arr[-2])
            if prev == 0.0:
                row["kwh_growth_rate_pct"] = None   # undefined, not 0 or infinity
            else:
                raw = 100.0 * (_py_float(arr[-1]) - prev) / prev
                # Clamp to ±9999.9999 so it fits NUMERIC(8,4)
                row["kwh_growth_rate_pct"] = round(
                    max(-_GROWTH_RATE_CLAMP, min(_GROWTH_RATE_CLAMP, raw)), 4
                )
        else:
            row["kwh_growth_rate_pct"] = None

        if len(arr) >= 6:
            z = (arr[-1] - np.mean(arr)) / (np.std(arr) + 1e-9)
            row["is_anomaly"] = _py_bool(abs(z) > 2.5)
        else:
            row["is_anomaly"] = False

    return sorted_hist