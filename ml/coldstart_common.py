# ml/coldstart_common.py
"""
Shared constants for the cold-start ramp/stabilized models
(ml/train_coldstart_ramp.py, ml/train_coldstart_stabilized_single.py,
ml/train_coldstart_stabilized_permonth.py) and their consumers
(ml/site_simulator.py, ml/coldstart_validation.py).

Single source of truth for the feature list so training and inference can't
drift out of sync. Deliberately excludes every consumption-history column
from ml/train.py's NUMERIC_FEATURES (avg_kwh_units, rolling_avg_3m_kwh,
rolling_avg_6m_kwh, kwh_growth_rate_overall, std_kwh_units, months_active,
pct_months_zero_consumption, kwh_coefficient_of_variation,
seasonal_summer_avg_kwh, seasonal_winter_avg_kwh, power_utilisation_pct) —
those are never available for a genuinely new station (see
ml/site_simulator.py's module docstring), and the general model has never
once seen a row where they're missing (100% populated in training — every
station's own first feature row already includes that month's own bill),
which is the root cause this whole module exists to work around.

months_since_active (re-zeroed at each station's first non-zero bill, not
supply_date — see db_manager.get_coldstart_training_matrix()'s docstring)
is added explicitly by callers, not included here, since whether it's a
live feature or a fixed bucketing key differs per model:
  - ramp model: live feature (varies 0-5 within the pooled window)
  - Family A stabilized model: live feature (varies 6+ )
  - Family B per-calendar-month models: live feature (varies within each
    calendar-month bucket, since the bucket is no longer position-fixed)
month_of_year is the one column that flips the other way — it's a live
feature everywhere except Family B's per-calendar-month models, where it's
the bucketing key itself (constant per model) and must be dropped from that
family's feature list.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ml.train import BOOL_FEATURES, CATEGORICAL_FEATURES  # reused as-is

COLDSTART_STATIC_FEATURES = [
    # Capacity
    "contracted_load_kva", "total_charger_count", "total_power_kw", "charger_mix_ratio",
    # Geo / highway position
    "dist_from_city_a_km", "dist_from_city_b_km", "dist_from_midpoint_km", "highway_position_ratio",
    # Competition / amenities
    "nearby_ev_stations_1km", "nearby_restaurants_1km", "nearby_hotels_1km", "nearby_petrol_pumps_1km",
    "competition_intensity", "amenity_score",
    # Calendar
    "month_of_year",
    # Weather
    "avg_temp_c", "max_temp_c", "total_rainfall_mm", "heatwave_days", "rainfall_days",
]

# is_ramp_up_phase is derived from the old months_since_opening clock (relative
# to supply_date), not the re-zeroed months_since_active clock these models
# use — excluded here since it would be actively misleading.
COLDSTART_BOOL_FEATURES = [c for c in BOOL_FEATURES if c != "is_ramp_up_phase"]
COLDSTART_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
COLDSTART_TARGET = "target_kwh"

RAMP_MIN, RAMP_MAX = 0, 5

# No upper bound here, deliberately — "stabilized" has no natural ceiling
# (once growth plateaus around month 6, it just stays plateaued), unlike the
# ramp window's 0-5 boundary, which IS data-driven (median-kWh growth rate
# decelerates sharply right around month 6 — see ml/train_coldstart_ramp.py's
# sibling docstrings / the coldstart_validation report). An earlier version
# of this constant included STABILIZED_MAX=11 to match the validation
# report's old fixed 12-month horizon, but that made Family A's stabilized
# model extrapolate for any station with more than 12 months of real
# history (now the common case, since the report predicts each station's
# full available span) — LightGBM can't differentiate months past whatever
# max months_since_active it saw in training, so every month beyond 11
# collapsed into the same leaf as month 11, producing wildly wrong
# predictions. Family A now trains on the same unbounded
# months_since_active >= 6 window as Family B.
STABILIZED_MIN = 6

# Resolved numeric-feature lists for inference (ml/site_simulator.py,
# ml/coldstart_validation.py) — kept here, not recomputed per call site, so
# both can resolve a trajectory step identically.
RAMP_NUMERIC_FEATURES = COLDSTART_STATIC_FEATURES + ["months_since_active"]
PERMONTH_NUMERIC_FEATURES = [c for c in COLDSTART_STATIC_FEATURES if c != "month_of_year"] + ["months_since_active"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared trajectory-vs-actuals comparison — used by ml.coldstart_validation's
# HTML report AND ml.site_simulator.predict_existing_station_trajectory's
# live "test prediction" API path, so both compare the exact same way.
# ─────────────────────────────────────────────────────────────────────────────

def month_diff(start: str, end: str) -> int:
    """Inclusive month count between two 'YYYY-MM...' strings — e.g.
    2023-11 to 2026-02 -> 28 (28 calendar months of data)."""
    y1, m1 = int(start[:4]), int(start[5:7])
    y2, m2 = int(end[:4]), int(end[5:7])
    return (y2 - y1) * 12 + (m2 - m1) + 1


def actual_series(history: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Sorted (bill_month, kwh) pairs, with the station's LEADING run of
    kwh=0 bills dropped — those are commissioning/admin-delay artifacts
    ("not started yet"), not real ramp-up signal, consistent with how
    db_manager.get_coldstart_training_matrix() treats training data.
    """
    series = sorted(
        (str(r["bill_month"])[:10], float(r.get("kwh_units") or r.get("billed_units") or 0))
        for r in history if r.get("bill_month")
    )
    first_real_idx = next((i for i, (_, v) in enumerate(series) if v > 0), None)
    if first_real_idx is None:
        return []
    return series[first_real_idx:]


def compare_trajectory_to_actuals(
    series: List[Tuple[str, float]],
    trajectory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aligns each predicted month with the REAL actual for that same calendar
    month (not month1/3mo/lifetime — a station's month-1 ramp-up and its
    year-3 steady state are different regimes, comparing a single point
    against both conflates them). Returns per-month rows plus the mean abs
    % error over however many months actually overlap.
    """
    actual_by_month = {m[:7]: v for m, v in series}
    rows = []
    for step in trajectory:
        month = step["feature_month"][:7]
        actual = actual_by_month.get(month)
        pct_error = None
        if actual is not None and actual != 0:
            pct_error = round(100 * (step["predicted_kwh"] - actual) / actual, 1)
        rows.append({
            "feature_month": month,
            "months_since_active": step.get("months_since_active"),
            "predicted_kwh": step["predicted_kwh"],
            "predicted_kwh_lower": step["predicted_kwh_lower"],
            "predicted_kwh_upper": step["predicted_kwh_upper"],
            "actual_kwh": actual,
            "pct_error": pct_error,
        })
    errs = [abs(r["pct_error"]) for r in rows if r["pct_error"] is not None]
    mape = round(float(np.mean(errs)), 1) if errs else None
    return {"rows": rows, "mape": mape, "n_matched": len(errs)}


# ─────────────────────────────────────────────────────────────────────────────
# "Estimate C" — disagreement-gated blend of Family A (single) and Family B
# (permonth). Investigated directly against station 114631197 (TECSO CHARGE
# ZONE LIMITED): Family B's 12 calendar-month models are each trained on far
# fewer rows than Family A's one pooled model (162-483 vs 3,877), and several
# of that station's months hit a B-specific collapse (e.g. 2025-09: B=463
# kWh vs A=6567, actual=8520) that's a small-sample artifact in one calendar
# bucket, not real seasonal signal. A plain average gets dragged toward
# whatever number is wrong — confirmed: avg(6567, 463)=3515 is FARTHER from
# the actual 8520 than A alone. So this only averages A and B when they
# roughly agree; when B diverges sharply from A, it falls back to A alone
# rather than let the average get pulled toward the outlier.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DIVERGENCE_THRESHOLD = 2.5


def blend_trajectory_steps(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Combine Family A's and Family B's prediction for ONE trajectory step
    (same feature_month). Falls back to step_a unchanged when the two
    disagree by more than divergence_threshold x in either direction;
    otherwise averages predicted_kwh and both bounds."""
    a_kwh, b_kwh = step_a["predicted_kwh"], step_b["predicted_kwh"]
    if a_kwh <= 0 or b_kwh <= 0 or max(a_kwh, b_kwh) / max(min(a_kwh, b_kwh), 1e-9) > divergence_threshold:
        return dict(step_a)
    return {
        **step_a,
        "predicted_kwh": round((a_kwh + b_kwh) / 2, 1),
        "predicted_kwh_lower": round((step_a["predicted_kwh_lower"] + step_b["predicted_kwh_lower"]) / 2, 1),
        "predicted_kwh_upper": round((step_a["predicted_kwh_upper"] + step_b["predicted_kwh_upper"]) / 2, 1),
    }


def blend_trajectories(
    trajectory_a: List[Dict[str, Any]],
    trajectory_b: List[Dict[str, Any]],
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """blend_trajectory_steps() applied across two full trajectories,
    aligned by feature_month."""
    by_month_b = {step["feature_month"]: step for step in trajectory_b}
    return [
        blend_trajectory_steps(step_a, by_month_b[step_a["feature_month"]], divergence_threshold)
        for step_a in trajectory_a
        if step_a["feature_month"] in by_month_b
    ]
