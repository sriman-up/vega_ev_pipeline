# features/feature_builder.py
"""
Assembles the final station_features row from all feature groups.

This is the single entry point called by pipeline.py — it pulls together:
  - Consumption features      (consumption_features.py)
  - Competition features       (competition_features.py)
  - Geo / highway features     (geo_features.py)
  - Weather features           (monthly_weather table)
  - Calendar / seasonality     (computed inline)
  - Station capacity features  (from stations table)
  - EV market maturity         (months_since_opening, ramp-up phase)

Feature rationale — why each group matters for EV station revenue:
─────────────────────────────────────────────────────────────────
CONSUMPTION HISTORY
  Rolling averages and growth rate capture momentum. A station growing
  at 15% MoM is fundamentally different from one that peaked 6 months ago.
  CoV flags volatile stations where the model needs wider confidence intervals.

STATION CAPACITY
  total_power_kw (sum of all charger ratings) is a hard ceiling on revenue.
  charger_mix_ratio (fast/total) matters because fast chargers attract
  long-distance EV drivers rather than local top-up sessions.
  power_utilisation_pct reveals whether the station is supply- or demand-constrained.

COMPETITION
  nearby_ev_stations_1km is the most direct competitor signal.
  nearby_petrol_pumps_1km is a proxy for traffic volume at that node.
  nearby_hotels_1km indicates overnight dwell time (guests will charge while sleeping).
  amenity_score captures how comfortable the stop is — key for 45-min DC fast charge waits.

LOCATION TYPE
  A station at a highway rest area has a very different demand profile than
  one at a shopping mall or hotel. Encoding location_type as a categorical
  lets the model learn these distinct curves.

HIGHWAY POSITION
  highway_position_ratio captures where on the route the station sits.
  Stations near the 30–50% mark (150–250 km from Hyderabad on NH65) are
  the "range anxiety zone" where EV drivers must charge. Stations at 0–15%
  or 85–100% serve top-up demand from city-edge drivers.
  direction_side matters because morning traffic (outgoing from Hyd) and
  evening return traffic peak at different stations.

WEATHER
  Heat (>38°C) increases cabin AC load → higher per-km consumption → more
  charging events. Monsoon rain reduces highway traffic, directly cutting revenue.
  heatwave_days is the single strongest weather predictor for the Deccan plateau.

CALENDAR / SEASONALITY
  is_festival_month (Oct/Nov for Dussehra/Diwali) drives road trip peaks.
  is_monsoon (Jul–Sep) is the low season for highway EV travel in Telangana.
  days_in_month is a simple exposure correction (28 vs 31 days = 10% difference).

EV MARKET MATURITY
  months_since_opening controls for the ramp-up curve. Most stations take
  4–8 months to reach steady-state utilisation as local EV owners discover them.
  is_ramp_up_phase (first 6 months) lets the model treat new stations differently.

Usage:
    from features.feature_builder import build_feature_row
    row = build_feature_row(station, history, weather, feature_month="2026-04-01")
"""

import calendar
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ev_pipeline.features.competition_features import compute_competition_features
from ev_pipeline.features.consumption_features import compute_consumption_features
from ev_pipeline.features.geo_features import compute_geo_features, nearest_city
from ev_pipeline.utils.helpers import iso_to_date, remove_none

log = logging.getLogger(__name__)

# Festival months (1-indexed): Oct=10, Nov=11 (Dussehra + Diwali period)
FESTIVAL_MONTHS = {10, 11}
SUMMER_MONTHS   = {4, 5, 6}
MONSOON_MONTHS  = {7, 8, 9}


def build_feature_row(
    station: Dict[str, Any],
    history: List[Dict[str, Any]],
    weather: Optional[Dict[str, Any]] = None,
    feature_month: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete station_features row for one station × month.

    Args:
        station:       Row from stations table (fully enriched with Places + geo).
        history:       All monthly_bills rows for this station (any order).
        weather:       Row from monthly_weather for this station × month (can be None).
        feature_month: 'YYYY-MM-01'. Defaults to current month.

    Returns:
        Dict ready for db.upsert_station_features().
    """
    if not feature_month:
        now = datetime.utcnow()
        feature_month = f"{now.year}-{now.month:02d}-01"

    fm_date = iso_to_date(feature_month)
    scno    = station["unique_scno"]

    row: Dict[str, Any] = {
        "station_id":    station.get("id"),
        "unique_scno":   scno,
        "feature_month": feature_month,
    }

    # ── 1. Consumption features ───────────────────────────────────────────────
    cons = compute_consumption_features(history, up_to_month=feature_month)
    row.update(cons)

    for k in ("kwh_mom_change", "kwh_yoy_change", "kwh_growth_rate_pct", "is_anomaly"):
        row.pop(k, None)

    # ── 2. Competition & amenity features ─────────────────────────────────────
    comp = compute_competition_features(station, history)
    # Merge only the fields that belong in station_features
    for key in (
        "nearby_ev_stations_1km", "nearby_restaurants_1km", "nearby_hotels_1km",
        "nearby_petrol_pumps_1km", "nearby_shopping_1km",
        "has_attached_restaurant", "places_rating", "places_user_ratings_total",
        "competition_intensity", "amenity_score",
        "kwh_coefficient_of_variation", "demand_cliff_detected",
    ):
        if key in comp:
            row[key] = comp[key]

    row["location_type"] = station.get("location_type")

    # ── 3. Station capacity & infrastructure ──────────────────────────────────
    row["contracted_load_kva"]   = station.get("contracted_load_kva")
    row["total_charger_count"]   = station.get("total_charger_count")
    row["charger_30kw_count"]    = station.get("charger_30kw_count")
    row["charger_60kw_count"]    = station.get("charger_60kw_count")
    row["charger_120kw_count"]   = station.get("charger_120kw_count")
    row["charger_150kw_count"]   = station.get("charger_150kw_count")
    row["meter_phase"]           = station.get("meter_phase")
    row["security_deposit"]      = station.get("security_deposit")

    # Total installed power (kW)
    total_power = _compute_total_power(station)
    row["total_power_kw"] = total_power

    # Fast charger mix ratio
    total = station.get("total_charger_count") or 0
    fast  = (station.get("charger_60kw_count")  or 0) + \
            (station.get("charger_120kw_count") or 0) + \
            (station.get("charger_150kw_count") or 0)
    row["charger_mix_ratio"] = round(fast / total, 4) if total > 0 else None

    # Power utilisation: avg_kwh consumed / theoretical max (kW × hours in month)
    avg_kwh = cons.get("avg_kwh_units")
    if avg_kwh and total_power and fm_date:
        hours_in_month = calendar.monthrange(fm_date.year, fm_date.month)[1] * 24
        theoretical_max = total_power * hours_in_month
        row["power_utilisation_pct"] = round(100 * avg_kwh / theoretical_max, 4)
        # Same signal in hours rather than percent — easier to read on a dashboard
        # ("ran at full rated power for ~150h out of 720h possible"). Assumes
        # chargers draw their full rated kW whenever in use; not fed to the model
        # since it's a near-perfect linear rescale of power_utilisation_pct
        # (collinear — would add no information, only noise from day-count).
        row["estimated_uptime_hours"] = round(avg_kwh / total_power, 2)

    # ── 4. Highway & geo features ─────────────────────────────────────────────
    for key in (
        "dist_from_city_a_km", "dist_from_city_b_km", "dist_from_midpoint_km",
        "dist_from_quarter_a_km", "dist_from_quarter_b_km",
        "highway_position_ratio", "direction_side", "total_highway_length_km",
        "travel_time_from_city_a_min",
    ):
        if station.get(key) is not None:
            row[key] = station[key]

    # ── 5. Weather context ────────────────────────────────────────────────────
    if weather:
        for key in (
            "avg_temp_c", "max_temp_c", "total_rainfall_mm",
            "avg_humidity_pct", "heatwave_days", "rainfall_days",
        ):
            if weather.get(key) is not None:
                row[key] = weather[key]

    # ── 6. Calendar / seasonality ─────────────────────────────────────────────
    if fm_date:
        m = fm_date.month
        row["month_of_year"]       = m
        row["quarter"]             = (m - 1) // 3 + 1
        row["is_summer"]           = m in SUMMER_MONTHS
        row["is_monsoon"]          = m in MONSOON_MONTHS
        row["is_festival_month"]   = m in FESTIVAL_MONTHS
        row["days_in_month"]       = calendar.monthrange(fm_date.year, m)[1]

    # ── 7. EV market maturity ─────────────────────────────────────────────────
    supply_date = iso_to_date(station.get("supply_date"))
    if supply_date and fm_date:
        months_active = (fm_date.year - supply_date.year) * 12 + \
                        (fm_date.month - supply_date.month)
        row["months_since_opening"] = max(months_active, 0)
        row["is_ramp_up_phase"]     = months_active <= 6

    # ── 8. Target variable: next month's kwh ─────────────────────────────────
    future = sorted(
        [r for r in history
         if r.get("bill_month") and str(r["bill_month"])[:7] > feature_month[:7]],
        key=lambda r: str(r["bill_month"]),
    )
    if future:
        first = future[0]
        row["next_month_kwh"] = float(
            first.get("kwh_units") or first.get("billed_units") or 0
        )

    return remove_none(row)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_total_power(station: Dict[str, Any]) -> Optional[float]:
    """Sum up charger capacities from the station record."""
    total = 0.0
    charger_kw = {
        "charger_30kw_count":  30,
        "charger_60kw_count":  60,
        "charger_120kw_count": 120,
        "charger_150kw_count": 150,
    }
    found_any = False
    for col, kw in charger_kw.items():
        count = station.get(col) or 0
        if count:
            total += count * kw
            found_any = True

    # Also parse charger_other_json e.g. {"50kw": 2, "240kw": 1}
    other = station.get("charger_other_json") or {}
    if isinstance(other, dict):
        for key, count in other.items():
            import re
            m = re.search(r"(\d+)", str(key))
            if m:
                total += int(m.group(1)) * (count or 0)
                found_any = True

    return round(total, 2) if found_any else None