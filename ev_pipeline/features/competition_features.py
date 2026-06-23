# features/competition_features.py
"""
Competition and amenity feature engineering for EV stations.

Builds competition-related features from:
  - Google Places nearby search results (stored on the stations table)
  - Monthly bill history (to detect if usage correlates with competition changes)

These features are relatively static (refreshed monthly via Places API),
but are included in each month's station_features row for completeness.

Usage:
    from features.competition_features import compute_competition_features
    feats = compute_competition_features(station_dict, history_rows)
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ─────────────────────────────────────────────────────────────────────────────

def compute_competition_features(
    station: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build competition and amenity features for one station.

    Args:
        station:  Station dict from DB (must include Places-enriched fields).
        history:  Optional list of monthly_bills dicts for demand-side signals.

    Returns:
        Dict of competition features ready to merge into station_features.
    """
    feats: Dict[str, Any] = {}

    # ── Direct counts from Places enrichment ─────────────────────────────────
    nearby_ev     = station.get("nearby_ev_stations_1km")
    nearby_rest   = station.get("nearby_restaurants_1km")
    nearby_hotels = station.get("nearby_hotels_1km")
    nearby_petrol = station.get("nearby_petrol_pumps_1km")
    nearby_shop   = station.get("nearby_shopping_1km")
    has_rest      = station.get("has_attached_restaurant", False)
    rating        = station.get("places_rating")
    rating_count  = station.get("places_user_ratings_total")

    feats["nearby_ev_stations_1km"]   = nearby_ev     if nearby_ev     is not None else 0
    feats["nearby_restaurants_1km"]   = nearby_rest   if nearby_rest   is not None else 0
    feats["nearby_hotels_1km"]        = nearby_hotels if nearby_hotels is not None else 0
    feats["nearby_petrol_pumps_1km"]  = nearby_petrol if nearby_petrol is not None else 0
    feats["nearby_shopping_1km"]      = nearby_shop   if nearby_shop   is not None else 0
    feats["has_attached_restaurant"]  = bool(has_rest)
    feats["places_rating"]            = rating
    feats["places_user_ratings_total"] = rating_count

    # ── Competition intensity score (0–1 normalised, higher = more competition) ──
    # Heuristic: each nearby EV station within 1 km is a direct competitor.
    # Capped at 10 for normalisation.
    ev_count = feats["nearby_ev_stations_1km"]
    feats["competition_intensity"] = round(min(ev_count / 10.0, 1.0), 4)

    # ── Amenity score (0–1, higher = better traveller draw) ──────────────────
    # Components: attached restaurant (strong signal), nearby restaurants, rating
    amenity = 0.0
    if feats["has_attached_restaurant"]:
        amenity += 0.5
    rest_count = feats["nearby_restaurants_1km"]
    amenity += min(rest_count / 20.0, 0.3)   # up to 0.3 for 20+ restaurants nearby
    if rating:
        amenity += (float(rating) / 5.0) * 0.2       # up to 0.2 for 5-star rating
    feats["amenity_score"] = round(min(amenity, 1.0), 4)

    # ── Demand-side competition signals from billing history ─────────────────
    if history:
        feats.update(_demand_competition_signals(history))

    # ── Staleness flag ────────────────────────────────────────────────────────
    last_updated = station.get("competition_last_updated")
    if last_updated:
        try:
            lu = datetime.fromisoformat(str(last_updated))
            days_stale = (datetime.utcnow() - lu).days
            feats["competition_data_days_stale"] = days_stale
        except Exception:
            pass

    log.debug(
        "Competition features for SCNo %s: ev_nearby=%d amenity=%.2f intensity=%.2f",
        station.get("unique_scno"),
        feats["nearby_ev_stations_1km"],
        feats["amenity_score"],
        feats["competition_intensity"],
    )
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Demand-side signals from billing history
# ─────────────────────────────────────────────────────────────────────────────

def _demand_competition_signals(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Derive signals that hint at competitive pressure from the consumption history.
    E.g. sustained drops in usage after a period of growth could indicate a
    new competitor opened nearby.

    Returns a small dict of extra signals to merge into competition features.
    """
    feats: Dict[str, Any] = {}

    if len(history) < 4:
        return feats

    sorted_hist = sorted(
        [r for r in history if r.get("bill_month")],
        key=lambda r: str(r["bill_month"]),
    )

    def kwh(r):
        return float(r.get("kwh_units") or r.get("billed_units") or 0)

    units = [kwh(r) for r in sorted_hist]

    # Detect a "demand cliff" — 2+ consecutive months of >20% decline
    # after at least 3 months of stable/growing usage
    cliff_detected = False
    cliff_month: Optional[str] = None
    for i in range(3, len(units)):
        prev = units[i - 1]
        if prev == 0:
            continue
        drop_pct = (units[i] - prev) / prev
        if drop_pct < -0.20:
            # Check if prior 3 months were not declining
            prior_trend = units[i - 3: i]
            if len(prior_trend) >= 2 and prior_trend[-1] >= prior_trend[0]:
                cliff_detected = True
                cliff_month = str(sorted_hist[i]["bill_month"])
                break

    feats["demand_cliff_detected"] = cliff_detected
    if cliff_month:
        feats["demand_cliff_month"] = cliff_month

    # Coefficient of variation — high CV may indicate external disruption
    import numpy as np
    arr = [u for u in units if u > 0]
    if arr:
        cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
        feats["kwh_coefficient_of_variation"] = round(cv, 4)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Utility: merge competition features into a station_features row
# ─────────────────────────────────────────────────────────────────────────────

def apply_competition_features(
    feature_row: Dict[str, Any],
    station: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper — compute and merge competition features directly
    into an existing station_features dict.
    """
    comp = compute_competition_features(station, history)
    feature_row.update(comp)
    return feature_row