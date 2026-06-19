# features/geo_features.py
"""
Highway and geographic feature engineering for EV stations.

For each station (with lat/lon) we:
  1. Find the nearest highway city-pair from config.settings.HIGHWAY_PAIRS
  2. Project the station onto the highway line
  3. Compute distances from both city endpoints, midpoint, and quarter milestones
  4. Determine 'direction side' — is it on the outgoing side from city A or city B?
     Uses a heuristic: if position_ratio < 0.5 -> closer to city A (outgoing from A),
     else outgoing from B. Edge cases: 0.4–0.6 range = midpoint zone.

Usage:
    from features.geo_features import compute_geo_features
    feats = compute_geo_features(lat=17.08, lon=79.47)
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

from ev_pipeline.config.settings import (
    HIGHWAY_AVG_SPEED_KMH,
    HIGHWAY_PAIR_MAX_DIST_KM,
    HIGHWAY_PAIRS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Haversine
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Coerce to float — psycopg2 returns Postgres `numeric` columns as
    # decimal.Decimal, which can't be mixed with float in arithmetic.
    lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Project point onto a great-circle segment (approximated as flat for short distances)
# Returns (t, perp_dist_km) where t ∈ [0,1] is position along A->B
# ─────────────────────────────────────────────────────────────────────────────

def _project_onto_segment(
    lat_p: float, lon_p: float,
    lat_a: float, lon_a: float,
    lat_b: float, lon_b: float,
) -> Tuple[float, float]:
    """
    Flat-Earth projection of P onto segment A->B.
    Returns (t, perpendicular_distance_km).
    t=0 means at A, t=1 means at B.
    """
    # Convert to approximate Cartesian (km)
    def to_xy(lat, lon):
        x = lon * math.cos(math.radians(lat_a)) * 111.32
        y = lat * 110.574
        return x, y

    px, py = to_xy(lat_p, lon_p)
    ax, ay = to_xy(lat_a, lon_a)
    bx, by = to_xy(lat_b, lon_b)

    ab_x, ab_y = bx - ax, by - ay
    ap_x, ap_y = px - ax, py - ay

    ab_len_sq = ab_x ** 2 + ab_y ** 2
    if ab_len_sq < 1e-10:
        return 0.0, math.hypot(ap_x, ap_y)

    t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
    t = max(0.0, min(1.0, t))

    # Perpendicular distance
    proj_x = ax + t * ab_x
    proj_y = ay + t * ab_y
    perp = math.hypot(px - proj_x, py - proj_y)
    return t, perp


# ─────────────────────────────────────────────────────────────────────────────
# Main feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_geo_features(lat: float, lon: float) -> Dict[str, Any]:
    """
    Given station coordinates, compute all highway geo features.
    Matches the station to the nearest highway pair within HIGHWAY_PAIR_MAX_DIST_KM.
    """
    if not lat or not lon:
        return {}
    lat, lon = float(lat), float(lon)

    best: Optional[Dict] = None
    best_perp = float("inf")
    best_t = 0.0
    best_hp = None

    for hp in HIGHWAY_PAIRS:
        t, perp = _project_onto_segment(
            lat, lon,
            hp["lat_a"], hp["lon_a"],
            hp["lat_b"], hp["lon_b"],
        )
        if perp < best_perp:
            best_perp = perp
            best_t = t
            best_hp = hp

    if best_hp is None or best_perp > HIGHWAY_PAIR_MAX_DIST_KM:
        log.info(
            "Station at (%.5f, %.5f) is %.1f km from nearest highway — no highway match",
            lat, lon, best_perp,
        )
        return {}

    hp = best_hp
    t = best_t

    total_km = haversine_km(hp["lat_a"], hp["lon_a"], hp["lat_b"], hp["lon_b"])
    dist_from_a = round(t * total_km, 2)
    dist_from_b = round((1 - t) * total_km, 2)
    dist_from_mid = round(abs(t - 0.5) * total_km, 2)
    dist_from_q1 = round(abs(t - 0.25) * total_km, 2)   # 25% milestone from A
    dist_from_q3 = round(abs(t - 0.75) * total_km, 2)   # 75% milestone (= 25% from B)

    # Direction side heuristic
    if t < 0.40:
        direction_side = f"outgoing_from_{hp['city_a'].lower().replace(' ','_')}"
    elif t > 0.60:
        direction_side = f"outgoing_from_{hp['city_b'].lower().replace(' ','_')}"
    else:
        direction_side = "midpoint_zone"

    log.info(
        "Station (%.5f,%.5f) -> highway '%s' t=%.3f dist_A=%.1fkm dist_B=%.1fkm side=%s",
        lat, lon, hp["name"], t, dist_from_a, dist_from_b, direction_side,
    )

    return {
        "highway_name":             hp["name"],
        "nearest_city_a":           hp["city_a"],
        "nearest_city_b":           hp["city_b"],
        "dist_from_city_a_km":      dist_from_a,
        "dist_from_city_b_km":      dist_from_b,
        "dist_from_midpoint_km":    dist_from_mid,
        "dist_from_quarter_a_km":   dist_from_q1,
        "dist_from_quarter_b_km":   dist_from_q3,
        "total_highway_length_km":  round(total_km, 2),
        "highway_position_ratio":   round(t, 4),
        "direction_side":           direction_side,
        # Heuristic estimate (distance / HIGHWAY_AVG_SPEED_KMH) — no Distance
        # Matrix API is configured, so this is not live traffic-aware travel time.
        "travel_time_from_city_a_min": round(dist_from_a / HIGHWAY_AVG_SPEED_KMH * 60, 1),
        "travel_time_from_city_b_min": round(dist_from_b / HIGHWAY_AVG_SPEED_KMH * 60, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nearest city (for stations not on a known highway)
# ─────────────────────────────────────────────────────────────────────────────

MAJOR_CITIES = [
    ("Hyderabad",   17.3850, 78.4867),
    ("Vijayawada",  16.5062, 80.6480),
    ("Warangal",    17.9784, 79.5941),
    ("Nalgonda",    17.0575, 79.2672),
    ("Suryapet",    17.1400, 79.6200),
    ("Khammam",     17.2473, 80.1514),
    ("Miryalaguda", 16.8700, 79.5660),
]


def nearest_city(lat: float, lon: float) -> Dict[str, Any]:
    """Return name and distance to nearest major city."""
    if not lat or not lon:
        return {}
    lat, lon = float(lat), float(lon)
    dists = [
        (haversine_km(lat, lon, clat, clon), name)
        for name, clat, clon in MAJOR_CITIES
    ]
    dists.sort()
    nearest_dist, nearest_name = dists[0]
    return {
        "nearest_major_city":      nearest_name,
        "dist_nearest_city_km":    round(nearest_dist, 2),
    }