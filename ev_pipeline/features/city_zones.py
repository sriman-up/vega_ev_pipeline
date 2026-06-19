# features/city_zones.py
"""
Concentric city-zone classification for EV stations.

Each station is assigned to its nearest anchor city (config.settings.ANCHOR_CITIES)
and banded by straight-line distance into:
  core       0-30 km   — dense urban demand, short-hop charging
  periurban  30-60 km  — commuter belt, mixed demand
  highway    60-120 km — intercity corridor, range-anxiety charging
  remote     120+ km   — sparse coverage, policy-gap candidates

Distinct from features.geo_features.nearest_city(), which matches against a
broader MAJOR_CITIES list for highway-pair projection — these are two
separate concerns living in separate stations columns
(nearest_city/dist_to_city_km/zone_band vs. nearest_major_city/dist_nearest_city_km).

Usage:
    from features.city_zones import compute_city_zone
    updates = compute_city_zone(lat, lon)   # -> {'nearest_city', 'dist_to_city_km', 'zone_band'}
"""

import logging
from typing import Any, Dict, Optional

from ev_pipeline.config.settings import ANCHOR_CITIES, ZONE_BAND_BOUNDS_KM
from ev_pipeline.features.geo_features import haversine_km

log = logging.getLogger(__name__)


def zone_band_for_distance(dist_km: float) -> str:
    """Map a distance (km) to its zone band using ZONE_BAND_BOUNDS_KM."""
    for bound_km, band in ZONE_BAND_BOUNDS_KM:
        if dist_km <= bound_km:
            return band
    return "remote"


def compute_city_zone(lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    """Given station coordinates, find the nearest anchor city and zone band."""
    if not lat or not lon:
        return {}
    lat, lon = float(lat), float(lon)

    dists = [
        (haversine_km(lat, lon, clat, clon), name)
        for name, clat, clon in ANCHOR_CITIES
    ]
    dists.sort()
    nearest_dist, nearest_name = dists[0]

    log.debug(
        "Station (%.5f,%.5f) -> nearest anchor city %s (%.1f km, zone=%s)",
        lat, lon, nearest_name, nearest_dist, zone_band_for_distance(nearest_dist),
    )

    return {
        "nearest_city":    nearest_name,
        "dist_to_city_km": round(nearest_dist, 2),
        "zone_band":       zone_band_for_distance(nearest_dist),
    }
