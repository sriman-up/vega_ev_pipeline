# features/spatial_h3.py
"""
H3 hex tiling for coverage-gap analysis.

Assigns each station an H3 cell index at resolutions 5/6/7, then builds a
full hex grid over the Telangana bounding box at each resolution and flags
hexes where the nearest station is farther than H3_COVERAGE_GAP_KM away —
see config.settings.H3_COVERAGE_GAP_KM for the policy rationale.

Usage:
    from features.spatial_h3 import assign_h3_indices, build_coverage_zones

    updates = assign_h3_indices(lat, lon)            # -> {'h3_res5': ..., ...}
    zones   = build_coverage_zones(stations, res=6)  # -> list of h3_coverage_zones rows
"""

import logging
from typing import Any, Dict, List, Optional

import h3
import numpy as np

from ev_pipeline.config.settings import H3_COVERAGE_GAP_KM, H3_RESOLUTIONS, TELANGANA_BBOX
from ev_pipeline.features.geo_features import compute_geo_features

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Station-level H3 index assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_h3_indices(lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    """Return {'h3_res5': ..., 'h3_res6': ..., 'h3_res7': ...} for one point."""
    if not lat or not lon:
        return {}
    lat, lon = float(lat), float(lon)
    return {f"h3_res{res}": h3.latlng_to_cell(lat, lon, res) for res in H3_RESOLUTIONS}


# ─────────────────────────────────────────────────────────────────────────────
# Coverage-gap grid
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_polygon():
    b = TELANGANA_BBOX
    return h3.LatLngPoly([
        (b["min_lat"], b["min_lon"]),
        (b["min_lat"], b["max_lon"]),
        (b["max_lat"], b["max_lon"]),
        (b["max_lat"], b["min_lon"]),
    ])


def _nearest_station_dist_km(
    centers_lat: List[float], centers_lon: List[float],
    station_lat: List[float], station_lon: List[float],
) -> np.ndarray:
    """Vectorized haversine — min distance (km) from each hex centre to any station."""
    R = 6371.0
    clat = np.radians(np.asarray(centers_lat))[:, None]
    clon = np.radians(np.asarray(centers_lon))[:, None]
    slat = np.radians(np.asarray(station_lat))[None, :]
    slon = np.radians(np.asarray(station_lon))[None, :]
    dlat = slat - clat
    dlon = slon - clon
    a = np.sin(dlat / 2) ** 2 + np.cos(clat) * np.cos(slat) * np.sin(dlon / 2) ** 2
    dist_matrix = R * 2 * np.arcsin(np.sqrt(a))
    return dist_matrix.min(axis=1)


def build_coverage_zones(
    stations: List[Dict[str, Any]],
    resolution: int,
    gap_threshold_km: float = H3_COVERAGE_GAP_KM,
) -> List[Dict[str, Any]]:
    """
    Build h3_coverage_zones rows for every hex in the Telangana bbox at `resolution`.

    For each hex: count stations whose h3_res{resolution} matches it, find the
    nearest station's distance from the hex centre, and flag is_coverage_gap
    when that distance exceeds gap_threshold_km.
    """
    located = [
        s for s in stations
        if s.get("latitude") is not None and s.get("longitude") is not None
    ]
    if not located:
        log.warning("No stations with coordinates — cannot build coverage zones")
        return []

    station_lat = [float(s["latitude"]) for s in located]
    station_lon = [float(s["longitude"]) for s in located]

    h3_col = f"h3_res{resolution}"
    station_counts: Dict[str, int] = {}
    for s, slat, slon in zip(located, station_lat, station_lon):
        cell = s.get(h3_col) or h3.latlng_to_cell(slat, slon, resolution)
        station_counts[cell] = station_counts.get(cell, 0) + 1

    cells = list(h3.polygon_to_cells(_bbox_polygon(), resolution))
    log.info("Resolution %d: %d hexes covering the Telangana bbox", resolution, len(cells))

    centers = [h3.cell_to_latlng(c) for c in cells]
    centers_lat = [c[0] for c in centers]
    centers_lon = [c[1] for c in centers]
    nearest_dists = _nearest_station_dist_km(centers_lat, centers_lon, station_lat, station_lon)

    rows: List[Dict[str, Any]] = []
    for cell, (clat, clon), dist in zip(cells, centers, nearest_dists):
        rows.append({
            "h3_index": cell,
            "resolution": resolution,
            "center_lat": clat,
            "center_lng": clon,
            "station_count": station_counts.get(cell, 0),
            "nearest_station_dist_km": round(float(dist), 3),
            "is_coverage_gap": bool(dist > gap_threshold_km),
            "highway_overlap": bool(compute_geo_features(clat, clon)),
        })

    n_gaps = sum(1 for r in rows if r["is_coverage_gap"])
    log.info(
        "Resolution %d: %d/%d hexes flagged as coverage gaps (>%.0f km to nearest station)",
        resolution, n_gaps, len(rows), gap_threshold_km,
    )
    return rows
