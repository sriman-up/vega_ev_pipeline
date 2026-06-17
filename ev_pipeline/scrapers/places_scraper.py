# scrapers/places_scraper.py
"""
Google Places API (New) enrichment for EV stations.

Endpoints used:
  POST /v1/places:searchNearby  — find station at known coords; competition queries
  GET  /v1/places/{name}        — place details (types, rating, evChargeOptions)

Key design:
  - If station dict already has lat/lon (from PDF maps URL), use those for
    a tight 100m nearby search to find the correct Places record rather than
    doing a text search which can return the wrong result.
  - Place resource names in Places API (New) look like "places/ChIJ..." — we
    store the full resource name as places_id and use it directly in the
    details URL (GET /v1/places/ChIJ...).
  - evChargeOptions is an Atmosphere-tier field (~$0.03/call extra).
    Only fetched when fetch_charger_info=True (seed runs), not monthly refresh.
"""

import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from ev_pipeline.config.settings import GOOGLE_PLACES_API_KEY, PLACES_NEARBY_RADIUS_M

log = logging.getLogger(__name__)

_BASE = "https://places.googleapis.com/v1"

# location_type priority — first matching Places type wins
_TYPE_PRIORITY = [
    ("gas_station",                    "gas_station"),
    ("shopping_mall",                  "shopping_mall"),
    ("supermarket",                    "supermarket"),
    ("lodging",                        "hotel"),
    ("highway_rest_area",              "rest_area"),
    ("restaurant",                     "restaurant"),
    ("parking",                        "parking"),
    ("tourist_attraction",             "tourist_attraction"),
    ("electric_vehicle_charging_station", "ev_charging"),
]

FOOD_TYPES = {
    "restaurant", "cafe", "food", "bakery", "bar",
    "meal_delivery", "meal_takeaway", "fast_food_restaurant",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_key() -> str:
    if not GOOGLE_PLACES_API_KEY or GOOGLE_PLACES_API_KEY == "YOUR_KEY_HERE":
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set in .env")
    return GOOGLE_PLACES_API_KEY


def _get_headers(field_mask: str) -> Dict[str, str]:
    return {
        "X-Goog-Api-Key":  _api_key(),
        "X-Goog-FieldMask": field_mask,
    }


def _post_headers(field_mask: str) -> Dict[str, str]:
    return {
        "Content-Type":    "application/json",
        "X-Goog-Api-Key":  _api_key(),
        "X-Goog-FieldMask": field_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nearby Search (New API — POST /v1/places:searchNearby)
# ─────────────────────────────────────────────────────────────────────────────

def _nearby_search(
    lat: float,
    lon: float,
    radius_m: int,
    included_types: Optional[List[str]] = None,
    max_results: int = 20,
) -> List[Dict]:
    """
    Returns a list of place dicts from Places API (New) Nearby Search.
    Each dict has: name (resource name), displayName.text, types, location.
    """
    body: Dict[str, Any] = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius_m),
            }
        },
        "maxResultCount": min(max_results, 20),
    }
    if included_types:
        body["includedTypes"] = included_types

    # Field mask uses "places.<field>" prefix for nearby search
    field_mask = "places.name,places.displayName,places.types,places.location"

    try:
        r = requests.post(
            f"{_BASE}/places:searchNearby",
            json=body,
            headers=_post_headers(field_mask),
            timeout=15,
        )
        if not r.ok:
            _log_api_error(r, "Nearby Search")
            return []
        return r.json().get("places", [])
    except Exception as e:
        log.error("Nearby search failed at (%.5f, %.5f): %s", lat, lon, e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Place Details (New API — GET /v1/places/{name})
# ─────────────────────────────────────────────────────────────────────────────

def _place_details(
    resource_name: str,
    include_ev_charge: bool = False,
) -> Optional[Dict]:
    """
    Fetch place details from Places API (New).

    resource_name: full resource name e.g. "places/ChIJ..."
    Basic-tier fields (no extra cost):
        displayName, types, formattedAddress, location, rating, userRatingCount
    Atmosphere-tier field (charged):
        evChargeOptions  (~$0.03/call extra — only when include_ev_charge=True)
    """
    # Details field mask uses NO "places." prefix (unlike nearby search)
    fields = "displayName,types,formattedAddress,location,rating,userRatingCount"
    if include_ev_charge:
        fields += ",evChargeOptions"

    try:
        r = requests.get(
            f"{_BASE}/{resource_name}",
            headers=_get_headers(fields),
            timeout=15,
        )
        if not r.ok:
            _log_api_error(r, f"Place Details {resource_name}")
            return None
        return r.json()
    except Exception as e:
        log.error("Place details failed for %s: %s", resource_name, e)
        return None


def _log_api_error(response, context: str):
    try:
        err = response.json()
        msg = err.get("error", {}).get("message") or str(err)[:200]
    except Exception:
        msg = response.text[:200]
    log.error("Places API error (%s) [%s %s]: %s",
              context, response.status_code, response.reason, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Geocode: find the correct Places record for a station
# ─────────────────────────────────────────────────────────────────────────────

def _find_place_at_location(lat: float, lon: float) -> Optional[Dict]:
    """
    Nearby Search with tight radius to find the EV station Places record at
    known coordinates (from the PDF maps URL).

    Tries:
      1. 100 m radius, type=electric_vehicle_charging_station
      2. 300 m radius, any type (catches stations registered as parking/fuel)
    Returns the closest result by haversine distance, or None.
    """
    for radius, types in [
        (100,  ["electric_vehicle_charging_station"]),
        (300,  None),
    ]:
        results = _nearby_search(lat, lon, radius_m=radius, included_types=types)
        if results:
            return min(results, key=lambda p: _dist_to_place(lat, lon, p))
    return None


def _dist_to_place(lat: float, lon: float, place: Dict) -> float:
    loc = place.get("location") or {}
    return _haversine_m(lat, lon, loc.get("latitude", 999.0), loc.get("longitude", 999.0))


def geocode_station(station: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
    """
    Returns (latitude, longitude, resource_name).

    Priority:
      1. lat/lon already in station dict (from PDF maps URL) — find Places
         record at those exact coordinates.
      2. No coords — fall back to legacy Text Search by name/address.
    """
    lat = station.get("latitude")
    lon = station.get("longitude")

    if lat and lon:
        place = _find_place_at_location(float(lat), float(lon))
        resource_name = place.get("name", "") if place else ""
        if not resource_name:
            log.info(
                "Station %s not found in Places DB at coords (%.5f, %.5f) — "
                "will proceed without places_id",
                station.get("unique_scno"), lat, lon,
            )
        return float(lat), float(lon), resource_name

    # Fallback: legacy text search
    result = _legacy_text_search(
        f"{station.get('consumer_name', '')} EV charging {station.get('address', '')}"
    )
    if result:
        loc = result["geometry"]["location"]
        return loc["lat"], loc["lng"], result.get("place_id", "")

    log.warning("Could not geocode station %s", station.get("unique_scno"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_station(
    station: Dict[str, Any],
    fetch_charger_info: bool = False,
) -> Dict[str, Any]:
    """
    Full enrichment for one station.

    Args:
        station:            Station dict. Must have lat/lon or consumer_name/address.
        fetch_charger_info: Fetch evChargeOptions (Atmosphere tier, ~$0.03/call).
                            True for seed runs; False for monthly refresh.

    Returns dict of fields to merge into the station record.
    """
    updates: Dict[str, Any] = {}

    # ── 1. Geocode ────────────────────────────────────────────────────────────
    geo = geocode_station(station)
    if not geo:
        log.warning("Skipping enrichment for %s - no geocode", station.get("unique_scno"))
        return updates

    lat, lon, resource_name = geo
    updates["latitude"]  = lat
    updates["longitude"] = lon

    # ── 2. Place details ──────────────────────────────────────────────────────
    if resource_name:
        updates["places_id"] = resource_name
        details = _place_details(resource_name, include_ev_charge=fetch_charger_info)
        if details:
            updates["places_name"] = (details.get("displayName") or {}).get("text")
            updates["places_rating"] = details.get("rating")
            updates["places_user_ratings_total"] = details.get("userRatingCount")

            types = details.get("types") or []
            updates["location_type_raw"] = types
            updates["location_type"]     = _derive_location_type(types)
            updates["has_attached_restaurant"] = bool(
                FOOD_TYPES.intersection(set(types))
            )

            if fetch_charger_info:
                # evChargeOptions key in the response JSON
                ev_opts = details.get("evChargeOptions") or {}
                charger_feats = _parse_ev_charge_options(ev_opts)
                if charger_feats:
                    updates.update(charger_feats)
                    log.info(
                        "Charger info for SCNo %s: total=%s details=%s",
                        station.get("unique_scno"),
                        charger_feats.get("total_charger_count"),
                        {k: v for k, v in charger_feats.items()
                         if k.startswith("charger_") and k.endswith("_count")},
                    )
                else:
                    log.info(
                        "No evChargeOptions data in Places for SCNo %s "
                        "(station may not be indexed with charger data yet)",
                        station.get("unique_scno"),
                    )
    else:
        updates.setdefault("has_attached_restaurant", False)

    # ── 3. Nearby EV stations (competition) ──────────────────────────────────
    ev_results = _nearby_search(
        lat, lon,
        radius_m=PLACES_NEARBY_RADIUS_M,
        included_types=["electric_vehicle_charging_station"],
    )
    ev_others = [p for p in ev_results if p.get("name") != resource_name]
    updates["nearby_ev_stations_1km"] = len(ev_others)
    updates["_nearby_ev_names"] = [
        (p.get("displayName") or {}).get("text", "") for p in ev_others[:10]
    ]

    # ── 4. Nearby restaurants ─────────────────────────────────────────────────
    rest_results = _nearby_search(
        lat, lon,
        radius_m=PLACES_NEARBY_RADIUS_M,
        included_types=["restaurant", "cafe", "fast_food_restaurant"],
    )
    updates["nearby_restaurants_1km"] = len(rest_results)

    # Co-location: restaurant within 50 m = attached
    if not updates.get("has_attached_restaurant"):
        for p in rest_results:
            if _dist_to_place(lat, lon, p) < 50:
                updates["has_attached_restaurant"] = True
                break
    updates.setdefault("has_attached_restaurant", False)

    # ── 5. Nearby hotels ──────────────────────────────────────────────────────
    updates["nearby_hotels_1km"] = len(
        _nearby_search(lat, lon, radius_m=PLACES_NEARBY_RADIUS_M,
                       included_types=["lodging"])
    )

    # ── 6. Nearby petrol pumps ────────────────────────────────────────────────
    updates["nearby_petrol_pumps_1km"] = len(
        _nearby_search(lat, lon, radius_m=PLACES_NEARBY_RADIUS_M,
                       included_types=["gas_station"])
    )

    # ── 7. Nearby shopping ────────────────────────────────────────────────────
    updates["nearby_shopping_1km"] = len(
        _nearby_search(lat, lon, radius_m=PLACES_NEARBY_RADIUS_M,
                       included_types=["shopping_mall", "supermarket"])
    )

    updates["competition_last_updated"] = datetime.utcnow().isoformat()

    log.info(
        "Enriched SCNo %s: (%.5f, %.5f) ev=%d rest=%d hotels=%d petrol=%d "
        "shop=%d attached_rest=%s location_type=%s",
        station.get("unique_scno"), lat, lon,
        updates["nearby_ev_stations_1km"],
        updates["nearby_restaurants_1km"],
        updates["nearby_hotels_1km"],
        updates["nearby_petrol_pumps_1km"],
        updates["nearby_shopping_1km"],
        updates.get("has_attached_restaurant"),
        updates.get("location_type"),
    )
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Charger inventory parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ev_charge_options(ev_options: Dict) -> Dict[str, Any]:
    """
    Parse evChargeOptions from Places API (New) Place Details response.

    Response shape:
      {
        "connectorCount": 4,
        "connectorAggregation": [
          {
            "type": "EV_CONNECTOR_TYPE_CCS_COMBO_1",
            "maxChargeRateKw": 50.0,
            "count": 2,
            "availableCount": 1,
            "outOfServiceCount": 0
          },
          ...
        ]
      }
    """
    if not ev_options:
        return {}

    chargers: Dict[str, Any] = {}
    other: Dict[str, int] = {}
    total = 0

    aggregations = ev_options.get("connectorAggregation") or []
    for agg in aggregations:
        kw    = float(agg.get("maxChargeRateKw") or 0)
        count = int(agg.get("count") or 0)
        if kw <= 0 or count <= 0:
            continue
        total += count
        kw_int = int(round(kw))
        col = f"charger_{kw_int}kw_count"
        if col in ("charger_30kw_count", "charger_60kw_count",
                   "charger_120kw_count", "charger_150kw_count"):
            chargers[col] = chargers.get(col, 0) + count
        else:
            other[f"{kw_int}kw"] = other.get(f"{kw_int}kw", 0) + count

    # Also try the top-level connectorCount as a fallback for total
    if total == 0:
        top_level = int(ev_options.get("connectorCount") or 0)
        if top_level:
            total = top_level

    if other:
        chargers["charger_other_json"] = other
    if total:
        chargers["total_charger_count"] = total

    return chargers


# ─────────────────────────────────────────────────────────────────────────────
# location_type
# ─────────────────────────────────────────────────────────────────────────────

def _derive_location_type(types: List[str]) -> Optional[str]:
    types_set = set(types)
    for place_type, label in _TYPE_PRIORITY:
        if place_type in types_set:
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: extract charger info from a stored places_id
# ─────────────────────────────────────────────────────────────────────────────

def extract_charger_info_from_places(resource_name: str) -> Dict[str, Any]:
    """Fetch charger info for a known resource_name (e.g. backfill after seed)."""
    if not resource_name:
        return {}
    details = _place_details(resource_name, include_ev_charge=True)
    if not details:
        return {}
    return _parse_ev_charge_options(details.get("evChargeOptions") or {})


# ─────────────────────────────────────────────────────────────────────────────
# Legacy fallback (used when no lat/lon in station dict)
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_text_search(query: str) -> Optional[Dict]:
    """Legacy Places Text Search — only used when no lat/lon is available."""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    try:
        r = requests.get(
            url,
            params={"query": query, "key": _api_key()},
            timeout=10,
        )
        if not r.ok:
            return None
        results = r.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        log.error("Legacy text search failed for '%s': %s", query, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))