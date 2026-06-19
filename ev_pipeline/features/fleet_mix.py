# features/fleet_mix.py
"""
Static fleet-mix reference data, seeded into the zone_fleet_mix table by
zone_band (see features.city_zones) rather than by individual station.

IMPORTANT — accuracy caveat:
  VEHICLE_SPECS and ZONE_VEHICLE_SHARE_PCT are illustrative defaults shaped
  after VAHAN's 2W/4W/bus vehicle-category split, NOT actual VAHAN
  registration counts for Telangana. Replace fleet_share_pct with real
  RTO/district-level VAHAN EV registration data (mapped to zone_band) when
  available — see SOURCE_NOTE, which is stored alongside each row so the
  provenance travels with the data.

Usage:
    from features.fleet_mix import seed_zone_fleet_mix, expected_kwh_per_session
    seed_zone_fleet_mix()                     # one-time / idempotent DB seed
    avg_kwh = expected_kwh_per_session("highway")
"""

import logging
from typing import Any, Dict, List, Optional

from ev_pipeline.db.db_manager import get_zone_fleet_mix, upsert_zone_fleet_mix

log = logging.getLogger(__name__)

SOURCE_NOTE = (
    "Illustrative defaults shaped after VAHAN's 2W/4W/bus category split — "
    "not actual Telangana VAHAN registration counts. Replace fleet_share_pct "
    "with real RTO-level VAHAN data per zone when available."
)
LAST_UPDATED = "2026-06-18"

# Representative Indian EV segment specs (used identically across zones —
# only the fleet_share_pct mix varies by zone_band).
VEHICLE_SPECS: Dict[str, Dict[str, float]] = {
    "2W":  {"range_km": 85,  "battery_kwh": 2.5,   "charge_rate_kw": 2.2,
            "soc_start_pct": 20, "soc_end_pct": 90},
    "4W":  {"range_km": 300, "battery_kwh": 40.0,  "charge_rate_kw": 30.0,
            "soc_start_pct": 20, "soc_end_pct": 80},
    "bus": {"range_km": 200, "battery_kwh": 250.0, "charge_rate_kw": 120.0,
            "soc_start_pct": 30, "soc_end_pct": 90},
}

# Fleet share (%) by zone_band — each zone's shares sum to ~100.
# 2W dominates dense urban/periurban short-hop charging; 4W and buses pick up
# share on highway/remote corridors where intercity and freight/transit
# routes run.
ZONE_VEHICLE_SHARE_PCT: Dict[str, Dict[str, float]] = {
    "core":      {"2W": 70, "4W": 27, "bus": 3},
    "periurban": {"2W": 65, "4W": 30, "bus": 5},
    "highway":   {"2W": 20, "4W": 65, "bus": 15},
    "remote":    {"2W": 30, "4W": 55, "bus": 15},
}


def _avg_session_kwh(vehicle_type: str) -> float:
    """Expected kWh for one charging session: battery_kwh x SoC swing."""
    spec = VEHICLE_SPECS[vehicle_type]
    swing_pct = spec["soc_end_pct"] - spec["soc_start_pct"]
    return round(spec["battery_kwh"] * swing_pct / 100.0, 3)


def build_seed_rows() -> List[Dict[str, Any]]:
    """Build the full zone_fleet_mix seed rows from the static dicts above."""
    rows = []
    for zone_band, shares in ZONE_VEHICLE_SHARE_PCT.items():
        for vehicle_type, share_pct in shares.items():
            spec = VEHICLE_SPECS[vehicle_type]
            rows.append({
                "zone_band":            zone_band,
                "vehicle_type":         vehicle_type,
                "fleet_share_pct":      share_pct,
                "range_km":             spec["range_km"],
                "battery_kwh":          spec["battery_kwh"],
                "charge_rate_kw":       spec["charge_rate_kw"],
                "typical_soc_start_pct": spec["soc_start_pct"],
                "typical_soc_end_pct":   spec["soc_end_pct"],
                "avg_session_kwh":      _avg_session_kwh(vehicle_type),
                "source_note":          SOURCE_NOTE,
                "last_updated":         LAST_UPDATED,
            })
    return rows


def seed_zone_fleet_mix() -> int:
    """Idempotent upsert of the static zone_fleet_mix reference table."""
    rows = build_seed_rows()
    for row in rows:
        upsert_zone_fleet_mix(row)
    log.info(
        "Seeded zone_fleet_mix: %d rows (%d zones x %d vehicle types)",
        len(rows), len(ZONE_VEHICLE_SHARE_PCT), len(VEHICLE_SPECS),
    )
    return len(rows)


def expected_kwh_per_session(zone_band: str) -> Optional[float]:
    """
    Fleet-mix-weighted expected kWh per charging session for a zone band.

    Reads from the zone_fleet_mix table (not the static dicts directly) so
    the result reflects whatever fleet_share_pct values are currently seeded,
    including future updates from real VAHAN data.
    """
    rows = get_zone_fleet_mix(zone_band)
    if not rows:
        return None

    total_share = sum(r.get("fleet_share_pct") or 0 for r in rows)
    if not total_share:
        return None

    weighted = sum(
        (r.get("fleet_share_pct") or 0) * (r.get("avg_session_kwh") or 0)
        for r in rows
    )
    return round(weighted / total_share, 3)
