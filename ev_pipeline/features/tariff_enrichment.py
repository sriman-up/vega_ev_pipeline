# features/tariff_enrichment.py
"""
CPO (Charge Point Operator) detection and tariff estimation.

Since no public API provides per-station EV charging tariffs, this module:
  1. Detects the likely CPO from the station's `places_name` (Google Places)
  2. Looks up an estimated AC/DC tariff (Rs/kWh) for that CPO from a static
     reference table (cpo_tariffs)
  3. Picks AC vs DC rate based on the station's fastest installed charger
  4. Returns `cpo` and `tariff_inr_per_kwh` fields ready to merge into the
     stations record

IMPORTANT — accuracy caveat:
  These are published *average* market rates per operator (see cpo_tariffs
  source_note / last_updated), NOT station-specific or time-of-day pricing.
  Treat tariff_inr_per_kwh as a rough proxy for relative comparison and
  revenue estimation, not ground truth. Refresh CPO_TARIFFS periodically.

Usage:
    from features.tariff_enrichment import enrich_tariff
    updates = enrich_tariff(station_dict)
    station_dict.update(updates)

    # Then estimate EV-driver-facing revenue (distinct from monthly_bills.collection_rs,
    # which is the DISCOM electricity cost paid BY the station owner):
    estimated_revenue_rs = monthly_kwh_units * station_dict["tariff_inr_per_kwh"]
"""

import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CPO detection patterns
# ─────────────────────────────────────────────────────────────────────────────
# Matched (case-insensitive) against `places_name`. Order matters slightly for
# overlapping brand names (e.g. "Tata Power EZ" vs generic "Tata"), so more
# specific patterns are listed first within each CPO's list.

CPO_PATTERNS = {
    "tata_power": ["tata power", "ez charge", "plugo", "tata ev"],
    "statiq":     ["statiq"],
    "chargezone": ["chargezone", "charge zone"],
    "ather":      ["ather grid", "ather energy", "ather"],
    "bses":       ["bses"],
    "magenta":    ["magenta", "chargegrid"],
    "exicom":     ["exicom"],
    "jio_bp":     ["jio-bp", "jio bp", "jiobp"],
    "relux":      ["relux"],
    "kazam":      ["kazam"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Static CPO tariff reference table (Rs/kWh)
# ─────────────────────────────────────────────────────────────────────────────
# AC = slower chargers (typically <60kW, e.g. 7.4kW/15kW/30kW AC units)
# DC = fast chargers (typically >=60kW DC fast charging)
#
# Sources are aggregated published averages as of the date in last_updated.
# These are estimates — refresh from operator apps/websites periodically.

CPO_TARIFFS: Dict[str, Dict[str, Any]] = {
    "tata_power": {
        "tariff_ac_inr_kwh": 14.00,
        "tariff_dc_inr_kwh": 17.00,
        "source_note": "GreenTax.in 2026 guide — Tata Power ~Rs 17/kWh",
        "last_updated": "2026-06-13",
    },
    "chargezone": {
        "tariff_ac_inr_kwh": 18.00,
        "tariff_dc_inr_kwh": 22.50,
        "source_note": "ChargeZone official (Rs 18-25/unit DC) + GreenTax",
        "last_updated": "2026-06-13",
    },
    "statiq": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 20.00,
        "source_note": "Multi-network aggregator, estimated mid-range",
        "last_updated": "2026-06-13",
    },
    "ather": {
        "tariff_ac_inr_kwh": 0.00,
        "tariff_dc_inr_kwh": 0.00,
        "source_note": "Ather Grid — free for Ather vehicle owners",
        "last_updated": "2026-06-13",
    },
    "bses": {
        "tariff_ac_inr_kwh": 14.00,
        "tariff_dc_inr_kwh": 18.00,
        "source_note": "Delhi DISCOM-linked EV tariff, estimated",
        "last_updated": "2026-06-13",
    },
    "magenta": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 21.00,
        "source_note": "ChargeGrid network, estimated from market range",
        "last_updated": "2026-06-13",
    },
    "exicom": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 20.00,
        "source_note": "Hardware vendor / operator, estimated mid-range",
        "last_updated": "2026-06-13",
    },
    "jio_bp": {
        "tariff_ac_inr_kwh": 14.00,
        "tariff_dc_inr_kwh": 19.00,
        "source_note": "Jio-bp pulse network, estimated mid-range",
        "last_updated": "2026-06-13",
    },
    "relux": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 20.00,
        "source_note": "Estimated India avg fallback (no specific data)",
        "last_updated": "2026-06-13",
    },
    "kazam": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 20.00,
        "source_note": "Estimated India avg fallback (no specific data)",
        "last_updated": "2026-06-13",
    },
    "unknown": {
        "tariff_ac_inr_kwh": 15.00,
        "tariff_dc_inr_kwh": 21.00,
        "source_note": "Fallback — India public charging avg (Pulse Energy / EVCommunity 2026)",
        "last_updated": "2026-06-13",
    },
}


# Minimum kW among "fast" / DC chargers — used to decide AC vs DC tariff
DC_FAST_THRESHOLD_KW = 60


# ─────────────────────────────────────────────────────────────────────────────
# CPO detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_cpo(places_name: Optional[str]) -> str:
    """
    Best-effort CPO detection from the Google Places display name.
    Returns a key into CPO_TARIFFS, or 'unknown' if no pattern matches
    or places_name is empty.
    """
    if not places_name:
        return "unknown"

    name = places_name.lower()
    for cpo, patterns in CPO_PATTERNS.items():
        for pattern in patterns:
            if pattern in name:
                return cpo
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Charger speed classification
# ─────────────────────────────────────────────────────────────────────────────

def _has_fast_charger(station: Dict[str, Any]) -> bool:
    """
    Return True if the station has any charger >= DC_FAST_THRESHOLD_KW.
    Checks the fixed charger_*kw_count columns plus charger_other_json
    (e.g. {"50kw": 2, "240kw": 1}).
    """
    fast_count = (
        (station.get("charger_60kw_count") or 0)
        + (station.get("charger_120kw_count") or 0)
        + (station.get("charger_150kw_count") or 0)
    )
    if fast_count > 0:
        return True

    other = station.get("charger_other_json") or {}
    if isinstance(other, dict):
        for key, count in other.items():
            if not count:
                continue
            m = re.search(r"(\d+)", str(key))
            if m and int(m.group(1)) >= DC_FAST_THRESHOLD_KW:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_tariff(station: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine CPO and an estimated tariff (Rs/kWh) for a station.

    Args:
        station: station dict — must include 'places_name' for CPO detection
                 and charger_*kw_count / charger_other_json for AC/DC selection.

    Returns:
        Dict with keys:
            cpo                  — detected CPO key (see CPO_PATTERNS / CPO_TARIFFS)
            tariff_inr_per_kwh   — estimated Rs/kWh rate (AC or DC, based on chargers)
            tariff_basis         — 'AC' or 'DC', indicating which rate was used
            tariff_source_note   — provenance string for the rate used
            tariff_last_updated  — date string for the rate used
    """
    cpo = detect_cpo(station.get("places_name"))
    tariff_row = CPO_TARIFFS.get(cpo, CPO_TARIFFS["unknown"])

    is_dc = _has_fast_charger(station)
    basis = "DC" if is_dc else "AC"
    rate = tariff_row["tariff_dc_inr_kwh"] if is_dc else tariff_row["tariff_ac_inr_kwh"]

    updates = {
        "cpo": cpo,
        "tariff_inr_per_kwh": rate,
        "tariff_basis": basis,
        "tariff_source_note": tariff_row["source_note"],
        "tariff_last_updated": tariff_row["last_updated"],
    }

    log.info(
        "Tariff enrichment for SCNo %s: places_name=%r -> cpo=%s basis=%s rate=Rs%.2f/kWh",
        station.get("unique_scno"), station.get("places_name"), cpo, basis, rate,
    )
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Revenue estimation helper
# ─────────────────────────────────────────────────────────────────────────────

def estimate_revenue_rs(kwh_units: Optional[float], tariff_inr_per_kwh: Optional[float]) -> Optional[float]:
    """
    Estimate EV-driver-facing revenue for a given month's consumption.

    NOTE: This is distinct from monthly_bills.collection_rs, which is the
    DISCOM electricity cost paid BY the station owner. estimate_revenue_rs
    represents the (estimated) amount the CPO/station owner charges EV
    drivers — the difference is the owner's gross margin.
    """
    if kwh_units is None or tariff_inr_per_kwh is None:
        return None
    return round(float(kwh_units) * float(tariff_inr_per_kwh), 2)