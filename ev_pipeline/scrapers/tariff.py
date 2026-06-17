# scrapers/tariff.py
"""
TGSPDCL Retail Supply Tariff Schedule — FY 2025-26
Effective: 01-May-2025 to 31-Mar-2026
Source: TGERC Retail Supply Tariff Order FY 2025-26
        https://tgsouthernpower.org/resources/PDF/Tariffs/63tarifffile.pdf

Used to back-calculate kWh consumption from the gross bill amount scraped
from the TGSPDCL portal, which only exposes financial totals (not meter readings).

Limitations:
- Fuel Surcharge Adjustment (FSA) is not included — it varies quarterly and is
  not published on the portal. This introduces ~3–8% underestimate of actual units.
- Electricity Duty rate is 6% for all non-domestic LT categories.
- Capacitor surcharge (25% of billed amount if capacitors defunct) is not
  detectable from the portal; assumed absent.
"""

from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

# Standard electricity duty rate on energy charges for non-domestic LT (%)
ELECTRICITY_DUTY_RATE = 0.06

# Customer charge for LT-IX (Rs/month) — single-phase vs three-phase
CUSTOMER_CHARGE_1PH = 65
CUSTOMER_CHARGE_3PH = 120

# ── LT Tariff Table ────────────────────────────────────────────────────────────
# Structure per category:
#   fixed_rs_per_kw   — Rs/kW/month demand charge (0 if no demand charge)
#   energy_rs_per_unit — Rs/kWh (flat or per-slab base rate for back-calc)
#   customer_charge_1ph / customer_charge_3ph — Rs/month
#   billing_unit      — "kWh" or "kVAh"
#   ed_exempt         — True for categories exempt from electricity duty (LT-I domestic, LT-V agri)

LT_TARIFF = {
    "LT-I":   {"fixed_rs_per_kw": 10,  "energy_rs_per_unit": 5.10, "customer_charge_1ph": 40,  "customer_charge_3ph": 100, "billing_unit": "kWh", "ed_exempt": True},
    "LT-II":  {"fixed_rs_per_kw": 70,  "energy_rs_per_unit": 8.50, "customer_charge_1ph": 50,  "customer_charge_3ph": 105, "billing_unit": "kWh", "ed_exempt": False},
    "LT-III": {"fixed_rs_per_kw": 100, "energy_rs_per_unit": 7.70, "customer_charge_1ph": 100, "customer_charge_3ph": 350, "billing_unit": "kWh", "ed_exempt": False},
    "LT-IV":  {"fixed_rs_per_kw": 20,  "energy_rs_per_unit": 4.00, "customer_charge_1ph": 50,  "customer_charge_3ph": 50,  "billing_unit": "kWh", "ed_exempt": False},
    "LT-V":   {"fixed_rs_per_kw": 0,   "energy_rs_per_unit": 0.00, "customer_charge_1ph": 30,  "customer_charge_3ph": 30,  "billing_unit": "kWh", "ed_exempt": True},
    "LT-VI":  {"fixed_rs_per_kw": 32,  "energy_rs_per_unit": 7.10, "customer_charge_1ph": 120, "customer_charge_3ph": 120, "billing_unit": "kWh", "ed_exempt": False},
    "LT-VII": {"fixed_rs_per_kw": 21,  "energy_rs_per_unit": 8.30, "customer_charge_1ph": 50,  "customer_charge_3ph": 100, "billing_unit": "kWh", "ed_exempt": False},
    "LT-VIII":{"fixed_rs_per_kw": 21,  "energy_rs_per_unit": 12.0, "customer_charge_1ph": 100, "customer_charge_3ph": 100, "billing_unit": "kWh", "ed_exempt": False},
    # LT-IX: EV Charging Stations — no demand charge, flat ₹6/kWh
    "LT-IX":  {"fixed_rs_per_kw": 0,   "energy_rs_per_unit": 6.00, "customer_charge_1ph": 65,  "customer_charge_3ph": 120, "billing_unit": "kWh", "ed_exempt": False},
}

# All stations in this pipeline are EV charging stations
DEFAULT_TARIFF_CATEGORY = "LT-IX"


def get_tariff(category: str) -> Optional[dict]:
    """Return tariff entry for a category string. Accepts 'LT-IX', 'LTIX', '9', etc."""
    key = category.upper().strip().replace(" ", "-")
    if not key.startswith("LT-"):
        key = f"LT-{key}"
    return LT_TARIFF.get(key)


def estimate_kwh_units(
    demand_rs: float,
    tariff_category: str = DEFAULT_TARIFF_CATEGORY,
    meter_phase: int = 3,
    contracted_load_kw: float = 0.0,
) -> Optional[float]:
    """
    Estimate kWh consumption from the gross bill amount shown on the TGSPDCL portal.

    The portal shows only the total payable (energy + customer charge + ED).
    FSA is not included in this estimate; actual units may be ~3–8% lower.

    Returns None if the tariff category is unknown or the net energy amount is ≤ 0.
    """
    entry = get_tariff(tariff_category)
    if not entry:
        return None

    customer_charge = entry["customer_charge_3ph"] if meter_phase != 1 else entry["customer_charge_1ph"]
    fixed_charges = entry["fixed_rs_per_kw"] * (contracted_load_kw or 0.0)
    energy_rate = entry["energy_rs_per_unit"]

    if energy_rate <= 0:
        return None

    ed_rate = 0.0 if entry["ed_exempt"] else ELECTRICITY_DUTY_RATE
    net_energy_amount = demand_rs - customer_charge - fixed_charges
    if net_energy_amount <= 0:
        return None

    return round(net_energy_amount / (energy_rate * (1.0 + ed_rate)), 2)


def get_fixed_charges(
    tariff_category: str = DEFAULT_TARIFF_CATEGORY,
    contracted_load_kw: float = 0.0,
) -> float:
    """Return the fixed/demand charges component in Rs for the given contracted load."""
    entry = get_tariff(tariff_category)
    if not entry:
        return 0.0
    return round(entry["fixed_rs_per_kw"] * (contracted_load_kw or 0.0), 2)


def get_customer_charge(
    tariff_category: str = DEFAULT_TARIFF_CATEGORY,
    meter_phase: int = 3,
) -> float:
    """Return the monthly customer charge in Rs."""
    entry = get_tariff(tariff_category)
    if not entry:
        return 0.0
    return float(entry["customer_charge_3ph"] if meter_phase != 1 else entry["customer_charge_1ph"])
