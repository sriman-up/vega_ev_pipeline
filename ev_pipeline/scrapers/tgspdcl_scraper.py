# scrapers/tgspdcl_scraper.py
"""
Monthly bill scraper for TGSPDCL billing portal.

POSTs to https://tgsouthernpower.org/paybillonline with the Unique SCNo.
Unlike the /billinginfo endpoint, this page reports billed Units directly,
so kWh consumption no longer needs to be back-calculated from the tariff —
the tariff module is now only used as a fallback for the rare case where
the portal omits the Units field, and to fill in the (always-zero) LT-IX
fixed/demand charge component.

Usage:
    from scrapers.tgspdcl_scraper import fetch_latest_bill
    bill = fetch_latest_bill("114313853", tariff_category="LT-IX", meter_phase=3)
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

from ..config.settings import (
    SCRAPER_RETRY_COUNT,
    SCRAPER_TIMEOUT_S,
    SCRAPER_USER_AGENT,
    TGSPDCL_BILL_URL,
)
from .tariff import (
    DEFAULT_TARIFF_CATEGORY,
    estimate_kwh_units,
    get_fixed_charges,
)

log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": SCRAPER_USER_AGENT})


# ─────────────────────────────────────────────────────────────────────────────
# Core fetch
# ─────────────────────────────────────────────────────────────────────────────

def _post_bill_enquiry(unique_scno: str) -> Optional[str]:
    """POST to TGSPDCL paybillonline endpoint. Returns raw HTML or None on failure."""
    payload = {
        "ukscno":   unique_scno,
        "row1":     "1",
        "ukscno1":  "",
        "cname1":   "",
        "billamt1": "",
        "actamt1":  "",
        "arrears1": "",
        "totamt1":  "",
    }
    for attempt in range(1, SCRAPER_RETRY_COUNT + 1):
        try:
            resp = SESSION.post(
                TGSPDCL_BILL_URL,
                data=payload,
                timeout=SCRAPER_TIMEOUT_S,
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed for SCNo %s: %s", attempt, SCRAPER_RETRY_COUNT, unique_scno, e)
            if attempt < SCRAPER_RETRY_COUNT:
                time.sleep(2 ** attempt)
    return None


def _to_f(s: Any) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_iso_date(raw: str) -> Optional[str]:
    """Convert "05-Jun-26" → "2026-06-05" (ISO date string)."""
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    m = re.match(r"(\d{2})-([A-Za-z]{3})-(\d{2,4})", raw.strip())
    if not m:
        return None
    day, mon_raw, yr = m.groups()
    mon = months.get(mon_raw.lower())
    if not mon:
        return None
    yr = f"20{yr}" if len(yr) == 2 else yr
    return f"{yr}-{mon}-{day}"


def _parse_bill_month(raw: str) -> Optional[str]:
    """1st-of-month ISO date from a "DD-Mon-YY" or "Mon/YY" string."""
    iso = _to_iso_date(raw)
    if iso:
        return f"{iso[:7]}-01"
    m = re.match(r"([A-Za-z]{3})/(\d{2,4})", raw.strip())
    if m:
        months = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        mon = months.get(m.group(1).lower())
        yr = m.group(2)
        yr = f"20{yr}" if len(yr) == 2 else yr
        if mon:
            return f"{yr}-{mon}-01"
    return None


def _parse_bill_html(html: str, unique_scno: str) -> Optional[Dict[str, Any]]:
    """
    Parse the TGSPDCL paybillonline response.

    The "Bill / Payment Details" panel is a flat th/td table (one key per row),
    unlike the old /billinginfo page — no repeated keys across sections, so a
    single flat key→value map is enough. Relevant rows:
        Units                         → kwh_units   (reported directly — no tariff guesswork)
        Bill Date / Due Date          → bill_date, bill_month (due date is not stored)
        Current Month Bill            → demand_rs
        ACD Amount                    → acd_amount (netted into collection_rs below)
        Arrears                       → arrears_rs
        Total Amount to be Paid       → outstanding balance (0 if bill already paid)

    Portal layout as observed June 2026.
    """
    soup = BeautifulSoup(html, "html.parser")

    kv: Dict[str, str] = {}
    for table in soup.find_all("table", class_="ctable"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) != 2:
                continue
            key = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True)
            if key and val:
                kv[key] = val

    if not kv:
        log.warning("No bill/payment details table found for SCNo %s", unique_scno)
        return None

    def kv_get(*substrings: str) -> Optional[str]:
        for key, val in kv.items():
            if all(s in key for s in substrings):
                return val
        return None

    bill: Dict[str, Any] = {
        "unique_scno": unique_scno,
        "source":      "tgspdcl_scrape",
        "scraped_at":  datetime.utcnow().isoformat(),
        "kwh_units":   _to_f(kv_get("units")),
        "demand_rs":   _to_f(kv_get("current month bill")),
        "arrears_rs":  _to_f(kv_get("arrears")),
    }
    if bill.get("kwh_units") is not None:
        bill["billed_units"] = bill["kwh_units"]   # LT-IX billed on kWh basis

    # "Bill Date / Due Date" combined cell: "05-Jun-26 /  19-Jun-26"
    bill_date_due = kv_get("bill date")
    bill_date_raw = None
    if bill_date_due:
        parts = [p.strip() for p in bill_date_due.split("/") if p.strip()]
        if parts:
            bill_date_raw = parts[0]

    if bill_date_raw:
        iso_date = _to_iso_date(bill_date_raw)
        if iso_date:
            bill["bill_date"] = iso_date
        iso_month = _parse_bill_month(bill_date_raw)
        if iso_month:
            bill["bill_month"] = iso_month

    if "bill_month" not in bill:
        now = datetime.utcnow()
        bill["bill_month"] = f"{now.year}-{now.month:02d}-01"
        log.warning("Could not parse bill month for SCNo %s; defaulting to current month", unique_scno)

    # collection_rs (amount already paid this cycle) is not shown directly here,
    # but can be derived: gross payable minus what's still outstanding.
    #   gross_payable = demand_rs + arrears_rs - acd_amount
    #   collection_rs = gross_payable - total_amount_to_be_paid
    # When the bill is fully paid, "Total Amount to be Paid" is 0, so
    # collection_rs == gross_payable (matches /billinginfo's "Total Amount Paid").
    acd_amount = _to_f(kv_get("acd amount")) or 0.0
    total_to_pay = _to_f(kv_get("total amount"))
    if bill.get("demand_rs") is not None and total_to_pay is not None:
        gross_payable = bill["demand_rs"] + (bill.get("arrears_rs") or 0.0) - acd_amount
        bill["collection_rs"] = round(gross_payable - total_to_pay, 2)

    # Drop None values
    return {k: v for k, v in bill.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_bill(
    unique_scno: str,
    tariff_category: str = DEFAULT_TARIFF_CATEGORY,
    meter_phase: int = 3,
    contracted_load_kw: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch and parse the latest bill for a given Unique SCNo.

    kWh units come directly from the portal's "Units" field. The tariff
    schedule (tariff.py) is only used to: (a) fill the fixed/demand charge
    component, which is a known flat fact per category (₹0 for LT-IX), and
    (b) as a fallback estimate if the portal ever omits Units.

    Args:
        unique_scno:        TGSPDCL Unique Service Connection Number.
        tariff_category:    Tariff category string e.g. "LT-IX" (default for EV stations).
        meter_phase:        1 or 3 — used only by the fallback estimate.
        contracted_load_kw: Contracted load in kW — used for fixed charge lookup.

    Returns a dict ready for db.upsert_monthly_bill(), or None on failure.
    """
    log.info("Fetching bill for SCNo %s", unique_scno)
    html = _post_bill_enquiry(unique_scno)
    if not html:
        log.error("Failed to retrieve bill page for SCNo %s", unique_scno)
        return None

    bill = _parse_bill_html(html, unique_scno)
    if not bill:
        return None

    # Fixed/demand charge component is a known flat fact per category (₹0 for LT-IX)
    fixed = get_fixed_charges(tariff_category, contracted_load_kw)
    if fixed > 0:
        bill["fixed_charges_rs"] = fixed

    # Fallback only — portal normally reports Units directly
    if "kwh_units" not in bill and bill.get("demand_rs") is not None:
        kwh = estimate_kwh_units(bill["demand_rs"], tariff_category, meter_phase, contracted_load_kw)
        if kwh is not None and kwh > 0:
            bill["kwh_units"]    = kwh
            bill["billed_units"] = kwh
            log.warning("Units not reported by portal for SCNo %s; using tariff estimate %.2f kWh", unique_scno, kwh)

    log.info(
        "Scraped bill for SCNo %s  month=%s  demand_rs=%s  kwh_units=%s",
        unique_scno, bill.get("bill_month"), bill.get("demand_rs"), bill.get("kwh_units"),
    )
    return bill


def fetch_bills_for_all_stations(stations: list) -> Dict[str, Optional[Dict]]:
    """
    Batch fetch for a list of station dicts (each must have 'unique_scno').
    Passes tariff context from each station dict if available.
    Returns {unique_scno: bill_dict or None}.
    """
    results = {}
    for i, station in enumerate(stations):
        scno = station["unique_scno"]
        results[scno] = fetch_latest_bill(
            scno,
            tariff_category=station.get("tariff_category", DEFAULT_TARIFF_CATEGORY),
            meter_phase=int(station.get("meter_phase") or 3),
            contracted_load_kw=float(station.get("contracted_load_kva") or 0.0),
        )
        if i < len(stations) - 1:
            time.sleep(1.5)  # ~40 req/min max
    return results
