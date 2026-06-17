# scrapers/pdf_parser.py
"""
Parse TGSPDCL consumer detail PDFs and billing history PDFs.

Handles two layouts:
  - Consumer Details (single record per PDF)
  - Consumption/Billing/Collection/Arrears History — single-row layout
  - Tata-style history — dual-row layout (IR row + 9N/LT row per month)
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) else s


def _to_float(s: Optional[str]) -> Optional[float]:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_date(s: Optional[str]) -> Optional[str]:
    """Convert DD-MON-YY or DD-MON-YYYY to ISO YYYY-MM-DD."""
    if not s:
        return None
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    s = s.strip().upper()
    m = re.match(r"(\d{2})-([A-Z]{3})-(\d{2,4})", s)
    if not m:
        return None
    day, mon_str, yr = m.groups()
    mon = months.get(mon_str)
    if not mon:
        return None
    yr = f"20{yr}" if len(yr) == 2 else yr
    return f"{yr}-{mon}-{day}"


def _bill_month_to_iso(month_year: str) -> Optional[str]:
    """'Apr/2026' or 'Apr/26' -> '2026-04-01'."""
    months = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    m = re.match(r"([A-Za-z]{3})/(\d{2,4})", month_year.strip())
    if not m:
        return None
    mon_str, yr = m.groups()
    mon = months.get(mon_str.capitalize())
    if not mon:
        return None
    yr = f"20{yr}" if len(yr) == 2 else yr
    return f"{yr}-{mon}-01"


# ─────────────────────────────────────────────────────────────────────────────
# Maps URL lat/lon extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_maps_latlon(full_text: str, pdf_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract lat/lon from an embedded Google Maps URL in the PDF.

    TGSPDCL embeds a clickable pin link in Consumer Details PDFs, e.g.:
      https://www.google.com/maps/search/?api=1&query=17.0306283333333,78.48833166666667&t=k

    Strategy:
      1. Scan plain text extracted by pdfplumber (sometimes the URL is rendered as text).
      2. Scan PDF annotation/hyperlink objects (the clickable area pdfplumber exposes).
    """
    _PATTERNS = [
        r"query=(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+)",   # maps search URL
        r"@(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+),\d+z",   # maps/@lat,lon,zoom
        r"ll=(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+)",       # legacy maps URL
    ]

    def _try_patterns(text: str) -> Tuple[Optional[float], Optional[float]]:
        for pat in _PATTERNS:
            m = re.search(pat, text)
            if m:
                try:
                    lat, lon = float(m.group(1)), float(m.group(2))
                    # Basic sanity check for India bounding box
                    if 6.0 <= lat <= 37.0 and 68.0 <= lon <= 97.0:
                        return lat, lon
                except ValueError:
                    pass
        return None, None

    # 1. Plain text
    lat, lon = _try_patterns(full_text)
    if lat is not None:
        return lat, lon

    # 2. PDF annotations (hyperlinks)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for annot in (page.annots or []):
                    # pdfplumber returns annotation dicts; URI may be nested
                    uri = (
                        annot.get("uri")
                        or annot.get("URI")
                        or (annot.get("data") or {}).get("URI")
                        or (annot.get("data") or {}).get("uri")
                        or ""
                    )
                    if uri:
                        lat, lon = _try_patterns(str(uri))
                        if lat is not None:
                            return lat, lon
    except Exception as e:
        log.debug("Annotation extraction failed for %s: %s", pdf_path, e)

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Consumer Details PDF
# ─────────────────────────────────────────────────────────────────────────────

def parse_consumer_details(pdf_path: str) -> Dict[str, Any]:
    """
    Parse a TGSPDCL Consumer Details PDF.
    Returns a dict suitable for db.upsert_station().
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    text_lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_lines += (page.extract_text() or "").splitlines()

    full_text = "\n".join(text_lines)

    def find(pattern: str, default=None):
        m = re.search(pattern, full_text, re.IGNORECASE)
        return _clean(m.group(1)) if m else default

    station: Dict[str, Any] = {
        "unique_scno":         find(r"Unique SCNo\s+(\d+)"),
        "service_number":      find(r"Service Number\s+([\d\s]+)"),
        "category":            find(r"Category/SubCat\s+(\d+)"),
        "sub_category":        find(r"Category/SubCat\s+\d+\s*/\s*(\d+)"),
        "section_code_name":   find(r"Section Code - Name\s+([\w\-]+)"),
        "sub_division":        find(r"Sub Division\s+(\d+)"),
        "area_code":           find(r"Area Code\s+(\d+)"),
        "consumer_type":       find(r"Consumer / Service Type\s+([A-Z]+)"),
        "contracted_load_kva": _to_float(find(r"Contracted Load\s+([\d.]+)")),
        "connected_load_kva":  _to_float(find(r"Connected Load\s+([\d.]+)")),
        "meter_number":        find(r"Meter Number\s+(\d+)"),
        "meter_phase":         _to_int(find(r"Phase\s+(\d)")),
        "multiplying_factor":  _to_float(find(r"M\.Factor\s+([\d.]+)")),
        "security_deposit":    _to_float(find(r"Deposit Available\s+([\d.,]+)")),
        "supply_date":         _parse_date(find(r"Supply Date\s+(\d{2}-[A-Z]{3}-\d{2,4})")),
        "phone":               find(r"Phone\s+(\d+)"),
        "pin_code":            find(r"Pin Code\s+(\d+)"),
    }

    # ── Consumer name — take only the first non-empty word group after the label,
    #    stopping at any all-caps label word that follows (avoids bleeding into
    #    adjacent cells from multi-column PDF layout).
    name_m = re.search(r"Consumer Name\s+([A-Z][A-Z ]+?)(?:\n|  |\t|Area Code|Pole)", full_text)
    if name_m:
        station["consumer_name"] = _clean(name_m.group(1))

    # ── ERO / Circle ─────────────────────────────────────────────────────────
    ero_m = re.search(r"ERO\s*[:\s]+(\d+\s+\S+)", full_text)
    if ero_m:
        station["ero_name"] = _clean(ero_m.group(1))
    circle_m = re.search(r"Circle\s*[:\s]+([A-Z ]+?)(?:\s{2,}|ERO)", full_text)
    if circle_m:
        station["circle_name"] = _clean(circle_m.group(1))

    # ── Address — lines between "Address" label and "Group / Cycle" ──────────
    addr_m = re.search(r"Address\s+(.+?)(?:Group / Cycle|Consumer / Service)", full_text, re.DOTALL)
    if addr_m:
        station["address"] = re.sub(r"\s+", " ", addr_m.group(1)).strip()

    # ── Service number — strip spaces ─────────────────────────────────────────
    if station.get("service_number"):
        station["service_number"] = re.sub(r"\s+", "", station["service_number"])

    # ── Lat/lon from embedded Google Maps URL — always prefer this over Places ─
    lat, lon = _extract_maps_latlon(full_text, pdf_path)
    if lat is not None and lon is not None:
        station["latitude"]  = lat
        station["longitude"] = lon
        log.info(
            "Extracted lat/lon from embedded maps URL for SCNo %s: %.7f, %.7f",
            station.get("unique_scno"), lat, lon,
        )
    else:
        log.info(
            "No maps URL found in PDF for SCNo %s — lat/lon will come from Places API",
            station.get("unique_scno"),
        )

    log.info("Parsed consumer details for SCNo %s", station.get("unique_scno"))
    return {k: v for k, v in station.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Billing History — single-row layout
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4}$", re.I)

# Matches the IR (first) line of a dual-row entry extracted as plain text, e.g.:
#   "Apr/2026 01 / IR 9 39665 2905 17876.00 0.00 17876.00 0.00 0.00 0"
# Groups: month, status, cat, kwh_reading, kwh_units, demand, je_debit,
#         collection, je_credit, arrears, fixed_charges
# This mirrors the column layout that extract_table() returns so the same
# parsing loop can consume both table rows and text-fallback rows.
_IR_TEXT_LINE_RE = re.compile(
    r"^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4})"
    r"\s+(\d+(?:\s*/\s*\w+)?)"   # status: "01 / IR" or "00"
    r"\s+(\d+)"                    # category
    r"\s+([\d.]+)"                 # kwh closing reading
    r"\s+([\d.]+)"                 # kwh units
    r"\s+([\d.]+)"                 # demand (Rs.)
    r"\s+([\d.]+)"                 # JE debit (Rs.)
    r"\s+([\d.]+)"                 # collection (Rs.)
    r"\s+([\d.]+)"                 # JE credit (Rs.)
    r"\s+([\d.]+)"                 # arrears (Rs.)
    r"\s+([\d.]+)",                # fixed charges
    re.I,
)


def parse_billing_history_single(pdf_path: str, unique_scno: str) -> List[Dict[str, Any]]:
    """Parse standard single-row-per-month history PDF (Swathi layout)."""
    rows: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for row in table:
                if not row or not row[0]:
                    continue
                cell0 = str(row[0]).strip()
                if not _MONTH_RE.match(cell0):
                    continue
                try:
                    rec: Dict[str, Any] = {
                        "unique_scno":         unique_scno,
                        "bill_month":          _bill_month_to_iso(cell0),
                        "status":              _clean(row[1]) if len(row) > 1 else None,
                        "kwh_closing_reading": _to_float(row[2]) if len(row) > 2 else None,
                        "kwh_units":           _to_float(row[3]) if len(row) > 3 else None,
                        "demand_rs":           _to_float(row[4]) if len(row) > 4 else None,
                        "je_debit_rs":         _to_float(row[5]) if len(row) > 5 else None,
                        "collection_rs":       _to_float(row[6]) if len(row) > 6 else None,
                        "je_credit_rs":        _to_float(row[7]) if len(row) > 7 else None,
                        "arrears_rs":          _to_float(row[8]) if len(row) > 8 else None,
                        "source":              "pdf_import",
                    }
                    if rec["bill_month"]:
                        rows.append({k: v for k, v in rec.items() if v is not None})
                except Exception as e:
                    log.warning("Skipping row %s: %s", row, e)
    log.info("Parsed %d single-row history records for SCNo %s", len(rows), unique_scno)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Billing History — dual-row layout (Tata EV / HV)
# ─────────────────────────────────────────────────────────────────────────────

def parse_billing_history_dual(pdf_path: str, unique_scno: str) -> List[Dict[str, Any]]:
    """Parse dual-row-per-month history PDF (Tata EV / HV layout)."""
    rows: List[Dict[str, Any]] = []
    all_table_rows: List[List] = []
    all_text_lines: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                all_table_rows.extend(table)
            # Always collect text too — needed to supplement pages where
            # extract_table() returns rows that don't carry the month cell
            # (e.g., mid-table page continuations in web-to-PDF documents).
            all_text_lines.extend(
                ln.strip()
                for ln in (page.extract_text() or "").splitlines()
                if ln.strip()
            )

    # Determine which bill_months the table pass already captured so we
    # don't create duplicates when the text pass finds the same data.
    table_months: set = set()
    for row in all_table_rows:
        if row and row[0]:
            tm = re.match(
                r"^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4})",
                str(row[0]).strip(), re.I,
            )
            if tm:
                iso = _bill_month_to_iso(tm.group(1))
                if iso:
                    table_months.add(iso)

    # Text-based supplement: add rows for any months the table pass missed.
    j = 0
    while j < len(all_text_lines):
        tm = _IR_TEXT_LINE_RE.match(all_text_lines[j])
        if tm:
            iso = _bill_month_to_iso(tm.group(1))
            if iso not in table_months:
                ir_row = list(tm.groups())
                lt_row = all_text_lines[j + 1].split() if j + 1 < len(all_text_lines) else []
                all_table_rows.append(ir_row)
                all_table_rows.append(lt_row)
                if iso:
                    table_months.add(iso)
            j += 2
        else:
            j += 1

    i = 0
    while i < len(all_table_rows):
        row = all_table_rows[i]
        if not row or not row[0]:
            i += 1
            continue
        cell0 = str(row[0]).strip()
        m = re.match(r"^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4})", cell0, re.I)
        if not m:
            i += 1
            continue

        month_str = m.group(1)
        ir_row = row
        lt_row = all_table_rows[i + 1] if i + 1 < len(all_table_rows) else []

        def g(r, idx, fn=_to_float):
            try:
                return fn(r[idx]) if len(r) > idx and r[idx] else None
            except Exception:
                return None

        status_raw = str(ir_row[1]).strip() if len(ir_row) > 1 else ""
        status_m = re.match(r"(\d+)\s*/\s*(\w+)", status_raw)
        status = status_m.group(1) if status_m else None
        cat_ab = status_m.group(2) if status_m else None

        bill_date_raw = str(lt_row[-1]).strip() if lt_row else None
        bill_date = None
        if bill_date_raw:
            bd_m = re.match(r"(\d{2})-([A-Za-z]{3})-?(\d{2,4})", bill_date_raw)
            if bd_m:
                bill_date = _parse_date(f"{bd_m.group(1)}-{bd_m.group(2).upper()}-{bd_m.group(3)}")

        rec: Dict[str, Any] = {
            "unique_scno":          unique_scno,
            "bill_month":           _bill_month_to_iso(month_str),
            "status":               status,
            "category_ab":          cat_ab,
            "kwh_closing_reading":  g(ir_row, 3),
            "kwh_units":            g(ir_row, 4),
            "demand_rs":            g(ir_row, 5),
            "je_debit_rs":          g(ir_row, 6),
            "collection_rs":        g(ir_row, 7),
            "je_credit_rs":         g(ir_row, 8),
            "arrears_rs":           g(ir_row, 9),
            "fixed_charges_rs":     g(ir_row, 10),
            "cmd_kva":              g(lt_row, 0),
            "lc_side":              str(lt_row[2]).strip() if len(lt_row) > 2 else None,
            "kvah_closing_reading": g(lt_row, 3),
            "kvah_units":           g(lt_row, 4),
            "billed_units":         g(lt_row, 5),
            "rmd_kva":              g(lt_row, 6),
            "comp_load":            g(lt_row, 7),
            "bill_md":              g(lt_row, 8),
            "power_factor":         g(lt_row, 9),
            "bill_date":            bill_date,
            "source":               "pdf_import",
        }

        if rec["bill_month"]:
            rows.append({k: v for k, v in rec.items() if v is not None})
        i += 2

    log.info("Parsed %d dual-row history records for SCNo %s", len(rows), unique_scno)
    return rows


def parse_billing_history(pdf_path: str, unique_scno: str, dual_row: bool = False) -> List[Dict[str, Any]]:
    """Entry point — choose layout automatically or via dual_row flag."""
    return (
        parse_billing_history_dual(pdf_path, unique_scno)
        if dual_row
        else parse_billing_history_single(pdf_path, unique_scno)
    )


def auto_parse_history(pdf_path: str, unique_scno: str) -> List[Dict[str, Any]]:
    """Auto-detect single vs dual-row layout, then parse."""
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    dual = bool(re.search(r"Cat/CatAB|9N\s+LT|CMD\(KVA\)", text, re.I))
    log.info("Auto-detected layout for %s: %s", pdf_path, "dual" if dual else "single")
    return parse_billing_history(pdf_path, unique_scno, dual_row=dual)