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

# Matches a single-row history line extracted as plain text, e.g.:
#   "Apr/2026 01 13321 229 2497.00 3090.00 5587.00 0.00 0.00"
# Groups: month, status, kwh_reading, kwh_units, demand, je_debit,
#         collection, je_credit, arrears.
# Not anchored at line start (matched via re.search): some PDF exports
# (ConsumptionReportHistory.jsp) don't expose this layout via
# extract_table() at all — extract_table() returns 0 rows even though the
# text is clean — so this is parsed line-by-line off extract_text() like
# the dual-row layout, which also tolerates any sidebar-nav text bleed.
_SINGLE_TEXT_LINE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4})"
    r"\s+(\d+)"                     # status
    r"\s+(-?[\d.]+)"                # kwh closing reading
    r"\s+(-?[\d.]+)"                # kwh units
    r"\s+(-?[\d.]+)"                # demand (Rs.)
    r"\s+(-?[\d.]+)"                # JE debit (Rs.)
    r"\s+(-?[\d.]+)"                # collection (Rs.)
    r"\s+(-?[\d.]+)"                # JE credit (Rs.)
    r"\s+(-?[\d.]+)\s*$",           # arrears (Rs.)
    re.I,
)

# Matches the IR (first) line of a dual-row entry extracted as plain text, e.g.:
#   "Apr/2026 01 / IR 9 39665 2905 17876.00 0.00 17876.00 0.00 0.00 0"
# Groups: month, status, cat, kwh_reading, kwh_units, demand, je_debit,
#         collection, je_credit, arrears, fixed_charges
# Arrears (and in principle any Rs. column) can be negative (credit balance),
# e.g. "-764.00" — every numeric group allows an optional leading '-'.
# No leading "^" anchor: the live portal's PDF export (ConsumptionLinkDetails/
# ConsumptionReportHistory.jsp) bleeds sidebar nav text ("Meeting Particulars",
# "Spot Billing Information", ...) onto the same extracted line as a data row
# whenever their y-coordinates happen to line up, so this is matched with
# re.search() rather than re.match().
_IR_TEXT_LINE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{2,4})"
    r"\s+(\d+(?:\s*/\s*\w+)?)"      # status: "01 / IR" or "00"
    r"\s+(\d+)"                     # category
    r"\s+(-?[\d.]+)"                # kwh closing reading
    r"\s+(-?[\d.]+)"                # kwh units
    r"\s+(-?[\d.]+)"                # demand (Rs.)
    r"\s+(-?[\d.]+)"                # JE debit (Rs.)
    r"\s+(-?[\d.]+)"                # collection (Rs.)
    r"\s+(-?[\d.]+)"                # JE credit (Rs.)
    r"\s+(-?[\d.]+)"                # arrears (Rs.)
    r"\s+(-?[\d.]+)\s*$",           # fixed charges
    re.I,
)

# Matches the 9N/LT (second) line of a dual-row entry, e.g.:
#   "0.00 9N LT 18 0 0.00 0.00 54.00 54.00 1.00"
# PF is sometimes omitted entirely (closing/zero-activity months), and the
# trailing bill date — when not split across PDF lines by cell wrapping — can
# appear inline as a single "DD-Mon-YY" token.
_LT_TEXT_LINE_RE = re.compile(
    r"(-?[\d.]+)"                        # cmd_kva
    r"\s+(\w+)"                           # lc
    r"\s+(\w+)"                           # side
    r"\s+(-?[\d.]+)"                      # kvah closing reading
    r"\s+(-?[\d.]+)"                      # kvah units
    r"\s+(-?[\d.]+)"                      # billed units
    r"\s+(-?[\d.]+)"                      # rmd_kva
    r"\s+(-?[\d.]+)"                      # comp load
    r"\s+(-?[\d.]+)"                      # bill MD
    r"(?:\s+(-?[\d.]+))?"                 # power factor (optional)
    r"(?:\s+(\d{1,2}-[A-Za-z]{3}-?\d{0,4}))?"  # inline bill date (optional)
    r"\s*$",
    re.I,
)

# A table cell that wraps ("DD-Mon-" + "YY" on separate lines) splits the
# bill date across two text lines straddling the LT line: the "DD-Mon-"
# fragment appears on the line *before* the LT data, and the bare "YY"/"YYYY"
# remainder appears on the line *after* it.
_DATE_PREFIX_RE = re.compile(r"^(\d{1,2}-[A-Za-z]{3})-\s*$", re.I)
_YEAR_SUFFIX_RE = re.compile(r"^(\d{2,4})\s*$")


def parse_billing_history_single(pdf_path: str, unique_scno: str) -> List[Dict[str, Any]]:
    """Parse standard single-row-per-month history PDF (Swathi layout)."""
    all_text_lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text_lines.extend(
                ln.strip()
                for ln in (page.extract_text() or "").splitlines()
                if ln.strip()
            )

    rows: List[Dict[str, Any]] = []
    for ln in all_text_lines:
        m = _SINGLE_TEXT_LINE_RE.search(ln)
        if not m:
            continue
        rec: Dict[str, Any] = {
            "unique_scno":         unique_scno,
            "bill_month":          _bill_month_to_iso(m.group(1)),
            "status":              m.group(2),
            "kwh_closing_reading": _to_float(m.group(3)),
            "kwh_units":           _to_float(m.group(4)),
            "demand_rs":           _to_float(m.group(5)),
            "je_debit_rs":         _to_float(m.group(6)),
            "collection_rs":       _to_float(m.group(7)),
            "je_credit_rs":        _to_float(m.group(8)),
            "arrears_rs":          _to_float(m.group(9)),
            "source":              "pdf_import",
        }
        if rec["bill_month"]:
            rows.append({k: v for k, v in rec.items() if v is not None})
    log.info("Parsed %d single-row history records for SCNo %s", len(rows), unique_scno)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Billing History — dual-row layout (Tata EV / HV)
# ─────────────────────────────────────────────────────────────────────────────

def parse_billing_history_dual(pdf_path: str, unique_scno: str) -> List[Dict[str, Any]]:
    """
    Parse dual-row-per-month history PDF (Tata EV / HV layout).

    pdfplumber's extract_table() is unreliable across page breaks for this
    layout: continuation pages can come back with phantom leading columns,
    which makes every row's first cell None and silently drops the whole
    page. Plain text extraction doesn't have that problem, so this parses
    line-by-line off extract_text() instead and reconstructs IR/LT pairs
    from the sequence of recognized line types — anything that isn't an
    IR line, LT line, or bill-date fragment (sidebar/header/footer noise)
    is simply skipped.
    """
    all_text_lines: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text_lines.extend(
                ln.strip()
                for ln in (page.extract_text() or "").splitlines()
                if ln.strip()
            )

    # Classify each line into a typed event; unrecognized lines (nav menu,
    # page headers/footers, table column headers) are dropped.
    events: List[Tuple[str, Any]] = []
    for ln in all_text_lines:
        m = _IR_TEXT_LINE_RE.search(ln)
        if m:
            events.append(("IR", m))
            continue
        m = _LT_TEXT_LINE_RE.search(ln)
        if m:
            events.append(("LT", m))
            continue
        m = _DATE_PREFIX_RE.match(ln)
        if m:
            events.append(("DATE_PREFIX", m.group(1)))
            continue
        m = _YEAR_SUFFIX_RE.match(ln)
        if m:
            events.append(("YEAR", m.group(1)))

    rows: List[Dict[str, Any]] = []
    pending_ir: Optional[re.Match] = None
    date_prefix: Optional[str] = None

    i = 0
    while i < len(events):
        etype, edata = events[i]
        if etype == "IR":
            pending_ir = edata
            date_prefix = None
        elif etype == "DATE_PREFIX":
            date_prefix = edata
        elif etype == "LT":
            inline_date = edata.group(11)
            bill_date = _parse_date(inline_date) if inline_date else None
            if bill_date is None and date_prefix:
                if i + 1 < len(events) and events[i + 1][0] == "YEAR":
                    bill_date = _parse_date(f"{date_prefix}-{events[i + 1][1]}")
                    i += 1  # consume the year fragment
            if pending_ir is not None:
                rows.append(_build_dual_row(unique_scno, pending_ir, edata, bill_date))
                pending_ir = None
            date_prefix = None
        # "YEAR" events not directly following an LT line are stray noise.
        i += 1

    log.info("Parsed %d dual-row history records for SCNo %s", len(rows), unique_scno)
    return rows


def _build_dual_row(
    unique_scno: str, ir: "re.Match", lt: "re.Match", bill_date: Optional[str],
) -> Dict[str, Any]:
    status_raw = ir.group(2).strip()
    status_m = re.match(r"(\d+)\s*/\s*(\w+)", status_raw)
    status = status_m.group(1) if status_m else status_raw or None
    cat_ab = status_m.group(2) if status_m else None

    rec: Dict[str, Any] = {
        "unique_scno":          unique_scno,
        "bill_month":           _bill_month_to_iso(ir.group(1)),
        "status":               status,
        "category_ab":          cat_ab,
        "kwh_closing_reading":  _to_float(ir.group(4)),
        "kwh_units":            _to_float(ir.group(5)),
        "demand_rs":            _to_float(ir.group(6)),
        "je_debit_rs":          _to_float(ir.group(7)),
        "collection_rs":        _to_float(ir.group(8)),
        "je_credit_rs":         _to_float(ir.group(9)),
        "arrears_rs":           _to_float(ir.group(10)),
        "fixed_charges_rs":     _to_float(ir.group(11)),
        "cmd_kva":              _to_float(lt.group(1)),
        "lc_side":              lt.group(3),
        "kvah_closing_reading": _to_float(lt.group(4)),
        "kvah_units":           _to_float(lt.group(5)),
        "billed_units":         _to_float(lt.group(6)),
        "rmd_kva":              _to_float(lt.group(7)),
        "comp_load":            _to_float(lt.group(8)),
        "bill_md":              _to_float(lt.group(9)),
        "power_factor":         _to_float(lt.group(10)),
        "bill_date":            bill_date,
        "source":               "pdf_import",
    }
    return {k: v for k, v in rec.items() if v is not None}


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