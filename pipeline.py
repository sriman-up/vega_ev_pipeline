# pipeline.py
"""
EV Station Performance Prediction Pipeline — Orchestrator

Modes:
  --mode seed       Parse PDFs and seed DB (one-time)
  --mode monthly    Scrape latest bills + rebuild features (run monthly)
  --mode enrich     Re-run Places API enrichment for all stations
  --mode features   Recompute all features without scraping
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env before importing settings
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ev_pipeline.config.settings import LOG_FILE
from ev_pipeline.db.db_manager import (
    close_pool,
    get_all_stations,
    get_station_billing_history,
    upsert_monthly_bill,
    upsert_station,
    upsert_station_features,
    ping as db_ping,
    _upsert,
)
from ev_pipeline.features.consumption_features import backfill_rolling_features
from ev_pipeline.features.feature_builder import build_feature_row
from ev_pipeline.features.geo_features import compute_geo_features, nearest_city
from ev_pipeline.scrapers.pdf_parser import parse_consumer_details, auto_parse_history
from ev_pipeline.scrapers.places_scraper import enrich_station
from ev_pipeline.scrapers.tgspdcl_scraper import fetch_latest_bill
from ev_pipeline.scrapers.weather_scraper import fetch_monthly_weather

# Keys backfill_rolling_features adds to bill rows but that don't exist in
# station_features — must not be passed to build_feature_row / upsert_station_features
_BILL_ONLY_KEYS = {"kwh_mom_change", "kwh_yoy_change", "kwh_growth_rate_pct", "is_anomaly"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────

def seed_from_pdfs(pdf_dir: str):
    """
    Walk pdf_dir for paired PDFs:
      <SCNO>-<Name>.pdf          consumer details
      <SCNO>-<Name>-Hist.pdf    billing history
    """
    pdf_path = Path(pdf_dir)
    detail_pdfs = sorted(p for p in pdf_path.glob("*.pdf") if "hist" not in p.name.lower())
    hist_pdfs   = sorted(p for p in pdf_path.glob("*.pdf") if "hist"     in p.name.lower())
    log.info("Found %d detail PDFs and %d history PDFs", len(detail_pdfs), len(hist_pdfs))

    # Build scno -> history pdf map from filename prefix
    hist_map = {}
    for hp in hist_pdfs:
        scno_candidate = hp.stem.split("-")[0]
        if scno_candidate.isdigit():
            hist_map[scno_candidate] = str(hp)

    for dp in detail_pdfs:
        log.info("Processing details PDF: %s", dp.name)
        try:
            station_data = parse_consumer_details(str(dp))
        except Exception as e:
            log.error("Failed to parse %s: %s", dp.name, e)
            continue

        scno = station_data.get("unique_scno")
        if not scno:
            log.warning("No SCNo found in %s, skipping", dp.name)
            continue

        # ── Places enrichment ─────────────────────────────────────────────────
        # If lat/lon was extracted from the PDF maps URL, it is already in
        # station_data — enrich_station will skip geocoding and use it directly.
        log.info("Enriching station %s via Places API", scno)
        places_updates = enrich_station(station_data, fetch_charger_info=True)
        station_data.update(places_updates)
        station_data.pop("_nearby_ev_names", None)

        # ── Geo / highway features ────────────────────────────────────────────
        if station_data.get("latitude") and station_data.get("longitude"):
            geo_feats = compute_geo_features(
                station_data["latitude"], station_data["longitude"]
            )
            station_data.update(geo_feats)
            station_data.update(
                nearest_city(station_data["latitude"], station_data["longitude"])
            )

        station_id = upsert_station(station_data)
        station_data["id"] = station_id   # FIX 1.6: needed by _build_station_features
        log.info("Upserted station id=%d for SCNo %s", station_id, scno)

        # ── History PDF ───────────────────────────────────────────────────────
        hist_pdf = hist_map.get(scno)
        if hist_pdf:
            log.info("Parsing history PDF: %s", hist_pdf)
            try:
                bills = auto_parse_history(hist_pdf, scno)
                bills = backfill_rolling_features(bills)
                for bill in bills:
                    bill["station_id"] = station_id
                    upsert_monthly_bill(bill)
                log.info("Inserted %d bill records for SCNo %s", len(bills), scno)
            except Exception as e:
                log.error("Failed to parse history for SCNo %s: %s", scno, e)
        else:
            log.warning("No history PDF found for SCNo %s", scno)

        _build_station_features(scno, station_data)

    log.info("Seed complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY
# ─────────────────────────────────────────────────────────────────────────────

def monthly_update():
    stations = get_all_stations()
    log.info("Running monthly update for %d stations", len(stations))
    now = datetime.utcnow()
    current_month = f"{now.year}-{now.month:02d}-01"

    for station in stations:
        scno = station["unique_scno"]
        bill = fetch_latest_bill(
            scno,
            tariff_category=station.get("tariff_category", "LT-IX"),
            meter_phase=int(station.get("meter_phase") or 3),
            contracted_load_kw=float(station.get("contracted_load_kva") or 0.0),
        )
        if bill:
            bill["station_id"] = station["id"]
            row_id = upsert_monthly_bill(bill)
            log.info("Saved bill id=%d for SCNo %s month=%s units=%s",
                     row_id, scno, bill.get("bill_month"), bill.get("kwh_units"))
            # Backfill rolling features on full updated history
            history = get_station_billing_history(scno)
            history = backfill_rolling_features(history)
            for rec in history:
                upsert_monthly_bill(rec)
        else:
            log.warning("No bill returned for SCNo %s", scno)

        _build_station_features(scno, station, feature_month=current_month)

    log.info("Monthly update complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENRICH
# ─────────────────────────────────────────────────────────────────────────────

def enrich_all_stations():
    stations = get_all_stations()
    for station in stations:
        scno = station["unique_scno"]
        log.info("Re-enriching station %s", scno)
        updates = enrich_station(station, fetch_charger_info=False)
        if updates.get("latitude"):
            updates.update(compute_geo_features(updates["latitude"], updates["longitude"]))
            updates.update(nearest_city(updates["latitude"], updates["longitude"]))
        updates["unique_scno"] = scno
        updates.pop("_nearby_ev_names", None)
        upsert_station(updates)
    log.info("Enrichment complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_station_features(
    scno: str,
    station: dict,
    feature_month: Optional[str] = None,
):
    if not feature_month:
        now = datetime.utcnow()
        feature_month = f"{now.year}-{now.month:02d}-01"

    # Fetch full history from DB (backfilled rolling features included)
    history = get_station_billing_history(scno)

    # Strip bill-level keys that don't exist as columns in station_features
    # (build_feature_row calls compute_consumption_features which re-derives
    # the station-level equivalents cleanly)
    clean_history = [
        {k: v for k, v in row.items() if k not in _BILL_ONLY_KEYS}
        for row in history
    ]

    # Weather for this feature month
    weather = None
    if station.get("latitude") and station.get("longitude"):
        y, mo = int(feature_month[:4]), int(feature_month[5:7])
        weather = fetch_monthly_weather(
            scno, station["latitude"], station["longitude"], y, mo
        )
        if weather:
            _upsert(
                "monthly_weather", weather,
                conflict_cols=["unique_scno", "weather_month"],
                returning=None,
            )

    feature_row = build_feature_row(station, clean_history, weather, feature_month)
    upsert_station_features(feature_row)
    log.info("Feature vector saved for SCNo %s month=%s", scno, feature_month)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Station Pipeline")
    parser.add_argument("--mode", choices=["seed", "monthly", "enrich", "features"], required=True)
    parser.add_argument("--pdf_dir", default="./pdfs/")
    args = parser.parse_args()

    if not db_ping():
        log.error("Cannot reach Supabase DB - check .env credentials. Exiting.")
        sys.exit(1)

    try:
        if args.mode == "seed":
            seed_from_pdfs(args.pdf_dir)
        elif args.mode == "monthly":
            monthly_update()
        elif args.mode == "enrich":
            enrich_all_stations()
        elif args.mode == "features":
            stations = get_all_stations()
            for s in stations:
                _build_station_features(s["unique_scno"], s)
            log.info("Feature rebuild complete.")
    finally:
        close_pool()