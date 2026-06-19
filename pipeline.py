# pipeline.py
"""
EV Station Performance Prediction Pipeline — Orchestrator

Modes:
  --mode seed       Parse PDFs and seed DB (one-time)
  --mode monthly    Scrape latest bills + rebuild features (run monthly)
  --mode enrich     Re-run Places API enrichment for all stations
  --mode features   Recompute all features without scraping
  --mode spatial    Rebuild H3 tiling, city zones, fleet mix reference table
  --mode backfill   Fill derived fields (station_name, commissioning_date,
                    last_bill_month, avg_monthly_kwh_lifetime, tariff, travel
                    time) for existing rows — no Places/scraper API calls
  --mode backfill-features
                    Write one station_features row per historical bill month
                    per station (required before ml/train.py has any data —
                    see backfill_feature_history() docstring)
                    --begin-scno <unique_scno> resumes from that station onward
                    (e.g. after interrupting a long backfill-features run)
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

from ev_pipeline.config.settings import H3_RESOLUTIONS, LOG_FILE
from ev_pipeline.db.db_manager import (
    bulk_backfill_commissioning_dates,
    bulk_backfill_station_names,
    bulk_upsert_h3_coverage_zones,
    close_pool,
    get_all_stations,
    get_monthly_weather,
    get_station_billing_history,
    refresh_all_station_bill_summaries,
    upsert_monthly_bill,
    upsert_station,
    upsert_station_features,
    ping as db_ping,
    _upsert,
)
from ev_pipeline.features.city_zones import compute_city_zone
from ev_pipeline.features.consumption_features import backfill_rolling_features
from ev_pipeline.features.feature_builder import build_feature_row
from ev_pipeline.features.fleet_mix import seed_zone_fleet_mix
from ev_pipeline.features.geo_features import compute_geo_features, nearest_city
from ev_pipeline.features.spatial_h3 import assign_h3_indices, build_coverage_zones
from ev_pipeline.features.tariff_enrichment import enrich_tariff
from ev_pipeline.scrapers.pdf_parser import parse_consumer_details, auto_parse_history
from ev_pipeline.scrapers.places_scraper import enrich_station
from ev_pipeline.scrapers.tgspdcl_scraper import fetch_latest_bill
from ev_pipeline.scrapers.weather_scraper import fetch_monthly_weather
from ev_pipeline.utils.helpers import derive_station_name

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
            lat, lon = station_data["latitude"], station_data["longitude"]
            geo_feats = compute_geo_features(lat, lon)
            station_data.update(geo_feats)
            station_data.update(nearest_city(lat, lon))
            station_data.update(assign_h3_indices(lat, lon))
            station_data.update(compute_city_zone(lat, lon))

        # ── Display name, tariff, commissioning date ──────────────────────────
        station_data["station_name"] = derive_station_name(
            station_data.get("consumer_name"), station_data.get("places_name")
        )
        station_data.update(enrich_tariff(station_data))
        if station_data.get("supply_date") and not station_data.get("commissioning_date"):
            station_data["commissioning_date"] = station_data["supply_date"]

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

    refresh_all_station_bill_summaries()
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

        # Merge onto the existing row so tariff/name derivation can see
        # charger counts and consumer_name that `updates` alone doesn't carry.
        merged = {**station, **updates}
        updates["station_name"] = derive_station_name(
            merged.get("consumer_name"), merged.get("places_name")
        )
        updates.update(enrich_tariff(merged))

        upsert_station(updates)
    log.info("Enrichment complete.")


# ─────────────────────────────────────────────────────────────────────────────
# BACKFILL — derived fields for existing rows, no external API calls
# ─────────────────────────────────────────────────────────────────────────────

def backfill_derived_fields():
    """
    Fill derived fields that don't require a fresh scrape/API call:
      - station_name        (consumer_name, fallback places_name)
      - commissioning_date  (mirrors supply_date — see README 'Known Data Gaps')
      - last_bill_month / avg_monthly_kwh_lifetime (aggregated from monthly_bills)
      - travel_time_from_city_a/b_min (heuristic, from existing lat/lon)
      - cpo / tariff_inr_per_kwh / tariff_basis / tariff_source_note / tariff_last_updated
    Safe to re-run. Use this after a schema change or for a one-time catch-up
    on rows seeded before this enrichment was wired into seed/enrich.
    """
    bulk_backfill_station_names()
    bulk_backfill_commissioning_dates()
    refresh_all_station_bill_summaries()

    stations = get_all_stations()
    for station in stations:
        updates = {"unique_scno": station["unique_scno"]}
        lat, lon = station.get("latitude"), station.get("longitude")
        if lat and lon:
            updates.update(compute_geo_features(lat, lon))
        updates.update(enrich_tariff(station))
        upsert_station(updates)

    log.info("Derived-field backfill complete for %d stations", len(stations))


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL — H3 tiling, concentric city zones, fleet mix
# ─────────────────────────────────────────────────────────────────────────────

def build_spatial_layers():
    """
    Rebuild all three spatial layers:
      1. Per-station H3 indices (res 5/6/7) + h3_coverage_zones gap grid
      2. Per-station nearest_city / dist_to_city_km / zone_band
      3. Static zone_fleet_mix reference table (idempotent seed)
    """
    stations = get_all_stations()
    log.info("Building spatial layers for %d stations", len(stations))

    for station in stations:
        lat, lon = station.get("latitude"), station.get("longitude")
        if lat is None or lon is None:
            continue
        updates = {"unique_scno": station["unique_scno"]}
        updates.update(assign_h3_indices(lat, lon))
        updates.update(compute_city_zone(lat, lon))
        upsert_station(updates)
        station.update(updates)  # so build_coverage_zones below sees fresh h3 indices

    for resolution in H3_RESOLUTIONS:
        zones = build_coverage_zones(stations, resolution)
        bulk_upsert_h3_coverage_zones(zones)

    seed_zone_fleet_mix()
    log.info("Spatial layer build complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature history backfill
# ─────────────────────────────────────────────────────────────────────────────

def backfill_feature_history(begin_scno: Optional[str] = None):
    """
    station_features only ever gets ONE row per station: seed/monthly/features
    all default feature_month to "now", so years of monthly_bills history were
    never turned into historical station_features rows. ml/train.py's
    get_feature_matrix(min_months=3) has nothing to return as a result.

    This walks every station's full bill history and writes one station_features
    row per historical month (using only bills up to that month — no leakage,
    via compute_consumption_features' up_to_month cutoff), so training has a
    real multi-month time series per station. Safe to re-run — each (scno, month)
    upserts in place. Weather is cached (get_monthly_weather) so re-runs are fast,
    but the first run still makes one Open-Meteo call per station-month and can
    take a while for hundreds of stations x dozens of months.

    Interrupting (Ctrl+C) is safe — every write commits as it happens, nothing
    spans multiple stations in one transaction. Pass begin_scno to skip straight
    to where you left off instead of re-walking already-completed stations
    (harmless either way since weather is cached and upserts are idempotent —
    this is purely to save time, matches the unique_scno ordering get_all_stations()
    already returns).
    """
    stations = get_all_stations()
    if begin_scno:
        before = len(stations)
        stations = [s for s in stations if s["unique_scno"] >= begin_scno]
        log.info("--begin-scno %s: skipping %d already-completed stations, %d remaining",
                  begin_scno, before - len(stations), len(stations))
    log.info("Backfilling feature history for %d stations", len(stations))
    total_rows = 0

    for station in stations:
        scno = station["unique_scno"]
        history = get_station_billing_history(scno)
        months = sorted({str(r["bill_month"])[:7] for r in history if r.get("bill_month")})
        for ym in months:
            _build_station_features(scno, station, feature_month=f"{ym}-01", history=history)
            total_rows += 1
        if months:
            log.info("SCNo %s: backfilled %d months", scno, len(months))

    log.info("Feature history backfill complete: %d station-months written", total_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_station_features(
    scno: str,
    station: dict,
    feature_month: Optional[str] = None,
    history: Optional[list] = None,
):
    if not feature_month:
        now = datetime.utcnow()
        feature_month = f"{now.year}-{now.month:02d}-01"

    # Fetch full history from DB unless the caller already has it (backfill
    # passes it in once per station instead of once per historical month)
    if history is None:
        history = get_station_billing_history(scno)

    # Strip bill-level keys that don't exist as columns in station_features
    # (build_feature_row calls compute_consumption_features which re-derives
    # the station-level equivalents cleanly)
    clean_history = [
        {k: v for k, v in row.items() if k not in _BILL_ONLY_KEYS}
        for row in history
    ]

    # Weather for this feature month — check the cache before hitting Open-Meteo
    weather = None
    if station.get("latitude") and station.get("longitude"):
        weather = get_monthly_weather(scno, feature_month)
        if not weather:
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
    parser.add_argument(
        "--mode",
        choices=["seed", "monthly", "enrich", "features", "spatial", "backfill", "backfill-features"],
        required=True,
    )
    parser.add_argument("--pdf_dir", default="./pdfs/")
    parser.add_argument(
        "--begin-scno", default=None,
        help="--mode backfill-features only: resume from this unique_scno onward",
    )
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
        elif args.mode == "spatial":
            build_spatial_layers()
        elif args.mode == "backfill":
            backfill_derived_fields()
        elif args.mode == "backfill-features":
            backfill_feature_history(begin_scno=args.begin_scno)
    finally:
        close_pool()