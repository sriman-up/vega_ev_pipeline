# Context: EV Station Performance Prediction Pipeline — Data for Research

## What this is
A pipeline that tracks ~468 EV charging stations in Telangana (India), all on the
TGSPDCL **LT-IX "EV Charging Stations"** tariff (category code `9` in the `stations`
table). Data is stored in Postgres (Supabase). Goal: predict/explain station-level
EV charging demand (kWh) using consumption history, location, competition, weather,
and seasonality features.

## Current data volume (as of 2026-06-17)
| Table | Rows | Notes |
|---|---|---|
| `stations` | 468 | One row per station (Unique SCNo) |
| `monthly_bills` | 6,299 | Spans 2023-05-01 → 2026-06-01 |
| `monthly_weather` | 468 | One row per station per month, Open-Meteo |
| `station_features` | 468 | Wide ML-ready feature table, rebuilt monthly |

## Schema (key tables — full DDL in `ev_pipeline/db/schema.sql`)

### `stations` — one row per station
- Identity: `unique_scno` (PK-ish, business key), `service_number`, `consumer_name`, `address`, `category`/`sub_category` (all `"9"` = LT-IX EV)
- Capacity: `contracted_load_kva`, `connected_load_kva`, `meter_phase`, `multiplying_factor`
- Chargers: `charger_30kw_count`, `charger_60kw_count`, `charger_120kw_count`, `charger_150kw_count`, `total_charger_count`, `total_power_kw`
- Location: `latitude`, `longitude`, Google Places enrichment (`places_rating`, `places_user_ratings_total`, `location_type`, `has_attached_restaurant`)
- Competition (1km radius): `nearby_ev_stations_1km`, `nearby_restaurants_1km`, `nearby_hotels_1km`, `nearby_petrol_pumps_1km`, `nearby_shopping_1km`
- Highway geo: `highway_name`, `dist_from_city_a_km`, `dist_from_city_b_km`, `dist_from_midpoint_km`, `highway_position_ratio`, `direction_side` (3 highway pairs defined: NH65 Hyd-Vijayawada, NH44 Hyd-Nagpur, NH167 Hyd-Nalgonda)

### `monthly_bills` — one row per station per month
- `kwh_units`, `billed_units` — actual billed consumption (see "Data collection" below for how reliable this is per source)
- `demand_rs`, `fixed_charges_rs`, `arrears_rs`, `collection_rs` — financials
- `bill_date`, `bill_month` (1st of month, unique key with `unique_scno`)
- `rolling_avg_3m_kwh`, `rolling_avg_6m_kwh`, `kwh_mom_change`, `kwh_yoy_change`, `kwh_growth_rate_pct`, `is_anomaly` — backfilled derived features
- `source` — `"pdf_import"` (historical, from seeded PDFs) or `"tgspdcl_scrape"` (monthly scrape)

### `monthly_weather` — Open-Meteo, per station per month
`avg_temp_c`, `max_temp_c`, `min_temp_c`, `total_rainfall_mm`, `avg_humidity_pct`, `heatwave_days` (>40°C), `rainfall_days` (>2.5mm)

### `station_features` — wide table, rebuilt monthly, one row per station per `feature_month`
Combines: consumption stats (avg/std/growth/seasonality/anomaly flags), station capacity, competition/amenity scores, highway position, weather, calendar flags (`is_summer`, `is_monsoon`, `is_festival_month`), `months_since_opening`/`is_ramp_up_phase`, and `next_month_kwh` (target variable, filled retroactively next cycle).

## Data collection — sources and reliability

1. **Seed (one-time)**: `pipeline.py --mode seed` parses paired PDFs (`<SCNO>-<Name>.pdf` for consumer details, `<SCNO>-<Name>-Hist.pdf` for billing history) via `ev_pipeline/scrapers/pdf_parser.py`. These are official TGSPDCL bill-history PDFs with real meter-read `kwh_units` — **most reliable** source, `source='pdf_import'`.

2. **Monthly scrape**: `pipeline.py --mode monthly` calls `ev_pipeline/scrapers/tgspdcl_scraper.py::fetch_latest_bill()`, which POSTs to `https://tgsouthernpower.org/paybillonline` with `ukscno=<SCNo>`. As of 2026-06-17 this endpoint reports **Units directly** (a recent fix — an earlier version of the scraper hit a different endpoint, `/billinginfo`, which only exposed bill amount, not units, and required back-calculating kWh from the TGSPDCL tariff schedule — see `tariff.py`). The tariff-based estimate is now only a fallback for the rare case the portal omits Units; it ignores FSA (Fuel Surcharge Adjustment) and is ~3–8% less accurate than the real reported value. **Rows with `source='tgspdcl_scrape'` should be assumed accurate (real Units) unless cross-checked**, since the fallback path logs a warning when used.

3. **Weather**: `ev_pipeline/scrapers/weather_scraper.py`, Open-Meteo API, keyed by station lat/lon + month.

4. **Places enrichment**: `ev_pipeline/scrapers/places_scraper.py`, Google Places API — competition counts, ratings, location type.

## Known caveats for analysis
- All stations are the same tariff category (LT-IX) — no cross-category comparison possible within this dataset.
- `kwh_units` reliability varies by `source` and scrape date: pre-fix scraped rows (if any landed before 2026-06-17) may carry tariff-estimated units rather than true meter reads — worth a sanity pass (e.g., compare `kwh_units` against `demand_rs / 6.00` expected ratio; large deviations flag estimated rows).
- `monthly_bills` history starts 2023-05-01 — station ages vary (`months_since_opening` in `station_features` accounts for ramp-up bias).
- Electricity duty (6%) and customer charge (₹120/month, 3-phase) are baked into bill amounts; not separately itemized in older PDF-imported rows necessarily — check `fixed_charges_rs` population consistency by `source`.

## How to connect
- Postgres via Supabase. Connection mode/credentials in `.env` (not included here — do not paste secrets into other chats). Use `SUPABASE_CONNECTION_MODE=pooler` if direct host DNS fails (project's direct host historically only exposed an IPv6 `AAAA` record, no IPv4).
- Read helpers in `ev_pipeline/db/db_manager.py`: `get_all_stations()`, `get_station_billing_history(scno)`, `get_feature_vectors(scno=None, from_month=None)`.

## Suggested research angles
- Demand growth/seasonality patterns by `highway_position_ratio` / `direction_side`
- Competition intensity (`nearby_ev_stations_1km`) vs `avg_kwh_units` — cannibalization vs hub effect
- `is_ramp_up_phase` stations vs mature stations — growth curve shape
- Weather sensitivity (`heatwave_days`, `avg_temp_c`) vs consumption, given EV battery/cooling load
- Anomaly-flagged months (`is_anomaly`) — investigate causes (outages, billing errors, real demand spikes)
