# EV Station Performance Prediction Pipeline

## Project Structure

```
ev_pipeline/
├── config/
│   └── settings.py          # API keys, DB config, constants
├── scrapers/
│   ├── pdf_parser.py         # Parse TGSPDCL consumer detail + history PDFs
│   ├── tgspdcl_scraper.py    # Monthly bill scraper from tgsouthernpower.org
│   └── places_scraper.py     # Google Places API enrichment
├── features/
│   ├── consumption_features.py  # Stats from billing history
│   ├── geo_features.py          # Highway position, milestones, city distances
│   └── competition_features.py  # Nearby EV stations, restaurants
├── db/
│   ├── schema.sql            # PostgreSQL schema
│   └── db_manager.py         # DB read/write helpers
├── utils/
│   └── helpers.py            # Shared utilities
├── pipeline.py               # Orchestrator: run full pipeline
└── scheduler.py              # Monthly cron job runner
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Configure API keys and DB in config/settings.py
# 2. Parse existing PDFs and seed the DB
python pipeline.py --mode seed --pdf_dir ./pdfs/

# 3. Run monthly update (scrapes TGSPDCL + updates features)
python pipeline.py --mode monthly

# 4. Run on a schedule (cron wrapper)
python scheduler.py
```

## Features Engineered

See `features/` modules for full list. Summary:

- **Geo**: lat/lon, highway side (incoming/outgoing), dist to nearest city,
  dist from city pair endpoints and midpoints (e.g. Hyd↔Vijayawada)
- **Consumption**: avg_kwh, std_kwh, months_active, kwh_growth_rate, rolling_avg_3m
- **Competition**: nearby_ev_stations_1km, nearby_restaurants_1km, has_attached_restaurant
- **Station**: charger_count, charger_ratings (30kW, 60kW, etc.), security_deposit,
  contracted_load, meter_phase, category

ev_pipeline/
├── config/
│ ├── **init**.py
│ └── settings.py # API keys, DB config, highway pairs, constants
├── scrapers/
│ ├── **init**.py
│ ├── pdf_parser.py # Parse TGSPDCL consumer detail + history PDFs
│ │ # auto-detects single-row vs dual-row (HV) layout
│ ├── tgspdcl_scraper.py # Monthly bill scraper from tgsouthernpower.org
│ └── places_scraper.py # Google Places API geocoding + competition search
├── features/
│ ├── **init**.py
│ ├── consumption_features.py # Rolling avgs, growth rate, seasonality, anomaly
│ ├── geo_features.py # Highway projection, city distances, direction side
│ └── competition_features.py # Competition intensity, amenity score, demand cliff
├── db/
│ ├── **init**.py
│ ├── schema.sql # PostgreSQL schema (stations, monthly_bills, station_features)
│ └── db_manager.py # Upsert + query helpers (psycopg2)
├── utils/
│ ├── **init**.py
│ └── helpers.py # Type coercions, retry decorator, cache, date utils
├── pipeline.py # Orchestrator: seed / monthly / enrich / features modes
├── scheduler.py # APScheduler wrapper for monthly cron job
└── requirements.txt # Python dependencies
