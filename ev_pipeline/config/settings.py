# config/settings.py
"""
Central configuration. Override via environment variables or a .env file.
"""

import os
from pathlib import Path

# Every constant below is computed once, the first time this module is
# imported anywhere — and that computation is cached for the life of the
# process. If .env hasn't been loaded yet at that moment (e.g. some other
# module imports ev_pipeline.* before an entry point's own load_dotenv()
# call runs), DB_URL etc. get permanently baked in with placeholder defaults,
# and calling load_dotenv() later does nothing to fix it. Loading it here —
# the first thing this module does — makes that bug structurally impossible
# regardless of which script imports what, in what order. Safe to also call
# load_dotenv() again in entry points; it's idempotent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Google Places API ─────────────────────────────────────────────────────────
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "YOUR_KEY_HERE")
PLACES_NEARBY_RADIUS_M = 1000  # 1 km radius for competition search

# ── TGSPDCL Bill Enquiry ──────────────────────────────────────────────────────
TGSPDCL_BILL_URL = "https://tgsouthernpower.org/paybillonline"
# Selenium / requests-html user-agent
SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
SCRAPER_TIMEOUT_S = 30
SCRAPER_RETRY_COUNT = 3

# ── Supabase / Database ───────────────────────────────────────────────────────
#
# Supabase gives you two ways to connect:
#
#   1. Direct connection  (best for long-lived scripts / pipelines)
#      Host:  db.<project-ref>.supabase.co   Port: 5432
#      Use this for the pipeline and scheduler.
#
#   2. Connection pooler  (best for serverless / many short connections)
#      Host:  aws-0-<region>.pooler.supabase.com   Port: 6543 (transaction mode)
#                                                    Port: 5432 (session mode)
#      Use session mode (5432) if you need prepared statements / LISTEN/NOTIFY.
#
# Find all values at: Supabase Dashboard → Settings → Database → Connection string
#
# Recommended: store secrets in a .env file and never commit it.
# pip install python-dotenv  then add `from dotenv import load_dotenv; load_dotenv()`
# at the top of pipeline.py / scheduler.py.

# ── Connection mode toggle ────────────────────────────────────────────────────
# "direct"  → connects straight to Postgres on port 5432 (recommended for pipeline)
# "pooler"  → connects via Supabase connection pooler
SUPABASE_CONNECTION_MODE = os.getenv("SUPABASE_CONNECTION_MODE", "direct")

# ── Supabase project credentials ──────────────────────────────────────────────
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "your-project-ref")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "your-db-password")
SUPABASE_REGION      = os.getenv("SUPABASE_REGION", "ap-south-1")  # Mumbai for India

# ── Direct connection (port 5432) ─────────────────────────────────────────────
SUPABASE_DIRECT_HOST = os.getenv(
    "SUPABASE_DIRECT_HOST",
    f"db.{SUPABASE_PROJECT_REF}.supabase.co",
)

# ── Pooler connection (transaction mode, port 6543) ──────────────────────────
SUPABASE_POOLER_HOST = os.getenv(
    "SUPABASE_POOLER_HOST",
    f"aws-1-{SUPABASE_REGION}.pooler.supabase.com",
)
SUPABASE_POOLER_PORT = int(os.getenv("SUPABASE_POOLER_PORT", "6543"))

# ── Derived DB_URL (used everywhere via db_manager) ──────────────────────────
_DB_USER = f"postgres.{SUPABASE_PROJECT_REF}"   # Supabase pooler requires this format
_DB_NAME = "postgres"                            # Supabase default DB name

if SUPABASE_CONNECTION_MODE == "pooler":
    DB_URL = (
        f"postgresql://{_DB_USER}:{SUPABASE_DB_PASSWORD}"
        f"@{SUPABASE_POOLER_HOST}:{SUPABASE_POOLER_PORT}/{_DB_NAME}"
        f"?sslmode=require"
    )
else:
    # Direct — plain postgres user on port 5432
    DB_URL = (
        f"postgresql://postgres:{SUPABASE_DB_PASSWORD}"
        f"@{SUPABASE_DIRECT_HOST}:5432/{_DB_NAME}"
        f"?sslmode=require"
    )

# ── Supabase REST / JS client (optional — for edge functions or dashboards) ───
SUPABASE_URL     = os.getenv("SUPABASE_URL",     f"https://{SUPABASE_PROJECT_REF}.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "your-anon-key")   # safe to expose in browser
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")           # keep secret — server only

# ── Highway City Pairs ─────────────────────────────────────────────────────────
# Each entry: (city_a, lat_a, lon_a, city_b, lat_b, lon_b, highway_name)
# Stations will be matched to the nearest highway pair and given:
#   dist_from_a, dist_from_b, dist_from_midpoint, quarter_markers (0.25, 0.75)
#   direction_side: "outgoing_from_a" | "outgoing_from_b"
HIGHWAY_PAIRS = [
    {
        "name": "NH65 Hyderabad-Vijayawada",
        "city_a": "Hyderabad",
        "lat_a": 17.3850, "lon_a": 78.4867,
        "city_b": "Vijayawada",
        "lat_b": 16.5062, "lon_b": 80.6480,
    },
    {
        "name": "NH44 Hyderabad-Nagpur",
        "city_a": "Hyderabad",
        "lat_a": 17.3850, "lon_a": 78.4867,
        "city_b": "Nagpur",
        "lat_b": 21.1458, "lon_b": 79.0882,
    },
    {
        "name": "NH167 Hyderabad-Nalgonda",
        "city_a": "Hyderabad",
        "lat_a": 17.3850, "lon_a": 78.4867,
        "city_b": "Nalgonda",
        "lat_b": 17.0575, "lon_b": 79.2672,
    },
]

# Max distance (km) from a highway pair to associate the station with it
HIGHWAY_PAIR_MAX_DIST_KM = 10

# Assumed average road speed (km/h) used to estimate travel_time_from_city_a/b_min
# from the highway-projection distances. Heuristic only — no Distance Matrix API
# is configured. ~55 km/h reflects mixed NH traffic (tolls, town stretches, trucks)
# rather than free-flow speed limits.
HIGHWAY_AVG_SPEED_KMH = 55

# ── H3 hex tiling ────────────────────────────────────────────────────────────
H3_RESOLUTIONS = [5, 6, 7]

# India's stated ~30 km inter-station spacing target for highway EV corridors
# implies a ~15 km service radius — a hex farther than this from every
# station is flagged as a coverage gap.
H3_COVERAGE_GAP_KM = 15.0

# Rectangular approximation of the Telangana state boundary, used to enumerate
# the full hex grid for coverage-gap analysis. Not the precise state polygon —
# a thin border of hexes just outside Telangana will also be generated (and
# likely flagged as gaps). Swap in a real GeoJSON polygon if that matters.
TELANGANA_BBOX = {
    "min_lat": 15.80, "max_lat": 19.95,
    "min_lon": 77.25, "max_lon": 81.35,
}

# ── Concentric city zones ─────────────────────────────────────────────────────
# Anchor cities for radial zone banding (distinct from HIGHWAY_PAIRS / the
# nearest_major_city lookup in geo_features.py, which serve highway-projection
# features). Each entry: (city_name, lat, lon).
ANCHOR_CITIES = [
    ("Hyderabad",  17.3850, 78.4867),
    ("Warangal",   17.9784, 79.5941),
    ("Nizamabad",  18.6725, 78.0941),
    ("Khammam",    17.2473, 80.1514),
    ("Karimnagar", 18.4386, 79.1288),
]

# Zone band upper bounds (km) from the nearest anchor city, checked in order.
# Anything beyond the last bound falls into "remote".
ZONE_BAND_BOUNDS_KM = [
    (30,  "core"),
    (60,  "periurban"),
    (120, "highway"),
]

# ── Scheduling ────────────────────────────────────────────────────────────────
# Day of month to run the monthly scrape (1-28)
MONTHLY_SCRAPE_DAY = 10

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"