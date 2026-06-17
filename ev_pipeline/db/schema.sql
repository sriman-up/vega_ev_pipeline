-- db/schema.sql
-- PostgreSQL / Supabase schema for EV station performance prediction pipeline
-- Run once:  psql "<connection-string>" -f schema.sql
-- Or paste into Supabase Dashboard → SQL Editor

-- ─────────────────────────────────────────────────────────────────────────────
-- STATIONS  (one row per unique service connection / SCNo)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stations (
    id                  SERIAL PRIMARY KEY,
    unique_scno         VARCHAR(20)  NOT NULL UNIQUE,
    service_number      VARCHAR(20),
    consumer_name       VARCHAR(200),
    address             TEXT,
    pin_code            VARCHAR(10),
    phone               VARCHAR(20),
    category            VARCHAR(10),
    sub_category        VARCHAR(10),
    consumer_type       VARCHAR(10),          -- HV / LT
    section_code_name   VARCHAR(100),
    ero_name            VARCHAR(100),
    circle_name         VARCHAR(100),
    sub_division        VARCHAR(20),
    area_code           VARCHAR(20),
    supply_date         DATE,
    contracted_load_kva NUMERIC(10,2),
    connected_load_kva  NUMERIC(10,2),
    meter_number        VARCHAR(30),
    meter_phase         SMALLINT,
    multiplying_factor  NUMERIC(6,3),
    security_deposit    NUMERIC(12,2),

    -- ── Charger inventory ──────────────────────────────────────────────────
    charger_30kw_count  SMALLINT,
    charger_60kw_count  SMALLINT,
    charger_120kw_count SMALLINT,
    charger_150kw_count SMALLINT,
    charger_other_json  JSONB,               -- {"50kw": 2, "240kw": 1} etc.
    total_charger_count SMALLINT,
    total_power_kw      NUMERIC(8,2),        -- sum of all charger capacities

    -- ── Google Places enrichment ───────────────────────────────────────────
    latitude            NUMERIC(12,8),
    longitude           NUMERIC(12,8),
    places_id           VARCHAR(100),
    places_name         VARCHAR(200),
    places_rating       NUMERIC(3,1),
    places_user_ratings_total INT,
    has_attached_restaurant   BOOLEAN,

    -- ── Location type (from Places 'types' field) ─────────────────────────
    -- e.g. gas_station, shopping_mall, highway_rest_area, hotel, parking
    location_type       VARCHAR(50),
    location_type_raw   JSONB,               -- full Places types array

    -- ── Competition (1 km radius, refreshed monthly) ───────────────────────
    nearby_ev_stations_1km    SMALLINT,
    nearby_restaurants_1km    SMALLINT,
    nearby_hotels_1km         SMALLINT,
    nearby_petrol_pumps_1km   SMALLINT,
    nearby_shopping_1km       SMALLINT,      -- malls, supermarkets
    competition_last_updated  TIMESTAMP,

    -- ── Highway geo features ───────────────────────────────────────────────
    highway_name              VARCHAR(100),
    nearest_city_a            VARCHAR(100),
    nearest_city_b            VARCHAR(100),
    dist_from_city_a_km       NUMERIC(8,2),
    dist_from_city_b_km       NUMERIC(8,2),
    dist_from_midpoint_km     NUMERIC(8,2),
    dist_from_quarter_a_km    NUMERIC(8,2),
    dist_from_quarter_b_km    NUMERIC(8,2),
    total_highway_length_km   NUMERIC(8,2),
    highway_position_ratio    NUMERIC(5,4),  -- 0.0=city_a  1.0=city_b
    direction_side            VARCHAR(40),   -- outgoing_from_a | outgoing_from_b | midpoint_zone

    -- ── Road traffic proxy (from Google Maps Distance Matrix, optional) ───
    -- average travel time from nearest city (minutes) — higher = more remote
    travel_time_from_city_a_min  NUMERIC(8,2),
    travel_time_from_city_b_min  NUMERIC(8,2),

    -- ── Nearest major amenity distances (km) ──────────────────────────────
    dist_nearest_city_km         NUMERIC(8,2),
    nearest_major_city           VARCHAR(100),
    dist_nearest_toll_plaza_km   NUMERIC(8,2),
    dist_nearest_rest_area_km    NUMERIC(8,2),

    -- ── Metadata ───────────────────────────────────────────────────────────
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- MONTHLY_BILLS  (raw billing data — one row per station per month)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monthly_bills (
    id                   SERIAL PRIMARY KEY,
    station_id           INT NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
    unique_scno          VARCHAR(20) NOT NULL,
    bill_month           DATE NOT NULL,

    -- Raw TGSPDCL fields
    status               VARCHAR(5),
    category_ab          VARCHAR(10),
    kwh_closing_reading  NUMERIC(12,2),
    kwh_units            NUMERIC(10,2),
    kvah_closing_reading NUMERIC(12,2),
    kvah_units           NUMERIC(10,2),
    billed_units         NUMERIC(10,2),
    demand_rs            NUMERIC(12,2),
    fixed_charges_rs     NUMERIC(12,2),
    je_debit_rs          NUMERIC(12,2),
    collection_rs        NUMERIC(12,2),
    je_credit_rs         NUMERIC(12,2),
    arrears_rs           NUMERIC(12,2),
    rmd_kva              NUMERIC(8,2),
    cmd_kva              NUMERIC(8,2),
    comp_load            NUMERIC(8,2),
    bill_md              NUMERIC(8,2),
    power_factor         NUMERIC(5,3),
    bill_date            DATE,
    lc_side              VARCHAR(10),

    -- Derived rolling features (backfilled by consumption_features.py)
    rolling_avg_3m_kwh   NUMERIC(10,2),
    rolling_avg_6m_kwh   NUMERIC(10,2),
    kwh_mom_change       NUMERIC(10,2),
    kwh_yoy_change       NUMERIC(10,2),
    kwh_growth_rate_pct  NUMERIC(8,4),
    is_anomaly           BOOLEAN,

    -- Source tracking
    source               VARCHAR(20) DEFAULT 'pdf_import',
    scraped_at           TIMESTAMP DEFAULT NOW(),

    UNIQUE (unique_scno, bill_month)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- MONTHLY_WEATHER  (weather context per station per month — from Open-Meteo)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monthly_weather (
    id                      SERIAL PRIMARY KEY,
    unique_scno             VARCHAR(20) NOT NULL,
    weather_month           DATE NOT NULL,           -- 1st of month
    avg_temp_c              NUMERIC(5,2),
    max_temp_c              NUMERIC(5,2),
    min_temp_c              NUMERIC(5,2),
    total_rainfall_mm       NUMERIC(8,2),
    avg_humidity_pct        NUMERIC(5,2),
    avg_wind_speed_kmh      NUMERIC(6,2),
    heatwave_days           SMALLINT,                -- days > 40°C
    rainfall_days           SMALLINT,                -- days with > 2.5mm rain
    fetched_at              TIMESTAMP DEFAULT NOW(),
    UNIQUE (unique_scno, weather_month)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- STATION_FEATURES  (wide ML-ready feature table, rebuilt monthly)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS station_features (
    id                              SERIAL PRIMARY KEY,
    station_id                      INT NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
    unique_scno                     VARCHAR(20) NOT NULL,
    feature_month                   DATE NOT NULL,

    -- ── Consumption history features ──────────────────────────────────────
    avg_kwh_units                   NUMERIC(10,2),
    std_kwh_units                   NUMERIC(10,2),
    months_active                   SMALLINT,
    kwh_growth_rate_overall         NUMERIC(8,4),
    rolling_avg_3m_kwh              NUMERIC(10,2),
    rolling_avg_6m_kwh              NUMERIC(10,2),
    max_kwh_units                   NUMERIC(10,2),
    min_kwh_units                   NUMERIC(10,2),
    seasonal_summer_avg_kwh         NUMERIC(10,2),
    seasonal_winter_avg_kwh         NUMERIC(10,2),
    pct_months_zero_consumption     NUMERIC(5,2),
    kwh_coefficient_of_variation    NUMERIC(8,4),
    demand_cliff_detected           BOOLEAN,

    -- ── Station capacity & infrastructure ────────────────────────────────
    contracted_load_kva             NUMERIC(10,2),
    total_charger_count             SMALLINT,
    charger_30kw_count              SMALLINT,
    charger_60kw_count              SMALLINT,
    charger_120kw_count             SMALLINT,
    charger_150kw_count             SMALLINT,
    total_power_kw                  NUMERIC(8,2),    -- total installed charging capacity
    charger_mix_ratio               NUMERIC(5,4),    -- fast (>=60kW) / total chargers
    power_utilisation_pct           NUMERIC(6,2),    -- avg_kwh / (total_power_kw * 720hrs)
    meter_phase                     SMALLINT,
    security_deposit                NUMERIC(12,2),

    -- ── Competition & amenity ─────────────────────────────────────────────
    nearby_ev_stations_1km          SMALLINT,
    nearby_restaurants_1km          SMALLINT,
    nearby_hotels_1km               SMALLINT,
    nearby_petrol_pumps_1km         SMALLINT,
    nearby_shopping_1km             SMALLINT,
    has_attached_restaurant         BOOLEAN,
    places_rating                   NUMERIC(3,1),
    places_user_ratings_total       INT,
    competition_intensity           NUMERIC(5,4),    -- 0–1 score
    amenity_score                   NUMERIC(5,4),    -- 0–1 score
    location_type                   VARCHAR(50),     -- gas_station / hotel / mall etc.

    -- ── Highway & geo ─────────────────────────────────────────────────────
    dist_from_city_a_km             NUMERIC(8,2),
    dist_from_city_b_km             NUMERIC(8,2),
    dist_from_midpoint_km           NUMERIC(8,2),
    dist_from_quarter_a_km          NUMERIC(8,2),
    dist_from_quarter_b_km          NUMERIC(8,2),
    highway_position_ratio          NUMERIC(5,4),
    direction_side                  VARCHAR(40),
    total_highway_length_km         NUMERIC(8,2),
    dist_nearest_toll_plaza_km      NUMERIC(8,2),
    dist_nearest_rest_area_km       NUMERIC(8,2),
    travel_time_from_city_a_min     NUMERIC(8,2),

    -- ── Weather context (month being predicted) ───────────────────────────
    avg_temp_c                      NUMERIC(5,2),
    max_temp_c                      NUMERIC(5,2),
    total_rainfall_mm               NUMERIC(8,2),
    avg_humidity_pct                NUMERIC(5,2),
    heatwave_days                   SMALLINT,
    rainfall_days                   SMALLINT,

    -- ── Calendar / seasonality ────────────────────────────────────────────
    month_of_year                   SMALLINT,        -- 1–12
    quarter                         SMALLINT,        -- 1–4
    is_summer                       BOOLEAN,         -- Apr–Jun (peak EV cooling load)
    is_monsoon                      BOOLEAN,         -- Jul–Sep
    is_festival_month               BOOLEAN,         -- Diwali/Dussehra months (Oct/Nov)
    days_in_month                   SMALLINT,

    -- ── EV market maturity proxy ──────────────────────────────────────────
    -- Months since station opened — controls for ramp-up period
    months_since_opening            SMALLINT,
    is_ramp_up_phase                BOOLEAN,         -- first 6 months

    -- ── Target variable ───────────────────────────────────────────────────
    next_month_kwh                  NUMERIC(10,2),   -- filled on next monthly cycle

    computed_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (unique_scno, feature_month)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- INDEXES
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_monthly_bills_scno_month   ON monthly_bills    (unique_scno, bill_month);
CREATE INDEX IF NOT EXISTS idx_station_features_scno      ON station_features (unique_scno, feature_month);
CREATE INDEX IF NOT EXISTS idx_station_features_month     ON station_features (feature_month);
CREATE INDEX IF NOT EXISTS idx_stations_scno              ON stations         (unique_scno);
CREATE INDEX IF NOT EXISTS idx_monthly_weather_scno_month ON monthly_weather  (unique_scno, weather_month);