-- ev_pipeline/db/migrations/002_spatial_layers.sql
--
-- Adds three spatial layers on top of the base station schema:
--   1. H3 hex tiling (resolutions 5/6/7) + coverage-gap grid
--   2. Concentric city-zone bands (core/periurban/highway/remote)
--   3. Static zone_fleet_mix reference table (2W/4W/bus + vehicle specs)
--
-- Idempotent — safe to re-run against an existing database.

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. H3 hex tiling
-- ─────────────────────────────────────────────────────────────────────────

ALTER TABLE public.stations
  ADD COLUMN IF NOT EXISTS h3_res5 text,
  ADD COLUMN IF NOT EXISTS h3_res6 text,
  ADD COLUMN IF NOT EXISTS h3_res7 text;

CREATE INDEX IF NOT EXISTS idx_stations_h3_res5 ON public.stations (h3_res5);
CREATE INDEX IF NOT EXISTS idx_stations_h3_res6 ON public.stations (h3_res6);
CREATE INDEX IF NOT EXISTS idx_stations_h3_res7 ON public.stations (h3_res7);

CREATE TABLE IF NOT EXISTS public.h3_coverage_zones (
  h3_index text NOT NULL,
  resolution integer,
  center_lat double precision,
  center_lng double precision,
  station_count integer,
  nearest_station_dist_km double precision,
  is_coverage_gap boolean,
  pop_density_per_sqkm double precision,
  highway_overlap boolean,
  CONSTRAINT h3_coverage_zones_pkey PRIMARY KEY (h3_index)
);

CREATE INDEX IF NOT EXISTS idx_h3_coverage_zones_resolution
  ON public.h3_coverage_zones (resolution);

CREATE INDEX IF NOT EXISTS idx_h3_coverage_zones_gap
  ON public.h3_coverage_zones (is_coverage_gap)
  WHERE is_coverage_gap;

-- ─────────────────────────────────────────────────────────────────────────
-- 2. Concentric city zones
-- ─────────────────────────────────────────────────────────────────────────
-- Distinct from the existing nearest_city_a/b + nearest_major_city columns,
-- which are highway-pair-projection fields owned by features/geo_features.py.
-- These three columns are the anchor-city radial banding owned by
-- features/city_zones.py.

ALTER TABLE public.stations
  ADD COLUMN IF NOT EXISTS nearest_city character varying,
  ADD COLUMN IF NOT EXISTS dist_to_city_km numeric,
  ADD COLUMN IF NOT EXISTS zone_band character varying;

CREATE INDEX IF NOT EXISTS idx_stations_zone_band ON public.stations (zone_band);

-- ─────────────────────────────────────────────────────────────────────────
-- 3. Fleet mix reference table
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.zone_fleet_mix (
  id serial PRIMARY KEY,
  zone_band character varying NOT NULL,
  vehicle_type character varying NOT NULL,
  fleet_share_pct numeric,
  range_km numeric,
  battery_kwh numeric,
  charge_rate_kw numeric,
  typical_soc_start_pct numeric,
  typical_soc_end_pct numeric,
  avg_session_kwh numeric,
  source_note text,
  last_updated date,
  CONSTRAINT zone_fleet_mix_zone_vehicle_unique UNIQUE (zone_band, vehicle_type)
);

COMMIT;
