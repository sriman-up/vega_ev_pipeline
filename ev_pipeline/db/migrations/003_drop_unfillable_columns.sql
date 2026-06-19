-- ev_pipeline/db/migrations/003_drop_unfillable_columns.sql
--
-- dist_nearest_toll_plaza_km / dist_nearest_rest_area_km were added to the
-- schema but never had a real data source (no toll-plaza / rest-area
-- coordinate dataset exists anywhere in this codebase). Rather than leave
-- them permanently NULL or fabricate plausible-looking coordinates for real
-- infrastructure, drop them. See README.md "Known Data Gaps" for re-adding
-- them later if a real NHAI/highway dataset becomes available.
--
-- Idempotent — safe to re-run against an existing database.

BEGIN;

ALTER TABLE public.stations
  DROP COLUMN IF EXISTS dist_nearest_toll_plaza_km,
  DROP COLUMN IF EXISTS dist_nearest_rest_area_km;

ALTER TABLE public.station_features
  DROP COLUMN IF EXISTS dist_nearest_toll_plaza_km,
  DROP COLUMN IF EXISTS dist_nearest_rest_area_km;

COMMIT;
