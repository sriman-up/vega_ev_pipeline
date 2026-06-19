-- ev_pipeline/db/migrations/005_uptime_estimate.sql
--
-- Adds estimated_uptime_hours alongside the existing power_utilisation_pct:
-- avg_kwh / total_power_kw = hours of full-rated-power-equivalent operation
-- in the month. Same underlying signal as power_utilisation_pct, just in
-- hours instead of percent — for dashboards/reporting, not a model feature
-- (it's a near-perfect linear rescale, so it's redundant as training input).
--
-- Idempotent — safe to re-run against an existing database.

BEGIN;

ALTER TABLE public.station_features
  ADD COLUMN IF NOT EXISTS estimated_uptime_hours numeric;

COMMIT;
