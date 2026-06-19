-- ev_pipeline/db/migrations/004_ml_tables.sql
--
-- ml/train.py and ml/predict.py import db_manager functions
-- (get_feature_matrix, get_latest_feature_vectors, insert_model_run,
-- upsert_prediction) and write to a stations.is_low_performer column —
-- none of which existed yet. This migration adds the backing tables/column.
--
-- Idempotent — safe to re-run against an existing database.

BEGIN;

ALTER TABLE public.stations
  ADD COLUMN IF NOT EXISTS is_low_performer boolean;

CREATE TABLE IF NOT EXISTS public.model_runs (
  id serial PRIMARY KEY,
  model_version character varying NOT NULL,
  trained_at timestamp without time zone DEFAULT now(),
  n_stations integer,
  n_training_rows integer,
  n_test_rows integer,
  cv_mae numeric,
  cv_rmse numeric,
  cv_mape numeric,
  test_mae numeric,
  test_rmse numeric,
  test_mape numeric,
  feature_importances jsonb,
  hyperparameters jsonb,
  CONSTRAINT model_runs_version_unique UNIQUE (model_version)
);

CREATE TABLE IF NOT EXISTS public.model_predictions (
  id serial PRIMARY KEY,
  station_id integer NOT NULL REFERENCES public.stations(id),
  unique_scno character varying NOT NULL,
  feature_month date NOT NULL,
  prediction_month date NOT NULL,
  predicted_kwh numeric,
  predicted_kwh_lower numeric,
  predicted_kwh_upper numeric,
  is_low_performer boolean,
  low_performer_threshold numeric,
  low_performer_reason text,
  model_version character varying,
  model_trained_on date,
  actual_kwh numeric,
  abs_error numeric,
  pct_error numeric,
  predicted_at timestamp without time zone DEFAULT now(),
  CONSTRAINT model_predictions_unique UNIQUE (unique_scno, prediction_month)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_station ON public.model_predictions (station_id);

COMMIT;
