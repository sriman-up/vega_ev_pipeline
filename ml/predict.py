# ml/predict.py
"""
Run inference with the trained model and write predictions to DB.

Each month after the bill scrape completes:
  1. Feature vectors for all stations are rebuilt (pipeline --mode monthly)
  2. This script loads the latest model + thresholds, predicts next month's kWh,
     flags low performers, and writes to model_predictions + updates stations table.

Also provides prediction intervals via LightGBM quantile regression
(10th / 90th percentile models trained alongside the main model).

Usage:
    python -m ml.predict
    python -m ml.predict --version 20260612_0800
"""

import argparse
import json
import logging
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml.train import (
    MODEL_DIR,
    TARGET,
    compute_low_performer_thresholds,
    flag_low_performers,
    load_model,
    prepare_features,
    run_training,
)

log = logging.getLogger(__name__)

THRESHOLD_FILE = MODEL_DIR / "low_performer_thresholds.json"
QUANTILE_LOWER_FILE = MODEL_DIR / "lgbm_q10_latest.pkl"
QUANTILE_UPPER_FILE = MODEL_DIR / "lgbm_q90_latest.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Quantile models for prediction intervals
# ─────────────────────────────────────────────────────────────────────────────

def _train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Train 10th and 90th percentile models for confidence intervals."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None

    from ml.train import LGBM_PARAMS

    q10_params = {**LGBM_PARAMS, "objective": "quantile", "alpha": 0.10}
    q90_params = {**LGBM_PARAMS, "objective": "quantile", "alpha": 0.90}

    q10 = lgb.LGBMRegressor(**q10_params)
    q90 = lgb.LGBMRegressor(**q90_params)
    q10.fit(X_train, y_train)
    q90.fit(X_train, y_train)
    log.info("Quantile models (p10/p90) trained")
    return q10, q90


def _save_quantile_models(q10, q90):
    with open(QUANTILE_LOWER_FILE, "wb") as f:
        pickle.dump(q10, f)
    with open(QUANTILE_UPPER_FILE, "wb") as f:
        pickle.dump(q90, f)


def _load_quantile_models():
    try:
        with open(QUANTILE_LOWER_FILE, "rb") as f:
            q10 = pickle.load(f)
        with open(QUANTILE_UPPER_FILE, "rb") as f:
            q90 = pickle.load(f)
        return q10, q90
    except FileNotFoundError:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────────────────────────────────────

def run_predictions(
    model_version: str = "latest",
    retrain_if_stale: bool = True,
) -> pd.DataFrame:
    """
    Load the latest feature vectors from DB, run inference, write predictions.

    Args:
        model_version:      Model version to load ('latest' or timestamp string).
        retrain_if_stale:   If True and no model exists, runs training first.

    Returns:
        DataFrame of predictions with columns:
            unique_scno, prediction_month, predicted_kwh,
            predicted_kwh_lower, predicted_kwh_upper,
            is_low_performer, low_performer_reason, model_version
    """
    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, feature_names, categories, version = load_model(model_version)
    except FileNotFoundError:
        if retrain_if_stale:
            log.info("No trained model found — running training first...")
            metrics = run_training()
            version = metrics["model_version"]
            model, feature_names, categories, version = load_model(version)
        else:
            raise

    # ── Load latest feature vectors ───────────────────────────────────────────
    from ev_pipeline.db.db_manager import get_latest_feature_vectors, get_feature_matrix
    log.info("Loading latest feature vectors for inference...")
    X_latest = get_latest_feature_vectors()

    if X_latest.empty:
        log.error("No feature vectors found. Run pipeline --mode features first.")
        return pd.DataFrame()

    log.info("Running inference for %d stations", len(X_latest))

    # ── Prepare features (same pipeline as training) ─────────────────────────
    # For inference we don't have a target — add a dummy column so prepare_features works
    X_latest_copy = X_latest.copy()
    X_latest_copy[TARGET] = 0   # dummy

    X_inf, _, _ = prepare_features(X_latest_copy, categories=categories)
    # Align to training feature set
    X_inf = X_inf.reindex(columns=feature_names)

    # ── Predict (log space, then expm1 back to kWh) ───────────────────────────
    preds_log = model.predict(X_inf)
    preds_kwh = np.expm1(np.maximum(preds_log, 0))

    # ── Prediction intervals from quantile models ─────────────────────────────
    q10_model, q90_model = _load_quantile_models()
    if q10_model and q90_model:
        lower = np.expm1(np.maximum(q10_model.predict(X_inf), 0))
        upper = np.expm1(np.maximum(q90_model.predict(X_inf), 0))
    else:
        # Fallback: ±25% of prediction as rough CI
        lower = preds_kwh * 0.75
        upper = preds_kwh * 1.25
        log.info("Quantile models not found — using ±25%% fallback CI")

    # ── Determine prediction month (next month from feature_month) ────────────
    def next_month(dt_str) -> str:
        d = pd.to_datetime(dt_str)
        if d.month == 12:
            return f"{d.year + 1}-01-01"
        return f"{d.year}-{d.month + 1:02d}-01"

    predictions = pd.DataFrame({
        "unique_scno":        X_latest["unique_scno"],
        "station_id":         X_latest["station_id"],
        "feature_month":      X_latest["feature_month"],
        "prediction_month":   X_latest["feature_month"].apply(next_month),
        "predicted_kwh":      np.round(preds_kwh, 2),
        "predicted_kwh_lower": np.round(lower, 2),
        "predicted_kwh_upper": np.round(upper, 2),
        "model_version":      version,
        "model_trained_on":   date.today().isoformat(),
    })

    # ── Low performer flagging ────────────────────────────────────────────────
    history_df = get_feature_matrix(min_months=1)
    if not history_df.empty:
        thresholds = compute_low_performer_thresholds(history_df)
    elif THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE) as f:
            thresholds = json.load(f)
        log.info("Loaded low performer thresholds from file")
    else:
        log.warning("No threshold data available — low performer flag will be False for all")
        thresholds = {}

    predictions = flag_low_performers(predictions, thresholds)

    log.info(
        "Predictions complete: %d stations, %d flagged as low performers",
        len(predictions),
        predictions["is_low_performer"].sum(),
    )

    # ── Write to DB ───────────────────────────────────────────────────────────
    _write_predictions_to_db(predictions)

    return predictions


def _write_predictions_to_db(predictions: pd.DataFrame):
    from ev_pipeline.db.db_manager import upsert_prediction, upsert_station

    for _, row in predictions.iterrows():
        # Write to model_predictions
        pred_record = {
            "station_id":             int(row["station_id"]),
            "unique_scno":            row["unique_scno"],
            "prediction_month":       row["prediction_month"],
            "predicted_kwh":          float(row["predicted_kwh"]),
            "predicted_kwh_lower":    float(row["predicted_kwh_lower"]),
            "predicted_kwh_upper":    float(row["predicted_kwh_upper"]),
            "is_low_performer":       bool(row["is_low_performer"]),
            "low_performer_threshold": float(row.get("low_performer_threshold") or 0),
            "low_performer_reason":   row.get("low_performer_reason"),
            "model_version":          row["model_version"],
            "model_trained_on":       row["model_trained_on"],
            "feature_month":          row["feature_month"],
        }
        upsert_prediction(pred_record)

        # Denorm latest prediction onto stations table for dashboard
        upsert_station({
            "unique_scno":                row["unique_scno"],
            "predicted_next_month_kwh":   float(row["predicted_kwh"]),
            "predicted_at":               datetime.utcnow().isoformat(),
            "is_low_performer":           bool(row["is_low_performer"]),
        })

    log.info("Wrote %d predictions to DB", len(predictions))


# ─────────────────────────────────────────────────────────────────────────────
# Backfill actuals — run after each monthly bill scrape to fill actual_kwh
# ─────────────────────────────────────────────────────────────────────────────

def backfill_actuals():
    """
    After bills arrive, match model_predictions rows to actual monthly_bills
    and compute abs_error / pct_error.
    Run this in pipeline monthly_update() after upsert_monthly_bill().
    """
    from ev_pipeline.db.db_manager import get_conn
    import psycopg2.extras

    sql_update = """
        UPDATE model_predictions mp
        SET
            actual_kwh  = mb.kwh_units,
            abs_error   = ABS(mp.predicted_kwh - mb.kwh_units),
            pct_error   = CASE WHEN mb.kwh_units > 0
                               THEN ROUND(ABS(mp.predicted_kwh - mb.kwh_units)
                                    / mb.kwh_units * 100, 4)
                               ELSE NULL END
        FROM monthly_bills mb
        WHERE mp.unique_scno     = mb.unique_scno
          AND mp.prediction_month = mb.bill_month
          AND mp.actual_kwh IS NULL
          AND mb.kwh_units IS NOT NULL
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_update)
            log.info("Backfilled actuals for %d prediction rows", cur.rowcount)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run EV station kWh predictions")
    parser.add_argument("--version", default="latest",
                        help="Model version to use (default: latest)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Fail instead of retraining if no model found")
    parser.add_argument("--backfill-actuals", action="store_true",
                        help="Only backfill actual kWh into existing predictions")
    args = parser.parse_args()

    if args.backfill_actuals:
        backfill_actuals()
    else:
        preds = run_predictions(
            model_version=args.version,
            retrain_if_stale=not args.no_retrain,
        )
        if not preds.empty:
            print("\n=== Predictions ===")
            print(preds[["unique_scno", "prediction_month", "predicted_kwh",
                          "predicted_kwh_lower", "predicted_kwh_upper",
                          "is_low_performer"]].to_string(index=False))