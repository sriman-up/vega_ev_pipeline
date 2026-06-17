# ml/train.py
"""
Train the EV station performance prediction model.

Two outputs from one training run:
  1. REGRESSION  — predict next_month_kwh (LightGBM, TimeSeriesSplit CV)
  2. LOW PERFORMER FLAG — threshold rule applied on top of regression output:
       is_low_performer = predicted_kwh < station's own 40th-percentile history
     This is deliberately a rule (not a second classifier) because with
     ~20 stations the data is too small for a reliable binary classifier.

Model choice rationale:
  - LightGBM handles missing values natively (important — many geo/weather
    features are NULL for some stations)
  - Trains in seconds on this dataset size
  - SHAP values work out of the box for feature importance
  - log(1+x) target transform stabilises variance across low/high stations

Cross-validation:
  TimeSeriesSplit (n_splits=4) — never trains on future data.
  Evaluation on the last 20% of rows (most recent months) per station.

Usage:
    python -m ev_pipeline.ml.train
    python -m ev_pipeline.ml.train --no-shap  # skip SHAP (faster)
"""

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── LightGBM ────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    log.error("lightgbm not installed. Run: pip install lightgbm")

# ── SHAP ────────────────────────────────────────────────────────────────────
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns used for training
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Consumption history
    "rolling_avg_3m_kwh", "rolling_avg_6m_kwh", "avg_kwh_units",
    "std_kwh_units", "kwh_growth_rate_overall", "months_active",
    "pct_months_zero_consumption", "kwh_coefficient_of_variation",
    "seasonal_summer_avg_kwh", "seasonal_winter_avg_kwh",
    # Capacity
    "contracted_load_kva", "total_charger_count", "total_power_kw",
    "charger_mix_ratio", "power_utilisation_pct",
    # Geo
    "dist_from_city_a_km", "dist_from_city_b_km", "dist_from_midpoint_km",
    "highway_position_ratio",
    # Competition
    "nearby_ev_stations_1km", "nearby_restaurants_1km", "nearby_hotels_1km",
    "nearby_petrol_pumps_1km", "competition_intensity", "amenity_score",
    # Calendar
    "month_of_year", "months_since_opening",
    # Weather
    "avg_temp_c", "max_temp_c", "total_rainfall_mm",
    "heatwave_days", "rainfall_days",
]

BOOL_FEATURES = [
    "has_attached_restaurant", "is_summer", "is_monsoon",
    "is_festival_month", "is_ramp_up_phase",
]

CATEGORICAL_FEATURES = [
    "direction_side",    # outgoing_from_hyderabad | outgoing_from_vijayawada | midpoint_zone
    "location_type",     # gas_station | hotel | rest_area | restaurant | ev_charging | ...
]

TARGET = "next_month_kwh"
LOG_TARGET = "log_next_month_kwh"   # training uses log(1+kwh)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Clean and encode the feature matrix.
    Returns (X, y, feature_names).
    """
    df = df.copy()
    df = df.sort_values(["unique_scno", "feature_month"]).reset_index(drop=True)

    # Bool → int
    for c in BOOL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # Categorical → LightGBM category dtype
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype("category")

    # Cyclical month encoding (sin/cos) — better than raw 1-12
    if "month_of_year" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)

    all_features = (
        [c for c in NUMERIC_FEATURES if c in df.columns and c != "month_of_year"]
        + (["month_sin", "month_cos"] if "month_sin" in df.columns else [])
        + [c for c in BOOL_FEATURES if c in df.columns]
        + [c for c in CATEGORICAL_FEATURES if c in df.columns]
    )

    X = df[all_features]
    y = np.log1p(df[TARGET].astype(float))   # log(1+x) transform

    log.info("Feature matrix: %d rows x %d features", len(X), len(all_features))
    log.info("Features used: %s", all_features)

    # Warn about low-variance features (may indicate missing data)
    low_var = [c for c in NUMERIC_FEATURES if c in X.columns
               and X[c].std(skipna=True) < 1e-6]
    if low_var:
        log.warning("Near-zero variance features (check for data issues): %s", low_var)

    return X, y, all_features


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Simple temporal train/test split.
    test set = most recent test_frac months (across all stations).
    This simulates the real deployment scenario: train on past, predict future.
    """
    df = df.sort_values("feature_month")
    cutoff_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:cutoff_idx]
    test  = df.iloc[cutoff_idx:]
    log.info("Train: %d rows up to %s | Test: %d rows from %s",
             len(train), train["feature_month"].max(),
             len(test),  test["feature_month"].min())
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM model
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective":        "regression_l1",   # MAE objective — robust to outliers
    "metric":           ["mae", "rmse"],
    "n_estimators":     400,
    "learning_rate":    0.05,
    "num_leaves":       15,                # small — prevents overfitting on tiny dataset
    "min_child_samples": 5,               # minimum samples per leaf
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,              # L1 regularisation
    "reg_lambda":       0.1,              # L2 regularisation
    "random_state":     42,
    "verbose":          -1,
    "n_jobs":           -1,
}


def _mape(y_true, y_pred) -> float:
    mask = y_true > 1
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
) -> Dict:
    """TimeSeriesSplit cross-validation. Returns dict of mean CV metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        preds_log = model.predict(X_val)
        preds     = np.expm1(np.maximum(preds_log, 0))
        actuals   = np.expm1(y_val.values)

        mae  = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mape = _mape(actuals, preds)

        maes.append(mae); rmses.append(rmse); mapes.append(mape)
        log.info("Fold %d: MAE=%.1f  RMSE=%.1f  MAPE=%.1f%%", fold+1, mae, rmse, mape)

    return {
        "cv_mae":  float(np.mean(maes)),
        "cv_rmse": float(np.mean(rmses)),
        "cv_mape": float(np.nanmean(mapes)),
    }


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMRegressor:
    """Train on all training data (no early stopping — use fixed n_estimators)."""
    params = {**LGBM_PARAMS, "n_estimators": 400}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    log.info("Final model trained on %d rows", len(X_train))
    return model


def evaluate_test(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    preds_log = model.predict(X_test)
    preds     = np.expm1(np.maximum(preds_log, 0))
    actuals   = np.expm1(y_test.values)
    return {
        "test_mae":  float(mean_absolute_error(actuals, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "test_mape": float(_mape(actuals, preds)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Optional[Dict]:
    if not _HAS_SHAP:
        log.warning("shap not installed — skipping SHAP analysis")
        return None

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    importance = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    log.info("Top 10 features by SHAP:\n%s", importance.head(10).to_string())

    if output_path and _HAS_MPL:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.3)))
        importance.head(20).sort_values().plot.barh(ax=ax, color="#3498db")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP) — Top 20")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("SHAP plot saved to %s", output_path)

    return importance.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Low performer threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_low_performer_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Per-station low performer threshold = 40th percentile of that station's
    own historical kWh.

    Using a per-station threshold rather than a global one accounts for the
    fact that a rural station doing 200 kWh/month is very different from a
    highway station doing 5000 kWh/month.
    """
    thresholds = {}
    for scno, grp in df.groupby("unique_scno"):
        vals = grp[TARGET].dropna()
        thresholds[scno] = float(vals.quantile(0.40)) if len(vals) >= 3 else float(vals.median())
    return thresholds


def flag_low_performers(
    predictions_df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    Add is_low_performer and low_performer_reason columns to predictions_df.
    predictions_df must have: unique_scno, predicted_kwh.
    """
    df = predictions_df.copy()
    df["low_performer_threshold"] = df["unique_scno"].map(thresholds)
    df["is_low_performer"] = df["predicted_kwh"] < df["low_performer_threshold"]
    df["low_performer_reason"] = df.apply(
        lambda r: (
            f"Predicted {r['predicted_kwh']:.0f} kWh is below this station's "
            f"40th-percentile history ({r['low_performer_threshold']:.0f} kWh)"
        ) if r["is_low_performer"] else None,
        axis=1,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, feature_names: List[str], version: str) -> Path:
    path = MODEL_DIR / f"lgbm_{version}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names,
                     "version": version, "trained_at": datetime.utcnow().isoformat()}, f)
    log.info("Model saved to %s", path)
    return path


def load_model(version: str = "latest") -> Tuple:
    if version == "latest":
        paths = sorted(MODEL_DIR.glob("lgbm_*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No model found in {MODEL_DIR}")
        path = paths[-1]
    else:
        path = MODEL_DIR / f"lgbm_{version}.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.info("Loaded model version %s from %s", obj["version"], path)
    return obj["model"], obj["feature_names"], obj["version"]


# ─────────────────────────────────────────────────────────────────────────────
# Main training run
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    df: Optional[pd.DataFrame] = None,
    run_shap: bool = True,
    save: bool = True,
    log_to_db: bool = True,
) -> Dict:
    """
    Full training pipeline. Returns metrics dict.

    Args:
        df:         Pre-loaded feature DataFrame. If None, loads from DB.
        run_shap:   Whether to compute SHAP (requires shap package).
        save:       Save model artifact to MODEL_DIR.
        log_to_db:  Write model run metrics to model_runs table.
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm is required. Run: pip install lightgbm")

    if df is None:
        from ev_pipeline.db.db_manager import get_feature_matrix
        log.info("Loading feature matrix from DB...")
        df = get_feature_matrix(min_months=3)

    if df.empty or TARGET not in df.columns:
        raise ValueError("Feature matrix is empty or missing target column.")

    df = df.dropna(subset=[TARGET]).copy()
    log.info("Training on %d rows from %d stations",
             len(df), df["unique_scno"].nunique())

    # ── Prepare ───────────────────────────────────────────────────────────────
    X, y, feature_names = prepare_features(df)
    train_df, test_df   = time_split(df)

    X_train, y_train, _ = prepare_features(train_df)
    X_test,  y_test,  _ = prepare_features(test_df)

    # Align columns (test may be missing some categories seen only in train)
    X_test = X_test.reindex(columns=X_train.columns)

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info("Running TimeSeriesSplit cross-validation (4 folds)...")
    cv_metrics = cross_validate(X_train, y_train, n_splits=4)
    log.info("CV results: MAE=%.1f  RMSE=%.1f  MAPE=%.1f%%",
             cv_metrics["cv_mae"], cv_metrics["cv_rmse"], cv_metrics["cv_mape"])

    # ── Final model ───────────────────────────────────────────────────────────
    model = train_final_model(X_train, y_train)
    test_metrics = evaluate_test(model, X_test, y_test)
    log.info("Test results: MAE=%.1f  RMSE=%.1f  MAPE=%.1f%%",
             test_metrics["test_mae"], test_metrics["test_rmse"], test_metrics["test_mape"])

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_importance = None
    if run_shap and _HAS_SHAP:
        shap_path = MODEL_DIR / "shap_importance.png"
        shap_importance = compute_shap(model, X_train, output_path=shap_path)

    # ── Low performer thresholds ──────────────────────────────────────────────
    thresholds = compute_low_performer_thresholds(df)
    thresh_path = MODEL_DIR / "low_performer_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Low performer thresholds saved to %s", thresh_path)

    # ── Version + save ────────────────────────────────────────────────────────
    version = datetime.utcnow().strftime("%Y%m%d_%H%M")
    if save:
        save_model(model, feature_names, version)

    # ── Log to DB ─────────────────────────────────────────────────────────────
    all_metrics = {
        **cv_metrics,
        **test_metrics,
        "model_version":    version,
        "n_stations":       int(df["unique_scno"].nunique()),
        "n_training_rows":  int(len(X_train)),
        "n_test_rows":      int(len(X_test)),
        "feature_importances": shap_importance or {
            f: float(v) for f, v in
            zip(feature_names, model.feature_importances_)
        },
        "hyperparameters": LGBM_PARAMS,
    }
    if log_to_db:
        try:
            from ev_pipeline.db.db_manager import insert_model_run
            run_id = insert_model_run(all_metrics)
            log.info("Model run logged to DB with id=%d", run_id)
        except Exception as e:
            log.warning("Could not log model run to DB: %s", e)

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    metrics = run_training(run_shap=not args.no_shap, save=not args.no_save)
    print("\n=== Training Complete ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")