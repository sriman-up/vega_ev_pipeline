# ml/train_coldstart_ramp.py
"""
Train the shared ramp-stage cold-start model (months_since_active 0-5).

Both cold-start architecture families (Family A "single" and Family B
"per-calendar-month" — see ml/train_coldstart_stabilized_single.py /
ml/train_coldstart_stabilized_permonth.py) use this same model for the
first 6 months of a predicted trajectory, since the ramp design (pooled
data, months_since_active as a feature) is identical either way — training
it twice would just produce two near-identical models. See
ml/coldstart_common.py's docstring for why this exists at all instead of
reusing ml/train.py's general model.

Usage:
    python -m ml.train_coldstart_ramp
"""

import argparse
import logging
from datetime import datetime

from ml.coldstart_common import (
    COLDSTART_BOOL_FEATURES,
    COLDSTART_CATEGORICAL_FEATURES,
    COLDSTART_STATIC_FEATURES,
    COLDSTART_TARGET,
    RAMP_MAX,
    RAMP_MIN,
)
from ml.train import run_training, save_model, save_quantile_models

log = logging.getLogger(__name__)

MODEL_NAME = "coldstart_ramp"


def run_ramp_training(save: bool = True, log_to_db: bool = False):
    from ev_pipeline.db.db_manager import get_coldstart_training_matrix

    log.info("Loading ramp-stage training matrix (months_since_active %d-%d)...", RAMP_MIN, RAMP_MAX)
    df = get_coldstart_training_matrix(RAMP_MIN, RAMP_MAX)
    if df.empty:
        raise ValueError("Ramp-stage training matrix is empty.")

    metrics = run_training(
        df=df,
        target=COLDSTART_TARGET,
        numeric_features=COLDSTART_STATIC_FEATURES + ["months_since_active"],
        bool_features=COLDSTART_BOOL_FEATURES,
        categorical_features=COLDSTART_CATEGORICAL_FEATURES,
        run_shap=False,
        save=False,
        log_to_db=log_to_db,
        compute_thresholds=False,
        train_quantiles=True,
    )

    if save:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M")
        save_model(
            metrics["_model"], metrics["_feature_names"], version,
            categories=metrics["_categories"], model_name=MODEL_NAME,
        )
        save_quantile_models(metrics["_q10_model"], metrics["_q90_model"], version, model_name=MODEL_NAME)
        metrics["model_version"] = version

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Train the shared cold-start ramp-stage model")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    metrics = run_ramp_training(save=not args.no_save)
    print("\n=== Ramp Model Training Complete ===")
    for k in ("cv_mae", "cv_rmse", "cv_mape", "test_mae", "test_rmse", "test_mape", "model_version"):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")
