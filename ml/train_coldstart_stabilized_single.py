# ml/train_coldstart_stabilized_single.py
"""
Family A: one stabilized-stage cold-start model — ALL months_since_active
>= 6 (unbounded, same scope as Family B), with months_since_active as an
ordinary input feature.

No upper bound deliberately: an earlier version capped this at 11 to match
the validation report's old fixed 12-month horizon, but "stabilized" has no
real ceiling, and once the report started predicting each station's full
available history span (often well past 12 months), that cap meant
predicting months 12+ on input values the model never saw in training —
LightGBM can't differentiate months past whatever max it was fit on, so
every such month collapsed into the same leaf as month 11, producing wildly
wrong predictions (confirmed directly: one 24-month holdout station hit
739% MAPE before this fix).

Used for trajectory steps 6+ alongside the shared ramp model (steps 0-5,
ml/train_coldstart_ramp.py). Compare its holdout performance against
Family B (ml/train_coldstart_stabilized_permonth.py) via
ml/coldstart_validation.py --compare-families and keep whichever wins.

Usage:
    python -m ml.train_coldstart_stabilized_single
"""

import argparse
import logging
from datetime import datetime

from ml.coldstart_common import (
    COLDSTART_BOOL_FEATURES,
    COLDSTART_CATEGORICAL_FEATURES,
    COLDSTART_STATIC_FEATURES,
    COLDSTART_TARGET,
    STABILIZED_MIN,
)
from ml.train import run_training, save_model, save_quantile_models

log = logging.getLogger(__name__)

MODEL_NAME = "coldstart_stabilized_single"


def run_stabilized_single_training(save: bool = True, log_to_db: bool = False):
    from ev_pipeline.db.db_manager import get_coldstart_training_matrix

    log.info("Loading stabilized-stage training matrix (months_since_active >= %d, unbounded)...", STABILIZED_MIN)
    df = get_coldstart_training_matrix(STABILIZED_MIN, None)
    if df.empty:
        raise ValueError("Stabilized-stage training matrix is empty.")

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

    parser = argparse.ArgumentParser(description="Train Family A's single stabilized-stage cold-start model")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    metrics = run_stabilized_single_training(save=not args.no_save)
    print("\n=== Family A Stabilized Model Training Complete ===")
    for k in ("cv_mae", "cv_rmse", "cv_mape", "test_mae", "test_rmse", "test_mape", "model_version"):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")
