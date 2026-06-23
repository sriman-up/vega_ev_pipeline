# ml/train_coldstart_stabilized_permonth.py
"""
Family B: 12 stabilized-stage cold-start models, one per calendar month
(Jan-Dec), each trained on ALL months_since_active >= 6 rows that fall in
that calendar month — no upper bound on months_since_active.

Why calendar month instead of ramp-position month (the more obvious "one
model per months_since_active value" reading of "split models per month"):
checked both directly against the DB. Keying by ramp position and capping
at 11 (to match the trajectory horizon) only uses ~1,653 rows, 245-301 per
bucket, and discards every row from stations that have been running longer
than 11 months. Keying by calendar month uses every stabilized-phase row
regardless of station age — 3,877 rows, 162-483 per bucket (2.3x more
data) — and growth has already plateaued by month 6 (see
ml/coldstart_common.py's sibling docstrings / the validation report), so
seasonality is plausibly a stronger differentiator within the stabilized
phase than exact ramp position anyway. months_since_active is kept as an
ordinary feature in each calendar-month model since it's no longer fixed by
the bucketing; month_of_year itself is dropped (it IS the bucketing key,
constant within each model).

Pickles all 12 models into one file (keyed 1-12), not 12 separate files —
load_coldstart_permonth() mirrors ml.train.load_model()'s "latest" glob
resolution.

Usage:
    python -m ml.train_coldstart_stabilized_permonth
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ml.coldstart_common import (
    COLDSTART_BOOL_FEATURES,
    COLDSTART_CATEGORICAL_FEATURES,
    COLDSTART_STATIC_FEATURES,
    COLDSTART_TARGET,
    STABILIZED_MIN,
)
from ml.train import MODEL_DIR, run_training

log = logging.getLogger(__name__)

MODEL_NAME = "coldstart_stabilized_permonth"
_NUMERIC_FEATURES = [c for c in COLDSTART_STATIC_FEATURES if c != "month_of_year"] + ["months_since_active"]


def run_permonth_training(save: bool = True) -> Dict[int, Dict[str, Any]]:
    from ev_pipeline.db.db_manager import get_coldstart_training_matrix

    log.info("Loading all stabilized-phase rows (months_since_active >= %d, unbounded)...", STABILIZED_MIN)
    df = get_coldstart_training_matrix(STABILIZED_MIN, None)
    if df.empty:
        raise ValueError("Stabilized-phase training matrix is empty.")

    models: Dict[int, Dict[str, Any]] = {}
    for cal_month in range(1, 13):
        sub = df[df["month_of_year"] == cal_month]
        if sub.empty:
            log.warning("No rows for calendar month %d — skipping", cal_month)
            continue
        log.info("Training calendar-month %d model on %d rows...", cal_month, len(sub))
        metrics = run_training(
            df=sub,
            target=COLDSTART_TARGET,
            numeric_features=_NUMERIC_FEATURES,
            bool_features=COLDSTART_BOOL_FEATURES,
            categorical_features=COLDSTART_CATEGORICAL_FEATURES,
            run_shap=False,
            save=False,
            log_to_db=False,
            compute_thresholds=False,
            train_quantiles=True,
        )
        models[cal_month] = {
            "model": metrics["_model"],
            "feature_names": metrics["_feature_names"],
            "categories": metrics["_categories"],
            "q10_model": metrics["_q10_model"],
            "q90_model": metrics["_q90_model"],
            "cv_mae": metrics.get("cv_mae"),
            "test_mae": metrics.get("test_mae"),
        }

    if save:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M")
        save_coldstart_permonth(models, version)

    return models


def save_coldstart_permonth(models: Dict[int, Dict[str, Any]], version: str) -> Path:
    path = MODEL_DIR / f"{MODEL_NAME}_{version}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"months": models, "version": version, "trained_at": datetime.utcnow().isoformat()}, f)
    log.info("Per-calendar-month models saved to %s", path)
    return path


def load_coldstart_permonth(version: str = "latest") -> Dict[int, Dict[str, Any]]:
    if version == "latest":
        paths = sorted(MODEL_DIR.glob(f"{MODEL_NAME}_*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No {MODEL_NAME} model found in {MODEL_DIR}")
        path = paths[-1]
    else:
        path = MODEL_DIR / f"{MODEL_NAME}_{version}.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    log.info("Loaded per-calendar-month models version %s from %s", obj["version"], path)
    return obj["months"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Train Family B's 12 per-calendar-month stabilized cold-start models")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    models = run_permonth_training(save=not args.no_save)
    print("\n=== Family B Per-Calendar-Month Training Complete ===")
    for cal_month, m in sorted(models.items()):
        print(f"  month {cal_month:2d}: cv_mae={m['cv_mae']:.1f}  test_mae={m['test_mae']:.1f}")
