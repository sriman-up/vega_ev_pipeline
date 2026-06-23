# api/main.py
"""
FastAPI service exposing ml/site_simulator.py over HTTP.

This exists because the "predict a new station" flow needs live LightGBM
inference plus the full feature-engineering pipeline (geo/competition/weather/
H3/zone derivation) — all Python, all already built in ev_pipeline/ and ml/.
Re-implementing that in the Next.js dashboard's language would mean
duplicating the feature engineering and risking train/serve skew. Map layers
and the vehicle/fleet-mix panel don't need this service — they're plain reads
of h3_coverage_zones / stations / zone_fleet_mix and should query Supabase
directly from the dashboard.

Run:
    uvicorn api.main:app --reload --port 8000

Then from the Next.js dashboard, call http://localhost:8000/predict and
.../scan (proxy through a Next.js API route in production so this service's
URL/port stays server-side, and restrict CORS_ALLOWED_ORIGINS below instead
of relying on the dashboard's own auth).
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from api.schemas import (
    HealthResponse,
    NewStationTrajectoryRequest,
    NewStationTrajectoryResponse,
    PredictRequest,
    PredictResponse,
    ScanRequest,
    ScanRow,
    StationTrajectoryRequest,
    StationTrajectoryResponse,
)
from ev_pipeline.db.db_manager import ping as db_ping
from ml.site_simulator import (
    COLD_START_CAVEAT,
    predict_existing_station_trajectory,
    predict_new_station,
    predict_new_station_trajectory,
    scan_configurations,
)
from ml.train import MODEL_DIR

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="EV Site Simulator API", version="1.0.0")

# Dev default is permissive; set CORS_ALLOWED_ORIGINS (comma-separated) in
# production to your dashboard's actual origin(s).
_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _chargers_from_payload(chargers: dict) -> dict:
    """JSON object keys are always strings — convert kW keys back to float."""
    return {float(kw): count for kw, count in chargers.items()}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        db_reachable=db_ping(),
        # /predict and /scan both resolve to the cold-start ramp model
        # (ml.site_simulator.predict_new_station/scan_configurations are
        # always months_since_active=0 by construction), not the general
        # lgbm_*.pkl model — check the dependency they actually have.
        model_loaded=any(MODEL_DIR.glob("coldstart_ramp_*.pkl")),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predict_new_station(
            lat=req.lat,
            lon=req.lon,
            chargers=_chargers_from_payload(req.chargers),
            prediction_month=req.prediction_month,
            contracted_load_kva=req.contracted_load_kva,
            has_attached_restaurant=req.has_attached_restaurant,
            location_type=req.location_type,
            direction_side=req.direction_side,
            fetch_competition=req.fetch_competition,
            model_version=req.model_version,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"No trained model available: {e}. Run `python -m ml.train` first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("predict_new_station failed")
        raise HTTPException(status_code=500, detail="Prediction failed — see server logs.")
    return PredictResponse(**result)


@app.post("/scan", response_model=list[ScanRow])
def scan(req: ScanRequest):
    try:
        df = scan_configurations(
            lat=req.lat,
            lon=req.lon,
            charger_grid=[_chargers_from_payload(c) for c in req.charger_grid],
            prediction_month=req.prediction_month,
            restaurant_options=tuple(req.restaurant_options) if req.restaurant_options else (None,),
            location_type_options=tuple(req.location_type_options) if req.location_type_options else (None,),
            contracted_load_options=tuple(req.contracted_load_options) if req.contracted_load_options else (None,),
            scan_direction=req.scan_direction,
            fetch_competition=req.fetch_competition,
            model_version=req.model_version,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"No trained model available: {e}. Run `python -m ml.train` first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("scan_configurations failed")
        raise HTTPException(status_code=500, detail="Scan failed — see server logs.")
    return df.to_dict("records")


@app.post("/predict/trajectory", response_model=NewStationTrajectoryResponse)
def predict_trajectory(req: NewStationTrajectoryRequest):
    """Ramp-up curve for a hypothetical new station — same inputs as
    /predict, but returns one point per month instead of a single value."""
    try:
        trajectory = predict_new_station_trajectory(
            lat=req.lat,
            lon=req.lon,
            chargers=_chargers_from_payload(req.chargers),
            start_month=req.start_month,
            horizon_months=req.horizon_months,
            contracted_load_kva=req.contracted_load_kva,
            has_attached_restaurant=req.has_attached_restaurant,
            location_type=req.location_type,
            direction_side=req.direction_side,
            fetch_competition=req.fetch_competition,
            family=req.family,
            blend_peer_benchmark=req.blend_peer_benchmark,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"No trained model available: {e}. Run `python -m ml.train_coldstart_ramp` first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("predict_new_station_trajectory failed")
        raise HTTPException(status_code=500, detail="Trajectory prediction failed — see server logs.")
    return NewStationTrajectoryResponse(trajectory=trajectory, caveats=[COLD_START_CAVEAT])


@app.post("/station/trajectory", response_model=StationTrajectoryResponse)
def station_trajectory(req: StationTrajectoryRequest):
    """'Test prediction' for a REAL existing station — predicts its
    cold-start ramp-up curve using the production-saved models and compares
    it month-by-month against that station's own real billing history."""
    try:
        result = predict_existing_station_trajectory(
            unique_scno=req.unique_scno,
            family=req.family,
            horizon_months=req.horizon_months,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"No trained model available: {e}. Run `python -m ml.train_coldstart_ramp` first.")
    except Exception:
        log.exception("predict_existing_station_trajectory failed")
        raise HTTPException(status_code=500, detail="Trajectory prediction failed — see server logs.")
    return StationTrajectoryResponse(**result)
