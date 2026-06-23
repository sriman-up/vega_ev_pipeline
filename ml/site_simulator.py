# ml/site_simulator.py
"""
"What if I built a station here?" predictor for sites that don't exist yet.

ml/train.py's general model is fundamentally autoregressive: it has never
once seen a row with no consumption-momentum features at all (every
station's own first feature row already has that month's own bill folded
in), so feeding it a hypothetical site with zero history produces an
uncalibrated, near-constant prediction regardless of capacity/geo/calendar
inputs — confirmed directly against this DB. This module instead uses
dedicated cold-start models (ml/train_coldstart_ramp.py +
ml/train_coldstart_stabilized_single.py / _permonth.py), trained
specifically on rows with no consumption-history features in the input at
all, with the station's position in its own ramp-up curve
(months_since_active, re-zeroed at first real usage — see
ml/coldstart_common.py) as an explicit input instead. A separate,
clearly-labeled peer-group benchmark is still reported as an independent
sanity check.

What a candidate site needs, split by source
──────────────────────────────────────────────
REQUIRED INPUT (only thing you must supply):
  latitude, longitude, chargers ({kw: count, ...})

DERIVED AUTOMATICALLY, free (geometry only):
  h3_res5/6/7, nearest_city / dist_to_city_km / zone_band, highway position
  (dist_from_city_a/b_km, direction_side, travel_time_*) if the site is near
  a tracked highway pair, total_charger_count / total_power_kw /
  charger_mix_ratio, calendar features for the prediction month,
  months_since_opening=0 (it's a new station).

DERIVED VIA LIVE LOOKUPS, optional (real-world data, costs API quota):
  competition counts + rating + location_type + has_attached_restaurant
  (Google Places nearby search at the candidate lat/lon — reuses
  scrapers.places_scraper.enrich_station as-is, since it already works off
  bare coordinates), weather climatology for the prediction month (Open-Meteo
  archive; falls back to the same month last year if the prediction month
  is in the future and the archive doesn't have it yet).

ITERABLE / what-if (the actual point of this module — pass several values
and compare):
  chargers (the headline ask — scan charger mixes/counts),
  has_attached_restaurant (True/False override),
  direction_side (override — answers "which side of the highway?";
  highway_direction_options() gives you the two real values for a site),
  location_type (override), contracted_load_kva (override).

Usage:
    from ml.site_simulator import predict_new_station, scan_configurations

    result = predict_new_station(
        lat=17.40, lon=78.55, chargers={60: 2, 120: 1},
        prediction_month="2026-09",
    )

    df = scan_configurations(
        lat=17.40, lon=78.55, prediction_month="2026-09",
        charger_grid=[{30: 4}, {60: 2, 120: 1}, {150: 1, 60: 2}],
        restaurant_options=(True, False),
        scan_direction=True,
    )

CLI:
    python -m ml.site_simulator --lat 17.40 --lon 78.55 --chargers 60:2,120:1
    python -m ml.site_simulator --lat 17.40 --lon 78.55 --scan \
        --chargers-grid "30:4|60:2,120:1|150:1,60:2" --scan-restaurant --scan-direction
"""

import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ev_pipeline.db.db_manager import get_all_stations
from ev_pipeline.features.city_zones import compute_city_zone
from ev_pipeline.features.feature_builder import build_feature_row
from ev_pipeline.features.geo_features import compute_geo_features, nearest_city
from ev_pipeline.features.spatial_h3 import assign_h3_indices
from ev_pipeline.scrapers.weather_scraper import fetch_monthly_weather
from ml.coldstart_common import (
    COLDSTART_BOOL_FEATURES,
    COLDSTART_CATEGORICAL_FEATURES,
    COLDSTART_TARGET,
    PERMONTH_NUMERIC_FEATURES,
    RAMP_MAX,
    RAMP_NUMERIC_FEATURES,
    actual_series,
    blend_trajectories,
    compare_trajectory_to_actuals,
    month_diff,
)
from ml.train import load_model, load_quantile_models, prepare_features
from ml.train_coldstart_stabilized_permonth import load_coldstart_permonth

log = logging.getLogger(__name__)

SITESIM_SCNO = "SITESIM"

COLD_START_CAVEAT = (
    "No billing history exists for this site, so this prediction uses a "
    "dedicated cold-start model trained only on capacity/geo/competition/"
    "calendar signals and the station's position in its own ramp-up curve — "
    "not the general autoregressive model (which requires real consumption "
    "history). Treat predicted_kwh as a model-based estimate for a "
    "comparable brand-new site, not a guarantee."
)


_STANDARD_KW_COLS = {
    30: "charger_30kw_count", 60: "charger_60kw_count",
    120: "charger_120kw_count", 150: "charger_150kw_count",
}


# ─────────────────────────────────────────────────────────────────────────────
# Charger config <-> DB columns
# ─────────────────────────────────────────────────────────────────────────────

def chargers_to_columns(chargers: Dict[float, int]) -> Dict[str, Any]:
    """{30: 2, 240: 1} -> {charger_30kw_count: 2, ..., charger_other_json: {'240kw': 1}, ...}"""
    cols = {col: 0 for col in _STANDARD_KW_COLS.values()}
    other: Dict[str, int] = {}
    total = 0
    for kw, count in chargers.items():
        if not count:
            continue
        total += count
        kw_int = int(round(float(kw)))
        col = _STANDARD_KW_COLS.get(kw_int)
        if col:
            cols[col] += count
        else:
            other[f"{kw_int}kw"] = other.get(f"{kw_int}kw", 0) + count
    cols["total_charger_count"] = total
    cols["charger_other_json"] = other or None
    return cols


def format_chargers(chargers: Dict[float, int]) -> str:
    return ", ".join(f"{int(kw)}kW x{count}" for kw, count in sorted(chargers.items()) if count) or "none"


def total_power_kw(chargers: Dict[float, int]) -> float:
    return round(sum(float(kw) * count for kw, count in chargers.items()), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Month helpers — mirrors predict.py's next_month() inverse, since
# build_feature_row computes calendar features at feature_month while
# next_month_kwh (here, our prediction target) corresponds to the month after.
# ─────────────────────────────────────────────────────────────────────────────

def normalize_month(s: str) -> str:
    """'2026-9' / '2026-09' / '2026-09-01' -> '2026-09-01'"""
    parts = s.split("-")
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-01"


def default_prediction_month() -> str:
    now = datetime.utcnow()
    y, m = (now.year, now.month + 1) if now.month < 12 else (now.year + 1, 1)
    return f"{y}-{m:02d}-01"


def prev_month(month_str: str) -> str:
    y, m = int(month_str[:4]), int(month_str[5:7])
    y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return f"{y}-{m:02d}-01"


def add_months(month_str: str, n: int) -> str:
    """'2024-05-01' + 3 -> '2024-08-01'. n must be >= 0."""
    y, m = int(month_str[:4]), int(month_str[5:7])
    total = (y * 12 + (m - 1)) + n
    y, m = total // 12, total % 12 + 1
    return f"{y:04d}-{m:02d}-01"


# ─────────────────────────────────────────────────────────────────────────────
# Candidate station construction
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_station(
    lat: float,
    lon: float,
    chargers: Dict[float, int],
    feature_month: str,
    contracted_load_kva: Optional[float] = None,
    meter_phase: int = 3,
    fetch_competition: bool = True,
) -> Dict[str, Any]:
    """
    Build the base station dict: charger config + geometry-derived fields +
    (optionally) live Places competition data. Overrides for has_attached_restaurant
    / location_type / direction_side are applied by the caller afterwards.
    """
    station: Dict[str, Any] = {
        "unique_scno": SITESIM_SCNO,
        "latitude": lat,
        "longitude": lon,
        "meter_phase": meter_phase,
        **chargers_to_columns(chargers),
        # supply_date = feature_month -> months_since_opening=0, is_ramp_up_phase=True
        "supply_date": feature_month,
    }
    station["contracted_load_kva"] = (
        contracted_load_kva if contracted_load_kva is not None else total_power_kw(chargers)
    )

    if fetch_competition:
        try:
            from ev_pipeline.scrapers.places_scraper import enrich_station
            places_updates = enrich_station(station, fetch_charger_info=False)
            places_updates.pop("_nearby_ev_names", None)
            station.update(places_updates)
        except Exception as e:
            log.warning("Places competition lookup skipped (%s) — those features will be missing", e)

    station.update(compute_geo_features(lat, lon))
    station.update(nearest_city(lat, lon))
    station.update(assign_h3_indices(lat, lon))
    station.update(compute_city_zone(lat, lon))
    return station


def fetch_weather_with_fallback(
    lat: float, lon: float, feature_month: str, cache_scno: Optional[str] = None,
) -> Tuple[Optional[Dict], bool]:
    """
    Weather for feature_month. If feature_month is in the future (beyond the
    Open-Meteo archive's ~5 day lag), falls back to the same calendar month
    last year as a climatology proxy. Returns (weather_dict_or_None, is_proxy).

    cache_scno: when set to a REAL station's unique_scno (e.g.
    ml.coldstart_validation.py predicting an existing station's trajectory),
    checks db_manager.get_monthly_weather() before hitting Open-Meteo, and
    writes any freshly-fetched result back into that cache — that table is
    usually already populated for these exact (station, month) pairs from
    the original station_features backfill, so this turns most of a
    validation run's weather calls into cache hits instead of live API
    calls. Left None (default) for genuinely new/hypothetical sites
    (predict_new_station / scan_configurations) — those don't have a real
    scno, and caching by the shared SITESIM_SCNO placeholder would collide
    across different candidate lat/lons, returning the wrong site's weather.
    """
    from ev_pipeline.db.db_manager import get_monthly_weather, upsert_monthly_weather

    y, m = int(feature_month[:4]), int(feature_month[5:7])

    def _get(year: int, month: int) -> Optional[Dict]:
        month_str = f"{year:04d}-{month:02d}-01"
        if cache_scno:
            cached = get_monthly_weather(cache_scno, month_str)
            if cached:
                return cached
        weather = fetch_monthly_weather(SITESIM_SCNO, lat, lon, year, month)
        if weather and cache_scno:
            upsert_monthly_weather({**weather, "unique_scno": cache_scno})
        return weather

    weather = _get(y, m)
    if weather:
        return weather, False
    weather = _get(y - 1, m)
    return weather, weather is not None


def _alias_months_since_active(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    The cold-start models were trained on months_since_active — each
    station's own ramp clock re-zeroed at its first non-zero bill, not
    supply_date (see db_manager.get_coldstart_training_matrix()'s docstring:
    74% of real stations have a leading run of zero-kWh "not started yet"
    bills before supply_date's clock means anything). A simulated brand-new
    station has no such commissioning gap by construction — it's "active"
    from month 0 — so build_feature_row's months_since_opening (relative to
    supply_date) and months_since_active are the same number here; this just
    copies it under the name the cold-start models expect as an input column.
    """
    row["months_since_active"] = row.get("months_since_opening")
    return row


def build_monthly_feature_rows(
    station: Dict[str, Any],
    start_feature_month: str,
    horizon_months: int = 12,
    cache_scno: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    One cold-start feature row per month, from start_feature_month for
    horizon_months — calendar/weather/months_since_active vary per month
    while geo/capacity/competition stay fixed at the station's own values.

    history=[] for every month: this is the "if I forecasted blind every
    month from day one, never incorporating real consumption" curve — NOT a
    rolling forecast that feeds in actuals as they arrive. Consistent with
    this module's cold-start design (see module docstring).

    cache_scno: passed straight through to fetch_weather_with_fallback() —
    set this to the station's real unique_scno when predicting an EXISTING
    station's trajectory (ml.coldstart_validation.py) to hit the weather
    cache instead of re-fetching from Open-Meteo every run.
    """
    base = dict(station)
    base["supply_date"] = start_feature_month  # months_since_opening increments naturally
    rows = []
    for m in range(horizon_months):
        feature_month = add_months(start_feature_month, m)
        weather = None
        if base.get("latitude") and base.get("longitude"):
            weather, _ = fetch_weather_with_fallback(base["latitude"], base["longitude"], feature_month, cache_scno=cache_scno)
        row = build_feature_row(base, history=[], weather=weather, feature_month=feature_month)
        rows.append(_alias_months_since_active(row))
    return rows


def apply_overrides(
    station: Dict[str, Any],
    has_attached_restaurant: Optional[bool] = None,
    location_type: Optional[str] = None,
    direction_side: Optional[str] = None,
) -> Dict[str, Any]:
    station = dict(station)
    if has_attached_restaurant is not None:
        station["has_attached_restaurant"] = has_attached_restaurant
    if location_type is not None:
        station["location_type"] = location_type
    if direction_side is not None:
        station["direction_side"] = direction_side
    return station


def highway_direction_options(station: Dict[str, Any]) -> List[Optional[str]]:
    """
    The two real direction_side values for whichever highway pair this site
    matched (answers "predict for both sides of the highway"). [None] if the
    site isn't within range of any tracked highway pair.
    """
    city_a, city_b = station.get("nearest_city_a"), station.get("nearest_city_b")
    if not city_a or not city_b:
        return [None]
    return [
        f"outgoing_from_{city_a.lower().replace(' ', '_')}",
        f"outgoing_from_{city_b.lower().replace(' ', '_')}",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Peer benchmark — independent of the model, never fed in as a feature
# ─────────────────────────────────────────────────────────────────────────────

def peer_benchmark_kwh(station: Dict[str, Any], all_stations: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Average avg_monthly_kwh_lifetime among existing stations in the same
    zone_band with comparable total_power_kw (0.5x-2x the candidate's).
    Falls back to zone_band-only matching if too few stations satisfy both.
    This is a sanity-check baseline, kept separate from the model's inputs.
    """
    stations = all_stations if all_stations is not None else get_all_stations()
    zone_band = station.get("zone_band")
    power = total_power_kw(
        {30: station.get("charger_30kw_count") or 0, 60: station.get("charger_60kw_count") or 0,
         120: station.get("charger_120kw_count") or 0, 150: station.get("charger_150kw_count") or 0}
    ) if "charger_30kw_count" in station else (station.get("total_power_kw") or 0)

    def has_kwh(s):
        return s.get("avg_monthly_kwh_lifetime") is not None and s.get("zone_band") == zone_band

    same_zone = [s for s in stations if has_kwh(s)]
    if power:
        narrowed = [s for s in same_zone if power * 0.5 <= (s.get("total_power_kw") or 0) <= power * 2.0]
    else:
        narrowed = same_zone

    peers = narrowed if len(narrowed) >= 3 else same_zone
    if not peers:
        return {"peer_avg_kwh": None, "peer_group_size": 0, "peer_group_criteria": "no comparable stations found"}

    vals = [float(s["avg_monthly_kwh_lifetime"]) for s in peers]
    criteria = (
        f"zone_band={zone_band}, total_power_kw in [{power * 0.5:.0f}, {power * 2.0:.0f}]"
        if peers is narrowed else f"zone_band={zone_band} (power-band relaxed — too few exact matches)"
    )
    return {
        "peer_avg_kwh": round(sum(vals) / len(vals), 1),
        "peer_group_size": len(peers),
        "peer_group_criteria": criteria,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_coldstart_models(family: str = "single") -> Dict[str, Any]:
    """
    Load the production-saved cold-start models ONCE per trajectory call
    (not once per month — predict_new_station_trajectory() /
    predict_existing_station_trajectory() can span 30+ months for stations
    with long real histories, and load_model()/load_coldstart_permonth()
    each do a disk read + unpickle). family: 'single', 'permonth', or
    'both' (needed by callers that want to compare both, e.g. a "test
    prediction" UI showing both lines on one chart).

    Returns {"ramp": {...}, "single": {...}?, "permonth": {cal_month: {...}}?}
    where each {...} is {"model", "feature_names", "categories", "q10_model",
    "q90_model"} — the latter two are None if this model has no trained
    quantile siblings yet (see ml/train.py's load_quantile_models()), in
    which case _run_inference() falls back to a rough +/-25% CI.
    """
    models: Dict[str, Any] = {}
    model, feature_names, categories, _ = load_model(model_name="coldstart_ramp")
    q10, q90 = load_quantile_models(model_name="coldstart_ramp")
    models["ramp"] = {"model": model, "feature_names": feature_names, "categories": categories, "q10_model": q10, "q90_model": q90}

    families = ("single", "permonth") if family == "both" else (family,)
    if "single" in families:
        model, feature_names, categories, _ = load_model(model_name="coldstart_stabilized_single")
        q10, q90 = load_quantile_models(model_name="coldstart_stabilized_single")
        models["single"] = {"model": model, "feature_names": feature_names, "categories": categories, "q10_model": q10, "q90_model": q90}
    if "permonth" in families:
        models["permonth"] = load_coldstart_permonth()
    return models


def _pick_coldstart_model(
    models: Dict[str, Any],
    months_since_active: int,
    feature_month: str,
    family: str,
) -> Tuple[Any, List[str], Dict[str, List[str]], List[str], Tuple[Optional[Any], Optional[Any]]]:
    """
    Pure lookup against an already-loaded models dict (see
    load_coldstart_models()) — returns (model, feature_names, categories,
    numeric_features, quantile_models) for one trajectory step.
    quantile_models is (q10_model, q90_model), possibly (None, None) if this
    particular model has no trained quantile siblings yet — see
    _run_inference()'s fallback. months_since_active <= RAMP_MAX always
    resolves to the shared ramp model, regardless of family — both
    architecture families use the identical ramp-stage design (see
    ml/coldstart_common.py). For months_since_active > RAMP_MAX, family
    picks the stabilized-stage model: 'single' = Family A's one model,
    'permonth' = Family B's model for feature_month's calendar month.
    """
    if months_since_active <= RAMP_MAX:
        entry = models["ramp"]
        return entry["model"], entry["feature_names"], entry["categories"], RAMP_NUMERIC_FEATURES, (entry.get("q10_model"), entry.get("q90_model"))
    if family == "single":
        entry = models["single"]
        return entry["model"], entry["feature_names"], entry["categories"], RAMP_NUMERIC_FEATURES, (entry.get("q10_model"), entry.get("q90_model"))
    if family == "permonth":
        cal_month = int(feature_month[5:7])
        entry = models["permonth"][cal_month]
        return entry["model"], entry["feature_names"], entry["categories"], PERMONTH_NUMERIC_FEATURES, (entry.get("q10_model"), entry.get("q90_model"))
    raise ValueError(f"Unknown cold-start family {family!r} — expected 'single' or 'permonth'")


def _run_inference(
    feature_row: Dict[str, Any],
    model,
    feature_names: List[str],
    categories: Optional[Dict[str, List[str]]] = None,
    quantile_models: Optional[Tuple[Any, Any]] = None,
    numeric_features: Optional[List[str]] = None,
) -> Dict[str, float]:
    df = pd.DataFrame([feature_row])
    df[COLDSTART_TARGET] = 0.0  # dummy target — prepare_features requires the column, unused for inference
    # categories must be the exact vocabulary the model was trained on — a
    # single-row DataFrame naturally has only one category value per column,
    # which LightGBM rejects as a mismatch if categories aren't pinned (see
    # train.py's prepare_features() docstring).
    X, _, _ = prepare_features(
        df, categories=categories, numeric_features=numeric_features,
        bool_features=COLDSTART_BOOL_FEATURES, categorical_features=COLDSTART_CATEGORICAL_FEATURES,
        target=COLDSTART_TARGET,
    )
    X = X.reindex(columns=feature_names)
    pred_log = model.predict(X)[0]
    pred_kwh = float(np.expm1(max(pred_log, 0)))

    q10_model, q90_model = quantile_models or (None, None)
    if q10_model and q90_model:
        lower = float(np.expm1(max(q10_model.predict(X)[0], 0)))
        upper = float(np.expm1(max(q90_model.predict(X)[0], 0)))
    else:
        lower, upper = pred_kwh * 0.75, pred_kwh * 1.25  # fallback CI, matches ml/predict.py

    return {
        "predicted_kwh": round(pred_kwh, 1),
        "predicted_kwh_lower": round(lower, 1),
        "predicted_kwh_upper": round(upper, 1),
    }


PEER_BLEND_WEIGHT = 0.7  # model's share of a blended stabilized-stage estimate; peer_avg_kwh gets the rest


def _blend_with_peer(inference: Dict[str, float], peer_avg_kwh: float, weight: float = PEER_BLEND_WEIGHT) -> Dict[str, float]:
    """Nudges one stabilized-stage prediction toward comparable REAL
    stations' own lifetime average (peer_benchmark_kwh()) — grounds a
    hypothetical new site's estimate in actual nearby outcomes, not just
    the model's geo/competition/calendar extrapolation. Scales the CI band
    by the same ratio as the point estimate, preserving its relative width
    instead of leaving it mismatched against the shifted center."""
    pred = inference["predicted_kwh"]
    blended = weight * pred + (1 - weight) * peer_avg_kwh
    scale = blended / pred if pred else 1.0
    return {
        **inference,
        "predicted_kwh": round(blended, 1),
        "predicted_kwh_lower": round(inference["predicted_kwh_lower"] * scale, 1),
        "predicted_kwh_upper": round(inference["predicted_kwh_upper"] * scale, 1),
    }


def _build_coldstart_trajectory_for_family(
    feature_rows: List[Dict[str, Any]],
    coldstart_models: Dict[str, Any],
    family: str,
    peer_avg_kwh: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """One trajectory (list of per-month dicts) for a single resolved family
    ('single' or 'permonth') — shared by predict_new_station_trajectory()
    and predict_existing_station_trajectory() so the per-row inference loop
    isn't duplicated, and so 'ensemble' can build both underlying
    trajectories the same way before blending them (see blend_trajectories()).
    Each row resolves its OWN quantile_models via _pick_coldstart_model() —
    the ramp/stabilized-single/each-calendar-month model all have distinct
    quantile siblings, so a single shared pair can't be hoisted out of the
    loop the way the point models briefly were.

    peer_avg_kwh: when given, blends it into every STABILIZED-stage step
    (months_since_active > RAMP_MAX) via _blend_with_peer() — left None
    (default) by predict_existing_station_trajectory(), since that function
    exists specifically to measure the model's own standalone accuracy
    against a real station's actuals; only predict_new_station_trajectory()
    passes it. Ramp-stage steps are never blended — peer_avg_kwh averages
    over each peer's ENTIRE history (including their own early ramp-up), so
    it's not a fair comparison for a literal month-1 prediction, only for
    the stabilized stage where both numbers represent steady-state-ish output."""
    trajectory = []
    for row in feature_rows:
        model, feature_names, categories, numeric_features, quantile_models = _pick_coldstart_model(
            coldstart_models, row["months_since_active"], row["feature_month"], family=family,
        )
        inference = _run_inference(
            row, model, feature_names, categories=categories,
            quantile_models=quantile_models, numeric_features=numeric_features,
        )
        if peer_avg_kwh is not None and row["months_since_active"] > RAMP_MAX:
            inference = _blend_with_peer(inference, peer_avg_kwh)
        trajectory.append({
            "feature_month": row["feature_month"],
            "months_since_active": row.get("months_since_active"),
            **inference,
        })
    return trajectory


def predict_new_station(
    lat: float,
    lon: float,
    chargers: Dict[float, int],
    prediction_month: Optional[str] = None,
    contracted_load_kva: Optional[float] = None,
    has_attached_restaurant: Optional[bool] = None,
    location_type: Optional[str] = None,
    direction_side: Optional[str] = None,
    fetch_competition: bool = True,
    model_version: str = "latest",
) -> Dict[str, Any]:
    """Single what-if prediction for a hypothetical new station — always the
    shared ramp model (months_since_active=0 by construction: a brand-new
    site with no prior bills, supply_date=feature_month)."""
    prediction_month = normalize_month(prediction_month) if prediction_month else default_prediction_month()
    feature_month = prev_month(prediction_month)

    station = build_candidate_station(
        lat, lon, chargers, feature_month,
        contracted_load_kva=contracted_load_kva, fetch_competition=fetch_competition,
    )
    station = apply_overrides(station, has_attached_restaurant, location_type, direction_side)

    weather, weather_is_proxy = fetch_weather_with_fallback(lat, lon, feature_month)
    feature_row = _alias_months_since_active(
        build_feature_row(station, history=[], weather=weather, feature_month=feature_month)
    )
    # Always the shared ramp model — a brand-new station's first prediction
    # is months_since_active=0 by construction, well within the ramp window.
    model, feature_names, categories, version = load_model(model_version, model_name="coldstart_ramp")
    quantile_models = load_quantile_models(model_version, model_name="coldstart_ramp")
    numeric_features = RAMP_NUMERIC_FEATURES

    inference = _run_inference(
        feature_row, model, feature_names, categories=categories,
        quantile_models=quantile_models, numeric_features=numeric_features,
    )
    peer = peer_benchmark_kwh(station)

    caveats = [COLD_START_CAVEAT]
    if not station.get("highway_name"):
        caveats.append(
            "Site is not within range of any tracked highway pair (config.settings.HIGHWAY_PAIRS) — "
            "highway position / direction_side / travel_time features are missing."
        )
    if weather_is_proxy:
        caveats.append(f"{feature_month[:7]} weather isn't in the archive yet — used {feature_month[:4]} from the prior year as a climatology proxy.")
    elif weather is None:
        caveats.append("Weather data unavailable for this site/month — weather features are missing.")

    return {
        "prediction_month": prediction_month,
        "model_version": version,
        **inference,
        **peer,
        "zone_band": station.get("zone_band"),
        "nearest_city": station.get("nearest_city"),
        "direction_side": station.get("direction_side"),
        "has_attached_restaurant": station.get("has_attached_restaurant"),
        "location_type": station.get("location_type"),
        "total_power_kw": total_power_kw(chargers),
        "contracted_load_kva": station.get("contracted_load_kva"),
        "caveats": caveats,
    }


def predict_new_station_trajectory(
    lat: float,
    lon: float,
    chargers: Dict[float, int],
    start_month: Optional[str] = None,
    horizon_months: int = 12,
    contracted_load_kva: Optional[float] = None,
    has_attached_restaurant: Optional[bool] = None,
    location_type: Optional[str] = None,
    direction_side: Optional[str] = None,
    fetch_competition: bool = True,
    family: str = "single",
    blend_peer_benchmark: bool = True,
) -> List[Dict[str, Any]]:
    """
    Cold-start prediction for each of the first horizon_months months after
    start_month (default: next calendar month), one row per month — a
    predicted ramp-up curve rather than predict_new_station()'s single point.
    Still never incorporates real consumption (there isn't any — it's a
    hypothetical site); months_since_active increments and weather/calendar
    vary per month, capacity/geo/competition stay fixed. See
    build_monthly_feature_rows() docstring.

    family: 'single' (Family A's one stabilized model), 'permonth' (Family
    B's 12 calendar-month models), or 'ensemble' ("Estimate C" — averages A
    and B per month, falling back to A alone wherever B diverges sharply,
    see ml/coldstart_common.py's blend_trajectories()). Only affects
    months_since_active > RAMP_MAX; the first RAMP_MAX+1 months always use
    the shared ramp model either way (A and B are identical there, so
    'ensemble' is a no-op for those months too). See
    ml/coldstart_validation.py's --compare-families report to compare A vs B.

    blend_peer_benchmark: when True (default) and a reliable peer group
    exists (>= 3 comparable real stations — see peer_benchmark_kwh()),
    nudges every STABILIZED-stage step toward those peers' own lifetime
    average kWh (see _blend_with_peer()) — grounds the estimate in real
    nearby outcomes instead of purely the model's extrapolation. Ramp-stage
    steps (months 0-5) are never blended, since peer_avg_kwh isn't a fair
    comparison for a literal month-1 prediction (see
    _build_coldstart_trajectory_for_family()'s docstring).
    """
    start_month = normalize_month(start_month) if start_month else default_prediction_month()
    coldstart_models = load_coldstart_models("both" if family == "ensemble" else family)

    station = build_candidate_station(
        lat, lon, chargers, start_month,
        contracted_load_kva=contracted_load_kva, fetch_competition=fetch_competition,
    )
    station = apply_overrides(station, has_attached_restaurant, location_type, direction_side)

    peer_avg_kwh = None
    if blend_peer_benchmark:
        peer = peer_benchmark_kwh(station)
        if peer.get("peer_avg_kwh") is not None and peer.get("peer_group_size", 0) >= 3:
            peer_avg_kwh = peer["peer_avg_kwh"]
            log.info(
                "Blending stabilized-stage trajectory with peer benchmark %.1f kWh (%s)",
                peer_avg_kwh, peer.get("peer_group_criteria"),
            )

    feature_rows = build_monthly_feature_rows(station, start_month, horizon_months)

    if family == "ensemble":
        trajectory_a = _build_coldstart_trajectory_for_family(feature_rows, coldstart_models, "single", peer_avg_kwh)
        trajectory_b = _build_coldstart_trajectory_for_family(feature_rows, coldstart_models, "permonth", peer_avg_kwh)
        return blend_trajectories(trajectory_a, trajectory_b)
    return _build_coldstart_trajectory_for_family(feature_rows, coldstart_models, family, peer_avg_kwh)


def predict_existing_station_trajectory(
    unique_scno: str,
    family: str = "both",
    horizon_months: Optional[int] = None,
) -> Dict[str, Any]:
    """
    "Test prediction" for a REAL station already in the DB — predicts its
    cold-start trajectory using the production-saved ramp/stabilized models
    and compares it month-by-month against that station's own real billing
    history, same methodology as ml/coldstart_validation.py's HTML report.

    Important difference from that report: this uses whichever models are
    currently deployed (load_coldstart_models(), reading ml/artifacts/
    directly), NOT a holdout retrain excluding this station — the production
    model has very likely already seen this station's data during its own
    training. This is a quick sanity-check / demo tool (fast — no retraining,
    typically a few seconds once weather is cached), not a rigorous
    generalization test; use ml.coldstart_validation's --compare-families
    report for that.

    family: 'single', 'permonth', 'ensemble' (just "Estimate C" — A/B
    averaged per month, falling back to A alone wherever B diverges sharply,
    see ml/coldstart_common.py's blend_trajectories()), or 'both' (default —
    returns single, permonth, AND ensemble together, so a UI can plot all
    three lines against the actual series in one chart; computing the
    ensemble is nearly free once both underlying trajectories already exist).
    horizon_months: cap on months predicted; None (default) predicts the
    station's own full available billing history span.

    Returns:
        {
          "unique_scno": ..., "station_name": ..., "start_month": ...,
          "horizon_months": <int actually used>,
          "families": {"single": {"rows": [...], "mape": ..., "n_matched": ...}, "permonth": {...}, "ensemble": {...}},
        }
    Raises LookupError if the station doesn't exist, ValueError if it has no
    non-zero billing history to compare against.
    """
    from ev_pipeline.db.db_manager import get_station, get_station_billing_history

    station = get_station(unique_scno)
    if not station:
        raise LookupError(f"No station found for unique_scno={unique_scno!r}")

    history = get_station_billing_history(unique_scno)
    series = actual_series(history)
    if not series:
        raise ValueError(f"Station {unique_scno} has no non-zero billing history to compare against")

    start_month = series[0][0][:7] + "-01"
    full_span = month_diff(series[0][0][:7], series[-1][0][:7])
    effective_horizon = full_span if horizon_months is None else min(full_span, horizon_months)

    need_single = family in ("single", "both", "ensemble")
    need_permonth = family in ("permonth", "both", "ensemble")
    coldstart_models = load_coldstart_models("both" if (need_single and need_permonth) else "single" if need_single else "permonth")
    feature_rows = build_monthly_feature_rows(
        station, start_month, effective_horizon, cache_scno=unique_scno,
    )

    raw_trajectories = {}
    if need_single:
        raw_trajectories["single"] = _build_coldstart_trajectory_for_family(feature_rows, coldstart_models, "single")
    if need_permonth:
        raw_trajectories["permonth"] = _build_coldstart_trajectory_for_family(feature_rows, coldstart_models, "permonth")

    result_families = {}
    if family in ("single", "both"):
        result_families["single"] = compare_trajectory_to_actuals(series, raw_trajectories["single"])
    if family in ("permonth", "both"):
        result_families["permonth"] = compare_trajectory_to_actuals(series, raw_trajectories["permonth"])
    if family in ("ensemble", "both"):
        blended = blend_trajectories(raw_trajectories["single"], raw_trajectories["permonth"])
        result_families["ensemble"] = compare_trajectory_to_actuals(series, blended)

    return {
        "unique_scno": unique_scno,
        "station_name": station.get("station_name"),
        "start_month": start_month,
        "horizon_months": effective_horizon,
        "families": result_families,
    }


def scan_configurations(
    lat: float,
    lon: float,
    charger_grid: List[Dict[float, int]],
    prediction_month: Optional[str] = None,
    restaurant_options: Tuple[Optional[bool], ...] = (None,),
    location_type_options: Tuple[Optional[str], ...] = (None,),
    contracted_load_options: Tuple[Optional[float], ...] = (None,),
    scan_direction: bool = False,
    fetch_competition: bool = True,
    model_version: str = "latest",
) -> pd.DataFrame:
    """
    Grid-search over the iterable what-if parameters for ONE fixed site
    (lat/lon don't change — only the developer's design choices do). Places/
    weather lookups happen once, not once per combination. Returns one row
    per configuration, sorted by predicted_kwh descending.
    """
    prediction_month = normalize_month(prediction_month) if prediction_month else default_prediction_month()
    feature_month = prev_month(prediction_month)

    # Always the shared ramp model — every scanned config is a brand-new
    # station's first prediction, months_since_active=0 by construction.
    model, feature_names, categories, version = load_model(model_version, model_name="coldstart_ramp")
    quantile_models = load_quantile_models(model_version, model_name="coldstart_ramp")
    base_station = build_candidate_station(lat, lon, {}, feature_month, fetch_competition=fetch_competition)
    weather, weather_is_proxy = fetch_weather_with_fallback(lat, lon, feature_month)
    direction_options = highway_direction_options(base_station) if scan_direction else (None,)
    all_stations = get_all_stations()

    rows = []
    for chargers in charger_grid:
        charger_cols = chargers_to_columns(chargers)
        power = total_power_kw(chargers)
        for restaurant in restaurant_options:
            for direction in direction_options:
                for loc_type in location_type_options:
                    for load in contracted_load_options:
                        station = dict(base_station)
                        station.update(charger_cols)
                        station["contracted_load_kva"] = load if load is not None else power
                        station = apply_overrides(station, restaurant, loc_type, direction)

                        feature_row = _alias_months_since_active(
                            build_feature_row(station, history=[], weather=weather, feature_month=feature_month)
                        )
                        inference = _run_inference(
                            feature_row, model, feature_names, categories=categories,
                            quantile_models=quantile_models, numeric_features=RAMP_NUMERIC_FEATURES,
                        )
                        peer = peer_benchmark_kwh(station, all_stations)

                        rows.append({
                            "chargers": format_chargers(chargers),
                            "total_power_kw": power,
                            "contracted_load_kva": station["contracted_load_kva"],
                            "has_attached_restaurant": station.get("has_attached_restaurant"),
                            "direction_side": station.get("direction_side"),
                            "location_type": station.get("location_type"),
                            **inference,
                            "peer_avg_kwh": peer.get("peer_avg_kwh"),
                            "peer_group_size": peer.get("peer_group_size"),
                        })

    df = pd.DataFrame(rows).sort_values("predicted_kwh", ascending=False).reset_index(drop=True)
    log.info(
        "Scanned %d configurations for (%.5f, %.5f), prediction_month=%s, model=%s%s",
        len(df), lat, lon, prediction_month, version,
        " [weather is prior-year proxy]" if weather_is_proxy else "",
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_chargers(s: str) -> Dict[float, int]:
    return {float(kw): int(count) for kw, count in (part.split(":") for part in s.split(",") if part)}


def _parse_chargers_grid(s: str) -> List[Dict[float, int]]:
    return [_parse_chargers(part) for part in s.split("|")]


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Predict performance for a hypothetical new EV station")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--month", default=None, help="Prediction month, e.g. 2026-09 (default: next calendar month)")
    parser.add_argument("--contracted-load", type=float, default=None)
    parser.add_argument("--restaurant", default=None, help="true/false override (default: use live Places data)")
    parser.add_argument("--location-type", default=None)
    parser.add_argument("--direction-side", default=None)
    parser.add_argument("--no-competition", action="store_true", help="Skip the Places API lookup")
    parser.add_argument("--model-version", default="latest")

    parser.add_argument("--chargers", default=None, help="Single config, e.g. '60:2,120:1'")

    parser.add_argument("--scan", action="store_true", help="Grid-search mode instead of a single prediction")
    parser.add_argument("--chargers-grid", default=None, help="Pipe-separated configs, e.g. '30:4|60:2,120:1'")
    parser.add_argument("--scan-restaurant", action="store_true", help="Try both has_attached_restaurant values")
    parser.add_argument("--scan-direction", action="store_true", help="Try both highway directions")
    parser.add_argument("--out", default=None, help="CSV path to save --scan results")

    args = parser.parse_args()

    if args.scan:
        if not args.chargers_grid:
            parser.error("--scan requires --chargers-grid")
        df = scan_configurations(
            lat=args.lat, lon=args.lon,
            charger_grid=_parse_chargers_grid(args.chargers_grid),
            prediction_month=args.month,
            restaurant_options=(True, False) if args.scan_restaurant else (None,),
            location_type_options=(args.location_type,),
            contracted_load_options=(args.contracted_load,),
            scan_direction=args.scan_direction,
            fetch_competition=not args.no_competition,
            model_version=args.model_version,
        )
        print(df.to_string(index=False))
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nSaved to {args.out}")
    else:
        if not args.chargers:
            parser.error("--chargers is required (e.g. --chargers 60:2,120:1), or use --scan")
        result = predict_new_station(
            lat=args.lat, lon=args.lon,
            chargers=_parse_chargers(args.chargers),
            prediction_month=args.month,
            contracted_load_kva=args.contracted_load,
            has_attached_restaurant=_parse_bool(args.restaurant) if args.restaurant is not None else None,
            location_type=args.location_type,
            direction_side=args.direction_side,
            fetch_competition=not args.no_competition,
            model_version=args.model_version,
        )
        for k, v in result.items():
            if k == "caveats":
                print("caveats:")
                for c in v:
                    print(f"  - {c}")
            else:
                print(f"{k}: {v}")
