# ml/site_simulator.py
"""
"What if I built a station here?" predictor for sites that don't exist yet.

The trained model (ml/train.py) is fundamentally an autoregressive demand
model: its strongest predictors are consumption-momentum features
(avg_kwh_units, rolling_avg_3m_kwh, kwh_growth_rate_overall, seasonality,
coefficient of variation...) computed from real billing history. A
hypothetical new station has none of that by definition. This module does
NOT fabricate fake history to fill the gap — it leaves those features
missing and relies on LightGBM's native missing-value routing (the same
property train.py's docstring already leans on), then reports a separate,
clearly-labeled peer-group benchmark as an independent sanity check.

What a candidate site needs, split by source
──────────────────────────────────────────────
REQUIRED INPUT (only thing you must supply):
  latitude, longitude, chargers ({kw: count, ...})

DERIVED AUTOMATICALLY, free (geometry only):
  h3_res5/6/7, nearest_city / dist_to_city_km / zone_band, highway position
  (dist_from_city_a/b_km, direction_side, travel_time_*) if the site is near
  a tracked highway pair, total_charger_count / total_power_kw /
  charger_mix_ratio, calendar features for the prediction month,
  months_since_opening=0 / is_ramp_up_phase=True (it's a new station).

DERIVED VIA LIVE LOOKUPS, optional (real-world data, costs API quota):
  competition counts + rating + location_type + has_attached_restaurant
  (Google Places nearby search at the candidate lat/lon — reuses
  scrapers.places_scraper.enrich_station as-is, since it already works off
  bare coordinates), weather climatology for the prediction month (Open-Meteo
  archive; falls back to the same month last year if the prediction month
  is in the future and the archive doesn't have it yet).

NOT AVAILABLE, by definition:
  avg_kwh_units, rolling_avg_3m_kwh, rolling_avg_6m_kwh, std_kwh_units,
  kwh_growth_rate_overall, seasonal_summer/winter_avg_kwh,
  kwh_coefficient_of_variation, pct_months_zero_consumption — left missing.

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
from ml.predict import _load_quantile_models
from ml.train import TARGET, load_model, prepare_features

log = logging.getLogger(__name__)

SITESIM_SCNO = "SITESIM"

CONSUMPTION_HISTORY_FEATURES = [
    "avg_kwh_units", "std_kwh_units", "rolling_avg_3m_kwh", "rolling_avg_6m_kwh",
    "kwh_growth_rate_overall", "seasonal_summer_avg_kwh", "seasonal_winter_avg_kwh",
    "pct_months_zero_consumption", "kwh_coefficient_of_variation",
]

COLD_START_CAVEAT = (
    "No billing history exists for this site, so consumption-momentum features "
    f"({', '.join(CONSUMPTION_HISTORY_FEATURES)}) are left missing rather than "
    "estimated, and the model routes the prediction using only capacity/geo/"
    "competition/calendar signals. Treat predicted_kwh as a rough location-and-"
    "capacity-based prior — materially less reliable than predictions for "
    "stations with real history."
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


def fetch_weather_with_fallback(lat: float, lon: float, feature_month: str) -> Tuple[Optional[Dict], bool]:
    """
    Weather for feature_month. If feature_month is in the future (beyond the
    Open-Meteo archive's ~5 day lag), falls back to the same calendar month
    last year as a climatology proxy. Returns (weather_dict_or_None, is_proxy).
    """
    y, m = int(feature_month[:4]), int(feature_month[5:7])
    weather = fetch_monthly_weather(SITESIM_SCNO, lat, lon, y, m)
    if weather:
        return weather, False
    weather = fetch_monthly_weather(SITESIM_SCNO, lat, lon, y - 1, m)
    return weather, weather is not None


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

def _run_inference(
    feature_row: Dict[str, Any],
    model,
    feature_names: List[str],
    quantile_models: Optional[Tuple[Any, Any]] = None,
) -> Dict[str, float]:
    df = pd.DataFrame([feature_row])
    df[TARGET] = 0.0  # dummy target — prepare_features requires the column, unused for inference
    X, _, _ = prepare_features(df)
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
    """Single what-if prediction for a hypothetical new station."""
    prediction_month = normalize_month(prediction_month) if prediction_month else default_prediction_month()
    feature_month = prev_month(prediction_month)

    model, feature_names, version = load_model(model_version)
    quantile_models = _load_quantile_models()

    station = build_candidate_station(
        lat, lon, chargers, feature_month,
        contracted_load_kva=contracted_load_kva, fetch_competition=fetch_competition,
    )
    station = apply_overrides(station, has_attached_restaurant, location_type, direction_side)

    weather, weather_is_proxy = fetch_weather_with_fallback(lat, lon, feature_month)
    feature_row = build_feature_row(station, history=[], weather=weather, feature_month=feature_month)

    inference = _run_inference(feature_row, model, feature_names, quantile_models)
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

    model, feature_names, version = load_model(model_version)
    quantile_models = _load_quantile_models()
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

                        feature_row = build_feature_row(station, history=[], weather=weather, feature_month=feature_month)
                        inference = _run_inference(feature_row, model, feature_names, quantile_models)
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
