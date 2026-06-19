# ml/coldstart_validation.py
"""
Validates ml/site_simulator.py's cold-start "predict from scratch" methodology
against real stations' actual billing history.

We can't validate a cold-start prediction against a truly new station — by
definition there's no ground truth for it yet. So instead: pick N existing
stations at random, retrain a model that has NEVER seen their data (a real
generalization test, not just re-scoring rows the model memorized), predict
each one exactly as if it were brand new (supply_date = its own real first
bill month, zero billing history, only the static geo/capacity/competition
features site_simulator.py would have had), and compare that single flat
"as of opening" prediction against the station's REAL monthly kWh trajectory
over its whole lifetime.

The holdout model is trained in-memory only (train.py's run_training(save=False))
and never touches ml/artifacts/ — it can't accidentally become "latest" and
get served to real predictions.

Usage:
    python -m ml.coldstart_validation --n 5 --seed 42 --out reports/coldstart_validation.html
"""

import argparse
import base64
import io
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ev_pipeline.db.db_manager import get_all_stations, get_feature_matrix, get_station_billing_history
from ev_pipeline.features.feature_builder import build_feature_row
from ml.site_simulator import _run_inference, fetch_weather_with_fallback
from ml.train import run_training

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    log.warning("matplotlib not installed — plots will be skipped in the report")

DEFAULT_MIN_MONTHS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Holdout selection + retraining
# ─────────────────────────────────────────────────────────────────────────────

def select_holdout_stations(n: int, seed: int, min_months: int = DEFAULT_MIN_MONTHS) -> List[Dict[str, Any]]:
    """
    Random sample of n stations with at least min_months of real billing
    history (otherwise there's nothing meaningful to compare the cold-start
    prediction against).
    """
    stations = get_all_stations()
    rng = random.Random(seed)
    rng.shuffle(stations)

    selected = []
    for station in stations:
        if len(selected) >= n:
            break
        history = get_station_billing_history(station["unique_scno"])
        if len(history) >= min_months:
            station["_history"] = history
            selected.append(station)

    if len(selected) < n:
        log.warning("Only found %d/%d stations with >= %d months of history", len(selected), n, min_months)
    return selected


def train_holdout_model(excluded_scnos: List[str]) -> Tuple[Any, List[str]]:
    """
    Retrain on every station EXCEPT the held-out ones. save=False/log_to_db=False
    — this model only ever lives in memory for this validation run.
    """
    df = get_feature_matrix(min_months=3)
    df = df[~df["unique_scno"].isin(excluded_scnos)].copy()
    log.info("Training holdout model on %d rows from %d stations (excluding %d held out)",
              len(df), df["unique_scno"].nunique(), len(excluded_scnos))
    metrics = run_training(df=df, run_shap=False, save=False, log_to_db=False)
    return metrics["_model"], metrics["_feature_names"]


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start prediction for a real (existing) station
# ─────────────────────────────────────────────────────────────────────────────

def build_coldstart_prediction(station: Dict[str, Any], model, feature_names: List[str]) -> Dict[str, Any]:
    """
    Predict this real station's performance exactly as site_simulator.py
    would for a brand-new site — but reusing its already-stored geo/
    competition/zone fields directly (no live Places re-fetch needed; this
    station already exists, so that data is real, not hypothetical) and
    overriding supply_date to its own real first bill month so
    months_since_opening=0 / is_ramp_up_phase=True, matching what a true
    "predict before I have any history" call would have looked like.
    """
    history = station.get("_history") or get_station_billing_history(station["unique_scno"])
    first_bill_month = str(history[0]["bill_month"])[:10]

    candidate = dict(station)
    candidate["supply_date"] = first_bill_month

    weather = None
    if station.get("latitude") and station.get("longitude"):
        weather, _ = fetch_weather_with_fallback(station["latitude"], station["longitude"], first_bill_month)

    feature_row = build_feature_row(candidate, history=[], weather=weather, feature_month=first_bill_month)
    inference = _run_inference(feature_row, model, feature_names, quantile_models=None)
    return {**inference, "feature_month": first_bill_month}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison against the real trajectory
# ─────────────────────────────────────────────────────────────────────────────

def actual_series(history: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    series = [
        (str(r["bill_month"])[:10], float(r.get("kwh_units") or r.get("billed_units") or 0))
        for r in history if r.get("bill_month")
    ]
    return sorted(series)


def compare_to_actuals(history: List[Dict[str, Any]], predicted_kwh: float) -> Dict[str, Any]:
    series = actual_series(history)
    values = [v for _, v in series]

    def pct_err(actual: Optional[float]) -> Optional[float]:
        if actual is None or actual == 0:
            return None
        return round(100 * (predicted_kwh - actual) / actual, 1)

    month1 = values[0] if values else None
    avg_3mo = float(np.mean(values[:3])) if values else None
    avg_lifetime = float(np.mean(values)) if values else None

    return {
        "month1_actual": month1, "month1_pct_error": pct_err(month1),
        "avg_3mo_actual": round(avg_3mo, 1) if avg_3mo is not None else None, "avg_3mo_pct_error": pct_err(avg_3mo),
        "avg_lifetime_actual": round(avg_lifetime, 1) if avg_lifetime is not None else None, "avg_lifetime_pct_error": pct_err(avg_lifetime),
        "n_months": len(values),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot + HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_station(scno: str, station_name: Optional[str], series: List[Tuple[str, float]], prediction: Dict[str, Any]) -> str:
    if not _HAS_MPL:
        return ""
    import pandas as pd
    months = [pd.Timestamp(m) for m, _ in series]
    values = [v for _, v in series]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(months, values, marker="o", markersize=4, linewidth=1.5, color="#3498db", label="Actual kWh")
    ax.axhline(prediction["predicted_kwh"], color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Cold-start prediction ({prediction['predicted_kwh']:.0f} kWh)")
    ax.axhspan(prediction["predicted_kwh_lower"], prediction["predicted_kwh_upper"], color="#e74c3c", alpha=0.1)
    ax.set_title(f"{station_name or scno} ({scno[-6:]}) — predicted as of {prediction['feature_month'][:7]}")
    ax.set_ylabel("kWh")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cold-Start Validation Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1000px; margin: 0 auto;
          padding: 20px; background: #f8f9fa; color: #2c3e50; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 8px; }}
  h2 {{ color: #34495e; margin-top: 32px; border-left: 4px solid #e74c3c; padding-left: 10px; }}
  table {{ background: white; border-collapse: collapse; width: 100%; font-size: 13px; }}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #ecf0f1; }} td:first-child, th:first-child {{ text-align: left; }}
  .card {{ background: white; padding: 14px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 16px 0; }}
  .note {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 10px 14px; border-radius: 4px; font-size: 13px; }}
</style></head><body>
<h1>Cold-Start Validation Report</h1>
<p style="color:#7f8c8d">Generated {generated_at} — holdout model trained on {n_train_rows} rows from {n_train_stations} stations, excluding the {n_holdout} held-out stations below.</p>
<div class="note">Each station's prediction uses ONLY static geo/capacity/competition features
and its own real opening month — no billing history, exactly as ml/site_simulator.py would
treat a brand-new site. pct_error = (predicted - actual) / actual * 100.</div>

<h2>Aggregate error (mean abs % error across {n_holdout} stations)</h2>
{aggregate_table}

<h2>Per-station detail</h2>
{station_cards}

<footer style="margin-top:40px;font-size:0.8em;color:#aaa;text-align:center">ml/coldstart_validation.py &bull; {generated_at}</footer>
</body></html>"""


def _aggregate_table(results: List[Dict[str, Any]]) -> str:
    rows = []
    for key, label in [("month1_pct_error", "vs month-1 actual"),
                        ("avg_3mo_pct_error", "vs first-3-month avg"),
                        ("avg_lifetime_pct_error", "vs lifetime avg")]:
        errs = [abs(r["comparison"][key]) for r in results if r["comparison"].get(key) is not None]
        mape = f"{np.mean(errs):.1f}%" if errs else "n/a"
        rows.append(f"<tr><td>{label}</td><td>{mape}</td><td>{len(errs)}/{len(results)} stations</td></tr>")
    return ("<table><tr><th>Comparison point</th><th>Mean abs % error</th><th>Coverage</th></tr>"
            + "".join(rows) + "</table>")


def _station_card(result: Dict[str, Any]) -> str:
    s, pred, cmp_ = result["station"], result["prediction"], result["comparison"]
    img = f'<img src="data:image/png;base64,{result["plot"]}" style="width:100%;max-width:850px">' if result["plot"] else ""
    return f"""
<div class="card">
  {img}
  <table>
    <tr><th>predicted_kwh</th><th>range</th><th>month-1 actual</th><th>err</th>
        <th>3mo avg actual</th><th>err</th><th>lifetime avg actual</th><th>err</th><th>months of data</th></tr>
    <tr>
      <td>{pred['predicted_kwh']:.0f}</td>
      <td>{pred['predicted_kwh_lower']:.0f}-{pred['predicted_kwh_upper']:.0f}</td>
      <td>{cmp_['month1_actual'] if cmp_['month1_actual'] is not None else 'n/a'}</td>
      <td>{cmp_['month1_pct_error']}%</td>
      <td>{cmp_['avg_3mo_actual'] if cmp_['avg_3mo_actual'] is not None else 'n/a'}</td>
      <td>{cmp_['avg_3mo_pct_error']}%</td>
      <td>{cmp_['avg_lifetime_actual'] if cmp_['avg_lifetime_actual'] is not None else 'n/a'}</td>
      <td>{cmp_['avg_lifetime_pct_error']}%</td>
      <td>{cmp_['n_months']}</td>
    </tr>
  </table>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(n: int = 5, seed: int = 42, min_months: int = DEFAULT_MIN_MONTHS, out: str = "reports/coldstart_validation.html"):
    from datetime import datetime
    from pathlib import Path

    holdout_stations = select_holdout_stations(n, seed, min_months)
    if not holdout_stations:
        log.error("No stations with >= %d months of history found — nothing to validate.", min_months)
        return

    excluded = [s["unique_scno"] for s in holdout_stations]
    model, feature_names = train_holdout_model(excluded)

    full_df = get_feature_matrix(min_months=3)
    n_train_rows = len(full_df[~full_df["unique_scno"].isin(excluded)])
    n_train_stations = full_df[~full_df["unique_scno"].isin(excluded)]["unique_scno"].nunique()

    results = []
    for station in holdout_stations:
        scno = station["unique_scno"]
        history = station["_history"]
        prediction = build_coldstart_prediction(station, model, feature_names)
        comparison = compare_to_actuals(history, prediction["predicted_kwh"])
        plot = plot_station(scno, station.get("station_name"), actual_series(history), prediction)
        results.append({"station": station, "prediction": prediction, "comparison": comparison, "plot": plot})
        log.info("SCNo %s: predicted=%.0f kWh, month1_actual=%s (err %s%%), lifetime_avg_actual=%s (err %s%%)",
                  scno, prediction["predicted_kwh"], comparison["month1_actual"], comparison["month1_pct_error"],
                  comparison["avg_lifetime_actual"], comparison["avg_lifetime_pct_error"])

    html = _HTML_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_train_rows=n_train_rows, n_train_stations=n_train_stations, n_holdout=len(results),
        aggregate_table=_aggregate_table(results),
        station_cards="".join(_station_card(r) for r in results),
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("Cold-start validation report written to %s", out_path.resolve())
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Validate cold-start predictions against real station history")
    parser.add_argument("--n", type=int, default=5, help="Number of stations to hold out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-months", type=int, default=DEFAULT_MIN_MONTHS)
    parser.add_argument("--out", default="reports/coldstart_validation.html")
    args = parser.parse_args()

    run_validation(n=args.n, seed=args.seed, min_months=args.min_months, out=args.out)
