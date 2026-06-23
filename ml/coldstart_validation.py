# ml/coldstart_validation.py
"""
Validates ml/site_simulator.py's cold-start "predict from scratch" methodology
against real stations' actual billing history.

We can't validate a cold-start prediction against a truly new station — by
definition there's no ground truth for it yet. So instead: pick N existing
stations at random, retrain the cold-start models (ml/train_coldstart_ramp.py
+ ml/train_coldstart_stabilized_single.py / _permonth.py) on every OTHER
station (a real generalization test, not just re-scoring rows the model
memorized), predict a full ramp-up TRAJECTORY for each held-out station
exactly as if it were brand new (supply_date = its own real first NON-ZERO
bill month — see db_manager.get_coldstart_training_matrix()'s docstring on
why leading zero-kWh bills are excluded — one cold-start prediction per
month for the first --horizon-months months), and compare each predicted
month against the REAL actual for that SAME calendar month — not a single
flat number compared against the station's entire multi-year lifetime,
which conflates "month 1" with "steady state" and makes any cold-start
prediction look wrong by construction.

--compare-families trains and reports BOTH stabilized-stage architectures
side by side (Family A "single" vs Family B "per-calendar-month" — see
ml/coldstart_common.py) so you can pick a winner; both share the same
holdout split and the same ramp-stage model, so the comparison isolates
just the stabilized-stage design.

All holdout models are trained in-memory only (run_training(save=False))
and never touch ml/artifacts/ — they can't accidentally become "latest" and
get served to real predictions.

Usage:
    python -m ml.coldstart_validation --n 5 --seed 42 --horizon-months 12
    python -m ml.coldstart_validation --n 5 --compare-families
"""

import argparse
import base64
import io
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ev_pipeline.db.db_manager import get_all_stations, get_coldstart_training_matrix, get_station_billing_history
from ml.coldstart_common import (
    COLDSTART_BOOL_FEATURES,
    COLDSTART_CATEGORICAL_FEATURES,
    COLDSTART_TARGET,
    PERMONTH_NUMERIC_FEATURES,
    RAMP_MAX,
    RAMP_MIN,
    RAMP_NUMERIC_FEATURES,
    STABILIZED_MIN,
    actual_series,
    compare_trajectory_to_actuals,
    month_diff,
)
from ml.site_simulator import _run_inference, build_monthly_feature_rows
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
FAMILY_LABELS = {"single": "Family A (single)", "permonth": "Family B (per-calendar-month)"}
FAMILY_COLORS = {"single": "#e74c3c", "permonth": "#27ae60"}


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


def _train_one(df, numeric_features) -> Dict[str, Any]:
    metrics = run_training(
        df=df, target=COLDSTART_TARGET,
        numeric_features=numeric_features, bool_features=COLDSTART_BOOL_FEATURES,
        categorical_features=COLDSTART_CATEGORICAL_FEATURES,
        run_shap=False, save=False, log_to_db=False, compute_thresholds=False,
    )
    return {"model": metrics["_model"], "feature_names": metrics["_feature_names"], "categories": metrics["_categories"]}


def train_holdout_coldstart_models(excluded_scnos: List[str], family: str = "single") -> Dict[str, Any]:
    """
    Retrain the shared ramp model plus the requested stabilized-stage
    family(ies), excluding the held-out stations. Returns:
        {"ramp": {...}, "single": {...}?, "permonth": {cal_month: {...}}?}
    Each {...} is {"model", "feature_names", "categories"}. Pass
    family="both" to train both stabilized families for --compare-families.
    """
    ramp_df = get_coldstart_training_matrix(RAMP_MIN, RAMP_MAX)
    ramp_df = ramp_df[~ramp_df["unique_scno"].isin(excluded_scnos)].copy()
    log.info("Training holdout ramp model on %d rows from %d stations (excluding %d held out)",
              len(ramp_df), ramp_df["unique_scno"].nunique(), len(excluded_scnos))
    holdout: Dict[str, Any] = {"ramp": _train_one(ramp_df, RAMP_NUMERIC_FEATURES)}

    families = ("single", "permonth") if family == "both" else (family,)

    if "single" in families:
        single_df = get_coldstart_training_matrix(STABILIZED_MIN, None)
        single_df = single_df[~single_df["unique_scno"].isin(excluded_scnos)].copy()
        log.info("Training holdout Family A stabilized model on %d rows from %d stations",
                  len(single_df), single_df["unique_scno"].nunique())
        holdout["single"] = _train_one(single_df, RAMP_NUMERIC_FEATURES)

    if "permonth" in families:
        permonth_df = get_coldstart_training_matrix(STABILIZED_MIN, None)
        permonth_df = permonth_df[~permonth_df["unique_scno"].isin(excluded_scnos)].copy()
        log.info("Training holdout Family B per-calendar-month models on %d total rows from %d stations",
                  len(permonth_df), permonth_df["unique_scno"].nunique())
        permonth: Dict[int, Dict[str, Any]] = {}
        for cal_month in range(1, 13):
            sub = permonth_df[permonth_df["month_of_year"] == cal_month]
            if sub.empty:
                continue
            permonth[cal_month] = _train_one(sub, PERMONTH_NUMERIC_FEATURES)
        holdout["permonth"] = permonth

    return holdout


def _resolve_holdout_model(
    holdout_models: Dict[str, Any],
    months_since_active: int,
    feature_month: str,
    family: str,
) -> Tuple[Any, List[str], Dict[str, List[str]], List[str]]:
    """In-memory mirror of ml.site_simulator._pick_coldstart_model() —
    same selection logic, against holdout-trained models instead of disk."""
    if months_since_active <= RAMP_MAX:
        entry = holdout_models["ramp"]
        return entry["model"], entry["feature_names"], entry["categories"], RAMP_NUMERIC_FEATURES
    if family == "single":
        entry = holdout_models["single"]
        return entry["model"], entry["feature_names"], entry["categories"], RAMP_NUMERIC_FEATURES
    cal_month = int(feature_month[5:7])
    entry = holdout_models["permonth"][cal_month]
    return entry["model"], entry["feature_names"], entry["categories"], PERMONTH_NUMERIC_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start trajectory for a real (existing) station
# ─────────────────────────────────────────────────────────────────────────────

def build_coldstart_trajectory(
    station: Dict[str, Any],
    holdout_models: Dict[str, Any],
    start_month: str,
    horizon_months: int = 12,
    family: str = "single",
) -> List[Dict[str, Any]]:
    """
    Predict this real station's performance for each of its first
    horizon_months months exactly as site_simulator.py would for a brand-new
    site — reusing its already-stored geo/competition/zone fields directly
    (no live Places re-fetch needed; this station already exists, so that
    data is real, not hypothetical) and overriding supply_date to its own
    first real (non-zero-bill) month so months_since_active increments
    naturally, matching what a true "predict before I have any history"
    call sequence would have looked like. See
    ml.site_simulator.build_monthly_feature_rows().
    """
    feature_rows = build_monthly_feature_rows(
        station, start_month, horizon_months, cache_scno=station["unique_scno"],
    )
    trajectory = []
    for row in feature_rows:
        model, feature_names, categories, numeric_features = _resolve_holdout_model(
            holdout_models, row["months_since_active"], row["feature_month"], family,
        )
        inference = _run_inference(
            row, model, feature_names, categories=categories,
            quantile_models=None, numeric_features=numeric_features,
        )
        trajectory.append({
            "feature_month": row["feature_month"],
            "months_since_active": row.get("months_since_active"),
            **inference,
        })
    return trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Comparison against the real trajectory — actual_series()/
# compare_trajectory_to_actuals() now live in ml.coldstart_common, shared
# with ml.site_simulator.predict_existing_station_trajectory()'s live
# "test prediction" API path so both compare the exact same way.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Plot + HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_station(
    scno: str,
    station_name: Optional[str],
    series: List[Tuple[str, float]],
    trajectories: Dict[str, List[Dict[str, Any]]],
) -> str:
    if not _HAS_MPL:
        return ""
    import pandas as pd
    actual_months = [pd.Timestamp(m) for m, _ in series]
    actual_values = [v for _, v in series]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(actual_months, actual_values, marker="o", markersize=4, linewidth=1.5, color="#3498db", label="Actual kWh")

    horizon = 0
    first_month = None
    for family, trajectory in trajectories.items():
        pred_months = [pd.Timestamp(step["feature_month"]) for step in trajectory]
        pred_values = [step["predicted_kwh"] for step in trajectory]
        pred_lower = [step["predicted_kwh_lower"] for step in trajectory]
        pred_upper = [step["predicted_kwh_upper"] for step in trajectory]
        color = FAMILY_COLORS.get(family, "#999999")
        ax.plot(pred_months, pred_values, marker="s", markersize=4, linewidth=1.5, linestyle="--",
                color=color, label=FAMILY_LABELS.get(family, family))
        ax.fill_between(pred_months, pred_lower, pred_upper, color=color, alpha=0.10)
        horizon = max(horizon, len(trajectory))
        first_month = first_month or trajectory[0]["feature_month"][:7]

    ax.set_title(f"{station_name or scno} ({scno[-6:]}) — {horizon}-month cold-start forecast from {first_month}")
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
  .subtable {{ margin-top: 8px; }}
</style></head><body>
<h1>Cold-Start Validation Report</h1>
<p style="color:#7f8c8d">Generated {generated_at} — shared ramp model trained on {n_ramp_rows} rows from
{n_ramp_stations} stations, excluding the {n_holdout} held-out stations below.</p>
<div class="note">Each station gets a predicted trajectory spanning its own full available billing history
(up to {horizon_months} months in this report) using ONLY static geo/capacity/competition features and its
own first real (non-zero-bill) month — no billing history, exactly as ml/site_simulator.py would treat a
brand-new site. Months 0-{ramp_max} use the shared ramp model; months {stabilized_min}+ use the
stabilized-stage model(s) below. Each predicted month is compared against the REAL actual for that SAME
calendar month. pct_error = (predicted - actual) / actual * 100; MAPE = mean absolute pct_error over the
months that have both a prediction and a real (non-zero) actual.</div>

<h2>Aggregate error (MAPE across {n_holdout} stations)</h2>
{aggregate_table}

<h2>Per-station detail</h2>
{station_cards}

<footer style="margin-top:40px;font-size:0.8em;color:#aaa;text-align:center">ml/coldstart_validation.py &bull; {generated_at}</footer>
</body></html>"""


def _aggregate_table(results: List[Dict[str, Any]], families: Tuple[str, ...]) -> str:
    rows = []
    overall_mapes = {fam: [] for fam in families}
    header = "<th>Station</th>" + "".join(f"<th>{FAMILY_LABELS[fam]} MAPE</th>" for fam in families) + "<th>Months matched</th>"
    for r in results:
        name = r["station"].get("station_name") or r["station"]["unique_scno"]
        cells = []
        n_matched = None
        for fam in families:
            cmp_ = r["comparisons"][fam]
            n_matched = cmp_["n_matched"]
            if cmp_["mape"] is not None:
                overall_mapes[fam].append(cmp_["mape"])
                cells.append(f"<td>{cmp_['mape']:.1f}%</td>")
            else:
                cells.append("<td>n/a</td>")
        rows.append(f"<tr><td>{name}</td>{''.join(cells)}<td>{n_matched}</td></tr>")

    summary_cells = []
    for fam in families:
        vals = overall_mapes[fam]
        summary_cells.append(f"{FAMILY_LABELS[fam]}: {np.mean(vals):.1f}%" if vals else f"{FAMILY_LABELS[fam]}: n/a")
    summary = "<p><b>Overall mean MAPE — " + " | ".join(summary_cells) + f" (across {len(results)} stations)</b></p>"

    table = f"<table><tr>{header}</tr>" + "".join(rows) + "</table>"
    return summary + table


def _station_card(result: Dict[str, Any], families: Tuple[str, ...]) -> str:
    img = f'<img src="data:image/png;base64,{result["plot"]}" style="width:100%;max-width:850px">' if result["plot"] else ""
    summary = " | ".join(
        f"{FAMILY_LABELS[fam]} MAPE {result['comparisons'][fam]['mape']:.1f}%"
        if result["comparisons"][fam]["mape"] is not None else f"{FAMILY_LABELS[fam]} MAPE n/a"
        for fam in families
    )
    n_matched = next(iter(result["comparisons"].values()))["n_matched"]

    subtables = ""
    for fam in families:
        cmp_ = result["comparisons"][fam]
        row_html = "".join(
            f"<tr><td>{r['feature_month']}</td><td>{r['months_since_active']}</td>"
            f"<td>{r['predicted_kwh']:.0f}</td><td>{r['predicted_kwh_lower']:.0f}-{r['predicted_kwh_upper']:.0f}</td>"
            f"<td>{r['actual_kwh'] if r['actual_kwh'] is not None else 'n/a'}</td>"
            f"<td>{r['pct_error'] if r['pct_error'] is not None else 'n/a'}{'%' if r['pct_error'] is not None else ''}</td></tr>"
            for r in cmp_["rows"]
        )
        subtables += f"""
  <table class="subtable">
    <caption style="text-align:left;font-weight:bold;margin-bottom:4px">{FAMILY_LABELS[fam]}</caption>
    <tr><th>Month</th><th>Months active</th><th>Predicted kWh</th><th>Range</th><th>Actual kWh</th><th>% error</th></tr>
    {row_html}
  </table>"""

    return f"""
<div class="card">
  <h3>{result['station'].get('station_name') or result['station']['unique_scno']} — {summary} ({n_matched} months matched)</h3>
  {img}
  {subtables}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    n: int = 5,
    seed: int = 42,
    min_months: int = DEFAULT_MIN_MONTHS,
    horizon_months: Optional[int] = None,
    family: str = "single",
    compare_families: bool = False,
    out: str = "reports/coldstart_validation.html",
):
    """
    horizon_months: cap on how many months to predict forward per station.
    None (default) predicts each station's own FULL available billing
    history span (e.g. a station with 28 months of real bills gets a
    28-month trajectory) — the fairest test of how far the cold-start
    models' ramp/stabilized design actually holds up, since it's not
    artificially truncated at some fixed number. Both stabilized-stage
    families (ml/train_coldstart_stabilized_single.py and
    ml/train_coldstart_stabilized_permonth.py) are trained on unbounded
    months_since_active >= 6, so neither is extrapolating outside its
    training range regardless of how long a station's history is.
    """
    from datetime import datetime
    from pathlib import Path

    holdout_stations = select_holdout_stations(n, seed, min_months)
    if not holdout_stations:
        log.error("No stations with >= %d months of history found — nothing to validate.", min_months)
        return

    families: Tuple[str, ...] = ("single", "permonth") if compare_families else (family,)
    excluded = [s["unique_scno"] for s in holdout_stations]
    holdout_models = train_holdout_coldstart_models(excluded, family="both" if compare_families else family)

    ramp_df = get_coldstart_training_matrix(RAMP_MIN, RAMP_MAX)
    ramp_df = ramp_df[~ramp_df["unique_scno"].isin(excluded)]
    n_ramp_rows, n_ramp_stations = len(ramp_df), ramp_df["unique_scno"].nunique()

    results = []
    max_horizon_used = 0
    for station in holdout_stations:
        scno = station["unique_scno"]
        series = actual_series(station["_history"])
        if not series:
            log.warning("SCNo %s has no non-zero billing history — skipping", scno)
            continue
        start_month = series[0][0][:7] + "-01"
        full_span = month_diff(series[0][0][:7], series[-1][0][:7])
        station_horizon = full_span if horizon_months is None else min(full_span, horizon_months)
        max_horizon_used = max(max_horizon_used, station_horizon)

        comparisons = {}
        trajectories = {}
        for fam in families:
            trajectory = build_coldstart_trajectory(
                station, holdout_models, start_month=start_month,
                horizon_months=station_horizon, family=fam,
            )
            comparisons[fam] = compare_trajectory_to_actuals(series, trajectory)
            trajectories[fam] = trajectory
            log.info("SCNo %s [%s]: %d-month trajectory (station has %d months of data), MAPE=%s%% over %d matched months",
                      scno, fam, station_horizon, full_span, comparisons[fam]["mape"], comparisons[fam]["n_matched"])

        plot = plot_station(scno, station.get("station_name"), series, trajectories)
        results.append({"station": station, "comparisons": comparisons, "plot": plot})

    html = _HTML_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_ramp_rows=n_ramp_rows, n_ramp_stations=n_ramp_stations, n_holdout=len(results),
        horizon_months=max_horizon_used, ramp_max=RAMP_MAX, stabilized_min=STABILIZED_MIN,
        aggregate_table=_aggregate_table(results, families),
        station_cards="".join(_station_card(r, families) for r in results),
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
    parser.add_argument("--horizon-months", type=int, default=None,
                        help="Cap on months to predict forward per station (default: each station's "
                             "own full available billing history span, uncapped)")
    parser.add_argument("--family", choices=("single", "permonth"), default="single",
                        help="Stabilized-stage architecture to use when not comparing both")
    parser.add_argument("--compare-families", action="store_true",
                        help="Train and report both Family A and Family B side by side")
    parser.add_argument("--out", default="reports/coldstart_validation.html")
    args = parser.parse_args()

    run_validation(
        n=args.n, seed=args.seed, min_months=args.min_months,
        horizon_months=args.horizon_months, family=args.family,
        compare_families=args.compare_families, out=args.out,
    )
