# ml/eda.py
"""
Exploratory Data Analysis for EV station feature matrix.

Produces a self-contained HTML report covering:
  1. Data completeness audit (missing values per feature)
  2. Target distribution (next_month_kwh histogram + per-station box plots)
  3. Feature correlation heatmap (numeric features vs target)
  4. Per-station time series (actual kWh over months)
  5. Seasonality (monthly average across all stations)
  6. Ramp-up curve (avg kWh by months_since_opening)
  7. Competition vs usage scatter (nearby_ev_stations_1km vs avg kWh)
  8. Highway position vs usage (position_ratio vs avg kWh)

Usage:
    python -m ev_pipeline.ml.eda --out reports/eda.html
    # or from pipeline:
    from ev_pipeline.ml.eda import run_eda
    run_eda(output_path="reports/eda.html")
"""

import argparse
import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Optional matplotlib — fail gracefully if not installed ───────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    log.warning("matplotlib not installed — plots will be skipped in EDA report")


# ─────────────────────────────────────────────────────────────────────────────
# Feature groups (for correlation heatmap)
# ─────────────────────────────────────────────────────────────────────────────

CONSUMPTION_FEATURES = [
    "rolling_avg_3m_kwh", "rolling_avg_6m_kwh", "avg_kwh_units",
    "std_kwh_units", "kwh_growth_rate_overall", "months_active",
    "pct_months_zero_consumption", "kwh_coefficient_of_variation",
    "seasonal_summer_avg_kwh", "seasonal_winter_avg_kwh",
]
CAPACITY_FEATURES = [
    "contracted_load_kva", "total_charger_count", "total_power_kw",
    "charger_mix_ratio", "power_utilisation_pct",
]
GEO_FEATURES = [
    "dist_from_city_a_km", "dist_from_city_b_km", "dist_from_midpoint_km",
    "highway_position_ratio",
]
COMPETITION_FEATURES = [
    "nearby_ev_stations_1km", "nearby_restaurants_1km", "nearby_hotels_1km",
    "nearby_petrol_pumps_1km", "competition_intensity", "amenity_score",
]
CALENDAR_FEATURES = [
    "month_of_year", "is_summer", "is_monsoon", "is_festival_month",
    "months_since_opening",
]
WEATHER_FEATURES = [
    "avg_temp_c", "max_temp_c", "total_rainfall_mm",
    "heatwave_days", "rainfall_days",
]

ALL_FEATURE_GROUPS = {
    "Consumption History":  CONSUMPTION_FEATURES,
    "Station Capacity":     CAPACITY_FEATURES,
    "Geo / Highway":        GEO_FEATURES,
    "Competition":          COMPETITION_FEATURES,
    "Calendar":             CALENDAR_FEATURES,
    "Weather":              WEATHER_FEATURES,
}

TARGET = "next_month_kwh"


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:900px">'


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing(df: pd.DataFrame) -> str:
    all_cols = [c for group in ALL_FEATURE_GROUPS.values() for c in group]
    existing = [c for c in all_cols if c in df.columns]
    pct_missing = (df[existing].isna().mean() * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(existing) * 0.25)))
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#2ecc71"
              for v in pct_missing.values]
    pct_missing.plot.barh(ax=ax, color=colors)
    ax.axvline(20, color="#f39c12", linestyle="--", linewidth=0.8, label="20% threshold")
    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=0.8, label="50% threshold")
    ax.set_xlabel("% Missing")
    ax.set_title("Feature Completeness (% Missing Values)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_target_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(df[TARGET].dropna(), bins=30, color="#3498db", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("next_month_kwh")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Target Distribution")
    median_val = df[TARGET].median()
    axes[0].axvline(median_val, color="#e74c3c", linestyle="--",
                    label=f"Median: {median_val:.0f}")
    axes[0].legend()

    # Per-station box plot
    station_data = [grp[TARGET].dropna().values
                    for _, grp in df.groupby("unique_scno")
                    if grp[TARGET].dropna().shape[0] > 0]
    labels = [scno[-6:] for scno in df.groupby("unique_scno").groups.keys()]
    axes[1].boxplot(station_data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.6))
    axes[1].set_xlabel("Station (last 6 digits of SCNo)")
    axes[1].set_ylabel("kWh")
    axes[1].set_title("kWh Distribution per Station")
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    all_cols = [c for group in ALL_FEATURE_GROUPS.values() for c in group] + [TARGET]
    existing = [c for c in all_cols if c in df.columns]
    numeric = df[existing].select_dtypes(include=[np.number])

    corr = numeric.corr()[TARGET].drop(TARGET).sort_values()

    fig, ax = plt.subplots(figsize=(6, max(6, len(corr) * 0.28)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr.values]
    corr.plot.barh(ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson r with {TARGET}")
    ax.set_title("Feature Correlation with Target")
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_time_series(df: pd.DataFrame) -> str:
    stations = df["unique_scno"].unique()
    n = len(stations)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    for i, scno in enumerate(sorted(stations)):
        ax = axes[i // cols][i % cols]
        sub = df[df["unique_scno"] == scno].sort_values("feature_month")
        ax.plot(pd.to_datetime(sub["feature_month"]), sub[TARGET],
                marker="o", markersize=3, linewidth=1.5, color="#3498db")
        ax.set_title(f"...{scno[-6:]}", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Hide empty subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Per-Station Monthly kWh (target)", fontsize=12, y=1.01)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_seasonality(df: pd.DataFrame) -> str:
    if "month_of_year" not in df.columns:
        df = df.copy()
        df["month_of_year"] = pd.to_datetime(df["feature_month"]).dt.month

    monthly = df.groupby("month_of_year")[TARGET].agg(["mean", "std"]).reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["month_of_year"], monthly["mean"],
           yerr=monthly["std"], capsize=4,
           color="#3498db", alpha=0.8, error_kw={"linewidth": 1})
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Avg next_month_kwh")
    ax.set_title("Seasonality — Average Monthly kWh Across All Stations")
    # Shade monsoon and festival periods
    ax.axvspan(6.5, 9.5, alpha=0.08, color="#3498db", label="Monsoon (Jul-Sep)")
    ax.axvspan(9.5, 11.5, alpha=0.08, color="#e67e22", label="Festival (Oct-Nov)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_rampup(df: pd.DataFrame) -> str:
    if "months_since_opening" not in df.columns:
        return ""
    ramp = df.groupby("months_since_opening")[TARGET].mean().reset_index()
    ramp = ramp[ramp["months_since_opening"] <= 24]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ramp["months_since_opening"], ramp[TARGET],
            marker="o", markersize=4, color="#2ecc71", linewidth=2)
    ax.axvline(6, color="#e74c3c", linestyle="--", linewidth=0.9,
               label="End of ramp-up (6 months)")
    ax.set_xlabel("Months Since Opening")
    ax.set_ylabel("Avg kWh")
    ax.set_title("Ramp-Up Curve — Average kWh by Station Age")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_competition(df: pd.DataFrame) -> str:
    if "nearby_ev_stations_1km" not in df.columns:
        return ""
    comp = df.groupby("nearby_ev_stations_1km")[TARGET].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comp["nearby_ev_stations_1km"].astype(str), comp[TARGET],
           color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Nearby EV Stations (1 km)")
    ax.set_ylabel("Avg next_month_kwh")
    ax.set_title("Competition Effect on Usage")
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_highway_position(df: pd.DataFrame) -> str:
    if "highway_position_ratio" not in df.columns or df["highway_position_ratio"].isna().all():
        return ""
    sub = df.dropna(subset=["highway_position_ratio", TARGET])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(sub["highway_position_ratio"], sub[TARGET],
               alpha=0.5, s=20, color="#9b59b6")
    # Trend line
    z = np.polyfit(sub["highway_position_ratio"], sub[TARGET], 2)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#e74c3c", linewidth=1.5,
            label="Quadratic trend")
    ax.set_xlabel("Highway Position (0=City A, 1=City B)")
    ax.set_ylabel("next_month_kwh")
    ax.set_title("kWh by Highway Position")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Midpoint")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Data audit table
# ─────────────────────────────────────────────────────────────────────────────

def build_audit_table(df: pd.DataFrame) -> str:
    rows_html = []
    for group_name, cols in ALL_FEATURE_GROUPS.items():
        existing = [c for c in cols if c in df.columns]
        missing  = [c for c in cols if c not in df.columns]
        for c in existing:
            pct = df[c].isna().mean() * 100
            color = "color:#e74c3c" if pct > 50 else "color:#e67e22" if pct > 20 else ""
            rows_html.append(
                f"<tr><td>{group_name}</td><td>{c}</td>"
                f"<td style='{color}'>{pct:.1f}%</td>"
                f"<td>{df[c].dtype}</td>"
                f"<td>{df[c].dropna().describe().to_dict().get('mean', '')!s:.40}</td></tr>"
            )
        for c in missing:
            rows_html.append(
                f"<tr style='color:#aaa'><td>{group_name}</td><td>{c}</td>"
                f"<td>NOT IN DB</td><td>—</td><td>—</td></tr>"
            )
    return (
        "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:12px'>"
        "<tr><th>Group</th><th>Feature</th><th>% Missing</th><th>dtype</th><th>Mean (sample)</th></tr>"
        + "".join(rows_html)
        + "</table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML report assembly
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EV Station EDA Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1100px;
          margin: 0 auto; padding: 20px; background: #f8f9fa; color: #2c3e50; }}
  h1   {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
  h2   {{ color: #34495e; margin-top: 40px; border-left: 4px solid #3498db;
          padding-left: 10px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
               margin: 20px 0; }}
  .stat {{ background: white; padding: 16px; border-radius: 8px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.1); text-align: center; }}
  .stat .num {{ font-size: 2em; font-weight: bold; color: #3498db; }}
  .stat .lbl {{ font-size: 0.85em; color: #7f8c8d; margin-top: 4px; }}
  .warn {{ background: #fef9e7; border-left: 4px solid #f39c12;
           padding: 10px 14px; margin: 12px 0; border-radius: 4px; }}
  .img-wrap {{ background: white; padding: 12px; border-radius: 8px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin: 16px 0; }}
  table {{ background: white; }}
  footer {{ margin-top: 60px; font-size: 0.8em; color: #aaa; text-align: center; }}
</style>
</head>
<body>
<h1>EV Station Performance — EDA Report</h1>
<p style="color:#7f8c8d">Generated {generated_at}</p>

<div class="stat-grid">
  <div class="stat"><div class="num">{n_stations}</div><div class="lbl">Stations</div></div>
  <div class="stat"><div class="num">{n_rows}</div><div class="lbl">Feature Rows</div></div>
  <div class="stat"><div class="num">{avg_months:.1f}</div><div class="lbl">Avg Months / Station</div></div>
  <div class="stat"><div class="num">{median_kwh:.0f}</div><div class="lbl">Median Monthly kWh</div></div>
</div>

{warnings_html}

<h2>1. Feature Completeness</h2>
<div class="img-wrap">{missing_plot}</div>
{audit_table}

<h2>2. Target Distribution</h2>
<div class="img-wrap">{target_dist_plot}</div>

<h2>3. Feature Correlation with Target (next_month_kwh)</h2>
<div class="img-wrap">{correlation_plot}</div>

<h2>4. Per-Station Time Series</h2>
<div class="img-wrap">{time_series_plot}</div>

<h2>5. Seasonality</h2>
<div class="img-wrap">{seasonality_plot}</div>

<h2>6. Ramp-Up Curve</h2>
<div class="img-wrap">{rampup_plot}</div>

<h2>7. Competition Effect</h2>
<div class="img-wrap">{competition_plot}</div>

<h2>8. Highway Position vs Usage</h2>
<div class="img-wrap">{highway_plot}</div>

<footer>EV Pipeline EDA &bull; {generated_at}</footer>
</body>
</html>"""


def _warn_block(warnings):
    if not warnings:
        return ""
    items = "".join(f"<li>{w}</li>" for w in warnings)
    return f'<div class="warn"><strong>⚠ Data Quality Warnings</strong><ul>{items}</ul></div>'


def _safe_plot(fn, *args) -> str:
    if not _HAS_MPL:
        return "<p><em>matplotlib not installed — install it to see this plot.</em></p>"
    try:
        b64 = fn(*args)
        return _img_tag(b64) if b64 else "<p><em>Insufficient data for this plot.</em></p>"
    except Exception as e:
        log.warning("Plot %s failed: %s", fn.__name__, e)
        return f"<p><em>Plot unavailable: {e}</em></p>"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(
    df: Optional[pd.DataFrame] = None,
    output_path: str = "reports/eda.html",
    min_months: int = 3,
):
    """
    Run full EDA and write HTML report.

    Args:
        df:           Pre-loaded feature matrix. If None, loads from DB.
        output_path:  Where to write the HTML file.
        min_months:   Min history rows per station to include.
    """
    if df is None:
        from ev_pipeline.db.db_manager import get_feature_matrix
        log.info("Loading feature matrix from DB...")
        df = get_feature_matrix(min_months=min_months)

    if df.empty:
        log.error("No data available for EDA. Run seed + feature build first.")
        return

    log.info("EDA: %d rows, %d stations, %d features",
             len(df), df["unique_scno"].nunique(), len(df.columns))

    # ── Summary stats ────────────────────────────────────────────────────────
    n_stations  = df["unique_scno"].nunique()
    n_rows      = len(df)
    avg_months  = df.groupby("unique_scno").size().mean()
    median_kwh  = df[TARGET].median() if TARGET in df.columns else 0

    # ── Data quality warnings ─────────────────────────────────────────────────
    warnings = []
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.5]
    if high_missing:
        warnings.append(f"Features with >50% missing: {', '.join(high_missing[:8])}")
    if n_rows < 50:
        warnings.append(
            f"Only {n_rows} training rows — model reliability will be limited. "
            "Seed more stations or wait for more monthly history."
        )
    if n_stations < 10:
        warnings.append(
            f"Only {n_stations} stations — GroupKFold CV may not be representative."
        )
    zero_pct = (df[TARGET] == 0).mean() * 100 if TARGET in df.columns else 0
    if zero_pct > 10:
        warnings.append(f"{zero_pct:.1f}% of target rows are zero — consider log(1+x) transform.")

    # ── Build plots ───────────────────────────────────────────────────────────
    html = _HTML_TEMPLATE.format(
        generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_stations     = n_stations,
        n_rows         = n_rows,
        avg_months     = avg_months,
        median_kwh     = median_kwh,
        warnings_html  = _warn_block(warnings),
        missing_plot   = _safe_plot(plot_missing, df),
        audit_table    = build_audit_table(df),
        target_dist_plot    = _safe_plot(plot_target_distribution, df),
        correlation_plot    = _safe_plot(plot_correlation_heatmap, df),
        time_series_plot    = _safe_plot(plot_time_series, df),
        seasonality_plot    = _safe_plot(plot_seasonality, df),
        rampup_plot         = _safe_plot(plot_rampup, df),
        competition_plot    = _safe_plot(plot_competition, df),
        highway_plot        = _safe_plot(plot_highway_position, df),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("EDA report written to %s", out.resolve())
    return df  # return df so caller can chain into train


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/eda.html")
    parser.add_argument("--min_months", type=int, default=3)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    run_eda(output_path=args.out, min_months=args.min_months)