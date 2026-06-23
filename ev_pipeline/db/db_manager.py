# db/db_manager.py
"""
Database helper — psycopg2 with a persistent connection pool sized for Supabase.

Supabase free tier allows ~60 direct connections; pooler (pgBouncer) allows more.
We use a ThreadedConnectionPool (min=1, max=5) which is safe for the pipeline's
single-threaded use and won't exhaust Supabase limits.

SSL is always required — Supabase rejects non-SSL connections.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
import psycopg2.extras

from ..config.settings import DB_URL

log = logging.getLogger(__name__)

# ── Connection pool ───────────────────────────────────────────────────────────
# Initialised lazily on first use so import doesn't fail if DB is unreachable.
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        log.info("Initialising Supabase connection pool")
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=DB_URL,
            # Supabase requires SSL; sslmode=require is already in DB_URL
            # but we set it here too as a safety net
            sslmode="require",
        )
    return _pool


def close_pool() -> None:
    """Call this at process exit to cleanly return all connections."""
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        log.info("Connection pool closed")


@contextmanager
def get_conn() -> Generator:
    """Context manager: borrow a connection from the pool, auto-commit or rollback."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# Generic upsert helper
# ─────────────────────────────────────────────────────────────────────────────
JSONB_COLS = {"location_type_raw", "amenities", "tags"}  # add your jsonb cols here

def _upsert(table, data, conflict_cols, returning="id", jsonb_cols=JSONB_COLS):
    """
    Generic INSERT … ON CONFLICT DO UPDATE.
    Returns the value of `returning` column, or None.
    """
    if not data:
        raise ValueError(f"No data provided for upsert into {table}")
    sanitized = {
        k: psycopg2.extras.Json(v) if (k in jsonb_cols and isinstance(v, (list, dict))) else v
        for k, v in data.items()
    }
    sanitized = {
        k: psycopg2.extras.Json(v) if isinstance(v, (list, dict)) else v
        for k, v in data.items()
    }
    cols = list(sanitized.keys())
    vals = list(sanitized.values())
    placeholders = ", ".join(["%s"] * len(cols))
    set_clause = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in cols if c not in conflict_cols
    )
    conflict_str = ", ".join(conflict_cols)

    # Add updated_at if the table has it and we're not setting it already
    updated_at_clause = ""
    if table == "stations" and "updated_at" not in cols:
        updated_at_clause = ", updated_at = NOW()"

    returning_clause = f"RETURNING {returning}" if returning else ""

    sql = f"""
        INSERT INTO {table} ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_str}) DO UPDATE
            SET {set_clause}{updated_at_clause}
        {returning_clause}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, vals)
            if returning:
                row = cur.fetchone()
                return row[0] if row else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public write API
# ─────────────────────────────────────────────────────────────────────────────

def upsert_station(data: Dict[str, Any]) -> int:
    """Insert or update a station row. Returns station.id."""
    result = _upsert("stations", data, conflict_cols=["unique_scno"])
    return int(result)


def upsert_monthly_bill(data: Dict[str, Any]) -> int:
    """Insert or update a monthly bill row. Returns row id."""
    result = _upsert("monthly_bills", data, conflict_cols=["unique_scno", "bill_month"])
    return int(result)


def upsert_station_features(data: Dict[str, Any]) -> None:
    """Insert or update a station_features row."""
    _upsert(
        "station_features",
        data,
        conflict_cols=["unique_scno", "feature_month"],
        returning="id",
    )


def bulk_upsert_monthly_bills(rows: List[Dict[str, Any]]) -> int:
    """
    Efficient bulk upsert for monthly bills (e.g. after PDF import).
    Uses executemany with individual ON CONFLICT logic.
    Returns number of rows processed.
    """
    if not rows:
        return 0
    count = 0
    for row in rows:
        upsert_monthly_bill(row)
        count += 1
    log.info("Bulk upserted %d monthly bill rows", count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Public read API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_stations() -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM stations ORDER BY unique_scno")
            return [dict(r) for r in cur.fetchall()]


def get_station(unique_scno: str) -> Optional[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM stations WHERE unique_scno = %s", (unique_scno,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_station_billing_history(unique_scno: str) -> List[Dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM monthly_bills WHERE unique_scno = %s ORDER BY bill_month",
                (unique_scno,),
            )
            return [dict(r) for r in cur.fetchall()]


def get_monthly_weather(unique_scno: str, weather_month: str) -> Optional[Dict]:
    """Cached weather lookup — avoids re-hitting Open-Meteo for a month already fetched."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM monthly_weather WHERE unique_scno = %s AND weather_month = %s",
                (unique_scno, weather_month),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def upsert_monthly_weather(weather: Dict[str, Any]) -> None:
    """Write a freshly-fetched weather_scraper.fetch_monthly_weather() result
    into the cache (same pattern pipeline.py's _build_station_features uses
    inline) so the next get_monthly_weather() call for this (scno, month)
    is a cache hit instead of a live Open-Meteo call."""
    _upsert("monthly_weather", weather, conflict_cols=["unique_scno", "weather_month"], returning=None)


def get_latest_bill_month(unique_scno: str) -> Optional[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(bill_month) FROM monthly_bills WHERE unique_scno = %s",
                (unique_scno,),
            )
            row = cur.fetchone()
            return str(row[0]) if row and row[0] else None


def get_feature_vectors(
    unique_scno: Optional[str] = None,
    from_month: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch station_features rows for model training.
    Optionally filter by station or date range.
    """
    clauses = []
    params = []
    if unique_scno:
        clauses.append("unique_scno = %s")
        params.append(unique_scno)
    if from_month:
        clauses.append("feature_month >= %s")
        params.append(from_month)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM station_features {where} ORDER BY unique_scno, feature_month",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Derived-field backfill helpers (station_name, commissioning_date, bill summary)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_backfill_station_names() -> int:
    """
    Set station_name from consumer_name (fallback places_name) for any row
    where it's still NULL. Matches utils.helpers.derive_station_name's
    preference order.
    """
    sql = """
        UPDATE stations
        SET station_name = COALESCE(NULLIF(TRIM(consumer_name), ''), places_name)
        WHERE station_name IS NULL
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            n = cur.rowcount
    log.info("Backfilled station_name for %d stations", n)
    return n


def bulk_backfill_commissioning_dates() -> int:
    """commissioning_date mirrors supply_date — there's no separate EV-charger
    go-live date source in this pipeline (see README 'Known Data Gaps')."""
    sql = """
        UPDATE stations
        SET commissioning_date = supply_date
        WHERE commissioning_date IS NULL AND supply_date IS NOT NULL
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            n = cur.rowcount
    log.info("Backfilled commissioning_date for %d stations", n)
    return n


def refresh_all_station_bill_summaries() -> int:
    """
    Recompute stations.last_bill_month and avg_monthly_kwh_lifetime from
    monthly_bills for every station in one statement.
    """
    sql = """
        UPDATE stations s
        SET last_bill_month         = agg.last_month,
            avg_monthly_kwh_lifetime = agg.avg_kwh
        FROM (
            SELECT station_id,
                   MAX(bill_month)        AS last_month,
                   AVG(kwh_units)         AS avg_kwh
            FROM monthly_bills
            WHERE kwh_units IS NOT NULL
            GROUP BY station_id
        ) agg
        WHERE s.id = agg.station_id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            n = cur.rowcount
    log.info("Refreshed bill summary (last_bill_month, avg_monthly_kwh_lifetime) for %d stations", n)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Spatial layer write/read API (H3 coverage grid + fleet mix reference table)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_upsert_h3_coverage_zones(rows: List[Dict[str, Any]]) -> int:
    """
    Bulk insert/update h3_coverage_zones rows in a single multi-row statement.
    Resolution-7 grids over Telangana run into the tens of thousands of hexes,
    so this avoids the per-row round-trip cost of the generic _upsert() helper.
    """
    if not rows:
        return 0
    cols = [
        "h3_index", "resolution", "center_lat", "center_lng",
        "station_count", "nearest_station_dist_km", "is_coverage_gap",
        "highway_overlap",
    ]
    values = [tuple(row.get(c) for c in cols) for row in rows]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "h3_index")
    sql = f"""
        INSERT INTO h3_coverage_zones ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (h3_index) DO UPDATE SET {set_clause}
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values, page_size=1000)
    log.info("Bulk upserted %d h3_coverage_zones rows", len(rows))
    return len(rows)


def get_h3_coverage_zones(resolution: Optional[int] = None, gaps_only: bool = False) -> List[Dict]:
    clauses: List[str] = []
    params: List[Any] = []
    if resolution is not None:
        clauses.append("resolution = %s")
        params.append(resolution)
    if gaps_only:
        clauses.append("is_coverage_gap = true")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM h3_coverage_zones {where} ORDER BY resolution, h3_index",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


def upsert_zone_fleet_mix(data: Dict[str, Any]) -> None:
    """Insert or update a single zone_fleet_mix row, keyed on (zone_band, vehicle_type)."""
    _upsert("zone_fleet_mix", data, conflict_cols=["zone_band", "vehicle_type"], returning=None)


def get_zone_fleet_mix(zone_band: Optional[str] = None) -> List[Dict]:
    clauses: List[str] = []
    params: List[Any] = []
    if zone_band:
        clauses.append("zone_band = %s")
        params.append(zone_band)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM zone_fleet_mix {where} ORDER BY zone_band, vehicle_type",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# ML feature matrix / model run / prediction API (used by ml/eda.py, train.py, predict.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_matrix(min_months: int = 1, min_months_since_opening: Optional[int] = None) -> pd.DataFrame:
    """
    Load every station_features row for stations that have at least
    `min_months` rows of history. Used for EDA and training, where we need
    the full time series per station rather than just the latest row.

    min_months_since_opening, if set, excludes rows still in their ramp-up
    window (sf.months_since_opening < min_months_since_opening) — keeps the
    general/autoregressive model's training set focused on steady-state
    dynamics, since that's its only production use (ml/predict.py forecasts
    next month for stations with ongoing real history). Default None keeps
    today's behavior (no filter) for every other caller.
    """
    sql = """
        SELECT sf.*
        FROM station_features sf
        WHERE sf.unique_scno IN (
            SELECT unique_scno FROM station_features
            GROUP BY unique_scno
            HAVING COUNT(*) >= %s
        )
        AND (%s IS NULL OR sf.months_since_opening >= %s)
        ORDER BY sf.unique_scno, sf.feature_month
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(min_months, min_months_since_opening, min_months_since_opening))


def get_latest_feature_vectors() -> pd.DataFrame:
    """Most recent station_features row per station — model input for next-month inference."""
    sql = """
        SELECT DISTINCT ON (unique_scno) *
        FROM station_features
        ORDER BY unique_scno, feature_month DESC
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn)


def get_coldstart_training_matrix(min_month: int, max_month: Optional[int] = None) -> pd.DataFrame:
    """
    Training data for the cold-start ramp/stabilized models (ml/train_coldstart_*.py).

    Unlike get_feature_matrix(), this re-zeros each station's ramp clock at
    its first NON-ZERO bill (months_since_active), not supply_date — 74% of
    stations have a leading run of kwh_units=0 bills before real usage
    starts (commissioning/admin delay, not ramp-up signal), so those rows
    are dropped entirely via the first_active join. Target is that exact
    month's real kWh (mb.kwh_units), not next_month_kwh — there's no
    "next month" concept for a station with zero billing history; we're
    predicting the calendar month itself from ramp position + static
    geo/capacity/competition/calendar traits, deliberately excluding every
    consumption-history column (those are never available for a genuinely
    new station — see ml/site_simulator.py's module docstring).

    max_month=None means no upper bound (used to pull all stabilized-phase
    rows regardless of station age, for the per-calendar-month model).
    """
    sql = """
        WITH first_active AS (
            SELECT unique_scno, MIN(bill_month) AS first_active_month
            FROM monthly_bills
            WHERE kwh_units > 0
            GROUP BY unique_scno
        )
        SELECT
            sf.unique_scno, sf.feature_month,
            (DATE_PART('year', sf.feature_month) - DATE_PART('year', fa.first_active_month)) * 12
              + (DATE_PART('month', sf.feature_month) - DATE_PART('month', fa.first_active_month))
              AS months_since_active,
            sf.contracted_load_kva, sf.total_charger_count, sf.total_power_kw, sf.charger_mix_ratio,
            sf.dist_from_city_a_km, sf.dist_from_city_b_km, sf.dist_from_midpoint_km,
            sf.highway_position_ratio, sf.direction_side,
            sf.nearby_ev_stations_1km, sf.nearby_restaurants_1km, sf.nearby_hotels_1km,
            sf.nearby_petrol_pumps_1km, sf.competition_intensity, sf.amenity_score,
            sf.has_attached_restaurant, sf.location_type,
            sf.month_of_year, sf.is_summer, sf.is_monsoon, sf.is_festival_month,
            sf.avg_temp_c, sf.max_temp_c, sf.total_rainfall_mm, sf.heatwave_days, sf.rainfall_days,
            mb.kwh_units AS target_kwh
        FROM station_features sf
        JOIN monthly_bills mb
          ON mb.unique_scno = sf.unique_scno AND mb.bill_month = sf.feature_month
        JOIN first_active fa ON fa.unique_scno = sf.unique_scno
        WHERE sf.feature_month >= fa.first_active_month
          AND mb.kwh_units IS NOT NULL
          AND (DATE_PART('year', sf.feature_month) - DATE_PART('year', fa.first_active_month)) * 12
              + (DATE_PART('month', sf.feature_month) - DATE_PART('month', fa.first_active_month)) >= %s
          AND (%s IS NULL OR
               (DATE_PART('year', sf.feature_month) - DATE_PART('year', fa.first_active_month)) * 12
               + (DATE_PART('month', sf.feature_month) - DATE_PART('month', fa.first_active_month)) <= %s)
        ORDER BY sf.unique_scno, sf.feature_month
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(min_month, max_month, max_month))


def insert_model_run(metrics: Dict[str, Any]) -> int:
    """Insert a training run's metrics into model_runs. Returns the new row id."""
    data = {
        "model_version":       metrics.get("model_version"),
        "n_stations":          metrics.get("n_stations"),
        "n_training_rows":     metrics.get("n_training_rows"),
        "n_test_rows":         metrics.get("n_test_rows"),
        "cv_mae":              metrics.get("cv_mae"),
        "cv_rmse":             metrics.get("cv_rmse"),
        "cv_mape":             metrics.get("cv_mape"),
        "test_mae":            metrics.get("test_mae"),
        "test_rmse":           metrics.get("test_rmse"),
        "test_mape":           metrics.get("test_mape"),
        "feature_importances": metrics.get("feature_importances"),
        "hyperparameters":     metrics.get("hyperparameters"),
    }
    result = _upsert(
        "model_runs", data, conflict_cols=["model_version"], returning="id",
        jsonb_cols=JSONB_COLS | {"feature_importances", "hyperparameters"},
    )
    return int(result)


def upsert_prediction(data: Dict[str, Any]) -> None:
    """Insert or update a model_predictions row, keyed on (unique_scno, prediction_month)."""
    _upsert("model_predictions", data, conflict_cols=["unique_scno", "prediction_month"], returning=None)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def ping() -> bool:
    """Return True if the DB connection is healthy."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        log.error("DB ping failed: %s", e)
        return False