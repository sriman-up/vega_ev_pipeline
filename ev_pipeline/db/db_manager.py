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