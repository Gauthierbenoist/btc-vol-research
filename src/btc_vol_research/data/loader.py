"""Lecture des snapshots btc_options depuis Neon."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from btc_vol_research.config import PostgresConfig

if TYPE_CHECKING:
    import psycopg2.extensions

_COLUMNS = """
    snapshot_date, snapshot_utc, instrument_name, expiry_code, maturity_date,
    strike, option_type, underlying_price, bid_price, ask_price, mid_price,
    mark_price, mark_iv, open_interest, volume_24h, time_to_expiry_years
"""


def get_connection(cfg: PostgresConfig | None = None):
    import psycopg2

    cfg = cfg or PostgresConfig()
    if not cfg.is_configured:
        raise RuntimeError("DATABASE_URL manquant — copiez .env.example vers .env")
    return psycopg2.connect(cfg.database_url)


def list_snapshot_dates(conn: "psycopg2.extensions.connection | None" = None) -> list[date]:
    own = conn is None
    conn = conn or get_connection()
    try:
        df = pd.read_sql(
            "SELECT snapshot_date, COUNT(*) AS n FROM btc_options GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
        return [d.date() if hasattr(d, "date") else d for d in df["snapshot_date"]]
    finally:
        if own:
            conn.close()


def latest_snapshot_date(conn: "psycopg2.extensions.connection | None" = None) -> date:
    own = conn is None
    conn = conn or get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(snapshot_date) FROM btc_options")
        row = cur.fetchone()
        if not row or row[0] is None:
            raise RuntimeError("Aucun snapshot dans btc_options")
        return row[0]
    finally:
        if own:
            conn.close()


def load_snapshot(
    snapshot_date: date | str | None = None,
    conn: "psycopg2.extensions.connection | None" = None,
) -> pd.DataFrame:
    """Charge toutes les lignes d'une date snapshot."""
    own = conn is None
    conn = conn or get_connection()
    try:
        if snapshot_date is None:
            snapshot_date = latest_snapshot_date(conn)
        query = f"SELECT {_COLUMNS} FROM btc_options WHERE snapshot_date = %s ORDER BY maturity_date, strike"
        df = pd.read_sql(query, conn, params=(snapshot_date,))
        if df.empty:
            raise RuntimeError(f"Aucune donnée pour snapshot_date={snapshot_date}")
        df["maturity_date"] = pd.to_datetime(df["maturity_date"], utc=True)
        df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
        return df
    finally:
        if own:
            conn.close()
