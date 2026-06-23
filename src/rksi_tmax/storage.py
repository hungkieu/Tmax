from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


OBSERVATION_TABLE = "observations"

ASOS_COLUMNS = [
    "station",
    "valid",
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "wxcodes",
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust",
    "peak_wind_drct",
    "peak_wind_time",
    "feel",
    "metar",
    "snowdepth",
]


def sync_duckdb_from_csv(
    csv_paths: Iterable[str | Path],
    db_path: str | Path,
    table: str = OBSERVATION_TABLE,
) -> dict:
    paths = [Path(path) for path in csv_paths]
    existing = [path for path in paths if path.exists()]
    missing = [str(path) for path in paths if not path.exists()]
    if not existing:
        raise FileNotFoundError(f"No CSV files found: {', '.join(str(path) for path in paths)}")

    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    path_list = [str(path) for path in existing]

    with duckdb.connect(str(database)) as connection:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {table}_raw AS
            SELECT * FROM read_csv(
                ?,
                header = true,
                all_varchar = true,
                nullstr = ['null', 'M', ''],
                ignore_errors = true,
                strict_mode = false,
                union_by_name = true
            )
            """,
            [path_list],
        )
        _ensure_columns(connection, f"{table}_raw")
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {table}_incoming AS
            SELECT {", ".join(_quote(column) for column in ASOS_COLUMNS)}
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY station, valid
                        ORDER BY metar DESC NULLS LAST
                    ) AS rn
                FROM {table}_raw
                WHERE station IS NOT NULL AND valid IS NOT NULL
            )
            WHERE rn = 1
            ORDER BY station, valid
            """
        )
        _create_table_if_missing(connection, table)
        connection.execute(
            f"""
            DELETE FROM {table} existing
            WHERE EXISTS (
                SELECT 1
                FROM {table}_incoming incoming
                WHERE existing.station = incoming.station
                  AND existing.valid = incoming.valid
            )
            """
        )
        connection.execute(
            f"""
            INSERT INTO {table}
            SELECT {", ".join(_quote(column) for column in ASOS_COLUMNS)}
            FROM {table}_incoming
            """
        )
        connection.execute(f"DROP TABLE {table}_incoming")
        connection.execute(f"DROP TABLE {table}_raw")
        row_count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        stations = [
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT station FROM {table} ORDER BY station"
            ).fetchall()
        ]

    return {
        "db_path": str(database),
        "table": table,
        "csv_files": [str(path) for path in existing],
        "missing_csv_files": missing,
        "rows": int(row_count),
        "stations": stations,
    }


def read_station_observations_from_duckdb(
    db_path: str | Path,
    station: str,
    columns: Iterable[str],
    table: str = OBSERVATION_TABLE,
) -> pd.DataFrame:
    database = Path(db_path)
    if not database.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database}")

    selected = list(dict.fromkeys(columns))
    with duckdb.connect(str(database), read_only=True) as connection:
        existing_columns = _table_columns(connection, table)
        available = [column for column in selected if column in existing_columns]
        if "station" not in available:
            available.insert(0, "station")
        if "valid" not in available:
            available.insert(1, "valid")
        query = (
            f"SELECT {', '.join(_quote(column) for column in available)} "
            f"FROM {table} WHERE station = ? ORDER BY valid"
        )
        return connection.execute(query, [station]).fetchdf()


def upsert_observation_rows(
    rows: list[dict[str, str]],
    db_path: str | Path,
    table: str = OBSERVATION_TABLE,
) -> dict:
    if not rows:
        return {"db_path": str(db_path), "table": table, "inserted": 0}

    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in ASOS_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[ASOS_COLUMNS].astype("string")

    with duckdb.connect(str(database)) as connection:
        _create_table_if_missing(connection, table)
        before = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        connection.register("incoming_rows", frame)
        connection.execute(
            f"""
            INSERT INTO {table}
            SELECT {", ".join(_quote(column) for column in ASOS_COLUMNS)}
            FROM incoming_rows incoming
            WHERE NOT EXISTS (
                SELECT 1
                FROM {table} existing
                WHERE existing.station = incoming.station
                  AND existing.valid = incoming.valid
            )
            """
        )
        after = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    return {"db_path": str(database), "table": table, "inserted": int(after - before)}


def _ensure_columns(connection: duckdb.DuckDBPyConnection, table: str) -> None:
    existing = set(_table_columns(connection, table))
    for column in ASOS_COLUMNS:
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {_quote(column)} VARCHAR")


def _create_table_if_missing(connection: duckdb.DuckDBPyConnection, table: str) -> None:
    columns = ", ".join(f"{_quote(column)} VARCHAR" for column in ASOS_COLUMNS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")


def _table_columns(connection: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    try:
        return [row[1] for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()]
    except duckdb.CatalogException:
        return []


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
