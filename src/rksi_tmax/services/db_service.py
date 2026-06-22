from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.storage import OBSERVATION_TABLE, read_station_observations_from_duckdb


def database_status(config: ProjectConfig) -> dict[str, object]:
    db_path = Path(config.input_db)
    status: dict[str, object] = {
        "db_path": str(db_path),
        "exists": db_path.exists(),
        "station": config.station,
        "row_count": 0,
        "latest_valid": None,
        "stations": [],
    }
    if not db_path.exists():
        return status

    with duckdb.connect(str(db_path), read_only=True) as connection:
        try:
            status["stations"] = [
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT station FROM {OBSERVATION_TABLE} ORDER BY station"
                ).fetchall()
            ]
            row_count, latest_valid = connection.execute(
                f"""
                SELECT count(*), max(valid)
                FROM {OBSERVATION_TABLE}
                WHERE station = ?
                """,
                [config.station],
            ).fetchone()
            status["row_count"] = int(row_count)
            status["latest_valid"] = latest_valid
        except duckdb.Error as exc:
            status["error"] = str(exc)
    return status


def latest_observation(config: ProjectConfig) -> dict[str, object] | None:
    observations = read_station_observations_from_duckdb(
        config.input_db,
        config.station,
        ["station", "valid", "tmpf", "metar"],
    )
    if observations.empty:
        return None
    latest = observations.sort_values("valid").iloc[-1]
    tmpf = pd.to_numeric(pd.Series([latest.get("tmpf")]), errors="coerce").iloc[0]
    tmpc = None if pd.isna(tmpf) else (float(tmpf) - 32.0) * (5.0 / 9.0)
    return {
        "station": latest.get("station"),
        "valid": latest.get("valid"),
        "tmpf": None if pd.isna(tmpf) else float(tmpf),
        "tmpc": tmpc,
        "metar": latest.get("metar"),
    }


def latest_local_date_cutoff(config: ProjectConfig) -> tuple[str, str] | None:
    latest = latest_observation(config)
    if latest is None or latest["valid"] is None:
        return None
    valid = pd.to_datetime(latest["valid"], utc=True, errors="coerce")
    if pd.isna(valid):
        return None
    local = valid.tz_convert(config.timezone)
    return local.date().isoformat(), f"{local.hour:02d}:{local.minute:02d}"
