from __future__ import annotations

from datetime import date
from pathlib import Path

from rksi_tmax.config import ProjectConfig
from rksi_tmax.metar_import import fetch_metar_text, import_metar_file
from rksi_tmax.storage import sync_duckdb_from_csv


def fetch_metar_for_stations(
    stations: list[str],
    hours: int,
    output_path: str | Path,
) -> dict[str, object]:
    normalized = [station.strip().upper() for station in stations if station.strip()]
    return fetch_metar_text(normalized, hours=hours, output_path=output_path)


def import_station_metar(
    config: ProjectConfig,
    metar_path: str | Path,
    reference_date: str | date | None,
) -> dict[str, object]:
    scoped_path = _station_scoped_metar_file(config.station, metar_path)
    return import_metar_file(
        metar_path=scoped_path,
        csv_path=config.input_csv,
        reference_date=reference_date,
        db_path=config.input_db if config.prefer_duckdb else None,
    )


def import_many_station_metars(
    configs: list[ProjectConfig],
    metar_path: str | Path,
    reference_date: str | date | None,
) -> dict[str, object]:
    results = []
    total_inserted = 0
    total_db_inserted = 0
    for config in configs:
        result = import_station_metar(config, metar_path, reference_date)
        results.append({"station": config.station, **result})
        total_inserted += int(result["inserted"])
        if result["db_inserted"] is not None:
            total_db_inserted += int(result["db_inserted"])
    return {
        "metar_file": str(metar_path),
        "stations": [config.station for config in configs],
        "inserted": total_inserted,
        "db_inserted": total_db_inserted,
        "results": results,
    }


def sync_station_database(config: ProjectConfig) -> dict[str, object]:
    return sync_duckdb_from_csv(config.raw_csv_files, config.input_db)


def sync_many_databases(configs: list[ProjectConfig]) -> dict[str, object]:
    results = []
    for config in configs:
        result = sync_station_database(config)
        results.append({"station": config.station, **result})
    return {
        "stations": [config.station for config in configs],
        "results": results,
    }


def _station_scoped_metar_file(station: str, metar_path: str | Path) -> Path:
    source = Path(metar_path)
    lines = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if _line_station(line) == station.upper()
    ]
    directory = Path("data") / station.lower()
    directory.mkdir(parents=True, exist_ok=True)
    scoped = directory / f"{source.stem}-{station.lower()}-scoped{source.suffix or '.txt'}"
    scoped.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return scoped


def _line_station(line: str) -> str | None:
    tokens = line.strip().replace("=", "").split()
    if not tokens:
        return None
    if tokens[0] in {"METAR", "SPECI"} and len(tokens) > 1:
        return tokens[1].upper()
    return tokens[0].upper()
