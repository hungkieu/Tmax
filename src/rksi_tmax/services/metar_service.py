from __future__ import annotations

from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.metar_import import fetch_metar_text, import_metar_file, parse_metar
from rksi_tmax.services import db_service, training_service
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


def update_live_data(
    configs: list[ProjectConfig],
    *,
    hours: int,
    metar_path: str | Path,
    reference_date: str | date | None,
    update_openmeteo: bool,
) -> dict[str, object]:
    if not configs:
        raise ValueError("At least one location config is required.")
    stations = [config.station for config in configs]
    fetch_result = fetch_metar_for_stations(stations, hours, metar_path)
    metar_summary = summarize_metar_file(configs, metar_path, reference_date)
    import_result = import_many_station_metars(configs, metar_path, reference_date)
    openmeteo_results = (
        _update_openmeteo_for_configs(configs) if update_openmeteo else _openmeteo_skipped(configs)
    )
    station_rows = _live_update_station_rows(configs, metar_summary, import_result, openmeteo_results)
    return {
        "stations": stations,
        "metar_file": str(metar_path),
        "hours": int(hours),
        "update_openmeteo": bool(update_openmeteo),
        "fetch": fetch_result,
        "metar_summary": metar_summary,
        "import": import_result,
        "openmeteo": openmeteo_results,
        "station_rows": station_rows,
        "warnings": _live_update_warnings(station_rows),
    }


def summarize_metar_file(
    configs: list[ProjectConfig],
    metar_path: str | Path,
    reference_date: str | date | None,
) -> dict[str, object]:
    config_by_station = {config.station.upper(): config for config in configs}
    rows_by_station: dict[str, list[dict[str, object]]] = {
        station: [] for station in config_by_station
    }
    path = Path(metar_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parse_errors = []
    for line in lines:
        station = _line_station(line)
        if station not in config_by_station:
            continue
        try:
            parsed = parse_metar(line, reference_date)
        except Exception as exc:
            parse_errors.append({"line": line, "error": str(exc)})
            continue
        row = parsed.row
        rows_by_station[station].append(
            {
                "valid_utc": parsed.valid,
                "tmpf": _csv_float(row.get("tmpf")),
                "drct": _csv_float(row.get("drct")),
                "sknt": _csv_float(row.get("sknt")),
                "gust": _csv_float(row.get("gust")),
                "cloud_layers": sum(
                    1
                    for column in ("skyc1", "skyc2", "skyc3", "skyc4")
                    if row.get(column) not in {None, "", "null"}
                ),
                "cavok": " CAVOK " in f" {line.upper()} ",
                "weather_codes": row.get("wxcodes"),
                "metar": row.get("metar"),
            }
        )

    station_summary = {}
    for station, rows in rows_by_station.items():
        config = config_by_station[station]
        station_summary[station] = _station_metar_summary(rows, config)
    return {
        "metar_file": str(path),
        "lines": len(lines),
        "stations": station_summary,
        "parse_errors": parse_errors,
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


def _update_openmeteo_for_configs(configs: list[ProjectConfig]) -> dict[str, object]:
    results = []
    for config in configs:
        local_date = pd.Timestamp.now(tz=ZoneInfo(config.timezone)).date().isoformat()
        before = training_service.openmeteo_daily_status(config, local_date)
        if not before["configured"]:
            results.append(
                {
                    "station": config.station,
                    "date": local_date,
                    "configured": False,
                    "exists_before": before["exists"],
                    "exists_after": before["exists"],
                    "fetched": False,
                    "created_new_file": False,
                    "output_path": before["output_path"],
                    "error": "Open-Meteo API is not configured for this location.",
                }
            )
            continue
        try:
            result = training_service.prepare_openmeteo_daily_data(config, local_date, force=False)
            after = training_service.openmeteo_daily_status(config, local_date)
            results.append(
                {
                    "station": config.station,
                    "date": local_date,
                    "configured": True,
                    "exists_before": before["exists"],
                    "exists_after": after["exists"],
                    "fetched": bool(result.get("fetched")),
                    "created_new_file": bool(result.get("created_new_file")),
                    "output_path": after["output_path"],
                    "size_bytes": after["size_bytes"],
                    "error": None,
                }
            )
        except Exception as exc:
            after = training_service.openmeteo_daily_status(config, local_date)
            results.append(
                {
                    "station": config.station,
                    "date": local_date,
                    "configured": True,
                    "exists_before": before["exists"],
                    "exists_after": after["exists"],
                    "fetched": False,
                    "created_new_file": False,
                    "output_path": after["output_path"],
                    "size_bytes": after["size_bytes"],
                    "error": str(exc),
                }
            )
    return {"results": results}


def _openmeteo_skipped(configs: list[ProjectConfig]) -> dict[str, object]:
    return {
        "results": [
            {
                "station": config.station,
                "configured": config.openmeteo_latitude is not None
                and config.openmeteo_longitude is not None
                and config.openmeteo_live_json_pattern is not None,
                "fetched": False,
                "created_new_file": False,
                "error": "Skipped by user.",
            }
            for config in configs
        ]
    }


def _live_update_station_rows(
    configs: list[ProjectConfig],
    metar_summary: dict[str, object],
    import_result: dict[str, object],
    openmeteo_results: dict[str, object],
) -> list[dict[str, object]]:
    import_by_station = {
        row["station"]: row for row in import_result.get("results", []) if isinstance(row, dict)
    }
    openmeteo_by_station = {
        row["station"]: row for row in openmeteo_results.get("results", []) if isinstance(row, dict)
    }
    summary_by_station = metar_summary.get("stations", {})
    rows = []
    for config in configs:
        station_summary = summary_by_station.get(config.station, {})
        imported = import_by_station.get(config.station, {})
        openmeteo = openmeteo_by_station.get(config.station, {})
        status = db_service.database_status(config)
        latest = db_service.latest_observation(config) if status.get("exists") else None
        rows.append(
            {
                "station": config.station,
                "metar_lines": station_summary.get("lines", 0),
                "metar_valid_start_local": station_summary.get("valid_start_local"),
                "metar_valid_end_local": station_summary.get("valid_end_local"),
                "temps_missing": station_summary.get("temps_missing", 0),
                "wind_missing": station_summary.get("wind_missing", 0),
                "cloud_missing": station_summary.get("cloud_missing", 0),
                "inserted_csv": imported.get("inserted", 0),
                "inserted_db": imported.get("db_inserted"),
                "skipped_existing": imported.get("skipped_existing", 0),
                "openmeteo_date": openmeteo.get("date"),
                "openmeteo_configured": openmeteo.get("configured"),
                "openmeteo_exists_after": openmeteo.get("exists_after"),
                "openmeteo_fetched": openmeteo.get("fetched"),
                "openmeteo_error": openmeteo.get("error"),
                "db_rows": status.get("row_count", 0),
                "db_latest_valid_utc": status.get("latest_valid"),
                "latest_temp_c": latest.get("tmpc") if latest else None,
            }
        )
    return rows


def _live_update_warnings(station_rows: list[dict[str, object]]) -> list[str]:
    warnings = []
    for row in station_rows:
        station = row["station"]
        if int(row.get("metar_lines") or 0) == 0:
            warnings.append(f"{station}: no METAR lines were fetched for this update window.")
        if int(row.get("temps_missing") or 0) > 0:
            warnings.append(f"{station}: {row['temps_missing']} METAR lines are missing temperature.")
        if int(row.get("wind_missing") or 0) > 0:
            warnings.append(f"{station}: {row['wind_missing']} METAR lines are missing wind.")
        if row.get("openmeteo_error") not in {None, "Skipped by user."}:
            warnings.append(f"{station}: Open-Meteo update failed: {row['openmeteo_error']}")
        if not row.get("openmeteo_exists_after") and row.get("openmeteo_configured"):
            warnings.append(f"{station}: Open-Meteo daily cache is still missing.")
    return warnings


def _station_metar_summary(rows: list[dict[str, object]], config: ProjectConfig) -> dict[str, object]:
    if not rows:
        return {
            "lines": 0,
            "valid_start_utc": None,
            "valid_end_utc": None,
            "valid_start_local": None,
            "valid_end_local": None,
            "temps_missing": 0,
            "wind_missing": 0,
            "cloud_missing": 0,
        }
    valid_times = [row["valid_utc"] for row in rows]
    start = min(valid_times)
    end = max(valid_times)
    return {
        "lines": len(rows),
        "valid_start_utc": start.strftime("%Y-%m-%d %H:%M"),
        "valid_end_utc": end.strftime("%Y-%m-%d %H:%M"),
        "valid_start_local": pd.Timestamp(start, tz="UTC").tz_convert(config.timezone).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "valid_end_local": pd.Timestamp(end, tz="UTC").tz_convert(config.timezone).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "temps_missing": sum(1 for row in rows if row["tmpf"] is None),
        "wind_missing": sum(1 for row in rows if row["sknt"] is None),
        "cloud_missing": sum(
            1 for row in rows if int(row["cloud_layers"] or 0) == 0 and not row["cavok"]
        ),
        "weather_code_lines": sum(1 for row in rows if row["weather_codes"] not in {None, "", "null"}),
    }


def _csv_float(value: object) -> float | None:
    if value in {None, "", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
