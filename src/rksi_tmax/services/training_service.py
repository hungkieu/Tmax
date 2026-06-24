from __future__ import annotations

from pathlib import Path

from rksi_tmax.config import ProjectConfig
from rksi_tmax.features import load_observations
from rksi_tmax.heat_risk import (
    _complete_observation_date_range,
    _openmeteo_training_date_range,
    build_heat_risk_dataset,
    train_heat_risk_model,
    validate_heat_risk_model,
)
from rksi_tmax.openmeteo import ensure_openmeteo_live_data, ensure_openmeteo_training_data


def openmeteo_status(config: ProjectConfig) -> dict[str, object]:
    history_path = Path(config.openmeteo_history_json) if config.openmeteo_history_json else None
    return {
        "configured": config.openmeteo_latitude is not None and config.openmeteo_longitude is not None,
        "latitude": config.openmeteo_latitude,
        "longitude": config.openmeteo_longitude,
        "timezone": config.openmeteo_timezone,
        "history_json": str(history_path) if history_path else None,
        "history_json_exists": bool(history_path and history_path.exists()),
        "live_json_pattern": config.openmeteo_live_json_pattern,
    }


def openmeteo_daily_status(config: ProjectConfig, local_date: str) -> dict[str, object]:
    path = (
        Path(config.openmeteo_live_json_pattern.format(date=local_date))
        if config.openmeteo_live_json_pattern
        else None
    )
    return {
        "configured": config.openmeteo_live_json_pattern is not None
        and config.openmeteo_latitude is not None
        and config.openmeteo_longitude is not None,
        "date": local_date,
        "output_path": str(path) if path else None,
        "exists": bool(path and path.exists()),
        "size_bytes": path.stat().st_size if path and path.exists() else None,
    }


def prepare_openmeteo_training_data(
    config: ProjectConfig,
    force: bool = False,
) -> dict[str, object]:
    observations = load_observations(config.input_csv, config)
    date_range = _complete_observation_date_range(observations, config)
    if date_range is None:
        raise ValueError("No completed observation days found for Open-Meteo training data.")
    date_range = _openmeteo_training_date_range(config, date_range)
    if date_range[0] > date_range[1]:
        raise ValueError(
            "Open-Meteo training date range is empty after applying config bounds: "
            f"start_date={date_range[0]}, end_date={date_range[1]}. "
            "Update openmeteo_training_start_date/openmeteo_training_end_date or add observations "
            "within that range."
        )
    result = ensure_openmeteo_training_data(
        config.openmeteo_history_json,
        config.openmeteo_latitude,
        config.openmeteo_longitude,
        date_range[0],
        date_range[1],
        timezone=config.openmeteo_timezone,
        force=force,
    )
    if result is None:
        raise ValueError("Open-Meteo API is not configured for this location.")
    return result


def prepare_openmeteo_daily_data(
    config: ProjectConfig,
    local_date: str,
    force: bool = False,
) -> dict[str, object]:
    before = openmeteo_daily_status(config, local_date)
    result = ensure_openmeteo_live_data(
        config.openmeteo_live_json_pattern,
        config.openmeteo_latitude,
        config.openmeteo_longitude,
        local_date,
        timezone=config.openmeteo_timezone,
        force=force,
    )
    if result is None:
        raise ValueError("Open-Meteo API is not configured for this location.")
    after = openmeteo_daily_status(config, local_date)
    result["file_existed_before"] = before["exists"]
    result["file_exists_after"] = after["exists"]
    result["created_new_file"] = bool(after["exists"] and not before["exists"])
    result["size_bytes"] = after["size_bytes"]
    return result


def build_dataset(config: ProjectConfig) -> dict[str, object]:
    dataset = build_heat_risk_dataset(config)
    return {
        "rows": int(len(dataset)),
        "output_path": str(config.heat_risk_dataset_parquet),
        "columns": list(dataset.columns),
    }


def train_model(config: ProjectConfig) -> dict[str, object]:
    return train_heat_risk_model(config)


def validate_model(config: ProjectConfig) -> dict[str, object]:
    return validate_heat_risk_model(config)
