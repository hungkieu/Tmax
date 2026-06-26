from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    station: str = "RKSI"
    timezone: str = "Asia/Seoul"
    cutoff_local: str = "09:00"
    complete_day_min_local: str = "23:00"
    target: str = "tmax"
    input_csv: Path = Path("data/rksi/asos.csv")
    input_db: Path = Path("artifacts/shared/observations.duckdb")
    prefer_duckdb: bool = True
    raw_csv_files: tuple[Path, ...] = (Path("data/rksi/asos.csv"),)
    openmeteo_history_csv: Path | None = None
    openmeteo_live_csv_pattern: str | None = None
    openmeteo_history_json: Path | None = None
    openmeteo_live_json_pattern: str | None = None
    openmeteo_latitude: float | None = None
    openmeteo_longitude: float | None = None
    openmeteo_timezone: str = "GMT"
    openmeteo_training_start_date: str | None = None
    openmeteo_training_end_date: str | None = None
    heat_risk_cutoffs: tuple[str, ...] = (
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
    )
    heat_risk_thresholds_c: tuple[float, ...] = (28.0, 29.0, 30.0, 31.0)
    heat_risk_dataset_parquet: Path = Path("artifacts/rksi/heat_risk_dataset.parquet")
    heat_risk_model_path: Path = Path("artifacts/rksi/heat_risk_model.joblib")
    heat_risk_metrics_path: Path = Path("artifacts/rksi/heat_risk_metrics.json")
    next_metar_temp_dataset_parquet: Path = Path(
        "artifacts/next_metar_temp/next_metar_temp_dataset.parquet"
    )
    next_metar_temp_model_path: Path = Path(
        "artifacts/next_metar_temp/next_metar_temp_model.joblib"
    )
    next_metar_temp_metrics_path: Path = Path(
        "artifacts/next_metar_temp/next_metar_temp_metrics.json"
    )
    test_fraction: float = 0.2
    random_state: int = 42
    feature_missing_threshold: float = 0.85

    @property
    def cutoff_minutes(self) -> int:
        return _hhmm_to_minutes(self.cutoff_local)

    @property
    def complete_day_min_minutes(self) -> int:
        return _hhmm_to_minutes(self.complete_day_min_local)


def _hhmm_to_minutes(value: str) -> int:
    hour_text, minute_text = value.split(":", maxsplit=1)
    return int(hour_text) * 60 + int(minute_text)


def _minutes_to_hhmm(minutes: int) -> str:
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _normalize_hhmm(value: str) -> str:
    return _minutes_to_hhmm(_hhmm_to_minutes(value))


def load_config(path: str | Path = "configs/default.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    path_fields = {
        "input_csv",
        "input_db",
        "heat_risk_dataset_parquet",
        "heat_risk_model_path",
        "heat_risk_metrics_path",
        "next_metar_temp_dataset_parquet",
        "next_metar_temp_model_path",
        "next_metar_temp_metrics_path",
        "openmeteo_history_csv",
        "openmeteo_history_json",
    }
    values = {
        key: Path(value) if key in path_fields and value is not None else value
        for key, value in raw.items()
    }
    if "raw_csv_files" in values:
        values["raw_csv_files"] = tuple(Path(value) for value in values["raw_csv_files"])
    if "cutoff_local" in values:
        values["cutoff_local"] = _normalize_hhmm(values["cutoff_local"])
    if "complete_day_min_local" in values:
        values["complete_day_min_local"] = _normalize_hhmm(values["complete_day_min_local"])
    if "heat_risk_cutoffs" in values:
        values["heat_risk_cutoffs"] = tuple(_normalize_hhmm(value) for value in values["heat_risk_cutoffs"])
    if "heat_risk_thresholds_c" in values:
        values["heat_risk_thresholds_c"] = tuple(float(value) for value in values["heat_risk_thresholds_c"])
    for key in ("openmeteo_latitude", "openmeteo_longitude"):
        if key in values and values[key] is not None:
            values[key] = float(values[key])
    for key in ("openmeteo_training_start_date", "openmeteo_training_end_date"):
        if key in values and values[key] is not None:
            values[key] = str(values[key])
    return ProjectConfig(**values)
