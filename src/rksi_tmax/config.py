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
    input_csv: Path = Path("asos.csv")
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
    heat_risk_dataset_parquet: Path = Path("artifacts/heat_risk_dataset.parquet")
    heat_risk_model_path: Path = Path("artifacts/heat_risk_model.joblib")
    heat_risk_metrics_path: Path = Path("artifacts/heat_risk_metrics.json")
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


def load_config(path: str | Path = "configs/default.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    path_fields = {
        "input_csv",
        "heat_risk_dataset_parquet",
        "heat_risk_model_path",
        "heat_risk_metrics_path",
    }
    values = {key: Path(value) if key in path_fields else value for key, value in raw.items()}
    if "heat_risk_cutoffs" in values:
        values["heat_risk_cutoffs"] = tuple(values["heat_risk_cutoffs"])
    if "heat_risk_thresholds_c" in values:
        values["heat_risk_thresholds_c"] = tuple(float(value) for value in values["heat_risk_thresholds_c"])
    return ProjectConfig(**values)
