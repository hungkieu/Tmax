from __future__ import annotations

from rksi_tmax.config import ProjectConfig
from rksi_tmax.heat_risk import (
    build_heat_risk_dataset,
    train_heat_risk_model,
    validate_heat_risk_model,
)


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
