from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rksi_tmax.config import ProjectConfig
from rksi_tmax.next_metar_temp import (
    build_next_metar_temp_dataset,
    predict_next_metar_temp,
    train_next_metar_temp_model,
    validate_next_metar_temp_model,
)
from rksi_tmax.services import metar_service, training_service


def artifact_status(config: ProjectConfig) -> dict[str, dict[str, object]]:
    artifacts = {
        "dataset": config.next_metar_temp_dataset_parquet,
        "model": config.next_metar_temp_model_path,
        "metrics": config.next_metar_temp_metrics_path,
    }
    return {
        name: {
            "path": str(path),
            "exists": Path(path).exists(),
            "size_bytes": Path(path).stat().st_size if Path(path).exists() else None,
        }
        for name, path in artifacts.items()
    }


def read_metrics(config: ProjectConfig) -> dict[str, object] | None:
    path = Path(config.next_metar_temp_metrics_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(config: ProjectConfig) -> dict[str, object]:
    dataset = build_next_metar_temp_dataset(config)
    return {
        "rows": int(len(dataset)),
        "columns": len(dataset.columns),
        "output_path": str(config.next_metar_temp_dataset_parquet),
    }


def train_model(config: ProjectConfig) -> dict[str, object]:
    return train_next_metar_temp_model(config)


def validate_model(config: ProjectConfig) -> dict[str, object]:
    return validate_next_metar_temp_model(config)


def run_live_nowcast(
    config: ProjectConfig,
    *,
    update_metar: bool,
    update_openmeteo: bool,
    metar_hours: int,
    metar_file: str,
    as_of_local: str | None = None,
) -> dict[str, object]:
    steps: dict[str, object] = {}
    warnings: list[str] = []

    local_date = _today_local(config)
    if update_metar:
        fetch_result = metar_service.fetch_metar_for_stations(
            [config.station],
            hours=metar_hours,
            output_path=metar_file,
        )
        steps["fetch_metar"] = fetch_result
        import_result = metar_service.import_station_metar(
            config,
            metar_file,
            reference_date=local_date,
        )
        steps["import_metar"] = import_result

    if update_openmeteo:
        try:
            steps["openmeteo"] = training_service.prepare_openmeteo_daily_data(
                config,
                local_date,
                force=False,
            )
        except Exception as exc:
            warnings.append(f"Open-Meteo update failed; prediction used existing cache if available: {exc}")

    prediction = predict_next_metar_temp(
        config,
        as_of_local=as_of_local or None,
        fetch_openmeteo=False,
    )
    return {
        "prediction": prediction,
        "steps": steps,
        "warnings": warnings,
    }


def _today_local(config: ProjectConfig) -> str:
    return datetime.now(ZoneInfo(config.timezone)).date().isoformat()
