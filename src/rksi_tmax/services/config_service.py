from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rksi_tmax.config import ProjectConfig, load_config
from rksi_tmax.metar_import import ASOS_COLUMNS


@dataclass(frozen=True)
class ConfigOption:
    label: str
    path: Path
    station: str


@dataclass(frozen=True)
class LocationConfigDraft:
    station: str
    timezone: str
    cutoff_local: str
    complete_day_min_local: str
    input_csv: str
    input_db: str
    raw_csv_files: tuple[str, ...]
    heat_risk_cutoffs: tuple[str, ...]
    heat_risk_thresholds_c: tuple[float, ...]
    openmeteo_history_csv: str | None = None
    openmeteo_live_csv_pattern: str | None = None


def discover_config_options(config_dir: str | Path = "configs") -> list[ConfigOption]:
    directory = Path(config_dir)
    options: list[ConfigOption] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            config = load_config(path)
        except Exception:
            continue
        label = f"{config.station} ({path.as_posix()})"
        options.append(ConfigOption(label=label, path=path, station=config.station))
    return options


def load_selected_config(path: str | Path) -> ProjectConfig:
    return load_config(path)


def create_location_config(
    draft: LocationConfigDraft,
    config_dir: str | Path = "configs",
    create_input_csv: bool = True,
) -> dict[str, object]:
    station = draft.station.strip().upper()
    if not station or not station.isalnum():
        raise ValueError("Station must be alphanumeric, for example RKSI.")
    config_path = Path(config_dir) / f"{station.lower()}.yaml"
    if config_path.exists():
        raise FileExistsError(f"Config already exists: {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _location_config_payload(station, draft)
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    config = load_config(config_path)

    csv_created = False
    if create_input_csv:
        csv_path = Path(config.input_csv)
        if not csv_path.exists():
            _write_empty_asos_csv(csv_path)
            csv_created = True

    return {
        "created": True,
        "station": config.station,
        "config_path": str(config_path),
        "input_csv": str(config.input_csv),
        "csv_created": csv_created,
    }


def delete_location_config(path: str | Path, expected_station: str) -> dict[str, object]:
    config_path = Path(path)
    config = load_config(config_path)
    station = config.station.upper()
    if station != expected_station.strip().upper():
        raise ValueError(f"Confirmation does not match station {station}.")
    if config_path.parent != Path("configs"):
        raise ValueError(f"Refusing to delete config outside configs/: {config_path}")
    config_path.unlink()
    return {
        "deleted": True,
        "station": station,
        "config_path": str(config_path),
    }


def default_location_draft(station: str) -> LocationConfigDraft:
    normalized = station.strip().upper() or "NEW"
    return LocationConfigDraft(
        station=normalized,
        timezone="Asia/Seoul",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv=f"{normalized}.csv",
        input_db="artifacts/observations.duckdb",
        raw_csv_files=(f"{normalized}.csv",),
        heat_risk_cutoffs=("09:00", "10:00", "11:00", "12:00", "13:00"),
        heat_risk_thresholds_c=(27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0),
        openmeteo_history_csv=None,
        openmeteo_live_csv_pattern=None,
    )


def summarize_config(config: ProjectConfig, config_path: str | Path) -> dict[str, object]:
    return {
        "config_path": str(config_path),
        "station": config.station,
        "timezone": config.timezone,
        "default_cutoff_local": config.cutoff_local,
        "input_csv": str(config.input_csv),
        "input_db": str(config.input_db),
        "prefer_duckdb": config.prefer_duckdb,
        "raw_csv_files": [str(path) for path in config.raw_csv_files],
        "model_path": str(config.heat_risk_model_path),
        "metrics_path": str(config.heat_risk_metrics_path),
        "openmeteo_history_csv": (
            str(config.openmeteo_history_csv) if config.openmeteo_history_csv else None
        ),
        "openmeteo_live_csv_pattern": config.openmeteo_live_csv_pattern,
    }


def _location_config_payload(station: str, draft: LocationConfigDraft) -> dict[str, object]:
    lower = station.lower()
    payload: dict[str, object] = {
        "station": station,
        "timezone": draft.timezone,
        "cutoff_local": draft.cutoff_local,
        "complete_day_min_local": draft.complete_day_min_local,
        "target": "tmax",
        "input_csv": draft.input_csv,
        "input_db": draft.input_db,
        "prefer_duckdb": True,
        "raw_csv_files": list(draft.raw_csv_files),
        "heat_risk_cutoffs": list(draft.heat_risk_cutoffs),
        "heat_risk_thresholds_c": list(draft.heat_risk_thresholds_c),
        "heat_risk_dataset_parquet": f"artifacts/{lower}_heat_risk_dataset.parquet",
        "heat_risk_model_path": f"artifacts/{lower}_heat_risk_model.joblib",
        "heat_risk_metrics_path": f"artifacts/{lower}_heat_risk_metrics.json",
        "test_fraction": 0.2,
        "random_state": 42,
        "feature_missing_threshold": 0.85,
    }
    if draft.openmeteo_history_csv:
        payload["openmeteo_history_csv"] = draft.openmeteo_history_csv
    if draft.openmeteo_live_csv_pattern:
        payload["openmeteo_live_csv_pattern"] = draft.openmeteo_live_csv_pattern
    return payload


def _write_empty_asos_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    path.write_text(",".join(ASOS_COLUMNS) + "\n", encoding="utf-8")
