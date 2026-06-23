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
    openmeteo_history_json: str | None = None
    openmeteo_live_json_pattern: str | None = None
    openmeteo_latitude: float | None = None
    openmeteo_longitude: float | None = None
    openmeteo_timezone: str = "GMT"
    openmeteo_training_start_date: str | None = "2023-01-01"
    openmeteo_training_end_date: str | None = None


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


def update_location_openmeteo_config(
    path: str | Path,
    latitude: float,
    longitude: float,
    timezone: str = "GMT",
    history_json: str | None = None,
    live_json_pattern: str | None = None,
    training_start_date: str | None = "2023-01-01",
    training_end_date: str | None = None,
) -> dict[str, object]:
    config_path = Path(path)
    if config_path.parent != Path("configs"):
        raise ValueError(f"Refusing to update config outside configs/: {config_path}")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("Open-Meteo latitude must be between -90 and 90.")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("Open-Meteo longitude must be between -180 and 180.")

    raw = _read_raw_config(config_path)
    station = str(raw.get("station") or config_path.stem).strip().upper()
    defaults = _default_openmeteo_paths(station)
    raw["openmeteo_history_json"] = history_json or defaults["history_json"]
    raw["openmeteo_live_json_pattern"] = live_json_pattern or defaults["live_json_pattern"]
    raw["openmeteo_latitude"] = float(latitude)
    raw["openmeteo_longitude"] = float(longitude)
    raw["openmeteo_timezone"] = timezone.strip() or "GMT"
    _set_optional(raw, "openmeteo_training_start_date", training_start_date)
    _set_optional(raw, "openmeteo_training_end_date", training_end_date)

    _write_raw_config(config_path, raw)
    config = load_config(config_path)
    return {
        "updated": True,
        "station": config.station,
        "config_path": str(config_path),
        "openmeteo_history_json": str(config.openmeteo_history_json),
        "openmeteo_live_json_pattern": config.openmeteo_live_json_pattern,
        "openmeteo_latitude": config.openmeteo_latitude,
        "openmeteo_longitude": config.openmeteo_longitude,
        "openmeteo_timezone": config.openmeteo_timezone,
        "openmeteo_training_start_date": config.openmeteo_training_start_date,
        "openmeteo_training_end_date": config.openmeteo_training_end_date,
    }


def default_location_draft(station: str) -> LocationConfigDraft:
    normalized = station.strip().upper() or "NEW"
    openmeteo_paths = _default_openmeteo_paths(normalized)
    return LocationConfigDraft(
        station=normalized,
        timezone="Asia/Seoul",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv=f"data/{normalized.lower()}/{normalized}.csv",
        input_db="artifacts/shared/observations.duckdb",
        raw_csv_files=(f"data/{normalized.lower()}/{normalized}.csv",),
        heat_risk_cutoffs=("09:00", "10:00", "11:00", "12:00", "13:00"),
        heat_risk_thresholds_c=(27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0),
        openmeteo_history_csv=None,
        openmeteo_live_csv_pattern=None,
        openmeteo_history_json=openmeteo_paths["history_json"],
        openmeteo_live_json_pattern=openmeteo_paths["live_json_pattern"],
        openmeteo_latitude=None,
        openmeteo_longitude=None,
        openmeteo_timezone="GMT",
        openmeteo_training_start_date="2023-01-01",
        openmeteo_training_end_date=None,
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
        "openmeteo_history_json": (
            str(config.openmeteo_history_json) if config.openmeteo_history_json else None
        ),
        "openmeteo_live_json_pattern": config.openmeteo_live_json_pattern,
        "openmeteo_latitude": config.openmeteo_latitude,
        "openmeteo_longitude": config.openmeteo_longitude,
        "openmeteo_timezone": config.openmeteo_timezone,
        "openmeteo_training_start_date": config.openmeteo_training_start_date,
        "openmeteo_training_end_date": config.openmeteo_training_end_date,
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
        "heat_risk_dataset_parquet": f"artifacts/{lower}/{lower}_heat_risk_dataset.parquet",
        "heat_risk_model_path": f"artifacts/{lower}/{lower}_heat_risk_model.joblib",
        "heat_risk_metrics_path": f"artifacts/{lower}/{lower}_heat_risk_metrics.json",
        "test_fraction": 0.2,
        "random_state": 42,
        "feature_missing_threshold": 0.85,
    }
    if draft.openmeteo_history_csv:
        payload["openmeteo_history_csv"] = draft.openmeteo_history_csv
    if draft.openmeteo_live_csv_pattern:
        payload["openmeteo_live_csv_pattern"] = draft.openmeteo_live_csv_pattern
    if draft.openmeteo_history_json:
        payload["openmeteo_history_json"] = draft.openmeteo_history_json
    if draft.openmeteo_live_json_pattern:
        payload["openmeteo_live_json_pattern"] = draft.openmeteo_live_json_pattern
    if draft.openmeteo_latitude is not None:
        payload["openmeteo_latitude"] = draft.openmeteo_latitude
    if draft.openmeteo_longitude is not None:
        payload["openmeteo_longitude"] = draft.openmeteo_longitude
    if draft.openmeteo_timezone:
        payload["openmeteo_timezone"] = draft.openmeteo_timezone
    if draft.openmeteo_training_start_date:
        payload["openmeteo_training_start_date"] = draft.openmeteo_training_start_date
    if draft.openmeteo_training_end_date:
        payload["openmeteo_training_end_date"] = draft.openmeteo_training_end_date
    return payload


def _read_raw_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return raw


def _write_raw_config(path: Path, raw: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _default_openmeteo_paths(station: str) -> dict[str, str]:
    station_lower = station.strip().lower() or "station"
    return {
        "history_json": f"data/{station_lower}/openmeteo-{station_lower}-history.json",
        "live_json_pattern": f"data/{station_lower}/openmeteo-{station_lower}-{{date}}.json",
    }


def _set_optional(raw: dict[str, object], key: str, value: str | None) -> None:
    normalized = value.strip() if value else ""
    if normalized:
        raw[key] = normalized
    else:
        raw.pop(key, None)


def _write_empty_asos_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    path.write_text(",".join(ASOS_COLUMNS) + "\n", encoding="utf-8")
