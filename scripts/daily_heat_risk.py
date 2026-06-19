from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rksi_tmax.config import ProjectConfig, _hhmm_to_minutes, _minutes_to_hhmm, load_config
from rksi_tmax.features import load_observations
from rksi_tmax.heat_risk import (
    build_heat_risk_dataset,
    format_heat_risk_explanation,
    predict_heat_risk,
    train_heat_risk_model,
)
from rksi_tmax.metar_import import fetch_metar_text, import_metar_file
from rksi_tmax.storage import sync_duckdb_from_csv


DEFAULT_CONFIGS = (
    "configs/default.yaml",
    "configs/rkpk.yaml",
    "configs/rjtt.yaml",
    "configs/wsss.yaml",
)


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Fetch/import METAR data, verify station freshness, and run heat-risk predictions."
    )
    parser.add_argument("--date", help="Local forecast date YYYY-MM-DD. Defaults to today in the first config timezone.")
    parser.add_argument("--cutoff-local", default="08:00", help="Local cutoff HH:MM.")
    parser.add_argument("--config", action="append", dest="configs", help="Config path. Repeat for multiple stations.")
    parser.add_argument("--metar-file", default="metar.txt")
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--max-lag-minutes", type=int, default=60)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="Force rebuild and train for all configs before prediction.")
    args = parser.parse_args()

    configs = [load_config(path) for path in (args.configs or DEFAULT_CONFIGS)]
    local_date = args.date or datetime.now(ZoneInfo(configs[0].timezone)).date().isoformat()
    cutoff_local = _minutes_to_hhmm(_hhmm_to_minutes(args.cutoff_local))
    metar_path = Path(args.metar_file)
    db_was_missing = any(config.prefer_duckdb and not Path(config.input_db).exists() for config in configs)

    if args.fetch:
        stations = sorted({config.station for config in configs})
        fetch_result = fetch_metar_text(stations=stations, hours=args.hours, output_path=metar_path)
        print(json.dumps({"fetch": fetch_result}, indent=2))

    _sync_if_history_is_short(configs)
    if metar_path.exists():
        import_result = import_metar_file(
            metar_path=metar_path,
            csv_path=configs[0].input_csv,
            reference_date=local_date,
            db_path=configs[0].input_db if configs[0].prefer_duckdb else None,
        )
        print(json.dumps({"import": import_result}, indent=2))

    if db_was_missing:
        _sync_duckdb(configs[0])
    _sync_if_history_is_short(configs)
    after = _check_all(configs, local_date, cutoff_local, args.max_lag_minutes)
    print(json.dumps({"data_check": after}, indent=2))

    for config in configs:
        _ensure_model(config, force=args.retrain)

    predictions = []
    for config in configs:
        if not next(row for row in after if row["station"] == config.station)["sufficient"]:
            predictions.append(
                {
                    "station": config.station,
                    "status": "skipped",
                    "reason": f"insufficient observations by {local_date} {cutoff_local}",
                }
            )
            continue

        result = predict_heat_risk(config, local_date, cutoff_local, dataset_path=None)
        if args.plot:
            from rksi_tmax.heat_risk import plot_prediction_curve

            safe_cutoff = cutoff_local.replace(":", "")
            plot_path = f"artifacts/{config.station.lower()}_{local_date}_{safe_cutoff}_temperature_curve.png"
            result["plot_path"] = str(plot_prediction_curve(config, result, plot_path))
        predictions.append(result)
        if args.explain:
            print(format_heat_risk_explanation(result))

    print(json.dumps({"predictions": predictions}, indent=2))


def _check_all(
    configs: list[ProjectConfig],
    local_date: str,
    cutoff_local: str,
    max_lag_minutes: int,
) -> list[dict]:
    return [
        _station_data_status(config, local_date, cutoff_local, max_lag_minutes)
        for config in configs
    ]


def _station_data_status(
    config: ProjectConfig,
    local_date: str,
    cutoff_local: str,
    max_lag_minutes: int,
) -> dict:
    cutoff_minutes = _hhmm_to_minutes(cutoff_local)
    observations = load_observations(config.input_csv, config)
    if observations.empty:
        return _status(config.station, local_date, cutoff_local, 0, 0, 0, None, False)

    local_times = pd.to_datetime(observations["valid_local"])
    minutes = local_times.dt.hour * 60 + local_times.dt.minute
    date_mask = local_times.dt.strftime("%Y-%m-%d") == local_date
    today_count = int(date_mask.sum())
    to_cutoff = observations[date_mask & (minutes <= cutoff_minutes)].copy()
    if to_cutoff.empty:
        return _status(
            config.station,
            local_date,
            cutoff_local,
            int(len(observations)),
            today_count,
            0,
            None,
            False,
        )

    to_cutoff_times = pd.to_datetime(to_cutoff["valid_local"])
    last_minutes = int((to_cutoff_times.dt.hour * 60 + to_cutoff_times.dt.minute).max())
    lag = cutoff_minutes - last_minutes
    return _status(
        config.station,
        local_date,
        cutoff_local,
        int(len(observations)),
        today_count,
        int(len(to_cutoff)),
        _minutes_to_hhmm(last_minutes),
        0 <= lag <= max_lag_minutes,
        lag,
    )


def _status(
    station: str,
    local_date: str,
    cutoff_local: str,
    total_rows: int,
    rows_today: int,
    rows_to_cutoff: int,
    last_observation_local: str | None,
    sufficient: bool,
    lag_minutes: int | None = None,
) -> dict:
    return {
        "station": station,
        "local_date": local_date,
        "cutoff_local": cutoff_local,
        "total_rows": total_rows,
        "rows_today": rows_today,
        "rows_to_cutoff": rows_to_cutoff,
        "last_observation_local": last_observation_local,
        "lag_minutes": lag_minutes,
        "sufficient": sufficient,
    }


def _sync_if_history_is_short(configs: list[ProjectConfig], minimum_rows: int = 100) -> None:
    if not any(config.prefer_duckdb and Path(config.input_db).exists() for config in configs):
        return

    for config in configs:
        if not config.prefer_duckdb or not Path(config.input_db).exists():
            continue
        try:
            observations = load_observations(config.input_csv, config)
        except Exception:
            _sync_duckdb(configs[0])
            return
        if len(observations) < minimum_rows:
            _sync_duckdb(configs[0])
            return


def _sync_duckdb(config: ProjectConfig) -> None:
    result = sync_duckdb_from_csv(config.raw_csv_files, config.input_db)
    print(json.dumps({"sync_duckdb": result}, indent=2))


def _ensure_model(config: ProjectConfig, force: bool) -> None:
    dataset_missing = not Path(config.heat_risk_dataset_parquet).exists()
    model_missing = not Path(config.heat_risk_model_path).exists()
    dataset_newer = (
        Path(config.heat_risk_dataset_parquet).exists()
        and Path(config.heat_risk_model_path).exists()
        and Path(config.heat_risk_dataset_parquet).stat().st_mtime
        > Path(config.heat_risk_model_path).stat().st_mtime
    )

    if force or dataset_missing:
        dataset = build_heat_risk_dataset(config)
        print(
            json.dumps(
                {
                    "build": {
                        "station": config.station,
                        "dataset": str(config.heat_risk_dataset_parquet),
                        "rows": len(dataset),
                    }
                },
                indent=2,
            )
        )

    if force or model_missing or dataset_newer:
        metrics = train_heat_risk_model(config)
        print(
            json.dumps(
                {
                    "train": {
                        "station": config.station,
                        "model": str(config.heat_risk_model_path),
                        "selected_prediction_method": metrics.get("selected_prediction_method"),
                    }
                },
                indent=2,
            )
        )


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
