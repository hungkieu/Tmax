from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rksi_tmax.config import load_config
from rksi_tmax.heat_risk import (
    build_heat_risk_dataset,
    format_heat_risk_explanation,
    plot_prediction_curve,
    predict_heat_risk,
    train_heat_risk_model,
    validate_heat_risk_model,
)
from rksi_tmax.metar_import import import_metar_file
from rksi_tmax.metar_import import fetch_metar_text
from rksi_tmax.storage import sync_duckdb_from_csv
from rksi_tmax.storage import read_station_observations_from_duckdb


SHORTCUT_STATIONS = {
    "rksi": "configs/default.yaml",
    "rkpk": "configs/rkpk.yaml",
    "rjtt": "configs/rjtt.yaml",
    "wsss": "configs/wsss.yaml",
}


def import_metar_main() -> None:
    parser = argparse.ArgumentParser(description="Append METAR text observations to ASOS CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metar-file", default="metar.txt")
    parser.add_argument("--csv", help="Target ASOS CSV. Defaults to config input_csv.")
    parser.add_argument(
        "--reference-date",
        help="UTC date YYYY-MM-DD used to infer month/year from METAR DDHHMMZ.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = import_metar_file(
        metar_path=args.metar_file,
        csv_path=args.csv or config.input_csv,
        reference_date=args.reference_date,
        db_path=config.input_db if config.prefer_duckdb else None,
    )
    print(json.dumps(result, indent=2))


def fetch_metar_main() -> None:
    parser = argparse.ArgumentParser(description="Fetch recent METAR text from Aviation Weather.")
    parser.add_argument(
        "--stations",
        default="RKSI,RKPK,RJTT,WSSS",
        help="Comma-separated station list, for example RKSI,RKPK,RJTT,WSSS.",
    )
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--output", default="metar.txt")
    args = parser.parse_args()

    stations = [station.strip() for station in args.stations.split(",") if station.strip()]
    result = fetch_metar_text(stations=stations, hours=args.hours, output_path=args.output)
    print(json.dumps(result, indent=2))


def sync_duckdb_main() -> None:
    parser = argparse.ArgumentParser(description="Sync one or more ASOS CSV files into DuckDB.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_files",
        help="CSV file to load. Repeat for multiple files. Defaults to raw_csv_files in config.",
    )
    parser.add_argument("--db", help="DuckDB path. Defaults to config input_db.")
    args = parser.parse_args()

    config = load_config(args.config)
    csv_files = args.csv_files or config.raw_csv_files
    result = sync_duckdb_from_csv(csv_files, args.db or config.input_db)
    print(json.dumps(result, indent=2))


def build_heat_risk_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-cutoff Tmax remaining heat table.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-csv")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = build_heat_risk_dataset(config, input_csv=args.input_csv, output_parquet=args.output)
    print(f"Wrote {len(dataset)} heat risk rows to {args.output or config.heat_risk_dataset_parquet}")


def train_heat_risk_main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-cutoff Tmax remaining heat model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train_heat_risk_model(config)
    print(json.dumps(metrics, indent=2))


def validate_heat_risk_main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-cutoff Tmax remaining heat model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    report = validate_heat_risk_model(config)
    print(json.dumps(report["summary"], indent=2))


def predict_heat_risk_main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Predict remaining Tmax heat from an arbitrary cutoff.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD format.")
    parser.add_argument("--cutoff-local", required=True, help="Local cutoff HH:MM, for example 10:00.")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--bet-temp-c",
        type=float,
        help="Estimate win probability for betting this Celsius value is not today's final Tmax.",
    )
    parser.add_argument(
        "--prediction-method",
        choices=["auto", "direct", "two_stage", "m1", "m3", "openmeteo"],
        default="auto",
        help="Override forecast method. M3/Open-Meteo only works for supported model bundles.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        help="Write a PNG temperature-curve chart. Omit the value to use an automatic artifacts path.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print a plain-language Vietnamese explanation after the JSON output.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = predict_heat_risk(
        config,
        args.date,
        args.cutoff_local,
        dataset_path=args.dataset,
        bet_temp_c=args.bet_temp_c,
        prediction_method_override=args.prediction_method,
    )
    if args.plot is not None:
        plot_path = args.plot or _default_prediction_plot_path(config.station, args.date, args.cutoff_local)
        result["plot_path"] = str(plot_prediction_curve(config, result, plot_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.explain:
        print(format_heat_risk_explanation(result))


def rksi_main() -> None:
    station_shortcut_main("configs/default.yaml")


def rkpk_main() -> None:
    station_shortcut_main("configs/rkpk.yaml")


def rjtt_main() -> None:
    station_shortcut_main("configs/rjtt.yaml")


def wsss_main() -> None:
    station_shortcut_main("configs/wsss.yaml")


def station_shortcut_main(config_path: str) -> None:
    _configure_stdout()
    config = load_config(config_path)
    ingest_config = load_config("configs/default.yaml")
    parser = argparse.ArgumentParser(
        description=(
            f"Fetch/import METAR and predict heat risk for {config.station}. "
            "Defaults are intended for one-command operational use."
        )
    )
    parser.add_argument("--date", help="Local forecast date YYYY-MM-DD. Defaults to today.")
    parser.add_argument(
        "--cutoff-local",
        help="Local cutoff HH:MM. Defaults to current local hour rounded down.",
    )
    parser.add_argument("--hours", type=int, default=4, help="METAR fetch lookback hours.")
    parser.add_argument("--metar-file", default="metar.txt")
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-duckdb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bet-temp-c",
        type=float,
        help="Estimate win probability for betting this Celsius value is not today's final Tmax.",
    )
    parser.add_argument(
        "--prediction-method",
        choices=["auto", "direct", "two_stage", "m1", "m3", "openmeteo"],
        default="auto",
        help="Override forecast method. M3/Open-Meteo only works for supported model bundles.",
    )
    args = parser.parse_args()

    now_local = datetime.now(ZoneInfo(config.timezone))
    local_date = args.date or now_local.date().isoformat()
    cutoff_local = args.cutoff_local or f"{now_local.hour:02d}:00"
    metar_file = Path(args.metar_file)

    if args.fetch:
        fetch_result = fetch_metar_text(
            stations=[station.upper() for station in SHORTCUT_STATIONS],
            hours=args.hours,
            output_path=metar_file,
        )
        print(json.dumps({"fetch": fetch_result}, indent=2, ensure_ascii=False))

    db_was_missing = ingest_config.prefer_duckdb and not ingest_config.input_db.exists()
    if metar_file.exists():
        import_result = import_metar_file(
            metar_path=metar_file,
            csv_path=ingest_config.input_csv,
            reference_date=local_date,
            db_path=ingest_config.input_db if ingest_config.prefer_duckdb else None,
        )
        print(json.dumps({"import": import_result}, indent=2, ensure_ascii=False))

    if args.sync_duckdb and (
        db_was_missing or _duckdb_station_history_is_short(config_path, minimum_rows=100)
    ):
        sync_result = sync_duckdb_from_csv(ingest_config.raw_csv_files, ingest_config.input_db)
        print(json.dumps({"sync_duckdb": sync_result}, indent=2, ensure_ascii=False))

    result = predict_heat_risk(
        config,
        local_date,
        cutoff_local,
        dataset_path=None,
        bet_temp_c=args.bet_temp_c,
        prediction_method_override=args.prediction_method,
    )
    if args.plot:
        plot_path = _default_prediction_plot_path(config.station, local_date, cutoff_local)
        result["plot_path"] = str(plot_prediction_curve(config, result, plot_path))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.explain:
        print(format_heat_risk_explanation(result))


def _default_prediction_plot_path(station: str, local_date: str, cutoff_local: str) -> str:
    safe_cutoff = cutoff_local.replace(":", "")
    return f"artifacts/{station.lower()}_{local_date}_{safe_cutoff}_temperature_curve.png"


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _duckdb_station_history_is_short(config_path: str, minimum_rows: int) -> bool:
    config = load_config(config_path)
    if not config.prefer_duckdb or not config.input_db.exists():
        return False
    try:
        observations = read_station_observations_from_duckdb(
            config.input_db,
            config.station,
            ["station", "valid"],
        )
    except Exception:
        return True
    return len(observations) < minimum_rows
