from __future__ import annotations

import argparse
import json

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
    parser = argparse.ArgumentParser(description="Predict remaining Tmax heat from an arbitrary cutoff.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD format.")
    parser.add_argument("--cutoff-local", required=True, help="Local cutoff HH:MM, for example 10:00.")
    parser.add_argument("--dataset")
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
    )
    if args.plot is not None:
        plot_path = args.plot or _default_prediction_plot_path(config.station, args.date, args.cutoff_local)
        result["plot_path"] = str(plot_prediction_curve(config, result, plot_path))
    print(json.dumps(result, indent=2))
    if args.explain:
        print(format_heat_risk_explanation(result))


def _default_prediction_plot_path(station: str, local_date: str, cutoff_local: str) -> str:
    safe_cutoff = cutoff_local.replace(":", "")
    return f"artifacts/{station.lower()}_{local_date}_{safe_cutoff}_temperature_curve.png"
