from __future__ import annotations

import argparse
import json

from rksi_tmax.config import load_config
from rksi_tmax.heat_risk import (
    build_heat_risk_dataset,
    predict_heat_risk,
    train_heat_risk_model,
    validate_heat_risk_model,
)
from rksi_tmax.metar_import import import_metar_file


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
    )
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
    args = parser.parse_args()

    config = load_config(args.config)
    result = predict_heat_risk(
        config,
        args.date,
        args.cutoff_local,
        dataset_path=args.dataset,
    )
    print(json.dumps(result, indent=2))
