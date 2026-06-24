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
from rksi_tmax.next_metar_temp import (
    HORIZONS,
    build_next_metar_dataset,
    korea_configs,
    predict_next_metars,
    train_next_metar_model,
    verify_next_metar_from_history,
    verify_next_metar_predictions,
)
from rksi_tmax.services.metar_service import import_many_station_metars
from rksi_tmax.services.training_service import (
    prepare_openmeteo_daily_data,
    prepare_openmeteo_training_data,
)
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
    parser.add_argument("--metar-file", default="data/shared/metar.txt")
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
    parser.add_argument("--output", default="data/shared/metar.txt")
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


def fetch_openmeteo_main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo API data into location cache files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["training", "daily"],
        default="training",
        help=(
            "training fetches historical forecast range; daily fetches one forecast date."
        ),
    )
    parser.add_argument("--date", help="Local date YYYY-MM-DD for --mode daily.")
    parser.add_argument("--force", action="store_true", help="Refresh cache even if a file already exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "daily":
        if not args.date:
            parser.error("--date is required for daily mode")
        result = prepare_openmeteo_daily_data(config, args.date, force=args.force)
    else:
        result = prepare_openmeteo_training_data(config, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


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
        choices=["auto", "direct", "two_stage", "m1", "m3", "openmeteo", "m4"],
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


def build_next_metar_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Build Korea next-METAR integer temperature dataset.")
    parser.add_argument(
        "--station",
        default="ALL",
        choices=["ALL", "RKSI", "RKPK"],
        help="Station to build, or ALL for the combined Korea model.",
    )
    parser.add_argument("--output", help="Output parquet path.")
    args = parser.parse_args()

    dataset = build_next_metar_dataset(
        korea_configs(args.station),
        output_path=args.output or "artifacts/next_metar_temp/next_metar_temp_dataset.parquet",
    )
    print(
        json.dumps(
            {
                "rows": len(dataset),
                "stations": sorted(dataset["station"].unique().tolist()),
                "output_path": args.output
                or "artifacts/next_metar_temp/next_metar_temp_dataset.parquet",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def train_next_metar_temp_main() -> None:
    parser = argparse.ArgumentParser(description="Train Korea next-METAR integer temperature model.")
    parser.add_argument(
        "--dataset",
        default="artifacts/next_metar_temp/next_metar_temp_dataset.parquet",
    )
    parser.add_argument(
        "--model",
        default="artifacts/next_metar_temp/next_metar_temp_model.joblib",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/next_metar_temp/next_metar_temp_metrics.json",
    )
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metrics = train_next_metar_model(
        args.dataset,
        args.model,
        args.metrics,
        promote=args.promote,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def predict_next_metar_temp_main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Predict the next 1-4 regular METAR integer temperatures."
    )
    parser.add_argument("--station", required=True, choices=["RKSI", "RKPK"])
    parser.add_argument(
        "--observed-at",
        help="Observation timestamp, ISO-8601 with timezone. Read from MongoDB when omitted.",
    )
    parser.add_argument(
        "--temp-c",
        type=float,
        help="Live observed temperature in Celsius. Read from MongoDB when omitted.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(HORIZONS),
        help="Which METARs ahead to predict (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--tmax-signal-c",
        type=float,
        help="Optional current-day Tmax forecast/signal. If omitted, the model imputes it.",
    )
    parser.add_argument(
        "--model",
        default="artifacts/next_metar_temp/next_metar_temp_model.joblib",
    )
    parser.add_argument(
        "--log",
        default="artifacts/next_metar_temp/next_metar_temp_predictions.jsonl",
    )
    args = parser.parse_args()

    observed_at = args.observed_at
    temp_c = args.temp_c
    if observed_at is None or temp_c is None:
        from rksi_tmax.mongo_source import get_current_temperature

        current = get_current_temperature(args.station)
        if current is None:
            raise SystemExit(
                f"No live temperature found in MongoDB for {args.station}. "
                "Provide --temp-c and --observed-at, or check the database."
            )
        if temp_c is None:
            temp_c = current.temp_c
        if observed_at is None:
            observed_at = current.observed_at_local_iso

    result = predict_next_metars(
        args.station,
        observed_at,
        temp_c,
        horizons=args.horizons,
        tmax_signal_c=args.tmax_signal_c,
        model_path=args.model,
        log_path=args.log,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def verify_next_metar_temp_main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Verify pending next-METAR predictions.")
    parser.add_argument("--station", default="ALL", choices=["ALL", "RKSI", "RKPK"])
    parser.add_argument(
        "--log",
        default="artifacts/next_metar_temp/next_metar_temp_predictions.jsonl",
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--from-db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify against observed temperatures read directly from MongoDB.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="How many hours of MongoDB history to load for --from-db.",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=int,
        default=300,
        help="Max gap when matching a METAR slot to the nearest MongoDB observation.",
    )
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--metar-file", default="data/shared/metar.txt")
    parser.add_argument("--reference-date", help="UTC reference date YYYY-MM-DD for METAR import.")
    args = parser.parse_args()

    configs = korea_configs(args.station)
    if args.from_db:
        from datetime import timedelta, timezone

        from rksi_tmax.mongo_source import get_temperature_history

        since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
        history_by_station = {
            config.station.upper(): get_temperature_history(config.station, since=since)
            for config in configs
        }
        result = verify_next_metar_from_history(
            history_by_station,
            log_path=args.log,
            window=args.window,
            tolerance_seconds=args.tolerance_seconds,
        )
    else:
        if args.fetch:
            fetch_result = fetch_metar_text(
                [config.station for config in configs],
                hours=args.hours,
                output_path=args.metar_file,
            )
            print(json.dumps({"fetch": fetch_result}, indent=2, ensure_ascii=False))
        if Path(args.metar_file).exists():
            import_result = import_many_station_metars(
                configs,
                args.metar_file,
                args.reference_date or datetime.now().date().isoformat(),
            )
            print(json.dumps({"import": import_result}, indent=2, ensure_ascii=False))
        result = verify_next_metar_predictions(configs, log_path=args.log, window=args.window)
    print(json.dumps(result, indent=2, ensure_ascii=False))


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
    ingest_configs = [load_config(path) for path in SHORTCUT_STATIONS.values()]
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
    parser.add_argument("--metar-file", default="data/shared/metar.txt")
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
        choices=["auto", "direct", "two_stage", "m1", "m3", "openmeteo", "m4"],
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
        import_result = import_many_station_metars(ingest_configs, metar_file, local_date)
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
    lower = station.lower()
    return f"artifacts/{lower}/{lower}_{local_date}_{safe_cutoff}_temperature_curve.png"


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
