from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rksi_tmax.config import ProjectConfig, load_config
from rksi_tmax.heat_risk import predict_heat_risk
from rksi_tmax.metar_import import fetch_metar_text, import_metar_file
from rksi_tmax.storage import read_station_observations_from_duckdb, sync_duckdb_from_csv


DEFAULT_CONFIG_PATHS = (
    "configs/default.yaml",
    "configs/rkpk.yaml",
    "configs/rjtt.yaml",
    "configs/wsss.yaml",
)
DEFAULT_STATIONS = ("RKSI", "RKPK", "RJTT", "WSSS")


def telegram_report_main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Build a combined Telegram heat-risk report.")
    parser.add_argument("--output", default="artifacts/telegram_report.md")
    parser.add_argument("--metar-file", default="metar.txt")
    parser.add_argument("--hours", type=int, default=4)
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-duckdb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", action="append", dest="config_paths")
    parser.add_argument("--reference-date", help="UTC date YYYY-MM-DD for METAR DDHHMMZ parsing.")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    report = build_telegram_report(
        config_paths=tuple(args.config_paths or DEFAULT_CONFIG_PATHS),
        metar_file=args.metar_file,
        output_path=args.output,
        hours=args.hours,
        fetch=args.fetch,
        sync_duckdb=args.sync_duckdb,
        now_utc=now_utc,
        reference_date=args.reference_date,
    )
    print(json.dumps(report["metadata"], indent=2, ensure_ascii=False))
    print(report["text"])


def build_telegram_report(
    config_paths: tuple[str, ...] = DEFAULT_CONFIG_PATHS,
    metar_file: str | Path = "metar.txt",
    output_path: str | Path = "artifacts/telegram_report.md",
    hours: int = 4,
    fetch: bool = True,
    sync_duckdb: bool = True,
    now_utc: datetime | None = None,
    reference_date: str | None = None,
) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    configs = [load_config(path) for path in config_paths]
    ingest_config = load_config("configs/default.yaml")
    metar_path = Path(metar_file)

    fetch_result = None
    append_result = None
    if fetch:
        fetch_output = metar_path.with_name(f"{metar_path.name}.fetch")
        fetch_result = fetch_metar_text(
            stations=list(DEFAULT_STATIONS),
            hours=hours,
            output_path=fetch_output,
        )
        append_result = _append_unique_lines(fetch_output, metar_path)
        fetch_output.unlink(missing_ok=True)

    db_was_missing = ingest_config.prefer_duckdb and not ingest_config.input_db.exists()
    import_result = None
    if metar_path.exists():
        import_result = import_metar_file(
            metar_path=metar_path,
            csv_path=ingest_config.input_csv,
            reference_date=reference_date or now_utc.date().isoformat(),
            db_path=ingest_config.input_db if ingest_config.prefer_duckdb else None,
        )

    sync_result = None
    can_sync_from_csv = all(path.exists() for path in ingest_config.raw_csv_files)
    if sync_duckdb and can_sync_from_csv and (
        db_was_missing or _any_station_history_is_short(configs, minimum_rows=100)
    ):
        sync_result = sync_duckdb_from_csv(ingest_config.raw_csv_files, ingest_config.input_db)

    entries = []
    for config in configs:
        local_now = now_utc.astimezone(ZoneInfo(config.timezone))
        local_date = local_now.date().isoformat()
        cutoff_local = f"{local_now.hour:02d}:00"
        try:
            prediction = predict_heat_risk(config, local_date, cutoff_local, dataset_path=None)
            entries.append({"status": "ok", "prediction": prediction})
        except Exception as exc:
            entries.append(
                {
                    "status": "error",
                    "station": config.station,
                    "local_date": local_date,
                    "cutoff_local": cutoff_local,
                    "error": str(exc),
                }
            )

    text = format_telegram_report(entries, generated_at_utc=now_utc)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    return {
        "text": text,
        "metadata": {
            "output": str(output),
            "generated_at_utc": now_utc.isoformat(),
            "fetch": fetch_result,
            "append_metar": append_result,
            "import": import_result,
            "sync_duckdb": sync_result,
            "stations": [
                entry.get("prediction", entry).get("station", "unknown")
                for entry in entries
            ],
            "errors": [
                entry for entry in entries if entry.get("status") == "error"
            ],
        },
    }


def format_telegram_report(entries: list[dict], generated_at_utc: datetime) -> str:
    generated = generated_at_utc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "Heat-risk report",
        f"Generated: {generated}",
        "",
    ]
    for entry in entries:
        if entry.get("status") == "error":
            lines.extend(_format_error_entry(entry))
        else:
            lines.extend(_format_prediction_entry(entry["prediction"]))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_prediction_entry(prediction: dict) -> list[str]:
    interval_low = prediction.get("prediction_interval_80_low_c")
    interval_high = prediction.get("prediction_interval_80_high_c")
    weather = prediction.get("weather_context", {})
    weather_summary = weather.get("summary") if isinstance(weather, dict) else None
    weather_lines = weather_summary[:2] if isinstance(weather_summary, list) else []
    output = [
        f"{prediction['station']} | {prediction['local_date']} | cutoff {prediction['cutoff_local']}",
        (
            f"Tmax: {_format_c(prediction.get('predicted_tmax_c'))} "
            f"(80% {_format_c(interval_low)}-{_format_c(interval_high)})"
        ),
        (
            f"Observed max: {_format_c(prediction.get('observed_max_to_cutoff_c'))}; "
            f"latest: {_format_c(prediction.get('last_temp_to_cutoff_c'))} "
            f"at {prediction.get('last_observation_local', 'unknown')}"
        ),
        (
            "Risk: "
            f">=29C {_format_percent(prediction.get('prob_tmax_ge_29c'))}, "
            f">=30C {_format_percent(prediction.get('prob_tmax_ge_30c'))}, "
            f"late warming {prediction.get('late_warming_warning', 'unknown')}"
        ),
    ]
    output.extend(f"Weather: {line}" for line in weather_lines)
    if prediction.get("next_update_local"):
        output.append(
            "Next update: "
            f"{prediction['next_update_local']} "
            f"({'recommended' if prediction.get('recommend_update_next_cutoff') else 'optional'})"
        )
    return output


def _format_error_entry(entry: dict) -> list[str]:
    return [
        f"{entry.get('station', 'unknown')} | {entry.get('local_date')} | cutoff {entry.get('cutoff_local')}",
        f"ERROR: {entry.get('error', 'unknown error')}",
    ]


def _format_c(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}C"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _any_station_history_is_short(configs: list[ProjectConfig], minimum_rows: int) -> bool:
    for config in configs:
        if not config.prefer_duckdb or not config.input_db.exists():
            continue
        try:
            observations = read_station_observations_from_duckdb(
                config.input_db,
                config.station,
                ["station", "valid"],
            )
        except Exception:
            return True
        if len(observations) < minimum_rows:
            return True
    return False


def _append_unique_lines(source_path: Path, target_path: Path) -> dict:
    source_lines = _read_nonempty_lines(source_path)
    target_lines = _read_nonempty_lines(target_path)
    existing = set(target_lines)
    new_lines = [line for line in source_lines if line not in existing]
    if new_lines:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8", newline="\n") as file:
            if target_lines:
                file.write("\n")
            file.write("\n".join(new_lines))
            file.write("\n")
    return {
        "source": str(source_path),
        "target": str(target_path),
        "read": len(source_lines),
        "appended": len(new_lines),
        "skipped_existing": len(source_lines) - len(new_lines),
    }


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
