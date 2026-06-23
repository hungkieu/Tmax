from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rksi_tmax.config import ProjectConfig, _hhmm_to_minutes, load_config
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
    parser.add_argument("--output", default="artifacts/shared/telegram_report.md")
    parser.add_argument("--metar-file", default="data/shared/metar.txt")
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
    metar_file: str | Path = "data/shared/metar.txt",
    output_path: str | Path = "artifacts/shared/telegram_report.md",
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
        cutoff_local = _select_configured_cutoff(config, local_now)
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
        "Báo cáo heat-risk",
        f"Tạo lúc: {generated}",
        "",
    ]
    for entry in entries:
        if entry.get("status") == "error":
            lines.extend(_format_error_entry(entry))
        else:
            lines.extend(_format_prediction_entry(entry["prediction"]))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _select_configured_cutoff(config: ProjectConfig, local_now: datetime) -> str:
    current_minutes = local_now.hour * 60 + local_now.minute
    configured = sorted(config.heat_risk_cutoffs, key=_hhmm_to_minutes)
    eligible = [cutoff for cutoff in configured if _hhmm_to_minutes(cutoff) <= current_minutes]
    return eligible[-1] if eligible else configured[0]


def _format_prediction_entry(prediction: dict) -> list[str]:
    interval_low = prediction.get("prediction_interval_80_low_c")
    interval_high = prediction.get("prediction_interval_80_high_c")
    tail_risk_upper = prediction.get("tail_risk_upper_c")
    weather = prediction.get("weather_context", {})
    weather_summary = weather.get("summary") if isinstance(weather, dict) else None
    weather_lines = weather_summary[:3] if isinstance(weather_summary, list) else []
    threshold_text = _format_threshold_probabilities(prediction)
    remaining_heat_text = _format_remaining_heat_probabilities(prediction)
    curve_text = _format_future_curve(prediction)
    warning_reasons = prediction.get("warning_reasons")
    warning_text = "; ".join(warning_reasons[:2]) if isinstance(warning_reasons, list) else None
    output = [
        f"{prediction['station']} | {prediction['local_date']} | cutoff {prediction['cutoff_local']}",
        (
            f"Dự báo Tmax: {_format_c(prediction.get('predicted_tmax_c'))}; "
            f"trung vị phân bố {_format_c(prediction.get('prediction_p50_c'))}; "
            f"khoảng 80% {_format_c(interval_low)}-{_format_c(interval_high)}"
        ),
        (
            f"Đã quan sát: max {_format_c(prediction.get('observed_max_to_cutoff_c'))}; "
            f"mới nhất {_format_c(prediction.get('last_temp_to_cutoff_c'))} "
            f"lúc {prediction.get('last_observation_local', 'không rõ')}"
        ),
        (
            f"Xác suất ngưỡng: {threshold_text}"
        ),
        remaining_heat_text,
        (
            f"Pha nhiệt: {_translate_phase(prediction.get('thermal_phase'))}; "
            f"mức tăng còn lại: {_translate_warming_strength(prediction.get('warming_strength'))}; "
            f"rủi ro tăng muộn: {_translate_warning(prediction.get('late_warming_warning'))}"
        ),
    ]
    if tail_risk_upper is not None and interval_high is not None and float(tail_risk_upper) > float(interval_high):
        output.append(f"Upper tail-risk: {_format_c(tail_risk_upper)}")
    if prediction.get("openmeteo_features_available"):
        openmeteo_model = prediction.get("openmeteo_predicted_tmax_c")
        selected = prediction.get("predicted_tmax_c")
        delta = (
            float(openmeteo_model) - float(selected)
            if openmeteo_model is not None and selected is not None
            else None
        )
        output.append(
            "Open-Meteo: "
            f"Tmax {_format_c(prediction.get('openmeteo_forecast_tmax_c'))}; "
            f"M3 sau hiệu chỉnh {_format_c(openmeteo_model)}; "
            f"chênh forecast chính {_format_signed_c(delta)}"
        )
    bet = prediction.get("not_highest_bet")
    if isinstance(bet, dict):
        output.append(
            "Cược "
            f"{_format_c(bet.get('bet_temp_c'))} không phải Tmax: "
            f"thắng {_format_percent(bet.get('win_probability'))}, "
            f"thua {_format_percent(bet.get('lose_probability'))}"
        )
    if curve_text:
        output.append(f"Đường nhiệt tới: {curve_text}")
    if warning_text:
        output.append(f"Lý do cảnh báo: {_translate_warning_reasons(warning_text)}")
    if prediction.get("recommended_action"):
        output.append(f"Khuyến nghị: {_translate_recommended_action(prediction['recommended_action'])}")
    output.extend(f"Weather: {line}" for line in weather_lines)
    if prediction.get("next_update_local"):
        output.append(
            "Cập nhật tiếp: "
            f"{prediction['next_update_local']} "
            f"({'nên cập nhật' if prediction.get('recommend_update_next_cutoff') else 'tuỳ chọn'})"
        )
    return output


def _format_error_entry(entry: dict) -> list[str]:
    return [
        f"{entry.get('station', 'unknown')} | {entry.get('local_date')} | cutoff {entry.get('cutoff_local')}",
        f"LỖI: {entry.get('error', 'unknown error')}",
    ]


def _format_threshold_probabilities(prediction: dict) -> str:
    probabilities = prediction.get("monotonic_threshold_probabilities")
    if isinstance(probabilities, dict) and probabilities:
        items = sorted(
            ((_threshold_from_slug(slug), value) for slug, value in probabilities.items()),
            key=lambda item: item[0],
        )
    else:
        items = sorted(
            (
                (_threshold_from_slug(match.group(1)), value)
                for key, value in prediction.items()
                if (match := re.fullmatch(r"prob_tmax_ge_([0-9p]+c)", key))
            ),
            key=lambda item: item[0],
        )
    if not items:
        return "không có"
    return ", ".join(f">={threshold:g}C {_format_percent(value)}" for threshold, value in items)


def _format_remaining_heat_probabilities(prediction: dict) -> str:
    parts = [
        f"dự báo còn tăng {_format_c(prediction.get('predicted_remaining_heat_c'))}",
        f"xác suất còn tăng >=2C {_format_percent(prediction.get('prob_remaining_heat_ge_2_0'))}",
        f">=3C {_format_percent(prediction.get('prob_remaining_heat_ge_3_0'))}",
        f">=4C {_format_percent(prediction.get('prob_remaining_heat_ge_4_0'))}",
    ]
    if prediction.get("prob_tmax_already_reached") is not None:
        parts.append(f"xác suất đã đạt Tmax {_format_percent(prediction.get('prob_tmax_already_reached'))}")
    return "Còn lại: " + ", ".join(parts)


def _format_future_curve(prediction: dict) -> str | None:
    future_curve = prediction.get("future_curve")
    if not isinstance(future_curve, dict) or not future_curve:
        return None
    items = sorted(future_curve.items())[:4]
    return " -> ".join(f"{timestamp[-5:]} {_format_c(value)}" for timestamp, value in items)


def _threshold_from_slug(slug: str) -> float:
    return float(slug.removesuffix("c").replace("p", "."))


def _translate_phase(value: object) -> str:
    return {
        "pre_peak_ramp": "đang tăng trước đỉnh",
        "peak_plateau": "gần vùng đỉnh/đi ngang",
        "post_peak_decline": "sau đỉnh, đang giảm",
        "uncertain_transition": "chuyển pha chưa chắc chắn",
    }.get(str(value), "không rõ")


def _translate_warning(value: object) -> str:
    return {
        "low": "thấp",
        "watch_false_plateau": "cần theo dõi plateau giả",
        "high_late_warming_risk": "cao",
        "extreme_late_warming_possible": "rất cao",
        "high_extreme_late_warming_risk": "rất cao",
    }.get(str(value), str(value) if value is not None else "không rõ")


def _translate_warming_strength(value: object) -> str:
    return {
        "no_or_weak_warming": "yếu/gần như không tăng",
        "mild_warming": "tăng nhẹ",
        "strong_warming": "tăng mạnh",
        "extreme_warming": "tăng rất mạnh",
    }.get(str(value), str(value) if value is not None else "không rõ")


def _translate_warning_reasons(text: str) -> str:
    replacements = {
        "classifier probability for remaining heat >= 2C is high": "xác suất còn tăng >=2C cao",
        "classifier probability for remaining heat >= 3C is high": "xác suất còn tăng >=3C cao",
        "classifier probability for remaining heat >= 4C is high": "xác suất còn tăng >=4C cao",
        "point forecast may be too low": "dự báo điểm có thể thấp hơn thực tế",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _translate_recommended_action(value: str) -> str:
    return {
        "Do not treat point forecast as final; update at next cutoff.": (
            "Không xem dự báo điểm là kết luận cuối; nên cập nhật ở cutoff kế tiếp."
        ),
        "Point forecast can be used with normal interval uncertainty.": (
            "Có thể dùng dự báo điểm cùng khoảng bất định như bình thường."
        ),
    }.get(value, value)


def _format_c(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}C"
    except (TypeError, ValueError):
        return str(value)


def _format_signed_c(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.1f}C"
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
