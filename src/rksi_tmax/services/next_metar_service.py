"""Orchestration for the one-click Next-METAR cycle used by the UI.

A single ``run_cycle`` call does everything under the hood:

1. read the latest live temperature for each Korea station from MongoDB and
   predict the next regular METAR temperature;
2. verify previously pending predictions against the MongoDB observation history;
3. return a structured, human-readable log of every step plus rolling health and
   the most recent predictions.

Errors (missing ``MONGODB_URI``, a station with no live document, etc.) are
captured as steps rather than raised, so the UI can always render a clear report.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from rksi_tmax.next_metar_temp import (
    PREDICTION_LOG_PATH,
    SUPPORTED_STATIONS,
    predict_next_metars,
    read_prediction_log,
    verification_health,
    verify_next_metar_from_history,
)


def run_cycle(
    stations: Iterable[str] | None = None,
    *,
    verify_hours: int = 48,
    tolerance_seconds: int = 300,
    recent_limit: int = 20,
    window: int = 100,
) -> dict[str, object]:
    targets = [s.upper() for s in (stations or SUPPORTED_STATIONS)]
    steps: list[dict[str, object]] = []

    # MongoDB access is imported lazily so the rest of the UI loads even when the
    # mongo driver / URI is not configured yet.
    try:
        from rksi_tmax.mongo_source import (
            get_current_temperature,
            get_temperature_history,
            mongodb_uri,
        )

        mongodb_uri()  # fail fast with a clear message if MONGODB_URI is missing
    except Exception as exc:
        return {
            "ran_at": _now_local_iso(),
            "ok": False,
            "error": str(exc),
            "steps": [
                {
                    "action": "connect",
                    "status": "error",
                    "title": "Kết nối MongoDB",
                    "summary": str(exc),
                    "detail": None,
                }
            ],
            "health": {},
            "recent": _recent_rows(recent_limit, targets),
        }

    # 1) Predict every station from its latest live observation.
    for station in targets:
        try:
            current = get_current_temperature(station)
        except Exception as exc:
            steps.append(_step("predict", station, "error", f"Lỗi đọc nhiệt live: {exc}"))
            continue
        if current is None:
            steps.append(
                _step(
                    "predict",
                    station,
                    "skipped",
                    "Không có nhiệt live trong MongoDB (sân bay chưa được ghi dữ liệu).",
                )
            )
            continue
        result = predict_next_metars(
            station,
            current.observed_at_local_iso,
            current.temp_c,
        )
        preds = result.get("predictions", [])
        forecast = ", ".join(
            f"{_hhmm(p['next_metar_at'])}→{p['predicted_temp_c']}°C" for p in preds
        )
        summary = (
            f"Quan trắc {current.temp_c:.1f}°C lúc {_hhmm(result['observed_at'])} "
            f"→ dự báo {len(preds)} METAR kế tiếp: {forecast}"
        )
        status = "ok"
        if any(p.get("fallback") for p in preds):
            status = "warning"
            summary += " (model lỗi → dùng fallback làm tròn)"
        steps.append(_step("predict", station, status, summary, result))

    # 2) Verify pending predictions against the MongoDB observation history.
    try:
        since = datetime.now(timezone.utc) - timedelta(hours=verify_hours)
        history_by_station = {
            station: get_temperature_history(station, since=since) for station in targets
        }
        verify_result = verify_next_metar_from_history(
            history_by_station,
            window=window,
            tolerance_seconds=tolerance_seconds,
        )
        scoped_station = targets[0] if len(targets) == 1 else None
        pending = _pending_count(read_prediction_log(), scoped_station)
        verify_summary = (
            f"Đã chấm điểm {verify_result['verified']} dự báo, còn {pending} chờ."
        )
        steps.append(_step("verify", None, "ok", verify_summary, verify_result))
        health = verification_health(read_prediction_log(), station=scoped_station, window=window)
    except Exception as exc:
        steps.append(_step("verify", None, "error", f"Lỗi verify: {exc}"))
        scoped_station = targets[0] if len(targets) == 1 else None
        health = verification_health(read_prediction_log(), station=scoped_station, window=window)

    return {
        "ran_at": _now_local_iso(),
        "ok": True,
        "error": None,
        "steps": steps,
        "health": health,
        "recent": _recent_rows(recent_limit, targets),
    }


def _pending_count(records: list[dict[str, object]], station: str | None) -> int:
    if station is not None:
        normalized = station.upper()
        records = [r for r in records if str(r.get("station", "")).upper() == normalized]
    return sum(1 for r in records if r.get("verification_status") != "verified")


def _recent_rows(limit: int, stations: Iterable[str] | None = None) -> list[dict[str, object]]:
    records = read_prediction_log(PREDICTION_LOG_PATH)
    if stations is not None:
        wanted = {s.upper() for s in stations}
        records = [r for r in records if str(r.get("station", "")).upper() in wanted]
    rows = []
    for record in reversed(records[-limit:]):
        rows.append(
            {
                "Sân bay": record.get("station"),
                "Bước": record.get("horizon"),
                "Quan trắc": _short(record.get("observed_at")),
                "METAR": _short(record.get("next_metar_at")),
                "Dự báo °C": record.get("predicted_temp_c"),
                "Thực tế °C": record.get("actual_temp_c"),
                "Lệch °C": record.get("verification_error_c"),
                "Trạng thái": record.get("verification_status"),
            }
        )
    return rows


def _step(
    action: str,
    station: str | None,
    status: str,
    summary: str,
    detail: object | None = None,
) -> dict[str, object]:
    titles = {"predict": "Dự báo", "verify": "Chấm điểm", "connect": "Kết nối"}
    title = titles.get(action, action)
    if station:
        title = f"{title} {station}"
    return {
        "action": action,
        "station": station,
        "status": status,
        "title": title,
        "summary": summary,
        "detail": detail,
    }


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _hhmm(value: object) -> str:
    try:
        return datetime.fromisoformat(str(value)).strftime("%H:%M")
    except (TypeError, ValueError):
        return str(value)


def _short(value: object) -> str | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value)).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)
