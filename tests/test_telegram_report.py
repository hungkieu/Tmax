from __future__ import annotations

from datetime import datetime, timezone

from rksi_tmax.telegram_report import format_telegram_report


def _prediction(station: str) -> dict:
    return {
        "station": station,
        "local_date": "2026-06-19",
        "cutoff_local": "12:00",
        "last_observation_local": "2026-06-19 12:00",
        "last_observation_lag_minutes": 0,
        "data_fresh_enough": True,
        "observed_max_to_cutoff_c": 30.0,
        "last_temp_to_cutoff_c": 30.0,
        "predicted_tmax_c": 31.2,
        "prediction_p50_c": 31.1,
        "prediction_interval_80_low_c": 30.4,
        "prediction_interval_80_high_c": 32.1,
        "monotonic_threshold_probabilities": {
            "27c": 1.0,
            "28c": 1.0,
            "29c": 1.0,
            "30c": 0.95,
            "31c": 0.55,
            "32c": 0.12,
            "33c": 0.01,
        },
        "predicted_remaining_heat_c": 1.2,
        "prob_remaining_heat_ge_2_0": 0.35,
        "prob_remaining_heat_ge_3_0": 0.12,
        "prob_remaining_heat_ge_4_0": 0.02,
        "prob_tmax_already_reached": 0.08,
        "thermal_phase": "peak_plateau",
        "late_warming_warning": "watch_false_plateau",
        "future_curve": {
            "2026-06-19 12:30": 30.5,
            "2026-06-19 13:00": 30.8,
        },
        "weather_context": {
            "summary": [
                "Có mây thấp trong 2 giờ gần đây; trần mây thấp nhất khoảng 1800 ft.",
                "Điều kiện bay có lúc ở mức MVFR hoặc xấu hơn.",
            ]
        },
        "next_update_local": "13:00",
        "recommend_update_next_cutoff": True,
    }


def test_format_telegram_report_includes_all_station_summaries() -> None:
    entries = [
        {"status": "ok", "prediction": _prediction("RKSI")},
        {"status": "ok", "prediction": _prediction("RKPK")},
        {"status": "ok", "prediction": _prediction("RJTT")},
        {"status": "ok", "prediction": _prediction("WSSS")},
    ]

    report = format_telegram_report(
        entries,
        generated_at_utc=datetime(2026, 6, 19, 5, 15, tzinfo=timezone.utc),
    )

    assert "Báo cáo heat-risk" in report
    assert "Tạo lúc: 2026-06-19 05:15 UTC" in report
    assert "RKSI | 2026-06-19 | cutoff 12:00" in report
    assert "WSSS | 2026-06-19 | cutoff 12:00" in report
    assert "Dự báo Tmax: 31.2C" in report
    assert ">=33C 1%" in report
    assert "Còn lại: còn tăng 1.2C" in report
    assert "Đường nhiệt tới: 12:30 30.5C -> 13:00 30.8C" in report
    assert "Weather: Có mây thấp" in report


def test_format_telegram_report_keeps_station_errors() -> None:
    report = format_telegram_report(
        [
            {"status": "ok", "prediction": _prediction("RKSI")},
            {
                "status": "error",
                "station": "WSSS",
                "local_date": "2026-06-19",
                "cutoff_local": "12:00",
                "error": "model artifact missing",
            },
        ],
        generated_at_utc=datetime(2026, 6, 19, 5, 15, tzinfo=timezone.utc),
    )

    assert "RKSI | 2026-06-19 | cutoff 12:00" in report
    assert "WSSS | 2026-06-19 | cutoff 12:00" in report
    assert "LỖI: model artifact missing" in report
