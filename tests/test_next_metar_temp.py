from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.next_metar_temp import (
    FEATURE_COLUMNS,
    HORIZONS,
    build_next_metar_dataset,
    next_regular_metar_time,
    nth_regular_metar_time,
    predict_next_metar_temperature,
    predict_next_metars,
    read_prediction_log,
    should_promote_model,
    verify_next_metar_from_history,
    verify_next_metar_predictions,
)


def test_next_regular_metar_time_uses_station_schedule() -> None:
    seoul = ZoneInfo("Asia/Seoul")

    assert next_regular_metar_time("RKSI", datetime(2026, 6, 23, 9, 30, tzinfo=seoul)).isoformat() == (
        "2026-06-23T10:00:00+09:00"
    )
    assert next_regular_metar_time("RKSI", datetime(2026, 6, 23, 9, 31, tzinfo=seoul)).isoformat() == (
        "2026-06-23T10:00:00+09:00"
    )
    assert next_regular_metar_time("RKPK", datetime(2026, 6, 23, 9, 30, tzinfo=seoul)).isoformat() == (
        "2026-06-23T10:00:00+09:00"
    )


def test_nth_regular_metar_time_steps_by_station_interval() -> None:
    seoul = ZoneInfo("Asia/Seoul")
    observed = datetime(2026, 6, 23, 9, 15, tzinfo=seoul)

    rksi = [nth_regular_metar_time("RKSI", observed, n).isoformat() for n in (1, 2, 3, 4)]
    assert rksi == [
        "2026-06-23T09:30:00+09:00",
        "2026-06-23T10:00:00+09:00",
        "2026-06-23T10:30:00+09:00",
        "2026-06-23T11:00:00+09:00",
    ]

    rkpk = [nth_regular_metar_time("RKPK", observed, n).isoformat() for n in (1, 2)]
    assert rkpk == ["2026-06-23T10:00:00+09:00", "2026-06-23T11:00:00+09:00"]


def test_predict_next_metars_returns_all_horizons_and_logs_each(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"

    result = predict_next_metars(
        "RKSI",
        "2026-06-23T09:30:00+09:00",
        27.5,
        model_path=tmp_path / "missing.joblib",  # forces fallback rounding
        log_path=log_path,
    )

    preds = result["predictions"]
    assert [p["horizon"] for p in preds] == list(HORIZONS)
    assert [p["next_metar_at"] for p in preds] == [
        "2026-06-23T10:00:00+09:00",
        "2026-06-23T10:30:00+09:00",
        "2026-06-23T11:00:00+09:00",
        "2026-06-23T11:30:00+09:00",
    ]
    assert all(p["predicted_temp_c"] == 28 for p in preds)  # fallback round-half-up
    assert len(read_prediction_log(log_path)) == len(HORIZONS)


def test_dataset_includes_multiple_horizons(tmp_path: Path) -> None:
    csv_path = tmp_path / "rksi.csv"
    _write_asos_csv(
        csv_path,
        [
            ("RKSI", "2026-06-23 00:00", 20.0),
            ("RKSI", "2026-06-23 00:30", 21.0),
            ("RKSI", "2026-06-23 01:00", 22.0),
            ("RKSI", "2026-06-23 01:30", 23.0),
            ("RKSI", "2026-06-23 02:00", 24.0),
        ],
    )
    config = ProjectConfig(station="RKSI", input_csv=csv_path, prefer_duckdb=False)

    dataset = build_next_metar_dataset([config], tmp_path / "dataset.parquet")

    assert set(dataset["horizon"].unique()) >= {1, 2}
    # First observation (09:00 local) at horizon 2 targets the 10:00 METAR (22C).
    h2_first = dataset[(dataset["observed_at_local"].str.startswith("2026-06-23T09:00")) & (dataset["horizon"] == 2)]
    assert int(h2_first.iloc[0]["target_temp_c_int"]) == 22


def test_dataset_target_uses_next_regular_metar_without_future_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "rksi.csv"
    _write_asos_csv(
        csv_path,
        [
            ("RKSI", "2026-06-23 00:00", 20.0),
            ("RKSI", "2026-06-23 00:30", 21.0),
            ("RKSI", "2026-06-23 01:00", 22.0),
        ],
    )
    config = ProjectConfig(station="RKSI", input_csv=csv_path, prefer_duckdb=False)

    dataset = build_next_metar_dataset([config], tmp_path / "dataset.parquet")

    first = dataset.iloc[0]
    assert first["observed_at_local"] == "2026-06-23T09:00:00+09:00"
    assert first["next_metar_at_local"] == "2026-06-23T09:30:00+09:00"
    assert first["temp_c"] == 20.0
    assert first["target_temp_c_int"] == 21


def test_prediction_fallback_is_integer_and_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"

    result = predict_next_metar_temperature(
        "RKSI",
        "2026-06-23T09:30:00+09:00",
        27.5,
        model_path=tmp_path / "missing.joblib",
        log_path=log_path,
    )

    assert result["predicted_temp_c"] == 28
    assert isinstance(result["predicted_temp_c"], int)
    assert result["status"] == "fallback"
    assert read_prediction_log(log_path)[0]["predicted_temp_c"] == 28


def test_feature_columns_do_not_include_full_m3_or_weather_subfeatures() -> None:
    blocked_fragments = ("openmeteo_hourly", "cloud", "wind", "weather", "curve", "skyc", "sknt")

    assert "tmax_signal_c" in FEATURE_COLUMNS
    assert all(not any(fragment in column for fragment in blocked_fragments) for column in FEATURE_COLUMNS)


def test_verify_prediction_joins_correct_next_metar(tmp_path: Path) -> None:
    csv_path = tmp_path / "rksi.csv"
    _write_asos_csv(csv_path, [("RKSI", "2026-06-23 01:00", 28.0)])
    log_path = tmp_path / "predictions.jsonl"
    predict_next_metar_temperature(
        "RKSI",
        "2026-06-23T09:30:00+09:00",
        27.4,
        model_path=tmp_path / "missing.joblib",
        log_path=log_path,
    )
    config = ProjectConfig(station="RKSI", input_csv=csv_path, prefer_duckdb=False)

    result = verify_next_metar_predictions([config], log_path=log_path)
    records = read_prediction_log(log_path)

    assert result["verified"] == 1
    assert records[0]["verification_status"] == "verified"
    assert records[0]["actual_temp_c"] == 28
    assert records[0]["within_1c"] is True


def test_verify_from_mongo_history_uses_nearest_observation(tmp_path: Path) -> None:
    utc = ZoneInfo("UTC")
    log_path = tmp_path / "predictions.jsonl"
    predict_next_metar_temperature(
        "RKSI",
        "2026-06-23T09:30:00+09:00",
        27.4,
        model_path=tmp_path / "missing.joblib",
        log_path=log_path,
    )
    # Continuous stream like MongoDB: not aligned to the 10:00 (01:00 UTC) METAR slot.
    history = pd.DataFrame(
        {
            "valid_local": [
                datetime(2026, 6, 23, 0, 58, 12, tzinfo=utc),
                datetime(2026, 6, 23, 0, 59, 47, tzinfo=utc),  # nearest to 01:00 UTC
                datetime(2026, 6, 23, 1, 3, 5, tzinfo=utc),
            ],
            "temp_c": [27.6, 28.0, 28.4],
        }
    )

    result = verify_next_metar_from_history({"RKSI": history}, log_path=log_path)
    records = read_prediction_log(log_path)

    assert result["verified"] == 1
    assert records[0]["actual_temp_c"] == 28
    assert records[0]["verification_status"] == "verified"


def test_verify_from_mongo_history_skips_when_no_observation_in_window(tmp_path: Path) -> None:
    utc = ZoneInfo("UTC")
    log_path = tmp_path / "predictions.jsonl"
    predict_next_metar_temperature(
        "RKSI",
        "2026-06-23T09:30:00+09:00",
        27.4,
        model_path=tmp_path / "missing.joblib",
        log_path=log_path,
    )
    history = pd.DataFrame(
        {
            "valid_local": [datetime(2026, 6, 23, 2, 0, tzinfo=utc)],  # ~1h from slot
            "temp_c": [30.0],
        }
    )

    result = verify_next_metar_from_history(
        {"RKSI": history}, log_path=log_path, tolerance_seconds=300
    )
    records = read_prediction_log(log_path)

    assert result["verified"] == 0
    assert records[0]["verification_status"] == "pending"


def test_promotion_rejects_degraded_candidate() -> None:
    decision = should_promote_model(
        {
            "validation_mae_c": 0.9,
            "validation_exact_accuracy": 0.60,
        },
        {
            "validation_mae_c": 0.7,
            "validation_exact_accuracy": 0.70,
        },
    )

    assert decision["promote"] is False
    assert decision["reason"] == "validation_mae_degraded"


def _write_asos_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    frame = pd.DataFrame(
        {
            "station": [row[0] for row in rows],
            "valid": [row[1] for row in rows],
            "tmpf": [_c_to_f(row[2]) for row in rows],
        }
    )
    frame.to_csv(path, index=False)


def _c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0
