from __future__ import annotations

from pathlib import Path

import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.features import make_daily_dataset
from rksi_tmax.openmeteo import (
    load_openmeteo_daily,
    load_openmeteo_features_for_dates,
    load_openmeteo_json,
)


def test_load_openmeteo_daily_skips_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "openmeteo.csv"
    _write_openmeteo_csv(csv_path, [("2026-06-20", 22.2, 55, 4.9)])

    frame = load_openmeteo_daily(csv_path)

    assert frame.loc[0, "local_date"] == "2026-06-20"
    assert frame.loc[0, "openmeteo_tmax_c"] == 22.2
    assert frame.loc[0, "openmeteo_weather_code"] == 55
    assert frame.loc[0, "openmeteo_precipitation_flag"] == 1


def test_live_openmeteo_file_overrides_history_for_date(tmp_path: Path) -> None:
    history_path = tmp_path / "openmeteo-rksi.csv"
    live_path = tmp_path / "openmeteo-rksi-2026-06-21.csv"
    _write_openmeteo_csv(history_path, [("2026-06-21", 24.0, 3, 0.0)])
    _write_openmeteo_csv(live_path, [("2026-06-21", 27.0, 61, 2.0)])

    frame = load_openmeteo_features_for_dates(
        history_path,
        str(tmp_path / "openmeteo-rksi-{date}.csv"),
        ["2026-06-21"],
    )

    assert frame is not None
    assert frame.loc[0, "openmeteo_tmax_c"] == 27.0
    assert frame.loc[0, "openmeteo_weather_code"] == 61


def test_load_openmeteo_json_adds_hourly_features(tmp_path: Path) -> None:
    json_path = tmp_path / "openmeteo-rksi-history.json"
    json_path.write_text(
        """
{
  "daily": {
    "time": ["2026-06-20"],
    "weather_code": [3],
    "temperature_2m_max": [29.5],
    "rain_sum": [1.2],
    "precipitation_sum": [1.5],
    "precipitation_hours": [2],
    "wind_speed_10m_max": [24.0],
    "wind_gusts_10m_max": [35.0]
  },
  "hourly": {
    "time": ["2026-06-20T00:00", "2026-06-20T09:00", "2026-06-20T12:00"],
    "temperature_2m": [20.0, 25.0, 30.0],
    "weather_code": [1, 2, 3],
    "wind_speed_10m": [5.0, 10.0, 20.0],
    "wind_gusts_10m": [7.0, 16.0, 28.0],
    "cloud_cover": [10, 40, 80],
    "visibility": [10000, 7000, 5000],
    "rain": [0.0, 0.2, 1.0],
    "precipitation": [0.0, 0.3, 1.2],
    "precipitation_probability": [5, 20, 70]
  }
}
""",
        encoding="utf-8",
    )

    frame = load_openmeteo_json(json_path)

    assert frame.loc[0, "openmeteo_tmax_c"] == 29.5
    assert frame.loc[0, "openmeteo_hourly_temp_max_c"] == 30.0
    assert frame.loc[0, "openmeteo_hourly_temp_09z_c"] == 25.0
    assert frame.loc[0, "openmeteo_hourly_precipitation_probability_max_pct"] == 70


def test_json_live_overrides_history_for_date(tmp_path: Path) -> None:
    history_path = tmp_path / "openmeteo-rksi-history.json"
    live_path = tmp_path / "openmeteo-rksi-2026-06-21.json"
    _write_openmeteo_json(history_path, "2026-06-21", 24.0)
    _write_openmeteo_json(live_path, "2026-06-21", 28.0)

    frame = load_openmeteo_features_for_dates(
        None,
        None,
        ["2026-06-21"],
        history_path,
        str(tmp_path / "openmeteo-rksi-{date}.json"),
    )

    assert frame is not None
    assert frame.loc[0, "openmeteo_tmax_c"] == 28.0


def test_make_daily_dataset_adds_openmeteo_features(tmp_path: Path) -> None:
    openmeteo_path = tmp_path / "openmeteo-rksi.csv"
    _write_openmeteo_csv(
        openmeteo_path,
        [
            ("2024-06-01", 25.0, 3, 0.0),
            ("2024-06-02", 28.0, 61, 3.0),
            ("2024-06-03", 29.0, 3, 0.0),
        ],
    )
    config = ProjectConfig(
        cutoff_local="09:00",
        complete_day_min_local="15:00",
        openmeteo_history_csv=openmeteo_path,
    )

    dataset = make_daily_dataset(_observations(), config)
    row = dataset[dataset["local_date"] == "2024-06-02"].iloc[0]

    assert row["openmeteo_tmax_c"] == 28.0
    assert row["openmeteo_expected_remaining_heat_c"] > 0.0
    assert row["openmeteo_precipitation_flag"] == 1


def _write_openmeteo_csv(path: Path, rows: list[tuple[str, float, int, float]]) -> None:
    lines = [
        "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation",
        "37.45,126.4375,25.0,0,GMT,GMT",
        "",
        (
            "time,temperature_2m_max (°C),weather_code (wmo code),precipitation_sum (mm),"
            "precipitation_hours (h),rain_sum (mm),wind_speed_10m_max (km/h),"
            "wind_gusts_10m_max (km/h)"
        ),
    ]
    for local_date, tmax, code, precipitation in rows:
        lines.append(f"{local_date},{tmax},{code},{precipitation},1.0,{precipitation},20.0,30.0")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_openmeteo_json(path: Path, local_date: str, tmax: float) -> None:
    path.write_text(
        (
            '{"daily":{"time":["%s"],"weather_code":[3],"temperature_2m_max":[%s],'
            '"rain_sum":[0],"precipitation_sum":[0],"precipitation_hours":[0],'
            '"wind_speed_10m_max":[10],"wind_gusts_10m_max":[20]},'
            '"hourly":{"time":["%sT09:00"],"temperature_2m":[%s],"weather_code":[3],'
            '"wind_speed_10m":[10],"wind_gusts_10m":[20],"cloud_cover":[20],'
            '"visibility":[10000],"rain":[0],"precipitation":[0],'
            '"precipitation_probability":[0]}}'
        )
        % (local_date, tmax, local_date, tmax),
        encoding="utf-8",
    )


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["RKSI"] * 6,
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 09:00+09:00",
                    "2024-06-01 15:00+09:00",
                    "2024-06-02 09:00+09:00",
                    "2024-06-02 15:00+09:00",
                    "2024-06-03 09:00+09:00",
                    "2024-06-03 15:00+09:00",
                ]
            ),
            "tmpf": [68.0, 86.0, 70.0, 91.0, 75.0, 96.0],
            "dwpf": [50.0, 65.0, 51.0, 66.0, 52.0, 67.0],
            "relh": [70.0, 50.0, 70.0, 50.0, 70.0, 50.0],
            "drct": [100.0, 200.0, 100.0, 200.0, 100.0, 200.0],
            "sknt": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0],
            "p01i": [0.0] * 6,
            "alti": [29.9] * 6,
            "mslp": [1010.0] * 6,
            "vsby": [6.0] * 6,
            "gust": [None] * 6,
            "skyl1": [3000.0] * 6,
            "skyl2": [None] * 6,
            "skyl3": [None] * 6,
            "skyl4": [None] * 6,
            "feel": [68.0, 86.0, 70.0, 91.0, 75.0, 96.0],
            "skyc1": ["FEW"] * 6,
            "skyc2": [None] * 6,
            "skyc3": [None] * 6,
            "skyc4": [None] * 6,
            "wxcodes": [None] * 6,
        }
    )
