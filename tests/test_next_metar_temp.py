from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rksi_tmax.config import ProjectConfig
from rksi_tmax.next_metar_temp import (
    make_next_metar_temp_dataset,
    next_metar_temp_feature_columns,
)


def test_next_metar_features_include_trough_prev3_weather_and_openmeteo(tmp_path: Path) -> None:
    openmeteo_path = tmp_path / "openmeteo-history.json"
    _write_openmeteo_json(openmeteo_path)
    config = ProjectConfig(
        station="RKSI",
        timezone="Asia/Seoul",
        openmeteo_history_json=openmeteo_path,
        openmeteo_timezone="GMT",
    )

    dataset = make_next_metar_temp_dataset(_observations(), config, include_target=True)
    row = dataset[
        dataset["valid_local"].astype(str).str.startswith("2024-06-04 08:00")
    ].iloc[0]

    assert row["target_temp_c_int"] == 25
    assert row["tmpc_min_so_far"] == pytest.approx(18.0)
    assert row["minutes_since_trough_so_far"] == 120
    assert row["temp_rise_from_trough_so_far_c"] == pytest.approx(6.0)
    assert row["prev3_rise_from_trough_mean_c"] == pytest.approx(3.0)
    assert row["today_vs_prev3_rise_from_trough_c"] == pytest.approx(3.0)
    assert row["cloud_cover_max_current"] == 3
    assert row["low_cloud_current"] == 1
    assert row["precip_current"] == 1
    assert row["wind_regime_e_current"] == 1
    assert row["openmeteo_current_temperature_2m"] == pytest.approx(23.0)
    assert row["openmeteo_target_temperature_2m"] == pytest.approx(25.0)
    assert row["openmeteo_target_minus_current_cloud_cover"] == pytest.approx(20.0)


def test_next_metar_feature_columns_exclude_targets_and_metadata() -> None:
    config = ProjectConfig(station="RKSI", timezone="Asia/Seoul")

    dataset = make_next_metar_temp_dataset(_observations(), config, include_target=True)
    columns = next_metar_temp_feature_columns(dataset)

    assert "target_temp_c_int" not in columns
    assert "target_delta_c_int" not in columns
    assert "valid_local" not in columns
    assert "temp_rise_from_trough_so_far_c" in columns
    assert "today_vs_prev3_rise_from_trough_c" in columns
    assert "cloud_cover_max_current" in columns


def _observations() -> pd.DataFrame:
    rows = []
    for day, temps in [
        ("2024-06-01", [18.0, 19.0, 21.0, 22.0]),
        ("2024-06-02", [18.0, 20.0, 21.0, 22.0]),
        ("2024-06-03", [18.0, 19.0, 21.0, 22.0]),
        ("2024-06-04", [18.0, 21.0, 24.0, 25.0]),
    ]:
        for time, temp_c in zip(["06:00", "07:00", "08:00", "09:00"], temps, strict=True):
            rows.append(
                {
                    "station": "RKSI",
                    "valid_local": pd.Timestamp(f"{day} {time}", tz="Asia/Seoul"),
                    "tmpf": temp_c * 9.0 / 5.0 + 32.0,
                    "dwpf": 15.0 * 9.0 / 5.0 + 32.0,
                    "relh": 70.0,
                    "drct": 90.0,
                    "sknt": 8.0,
                    "gust": 12.0,
                    "vsby": 6.0,
                    "alti": 29.9,
                    "mslp": 1010.0,
                    "skyc1": "BKN",
                    "skyc2": None,
                    "skyc3": None,
                    "skyc4": None,
                    "skyl1": 1500.0,
                    "skyl2": None,
                    "skyl3": None,
                    "skyl4": None,
                    "wxcodes": "RA" if day == "2024-06-04" and time == "08:00" else "",
                    "metar": f"METAR RKSI {day} {time}",
                }
            )
    return pd.DataFrame(rows)


def _write_openmeteo_json(path: Path) -> None:
    path.write_text(
        """
{
  "hourly": {
    "time": [
      "2024-06-03T23:00",
      "2024-06-04T00:00"
    ],
    "temperature_2m": [23.0, 25.0],
    "weather_code": [61, 61],
    "wind_speed_10m": [12.0, 15.0],
    "wind_gusts_10m": [18.0, 22.0],
    "cloud_cover": [60.0, 80.0],
    "visibility": [8000.0, 7000.0],
    "rain": [0.2, 0.4],
    "precipitation": [0.2, 0.5],
    "precipitation_probability": [40.0, 70.0]
  }
}
""",
        encoding="utf-8",
    )
