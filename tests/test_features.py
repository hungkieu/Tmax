from __future__ import annotations

import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.features import make_daily_dataset


def test_local_day_grouping_uses_seoul_date() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="10:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI", "RKSI", "RKSI", "RKSI"],
            "valid_local": pd.to_datetime(
                [
                    "2024-01-01 08:30+09:00",
                    "2024-01-01 10:30+09:00",
                    "2024-01-02 08:30+09:00",
                    "2024-01-02 10:30+09:00",
                ]
            ),
            "tmpf": [32.0, 50.0, 35.0, 55.0],
            "dwpf": [25.0, 30.0, 28.0, 33.0],
            "relh": [80.0, 60.0, 75.0, 55.0],
            "drct": [100.0, 120.0, 100.0, 120.0],
            "sknt": [4.0, 6.0, 4.0, 6.0],
            "p01i": [0.0, 0.0, 0.0, 0.0],
            "alti": [30.0, 30.0, 30.0, 30.0],
            "mslp": [1015.0, 1014.0, 1013.0, 1012.0],
            "vsby": [6.0, 6.0, 6.0, 6.0],
            "gust": [None, None, None, None],
            "skyl1": [3000.0, 3000.0, 3000.0, 3000.0],
            "skyl2": [None, None, None, None],
            "skyl3": [None, None, None, None],
            "skyl4": [None, None, None, None],
            "feel": [30.0, 50.0, 35.0, 55.0],
            "skyc1": ["SCT", "BKN", "FEW", "OVC"],
            "skyc2": [None, None, None, None],
            "skyc3": [None, None, None, None],
            "skyc4": [None, None, None, None],
            "wxcodes": ["BR", None, None, "RA"],
        }
    )

    dataset = make_daily_dataset(observations, config)

    assert dataset["local_date"].tolist() == ["2024-01-02"]
    assert dataset["tmax_f"].iloc[0] == 55.0


def test_features_do_not_use_after_cutoff_observations() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="10:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI"] * 4,
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 08:00+09:00",
                    "2024-06-01 12:00+09:00",
                    "2024-06-02 08:00+09:00",
                    "2024-06-02 12:00+09:00",
                ]
            ),
            "tmpf": [60.0, 90.0, 61.0, 100.0],
            "dwpf": [50.0, 70.0, 51.0, 75.0],
            "relh": [70.0, 40.0, 70.0, 40.0],
            "drct": [100.0, 200.0, 100.0, 200.0],
            "sknt": [5.0, 20.0, 5.0, 20.0],
            "p01i": [0.0, 0.0, 0.0, 0.0],
            "alti": [29.9, 29.8, 29.9, 29.8],
            "mslp": [1010.0, 1008.0, 1010.0, 1008.0],
            "vsby": [6.0, 6.0, 6.0, 6.0],
            "gust": [None, 30.0, None, 30.0],
            "skyl1": [3000.0, 1000.0, 3000.0, 1000.0],
            "skyl2": [None, None, None, None],
            "skyl3": [None, None, None, None],
            "skyl4": [None, None, None, None],
            "feel": [60.0, 95.0, 61.0, 105.0],
            "skyc1": ["FEW", "OVC", "FEW", "OVC"],
            "skyc2": [None, None, None, None],
            "skyc3": [None, None, None, None],
            "skyc4": [None, None, None, None],
            "wxcodes": [None, "RA", None, "RA"],
        }
    )

    dataset = make_daily_dataset(observations, config)
    row = dataset.iloc[0]

    assert row["local_date"] == "2024-06-02"
    assert row["tmpf_last_to_cutoff"] == 61.0
    assert row["tmpf_max_to_cutoff"] == 61.0
    assert row["tmax_f"] == 100.0
    assert row["precip_observed_to_cutoff"] == 0


def test_incomplete_local_day_is_kept_for_prediction_without_target() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="23:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI", "RKSI", "RKSI", "RKSI", "RKSI", "RKSI"],
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 08:00+09:00",
                    "2024-06-01 23:30+09:00",
                    "2024-06-02 08:00+09:00",
                    "2024-06-02 23:30+09:00",
                    "2024-06-03 08:00+09:00",
                    "2024-06-03 12:00+09:00",
                ]
            ),
            "tmpf": [60.0, 90.0, 61.0, 95.0, 62.0, 100.0],
            "dwpf": [50.0, 70.0, 51.0, 72.0, 52.0, 75.0],
            "relh": [70.0, 40.0, 70.0, 40.0, 70.0, 40.0],
            "drct": [100.0, 200.0, 100.0, 200.0, 100.0, 200.0],
            "sknt": [5.0, 20.0, 5.0, 20.0, 5.0, 20.0],
            "p01i": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "alti": [29.9, 29.8, 29.9, 29.8, 29.9, 29.8],
            "mslp": [1010.0, 1008.0, 1010.0, 1008.0, 1010.0, 1008.0],
            "vsby": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            "gust": [None, 30.0, None, 30.0, None, 30.0],
            "skyl1": [3000.0, 1000.0, 3000.0, 1000.0, 3000.0, 1000.0],
            "skyl2": [None, None, None, None, None, None],
            "skyl3": [None, None, None, None, None, None],
            "skyl4": [None, None, None, None, None, None],
            "feel": [60.0, 95.0, 61.0, 98.0, 62.0, 105.0],
            "skyc1": ["FEW", "OVC", "FEW", "OVC", "FEW", "OVC"],
            "skyc2": [None, None, None, None, None, None],
            "skyc3": [None, None, None, None, None, None],
            "skyc4": [None, None, None, None, None, None],
            "wxcodes": [None, "RA", None, "RA", None, "RA"],
        }
    )

    dataset = make_daily_dataset(observations, config)

    assert dataset["local_date"].tolist() == ["2024-06-02", "2024-06-03"]
    assert dataset.loc[0, "target_complete"] == 1
    assert dataset.loc[1, "target_complete"] == 0
    assert pd.isna(dataset.loc[1, "tmax_c"])
    assert dataset.loc[1, "tmpf_last_to_cutoff"] == 62.0


def test_remaining_heat_climatology_uses_prior_days_only() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="10:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI"] * 6,
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 08:00+09:00",
                    "2024-06-01 12:00+09:00",
                    "2024-06-02 08:00+09:00",
                    "2024-06-02 12:00+09:00",
                    "2024-06-03 08:00+09:00",
                    "2024-06-03 12:00+09:00",
                ]
            ),
            "tmpf": [68.0, 86.0, 68.0, 95.0, 68.0, 104.0],
            "dwpf": [50.0, 70.0, 51.0, 72.0, 52.0, 75.0],
            "relh": [70.0, 40.0, 70.0, 40.0, 70.0, 40.0],
            "drct": [100.0, 200.0, 100.0, 200.0, 100.0, 200.0],
            "sknt": [5.0, 20.0, 5.0, 20.0, 5.0, 20.0],
            "p01i": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "alti": [29.9, 29.8, 29.9, 29.8, 29.9, 29.8],
            "mslp": [1010.0, 1008.0, 1010.0, 1008.0, 1010.0, 1008.0],
            "vsby": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            "gust": [None, 30.0, None, 30.0, None, 30.0],
            "skyl1": [3000.0, 1000.0, 3000.0, 1000.0, 3000.0, 1000.0],
            "skyl2": [None, None, None, None, None, None],
            "skyl3": [None, None, None, None, None, None],
            "skyl4": [None, None, None, None, None, None],
            "feel": [68.0, 86.0, 68.0, 95.0, 68.0, 104.0],
            "skyc1": ["FEW", "OVC", "FEW", "OVC", "FEW", "OVC"],
            "skyc2": [None, None, None, None, None, None],
            "skyc3": [None, None, None, None, None, None],
            "skyc4": [None, None, None, None, None, None],
            "wxcodes": [None, "RA", None, "RA", None, "RA"],
        }
    )

    dataset = make_daily_dataset(observations, config)

    assert dataset["local_date"].tolist() == ["2024-06-02", "2024-06-03"]
    assert dataset.loc[0, "remaining_heat_climo_month_c"] == 10.0
    assert dataset.loc[0, "expected_tmax_from_cutoff_c"] == 30.0
    assert dataset.loc[1, "remaining_heat_climo_month_c"] == 12.5
    assert dataset.loc[1, "expected_tmax_from_cutoff_c"] == 32.5


def test_wind_regime_and_cloud_fog_clearing_use_cutoff_window() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="12:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI"] * 6,
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 07:00+09:00",
                    "2024-06-01 08:00+09:00",
                    "2024-06-01 12:00+09:00",
                    "2024-06-02 07:00+09:00",
                    "2024-06-02 08:00+09:00",
                    "2024-06-02 12:00+09:00",
                ]
            ),
            "tmpf": [68.0, 70.0, 86.0, 68.0, 72.0, 95.0],
            "dwpf": [60.0, 60.0, 65.0, 60.0, 60.0, 65.0],
            "relh": [90.0, 80.0, 50.0, 90.0, 75.0, 45.0],
            "drct": [270.0, 80.0, 180.0, 270.0, 90.0, 180.0],
            "sknt": [5.0, 5.0, 10.0, 5.0, 5.0, 10.0],
            "p01i": [0.0] * 6,
            "alti": [29.9] * 6,
            "mslp": [1010.0] * 6,
            "vsby": [6.0] * 6,
            "gust": [None] * 6,
            "skyl1": [1000.0, 3000.0, 500.0, 1000.0, 3000.0, 500.0],
            "skyl2": [None] * 6,
            "skyl3": [None] * 6,
            "skyl4": [None] * 6,
            "feel": [68.0, 70.0, 86.0, 68.0, 72.0, 95.0],
            "skyc1": ["BKN", "FEW", "OVC", "BKN", "FEW", "OVC"],
            "skyc2": [None] * 6,
            "skyc3": [None] * 6,
            "skyc4": [None] * 6,
            "wxcodes": ["BR", None, "RA", "BR", None, "RA"],
        }
    )

    dataset = make_daily_dataset(observations, config)
    row = dataset[dataset["local_date"] == "2024-06-02"].iloc[0]

    assert row["first_cloud_cover_to_cutoff"] == 3
    assert row["last_cloud_cover_to_cutoff"] == 1
    assert row["cloud_clearing_to_cutoff"] == 2
    assert row["fog_cleared_to_cutoff"] == 1
    assert row["fog_developed_to_cutoff"] == 0
    assert row["wind_regime_e_last_to_cutoff"] == 1
    assert row["wind_regime_w_last_to_cutoff"] == 0


def test_phase_plateau_features_use_only_cutoff_window() -> None:
    config = ProjectConfig(cutoff_local="10:00", complete_day_min_local="12:00")
    observations = pd.DataFrame(
        {
            "station": ["RKSI"] * 8,
            "valid_local": pd.to_datetime(
                [
                    "2024-06-01 09:00+09:00",
                    "2024-06-01 10:00+09:00",
                    "2024-06-01 12:00+09:00",
                    "2024-06-01 13:00+09:00",
                    "2024-06-02 09:00+09:00",
                    "2024-06-02 09:30+09:00",
                    "2024-06-02 10:00+09:00",
                    "2024-06-02 12:00+09:00",
                ]
            ),
            "tmpf": [68.0, 70.0, 90.0, 95.0, 68.0, 72.0, 72.0, 100.0],
            "dwpf": [55.0] * 8,
            "relh": [70.0] * 8,
            "drct": [100.0] * 8,
            "sknt": [5.0] * 8,
            "p01i": [0.0] * 8,
            "alti": [29.9] * 8,
            "mslp": [1010.0] * 8,
            "vsby": [6.0] * 8,
            "gust": [None] * 8,
            "skyl1": [3000.0] * 8,
            "skyl2": [None] * 8,
            "skyl3": [None] * 8,
            "skyl4": [None] * 8,
            "feel": [68.0, 70.0, 90.0, 95.0, 68.0, 72.0, 72.0, 100.0],
            "skyc1": ["FEW"] * 8,
            "skyc2": [None] * 8,
            "skyc3": [None] * 8,
            "skyc4": [None] * 8,
            "wxcodes": [None] * 8,
        }
    )

    dataset = make_daily_dataset(observations, config)
    row = dataset[dataset["local_date"] == "2024-06-02"].iloc[0]

    assert row["tmpf_max_to_cutoff"] == 72.0
    assert row["last_temp_equals_observed_max"] == 1
    assert row["minutes_since_observed_max"] == 0
    assert row["observed_max_count_so_far"] == 2
    assert row["tmax_f"] == 100.0
