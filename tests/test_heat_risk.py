from __future__ import annotations

import numpy as np
import pandas as pd

from rksi_tmax.config import ProjectConfig
from rksi_tmax.heat_risk import (
    _make_single_cutoff_dataset,
    _threshold_probabilities,
    _threshold_probability,
    _update_recommendation,
)


class DummyClassifier:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.full(len(features), 0.8), np.full(len(features), 0.2)])


class FixedProbabilityClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [np.full(len(features), 1.0 - self.probability), np.full(len(features), self.probability)]
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


def test_remaining_heat_target_uses_observed_max_to_cutoff() -> None:
    config = ProjectConfig(cutoff_local="09:00", complete_day_min_local="15:00")

    dataset = _make_single_cutoff_dataset(_observations(), config, "09:00")

    row = dataset[dataset["local_date"] == "2024-06-02"].iloc[0]
    assert row["remaining_heat_target_c"] == row["tmax_c"] - row["tmpc_max_to_cutoff"]
    assert row["remaining_heat_target_c"] >= 0.0


def test_threshold_probability_is_one_when_threshold_already_observed() -> None:
    features = pd.DataFrame({"x": [1.0, 2.0]})
    observed_max = pd.Series([31.0, 29.0])

    probability = _threshold_probability(DummyClassifier(), features, observed_max, 30.0)

    assert probability.tolist() == [1.0, 0.2]


def test_threshold_probabilities_are_monotonic() -> None:
    features = pd.DataFrame({"x": [1.0]})
    observed_max = pd.Series([27.0])
    classifiers = {
        "28.0": FixedProbabilityClassifier(0.2),
        "29.0": FixedProbabilityClassifier(0.7),
        "30.0": FixedProbabilityClassifier(0.4),
    }

    probabilities = _threshold_probabilities(classifiers, features, observed_max)

    assert probabilities["28.0"][0] == 0.2
    assert probabilities["29.0"][0] == 0.2
    assert probabilities["30.0"][0] == 0.2


def test_update_recommendation_uses_policy_and_interval_width() -> None:
    interval = {"prediction_interval_80_low_c": 25.0, "prediction_interval_80_high_c": 27.0}
    policy = {
        "10:00": {
            "next_cutoff_local": "11:00",
            "median_abs_error_improvement_c": 0.05,
        }
    }

    recommendation = _update_recommendation("10:00", interval, policy)

    assert recommendation["next_update_local"] == "11:00"
    assert recommendation["recommend_update_next_cutoff"]


def test_update_recommendation_uses_nearest_policy_for_arbitrary_cutoff() -> None:
    interval = {"prediction_interval_80_low_c": 25.0, "prediction_interval_80_high_c": 26.0}
    policy = {
        "10:00": {
            "next_cutoff_local": "11:00",
            "median_abs_error_improvement_c": 0.2,
        }
    }

    recommendation = _update_recommendation("10:30", interval, policy)

    assert recommendation["next_update_local"] == "11:00"
    assert recommendation["recommend_update_next_cutoff"]
