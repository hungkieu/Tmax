from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rksi_tmax.config import ProjectConfig
from rksi_tmax.heat_risk import (
    M4_EXPERT_NAMES,
    _add_future_curve_targets,
    _false_plateau_rule,
    _integer_tmax_win_rates,
    _m4_gating_feature_columns,
    _m4_bundle_available,
    _m4_oof_folds,
    _m4_predict_remaining_heat_and_weights,
    _possible_new_peak_output,
    format_heat_risk_explanation,
    _make_single_cutoff_dataset,
    _metrics_by_cutoff,
    _not_highest_bet_output,
    openmeteo_heat_risk_feature_columns,
    _prediction_interval,
    _resolve_prediction_method,
    _regime_break_output,
    _threshold_probabilities,
    _threshold_probability,
    _tail_risk_interval,
    _underprediction_probabilities,
    _update_recommendation,
    _warming_strength_output,
)


def test_possible_new_peak_flags_suppressed_dip() -> None:
    out = _possible_new_peak_output(
        {"false_plateau_score": 2.5, "prob_tmax_already_reached": 0.5},
        conditional_tmax_c=32.1,
        predicted_tmax_c=31.4,
    )
    assert out["possible_new_peak"] is True
    assert out["planning_tmax_rounded_c"] == 32
    assert "possible_new_peak_warning" in out


def test_possible_new_peak_silent_when_not_suppressed() -> None:
    out = _possible_new_peak_output(
        {"false_plateau_score": 1.0, "prob_tmax_already_reached": 0.5},
        conditional_tmax_c=32.1,
        predicted_tmax_c=31.4,
    )
    assert out["possible_new_peak"] is False
    assert out["planning_tmax_c"] == 31.4
    assert "possible_new_peak_warning" not in out


def test_possible_new_peak_silent_when_peak_already_passed() -> None:
    out = _possible_new_peak_output(
        {"false_plateau_score": 2.5, "prob_tmax_already_reached": 0.85},
        conditional_tmax_c=33.0,
        predicted_tmax_c=32.0,
    )
    assert out["possible_new_peak"] is False


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


class FixedRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value)


class FixedM4GatingModel:
    classes_ = np.asarray(["A", "B"])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.full(len(features), 0.25), np.full(len(features), 0.75)])


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


def test_underprediction_probabilities_are_monotonic() -> None:
    features = pd.DataFrame({"x": [1.0]})
    classifiers = {
        "1.5": FixedProbabilityClassifier(0.6),
        "2.0": FixedProbabilityClassifier(0.8),
    }

    probabilities = _underprediction_probabilities(classifiers, features)

    assert probabilities["1.5"][0] == 0.6
    assert probabilities["2.0"][0] == 0.6


def test_integer_tmax_win_rates_use_half_up_rounded_candidates() -> None:
    frame = pd.DataFrame(
        {
            "predicted_tmax_c": [27.6, 27.4, 28.4],
            "tmax_c": [28.0, 28.0, 27.0],
        }
    )

    rates = _integer_tmax_win_rates(frame)

    assert rates["n"] == 3
    assert rates["tmax_win_count"] == 1
    assert rates["tmax_win_rate"] == pytest.approx(1 / 3)
    assert rates["tmax_plus_1_win_count"] == 1
    assert rates["tmax_plus_1_win_rate"] == pytest.approx(1 / 3)
    assert rates["tmax_minus_1_win_count"] == 1
    assert rates["tmax_minus_1_win_rate"] == pytest.approx(1 / 3)
    assert rates["combined_tmax_minus_1_to_plus_1_win_count"] == 3
    assert rates["combined_tmax_minus_1_to_plus_1_win_rate"] == 1.0


def test_openmeteo_feature_columns_use_available_cache_rows_for_missingness() -> None:
    frame = pd.DataFrame(
        {
            "base_feature": range(1000),
            "openmeteo_tmax_c": [np.nan] * 890 + list(range(110)),
            "openmeteo_hourly_temp_mean_c": [np.nan] * 890 + list(range(110)),
        }
    )

    columns = openmeteo_heat_risk_feature_columns(
        frame,
        ["base_feature"],
        missing_threshold=0.85,
    )

    assert "base_feature" in columns
    assert "openmeteo_tmax_c" in columns
    assert "openmeteo_hourly_temp_mean_c" in columns


def test_metrics_by_cutoff_include_integer_tmax_win_rates() -> None:
    frame = pd.DataFrame(
        {
            "cutoff_local": ["09:00", "09:00", "10:00"],
            "predicted_tmax_c": [27.6, 27.4, 28.4],
            "predicted_remaining_heat_c": [1.0, 1.0, 1.0],
            "remaining_heat_target_c": [1.0, 2.0, 0.0],
            "tmax_c": [28.0, 28.0, 27.0],
            "tmpc_max_to_cutoff": [27.0, 26.0, 27.0],
        }
    )

    rows = _metrics_by_cutoff(frame)

    first = rows[0]
    assert first["cutoff_local"] == "09:00"
    assert first["n"] == 2
    assert first["tmax_win_count"] == 1
    assert first["tmax_win_rate"] == 0.5
    assert first["tmax_plus_1_win_count"] == 1
    assert first["combined_tmax_minus_1_to_plus_1_win_rate"] == 1.0


def test_warming_strength_output_derives_class_probabilities() -> None:
    probabilities = {
        "0.5": np.asarray([0.8]),
        "2.0": np.asarray([0.6]),
        "4.0": np.asarray([0.2]),
    }

    output = _warming_strength_output(probabilities)

    assert output["warming_strength"] == "strong_warming"
    assert np.isclose(output["prob_no_or_weak_warming"], 0.2)
    assert np.isclose(output["prob_mild_warming"], 0.2)
    assert np.isclose(output["prob_strong_warming"], 0.4)
    assert np.isclose(output["prob_extreme_warming"], 0.2)


def test_tail_risk_interval_uses_late_warming_thresholds() -> None:
    interval = {"prediction_interval_80_high_c": 25.4}
    probabilities = {
        "2.0": np.asarray([0.70]),
        "3.0": np.asarray([0.35]),
        "4.0": np.asarray([0.20]),
    }

    output = _tail_risk_interval(interval, observed_max_c=22.0, late_warming_probabilities=probabilities)

    assert output["tail_risk_upper_c"] == 26.0
    assert output["tail_risk_interval_80_high_c"] == 26.0
    assert "prob_remaining_heat_ge_4_0" in output["tail_risk_reasons"]


def test_not_highest_bet_is_won_when_observed_max_already_exceeds_bet() -> None:
    output = _not_highest_bet_output(
        24.0,
        observed_max_c=25.0,
        tmax_threshold_probabilities={},
        remaining_heat_probabilities={},
    )

    assert output["win_probability"] == 1.0
    assert output["lose_probability"] == 0.0
    assert output["probability_basis"] == "observed_max_already_above_bet"


def test_not_highest_bet_interpolates_tmax_threshold_probability() -> None:
    output = _not_highest_bet_output(
        29.0,
        observed_max_c=27.0,
        tmax_threshold_probabilities={
            "28.0": np.asarray([0.8]),
            "30.0": np.asarray([0.2]),
        },
        remaining_heat_probabilities={"2.0": np.asarray([0.4])},
    )

    assert np.isclose(output["win_probability"], 0.5)
    assert output["probability_basis"] == "final_tmax_threshold_classifier_interpolated"


def test_not_highest_bet_uses_remaining_heat_when_tmax_threshold_is_out_of_range() -> None:
    output = _not_highest_bet_output(
        23.0,
        observed_max_c=22.0,
        tmax_threshold_probabilities={
            "28.0": np.asarray([0.8]),
            "30.0": np.asarray([0.2]),
        },
        remaining_heat_probabilities={
            "0.5": np.asarray([0.7]),
            "2.0": np.asarray([0.3]),
        },
    )

    assert np.isclose(output["win_probability"], 0.5666666666666667)
    assert output["probability_basis"] == "remaining_heat_classifier_interpolated"


def test_resolve_prediction_method_allows_supported_m3_alias() -> None:
    bundle = {
        "metrics": {"selected_prediction_method": "m1"},
        "openmeteo_regressor": object(),
        "openmeteo_feature_columns": ["openmeteo_tmax_c"],
    }

    assert _resolve_prediction_method(bundle, "m3") == "openmeteo"
    assert _resolve_prediction_method(bundle, "auto") == "m1"


def test_resolve_prediction_method_allows_supported_m4() -> None:
    bundle = {
        "metrics": {"selected_prediction_method": "m4"},
        "m4_experts": {"A": object()},
        "m4_expert_columns": {"A": ["base_feature"]},
        "m4_gating_model": object(),
        "m4_expert_names": ["A"],
    }

    assert _resolve_prediction_method(bundle, "m4") == "m4"
    assert _resolve_prediction_method(bundle, "auto") == "m4"


def test_m4_oof_folds_keep_dates_grouped() -> None:
    frame = pd.DataFrame(
        {
            "local_date": ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02", "2024-06-03"],
            "x": [1, 2, 3, 4, 5],
        }
    )

    folds = _m4_oof_folds(frame, fold_count=3)

    assert folds
    for train_dates, validation_dates in folds:
        assert train_dates.isdisjoint(validation_dates)
    validation_dates = set().union(*(validation for _, validation in folds))
    assert validation_dates == set(frame["local_date"].unique())


def test_m4_prediction_blends_experts_with_soft_weights() -> None:
    frame = pd.DataFrame({"base_feature": [1.0, 2.0], "gate_feature": [3.0, 4.0]})
    bundle = {
        "m4_experts": {"A": FixedRegressor(1.0), "B": FixedRegressor(3.0)},
        "m4_expert_columns": {"A": ["base_feature"], "B": ["base_feature"]},
        "m4_gating_model": FixedM4GatingModel(),
        "m4_gating_columns": [
            "gate_feature",
            "m4_expert_A_remaining_heat_c",
            "m4_expert_B_remaining_heat_c",
        ],
        "m4_expert_names": ["A", "B"],
    }

    prediction, weights = _m4_predict_remaining_heat_and_weights(bundle, frame)

    assert prediction is not None
    assert weights is not None
    assert prediction.tolist() == [2.5, 2.5]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert list(weights.columns) == ["A", "B"]


def test_m4_expert_set_excludes_solar_expert_and_gating_features() -> None:
    frame = pd.DataFrame(
        {
            "cutoff_minutes": [600],
            "openmeteo_tmax_c": [25.0],
            "solar_shortwave_sum_to_cutoff": [1000.0],
            "solar_high_but_temp_flat_flag": [1],
            "temp_rise_per_1000_ghi": [1.2],
            "m4_expert_A_remaining_heat_c": [1.0],
            "m4_expert_G_remaining_heat_c": [2.0],
        }
    )

    columns = _m4_gating_feature_columns(frame)

    assert "I" not in M4_EXPERT_NAMES
    assert "solar_shortwave_sum_to_cutoff" not in columns
    assert "solar_high_but_temp_flat_flag" not in columns
    assert "temp_rise_per_1000_ghi" not in columns


def test_m4_legacy_bundle_with_solar_expert_is_unavailable() -> None:
    bundle = {
        "metrics": {"selected_prediction_method": "m4"},
        "m4_experts": {"A": object(), "I": object()},
        "m4_expert_columns": {"A": ["base_feature"], "I": ["solar_shortwave_sum_to_cutoff"]},
        "m4_gating_model": object(),
        "m4_expert_names": ["A", "I"],
    }

    assert not _m4_bundle_available(bundle)
    with pytest.raises(ValueError, match="legacy|unavailable"):
        _resolve_prediction_method(bundle, "auto")


def test_resolve_prediction_method_rejects_unsupported_m3() -> None:
    bundle = {"metrics": {"selected_prediction_method": "m1"}}

    with pytest.raises(ValueError, match="M3/Open-Meteo"):
        _resolve_prediction_method(bundle, "m3")


def test_resolve_prediction_method_rejects_unsupported_m4() -> None:
    bundle = {"metrics": {"selected_prediction_method": "m1"}}

    with pytest.raises(ValueError, match="M4/Mixture-of-Experts"):
        _resolve_prediction_method(bundle, "m4")


def test_false_plateau_rule_detects_suppressed_heating() -> None:
    row = pd.Series(
        {
            "cutoff_minutes": 600,
            "temp_flat_duration_last_2h": 120,
            "temp_range_last_2h": 0.5,
            "weather_suppression_score": 1.7,
            "last_temp_equals_observed_max": 1,
            "cutoff_before_typical_peak": 1,
        }
    )

    result = _false_plateau_rule(row)

    assert result["suppressed_late_warming_warning"]
    assert result["warning_type"] == "weather_suppressed_false_plateau"


def test_format_heat_risk_explanation_includes_key_fields() -> None:
    text = format_heat_risk_explanation(
        {
            "station": "RJTT",
            "local_date": "2026-06-19",
            "cutoff_local": "10:00",
            "observed_max_to_cutoff_c": 22.0,
            "last_temp_to_cutoff_c": 22.0,
            "predicted_remaining_heat_c": 2.0,
            "predicted_tmax_c": 24.0,
            "prediction_method": "openmeteo",
            "prediction_interval_80_low_c": 22.5,
            "prediction_interval_80_high_c": 25.5,
            "thermal_phase": "uncertain_transition",
            "late_warming_risk": "high",
            "late_warming_warning": "elevated_late_warming_risk",
            "openmeteo_predicted_remaining_heat_c": 1.5,
            "weather_context": {
                "summary": [
                    "Có mưa trong 2 giờ gần đây.",
                    "Có mây thấp trong 2 giờ gần đây.",
                ]
            },
        }
    )

    assert "RJTT" in text
    assert "24.0C" in text
    assert "có rủi ro tăng nhiệt muộn" in text
    assert "Nhận xét thời tiết METAR" in text
    assert "Có mưa trong 2 giờ gần đây." in text


def test_update_recommendation_uses_policy_and_interval_width() -> None:
    interval = {"prediction_interval_80_low_c": 25.0, "prediction_interval_80_high_c": 27.0}
    row = pd.Series({"tmpc_last_to_cutoff": 26.0, "tmpc_max_to_cutoff": 26.0})
    policy = {
        "10:00": {
            "next_cutoff_local": "11:00",
            "median_abs_error_improvement_c": 0.05,
        }
    }

    recommendation = _update_recommendation("10:00", interval, policy, row=row)

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


def test_update_recommendation_does_not_recommend_same_or_past_cutoff() -> None:
    interval = {"prediction_interval_80_low_c": 25.0, "prediction_interval_80_high_c": 27.0}
    policy = {
        "12:30": {
            "next_cutoff_local": "13:00",
            "median_abs_error_improvement_c": 0.2,
        }
    }

    recommendation = _update_recommendation("13:00", interval, policy)

    assert recommendation["next_update_local"] is None
    assert not recommendation["recommend_update_next_cutoff"]


def test_prediction_interval_reports_practical_low() -> None:
    calibration = {
        "method": "conformal_by_cutoff",
        "overall": {"residual_p10_c": -2.0, "residual_p50_c": 0.0, "residual_p90_c": 1.0},
        "by_cutoff": {
            "12:00": {"residual_p10_c": -1.5, "residual_p50_c": 0.1, "residual_p90_c": 0.8}
        },
    }

    interval = _prediction_interval(28.5, calibration, "12:00", observed_max_c=28.0)

    assert interval["prediction_interval_80_low_raw_c"] == 27.0
    assert interval["prediction_interval_80_low_practical_c"] == 28.0
    assert interval["prediction_interval_80_low_c"] == 28.0


def test_future_curve_targets_use_local_minutes_and_keep_missing_horizons() -> None:
    frame = pd.DataFrame({"local_date": ["2024-06-02"], "cutoff_minutes": [600]})
    observations = pd.DataFrame(
        {
            "valid_local": pd.to_datetime(
                [
                    "2024-06-02 10:30+09:00",
                    "2024-06-02 12:00+09:00",
                ]
            ),
            "tmpf": [68.0, 77.0],
        }
    )

    result = _add_future_curve_targets(frame, observations, cutoff_minutes=600)

    assert result.loc[0, "future_tmpc_plus_30m"] == 20.0
    assert pd.isna(result.loc[0, "future_tmpc_plus_60m"])
    assert result.loc[0, "future_tmpc_plus_120m"] == 25.0


def test_future_curve_targets_use_nearest_observation_within_tolerance() -> None:
    frame = pd.DataFrame({"local_date": ["2024-06-02"], "cutoff_minutes": [600]})
    observations = pd.DataFrame(
        {
            "valid_local": pd.to_datetime(
                [
                    "2024-06-02 10:20+09:00",
                    "2024-06-02 11:52+09:00",
                ]
            ),
            "tmpf": [68.0, 77.0],
        }
    )

    result = _add_future_curve_targets(frame, observations, cutoff_minutes=600)

    assert result.loc[0, "future_tmpc_plus_30m"] == 20.0
    assert result.loc[0, "future_tmpc_plus_120m"] == 25.0


def test_regime_break_output_classifies_cooler_warmer_similar() -> None:
    cooler = _regime_break_output(
        pd.Series({"regime_break_cooler_than_recent": 1, "regime_break_score": 4.0})
    )
    warmer = _regime_break_output(
        pd.Series({"regime_break_warmer_than_recent": 1, "regime_break_score": 3.0})
    )
    similar = _regime_break_output(pd.Series({"regime_break_score": 0.5}))

    assert cooler["regime_break_type"] == "cooler_than_recent"
    assert warmer["regime_break_type"] == "warmer_than_recent"
    assert similar["regime_break_type"] == "similar_to_recent"
