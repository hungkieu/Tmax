from __future__ import annotations

import json
import os
import pathlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from rksi_tmax.config import ProjectConfig, _hhmm_to_minutes, _minutes_to_hhmm
from rksi_tmax.features import load_observations, make_daily_dataset


TARGET_COLUMN = "remaining_heat_target_c"
FINAL_TMAX_COLUMN = "tmax_c"
NON_FEATURE_COLUMNS = {
    "local_date",
    "cutoff_local",
    "tmax_f",
    "tmax_c",
    "tmin_f",
    "tmin_c",
    "obs_count_full_day",
    "last_full_day_minute",
    "tmax_minute",
    "target_complete",
    "remaining_heat_target_c",
    "thermal_phase_target",
}
UPDATE_MIN_IMPROVEMENT_C = 0.15
UPDATE_MIN_INTERVAL_WIDTH_C = 1.5
REMAINING_HEAT_THRESHOLDS_C = (0.5, 1.0, 2.0, 3.0, 4.0)
UNDERPREDICTION_THRESHOLDS_C = (1.5, 2.0)
CONTINUING_HEAT_THRESHOLD_C = 0.5
SENSITIVE_PROBABILITY_RANGE = (0.20, 0.70)
FUTURE_CURVE_HORIZONS_MINUTES = (30, 60, 90, 120, 150, 180)
THERMAL_PHASE_LABELS = (
    "pre_peak_ramp",
    "peak_plateau",
    "post_peak_decline",
    "uncertain_transition",
)
M1_FEATURE_PREFIXES = (
    "temp_rise_",
    "last_temp_equals_observed_max",
    "minutes_since_observed_max",
    "observed_max_is_latest_observation",
    "observed_max_count_so_far",
    "duration_within_",
    "temp_range_last_2h",
    "temp_std_last_2h",
    "month_tmax_",
    "month_median_tmax_minute",
    "cutoff_minutes_before_monthly_median_tmax_time",
    "month_remaining_heat_",
    "cutoff_before_typical_peak",
    "temp_flat_duration_",
    "rain_seen_",
    "low_cloud_",
    "ceiling_min_",
    "visibility_min_",
    "visibility_low_",
    "mvfr_or_worse_",
    "weather_suppression_score",
    "false_plateau_candidate",
    "last3_",
    "today_vs_last3_",
    "regime_break_",
    "tmax_lag_",
)


def _load_model_bundle(model_path: str | Path) -> dict:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]
    return joblib.load(model_path)


def build_heat_risk_dataset(
    config: ProjectConfig,
    input_csv: str | Path | None = None,
    output_parquet: str | Path | None = None,
) -> pd.DataFrame:
    observations = load_observations(input_csv or config.input_csv, config)
    dataset = make_heat_risk_dataset(observations, config)

    output_path = Path(output_parquet or config.heat_risk_dataset_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(dataset).write_parquet(output_path)
    return dataset


def make_heat_risk_dataset(observations: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    frames = [
        _make_single_cutoff_dataset(observations, config, cutoff)
        for cutoff in config.heat_risk_cutoffs
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["local_date", "cutoff_minutes"])


def _make_single_cutoff_dataset(
    observations: pd.DataFrame,
    config: ProjectConfig,
    cutoff_local: str,
) -> pd.DataFrame:
    cutoff_local = _normalize_cutoff(cutoff_local)
    cutoff_config = replace(config, cutoff_local=cutoff_local)
    frame = make_daily_dataset(observations, cutoff_config)
    frame["cutoff_local"] = cutoff_local
    frame["cutoff_minutes"] = _hhmm_to_minutes(cutoff_local)
    frame[TARGET_COLUMN] = frame[FINAL_TMAX_COLUMN] - frame["tmpc_max_to_cutoff"]
    frame.loc[frame[TARGET_COLUMN] < 0.0, TARGET_COLUMN] = 0.0
    frame = _add_future_curve_targets(frame, observations, frame["cutoff_minutes"].iloc[0])
    frame["thermal_phase_target"] = frame.apply(_thermal_phase_target, axis=1)
    return frame


def _add_future_curve_targets(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    cutoff_minutes: int,
) -> pd.DataFrame:
    data = observations.copy()
    data["valid_local"] = pd.to_datetime(data["valid_local"])
    data["local_date"] = data["valid_local"].dt.date.astype(str)
    data["local_minutes"] = data["valid_local"].dt.hour * 60 + data["valid_local"].dt.minute
    data["tmpc"] = (pd.to_numeric(data["tmpf"], errors="coerce") - 32.0) * (5.0 / 9.0)
    output = frame.copy()
    for horizon in FUTURE_CURVE_HORIZONS_MINUTES:
        target_minute = cutoff_minutes + horizon
        target = (
            data[data["local_minutes"] == target_minute][["local_date", "tmpc"]]
            .dropna(subset=["tmpc"])
            .rename(columns={"tmpc": _future_target_column(horizon)})
        )
        output = output.merge(target, on="local_date", how="left")
    return output


def _thermal_phase_target(row: pd.Series) -> str:
    if pd.isna(row.get("tmax_minute")) or pd.isna(row.get(TARGET_COLUMN)):
        return "uncertain_transition"
    if float(row["tmax_minute"]) <= float(row["cutoff_minutes"]):
        return "post_peak_decline"
    remaining_heat = float(row[TARGET_COLUMN])
    if remaining_heat <= 0.5 or (
        pd.notna(row.get("tmpc_last_to_cutoff"))
        and pd.notna(row.get(FINAL_TMAX_COLUMN))
        and abs(float(row[FINAL_TMAX_COLUMN]) - float(row["tmpc_last_to_cutoff"])) <= 1.0
    ):
        return "peak_plateau"
    if float(row.get("temp_rise_last_60m", 0.0) or 0.0) > 0.1:
        return "pre_peak_ramp"
    return "uncertain_transition"


def load_heat_risk_table(path: str | Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas().sort_values(["local_date", "cutoff_minutes"])


def heat_risk_feature_columns(dataset: pd.DataFrame, missing_threshold: float = 1.0) -> list[str]:
    threshold_columns = {
        column
        for column in dataset.columns
        if column.startswith("target_tmax_ge_")
        or column.startswith("target_remaining_heat_ge_")
        or column.startswith("future_tmpc_plus_")
    }
    excluded = NON_FEATURE_COLUMNS | threshold_columns
    return [
        column
        for column in dataset.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(dataset[column])
        and not dataset[column].isna().all()
        and dataset[column].isna().mean() < missing_threshold
    ]


def m0_heat_risk_feature_columns(
    dataset: pd.DataFrame,
    missing_threshold: float = 1.0,
) -> list[str]:
    return [
        column
        for column in heat_risk_feature_columns(dataset, missing_threshold)
        if not _is_m1_feature(column)
    ]


def _is_m1_feature(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in M1_FEATURE_PREFIXES)


def train_heat_risk_model(config: ProjectConfig) -> dict:
    dataset = load_heat_risk_table(config.heat_risk_dataset_parquet).dropna(
        subset=[TARGET_COLUMN, FINAL_TMAX_COLUMN]
    )
    if len(dataset) < 100:
        raise ValueError("Need at least 100 multi-cutoff rows to train heat risk models.")

    dates = pd.Series(sorted(dataset["local_date"].unique()))
    split_index = max(1, int(len(dates) * (1.0 - config.test_fraction)))
    if split_index >= len(dates):
        split_index = len(dates) - 1
    train_dates = set(dates.iloc[:split_index])
    test_dates = set(dates.iloc[split_index:])
    train = dataset[dataset["local_date"].isin(train_dates)].copy()
    test = dataset[dataset["local_date"].isin(test_dates)].copy()

    columns = m0_heat_risk_feature_columns(train, missing_threshold=config.feature_missing_threshold)
    extended_columns = heat_risk_feature_columns(
        train,
        missing_threshold=config.feature_missing_threshold,
    )
    regressor = _regression_pipeline(config)
    regressor.fit(train[columns], train[TARGET_COLUMN])

    m1_regressor = _regression_pipeline(config)
    m1_regressor.fit(train[extended_columns], train[TARGET_COLUMN])

    continuation_classifiers = {}
    for threshold in REMAINING_HEAT_THRESHOLDS_C:
        label = _remaining_heat_label(threshold)
        train[label] = (train[TARGET_COLUMN] >= threshold).astype(int)
        test[label] = (test[TARGET_COLUMN] >= threshold).astype(int)
        if train[label].nunique() < 2:
            continue
        classifier = _classifier_pipeline(config)
        classifier.fit(train[columns], train[label])
        continuation_classifiers[str(threshold)] = classifier

    late_warming_classifiers = {}
    for threshold in REMAINING_HEAT_THRESHOLDS_C:
        label = _remaining_heat_label(threshold)
        if train[label].nunique() < 2:
            continue
        classifier = _classifier_pipeline(config)
        classifier.fit(train[extended_columns], train[label])
        late_warming_classifiers[str(threshold)] = classifier

    continuing_train = train[train[TARGET_COLUMN] >= CONTINUING_HEAT_THRESHOLD_C]
    conditional_regressor = _regression_pipeline(config)
    if len(continuing_train) >= 100:
        conditional_regressor.fit(continuing_train[columns], continuing_train[TARGET_COLUMN])
    else:
        conditional_regressor.fit(train[columns], train[TARGET_COLUMN])

    classifiers = {}
    for threshold in config.heat_risk_thresholds_c:
        label = _threshold_label(threshold)
        train[label] = (train[FINAL_TMAX_COLUMN] >= threshold).astype(int)
        test[label] = (test[FINAL_TMAX_COLUMN] >= threshold).astype(int)
        if train[label].nunique() < 2:
            continue
        classifier = _classifier_pipeline(config)
        classifier.fit(train[columns], train[label])
        classifiers[str(threshold)] = classifier

    phase_classifier = None
    if train["thermal_phase_target"].nunique() > 1:
        phase_classifier = _classifier_pipeline(config)
        phase_classifier.fit(train[extended_columns], train["thermal_phase_target"])

    curve_models = {}
    for horizon in FUTURE_CURVE_HORIZONS_MINUTES:
        target_column = _future_target_column(horizon)
        available_train = train.dropna(subset=[target_column])
        if len(available_train) < 100:
            continue
        model = _regression_pipeline(config)
        model.fit(available_train[extended_columns], available_train[target_column])
        curve_models[str(horizon)] = model

    raw_threshold_probabilities = _raw_threshold_probabilities(
        classifiers,
        test[columns],
    )
    threshold_probabilities = _monotonic_operational_threshold_probabilities(
        classifiers,
        test[columns],
        test["tmpc_max_to_cutoff"],
    )
    threshold_metrics = {
        threshold_text: _threshold_metrics(
            train[_threshold_label(float(threshold_text))],
            test[_threshold_label(float(threshold_text))],
            raw_threshold_probabilities[threshold_text],
            threshold_probabilities[threshold_text],
            test["tmpc_max_to_cutoff"],
            float(threshold_text),
        )
        for threshold_text in threshold_probabilities
    }

    direct_remaining_prediction = _clip_remaining(regressor.predict(test[columns]))
    m1_remaining_prediction = _clip_remaining(m1_regressor.predict(test[extended_columns]))
    continuation_probabilities = _remaining_heat_probabilities(
        continuation_classifiers,
        test[columns],
    )
    late_warming_probabilities = _remaining_heat_probabilities(
        late_warming_classifiers,
        test[extended_columns],
    )
    remaining_heat_probability_metrics = {
        threshold_text: _remaining_heat_probability_metrics(
            train[_remaining_heat_label(float(threshold_text))],
            test[_remaining_heat_label(float(threshold_text))],
            probability,
        )
        for threshold_text, probability in late_warming_probabilities.items()
    }
    conditional_remaining_prediction = _clip_remaining(conditional_regressor.predict(test[columns]))
    two_stage_remaining_prediction = _two_stage_remaining_prediction(
        continuation_probabilities,
        conditional_remaining_prediction,
    )
    direct_mae = float(mean_absolute_error(test[TARGET_COLUMN], direct_remaining_prediction))
    two_stage_mae = float(mean_absolute_error(test[TARGET_COLUMN], two_stage_remaining_prediction))
    selected_prediction_method = "two_stage" if two_stage_mae <= direct_mae else "direct"
    remaining_prediction = (
        two_stage_remaining_prediction
        if selected_prediction_method == "two_stage"
        else direct_remaining_prediction
    )
    train_direct_remaining_prediction = _clip_remaining(regressor.predict(train[columns]))
    train_continuation_probabilities = _remaining_heat_probabilities(
        continuation_classifiers,
        train[columns],
    )
    train_conditional_remaining_prediction = _clip_remaining(
        conditional_regressor.predict(train[columns])
    )
    train_two_stage_remaining_prediction = _two_stage_remaining_prediction(
        train_continuation_probabilities,
        train_conditional_remaining_prediction,
    )
    train_remaining_prediction = (
        train_two_stage_remaining_prediction
        if selected_prediction_method == "two_stage"
        else train_direct_remaining_prediction
    )
    tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + remaining_prediction
    train_tmax_prediction = train["tmpc_max_to_cutoff"].to_numpy() + train_remaining_prediction
    m1_tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + m1_remaining_prediction
    underprediction_classifiers = _fit_underprediction_classifiers(
        config,
        train,
        extended_columns,
        train_tmax_prediction,
    )
    underprediction_probabilities = _underprediction_probabilities(
        underprediction_classifiers,
        test[extended_columns],
    )
    curve_predictions = _curve_predictions(curve_models, test[extended_columns])
    curve_tmax_prediction = _curve_tmax_prediction(test["tmpc_max_to_cutoff"], curve_predictions)
    residual = test[FINAL_TMAX_COLUMN].to_numpy() - tmax_prediction
    update_policy = _build_update_policy(test, tmax_prediction)
    interval_calibration = _build_interval_calibration(test, residual)
    phase_metrics = _phase_metrics(phase_classifier, test, extended_columns)
    curve_metrics = _curve_metrics(test, curve_predictions, curve_tmax_prediction)
    late_warming_metrics = _late_warming_event_metrics(
        test,
        late_warming_probabilities,
    )
    underprediction_metrics = _underprediction_event_metrics(
        test,
        underprediction_probabilities,
        tmax_prediction,
    )

    metrics = {
        "station": config.station,
        "target": TARGET_COLUMN,
        "cutoffs": list(config.heat_risk_cutoffs),
        "thresholds_c": list(config.heat_risk_thresholds_c),
        "n_rows": int(len(dataset)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_start": str(min(train_dates)),
        "train_end": str(max(train_dates)),
        "test_start": str(min(test_dates)),
        "test_end": str(max(test_dates)),
        "remaining_heat_mae_c": float(mean_absolute_error(test[TARGET_COLUMN], remaining_prediction)),
        "direct_remaining_heat_mae_c": direct_mae,
        "two_stage_remaining_heat_mae_c": two_stage_mae,
        "selected_prediction_method": selected_prediction_method,
        "tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], tmax_prediction)),
        "tmax_rmse_c": _rmse(test[FINAL_TMAX_COLUMN], tmax_prediction),
        "tmax_bias_c": float(np.mean(tmax_prediction - test[FINAL_TMAX_COLUMN].to_numpy())),
        "m0_heat_risk_tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], tmax_prediction)),
        "m1_phase_feature_tmax_mae_c": float(
            mean_absolute_error(test[FINAL_TMAX_COLUMN], m1_tmax_prediction)
        ),
        "curve_predicted_tmax_mae_c": curve_metrics.get("curve_predicted_tmax_mae_c"),
        "observed_max_baseline_mae_c": float(
            mean_absolute_error(test[FINAL_TMAX_COLUMN], test["tmpc_max_to_cutoff"])
        ),
        "threshold_metrics": threshold_metrics,
        "remaining_heat_probability_metrics": remaining_heat_probability_metrics,
        "late_warming_metrics": late_warming_metrics,
        "underprediction_metrics": underprediction_metrics,
        "phase_metrics": phase_metrics,
        "curve_metrics": curve_metrics,
        "update_policy": update_policy,
        "interval_calibration": interval_calibration,
        "feature_count": len(columns),
        "extended_feature_count": len(extended_columns),
        "feature_missing_threshold": config.feature_missing_threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    config.heat_risk_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "m1_regressor": m1_regressor,
            "conditional_regressor": conditional_regressor,
            "classifiers": classifiers,
            "continuation_classifiers": continuation_classifiers,
            "late_warming_classifiers": late_warming_classifiers,
            "underprediction_classifiers": underprediction_classifiers,
            "phase_classifier": phase_classifier,
            "curve_models": curve_models,
            "feature_columns": columns,
            "extended_feature_columns": extended_columns,
            "metrics": metrics,
            "config": config,
            "update_policy": update_policy,
            "interval_calibration": interval_calibration,
        },
        config.heat_risk_model_path,
    )
    config.heat_risk_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.heat_risk_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def validate_heat_risk_model(config: ProjectConfig) -> dict:
    dataset = load_heat_risk_table(config.heat_risk_dataset_parquet).dropna(
        subset=[TARGET_COLUMN, FINAL_TMAX_COLUMN]
    )
    bundle = _load_model_bundle(config.heat_risk_model_path)
    metrics = bundle["metrics"]
    columns = bundle["feature_columns"]
    extended_columns = bundle.get("extended_feature_columns", columns)

    dates = pd.Series(sorted(dataset["local_date"].unique()))
    test_dates = set(
        date
        for date in dates
        if metrics["test_start"] <= str(date) <= metrics["test_end"]
    )
    test = dataset[dataset["local_date"].isin(test_dates)].copy()

    remaining_prediction = _predict_remaining_heat(bundle, test[columns])
    m1_remaining_prediction = _clip_remaining(
        bundle.get("m1_regressor", bundle["regressor"]).predict(test[extended_columns])
    )
    tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + remaining_prediction
    m1_tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + m1_remaining_prediction
    curve_predictions = _curve_predictions(bundle.get("curve_models", {}), test[extended_columns])
    curve_tmax_prediction = _curve_tmax_prediction(test["tmpc_max_to_cutoff"], curve_predictions)
    test["predicted_remaining_heat_c"] = remaining_prediction
    test["predicted_tmax_c"] = tmax_prediction
    test["m1_predicted_tmax_c"] = m1_tmax_prediction
    test["curve_predicted_tmax_c"] = curve_tmax_prediction
    test["error_c"] = test["predicted_tmax_c"] - test[FINAL_TMAX_COLUMN]
    test["abs_error_c"] = test["error_c"].abs()
    test["curve_abs_error_c"] = (test["curve_predicted_tmax_c"] - test[FINAL_TMAX_COLUMN]).abs()

    report = {
        "summary": {
            "station": config.station,
            "test_start": metrics["test_start"],
            "test_end": metrics["test_end"],
            "n_test": int(len(test)),
            "tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], tmax_prediction)),
            "tmax_rmse_c": _rmse(test[FINAL_TMAX_COLUMN], tmax_prediction),
            "remaining_heat_mae_c": float(
                mean_absolute_error(test[TARGET_COLUMN], remaining_prediction)
            ),
            "observed_max_baseline_mae_c": float(
                mean_absolute_error(test[FINAL_TMAX_COLUMN], test["tmpc_max_to_cutoff"])
            ),
        },
        "metrics_by_cutoff": _metrics_by_cutoff(test),
        "threshold_metrics": metrics["threshold_metrics"],
        "remaining_heat_probability_metrics": metrics.get("remaining_heat_probability_metrics", {}),
        "update_policy": metrics["update_policy"],
        "interval_calibration": metrics["interval_calibration"],
        "interval_coverage": _interval_coverage(test, tmax_prediction, metrics["interval_calibration"]),
        "model_comparison": _model_comparison(test),
        "phase_metrics": metrics.get("phase_metrics", {}),
        "curve_metrics": _curve_metrics(test, curve_predictions, curve_tmax_prediction),
        "late_warming_metrics": metrics.get("late_warming_metrics", {}),
        "underprediction_metrics": metrics.get("underprediction_metrics", {}),
        "top_errors": _top_heat_risk_errors(test).to_dict(orient="records"),
        "top_error_days": _top_error_days(test).to_dict(orient="records"),
        "top_error_reduction_vs_curve": _top_error_reduction(test),
    }

    artifacts_dir = Path(config.heat_risk_metrics_path).parent
    artifact_stem = _artifact_stem(config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"{artifact_stem}_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report["top_errors"]).to_csv(
        artifacts_dir / f"{artifact_stem}_top_errors.csv",
        index=False,
    )
    pd.DataFrame(report["top_error_days"]).to_csv(
        artifacts_dir / f"{artifact_stem}_top_error_days.csv",
        index=False,
    )
    _plot_heat_risk_diagnostics(test, report, artifacts_dir / f"{artifact_stem}_diagnostics.png")
    _plot_thermal_curve_diagnostics(
        test,
        report,
        artifacts_dir / f"{artifact_stem}_thermal_curve_diagnostics.png",
    )
    return report


def predict_heat_risk(
    config: ProjectConfig,
    local_date: str,
    cutoff_local: str,
    dataset_path: str | Path | None = None,
) -> dict:
    cutoff_local = _normalize_cutoff(cutoff_local)
    bundle = _load_model_bundle(config.heat_risk_model_path)
    columns = bundle["feature_columns"]
    extended_columns = bundle.get("extended_feature_columns", columns)
    row = _prediction_row(config, local_date, cutoff_local, dataset_path)
    missing = [column for column in columns if column not in row.columns]
    if missing:
        raise ValueError(f"Prediction row is missing model features: {missing}")
    extended_missing = [column for column in extended_columns if column not in row.columns]
    if extended_missing:
        raise ValueError(f"Prediction row is missing extended model features: {extended_missing}")

    direct_remaining_heat = float(_clip_remaining(bundle["regressor"].predict(row[columns]))[0])
    remaining_probabilities = _remaining_heat_probabilities(
        bundle.get("continuation_classifiers", {}),
        row[columns],
    )
    late_warming_probabilities = _remaining_heat_probabilities(
        bundle.get("late_warming_classifiers", bundle.get("continuation_classifiers", {})),
        row[extended_columns],
    )
    underprediction_probabilities = _underprediction_probabilities(
        bundle.get("underprediction_classifiers", {}),
        row[extended_columns],
    )
    conditional_remaining_heat = float(
        _clip_remaining(bundle.get("conditional_regressor", bundle["regressor"]).predict(row[columns]))[0]
    )
    two_stage_remaining_heat = float(
        _two_stage_remaining_prediction(
            remaining_probabilities,
            np.asarray([conditional_remaining_heat]),
        )[0]
    )
    remaining_heat = float(_predict_remaining_heat(bundle, row[columns])[0])
    observed_max_c = float(row["tmpc_max_to_cutoff"].iloc[0])
    last_observation_minute = int(row["last_observation_minute"].iloc[0])
    cutoff_minutes = _hhmm_to_minutes(cutoff_local)
    observation_lag_minutes = cutoff_minutes - last_observation_minute
    predicted_tmax_c = observed_max_c + remaining_heat
    interval = _prediction_interval(
        predicted_tmax_c,
        bundle["interval_calibration"],
        cutoff_local,
        observed_max_c,
    )

    result = {
        "station": config.station,
        "local_date": local_date,
        "cutoff_local": cutoff_local,
        "last_observation_local": f"{local_date} {_minutes_to_hhmm(last_observation_minute)}",
        "last_observation_lag_minutes": observation_lag_minutes,
        "data_fresh_enough": observation_lag_minutes <= 60,
        "observed_max_to_cutoff_c": observed_max_c,
        "last_temp_to_cutoff_c": float(row["tmpc_last_to_cutoff"].iloc[0]),
        "predicted_remaining_heat_c": remaining_heat,
        "prediction_method": bundle["metrics"].get("selected_prediction_method", "direct"),
        "direct_predicted_remaining_heat_c": direct_remaining_heat,
        "conditional_predicted_remaining_heat_if_continues_c": conditional_remaining_heat,
        "two_stage_predicted_remaining_heat_c": two_stage_remaining_heat,
        "predicted_tmax_c": predicted_tmax_c,
        "predicted_tmax_f": predicted_tmax_c * 9.0 / 5.0 + 32.0,
        **interval,
        "target_complete": bool(row.get("target_complete", pd.Series([0])).iloc[0]),
    }

    for threshold_text, probability_values in late_warming_probabilities.items():
        threshold = float(threshold_text)
        probability = float(probability_values[0])
        result[f"prob_remaining_heat_ge_{_remaining_heat_slug(threshold)}"] = probability
    for threshold_text, probability_values in underprediction_probabilities.items():
        threshold = float(threshold_text)
        probability = float(probability_values[0])
        result[f"prob_m0_underpredict_ge_{_remaining_heat_slug(threshold)}"] = probability
    if "0.5" in late_warming_probabilities:
        result["prob_tmax_already_reached"] = float(1.0 - late_warming_probabilities["0.5"][0])
    result["late_warming_risk"] = _late_warming_risk_label(
        float(late_warming_probabilities.get("2.0", np.asarray([0.0]))[0])
    )
    result.update(
        _warning_output(
            row.iloc[0],
            late_warming_probabilities,
            underprediction_probabilities,
        )
    )
    result["weather_context"] = _weather_context(row.iloc[0])
    result.update(_phase_prediction(bundle.get("phase_classifier"), row[extended_columns]))
    future_curve = _future_curve_output(
        bundle.get("curve_models", {}),
        row[extended_columns],
        local_date,
        cutoff_minutes,
    )
    result["future_curve"] = future_curve
    result["curve_predicted_tmax_c"] = _curve_prediction_tmax_value(observed_max_c, future_curve)
    result.update(_regime_break_output(row.iloc[0]))

    raw_probabilities = _raw_threshold_probabilities(
        bundle["classifiers"],
        row[columns],
    )
    probabilities = _monotonic_operational_threshold_probabilities(
        bundle["classifiers"],
        row[columns],
        row["tmpc_max_to_cutoff"],
    )
    result["raw_threshold_probabilities"] = {
        _threshold_slug(float(threshold_text)): float(values[0])
        for threshold_text, values in raw_probabilities.items()
    }
    result["monotonic_threshold_probabilities"] = {
        _threshold_slug(float(threshold_text)): float(values[0])
        for threshold_text, values in probabilities.items()
    }
    for threshold_text, probability_values in probabilities.items():
        threshold = float(threshold_text)
        probability = probability_values[0]
        result[f"prob_tmax_ge_{_threshold_slug(threshold)}"] = float(probability)

    update = _update_recommendation(
        cutoff_local,
        interval,
        bundle["update_policy"],
        row.iloc[0],
        probabilities,
    )
    result.update(update)

    if pd.notna(row[FINAL_TMAX_COLUMN].iloc[0]):
        result["actual_tmax_c"] = float(row[FINAL_TMAX_COLUMN].iloc[0])
        result["actual_remaining_heat_c"] = float(row[TARGET_COLUMN].iloc[0])
    return result


def format_heat_risk_explanation(prediction: dict) -> str:
    station = prediction["station"]
    local_date = prediction["local_date"]
    cutoff = prediction["cutoff_local"]
    observed_max = prediction.get("observed_max_to_cutoff_c")
    last_temp = prediction.get("last_temp_to_cutoff_c")
    remaining = prediction.get("predicted_remaining_heat_c")
    predicted_tmax = prediction.get("predicted_tmax_c")
    interval_low = prediction.get("prediction_interval_80_low_c")
    interval_high = prediction.get("prediction_interval_80_high_c")
    phase = prediction.get("thermal_phase", "unknown")
    late_risk = prediction.get("late_warming_risk", "unknown")
    warning = prediction.get("late_warming_warning", "unknown")
    update_next = prediction.get("next_update_local")
    update_recommended = prediction.get("recommend_update_next_cutoff")
    p_reached = prediction.get("prob_tmax_already_reached")
    p_ge_2 = prediction.get("prob_remaining_heat_ge_2_0")
    p_ge_3 = prediction.get("prob_remaining_heat_ge_3_0")
    p_ge_4 = prediction.get("prob_remaining_heat_ge_4_0")

    lines = [
        "",
        "=== Giải thích dễ đọc ===",
        f"Station {station}, ngày local {local_date}, dữ liệu tính đến {cutoff}.",
        (
            "Nhiệt độ cao nhất đã quan sát đến cutoff là "
            f"{_format_c(observed_max)}; nhiệt độ mới nhất là {_format_c(last_temp)}."
        ),
        (
            "Model dự báo từ sau cutoff đến cuối ngày còn có thể tăng thêm khoảng "
            f"{_format_c(remaining)}, nên Tmax dự báo là {_format_c(predicted_tmax)}."
        ),
        (
            "Khoảng bất định 80% cho Tmax nằm trong khoảng "
            f"{_format_c(interval_low)} đến {_format_c(interval_high)}."
        ),
        (
            "Trạng thái nhiệt hiện tại: "
            f"{_translate_phase(phase)}. Mức rủi ro tăng muộn: {_translate_risk(late_risk)}."
        ),
        f"Cảnh báo vận hành: {_translate_warning(warning)}.",
    ]
    weather_lines = _format_weather_context(prediction.get("weather_context"))
    if weather_lines:
        lines.append("Nhận xét thời tiết METAR:")
        lines.extend(f"- {line}" for line in weather_lines)
    if p_reached is not None:
        lines.append(f"Xác suất Tmax đã đạt đỉnh lúc này: {_format_percent(p_reached)}.")
    probabilities = []
    if p_ge_2 is not None:
        probabilities.append(f">=2C: {_format_percent(p_ge_2)}")
    if p_ge_3 is not None:
        probabilities.append(f">=3C: {_format_percent(p_ge_3)}")
    if p_ge_4 is not None:
        probabilities.append(f">=4C: {_format_percent(p_ge_4)}")
    if probabilities:
        lines.append("Xác suất còn tăng thêm sau cutoff: " + ", ".join(probabilities) + ".")

    reasons = prediction.get("warning_reasons") or []
    if reasons:
        lines.append("Lý do cần chú ý:")
        lines.extend(f"- {_translate_warning_reason(reason)}" for reason in reasons)

    if update_next:
        action = "nên cập nhật" if update_recommended else "không bắt buộc cập nhật"
        lines.append(f"Mốc tiếp theo: {update_next}; hệ thống đánh giá là {action}.")
    if prediction.get("recommended_action"):
        lines.append(f"Hành động gợi ý: {_translate_recommended_action(prediction['recommended_action'])}")
    if prediction.get("plot_path"):
        lines.append(f"Biểu đồ curve đã ghi tại: {prediction['plot_path']}")
    return "\n".join(lines)


def _format_c(value: object) -> str:
    if value is None:
        return "không có dữ liệu"
    try:
        if pd.isna(value):
            return "không có dữ liệu"
        return f"{float(value):.1f}C"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "không có dữ liệu"
    try:
        if pd.isna(value):
            return "không có dữ liệu"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _translate_phase(value: str) -> str:
    translations = {
        "pre_peak_ramp": "đang tăng trước đỉnh nhiệt",
        "peak_plateau": "đang ở vùng đỉnh/đi ngang",
        "post_peak_decline": "có khả năng đã qua đỉnh và đang giảm",
        "uncertain_transition": "chuyển pha chưa chắc chắn",
        "unknown": "không xác định",
    }
    return translations.get(value, value)


def _translate_risk(value: str) -> str:
    translations = {
        "low": "thấp",
        "moderate": "vừa phải",
        "elevated": "cao hơn bình thường",
        "high": "cao",
        "unknown": "không xác định",
    }
    return translations.get(value, value)


def _translate_warning(value: str) -> str:
    translations = {
        "low": "thấp, chưa có tín hiệu cảnh báo đặc biệt",
        "watch_false_plateau": "cần theo dõi false plateau",
        "elevated_late_warming_risk": "có rủi ro tăng nhiệt muộn",
        "high_late_warming_risk": "rủi ro tăng nhiệt muộn cao",
        "high_extreme_late_warming_risk": "rủi ro tăng nhiệt muộn rất cao",
        "unknown": "không xác định",
    }
    return translations.get(value, value)


def _translate_warning_reason(value: str) -> str:
    translations = {
        "cutoff before or near noon": "cutoff vẫn còn trước hoặc gần giữa trưa",
        "temperature flat for at least 90 minutes": "nhiệt độ đi ngang ít nhất 90 phút",
        "rain/low cloud/MVFR or low visibility recently": (
            "gần đây có mưa, mây thấp, điều kiện bay kém hoặc tầm nhìn thấp"
        ),
        "latest temperature equals observed max": "nhiệt độ mới nhất đang bằng mức cao nhất đã quan sát",
        "cutoff before typical monthly peak time": "cutoff vẫn trước giờ đỉnh nhiệt thường gặp của tháng",
        "classifier probability for remaining heat >= 2C is high": (
            "model đánh giá khả năng còn tăng thêm ít nhất 2C là cao"
        ),
        "classifier probability for remaining heat >= 3C is high": (
            "model đánh giá khả năng còn tăng thêm ít nhất 3C là cao"
        ),
        "classifier probability for remaining heat >= 4C is high": (
            "model đánh giá khả năng còn tăng thêm ít nhất 4C là cao"
        ),
        "point forecast may be too low": "dự báo Tmax đang có nguy cơ bị thấp",
    }
    return translations.get(value, value)


def _translate_recommended_action(value: str) -> str:
    translations = {
        "Do not treat point forecast as final; update at next cutoff.": (
            "Không nên xem dự báo điểm là kết luận cuối; nên cập nhật lại ở cutoff tiếp theo."
        ),
        "Point forecast can be used with normal interval uncertainty.": (
            "Có thể dùng dự báo điểm kèm khoảng bất định thông thường."
        ),
    }
    return translations.get(value, value)


def _weather_context(row: pd.Series) -> dict:
    context = {
        "rain_seen_last_2h": _truthy_feature(row, "rain_seen_last_2h"),
        "rain_seen_at_cutoff": _truthy_feature(row, "rain_seen_at_cutoff"),
        "low_cloud_seen_last_2h": _truthy_feature(row, "low_cloud_seen_last_2h"),
        "visibility_low_last_2h": _truthy_feature(row, "visibility_low_last_2h"),
        "mvfr_or_worse_last_2h": _truthy_feature(row, "mvfr_or_worse_last_2h"),
        "fog_observed_to_cutoff": _truthy_feature(row, "fog_observed_to_cutoff"),
        "fog_cleared_to_cutoff": _truthy_feature(row, "fog_cleared_to_cutoff"),
        "fog_developed_to_cutoff": _truthy_feature(row, "fog_developed_to_cutoff"),
        "weather_suppression_score": _optional_float(row.get("weather_suppression_score")),
        "visibility_min_last_2h_sm": _optional_float(row.get("visibility_min_last_2h")),
        "ceiling_min_last_2h_ft": _optional_float(row.get("ceiling_min_last_2h")),
        "lowest_ceiling_ft_to_cutoff": _optional_float(row.get("lowest_ceiling_ft_to_cutoff")),
        "cloud_clearing_to_cutoff": _optional_float(row.get("cloud_clearing_to_cutoff")),
        "cloud_increasing_to_cutoff": _optional_float(row.get("cloud_increasing_to_cutoff")),
        "max_cloud_cover_to_cutoff": _optional_float(row.get("max_cloud_cover_to_cutoff")),
        "last_cloud_cover_to_cutoff": _optional_float(row.get("last_cloud_cover_to_cutoff")),
    }
    context["summary"] = _weather_context_lines(context)
    return context


def _format_weather_context(context: object) -> list[str]:
    if not isinstance(context, dict):
        return []
    summary = context.get("summary")
    if isinstance(summary, list):
        return [str(line) for line in summary if line]
    return _weather_context_lines(context)


def _weather_context_lines(context: dict) -> list[str]:
    lines = []
    if context.get("rain_seen_last_2h"):
        when = "ngay tại cutoff" if context.get("rain_seen_at_cutoff") else "trong 2 giờ gần đây"
        lines.append(f"Có dấu hiệu mưa/giáng thủy {when}; nắng lên sau cutoff có thể bị trì hoãn.")
    if context.get("fog_observed_to_cutoff"):
        if context.get("fog_cleared_to_cutoff"):
            lines.append("Có sương mù/mù trước cutoff nhưng tín hiệu đã cải thiện dần.")
        elif context.get("fog_developed_to_cutoff"):
            lines.append("Sương mù/mù xuất hiện về gần cutoff, có thể kìm tăng nhiệt buổi sáng.")
        else:
            lines.append("Có sương mù/mù trong cửa sổ quan sát trước cutoff.")
    if context.get("low_cloud_seen_last_2h"):
        ceiling = context.get("ceiling_min_last_2h_ft")
        if ceiling is not None:
            lines.append(f"Có mây thấp trong 2 giờ gần đây; trần mây thấp nhất khoảng {ceiling:.0f} ft.")
        else:
            lines.append("Có mây thấp trong 2 giờ gần đây.")
    if context.get("visibility_low_last_2h"):
        visibility = context.get("visibility_min_last_2h_sm")
        if visibility is not None:
            lines.append(f"Tầm nhìn từng giảm thấp, tối thiểu khoảng {visibility:.1f} statute mile.")
        else:
            lines.append("Tầm nhìn từng giảm thấp trong 2 giờ gần đây.")
    if context.get("mvfr_or_worse_last_2h"):
        lines.append("Điều kiện bay có lúc ở mức MVFR hoặc xấu hơn, thường là tín hiệu hạn chế bức xạ mặt trời.")

    clearing = context.get("cloud_clearing_to_cutoff")
    increasing = context.get("cloud_increasing_to_cutoff")
    if clearing is not None and clearing >= 2:
        lines.append("Mây đang có xu hướng tan bớt trước cutoff, có thể mở cửa cho tăng nhiệt muộn.")
    elif increasing is not None and increasing >= 2:
        lines.append("Mây đang tăng lên trước cutoff, có thể làm chậm nhịp tăng nhiệt.")

    score = context.get("weather_suppression_score")
    if score is not None and score >= 1.0:
        lines.append(f"Điểm ức chế thời tiết là {score:.1f}, nên cần thận trọng với kịch bản false plateau.")
    if not lines:
        lines.append("Không thấy tín hiệu mưa, mây thấp, tầm nhìn thấp hoặc điều kiện bay xấu đáng kể trong METAR gần cutoff.")
    return lines


def _truthy_feature(row: pd.Series, column: str) -> bool:
    value = row.get(column)
    if value is None or pd.isna(value):
        return False
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return bool(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_prediction_curve(
    config: ProjectConfig,
    prediction: dict,
    output_path: str | Path,
) -> Path:
    observations = load_observations(config.input_csv, config)
    local_date = prediction["local_date"]
    cutoff_minutes = _hhmm_to_minutes(prediction["cutoff_local"])
    day = observations.copy()
    day["valid_local"] = pd.to_datetime(day["valid_local"])
    day["local_date"] = day["valid_local"].dt.date.astype(str)
    day["local_minutes"] = day["valid_local"].dt.hour * 60 + day["valid_local"].dt.minute
    day["tmpc"] = (pd.to_numeric(day["tmpf"], errors="coerce") - 32.0) * (5.0 / 9.0)
    day = day[(day["local_date"] == local_date) & day["tmpc"].notna()].sort_values("local_minutes")

    observed_to_cutoff = day[day["local_minutes"] <= cutoff_minutes]
    observed_after_cutoff = day[day["local_minutes"] > cutoff_minutes]
    future_curve = _future_curve_to_frame(prediction.get("future_curve", {}))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    if not observed_to_cutoff.empty:
        ax.plot(
            observed_to_cutoff["local_minutes"],
            observed_to_cutoff["tmpc"],
            marker="o",
            linewidth=2.4,
            color="#1f77b4",
            label="Observed to cutoff",
        )
    if not observed_after_cutoff.empty:
        ax.plot(
            observed_after_cutoff["local_minutes"],
            observed_after_cutoff["tmpc"],
            marker="o",
            linewidth=1.8,
            linestyle="--",
            color="#9aa0a6",
            label="Observed after cutoff",
        )
    if not future_curve.empty:
        ax.plot(
            future_curve["local_minutes"],
            future_curve["tmpc"],
            marker="o",
            linewidth=2.6,
            color="#f28e2b",
            label="Forecast curve",
        )

    ax.axvline(cutoff_minutes, color="#d1495b", linewidth=1.8, linestyle="--", label="Cutoff")
    ax.axhline(
        prediction["predicted_tmax_c"],
        color="#2a9d8f",
        linewidth=1.4,
        linestyle=":",
        label="M0 predicted Tmax",
    )
    if "curve_predicted_tmax_c" in prediction and prediction["curve_predicted_tmax_c"] is not None:
        ax.axhline(
            prediction["curve_predicted_tmax_c"],
            color="#f28e2b",
            linewidth=1.2,
            linestyle=":",
            label="Curve predicted Tmax",
        )

    cutoff_temp = prediction.get("last_temp_to_cutoff_c")
    if cutoff_temp is not None:
        ax.scatter(
            [cutoff_minutes],
            [cutoff_temp],
            s=90,
            color="#d1495b",
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            f"Cutoff {prediction['cutoff_local']}\n{cutoff_temp:.1f}C",
            xy=(cutoff_minutes, cutoff_temp),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=10,
            color="#6b1f2b",
            arrowprops={"arrowstyle": "->", "color": "#d1495b", "linewidth": 1.0},
        )

    x_min = min(360, int(day["local_minutes"].min()) if not day.empty else cutoff_minutes - 180)
    x_max = max(1080, int(day["local_minutes"].max()) if not day.empty else cutoff_minutes + 180)
    if not future_curve.empty:
        x_max = max(x_max, int(future_curve["local_minutes"].max()) + 30)
    ax.set_xlim(x_min, x_max)
    tick_start = (x_min // 60) * 60
    ticks = list(range(tick_start, x_max + 1, 60))
    ax.set_xticks(ticks, [_minutes_to_hhmm(tick) for tick in ticks])
    ax.set_ylabel("Temperature (C)")
    ax.set_xlabel("Local time")
    ax.set_title(
        f"{prediction['station']} {local_date} Temperature Curve "
        f"({prediction['thermal_phase']}, late warming: {prediction['late_warming_risk']})"
    )
    subtitle = (
        f"Observed max {prediction['observed_max_to_cutoff_c']:.1f}C | "
        f"M0 Tmax {prediction['predicted_tmax_c']:.1f}C | "
        f"80% interval {prediction['prediction_interval_80_low_c']:.1f}-"
        f"{prediction['prediction_interval_80_high_c']:.1f}C"
    )
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="best")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _future_curve_to_frame(future_curve: dict) -> pd.DataFrame:
    rows = []
    for timestamp, value in future_curve.items():
        dt = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({"local_minutes": int(dt.hour * 60 + dt.minute), "tmpc": float(value)})
    return pd.DataFrame(rows).sort_values("local_minutes") if rows else pd.DataFrame()


def _prediction_row(
    config: ProjectConfig,
    local_date: str,
    cutoff_local: str,
    dataset_path: str | Path | None,
) -> pd.DataFrame:
    if dataset_path is not None:
        dataset = load_heat_risk_table(dataset_path)
        match = dataset[
            (dataset["local_date"] == local_date) & (dataset["cutoff_local"] == cutoff_local)
        ]
        if not match.empty:
            return match.iloc[[0]]

    observations = load_observations(config.input_csv, config)
    row = _make_single_cutoff_dataset(observations, config, cutoff_local)
    match = row[row["local_date"] == local_date]
    if match.empty:
        raise ValueError(f"No feature row found for {local_date} at cutoff {cutoff_local}.")
    return match.iloc[[0]]


def _artifact_stem(config: ProjectConfig) -> str:
    stem = Path(config.heat_risk_metrics_path).stem
    return stem.removesuffix("_metrics")


def _regression_pipeline(config: ProjectConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.03,
                    max_iter=600,
                    l2_regularization=0.1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _classifier_pipeline(config: ProjectConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=400,
                    l2_regularization=0.1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _threshold_probability(
    classifier: Pipeline,
    features: pd.DataFrame,
    observed_max_c: pd.Series,
    threshold: float,
) -> np.ndarray:
    probability = classifier.predict_proba(features)[:, 1]
    return np.where(observed_max_c.to_numpy() >= threshold, 1.0, probability)


def _raw_threshold_probabilities(
    classifiers: dict,
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        threshold_text: classifier.predict_proba(features)[:, 1]
        for threshold_text, classifier in sorted(classifiers.items(), key=lambda item: float(item[0]))
    }


def _monotonic_operational_threshold_probabilities(
    classifiers: dict,
    features: pd.DataFrame,
    observed_max_c: pd.Series,
) -> dict[str, np.ndarray]:
    probabilities = {}
    previous = None
    for threshold_text, classifier in sorted(classifiers.items(), key=lambda item: float(item[0])):
        threshold = float(threshold_text)
        probability = _threshold_probability(classifier, features, observed_max_c, threshold)
        if previous is not None:
            probability = np.minimum(probability, previous)
        probabilities[threshold_text] = probability
        previous = probability
    return probabilities


def _threshold_probabilities(
    classifiers: dict,
    features: pd.DataFrame,
    observed_max_c: pd.Series,
) -> dict[str, np.ndarray]:
    return _monotonic_operational_threshold_probabilities(classifiers, features, observed_max_c)


def _threshold_metrics(
    train_label: pd.Series,
    test_label: pd.Series,
    raw_probability: np.ndarray,
    operational_probability: np.ndarray,
    observed_max_c: pd.Series,
    threshold: float,
) -> dict:
    climatology_probability = float(train_label.mean())
    climatology = np.full(len(test_label), climatology_probability)
    climatology_brier = float(brier_score_loss(test_label, climatology))
    operational_brier = float(brier_score_loss(test_label, operational_probability))
    output = {
        "brier": operational_brier,
        "climatology_brier": climatology_brier,
        "brier_skill_score": _brier_skill_score(operational_brier, climatology_brier),
        "event_rate": float(test_label.mean()),
        "n": int(len(test_label)),
    }
    if test_label.nunique() > 1:
        output["roc_auc"] = float(roc_auc_score(test_label, operational_probability))
    else:
        output["roc_auc"] = None

    not_crossed = observed_max_c.to_numpy() < threshold
    not_crossed_label = test_label.to_numpy()[not_crossed]
    not_crossed_probability = raw_probability[not_crossed]
    if len(not_crossed_label) > 0:
        not_crossed_climatology = np.full(len(not_crossed_label), climatology_probability)
        not_crossed_climatology_brier = float(
            brier_score_loss(not_crossed_label, not_crossed_climatology)
        )
        not_crossed_brier = float(brier_score_loss(not_crossed_label, not_crossed_probability))
        output["not_yet_crossed"] = {
            "n": int(len(not_crossed_label)),
            "event_rate": float(np.mean(not_crossed_label)),
            "brier": not_crossed_brier,
            "climatology_brier": not_crossed_climatology_brier,
            "brier_skill_score": _brier_skill_score(
                not_crossed_brier,
                not_crossed_climatology_brier,
            ),
            "roc_auc": (
                float(roc_auc_score(not_crossed_label, not_crossed_probability))
                if len(np.unique(not_crossed_label)) > 1
                else None
            ),
        }
    else:
        output["not_yet_crossed"] = {"n": 0}
    return output


def _remaining_heat_probabilities(classifiers: dict, features: pd.DataFrame) -> dict[str, np.ndarray]:
    probabilities = {}
    previous = None
    for threshold_text, classifier in sorted(classifiers.items(), key=lambda item: float(item[0])):
        probability = classifier.predict_proba(features)[:, 1]
        if previous is not None:
            probability = np.minimum(probability, previous)
        probabilities[threshold_text] = probability
        previous = probability
    return probabilities


def _remaining_heat_probability_metrics(
    train_label: pd.Series,
    test_label: pd.Series,
    probability: np.ndarray,
) -> dict:
    climatology_probability = float(train_label.mean())
    climatology = np.full(len(test_label), climatology_probability)
    brier = float(brier_score_loss(test_label, probability))
    climatology_brier = float(brier_score_loss(test_label, climatology))
    return {
        "n": int(len(test_label)),
        "event_rate": float(test_label.mean()),
        "brier": brier,
        "climatology_brier": climatology_brier,
        "brier_skill_score": _brier_skill_score(brier, climatology_brier),
        "roc_auc": (
            float(roc_auc_score(test_label, probability))
            if test_label.nunique() > 1
            else None
        ),
    }


def _phase_metrics(
    classifier: Pipeline | None,
    test: pd.DataFrame,
    columns: list[str],
) -> dict:
    if classifier is None:
        return {"available": False}
    actual = test["thermal_phase_target"]
    predicted = pd.Series(classifier.predict(test[columns]), index=test.index)
    labels = [label for label in THERMAL_PHASE_LABELS if label in set(actual) | set(predicted)]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    return {
        "available": True,
        "accuracy": float((actual == predicted).mean()),
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _curve_predictions(
    curve_models: dict,
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        horizon_text: model.predict(features)
        for horizon_text, model in sorted(curve_models.items(), key=lambda item: int(item[0]))
    }


def _curve_tmax_prediction(
    observed_max_c: pd.Series,
    curve_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    if not curve_predictions:
        return np.full(len(observed_max_c), np.nan)
    stacked = np.column_stack([observed_max_c.to_numpy(dtype=float), *curve_predictions.values()])
    return np.nanmax(stacked, axis=1)


def _curve_metrics(
    test: pd.DataFrame,
    curve_predictions: dict[str, np.ndarray],
    curve_tmax_prediction: np.ndarray,
) -> dict:
    horizon_metrics = {}
    for horizon_text, prediction in curve_predictions.items():
        target = _future_target_column(int(horizon_text))
        mask = test[target].notna()
        if not mask.any():
            continue
        horizon_metrics[horizon_text] = {
            "target": target,
            "mae_c": float(mean_absolute_error(test.loc[mask, target], prediction[mask.to_numpy()])),
            "n": int(mask.sum()),
        }
    tmax_mask = np.isfinite(curve_tmax_prediction) & test[FINAL_TMAX_COLUMN].notna().to_numpy()
    return {
        "horizons": horizon_metrics,
        "curve_predicted_tmax_mae_c": (
            float(mean_absolute_error(test.loc[tmax_mask, FINAL_TMAX_COLUMN], curve_tmax_prediction[tmax_mask]))
            if tmax_mask.any()
            else None
        ),
        "n_curve_tmax": int(tmax_mask.sum()),
    }


def _late_warming_event_metrics(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
) -> dict:
    output = {}
    for threshold_text in ("2.0", "3.0", "4.0"):
        probability = probabilities.get(threshold_text)
        if probability is None:
            continue
        actual = (test[TARGET_COLUMN] >= float(threshold_text)).astype(int)
        predicted = (probability >= 0.30).astype(int)
        output[threshold_text] = {
            "event_rate": float(actual.mean()),
            "recall_at_30pct": (
                float(recall_score(actual, predicted, zero_division=0))
                if actual.nunique() > 1
                else None
            ),
            "precision_at_30pct": (
                float(precision_score(actual, predicted, zero_division=0))
                if predicted.sum() > 0
                else None
            ),
            "false_alarm_rate_at_30pct": _false_alarm_rate(actual.to_numpy(), predicted),
        }
    return output


def _fit_underprediction_classifiers(
    config: ProjectConfig,
    train: pd.DataFrame,
    columns: list[str],
    train_tmax_prediction: np.ndarray,
) -> dict:
    classifiers = {}
    residual = train[FINAL_TMAX_COLUMN].to_numpy() - train_tmax_prediction
    for threshold in UNDERPREDICTION_THRESHOLDS_C:
        label = (residual >= threshold).astype(int)
        if len(np.unique(label)) < 2:
            continue
        classifier = _classifier_pipeline(config)
        classifier.fit(train[columns], label)
        classifiers[str(threshold)] = classifier
    return classifiers


def _underprediction_probabilities(classifiers: dict, features: pd.DataFrame) -> dict[str, np.ndarray]:
    probabilities = {}
    previous = None
    for threshold_text, classifier in sorted(classifiers.items(), key=lambda item: float(item[0])):
        probability = classifier.predict_proba(features)[:, 1]
        if previous is not None:
            probability = np.minimum(probability, previous)
        probabilities[threshold_text] = probability
        previous = probability
    return probabilities


def _underprediction_event_metrics(
    test: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    tmax_prediction: np.ndarray,
) -> dict:
    output = {}
    residual = test[FINAL_TMAX_COLUMN].to_numpy() - tmax_prediction
    for threshold_text in ("1.5", "2.0"):
        probability = probabilities.get(threshold_text)
        if probability is None:
            continue
        actual = (residual >= float(threshold_text)).astype(int)
        predicted = (probability >= 0.30).astype(int)
        output[threshold_text] = {
            "event_rate": float(actual.mean()),
            "recall_at_30pct": (
                float(recall_score(actual, predicted, zero_division=0))
                if len(np.unique(actual)) > 1
                else None
            ),
            "precision_at_30pct": (
                float(precision_score(actual, predicted, zero_division=0))
                if predicted.sum() > 0
                else None
            ),
            "false_alarm_rate_at_30pct": _false_alarm_rate(actual, predicted),
        }
    return output


def _false_alarm_rate(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    predicted_positive = predicted == 1
    if not predicted_positive.any():
        return None
    return float((actual[predicted_positive] == 0).mean())


def _two_stage_remaining_prediction(
    continuation_probabilities: dict[str, np.ndarray],
    conditional_remaining_prediction: np.ndarray,
) -> np.ndarray:
    heat_continues_probability = continuation_probabilities.get(str(CONTINUING_HEAT_THRESHOLD_C))
    if heat_continues_probability is None:
        heat_continues_probability = np.ones(len(conditional_remaining_prediction))
    return _clip_remaining(heat_continues_probability * conditional_remaining_prediction)


def _predict_remaining_heat(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    method = bundle["metrics"].get("selected_prediction_method", "direct")
    if method == "two_stage" and "conditional_regressor" in bundle:
        probabilities = _remaining_heat_probabilities(
            bundle.get("continuation_classifiers", {}),
            features,
        )
        conditional_prediction = _clip_remaining(bundle["conditional_regressor"].predict(features))
        return _two_stage_remaining_prediction(probabilities, conditional_prediction)
    return _clip_remaining(bundle["regressor"].predict(features))


def _build_interval_calibration(test: pd.DataFrame, residual: np.ndarray) -> dict:
    frame = test[["cutoff_local"]].copy()
    frame["residual"] = residual
    by_cutoff = {}
    for cutoff, group in frame.groupby("cutoff_local"):
        if len(group) < 30:
            continue
        by_cutoff[str(cutoff)] = _residual_quantiles(group["residual"].to_numpy())
    return {
        "method": "conformal_by_cutoff",
        "overall": _residual_quantiles(residual),
        "by_cutoff": by_cutoff,
    }


def _residual_quantiles(residual: np.ndarray) -> dict:
    return {
        "residual_p10_c": float(np.quantile(residual, 0.10)),
        "residual_p50_c": float(np.quantile(residual, 0.50)),
        "residual_p90_c": float(np.quantile(residual, 0.90)),
    }


def _prediction_interval(
    prediction_c: float,
    calibration: dict,
    cutoff_local: str,
    observed_max_c: float,
) -> dict:
    quantiles = calibration.get("by_cutoff", {}).get(cutoff_local, calibration.get("overall", calibration))
    p10 = prediction_c + quantiles["residual_p10_c"]
    p50 = prediction_c + quantiles["residual_p50_c"]
    p90 = prediction_c + quantiles["residual_p90_c"]
    p10_practical = max(observed_max_c, p10)
    return {
        "interval_method": calibration.get("method", "global_residual"),
        "prediction_p10_c": p10,
        "prediction_p50_c": p50,
        "prediction_p90_c": p90,
        "prediction_interval_80_low_raw_c": p10,
        "prediction_interval_80_low_practical_c": p10_practical,
        "prediction_interval_80_low_c": p10_practical,
        "prediction_interval_80_high_c": p90,
    }


def _build_update_policy(test: pd.DataFrame, prediction: np.ndarray) -> dict:
    frame = test[["local_date", "cutoff_local", "cutoff_minutes", FINAL_TMAX_COLUMN]].copy()
    frame["prediction"] = prediction
    frame["abs_error"] = (frame["prediction"] - frame[FINAL_TMAX_COLUMN]).abs()
    records = []
    for _, group in frame.sort_values("cutoff_minutes").groupby("local_date"):
        current = group.iloc[:-1].copy()
        next_rows = group.iloc[1:].copy()
        if current.empty:
            continue
        current["next_cutoff_local"] = next_rows["cutoff_local"].to_numpy()
        current["next_abs_error"] = next_rows["abs_error"].to_numpy()
        current["improvement_c"] = current["abs_error"] - current["next_abs_error"]
        records.append(current)
    if not records:
        return {}
    transitions = pd.concat(records, ignore_index=True)
    output = {}
    for cutoff, group in transitions.groupby("cutoff_local"):
        output[str(cutoff)] = {
            "next_cutoff_local": str(group["next_cutoff_local"].mode().iloc[0]),
            "mean_abs_error_improvement_c": float(group["improvement_c"].mean()),
            "median_abs_error_improvement_c": float(group["improvement_c"].median()),
            "update_helped_rate": float((group["improvement_c"] > 0.0).mean()),
            "n": int(len(group)),
        }
    return output


def _update_recommendation(
    cutoff_local: str,
    interval: dict,
    update_policy: dict,
    row: pd.Series | None = None,
    threshold_probabilities: dict[str, np.ndarray] | None = None,
) -> dict:
    policy = update_policy.get(cutoff_local)
    if policy is None:
        policy = _nearest_update_policy(cutoff_local, update_policy)
    interval_width = interval["prediction_interval_80_high_c"] - interval["prediction_interval_80_low_c"]
    if policy is None:
        return {
            "next_update_local": None,
            "recommend_update_next_cutoff": False,
            "update_reason": "No historical next-cutoff policy for this cutoff.",
        }

    expected_improvement = policy["median_abs_error_improvement_c"]
    last_temp_is_observed_max = _last_temp_is_observed_max(row)
    near_sensitive_threshold = _near_sensitive_threshold(threshold_probabilities)
    worth_update = (
        expected_improvement >= UPDATE_MIN_IMPROVEMENT_C
        or (interval_width >= UPDATE_MIN_INTERVAL_WIDTH_C and last_temp_is_observed_max)
        or near_sensitive_threshold
    )
    reason = (
        f"historical median improvement {expected_improvement:.2f}C; "
        f"80% interval width {interval_width:.2f}C; "
        f"last temp equals observed max: {last_temp_is_observed_max}; "
        f"near sensitive threshold: {near_sensitive_threshold}"
    )
    return {
        "next_update_local": policy["next_cutoff_local"],
        "recommend_update_next_cutoff": bool(worth_update),
        "update_reason": reason,
    }


def _last_temp_is_observed_max(row: pd.Series | None) -> bool:
    if row is None:
        return False
    if "tmpc_last_to_cutoff" not in row or "tmpc_max_to_cutoff" not in row:
        return False
    if pd.isna(row["tmpc_last_to_cutoff"]) or pd.isna(row["tmpc_max_to_cutoff"]):
        return False
    return abs(float(row["tmpc_last_to_cutoff"]) - float(row["tmpc_max_to_cutoff"])) < 0.05


def _near_sensitive_threshold(threshold_probabilities: dict[str, np.ndarray] | None) -> bool:
    if threshold_probabilities is None:
        return False
    lower, upper = SENSITIVE_PROBABILITY_RANGE
    for probability_values in threshold_probabilities.values():
        probability = float(probability_values[0])
        if lower <= probability <= upper:
            return True
    return False


def _nearest_update_policy(cutoff_local: str, update_policy: dict) -> dict | None:
    if not update_policy:
        return None
    cutoff_minutes = _hhmm_to_minutes(cutoff_local)
    candidates = []
    for key, value in update_policy.items():
        next_cutoff = value.get("next_cutoff_local")
        if next_cutoff is None:
            continue
        next_minutes = _hhmm_to_minutes(next_cutoff)
        if next_minutes > cutoff_minutes:
            candidates.append((_hhmm_to_minutes(key), next_minutes, value))
    if not candidates:
        return None

    earlier = [candidate for candidate in candidates if candidate[0] <= cutoff_minutes]
    if earlier:
        return max(earlier, key=lambda candidate: candidate[0])[2]
    return min(candidates, key=lambda candidate: candidate[1])[2]


def _metrics_by_cutoff(test: pd.DataFrame) -> list[dict]:
    output = []
    for cutoff, group in test.groupby("cutoff_local"):
        output.append(
            {
                "cutoff_local": str(cutoff),
                "n": int(len(group)),
                "tmax_mae_c": float(mean_absolute_error(group[FINAL_TMAX_COLUMN], group["predicted_tmax_c"])),
                "remaining_heat_mae_c": float(
                    mean_absolute_error(group[TARGET_COLUMN], group["predicted_remaining_heat_c"])
                ),
                "baseline_observed_max_mae_c": float(
                    mean_absolute_error(group[FINAL_TMAX_COLUMN], group["tmpc_max_to_cutoff"])
                ),
                "bias_c": float(np.mean(group["predicted_tmax_c"] - group[FINAL_TMAX_COLUMN])),
            }
        )
    return sorted(output, key=lambda row: _hhmm_to_minutes(row["cutoff_local"]))


def _top_heat_risk_errors(test: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    columns = [
        "local_date",
        "cutoff_local",
        "tmpc_last_to_cutoff",
        "tmpc_max_to_cutoff",
        "predicted_remaining_heat_c",
        "predicted_tmax_c",
        FINAL_TMAX_COLUMN,
        "error_c",
        "abs_error_c",
        "drct_last_to_cutoff",
        "sknt_last_to_cutoff",
        "max_cloud_cover_to_cutoff",
        "precip_observed_to_cutoff",
        "fog_observed_to_cutoff",
    ]
    available = [column for column in columns if column in test.columns]
    return test[available].sort_values("abs_error_c", ascending=False).head(n)


def _top_error_days(test: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    rows = []
    for local_date, group in test.groupby("local_date"):
        worst = group.sort_values("abs_error_c", ascending=False).iloc[0]
        rows.append(
            {
                "local_date": local_date,
                "max_abs_error_c": float(group["abs_error_c"].max()),
                "mean_abs_error_c": float(group["abs_error_c"].mean()),
                "worst_cutoff_local": worst["cutoff_local"],
                "actual_tmax_c": float(worst[FINAL_TMAX_COLUMN]),
                "morning_observed_max_c": float(group.sort_values("cutoff_minutes").iloc[0]["tmpc_max_to_cutoff"]),
                "worst_predicted_tmax_c": float(worst["predicted_tmax_c"]),
                "worst_error_c": float(worst["error_c"]),
                "precip_observed_any_cutoff": int(group.get("precip_observed_to_cutoff", pd.Series([0])).max()),
                "fog_observed_any_cutoff": int(group.get("fog_observed_to_cutoff", pd.Series([0])).max()),
                "fog_cleared_any_cutoff": int(group.get("fog_cleared_to_cutoff", pd.Series([0])).max()),
                "cloud_clearing_max": float(group.get("cloud_clearing_to_cutoff", pd.Series([np.nan])).max()),
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_error_c", ascending=False).head(n)


def _interval_coverage(
    test: pd.DataFrame,
    prediction: np.ndarray,
    calibration: dict,
) -> dict:
    rows = []
    for index, row in test.reset_index(drop=True).iterrows():
        interval = _prediction_interval(
            float(prediction[index]),
            calibration,
            row["cutoff_local"],
            float(row["tmpc_max_to_cutoff"]),
        )
        actual = float(row[FINAL_TMAX_COLUMN])
        rows.append(
            {
                "cutoff_local": row["cutoff_local"],
                "covered": (
                    interval["prediction_interval_80_low_c"]
                    <= actual
                    <= interval["prediction_interval_80_high_c"]
                ),
                "width_c": interval["prediction_interval_80_high_c"]
                - interval["prediction_interval_80_low_c"],
            }
        )
    frame = pd.DataFrame(rows)
    by_cutoff = []
    for cutoff, group in frame.groupby("cutoff_local"):
        by_cutoff.append(
            {
                "cutoff_local": cutoff,
                "coverage": float(group["covered"].mean()),
                "mean_width_c": float(group["width_c"].mean()),
                "n": int(len(group)),
            }
        )
    return {
        "overall": {
            "coverage": float(frame["covered"].mean()),
            "mean_width_c": float(frame["width_c"].mean()),
            "n": int(len(frame)),
        },
        "by_cutoff": sorted(by_cutoff, key=lambda row: _hhmm_to_minutes(row["cutoff_local"])),
    }


def _model_comparison(test: pd.DataFrame) -> dict:
    output = {
        "m0_heat_risk": {
            "tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], test["predicted_tmax_c"])),
        },
        "m1_phase_features": {
            "tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], test["m1_predicted_tmax_c"])),
        },
    }
    mask = test["curve_predicted_tmax_c"].notna()
    output["m2_future_curve"] = {
        "tmax_mae_c": (
            float(mean_absolute_error(test.loc[mask, FINAL_TMAX_COLUMN], test.loc[mask, "curve_predicted_tmax_c"]))
            if mask.any()
            else None
        ),
        "n": int(mask.sum()),
    }
    return output


def _top_error_reduction(test: pd.DataFrame, n: int = 20) -> dict:
    m0 = test.sort_values("abs_error_c", ascending=False).head(n)
    curve = test.dropna(subset=["curve_abs_error_c"]).sort_values(
        "curve_abs_error_c",
        ascending=False,
    ).head(n)
    return {
        "m0_top_abs_error_mean_c": float(m0["abs_error_c"].mean()) if not m0.empty else None,
        "curve_top_abs_error_mean_c": (
            float(curve["curve_abs_error_c"].mean()) if not curve.empty else None
        ),
        "n": int(min(len(m0), len(curve))),
    }


def _phase_prediction(classifier: Pipeline | None, features: pd.DataFrame) -> dict:
    if classifier is None:
        return {
            "thermal_phase": "unknown",
            "prob_pre_peak_ramp": None,
            "prob_peak_plateau": None,
            "prob_post_peak_decline": None,
        }
    probabilities = classifier.predict_proba(features)[0]
    classes = list(classifier.classes_)
    probability_by_class = {
        str(label): float(probabilities[index])
        for index, label in enumerate(classes)
    }
    phase = str(classes[int(np.argmax(probabilities))])
    return {
        "thermal_phase": phase,
        "prob_pre_peak_ramp": probability_by_class.get("pre_peak_ramp", 0.0),
        "prob_peak_plateau": probability_by_class.get("peak_plateau", 0.0),
        "prob_post_peak_decline": probability_by_class.get("post_peak_decline", 0.0),
        "prob_uncertain_transition": probability_by_class.get("uncertain_transition", 0.0),
    }


def _future_curve_output(
    curve_models: dict,
    features: pd.DataFrame,
    local_date: str,
    cutoff_minutes: int,
) -> dict:
    output = {}
    for horizon_text, model in sorted(curve_models.items(), key=lambda item: int(item[0])):
        target_minutes = cutoff_minutes + int(horizon_text)
        output[f"{local_date} {_minutes_to_hhmm(target_minutes)}"] = float(model.predict(features)[0])
    return output


def _curve_prediction_tmax_value(observed_max_c: float, future_curve: dict) -> float | None:
    if not future_curve:
        return None
    return float(max([observed_max_c, *future_curve.values()]))


def _late_warming_risk_label(probability_ge_2c: float) -> str:
    if probability_ge_2c < 0.10:
        return "low"
    if probability_ge_2c < 0.30:
        return "moderate"
    if probability_ge_2c < 0.50:
        return "elevated"
    return "high"


def _warning_output(
    row: pd.Series,
    late_warming_probabilities: dict[str, np.ndarray],
    underprediction_probabilities: dict[str, np.ndarray],
) -> dict:
    probability_ge_2c = float(late_warming_probabilities.get("2.0", np.asarray([0.0]))[0])
    probability_ge_3c = float(late_warming_probabilities.get("3.0", np.asarray([0.0]))[0])
    probability_ge_4c = float(late_warming_probabilities.get("4.0", np.asarray([0.0]))[0])
    probability_under_1_5c = float(
        underprediction_probabilities.get("1.5", np.asarray([0.0]))[0]
    )
    probability_under_2c = float(
        underprediction_probabilities.get("2.0", np.asarray([0.0]))[0]
    )
    rule = _false_plateau_rule(row)

    if probability_ge_4c >= 0.15:
        warning = "high_extreme_late_warming_risk"
        warning_type = "classifier_remaining_heat_ge_4c"
    elif probability_ge_3c >= 0.30:
        warning = "high_late_warming_risk"
        warning_type = "classifier_remaining_heat_ge_3c"
    elif probability_ge_2c >= 0.50:
        warning = "elevated_late_warming_risk"
        warning_type = "classifier_remaining_heat_ge_2c"
    elif rule["suppressed_late_warming_warning"]:
        warning = "watch_false_plateau"
        warning_type = rule["warning_type"]
    else:
        warning = "low"
        warning_type = "none"

    forecast_underprediction_warning = (
        probability_under_1_5c >= 0.30
        or probability_under_2c >= 0.20
        or probability_ge_4c >= 0.15
        or (rule["suppressed_late_warming_warning"] and probability_ge_2c >= 0.50)
    )
    reasons = list(rule["warning_reasons"])
    if probability_ge_2c >= 0.50:
        reasons.append("classifier probability for remaining heat >= 2C is high")
    if probability_ge_3c >= 0.30:
        reasons.append("classifier probability for remaining heat >= 3C is high")
    if probability_ge_4c >= 0.15:
        reasons.append("classifier probability for remaining heat >= 4C is high")
    if forecast_underprediction_warning:
        reasons.append("point forecast may be too low")

    recommended_action = (
        "Do not treat point forecast as final; update at next cutoff."
        if warning != "low" or forecast_underprediction_warning
        else "Point forecast can be used with normal interval uncertainty."
    )
    return {
        "suppressed_late_warming_warning": bool(rule["suppressed_late_warming_warning"]),
        "late_warming_warning": warning,
        "warning_type": warning_type,
        "forecast_underprediction_warning": bool(forecast_underprediction_warning),
        "warning_reasons": reasons,
        "recommended_action": recommended_action,
        "false_plateau_score": rule["false_plateau_score"],
    }


def _false_plateau_rule(row: pd.Series) -> dict:
    cutoff_minutes = float(row.get("cutoff_minutes", row.get("last_observation_minute", 9999)) or 9999)
    flat_duration = float(row.get("temp_flat_duration_last_2h", 0.0) or 0.0)
    temp_range = float(row.get("temp_range_last_2h", 999.0) or 999.0)
    suppression_score = float(row.get("weather_suppression_score", 0.0) or 0.0)
    latest_is_max = int(row.get("last_temp_equals_observed_max", 0) or 0) == 1
    before_peak = int(row.get("cutoff_before_typical_peak", 0) or 0) == 1
    score = 0.0
    reasons = []
    if cutoff_minutes <= 12 * 60:
        score += 1.0
        reasons.append("cutoff before or near noon")
    if flat_duration >= 90 or temp_range <= 1.0:
        score += 1.0
        reasons.append("temperature flat for at least 90 minutes")
    if suppression_score >= 1.0:
        score += 1.0
        reasons.append("rain/low cloud/MVFR or low visibility recently")
    if latest_is_max:
        score += 0.5
        reasons.append("latest temperature equals observed max")
    if before_peak:
        score += 0.5
        reasons.append("cutoff before typical monthly peak time")
    warning = (
        cutoff_minutes <= 12 * 60
        and (flat_duration >= 90 or temp_range <= 1.0)
        and suppression_score >= 1.0
        and latest_is_max
        and before_peak
    )
    return {
        "suppressed_late_warming_warning": bool(warning),
        "warning_type": "weather_suppressed_false_plateau" if warning else "none",
        "warning_reasons": reasons if warning else [],
        "false_plateau_score": float(score),
    }


def _regime_break_output(row: pd.Series) -> dict:
    cooler = int(row.get("regime_break_cooler_than_recent", 0) or 0)
    warmer = int(row.get("regime_break_warmer_than_recent", 0) or 0)
    if cooler:
        regime_type = "cooler_than_recent"
    elif warmer:
        regime_type = "warmer_than_recent"
    else:
        regime_type = "similar_to_recent"
    score = float(row.get("regime_break_score", 0.0) or 0.0)
    return {
        "regime_break_score": score,
        "regime_break_type": regime_type,
        "last3_weight_hint": float(max(0.0, min(1.0, 1.0 - score / 6.0))),
    }


def _plot_heat_risk_diagnostics(test: pd.DataFrame, report: dict, output_path: Path) -> None:
    metrics_by_cutoff = pd.DataFrame(report["metrics_by_cutoff"])
    top_errors = pd.DataFrame(report["top_errors"]).sort_values("abs_error_c", ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)

    ax_cutoff = fig.add_subplot(grid[0, 0])
    ax_cutoff.plot(metrics_by_cutoff["cutoff_local"], metrics_by_cutoff["tmax_mae_c"], marker="o", label="ML")
    ax_cutoff.plot(
        metrics_by_cutoff["cutoff_local"],
        metrics_by_cutoff["baseline_observed_max_mae_c"],
        marker="o",
        label="Observed max baseline",
    )
    ax_cutoff.set_title("Tmax MAE by Cutoff")
    ax_cutoff.set_ylabel("MAE (C)")
    ax_cutoff.tick_params(axis="x", rotation=30)
    ax_cutoff.legend()

    ax_remaining = fig.add_subplot(grid[0, 1])
    ax_remaining.scatter(
        test[TARGET_COLUMN],
        test["predicted_remaining_heat_c"],
        s=16,
        alpha=0.45,
        color="#457b9d",
        edgecolor="none",
    )
    upper = max(test[TARGET_COLUMN].max(), test["predicted_remaining_heat_c"].max()) + 0.5
    ax_remaining.plot([0, upper], [0, upper], color="#333333", linewidth=1)
    ax_remaining.set_title("Remaining Heat: Actual vs Predicted")
    ax_remaining.set_xlabel("Actual remaining heat (C)")
    ax_remaining.set_ylabel("Predicted remaining heat (C)")

    ax_error = fig.add_subplot(grid[1, 0])
    for cutoff, group in test.groupby("cutoff_local"):
        if cutoff in {"06:00", "09:00", "12:00", "15:00"}:
            ax_error.hist(group["error_c"], bins=24, alpha=0.45, label=cutoff)
    ax_error.axvline(0, color="#333333", linewidth=1)
    ax_error.set_title("Tmax Error Distribution by Selected Cutoffs")
    ax_error.set_xlabel("Prediction - Actual (C)")
    ax_error.legend()

    ax_top = fig.add_subplot(grid[1, 1])
    labels = top_errors["local_date"] + " " + top_errors["cutoff_local"]
    ax_top.barh(labels, top_errors["abs_error_c"], color="#f4a261", edgecolor="#9c6644")
    ax_top.set_title("Top Heat Risk Errors")
    ax_top.set_xlabel("Absolute error (C)")

    ax_threshold = fig.add_subplot(grid[2, 0])
    threshold_metrics = pd.DataFrame(
        [
            {"threshold": key, **value}
            for key, value in report["threshold_metrics"].items()
        ]
    )
    if not threshold_metrics.empty:
        ax_threshold.bar(threshold_metrics["threshold"], threshold_metrics["brier"], color="#2a9d8f")
    ax_threshold.set_title("Hot Threshold Probability Brier Score")
    ax_threshold.set_xlabel("Threshold (C)")
    ax_threshold.set_ylabel("Brier score")

    ax_update = fig.add_subplot(grid[2, 1])
    update_policy = pd.DataFrame(
        [
            {"cutoff_local": key, **value}
            for key, value in report["update_policy"].items()
        ]
    )
    if not update_policy.empty:
        ax_update.bar(
            update_policy["cutoff_local"],
            update_policy["median_abs_error_improvement_c"],
            color="#457b9d",
        )
    ax_update.axhline(UPDATE_MIN_IMPROVEMENT_C, color="#d1495b", linewidth=1, linestyle="--")
    ax_update.set_title("Median Error Improvement at Next Cutoff")
    ax_update.set_ylabel("Improvement (C)")
    ax_update.tick_params(axis="x", rotation=30)

    fig.suptitle("Tmax Remaining Heat and Update Value Diagnostics", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_thermal_curve_diagnostics(test: pd.DataFrame, report: dict, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax_phase = fig.add_subplot(grid[0, 0])
    phase_metrics = report.get("phase_metrics", {})
    labels = phase_metrics.get("labels", [])
    matrix = np.asarray(phase_metrics.get("confusion_matrix", []))
    if labels and matrix.size:
        image = ax_phase.imshow(matrix, cmap="Blues")
        ax_phase.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
        ax_phase.set_yticks(range(len(labels)), labels=labels)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax_phase.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center")
        fig.colorbar(image, ax=ax_phase, fraction=0.046)
    ax_phase.set_title("Thermal Phase Confusion Matrix")

    ax_curve = fig.add_subplot(grid[0, 1])
    horizon_rows = []
    for horizon in FUTURE_CURVE_HORIZONS_MINUTES:
        target = _future_target_column(horizon)
        if target not in test:
            continue
        metric = report.get("curve_metrics", {}).get("horizons", {}).get(str(horizon))
        if metric:
            horizon_rows.append({"horizon": f"+{horizon}m", "mae_c": metric["mae_c"]})
    if horizon_rows:
        horizon_frame = pd.DataFrame(horizon_rows)
        ax_curve.bar(horizon_frame["horizon"], horizon_frame["mae_c"], color="#457b9d")
    ax_curve.set_title("Future Curve MAE by Horizon")
    ax_curve.set_ylabel("MAE (C)")

    ax_late = fig.add_subplot(grid[1, 0])
    late_rows = []
    for threshold, values in report.get("late_warming_metrics", {}).items():
        late_rows.append(
            {
                "threshold": f">={threshold}C",
                "recall": values.get("recall_at_30pct") or 0.0,
                "precision": values.get("precision_at_30pct") or 0.0,
            }
        )
    if late_rows:
        late_frame = pd.DataFrame(late_rows)
        positions = np.arange(len(late_frame))
        ax_late.bar(positions - 0.18, late_frame["recall"], width=0.36, label="Recall")
        ax_late.bar(positions + 0.18, late_frame["precision"], width=0.36, label="Precision")
        ax_late.set_xticks(positions, late_frame["threshold"])
        ax_late.legend()
    ax_late.set_ylim(0, 1)
    ax_late.set_title("Late Warming Detection at 30% Probability")

    ax_compare = fig.add_subplot(grid[1, 1])
    comparison = report.get("model_comparison", {})
    rows = [
        {"model": "M0", "mae": comparison.get("m0_heat_risk", {}).get("tmax_mae_c")},
        {"model": "M1", "mae": comparison.get("m1_phase_features", {}).get("tmax_mae_c")},
        {"model": "M2 Curve", "mae": comparison.get("m2_future_curve", {}).get("tmax_mae_c")},
    ]
    compare_frame = pd.DataFrame(rows).dropna(subset=["mae"])
    if not compare_frame.empty:
        ax_compare.bar(compare_frame["model"], compare_frame["mae"], color="#2a9d8f")
    ax_compare.set_title("Model Tmax MAE Comparison")
    ax_compare.set_ylabel("MAE (C)")

    fig.suptitle("Thermal Phase and Future Curve Diagnostics", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _threshold_label(threshold: float) -> str:
    return f"target_tmax_ge_{_threshold_slug(threshold)}"


def _threshold_slug(threshold: float) -> str:
    return f"{threshold:g}c".replace(".", "p")


def _remaining_heat_label(threshold: float) -> str:
    return f"target_remaining_heat_ge_{_remaining_heat_slug(threshold)}"


def _remaining_heat_slug(threshold: float) -> str:
    return f"{threshold:.1f}".replace(".", "_")


def _future_target_column(horizon_minutes: int) -> str:
    return f"future_tmpc_plus_{horizon_minutes}m"


def _normalize_cutoff(cutoff_local: str) -> str:
    return _minutes_to_hhmm(_hhmm_to_minutes(cutoff_local))


def _brier_skill_score(brier: float, climatology_brier: float) -> float | None:
    if climatology_brier == 0.0:
        return None
    return float(1.0 - brier / climatology_brier)


def _clip_remaining(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
