from __future__ import annotations

import json
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
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from rksi_tmax.config import ProjectConfig, _hhmm_to_minutes
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
    "target_complete",
    "remaining_heat_target_c",
}
UPDATE_MIN_IMPROVEMENT_C = 0.15
UPDATE_MIN_INTERVAL_WIDTH_C = 1.5


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
    cutoff_config = replace(config, cutoff_local=cutoff_local)
    frame = make_daily_dataset(observations, cutoff_config)
    frame["cutoff_local"] = cutoff_local
    frame["cutoff_minutes"] = _hhmm_to_minutes(cutoff_local)
    frame[TARGET_COLUMN] = frame[FINAL_TMAX_COLUMN] - frame["tmpc_max_to_cutoff"]
    frame.loc[frame[TARGET_COLUMN] < 0.0, TARGET_COLUMN] = 0.0
    return frame


def load_heat_risk_table(path: str | Path) -> pd.DataFrame:
    return pl.read_parquet(path).to_pandas().sort_values(["local_date", "cutoff_minutes"])


def heat_risk_feature_columns(dataset: pd.DataFrame, missing_threshold: float = 1.0) -> list[str]:
    threshold_columns = {
        column for column in dataset.columns if column.startswith("target_tmax_ge_")
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

    columns = heat_risk_feature_columns(train, missing_threshold=config.feature_missing_threshold)
    regressor = _regression_pipeline(config)
    regressor.fit(train[columns], train[TARGET_COLUMN])

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

    threshold_probabilities = _threshold_probabilities(
        classifiers,
        test[columns],
        test["tmpc_max_to_cutoff"],
    )
    threshold_metrics = {
        threshold_text: _threshold_metrics(test[_threshold_label(float(threshold_text))], probability)
        for threshold_text, probability in threshold_probabilities.items()
    }

    remaining_prediction = _clip_remaining(regressor.predict(test[columns]))
    tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + remaining_prediction
    residual = test[FINAL_TMAX_COLUMN].to_numpy() - tmax_prediction
    update_policy = _build_update_policy(test, tmax_prediction)
    interval_calibration = _build_interval_calibration(residual)

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
        "tmax_mae_c": float(mean_absolute_error(test[FINAL_TMAX_COLUMN], tmax_prediction)),
        "tmax_rmse_c": _rmse(test[FINAL_TMAX_COLUMN], tmax_prediction),
        "tmax_bias_c": float(np.mean(tmax_prediction - test[FINAL_TMAX_COLUMN].to_numpy())),
        "observed_max_baseline_mae_c": float(
            mean_absolute_error(test[FINAL_TMAX_COLUMN], test["tmpc_max_to_cutoff"])
        ),
        "threshold_metrics": threshold_metrics,
        "update_policy": update_policy,
        "interval_calibration": interval_calibration,
        "feature_count": len(columns),
        "feature_missing_threshold": config.feature_missing_threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    config.heat_risk_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regressor": regressor,
            "classifiers": classifiers,
            "feature_columns": columns,
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
    bundle = joblib.load(config.heat_risk_model_path)
    metrics = bundle["metrics"]
    columns = bundle["feature_columns"]

    dates = pd.Series(sorted(dataset["local_date"].unique()))
    test_dates = set(
        date
        for date in dates
        if metrics["test_start"] <= str(date) <= metrics["test_end"]
    )
    test = dataset[dataset["local_date"].isin(test_dates)].copy()

    remaining_prediction = _clip_remaining(bundle["regressor"].predict(test[columns]))
    tmax_prediction = test["tmpc_max_to_cutoff"].to_numpy() + remaining_prediction
    test["predicted_remaining_heat_c"] = remaining_prediction
    test["predicted_tmax_c"] = tmax_prediction
    test["error_c"] = test["predicted_tmax_c"] - test[FINAL_TMAX_COLUMN]
    test["abs_error_c"] = test["error_c"].abs()

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
        "update_policy": metrics["update_policy"],
        "interval_calibration": metrics["interval_calibration"],
        "top_errors": _top_heat_risk_errors(test).to_dict(orient="records"),
    }

    artifacts_dir = Path(config.heat_risk_metrics_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "heat_risk_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report["top_errors"]).to_csv(artifacts_dir / "heat_risk_top_errors.csv", index=False)
    _plot_heat_risk_diagnostics(test, report, artifacts_dir / "heat_risk_diagnostics.png")
    return report


def predict_heat_risk(
    config: ProjectConfig,
    local_date: str,
    cutoff_local: str,
    dataset_path: str | Path | None = None,
) -> dict:
    bundle = joblib.load(config.heat_risk_model_path)
    columns = bundle["feature_columns"]
    row = _prediction_row(config, local_date, cutoff_local, dataset_path)
    missing = [column for column in columns if column not in row.columns]
    if missing:
        raise ValueError(f"Prediction row is missing model features: {missing}")

    remaining_heat = float(_clip_remaining(bundle["regressor"].predict(row[columns]))[0])
    observed_max_c = float(row["tmpc_max_to_cutoff"].iloc[0])
    predicted_tmax_c = observed_max_c + remaining_heat
    interval = _prediction_interval(predicted_tmax_c, bundle["interval_calibration"])
    update = _update_recommendation(cutoff_local, interval, bundle["update_policy"])

    result = {
        "station": config.station,
        "local_date": local_date,
        "cutoff_local": cutoff_local,
        "observed_max_to_cutoff_c": observed_max_c,
        "last_temp_to_cutoff_c": float(row["tmpc_last_to_cutoff"].iloc[0]),
        "predicted_remaining_heat_c": remaining_heat,
        "predicted_tmax_c": predicted_tmax_c,
        "predicted_tmax_f": predicted_tmax_c * 9.0 / 5.0 + 32.0,
        **interval,
        **update,
        "target_complete": bool(row.get("target_complete", pd.Series([0])).iloc[0]),
    }

    probabilities = _threshold_probabilities(
        bundle["classifiers"],
        row[columns],
        row["tmpc_max_to_cutoff"],
    )
    for threshold_text, probability_values in probabilities.items():
        threshold = float(threshold_text)
        probability = probability_values[0]
        result[f"prob_tmax_ge_{_threshold_slug(threshold)}"] = float(probability)

    if pd.notna(row[FINAL_TMAX_COLUMN].iloc[0]):
        result["actual_tmax_c"] = float(row[FINAL_TMAX_COLUMN].iloc[0])
        result["actual_remaining_heat_c"] = float(row[TARGET_COLUMN].iloc[0])
    return result


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


def _threshold_probabilities(
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


def _threshold_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    output = {
        "brier": float(brier_score_loss(y_true, probability)),
        "event_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    if y_true.nunique() > 1:
        output["roc_auc"] = float(roc_auc_score(y_true, probability))
    else:
        output["roc_auc"] = None
    return output


def _build_interval_calibration(residual: np.ndarray) -> dict:
    return {
        "residual_p10_c": float(np.quantile(residual, 0.10)),
        "residual_p50_c": float(np.quantile(residual, 0.50)),
        "residual_p90_c": float(np.quantile(residual, 0.90)),
    }


def _prediction_interval(prediction_c: float, calibration: dict) -> dict:
    p10 = prediction_c + calibration["residual_p10_c"]
    p50 = prediction_c + calibration["residual_p50_c"]
    p90 = prediction_c + calibration["residual_p90_c"]
    return {
        "prediction_p10_c": p10,
        "prediction_p50_c": p50,
        "prediction_p90_c": p90,
        "prediction_interval_80_low_c": p10,
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


def _update_recommendation(cutoff_local: str, interval: dict, update_policy: dict) -> dict:
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

    worth_update = (
        policy["median_abs_error_improvement_c"] >= UPDATE_MIN_IMPROVEMENT_C
        or interval_width >= UPDATE_MIN_INTERVAL_WIDTH_C
    )
    reason = (
        f"historical median improvement {policy['median_abs_error_improvement_c']:.2f}C; "
        f"80% interval width {interval_width:.2f}C"
    )
    return {
        "next_update_local": policy["next_cutoff_local"],
        "recommend_update_next_cutoff": bool(worth_update),
        "update_reason": reason,
    }


def _nearest_update_policy(cutoff_local: str, update_policy: dict) -> dict | None:
    if not update_policy:
        return None
    cutoff_minutes = _hhmm_to_minutes(cutoff_local)
    parsed = sorted((_hhmm_to_minutes(key), value) for key, value in update_policy.items())
    earlier = [value for minutes, value in parsed if minutes <= cutoff_minutes]
    if earlier:
        return earlier[-1]
    return parsed[0][1]


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
    return sorted(output, key=lambda row: row["cutoff_local"])


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


def _threshold_label(threshold: float) -> str:
    return f"target_tmax_ge_{_threshold_slug(threshold)}"


def _threshold_slug(threshold: float) -> str:
    return f"{threshold:g}c".replace(".", "p")


def _clip_remaining(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
