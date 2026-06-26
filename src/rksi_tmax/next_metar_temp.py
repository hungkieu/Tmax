from __future__ import annotations

import json
import math
import os
import pathlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline

from rksi_tmax.config import ProjectConfig
from rksi_tmax.features import (
    CLOUD_COLUMNS,
    CLOUD_COVER,
    FOG_CODES,
    NUMERIC_COLUMNS,
    PRECIP_CODES,
    WIND_REGIMES,
    fahrenheit_to_celsius,
    load_observations,
)
from rksi_tmax.openmeteo import (
    ensure_openmeteo_live_data,
    ensure_openmeteo_training_data,
    load_openmeteo_features_for_dates,
)


MODEL_NAME = "next_metar_temp"
MODEL_VERSION = 3
TARGET_HORIZON_MINUTES = 60
TARGET_TOLERANCE_MINUTES = 25
MIN_TRAIN_ROWS = 200
NON_FEATURE_COLUMNS = {
    "station",
    "valid_local",
    "observed_at_local",
    "target_valid_local",
    "local_date",
    "target_temp_c_int",
    "target_delta_c_int",
    "target_available",
    "metar",
}
REQUIRED_V3_DATASET_COLUMNS = {
    "temp_rise_from_trough_so_far_c",
    "minutes_since_trough_so_far",
    "today_vs_prev3_rise_from_trough_c",
    "cloud_cover_max_current",
    "precip_current",
    "target_temp_c_int",
}
OPENMETEO_HOURLY_COLUMNS = (
    "temperature_2m",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "visibility",
    "rain",
    "precipitation",
    "precipitation_probability",
)


def build_next_metar_temp_dataset(
    config: ProjectConfig,
    input_csv: str | Path | None = None,
    output_parquet: str | Path | None = None,
) -> pd.DataFrame:
    observations = load_observations(input_csv or config.input_csv, config)
    date_range = _observation_date_range(observations)
    if date_range is not None:
        ensure_openmeteo_training_data(
            config.openmeteo_history_json,
            config.openmeteo_latitude,
            config.openmeteo_longitude,
            date_range[0],
            date_range[1],
            timezone=config.openmeteo_timezone,
        )
    dataset = make_next_metar_temp_dataset(observations, config, include_target=True)
    output = Path(output_parquet or config.next_metar_temp_dataset_parquet)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(dataset).write_parquet(output)
    return dataset


def make_next_metar_temp_dataset(
    observations: pd.DataFrame,
    config: ProjectConfig,
    *,
    include_target: bool,
    horizon_minutes: int = TARGET_HORIZON_MINUTES,
    target_tolerance_minutes: int = TARGET_TOLERANCE_MINUTES,
) -> pd.DataFrame:
    frame = _base_feature_frame(observations, config)
    if frame.empty:
        return frame
    frame = _add_station_features(frame)
    frame = _add_time_features(frame)
    frame = _add_current_weather_features(frame)
    frame = _add_recent_temperature_features(frame)
    frame = _add_daily_trough_features(frame)
    frame = _add_previous_3_day_features(frame)
    frame = _add_openmeteo_features(frame, config, horizon_minutes)
    if include_target:
        frame = _add_target_temperature(
            frame,
            horizon_minutes=horizon_minutes,
            tolerance_minutes=target_tolerance_minutes,
        )
        frame = frame[frame["target_available"] == 1].reset_index(drop=True)
    frame["observed_at_local"] = frame["valid_local"].astype(str)
    return frame


def train_next_metar_temp_model(config: ProjectConfig) -> dict:
    dataset = _load_or_build_dataset(config)
    dataset = dataset.dropna(subset=["target_temp_c_int"]).sort_values("valid_local").reset_index(
        drop=True
    )
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} next-METAR rows to train.")

    feature_columns = next_metar_temp_feature_columns(dataset, config.feature_missing_threshold)
    split_index = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_fraction))))
    train = dataset.iloc[:split_index].copy()
    test = dataset.iloc[split_index:].copy()

    model = _classifier_pipeline(config)
    model.fit(train[feature_columns], train["target_temp_c_int"].astype(int))
    predicted = model.predict(test[feature_columns]).astype(int)
    probabilities = model.predict_proba(test[feature_columns])
    classes = model.named_steps["classifier"].classes_.astype(int)
    metrics = _metrics(dataset, train, test, predicted, probabilities, classes, feature_columns)

    output = Path(config.next_metar_temp_model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metrics": metrics,
            "feature_columns": feature_columns,
            "model": model,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "horizon_minutes": TARGET_HORIZON_MINUTES,
            "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
        },
        output,
        compress=3,
    )
    metrics_path = Path(config.next_metar_temp_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def validate_next_metar_temp_model(config: ProjectConfig) -> dict:
    bundle = _load_next_metar_model_bundle(config.next_metar_temp_model_path)
    dataset = _load_or_build_dataset(config)
    dataset = dataset.dropna(subset=["target_temp_c_int"]).sort_values("valid_local").reset_index(
        drop=True
    )
    split_index = max(1, min(len(dataset) - 1, int(len(dataset) * (1.0 - config.test_fraction))))
    test = dataset.iloc[split_index:].copy()
    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in test.columns]
    if missing:
        raise ValueError(f"Validation dataset is missing model features: {missing}")
    probabilities = bundle["model"].predict_proba(test[feature_columns])
    classes = bundle["model"].named_steps["classifier"].classes_.astype(int)
    predicted = classes[np.argmax(probabilities, axis=1)]
    report = {
        "summary": _prediction_metrics(test, predicted, probabilities, classes),
        "by_station": _metrics_by_group(test, predicted, probabilities, classes, "station"),
        "by_hour": _metrics_by_group(test, predicted, probabilities, classes, "hour"),
        "model_path": str(config.next_metar_temp_model_path),
        "dataset_path": str(config.next_metar_temp_dataset_parquet),
        "n_test": int(len(test)),
    }
    report_path = Path(config.next_metar_temp_metrics_path).with_name(
        f"{Path(config.next_metar_temp_metrics_path).stem}_validation_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def predict_next_metar_temp(
    config: ProjectConfig,
    *,
    as_of_local: str | None = None,
    fetch_openmeteo: bool = False,
) -> dict:
    bundle = _load_next_metar_model_bundle(config.next_metar_temp_model_path)
    observations = load_observations(config.input_csv, config)
    if observations.empty:
        raise ValueError(f"No observations found for {config.station}.")

    if fetch_openmeteo:
        local_date = _prediction_local_date(observations, config, as_of_local)
        ensure_openmeteo_live_data(
            config.openmeteo_live_json_pattern,
            config.openmeteo_latitude,
            config.openmeteo_longitude,
            local_date,
            timezone=config.openmeteo_timezone,
        )

    features = make_next_metar_temp_dataset(observations, config, include_target=False)
    row = _select_prediction_row(features, config, as_of_local)
    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in row.columns]
    if missing:
        raise ValueError(f"Prediction row is missing model features: {missing}")

    probabilities = bundle["model"].predict_proba(row[feature_columns])[0]
    classes = bundle["model"].named_steps["classifier"].classes_.astype(int)
    probability_by_temp = {
        int(temp): float(probabilities[index]) for index, temp in enumerate(classes)
    }
    predicted_temp = int(classes[int(np.argmax(probabilities))])
    current_temp = int(row["temp_c_int"].iloc[0])
    expected_temp = float(np.sum(classes * probabilities))
    target_time = pd.Timestamp(row["valid_local"].iloc[0]) + pd.Timedelta(
        minutes=bundle.get("horizon_minutes", TARGET_HORIZON_MINUTES)
    )
    output = {
        "model": bundle.get("model_name", MODEL_NAME),
        "model_version": int(bundle.get("model_version", MODEL_VERSION)),
        "station": config.station,
        "observed_at_local": str(row["valid_local"].iloc[0]),
        "target_time_local": str(target_time),
        "input_metar": row["metar"].iloc[0] if "metar" in row and pd.notna(row["metar"].iloc[0]) else None,
        "current_temp_c": float(row["temp_c"].iloc[0]),
        "current_temp_c_int": current_temp,
        "predicted_temp_c": predicted_temp,
        "expected_temp_c": round(expected_temp, 2),
        "predicted_delta_c": int(predicted_temp - current_temp),
        "probabilities_by_temp_c": {
            str(temp): round(probability, 4)
            for temp, probability in sorted(probability_by_temp.items())
            if probability >= 0.001
        },
        "prob_next_temp_eq_current_minus_1c": _probability_for_temp(
            probability_by_temp, current_temp - 1
        ),
        "prob_next_temp_eq_current_c": _probability_for_temp(probability_by_temp, current_temp),
        "prob_next_temp_eq_current_plus_1c": _probability_for_temp(
            probability_by_temp, current_temp + 1
        ),
        "prob_next_temp_le_current_minus_1c": _probability_below_or_equal(
            probability_by_temp, current_temp - 1
        ),
        "prob_next_temp_ge_current_plus_1c": _probability_above_or_equal(
            probability_by_temp, current_temp + 1
        ),
        "temperature_context": _temperature_context(row),
        "weather_context": _weather_context(row),
        "openmeteo_context": _openmeteo_context(row),
    }
    return output


def next_metar_temp_feature_columns(
    dataset: pd.DataFrame,
    missing_threshold: float = 0.85,
) -> list[str]:
    columns = []
    for column in dataset.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(dataset[column]):
            continue
        if dataset[column].isna().mean() > missing_threshold:
            continue
        columns.append(column)
    return columns


def _base_feature_frame(observations: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    frame = observations.copy()
    if "valid_local" not in frame.columns:
        frame["valid_local"] = (
            pd.to_datetime(frame["valid"], format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
            .dt.tz_convert(config.timezone)
        )
    frame["valid_local"] = pd.to_datetime(frame["valid_local"], errors="coerce")
    frame = frame.dropna(subset=["valid_local"]).sort_values("valid_local").reset_index(drop=True)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["temp_c"] = fahrenheit_to_celsius(frame["tmpf"])
    frame = frame.dropna(subset=["temp_c"]).reset_index(drop=True)
    frame["temp_c_int"] = _round_half_up(frame["temp_c"]).astype(int)
    frame["temp_fraction_c"] = frame["temp_c"] - frame["temp_c_int"]
    if "dwpf" in frame.columns:
        frame["dewpoint_c"] = fahrenheit_to_celsius(frame["dwpf"])
        frame["dewpoint_spread_c"] = frame["temp_c"] - frame["dewpoint_c"]
    if "feel" in frame.columns:
        frame["feel_c"] = fahrenheit_to_celsius(frame["feel"])
        frame["feel_minus_temp_c"] = frame["feel_c"] - frame["temp_c"]
    frame["local_date"] = frame["valid_local"].dt.date.astype(str)
    frame["local_date_dt"] = pd.to_datetime(frame["local_date"])
    frame["local_minutes"] = frame["valid_local"].dt.hour * 60 + frame["valid_local"].dt.minute
    return frame


def _add_station_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output[f"station_{str(output['station'].iloc[0]).lower()}"] = 1
    return output


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["hour"] = output["valid_local"].dt.hour
    output["month"] = output["valid_local"].dt.month
    output["day_of_year"] = output["valid_local"].dt.dayofyear
    output["sin_minute_of_day"] = np.sin(2.0 * np.pi * output["local_minutes"] / 1440.0)
    output["cos_minute_of_day"] = np.cos(2.0 * np.pi * output["local_minutes"] / 1440.0)
    output["sin_day_of_year"] = np.sin(2.0 * np.pi * output["day_of_year"] / 366.0)
    output["cos_day_of_year"] = np.cos(2.0 * np.pi * output["day_of_year"] / 366.0)
    return output


def _add_current_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    cover_columns = []
    for column in CLOUD_COLUMNS:
        if column not in output.columns:
            continue
        cover_column = f"{column}_cover"
        output[cover_column] = output[column].map(CLOUD_COVER)
        cover_columns.append(cover_column)
    if cover_columns:
        output["cloud_cover_max_current"] = output[cover_columns].max(axis=1)
        output["cloud_cover_lowest_layer_current"] = output[cover_columns].bfill(axis=1).iloc[:, 0]
    ceiling_columns = [column for column in ("skyl1", "skyl2", "skyl3", "skyl4") if column in output]
    if ceiling_columns:
        output["ceiling_ft_current"] = output[ceiling_columns].min(axis=1)
        output["low_cloud_current"] = (output["ceiling_ft_current"] <= 3000.0).astype(int)
    codes = output.get("wxcodes", pd.Series("", index=output.index)).fillna("").astype(str)
    output["precip_current"] = codes.str.contains("|".join(PRECIP_CODES), regex=True).astype(int)
    output["fog_current"] = codes.str.contains("|".join(FOG_CODES), regex=True).astype(int)
    output["thunder_current"] = codes.str.contains("TS", regex=False).astype(int)
    output["shower_current"] = codes.str.contains("SH", regex=False).astype(int)
    if "drct" in output.columns:
        direction = output["drct"].where(output["drct"].between(0.0, 360.0)) % 360.0
        radians = np.deg2rad(direction)
        output["wind_dir_sin_current"] = np.sin(radians)
        output["wind_dir_cos_current"] = np.cos(radians)
        for regime, sector in WIND_REGIMES.items():
            if len(sector) == 4:
                lower_1, upper_1, lower_2, upper_2 = sector
                mask = direction.between(lower_1, upper_1, inclusive="left") | direction.between(
                    lower_2, upper_2, inclusive="left"
                )
            else:
                lower, upper = sector
                mask = direction.between(lower, upper, inclusive="left")
            output[f"wind_regime_{regime}_current"] = mask.astype(int)
    return output


def _add_recent_temperature_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().sort_values(["station", "valid_local"]).reset_index(drop=True)
    for minutes in (30, 60, 120, 180):
        past = output[["station", "valid_local", "temp_c"]].copy()
        past = past.rename(columns={"valid_local": "past_valid_local", "temp_c": f"temp_c_past_{minutes}m"})
        left = output[["station", "valid_local"]].copy()
        left["row_id"] = np.arange(len(left))
        left["wanted_past_local"] = left["valid_local"] - pd.Timedelta(minutes=minutes)
        merged = pd.merge_asof(
            left.sort_values("wanted_past_local"),
            past.sort_values("past_valid_local"),
            left_on="wanted_past_local",
            right_on="past_valid_local",
            by="station",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=20),
        ).sort_values("row_id")
        output[f"temp_c_past_{minutes}m"] = merged[f"temp_c_past_{minutes}m"].to_numpy()
        output[f"temp_rise_last_{minutes}m"] = output["temp_c"] - output[f"temp_c_past_{minutes}m"]
    return output


def _add_daily_trough_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), group in frame.sort_values("valid_local").groupby(["station", "local_date"], sort=False):
        group = group.copy()
        group["tmpc_min_so_far"] = group["temp_c"].cummin()
        trough_minutes = []
        trough_minute = None
        trough_value = math.inf
        for _, row in group.iterrows():
            value = float(row["temp_c"])
            if value <= trough_value:
                trough_value = value
                trough_minute = int(row["local_minutes"])
            trough_minutes.append(trough_minute)
        group["trough_minute_so_far"] = trough_minutes
        group["minutes_since_trough_so_far"] = group["local_minutes"] - group["trough_minute_so_far"]
        group["temp_rise_from_trough_so_far_c"] = group["temp_c"] - group["tmpc_min_so_far"]
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(["station", "valid_local"]).reset_index(drop=True)


def _add_previous_3_day_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    keys = [
        "station",
        "local_date_dt",
        "local_minutes",
        "temp_c",
        "tmpc_min_so_far",
        "temp_rise_from_trough_so_far_c",
    ]
    previous_base = output[keys].copy()
    for offset in (1, 2, 3):
        previous = previous_base.copy()
        previous["local_date_dt"] = previous["local_date_dt"] + pd.Timedelta(days=offset)
        previous = previous.rename(
            columns={
                "temp_c": f"prev{offset}_temp_same_minute_c",
                "tmpc_min_so_far": f"prev{offset}_trough_so_far_c",
                "temp_rise_from_trough_so_far_c": f"prev{offset}_rise_from_trough_c",
            }
        )
        output = output.merge(
            previous,
            on=["station", "local_date_dt", "local_minutes"],
            how="left",
        )
    temp_columns = [f"prev{offset}_temp_same_minute_c" for offset in (1, 2, 3)]
    rise_columns = [f"prev{offset}_rise_from_trough_c" for offset in (1, 2, 3)]
    trough_columns = [f"prev{offset}_trough_so_far_c" for offset in (1, 2, 3)]
    output["prev3_temp_same_minute_mean_c"] = output[temp_columns].mean(axis=1)
    output["prev3_rise_from_trough_mean_c"] = output[rise_columns].mean(axis=1)
    output["prev3_trough_so_far_mean_c"] = output[trough_columns].mean(axis=1)
    output["today_vs_prev3_temp_same_minute_c"] = (
        output["temp_c"] - output["prev3_temp_same_minute_mean_c"]
    )
    output["today_vs_prev3_rise_from_trough_c"] = (
        output["temp_rise_from_trough_so_far_c"] - output["prev3_rise_from_trough_mean_c"]
    )
    output["today_vs_prev3_trough_c"] = output["tmpc_min_so_far"] - output["prev3_trough_so_far_mean_c"]
    return output


def _add_openmeteo_features(
    frame: pd.DataFrame,
    config: ProjectConfig,
    horizon_minutes: int,
) -> pd.DataFrame:
    output = frame.copy()
    dates = sorted(output["local_date"].dropna().astype(str).unique())
    daily = load_openmeteo_features_for_dates(
        config.openmeteo_history_csv,
        config.openmeteo_live_csv_pattern,
        dates,
        config.openmeteo_history_json,
        config.openmeteo_live_json_pattern,
    )
    if daily is not None:
        output = output.merge(daily, on="local_date", how="left")

    hourly = _load_openmeteo_hourly_features(config, dates)
    if hourly.empty:
        return output
    current = hourly.add_prefix("openmeteo_current_").rename(
        columns={
            "openmeteo_current_local_date": "local_date",
            "openmeteo_current_local_hour": "hour",
        }
    )
    output = output.merge(current, on=["local_date", "hour"], how="left")
    output["target_valid_local_for_openmeteo"] = output["valid_local"] + pd.Timedelta(
        minutes=horizon_minutes
    )
    output["target_local_date"] = output["target_valid_local_for_openmeteo"].dt.date.astype(str)
    output["target_hour"] = output["target_valid_local_for_openmeteo"].dt.hour
    target = hourly.add_prefix("openmeteo_target_").rename(
        columns={
            "openmeteo_target_local_date": "target_local_date",
            "openmeteo_target_local_hour": "target_hour",
        }
    )
    output = output.merge(target, on=["target_local_date", "target_hour"], how="left")
    for column in (
        "temperature_2m",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "rain",
        "precipitation",
        "precipitation_probability",
    ):
        current_column = f"openmeteo_current_{column}"
        target_column = f"openmeteo_target_{column}"
        if current_column in output and target_column in output:
            output[f"openmeteo_target_minus_current_{column}"] = (
                output[target_column] - output[current_column]
            )
    return output.drop(columns=["target_valid_local_for_openmeteo"], errors="ignore")


def _add_target_temperature(
    frame: pd.DataFrame,
    *,
    horizon_minutes: int,
    tolerance_minutes: int,
) -> pd.DataFrame:
    output = frame.copy().sort_values(["station", "valid_local"]).reset_index(drop=True)
    targets = output[["station", "valid_local", "temp_c_int"]].rename(
        columns={"valid_local": "target_valid_local", "temp_c_int": "target_temp_c_int"}
    )
    left = output[["station", "valid_local", "temp_c_int"]].copy()
    left["row_id"] = np.arange(len(left))
    left["wanted_target_local"] = left["valid_local"] + pd.Timedelta(minutes=horizon_minutes)
    merged = pd.merge_asof(
        left.sort_values("wanted_target_local"),
        targets.sort_values("target_valid_local"),
        left_on="wanted_target_local",
        right_on="target_valid_local",
        by="station",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    ).sort_values("row_id")
    output["target_valid_local"] = merged["target_valid_local"].to_numpy()
    output["target_temp_c_int"] = merged["target_temp_c_int"].to_numpy()
    output["target_available"] = output["target_temp_c_int"].notna().astype(int)
    output["target_delta_c_int"] = output["target_temp_c_int"] - output["temp_c_int"]
    return output


def _load_openmeteo_hourly_features(config: ProjectConfig, local_dates: list[str]) -> pd.DataFrame:
    paths = []
    if config.openmeteo_history_json and Path(config.openmeteo_history_json).exists():
        paths.append(Path(config.openmeteo_history_json))
    if config.openmeteo_live_json_pattern:
        for local_date in local_dates:
            path = Path(config.openmeteo_live_json_pattern.format(date=local_date))
            if path.exists():
                paths.append(path)
    frames = [_openmeteo_hourly_from_json(path, config) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    output = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["local_date", "local_hour"], keep="last")
        .sort_values(["local_date", "local_hour"])
        .reset_index(drop=True)
    )
    return output[output["local_date"].isin(local_dates)]


def _openmeteo_hourly_from_json(path: Path, config: ProjectConfig) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    source_tz = _zoneinfo(config.openmeteo_timezone)
    station_tz = _zoneinfo(config.timezone)
    timestamps = pd.to_datetime(pd.Series(times), errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(source_tz)
    timestamps = timestamps.dt.tz_convert(station_tz)
    frame = pd.DataFrame({"valid_local": timestamps})
    frame = frame.dropna(subset=["valid_local"]).reset_index(drop=True)
    frame["local_date"] = frame["valid_local"].dt.date.astype(str)
    frame["local_hour"] = frame["valid_local"].dt.hour
    for column in OPENMETEO_HOURLY_COLUMNS:
        values = hourly.get(column)
        if values is None:
            frame[column] = np.nan
        else:
            frame[column] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    return frame.drop(columns=["valid_local"])


def _classifier_pipeline(config: ProjectConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=90,
                    max_depth=22,
                    max_features=0.65,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _load_or_build_dataset(config: ProjectConfig) -> pd.DataFrame:
    path = Path(config.next_metar_temp_dataset_parquet)
    if path.exists():
        dataset = pl.read_parquet(path).to_pandas()
        if REQUIRED_V3_DATASET_COLUMNS.issubset(dataset.columns):
            return dataset
    return build_next_metar_temp_dataset(config)


def _metrics(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    feature_columns: list[str],
) -> dict:
    metrics = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_rows": int(len(dataset)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "stations": sorted(dataset["station"].dropna().astype(str).unique().tolist()),
        "feature_columns": feature_columns,
        "target_horizon_minutes": TARGET_HORIZON_MINUTES,
        "target_tolerance_minutes": TARGET_TOLERANCE_MINUTES,
        **_prediction_metrics(test, predicted, probabilities, classes),
        "by_station": _metrics_by_group(test, predicted, probabilities, classes, "station"),
        "by_hour": _metrics_by_group(test, predicted, probabilities, classes, "hour"),
        "trained_at": datetime.now().astimezone().isoformat(),
    }
    return metrics


def _prediction_metrics(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict:
    actual = frame["target_temp_c_int"].astype(int).to_numpy()
    current = frame["temp_c_int"].astype(int).to_numpy()
    baseline = current
    output = {
        "mae_c": float(mean_absolute_error(actual, predicted)),
        "baseline_persistence_mae_c": float(mean_absolute_error(actual, baseline)),
        "exact_accuracy": float(accuracy_score(actual, predicted)),
        "within_1c_accuracy": float((np.abs(actual - predicted) <= 1).mean()),
        "bias_c": float(np.mean(predicted - actual)),
        "delta_exact_accuracy": float(accuracy_score(actual - current, predicted - current)),
        "delta_direction_accuracy": float(
            accuracy_score(np.sign(actual - current), np.sign(predicted - current))
        ),
        "persistence_exact_accuracy": float(accuracy_score(actual, baseline)),
        "n": int(len(frame)),
    }
    try:
        output["log_loss"] = float(log_loss(actual, probabilities, labels=classes))
    except ValueError:
        output["log_loss"] = None
    return output


def _metrics_by_group(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    column: str,
) -> dict:
    output = {}
    if column not in frame:
        return output
    prediction = pd.Series(predicted, index=frame.index)
    for key, group in frame.groupby(column):
        indexes = group.index
        probability_rows = probabilities[[frame.index.get_loc(index) for index in indexes]]
        output[str(key)] = _prediction_metrics(
            group,
            prediction.loc[indexes].to_numpy(),
            probability_rows,
            classes,
        )
    return output


def _select_prediction_row(
    features: pd.DataFrame,
    config: ProjectConfig,
    as_of_local: str | None,
) -> pd.DataFrame:
    frame = features.sort_values("valid_local")
    if as_of_local:
        as_of = _parse_local_datetime(as_of_local, config)
        candidates = frame[frame["valid_local"] <= as_of]
        if candidates.empty:
            raise ValueError(f"No {config.station} METAR observation at or before {as_of_local}.")
        return candidates.tail(1).copy()
    return frame.tail(1).copy()


def _prediction_local_date(
    observations: pd.DataFrame,
    config: ProjectConfig,
    as_of_local: str | None,
) -> str:
    if as_of_local:
        return _parse_local_datetime(as_of_local, config).date().isoformat()
    frame = observations.copy()
    frame["valid_local"] = pd.to_datetime(frame["valid_local"], errors="coerce")
    latest = frame.dropna(subset=["valid_local"]).sort_values("valid_local").iloc[-1]
    return pd.Timestamp(latest["valid_local"]).date().isoformat()


def _temperature_context(row: pd.DataFrame) -> dict:
    data = row.iloc[0]
    keys = [
        "tmpc_min_so_far",
        "minutes_since_trough_so_far",
        "temp_rise_from_trough_so_far_c",
        "temp_rise_last_30m",
        "temp_rise_last_60m",
        "temp_rise_last_120m",
        "prev3_temp_same_minute_mean_c",
        "prev3_rise_from_trough_mean_c",
        "today_vs_prev3_temp_same_minute_c",
        "today_vs_prev3_rise_from_trough_c",
    ]
    return {key: _optional_float(data.get(key)) for key in keys}


def _weather_context(row: pd.DataFrame) -> dict:
    data = row.iloc[0]
    keys = [
        "cloud_cover_max_current",
        "ceiling_ft_current",
        "low_cloud_current",
        "sknt",
        "gust",
        "drct",
        "precip_current",
        "fog_current",
        "vsby",
        "dewpoint_spread_c",
    ]
    return {key: _optional_float(data.get(key)) for key in keys}


def _openmeteo_context(row: pd.DataFrame) -> dict:
    data = row.iloc[0]
    keys = [
        "openmeteo_current_temperature_2m",
        "openmeteo_target_temperature_2m",
        "openmeteo_target_minus_current_temperature_2m",
        "openmeteo_current_cloud_cover",
        "openmeteo_target_cloud_cover",
        "openmeteo_current_precipitation_probability",
        "openmeteo_target_precipitation_probability",
        "openmeteo_target_minus_current_precipitation_probability",
        "openmeteo_target_minus_current_wind_speed_10m",
    ]
    return {key: _optional_float(data.get(key)) for key in keys if key in data}


def _load_model_bundle(model_path: str | Path) -> dict:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]
    return joblib.load(model_path)


def _load_next_metar_model_bundle(model_path: str | Path) -> dict:
    bundle = _load_model_bundle(model_path)
    if bundle.get("model_name") != MODEL_NAME or int(bundle.get("model_version", 0)) < MODEL_VERSION:
        raise ValueError(
            "Existing next-METAR artifact is not the v3 METAR/weather/Open-Meteo model. "
            "Run rksi-build-next-metar-temp-dataset and rksi-train-next-metar-temp first."
        )
    return bundle


def _observation_date_range(observations: pd.DataFrame) -> tuple[str, str] | None:
    if observations.empty or "valid_local" not in observations:
        return None
    dates = pd.to_datetime(observations["valid_local"], errors="coerce").dropna().dt.date.astype(str)
    if dates.empty:
        return None
    return str(dates.min()), str(dates.max())


def _parse_local_datetime(value: str, config: ProjectConfig) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(_zoneinfo(config.timezone))
    return timestamp.tz_convert(_zoneinfo(config.timezone))


def _zoneinfo(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        if value.upper() in {"GMT", "UTC"}:
            return ZoneInfo("UTC")
        raise


def _round_half_up(series: pd.Series) -> pd.Series:
    return np.floor(pd.to_numeric(series, errors="coerce") + 0.5)


def _probability_for_temp(probability_by_temp: dict[int, float], temp: int) -> float:
    return round(float(probability_by_temp.get(int(temp), 0.0)), 4)


def _probability_below_or_equal(probability_by_temp: dict[int, float], temp: int) -> float:
    return round(float(sum(prob for value, prob in probability_by_temp.items() if value <= temp)), 4)


def _probability_above_or_equal(probability_by_temp: dict[int, float], temp: int) -> float:
    return round(float(sum(prob for value, prob in probability_by_temp.items() if value >= temp)), 4)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
