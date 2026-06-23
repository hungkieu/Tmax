from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from rksi_tmax.config import ProjectConfig, load_config
from rksi_tmax.features import fahrenheit_to_celsius, load_observations
from rksi_tmax.openmeteo import load_openmeteo_daily


KOREA_STATION_CONFIGS = {
    "RKSI": "configs/default.yaml",
    "RKPK": "configs/rkpk.yaml",
}
SUPPORTED_STATIONS = tuple(KOREA_STATION_CONFIGS)
ARTIFACT_DIR = Path("artifacts/next_metar_temp")
DATASET_PATH = ARTIFACT_DIR / "next_metar_temp_dataset.parquet"
MODEL_PATH = ARTIFACT_DIR / "next_metar_temp_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "next_metar_temp_metrics.json"
CANDIDATE_MODEL_PATH = ARTIFACT_DIR / "next_metar_temp_candidate.joblib"
CANDIDATE_METRICS_PATH = ARTIFACT_DIR / "next_metar_temp_candidate_metrics.json"
PREDICTION_LOG_PATH = ARTIFACT_DIR / "next_metar_temp_predictions.jsonl"
MODEL_NAME = "next_metar_integer_tmax"
MODEL_VERSION = 2
METAR_INTERVAL_MINUTES = {
    "RKSI": 30,
    "RKPK": 60,
}
# Forecast the next N regular METARs (1 = immediate next, up to 4 ahead).
HORIZONS = (1, 2, 3, 4)
FEATURE_COLUMNS = (
    "station_rksi",
    "station_rkpk",
    "temp_c",
    "temp_c_int",
    "temp_fraction_c",
    "horizon",
    "minutes_until_next_metar",
    "minute_of_day",
    "hour",
    "month",
    "day_of_year",
    "sin_minute_of_day",
    "cos_minute_of_day",
    "sin_day_of_year",
    "cos_day_of_year",
    "tmax_signal_c",
    "tmax_signal_minus_temp_c",
)
MIN_TRAIN_ROWS = 50
PROMOTION_MAX_MAE_DEGRADATION_C = 0.05
PROMOTION_MAX_EXACT_DEGRADATION = 0.02
HEALTH_MAE_UNHEALTHY_C = 1.0
HEALTH_BIAS_UNHEALTHY_C = 0.75
HEALTH_MIN_VERIFICATION_COVERAGE = 0.80
HEALTH_MAX_CONSECUTIVE_MISSES = 5


@dataclass(frozen=True)
class NextMetarPredictionInput:
    station: str
    observed_at: str
    temp_c: float
    tmax_signal_c: float | None = None


def build_next_metar_dataset(
    configs: Iterable[ProjectConfig],
    output_path: str | Path = DATASET_PATH,
) -> pd.DataFrame:
    frames = [_station_training_frame(config) for config in configs]
    dataset = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["target_temp_c_int"])
        .sort_values(["station", "observed_at_local"])
        .reset_index(drop=True)
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(dataset).write_parquet(output)
    return dataset


def train_next_metar_model(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
    *,
    promote: bool = True,
) -> dict[str, object]:
    dataset = pl.read_parquet(dataset_path).to_pandas()
    dataset = dataset.dropna(subset=["target_temp_c_int"]).sort_values(
        ["station", "observed_at_local"]
    )
    if len(dataset) < MIN_TRAIN_ROWS:
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} rows, found {len(dataset)}.")

    train, test = _temporal_split(dataset)
    model = _model_pipeline()
    model.fit(train[list(FEATURE_COLUMNS)], train["target_temp_c_int"])

    predictions = _round_predictions(model.predict(test[list(FEATURE_COLUMNS)]))
    actual = test["target_temp_c_int"].astype(int).to_numpy()
    errors = predictions - actual
    metrics = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_rows": int(len(dataset)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "stations": sorted(dataset["station"].unique().tolist()),
        "feature_columns": list(FEATURE_COLUMNS),
        "validation_mae_c": float(mean_absolute_error(actual, predictions)),
        "validation_rmse_c": float(mean_squared_error(actual, predictions) ** 0.5),
        "validation_bias_c": float(np.mean(errors)),
        "validation_exact_accuracy": float(np.mean(errors == 0)),
        "validation_within_1c_accuracy": float(np.mean(np.abs(errors) <= 1)),
        "trained_at": datetime.now().astimezone().isoformat(),
    }
    metrics["by_station"] = _metrics_by_station(test, predictions)
    metrics["by_horizon"] = _metrics_by_group(test, predictions, "horizon")

    target_model_path = Path(model_path)
    target_metrics_path = Path(metrics_path)
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    current_metrics = _load_json(target_metrics_path)
    promotion = should_promote_model(metrics, current_metrics)
    metrics["promoted"] = bool(promote and promotion["promote"])
    metrics["promotion"] = promotion

    if promote and promotion["promote"]:
        joblib.dump({"model": model, "metrics": metrics}, target_model_path)
        _write_json(target_metrics_path, metrics)
    else:
        joblib.dump({"model": model, "metrics": metrics}, CANDIDATE_MODEL_PATH)
        _write_json(CANDIDATE_METRICS_PATH, metrics)
    return metrics


def predict_next_metars(
    station: str,
    observed_at: str,
    temp_c: float,
    *,
    horizons: Iterable[int] = HORIZONS,
    tmax_signal_c: float | None = None,
    model_path: str | Path = MODEL_PATH,
    log_path: str | Path = PREDICTION_LOG_PATH,
) -> dict[str, object]:
    """Predict the integer temperature of the next N regular METARs.

    One prediction record per horizon is appended to the log (each verifiable
    independently against its own METAR slot).
    """
    normalized_station = _normalize_station(station)
    observed_dt = _parse_observed_at(observed_at, normalized_station)

    model = None
    model_version: int | None = MODEL_VERSION
    load_error: str | None = None
    try:
        bundle = _load_model_bundle(model_path)
        model = bundle["model"]
        model_version = bundle.get("metrics", {}).get("model_version", MODEL_VERSION)
    except Exception as exc:
        load_error = str(exc)
        model_version = None

    predictions = []
    for horizon in horizons:
        target_at = nth_regular_metar_time(normalized_station, observed_dt, horizon)
        fallback = False
        fallback_reason = load_error
        status = "ok"
        if model is None:
            predicted_temp_c = round_half_up(float(temp_c))
            fallback = True
            status = "fallback"
        else:
            try:
                row = make_prediction_feature_row(
                    normalized_station,
                    observed_dt,
                    float(temp_c),
                    horizon=horizon,
                    target_metar_at=target_at,
                    tmax_signal_c=tmax_signal_c,
                )
                predicted_temp_c = int(_round_predictions(model.predict(row[list(FEATURE_COLUMNS)]))[0])
            except Exception as exc:
                predicted_temp_c = round_half_up(float(temp_c))
                fallback = True
                fallback_reason = str(exc)
                status = "fallback"
        record = {
            "station": normalized_station,
            "observed_at": observed_dt.isoformat(),
            "horizon": int(horizon),
            "next_metar_at": target_at.isoformat(),
            "minutes_ahead": int((target_at - observed_dt).total_seconds() // 60),
            "input_temp_c": float(temp_c),
            "tmax_signal_c": _optional_float(tmax_signal_c),
            "predicted_temp_c": int(predicted_temp_c),
            "model": MODEL_NAME,
            "model_version": model_version,
            "status": status,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "verification_status": "pending",
        }
        append_prediction_log(record, log_path)
        predictions.append(record)

    return {
        "station": normalized_station,
        "observed_at": observed_dt.isoformat(),
        "input_temp_c": float(temp_c),
        "tmax_signal_c": _optional_float(tmax_signal_c),
        "model": MODEL_NAME,
        "model_version": model_version,
        "predictions": predictions,
    }


def predict_next_metar_temperature(
    station: str,
    observed_at: str,
    temp_c: float,
    *,
    tmax_signal_c: float | None = None,
    model_path: str | Path = MODEL_PATH,
    log_path: str | Path = PREDICTION_LOG_PATH,
) -> dict[str, object]:
    """Backward-compatible single-horizon predict (the immediate next METAR)."""
    result = predict_next_metars(
        station,
        observed_at,
        temp_c,
        horizons=(1,),
        tmax_signal_c=tmax_signal_c,
        model_path=model_path,
        log_path=log_path,
    )
    return result["predictions"][0]


def verify_next_metar_predictions(
    configs: Iterable[ProjectConfig],
    *,
    log_path: str | Path = PREDICTION_LOG_PATH,
    window: int = 100,
) -> dict[str, object]:
    records = read_prediction_log(log_path)
    if not records:
        return {"log_path": str(log_path), "verified": 0, "pending": 0, "health": {}}

    observations_by_station = {
        config.station.upper(): _official_actuals(config)
        for config in configs
        if config.station.upper() in SUPPORTED_STATIONS
    }
    verified = 0
    for record in records:
        if record.get("verification_status") == "verified":
            continue
        station = str(record.get("station", "")).upper()
        actuals = observations_by_station.get(station)
        if actuals is None or actuals.empty:
            continue
        next_metar_at = pd.Timestamp(str(record["next_metar_at"]))
        match = actuals[actuals["valid_local_iso"] == next_metar_at.isoformat()]
        if match.empty:
            continue
        if _mark_record_verified(record, int(match.iloc[-1]["actual_temp_c"])):
            verified += 1

    _write_prediction_log(records, log_path)
    return _verify_summary(log_path, verified, records, window)


def verify_next_metar_from_history(
    history_by_station: dict[str, pd.DataFrame],
    *,
    log_path: str | Path = PREDICTION_LOG_PATH,
    window: int = 100,
    tolerance_seconds: int = 300,
) -> dict[str, object]:
    """Verify pending predictions against a live observation stream from MongoDB.

    The stream is not aligned to METAR slots, so each pending prediction is
    matched to the observation nearest its ``next_metar_at`` within
    ``tolerance_seconds``.
    """
    records = read_prediction_log(log_path)
    if not records:
        return {"log_path": str(log_path), "verified": 0, "pending": 0, "health": {}}

    prepared = {
        station.upper(): _prepare_history_stream(history)
        for station, history in history_by_station.items()
        if station.upper() in SUPPORTED_STATIONS
    }
    verified = 0
    for record in records:
        if record.get("verification_status") == "verified":
            continue
        station = str(record.get("station", "")).upper()
        history = prepared.get(station)
        if history is None or history.empty:
            continue
        target = pd.Timestamp(str(record["next_metar_at"]))
        gaps = (history["valid_local"] - target).abs()
        nearest = gaps.idxmin()
        if gaps.loc[nearest].total_seconds() > tolerance_seconds:
            continue
        if _mark_record_verified(record, round_half_up(float(history.loc[nearest, "temp_c"]))):
            verified += 1

    _write_prediction_log(records, log_path)
    return _verify_summary(log_path, verified, records, window)


def _prepare_history_stream(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["valid_local", "temp_c"])
    frame = history[["valid_local", "temp_c"]].copy()
    frame["valid_local"] = pd.to_datetime(frame["valid_local"], utc=True)
    frame["temp_c"] = pd.to_numeric(frame["temp_c"], errors="coerce")
    return frame.dropna(subset=["valid_local", "temp_c"]).reset_index(drop=True)


def _mark_record_verified(record: dict[str, object], actual: int) -> bool:
    actual = int(actual)
    predicted = int(record["predicted_temp_c"])
    error = predicted - actual
    record["actual_temp_c"] = actual
    record["verification_error_c"] = error
    record["exact_hit"] = error == 0
    record["within_1c"] = abs(error) <= 1
    record["verification_status"] = "verified"
    record["verified_at"] = datetime.now().astimezone().isoformat()
    return True


def _verify_summary(
    log_path: str | Path,
    verified: int,
    records: list[dict[str, object]],
    window: int,
) -> dict[str, object]:
    pending = sum(1 for record in records if record.get("verification_status") != "verified")
    return {
        "log_path": str(log_path),
        "verified": verified,
        "pending": pending,
        "health": verification_health(records, window=window),
    }


def verification_health(
    records: list[dict[str, object]],
    *,
    station: str | None = None,
    window: int = 100,
) -> dict[str, object]:
    selected = records
    if station:
        normalized = station.upper()
        selected = [record for record in records if str(record.get("station", "")).upper() == normalized]
    recent = selected[-window:]
    verified = [record for record in recent if record.get("verification_status") == "verified"]
    pending_or_missing = len(recent) - len(verified)
    if not recent:
        return {"status": "unknown", "n": 0, "reasons": ["no_predictions"]}
    if not verified:
        return {
            "status": "unhealthy",
            "n": len(recent),
            "verified": 0,
            "verification_coverage": 0.0,
            "reasons": ["no_verified_predictions"],
        }

    errors = np.asarray([float(record["verification_error_c"]) for record in verified])
    exact = np.asarray([bool(record["exact_hit"]) for record in verified])
    within_1c = np.asarray([bool(record["within_1c"]) for record in verified])
    consecutive_misses = _consecutive_recent_misses(recent)
    coverage = len(verified) / len(recent)
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    reasons = []
    if mae > HEALTH_MAE_UNHEALTHY_C:
        reasons.append("rolling_mae_high")
    if abs(bias) > HEALTH_BIAS_UNHEALTHY_C:
        reasons.append("rolling_bias_high")
    if coverage < HEALTH_MIN_VERIFICATION_COVERAGE:
        reasons.append("verification_coverage_low")
    if consecutive_misses > HEALTH_MAX_CONSECUTIVE_MISSES:
        reasons.append("consecutive_misses_high")
    return {
        "status": "unhealthy" if reasons else "healthy",
        "n": len(recent),
        "verified": len(verified),
        "pending_or_missing": pending_or_missing,
        "verification_coverage": coverage,
        "mae_c": mae,
        "bias_c": bias,
        "exact_accuracy": float(np.mean(exact)),
        "within_1c_accuracy": float(np.mean(within_1c)),
        "consecutive_misses": consecutive_misses,
        "reasons": reasons,
    }


def should_promote_model(
    candidate_metrics: dict[str, object],
    current_metrics: dict[str, object] | None,
) -> dict[str, object]:
    if not current_metrics:
        return {"promote": True, "reason": "no_current_model"}
    if current_metrics.get("model_version") != candidate_metrics.get("model_version"):
        # A version bump is an intentional model change; the old version's metrics
        # are not comparable, so do not gate promotion on them.
        return {
            "promote": True,
            "reason": "model_version_changed",
            "candidate_model_version": candidate_metrics.get("model_version"),
            "current_model_version": current_metrics.get("model_version"),
        }
    candidate_mae = float(candidate_metrics["validation_mae_c"])
    current_mae = float(current_metrics["validation_mae_c"])
    candidate_exact = float(candidate_metrics["validation_exact_accuracy"])
    current_exact = float(current_metrics["validation_exact_accuracy"])
    if candidate_mae > current_mae + PROMOTION_MAX_MAE_DEGRADATION_C:
        return {
            "promote": False,
            "reason": "validation_mae_degraded",
            "candidate_validation_mae_c": candidate_mae,
            "current_validation_mae_c": current_mae,
        }
    if candidate_exact < current_exact - PROMOTION_MAX_EXACT_DEGRADATION:
        return {
            "promote": False,
            "reason": "exact_accuracy_degraded",
            "candidate_validation_exact_accuracy": candidate_exact,
            "current_validation_exact_accuracy": current_exact,
        }
    return {
        "promote": True,
        "reason": "validation_not_degraded",
        "candidate_validation_mae_c": candidate_mae,
        "current_validation_mae_c": current_mae,
        "candidate_validation_exact_accuracy": candidate_exact,
        "current_validation_exact_accuracy": current_exact,
    }


def next_regular_metar_time(station: str, observed_at: datetime) -> datetime:
    normalized_station = _normalize_station(station)
    local = _ensure_station_timezone(observed_at, normalized_station)
    interval = METAR_INTERVAL_MINUTES[normalized_station]
    minute_of_day = local.hour * 60 + local.minute
    next_minute = ((minute_of_day // interval) + 1) * interval
    next_day = local.date()
    if next_minute >= 24 * 60:
        next_minute -= 24 * 60
        next_day = next_day + timedelta(days=1)
    return datetime(
        next_day.year,
        next_day.month,
        next_day.day,
        next_minute // 60,
        next_minute % 60,
        tzinfo=ZoneInfo(_station_timezone(normalized_station)),
    )


def nth_regular_metar_time(station: str, observed_at: datetime, n: int) -> datetime:
    """Time of the n-th regular METAR after ``observed_at`` (n=1 is the next one)."""
    if n < 1:
        raise ValueError(f"Horizon must be >= 1, got {n}.")
    normalized_station = _normalize_station(station)
    base = next_regular_metar_time(normalized_station, observed_at)
    interval = METAR_INTERVAL_MINUTES[normalized_station]
    return base + timedelta(minutes=(n - 1) * interval)


def make_prediction_feature_row(
    station: str,
    observed_at: datetime,
    temp_c: float,
    *,
    horizon: int = 1,
    target_metar_at: datetime | None = None,
    tmax_signal_c: float | None = None,
) -> pd.DataFrame:
    normalized_station = _normalize_station(station)
    local = _ensure_station_timezone(observed_at, normalized_station)
    resolved_target = target_metar_at or nth_regular_metar_time(normalized_station, local, horizon)
    minute_of_day = local.hour * 60 + local.minute
    day_of_year = local.timetuple().tm_yday
    minutes_until_next = int((resolved_target - local).total_seconds() // 60)
    tmax_value = np.nan if tmax_signal_c is None else float(tmax_signal_c)
    row = {
        "station_rksi": int(normalized_station == "RKSI"),
        "station_rkpk": int(normalized_station == "RKPK"),
        "temp_c": float(temp_c),
        "temp_c_int": round_half_up(float(temp_c)),
        "temp_fraction_c": float(temp_c) - round_half_up(float(temp_c)),
        "horizon": int(horizon),
        "minutes_until_next_metar": minutes_until_next,
        "minute_of_day": minute_of_day,
        "hour": local.hour,
        "month": local.month,
        "day_of_year": day_of_year,
        "sin_minute_of_day": math.sin(2 * math.pi * minute_of_day / (24 * 60)),
        "cos_minute_of_day": math.cos(2 * math.pi * minute_of_day / (24 * 60)),
        "sin_day_of_year": math.sin(2 * math.pi * day_of_year / 366),
        "cos_day_of_year": math.cos(2 * math.pi * day_of_year / 366),
        "tmax_signal_c": tmax_value,
        "tmax_signal_minus_temp_c": tmax_value - float(temp_c) if pd.notna(tmax_value) else np.nan,
    }
    return pd.DataFrame([row])


def read_prediction_log(path: str | Path = PREDICTION_LOG_PATH) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_prediction_log(record: dict[str, object], path: str | Path = PREDICTION_LOG_PATH) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def korea_configs(station: str | None = None) -> list[ProjectConfig]:
    if station and station.upper() != "ALL":
        return [load_config(KOREA_STATION_CONFIGS[_normalize_station(station)])]
    return [load_config(path) for path in KOREA_STATION_CONFIGS.values()]


def _station_training_frame(
    config: ProjectConfig,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    station = _normalize_station(config.station)
    observations = load_observations(config.input_csv, config)
    frame = observations[["station", "valid_local", "tmpf"]].dropna(subset=["valid_local", "tmpf"]).copy()
    frame["station"] = station
    frame["valid_local"] = pd.to_datetime(frame["valid_local"])
    frame["temp_c"] = fahrenheit_to_celsius(pd.to_numeric(frame["tmpf"], errors="coerce"))
    frame = frame.dropna(subset=["temp_c"]).sort_values("valid_local").reset_index(drop=True)

    # Integer temperature of each regular METAR, keyed by its clean slot timestamp.
    regular_temp_by_slot = _regular_temp_by_slot(station, frame)
    # First regular slot strictly after each observation (clean :00/:30 boundary).
    base_slot = _next_regular_slot_series(station, frame["valid_local"])
    interval = METAR_INTERVAL_MINUTES[station]
    tmax_by_date = _load_tmax_signal_by_date(config, frame["valid_local"])

    parts = []
    for horizon in horizons:
        target_slot = base_slot + pd.to_timedelta((horizon - 1) * interval, unit="m")
        target_iso = target_slot.apply(lambda value: value.isoformat())
        target_temp = target_iso.map(regular_temp_by_slot).astype(float)
        rows = _vectorized_feature_frame(station, frame["valid_local"], frame["temp_c"], target_slot, horizon, tmax_by_date)
        rows["station"] = station
        rows["observed_at_local"] = frame["valid_local"].apply(lambda value: value.isoformat())
        rows["next_metar_at_local"] = target_iso.to_numpy()
        rows["target_temp_c_int"] = target_temp.to_numpy()
        parts.append(rows[target_temp.notna().to_numpy()])
    return pd.concat(parts, ignore_index=True)


def _regular_temp_by_slot(station: str, frame: pd.DataFrame) -> dict[str, float]:
    is_regular = frame["valid_local"].apply(lambda value: _is_regular_metar_time(station, value))
    regular = frame[is_regular]
    if regular.empty:
        return {}
    slot = _floor_to_minute_series(regular["valid_local"])
    slot_iso = slot.apply(lambda value: value.isoformat())
    temp_int = regular["temp_c"].apply(round_half_up)
    return dict(zip(slot_iso, temp_int))


def _next_regular_slot_series(station: str, valid_local: pd.Series) -> pd.Series:
    local = pd.to_datetime(valid_local)
    interval = METAR_INTERVAL_MINUTES[station]
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    next_minute = ((minute_of_day // interval) + 1) * interval
    return local.dt.normalize() + pd.to_timedelta(next_minute, unit="m")


def _floor_to_minute_series(valid_local: pd.Series) -> pd.Series:
    local = pd.to_datetime(valid_local)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    return local.dt.normalize() + pd.to_timedelta(minute_of_day, unit="m")


def _load_tmax_signal_by_date(config: ProjectConfig, valid_local: pd.Series) -> dict[str, float]:
    dates = sorted({pd.Timestamp(value).date().isoformat() for value in valid_local})
    if not dates:
        return {}
    frames = []
    if config.openmeteo_history_csv and Path(config.openmeteo_history_csv).exists():
        frames.append(load_openmeteo_daily(config.openmeteo_history_csv))
    if config.openmeteo_history_json and Path(config.openmeteo_history_json).exists():
        frames.append(_load_openmeteo_json_daily_tmax(config.openmeteo_history_json))
    if config.openmeteo_live_csv_pattern:
        for local_date in dates:
            live_path = Path(config.openmeteo_live_csv_pattern.format(date=local_date))
            if live_path.exists():
                frames.append(load_openmeteo_daily(live_path))
    if config.openmeteo_live_json_pattern:
        for local_date in dates:
            live_path = Path(config.openmeteo_live_json_pattern.format(date=local_date))
            if live_path.exists():
                frames.append(_load_openmeteo_json_daily_tmax(live_path))
    if not frames:
        return {}
    forecast = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["local_date"], keep="last")
        .sort_values("local_date")
    )
    forecast = forecast[forecast["local_date"].isin(dates)]
    clean = forecast[["local_date", "openmeteo_tmax_c"]].dropna(subset=["openmeteo_tmax_c"])
    return {
        str(row.local_date): float(row.openmeteo_tmax_c)
        for row in clean.itertuples(index=False)
    }


def _vectorized_feature_frame(
    station: str,
    valid_local: pd.Series,
    temp_c: pd.Series,
    target_slot: pd.Series,
    horizon: int,
    tmax_by_date: dict[str, float],
) -> pd.DataFrame:
    valid_local = pd.to_datetime(valid_local)
    target_slot = pd.to_datetime(target_slot)
    minute_of_day = valid_local.dt.hour * 60 + valid_local.dt.minute
    day_of_year = valid_local.dt.dayofyear
    temp_c = temp_c.astype(float)
    temp_c_int = temp_c.apply(round_half_up)
    local_date = valid_local.dt.date.astype(str)
    tmax_signal = local_date.map(tmax_by_date).astype(float)
    return pd.DataFrame(
        {
            "station_rksi": int(station == "RKSI"),
            "station_rkpk": int(station == "RKPK"),
            "temp_c": temp_c,
            "temp_c_int": temp_c_int,
            "temp_fraction_c": temp_c - temp_c_int,
            "horizon": int(horizon),
            "minutes_until_next_metar": (
                (target_slot - valid_local).dt.total_seconds() / 60.0
            ),
            "minute_of_day": minute_of_day,
            "hour": valid_local.dt.hour,
            "month": valid_local.dt.month,
            "day_of_year": day_of_year,
            "sin_minute_of_day": np.sin(2 * np.pi * minute_of_day / (24 * 60)),
            "cos_minute_of_day": np.cos(2 * np.pi * minute_of_day / (24 * 60)),
            "sin_day_of_year": np.sin(2 * np.pi * day_of_year / 366),
            "cos_day_of_year": np.cos(2 * np.pi * day_of_year / 366),
            "tmax_signal_c": tmax_signal,
            "tmax_signal_minus_temp_c": tmax_signal - temp_c,
        }
    )


def _load_openmeteo_json_daily_tmax(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    daily = payload.get("daily", {})
    dates = pd.Series(pd.to_datetime(daily.get("time", []), errors="coerce")).dt.date.astype("string")
    return pd.DataFrame(
        {
            "local_date": dates,
            "openmeteo_tmax_c": pd.to_numeric(
                pd.Series(daily.get("temperature_2m_max", [])),
                errors="coerce",
            ),
        }
    ).dropna(subset=["local_date"])


def _official_actuals(config: ProjectConfig) -> pd.DataFrame:
    observations = load_observations(config.input_csv, config)
    frame = observations[["valid_local", "tmpf"]].dropna(subset=["valid_local", "tmpf"]).copy()
    frame["valid_local"] = pd.to_datetime(frame["valid_local"])
    frame = frame[frame["valid_local"].apply(lambda value: _is_regular_metar_time(config.station, value))]
    frame["actual_temp_c"] = fahrenheit_to_celsius(pd.to_numeric(frame["tmpf"], errors="coerce")).apply(
        round_half_up
    )
    frame["valid_local_iso"] = frame["valid_local"].apply(lambda value: value.isoformat())
    return frame.dropna(subset=["actual_temp_c"])


def _temporal_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    tests = []
    for _, group in dataset.groupby("station", sort=False):
        split = max(1, int(len(group) * 0.8))
        if split >= len(group):
            split = len(group) - 1
        frames.append(group.iloc[:split])
        tests.append(group.iloc[split:])
    return pd.concat(frames, ignore_index=True), pd.concat(tests, ignore_index=True)


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    l2_regularization=0.05,
                    random_state=42,
                ),
            ),
        ]
    )


def _metrics_by_station(test: pd.DataFrame, predictions: np.ndarray) -> dict[str, dict[str, float | int]]:
    return _metrics_by_group(test, predictions, "station")


def _metrics_by_group(
    test: pd.DataFrame,
    predictions: np.ndarray,
    column: str,
) -> dict[str, dict[str, float | int]]:
    output = {}
    frame = test[[column, "target_temp_c_int"]].copy()
    frame["prediction"] = predictions
    for key, group in frame.groupby(column):
        errors = group["prediction"].to_numpy() - group["target_temp_c_int"].to_numpy()
        output[str(key)] = {
            "n": int(len(group)),
            "mae_c": float(np.mean(np.abs(errors))),
            "bias_c": float(np.mean(errors)),
            "exact_accuracy": float(np.mean(errors == 0)),
            "within_1c_accuracy": float(np.mean(np.abs(errors) <= 1)),
        }
    return output


def _load_model_bundle(model_path: str | Path) -> dict[str, object]:
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[misc,assignment]
    return joblib.load(model_path)


def _round_predictions(values: np.ndarray) -> np.ndarray:
    return np.asarray([round_half_up(float(value)) for value in values], dtype=int)


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _is_regular_metar_time(station: str, value: pd.Timestamp | datetime) -> bool:
    normalized_station = _normalize_station(station)
    local = _ensure_station_timezone(value.to_pydatetime() if isinstance(value, pd.Timestamp) else value, normalized_station)
    if normalized_station == "RKSI":
        return local.minute in {0, 30}
    if normalized_station == "RKPK":
        return local.minute == 0
    return False


def _parse_observed_at(value: str, station: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _ensure_station_timezone(parsed, station)


def _ensure_station_timezone(value: datetime, station: str) -> datetime:
    timezone = ZoneInfo(_station_timezone(station))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _station_timezone(station: str) -> str:
    _normalize_station(station)
    return "Asia/Seoul"


def _normalize_station(station: str) -> str:
    normalized = station.strip().upper()
    if normalized not in SUPPORTED_STATIONS:
        raise ValueError(f"Unsupported next-METAR station: {station}. Use RKSI or RKPK.")
    return normalized


def _optional_float(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_json(path: str | Path) -> dict[str, object] | None:
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_prediction_log(records: list[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _consecutive_recent_misses(records: list[dict[str, object]]) -> int:
    count = 0
    for record in reversed(records):
        if record.get("verification_status") != "verified":
            continue
        if bool(record.get("exact_hit")):
            break
        count += 1
    return count
