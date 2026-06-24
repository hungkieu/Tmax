from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from rksi_tmax.config import ProjectConfig
from rksi_tmax.openmeteo import load_openmeteo_features_for_dates
from rksi_tmax.storage import read_station_observations_from_duckdb


NUMERIC_COLUMNS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "feel",
]

CLOUD_COLUMNS = ["skyc1", "skyc2", "skyc3", "skyc4"]
CLOUD_COVER = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
PRECIP_CODES = ("RA", "SN", "DZ", "TS", "SH", "GR", "PL")
FOG_CODES = ("BR", "FG")
WIND_REGIMES = {
    "n": (337.5, 360.0, 0.0, 22.5),
    "ne": (22.5, 67.5),
    "e": (67.5, 112.5),
    "se": (112.5, 157.5),
    "s": (157.5, 202.5),
    "sw": (202.5, 247.5),
    "w": (247.5, 292.5),
    "nw": (292.5, 337.5),
}


def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    return (series - 32.0) * (5.0 / 9.0)


def load_observations(input_csv: str | Path, config: ProjectConfig) -> pd.DataFrame:
    """Read only the configured station from the large ASOS CSV."""
    wanted = [
        "station",
        "valid",
        *NUMERIC_COLUMNS,
        *CLOUD_COLUMNS,
        "wxcodes",
        "metar",
    ]
    if config.prefer_duckdb and Path(config.input_db).exists() and Path(input_csv) == config.input_csv:
        observations = read_station_observations_from_duckdb(
            config.input_db,
            config.station,
            wanted,
        )
        return _finalize_observations(observations, config)

    lazy = pl.scan_csv(
        input_csv,
        null_values=["null", "M", ""],
        infer_schema_length=10_000,
        ignore_errors=True,
    )
    schema_names = set(lazy.collect_schema().names())
    selected = [column for column in wanted if column in schema_names]

    numeric_present = [column for column in NUMERIC_COLUMNS if column in selected]
    observations = (
        lazy.select(selected)
        .filter(pl.col("station") == config.station)
        .with_columns(
            [
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in numeric_present
            ]
        )
        .with_columns(
            pl.col("valid")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M", strict=False)
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(config.timezone)
            .alias("valid_local")
        )
        .drop_nulls(["valid_local"])
        .sort("valid_local")
        .collect()
    )
    return observations.to_pandas()


def _finalize_observations(observations: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    frame = observations.copy()
    numeric_present = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    for column in numeric_present:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["valid_local"] = (
        pd.to_datetime(frame["valid"], format="%Y-%m-%d %H:%M", errors="coerce", utc=True)
        .dt.tz_convert(config.timezone)
    )
    frame = frame.dropna(subset=["valid_local"]).sort_values("valid_local").reset_index(drop=True)
    return frame


def make_daily_dataset(observations: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    data = observations.copy()
    data["valid_local"] = pd.to_datetime(data["valid_local"])
    data["local_date"] = data["valid_local"].dt.date.astype(str)
    data["local_minutes"] = data["valid_local"].dt.hour * 60 + data["valid_local"].dt.minute
    data = data.sort_values("valid_local")

    target = (
        data.groupby("local_date", as_index=False)
        .agg(
            tmax_f=("tmpf", "max"),
            tmin_f=("tmpf", "min"),
            obs_count_full_day=("tmpf", "count"),
            last_full_day_minute=("local_minutes", "max"),
        )
        .sort_values("local_date")
    )
    tmax_times = _tmax_time_features(data)
    target = target.merge(tmax_times, on="local_date", how="left")
    complete_mask = target["last_full_day_minute"] >= config.complete_day_min_minutes
    target["target_complete"] = complete_mask.astype(int)
    incomplete_columns = ["tmax_f", "tmin_f"]
    target.loc[~complete_mask, incomplete_columns] = np.nan
    target["tmax_c"] = fahrenheit_to_celsius(target["tmax_f"])
    target["tmin_c"] = fahrenheit_to_celsius(target["tmin_f"])

    morning = data[data["local_minutes"] <= config.cutoff_minutes].copy()
    if morning.empty:
        raise ValueError(f"No {config.station} observations found before cutoff {config.cutoff_local}.")

    numeric_present = [column for column in NUMERIC_COLUMNS if column in morning.columns]
    aggregations = {}
    for column in numeric_present:
        aggregations[f"{column}_mean_to_cutoff"] = (column, "mean")
        aggregations[f"{column}_min_to_cutoff"] = (column, "min")
        aggregations[f"{column}_max_to_cutoff"] = (column, "max")
        aggregations[f"{column}_last_to_cutoff"] = (column, "last")

    features = morning.groupby("local_date", as_index=False).agg(
        obs_count_to_cutoff=("tmpf", "count"),
        last_observation_minute=("local_minutes", "last"),
        **aggregations,
    )

    features["tmpc_last_to_cutoff"] = fahrenheit_to_celsius(features["tmpf_last_to_cutoff"])
    features["dwpc_last_to_cutoff"] = fahrenheit_to_celsius(features["dwpf_last_to_cutoff"])
    features["tmpc_max_to_cutoff"] = fahrenheit_to_celsius(features["tmpf_max_to_cutoff"])
    features["tmpc_min_to_cutoff"] = fahrenheit_to_celsius(features["tmpf_min_to_cutoff"])

    phase_features = _phase_plateau_features(morning, config.cutoff_minutes)
    cloud_features = _cloud_features(morning)
    weather_features = _weather_code_features(morning)
    wind_features = _wind_direction_features(morning)
    suppression_features = _weather_suppression_features(morning, config.cutoff_minutes)
    features = features.merge(phase_features, on="local_date", how="left")
    features = features.merge(cloud_features, on="local_date", how="left")
    features = features.merge(weather_features, on="local_date", how="left")
    features = features.merge(wind_features, on="local_date", how="left")
    features = features.merge(suppression_features, on="local_date", how="left")

    dataset = features.merge(target, on="local_date", how="left").sort_values("local_date")
    dataset["target_complete"] = dataset["target_complete"].fillna(0).astype(int)
    dataset = _add_calendar_features(dataset)
    dataset = _add_lag_features(dataset)
    dataset = _add_remaining_heat_climatology(dataset)
    dataset = _add_openmeteo_features(dataset, config)
    dataset = _add_last3_regime_features(dataset)
    dataset = dataset.dropna(subset=["tmax_prev1_c", "remaining_heat_climo_global_c"]).reset_index(
        drop=True
    )
    return dataset


def _tmax_time_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[["local_date", "local_minutes", "tmpf"]].dropna(subset=["tmpf"]).copy()
    if frame.empty:
        return pd.DataFrame({"local_date": data["local_date"].unique(), "tmax_minute": np.nan})
    idx = frame.groupby("local_date")["tmpf"].idxmax()
    return frame.loc[idx, ["local_date", "local_minutes"]].rename(
        columns={"local_minutes": "tmax_minute"}
    )


def _phase_plateau_features(morning: pd.DataFrame, cutoff_minutes: int) -> pd.DataFrame:
    frame = morning[["local_date", "local_minutes", "tmpf"]].dropna(subset=["tmpf"]).copy()
    if frame.empty:
        return pd.DataFrame({"local_date": morning["local_date"].unique()})
    frame["tmpc"] = fahrenheit_to_celsius(frame["tmpf"])
    rows = []
    for local_date, group in frame.sort_values("local_minutes").groupby("local_date"):
        last = group.iloc[-1]
        last_temp = float(last["tmpc"])
        max_temp = float(group["tmpc"].max())
        temp_at_06 = _temp_at_or_before(group, 6 * 60)
        temp_06_available = int((group["local_minutes"] <= 6 * 60).any())
        max_rows = group[np.isclose(group["tmpc"], max_temp, atol=0.05)]
        latest_max_minute = int(max_rows["local_minutes"].max())
        last_2h = group[group["local_minutes"] >= cutoff_minutes - 120]
        row = {
            "local_date": local_date,
            "last_temp_equals_observed_max": int(abs(last_temp - max_temp) < 0.05),
            "minutes_since_observed_max": int(last["local_minutes"] - latest_max_minute),
            "observed_max_is_latest_observation": int(int(last["local_minutes"]) == latest_max_minute),
            "observed_max_count_so_far": int(len(max_rows)),
            "duration_within_1c_of_observed_max": int((group["tmpc"] >= max_temp - 1.0).sum() * 30),
            "duration_within_2c_of_observed_max": int((group["tmpc"] >= max_temp - 2.0).sum() * 30),
            "temp_range_last_2h": float(last_2h["tmpc"].max() - last_2h["tmpc"].min()),
            "temp_std_last_2h": float(last_2h["tmpc"].std(ddof=0)) if len(last_2h) > 1 else 0.0,
            "temp_flat_duration_last_2h": _flat_duration_minutes(last_2h, last_temp),
            "tmpc_at_or_before_06": temp_at_06,
            "temp_06_observation_available": temp_06_available,
            "temp_rise_since_06_c": last_temp - temp_at_06,
            "observed_max_gain_since_06_c": max_temp - temp_at_06,
        }
        for minutes in (30, 60, 90, 120):
            row[f"temp_rise_last_{minutes}m"] = last_temp - _temp_at_or_before(
                group,
                cutoff_minutes - minutes,
            )
        for hour in (9, 10):
            key = f"temp_rise_since_{hour:02d}"
            row[key] = last_temp - _temp_at_or_before(group, hour * 60)
        rows.append(row)
    return pd.DataFrame(rows)


def _flat_duration_minutes(last_2h: pd.DataFrame, last_temp_c: float) -> int:
    if last_2h.empty:
        return 0
    duration = 0
    for _, row in last_2h.sort_values("local_minutes", ascending=False).iterrows():
        if abs(float(row["tmpc"]) - last_temp_c) > 0.5:
            break
        duration += 30
    return duration


def _temp_at_or_before(group: pd.DataFrame, minute: int) -> float:
    candidates = group[group["local_minutes"] <= minute]
    if candidates.empty:
        return float(group.iloc[0]["tmpc"])
    return float(candidates.iloc[-1]["tmpc"])


def _cloud_features(morning: pd.DataFrame) -> pd.DataFrame:
    frame = morning[["local_date", *[column for column in CLOUD_COLUMNS if column in morning.columns]]].copy()
    cover_columns = []
    for column in CLOUD_COLUMNS:
        if column not in frame.columns:
            continue
        cover_column = f"{column}_cover"
        frame[cover_column] = frame[column].map(CLOUD_COVER)
        cover_columns.append(cover_column)

    if cover_columns:
        frame["max_cloud_cover_to_cutoff"] = frame[cover_columns].max(axis=1)
        frame["last_cloud_cover_to_cutoff"] = frame[cover_columns].bfill(axis=1).iloc[:, 0]
        frame["first_cloud_cover_to_cutoff"] = frame[cover_columns].bfill(axis=1).iloc[:, 0]
    else:
        frame["max_cloud_cover_to_cutoff"] = np.nan
        frame["last_cloud_cover_to_cutoff"] = np.nan
        frame["first_cloud_cover_to_cutoff"] = np.nan

    ceiling_columns = [column for column in ["skyl1", "skyl2", "skyl3", "skyl4"] if column in morning.columns]
    if ceiling_columns:
        frame["lowest_ceiling_ft_to_cutoff"] = morning[ceiling_columns].min(axis=1)
        frame["first_lowest_ceiling_ft_to_cutoff"] = frame["lowest_ceiling_ft_to_cutoff"]
        frame["last_lowest_ceiling_ft_to_cutoff"] = frame["lowest_ceiling_ft_to_cutoff"]
    else:
        frame["lowest_ceiling_ft_to_cutoff"] = np.nan
        frame["first_lowest_ceiling_ft_to_cutoff"] = np.nan
        frame["last_lowest_ceiling_ft_to_cutoff"] = np.nan

    grouped = frame.groupby("local_date", as_index=False).agg(
        max_cloud_cover_to_cutoff=("max_cloud_cover_to_cutoff", "max"),
        first_cloud_cover_to_cutoff=("first_cloud_cover_to_cutoff", "first"),
        last_cloud_cover_to_cutoff=("last_cloud_cover_to_cutoff", "last"),
        lowest_ceiling_ft_to_cutoff=("lowest_ceiling_ft_to_cutoff", "min"),
        first_lowest_ceiling_ft_to_cutoff=("first_lowest_ceiling_ft_to_cutoff", "first"),
        last_lowest_ceiling_ft_to_cutoff=("last_lowest_ceiling_ft_to_cutoff", "last"),
    )
    grouped["cloud_cover_change_to_cutoff"] = (
        grouped["last_cloud_cover_to_cutoff"] - grouped["first_cloud_cover_to_cutoff"]
    )
    grouped["cloud_clearing_to_cutoff"] = (
        grouped["first_cloud_cover_to_cutoff"] - grouped["last_cloud_cover_to_cutoff"]
    ).clip(lower=0)
    grouped["cloud_increasing_to_cutoff"] = (
        grouped["last_cloud_cover_to_cutoff"] - grouped["first_cloud_cover_to_cutoff"]
    ).clip(lower=0)
    grouped["ceiling_lift_ft_to_cutoff"] = (
        grouped["last_lowest_ceiling_ft_to_cutoff"]
        - grouped["first_lowest_ceiling_ft_to_cutoff"]
    )
    return grouped


def _weather_code_features(morning: pd.DataFrame) -> pd.DataFrame:
    if "wxcodes" not in morning.columns:
        return pd.DataFrame(
            {
                "local_date": morning["local_date"].unique(),
                "fog_observed_to_cutoff": 0,
                "fog_first_to_cutoff": 0,
                "fog_last_to_cutoff": 0,
                "fog_cleared_to_cutoff": 0,
                "fog_developed_to_cutoff": 0,
                "precip_observed_to_cutoff": 0,
            }
        )

    weather = morning[["local_date", "wxcodes"]].copy()
    codes = weather["wxcodes"].fillna("").astype(str)
    weather["fog_observed_to_cutoff"] = codes.str.contains("|".join(FOG_CODES), regex=True).astype(int)
    weather["precip_observed_to_cutoff"] = codes.str.contains("|".join(PRECIP_CODES), regex=True).astype(int)
    grouped = weather.groupby("local_date", as_index=False).agg(
        fog_observed_to_cutoff=("fog_observed_to_cutoff", "max"),
        fog_first_to_cutoff=("fog_observed_to_cutoff", "first"),
        fog_last_to_cutoff=("fog_observed_to_cutoff", "last"),
        precip_observed_to_cutoff=("precip_observed_to_cutoff", "max"),
    )
    grouped["fog_cleared_to_cutoff"] = (
        (grouped["fog_first_to_cutoff"] == 1) & (grouped["fog_last_to_cutoff"] == 0)
    ).astype(int)
    grouped["fog_developed_to_cutoff"] = (
        (grouped["fog_first_to_cutoff"] == 0) & (grouped["fog_last_to_cutoff"] == 1)
    ).astype(int)
    return grouped


def _wind_direction_features(morning: pd.DataFrame) -> pd.DataFrame:
    if "drct" not in morning.columns:
        return pd.DataFrame({"local_date": morning["local_date"].unique()})

    wind = morning[["local_date", "drct"]].copy()
    direction = wind["drct"].where(wind["drct"].between(0.0, 360.0)) % 360.0
    radians = np.deg2rad(direction)
    wind["wind_dir_sin_to_cutoff"] = np.sin(radians)
    wind["wind_dir_cos_to_cutoff"] = np.cos(radians)
    wind["wind_dir_sin_last_to_cutoff"] = wind["wind_dir_sin_to_cutoff"]
    wind["wind_dir_cos_last_to_cutoff"] = wind["wind_dir_cos_to_cutoff"]

    for regime, sector in WIND_REGIMES.items():
        column = f"wind_regime_{regime}_last_to_cutoff"
        if len(sector) == 4:
            lower_1, upper_1, lower_2, upper_2 = sector
            wind[column] = (
                direction.between(lower_1, upper_1, inclusive="left")
                | direction.between(lower_2, upper_2, inclusive="left")
            ).astype(int)
        else:
            lower, upper = sector
            wind[column] = direction.between(lower, upper, inclusive="left").astype(int)

    aggregations = {
        "wind_dir_sin_mean_to_cutoff": ("wind_dir_sin_to_cutoff", "mean"),
        "wind_dir_cos_mean_to_cutoff": ("wind_dir_cos_to_cutoff", "mean"),
        "wind_dir_sin_last_to_cutoff": ("wind_dir_sin_last_to_cutoff", "last"),
        "wind_dir_cos_last_to_cutoff": ("wind_dir_cos_last_to_cutoff", "last"),
    }
    for regime in WIND_REGIMES:
        column = f"wind_regime_{regime}_last_to_cutoff"
        aggregations[column] = (column, "last")

    return wind.groupby("local_date", as_index=False).agg(**aggregations)


def _weather_suppression_features(morning: pd.DataFrame, cutoff_minutes: int) -> pd.DataFrame:
    wanted = ["local_date", "local_minutes", "wxcodes", "vsby", "skyl1", "skyl2", "skyl3", "skyl4"]
    present = [column for column in wanted if column in morning.columns]
    if "local_date" not in present or "local_minutes" not in present:
        return pd.DataFrame({"local_date": morning["local_date"].unique()})

    frame = morning[present].copy()
    frame = frame[frame["local_minutes"] >= cutoff_minutes - 120]
    if frame.empty:
        return pd.DataFrame({"local_date": morning["local_date"].unique()})

    codes = frame.get("wxcodes", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["rain_flag"] = codes.str.contains("|".join(PRECIP_CODES), regex=True).astype(int)
    frame["visibility_low_flag"] = (
        pd.to_numeric(frame.get("vsby", pd.Series(np.nan, index=frame.index)), errors="coerce") < 5.0
    ).astype(int)

    ceiling_columns = [column for column in ["skyl1", "skyl2", "skyl3", "skyl4"] if column in frame.columns]
    if ceiling_columns:
        ceiling = frame[ceiling_columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
    else:
        ceiling = pd.Series(np.nan, index=frame.index)
    frame["ceiling_ft"] = ceiling
    frame["low_cloud_flag"] = (ceiling <= 3000.0).astype(int)
    frame["mvfr_or_worse_flag"] = (
        (frame["low_cloud_flag"] == 1) | (frame["visibility_low_flag"] == 1)
    ).astype(int)

    grouped = frame.sort_values("local_minutes").groupby("local_date", as_index=False).agg(
        rain_seen_last_2h=("rain_flag", "max"),
        rain_seen_at_cutoff=("rain_flag", "last"),
        low_cloud_seen_last_2h=("low_cloud_flag", "max"),
        ceiling_min_last_2h=("ceiling_ft", "min"),
        visibility_min_last_2h=("vsby", "min"),
        visibility_low_last_2h=("visibility_low_flag", "max"),
        mvfr_or_worse_last_2h=("mvfr_or_worse_flag", "max"),
    )
    grouped["weather_suppression_score"] = (
        grouped["rain_seen_last_2h"].fillna(0.0)
        + grouped["low_cloud_seen_last_2h"].fillna(0.0)
        + grouped["mvfr_or_worse_last_2h"].fillna(0.0) * 0.7
        + grouped["visibility_low_last_2h"].fillna(0.0) * 0.5
    )
    return grouped


def _add_calendar_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    dates = pd.to_datetime(frame["local_date"])
    day_of_year = dates.dt.dayofyear
    frame["dayofyear_sin"] = np.sin(2 * math.pi * day_of_year / 366.0)
    frame["dayofyear_cos"] = np.cos(2 * math.pi * day_of_year / 366.0)
    frame["month"] = dates.dt.month
    return frame


def _add_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy().sort_values("local_date")
    frame["tmax_prev1_c"] = _previous_non_null(frame["tmax_c"], lag=1)
    frame["tmin_prev1_c"] = _previous_non_null(frame["tmin_c"], lag=1)
    frame["tmpc_09_prev1"] = frame["tmpc_last_to_cutoff"].shift(1)
    for lag in (1, 2, 3):
        frame[f"tmax_lag_{lag}_c"] = _previous_non_null(frame["tmax_c"], lag=lag)
    for window in (3, 7, 14):
        frame[f"tmax_roll{window}_mean_c"] = _previous_non_null_rolling_mean(
            frame["tmax_c"],
            window,
        )
        frame[f"tmin_roll{window}_mean_c"] = _previous_non_null_rolling_mean(
            frame["tmin_c"],
            window,
        )
    return frame


def _previous_non_null(series: pd.Series, lag: int) -> pd.Series:
    history: list[float] = []
    values = []
    for value in series:
        values.append(history[-lag] if len(history) >= lag else np.nan)
        if pd.notna(value):
            history.append(float(value))
    return pd.Series(values, index=series.index)


def _previous_non_null_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    history: list[float] = []
    values = []
    for value in series:
        values.append(float(np.mean(history[-window:])) if history else np.nan)
        if pd.notna(value):
            history.append(float(value))
    return pd.Series(values, index=series.index)


def _add_remaining_heat_climatology(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy().sort_values("local_date").reset_index(drop=True)
    frame["remaining_heat_c"] = frame["tmax_c"] - frame["tmpc_last_to_cutoff"]
    frame["tmax_climo_global_c"] = frame["tmax_c"].shift(1).expanding(min_periods=1).median()
    frame["tmax_climo_month_c"] = frame.groupby("month", group_keys=False)["tmax_c"].apply(
        lambda series: series.shift(1).expanding(min_periods=1).median()
    )
    frame["tmax_climo_month_c"] = frame["tmax_climo_month_c"].fillna(
        frame["tmax_climo_global_c"]
    )
    frame["month_tmax_p50_c"] = frame["tmax_climo_month_c"]
    frame["month_tmax_p90_c"] = frame.groupby("month", group_keys=False)["tmax_c"].apply(
        lambda series: series.shift(1).expanding(min_periods=1).quantile(0.90)
    )
    frame["month_tmax_p90_c"] = frame["month_tmax_p90_c"].fillna(frame["tmax_climo_global_c"])
    frame["month_median_tmax_minute"] = frame.groupby("month", group_keys=False)[
        "tmax_minute"
    ].apply(lambda series: series.shift(1).expanding(min_periods=1).median())
    frame["month_median_tmax_minute"] = frame["month_median_tmax_minute"].fillna(
        frame["tmax_minute"].shift(1).expanding(min_periods=1).median()
    )
    frame["cutoff_minutes_before_monthly_median_tmax_time"] = (
        frame["month_median_tmax_minute"] - frame["last_observation_minute"]
    )
    frame["cutoff_before_typical_peak"] = (
        frame["cutoff_minutes_before_monthly_median_tmax_time"] > 30
    ).astype(int)
    frame["false_plateau_candidate"] = (
        (frame.get("temp_flat_duration_last_2h", 0) >= 90)
        & (frame.get("weather_suppression_score", 0.0) >= 1.0)
        & (frame["cutoff_before_typical_peak"] == 1)
        & (frame.get("last_temp_equals_observed_max", 0) == 1)
    ).astype(int)
    frame["remaining_heat_climo_global_c"] = (
        frame["remaining_heat_c"].shift(1).expanding(min_periods=1).median()
    )
    frame["remaining_heat_climo_month_c"] = frame.groupby("month", group_keys=False)[
        "remaining_heat_c"
    ].apply(lambda series: series.shift(1).expanding(min_periods=1).median())
    frame["remaining_heat_climo_month_c"] = frame["remaining_heat_climo_month_c"].fillna(
        frame["remaining_heat_climo_global_c"]
    )
    frame["expected_tmax_from_cutoff_c"] = (
        frame["tmpc_last_to_cutoff"] + frame["remaining_heat_climo_month_c"]
    )
    frame["month_remaining_heat_p50_by_cutoff"] = frame["remaining_heat_climo_month_c"]
    frame["month_remaining_heat_p90_by_cutoff"] = frame.groupby(
        ["month"],
        group_keys=False,
    )["remaining_heat_c"].apply(lambda series: series.shift(1).expanding(min_periods=1).quantile(0.90))
    frame["month_remaining_heat_p90_by_cutoff"] = frame[
        "month_remaining_heat_p90_by_cutoff"
    ].fillna(frame["remaining_heat_climo_global_c"])
    return frame.drop(columns=["remaining_heat_c"])


def _add_openmeteo_features(dataset: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    forecast = load_openmeteo_features_for_dates(
        config.openmeteo_history_csv,
        config.openmeteo_live_csv_pattern,
        dataset["local_date"],
        config.openmeteo_history_json,
        config.openmeteo_live_json_pattern,
    )
    if forecast is None:
        return dataset

    frame = dataset.merge(forecast, on="local_date", how="left")
    frame["openmeteo_expected_remaining_heat_c"] = (
        frame["openmeteo_tmax_c"] - frame["tmpc_max_to_cutoff"]
    ).clip(lower=0.0)
    frame["openmeteo_tmax_minus_observed_max_c"] = (
        frame["openmeteo_tmax_c"] - frame["tmpc_max_to_cutoff"]
    )
    frame["openmeteo_tmax_minus_last_temp_c"] = (
        frame["openmeteo_tmax_c"] - frame["tmpc_last_to_cutoff"]
    )
    frame["openmeteo_tmax_minus_climo_c"] = (
        frame["openmeteo_tmax_c"] - frame["tmax_climo_month_c"]
    )
    return frame


def _add_last3_regime_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy().sort_values("local_date").reset_index(drop=True)
    for lag in (1, 2, 3):
        frame[f"tmax_lag_{lag}_anomaly_c"] = frame[f"tmax_lag_{lag}_c"] - frame[
            "tmax_climo_month_c"
        ]
    anomaly_columns = [f"tmax_lag_{lag}_anomaly_c" for lag in (1, 2, 3)]
    frame["last3_tmax_anomaly_mean_c"] = frame[anomaly_columns].mean(axis=1)
    frame["last3_tmax_anomaly_trend_c"] = (
        frame["tmax_lag_1_anomaly_c"] - frame["tmax_lag_3_anomaly_c"]
    )

    frame["last3_temp_to_cutoff_mean_c"] = (
        frame["tmpc_last_to_cutoff"].shift(1).rolling(3, min_periods=1).mean()
    )
    frame["last3_max_to_cutoff_mean_c"] = (
        frame["tmpc_max_to_cutoff"].shift(1).rolling(3, min_periods=1).mean()
    )
    frame["last3_warming_rate_mean_c"] = (
        frame["temp_rise_last_120m"].shift(1).rolling(3, min_periods=1).mean()
    )
    frame["today_vs_last3_temp_to_cutoff_diff_c"] = (
        frame["tmpc_last_to_cutoff"] - frame["last3_temp_to_cutoff_mean_c"]
    )
    frame["today_vs_last3_max_to_cutoff_diff_c"] = (
        frame["tmpc_max_to_cutoff"] - frame["last3_max_to_cutoff_mean_c"]
    )
    frame["today_vs_last3_warming_rate_diff_c"] = (
        frame["temp_rise_last_120m"] - frame["last3_warming_rate_mean_c"]
    )
    frame["regime_break_score"] = (
        frame["today_vs_last3_temp_to_cutoff_diff_c"].abs().fillna(0.0)
        + frame["today_vs_last3_max_to_cutoff_diff_c"].abs().fillna(0.0)
        + frame["today_vs_last3_warming_rate_diff_c"].abs().fillna(0.0) * 0.5
    )
    frame["regime_break_cooler_than_recent"] = (
        frame["today_vs_last3_temp_to_cutoff_diff_c"] <= -2.0
    ).astype(int)
    frame["regime_break_warmer_than_recent"] = (
        frame["today_vs_last3_temp_to_cutoff_diff_c"] >= 2.0
    ).astype(int)
    frame["regime_break_similar_to_recent"] = (
        (frame["regime_break_cooler_than_recent"] == 0)
        & (frame["regime_break_warmer_than_recent"] == 0)
    ).astype(int)
    return frame
