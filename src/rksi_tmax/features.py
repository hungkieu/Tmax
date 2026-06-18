from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from rksi_tmax.config import ProjectConfig


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
    lazy = pl.scan_csv(
        input_csv,
        null_values=["null", "M", ""],
        infer_schema_length=10_000,
        ignore_errors=True,
    )
    schema_names = set(lazy.collect_schema().names())
    wanted = [
        "station",
        "valid",
        *NUMERIC_COLUMNS,
        *CLOUD_COLUMNS,
        "wxcodes",
        "metar",
    ]
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

    cloud_features = _cloud_features(morning)
    weather_features = _weather_code_features(morning)
    wind_features = _wind_direction_features(morning)
    features = features.merge(cloud_features, on="local_date", how="left")
    features = features.merge(weather_features, on="local_date", how="left")
    features = features.merge(wind_features, on="local_date", how="left")

    dataset = features.merge(target, on="local_date", how="left").sort_values("local_date")
    dataset["target_complete"] = dataset["target_complete"].fillna(0).astype(int)
    dataset = _add_calendar_features(dataset)
    dataset = _add_lag_features(dataset)
    dataset = _add_remaining_heat_climatology(dataset)
    dataset = dataset.dropna(subset=["tmax_prev1_c", "remaining_heat_climo_global_c"]).reset_index(
        drop=True
    )
    return dataset


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
    frame["tmax_prev1_c"] = frame["tmax_c"].shift(1)
    frame["tmin_prev1_c"] = frame["tmin_c"].shift(1)
    frame["tmpc_09_prev1"] = frame["tmpc_last_to_cutoff"].shift(1)
    for window in (3, 7, 14):
        shifted_tmax = frame["tmax_c"].shift(1)
        shifted_tmin = frame["tmin_c"].shift(1)
        frame[f"tmax_roll{window}_mean_c"] = shifted_tmax.rolling(window, min_periods=1).mean()
        frame[f"tmin_roll{window}_mean_c"] = shifted_tmin.rolling(window, min_periods=1).mean()
    return frame


def _add_remaining_heat_climatology(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy().sort_values("local_date").reset_index(drop=True)
    frame["remaining_heat_c"] = frame["tmax_c"] - frame["tmpc_last_to_cutoff"]
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
    return frame.drop(columns=["remaining_heat_c"])
