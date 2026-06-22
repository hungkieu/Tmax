from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OPENMETEO_BASE_FEATURE_COLUMNS = (
    "openmeteo_tmax_c",
    "openmeteo_weather_code",
    "openmeteo_precipitation_sum_mm",
    "openmeteo_precipitation_hours",
    "openmeteo_rain_sum_mm",
    "openmeteo_wind_speed_10m_max_kmh",
    "openmeteo_wind_gusts_10m_max_kmh",
    "openmeteo_precipitation_flag",
    "openmeteo_rain_flag",
)


def load_openmeteo_daily(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    rows = source.read_text(encoding="utf-8-sig").splitlines()
    try:
        header_index = next(index for index, row in enumerate(rows) if row.startswith("time,"))
    except StopIteration as exc:
        raise ValueError(f"Open-Meteo CSV does not contain a time header: {source}") from exc

    raw = pd.read_csv(StringIO("\n".join(rows[header_index:])))
    output = pd.DataFrame()
    output["local_date"] = pd.to_datetime(raw["time"], errors="coerce").dt.date.astype("string")
    output["openmeteo_tmax_c"] = _numeric_column(raw, "temperature2mmax")
    output["openmeteo_weather_code"] = _numeric_column(raw, "weathercode")
    output["openmeteo_precipitation_sum_mm"] = _numeric_column(raw, "precipitationsum")
    output["openmeteo_precipitation_hours"] = _numeric_column(raw, "precipitationhours")
    output["openmeteo_rain_sum_mm"] = _numeric_column(raw, "rainsum")
    output["openmeteo_wind_speed_10m_max_kmh"] = _numeric_column(raw, "windspeed10mmax")
    output["openmeteo_wind_gusts_10m_max_kmh"] = _numeric_column(raw, "windgusts10mmax")
    output["openmeteo_precipitation_flag"] = (
        output["openmeteo_precipitation_sum_mm"].fillna(0.0) > 0.0
    ).astype(int)
    output["openmeteo_rain_flag"] = (output["openmeteo_rain_sum_mm"].fillna(0.0) > 0.0).astype(int)
    output = output.dropna(subset=["local_date"]).sort_values("local_date")
    return output.drop_duplicates(subset=["local_date"], keep="last").reset_index(drop=True)


def load_openmeteo_features_for_dates(
    history_csv: str | Path | None,
    live_csv_pattern: str | None,
    local_dates: Iterable[str],
) -> pd.DataFrame | None:
    dates = sorted({str(date) for date in local_dates})
    if not history_csv and not live_csv_pattern:
        return None

    requested = pd.DataFrame({"local_date": dates})
    frames = []
    if history_csv and Path(history_csv).exists():
        frames.append(load_openmeteo_daily(history_csv))
    if live_csv_pattern:
        for local_date in dates:
            live_path = Path(live_csv_pattern.format(date=local_date))
            if live_path.exists():
                frames.append(load_openmeteo_daily(live_path))

    if frames:
        forecast = (
            pd.concat(frames, ignore_index=True)
            .sort_values("local_date")
            .drop_duplicates(subset=["local_date"], keep="last")
        )
        forecast = forecast[forecast["local_date"].isin(dates)]
        output = requested.merge(forecast, on="local_date", how="left")
    else:
        output = requested.copy()
        for column in OPENMETEO_BASE_FEATURE_COLUMNS:
            output[column] = np.nan
    return output


def _numeric_column(frame: pd.DataFrame, needle: str) -> pd.Series:
    column = _find_column(frame.columns, needle)
    if column is None:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _find_column(columns: Iterable[str], needle: str) -> str | None:
    for column in columns:
        if needle in _normalize_column(column):
            return column
    return None


def _normalize_column(column: str) -> str:
    return "".join(character for character in column.lower() if character.isalnum())
