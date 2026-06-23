from __future__ import annotations

import json
import ssl
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

import certifi
import numpy as np
import pandas as pd


HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARIABLES = (
    "weather_code",
    "temperature_2m_max",
    "rain_sum",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
)
HOURLY_VARIABLES = (
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
    "openmeteo_hourly_temp_max_c",
    "openmeteo_hourly_temp_min_c",
    "openmeteo_hourly_temp_mean_c",
    "openmeteo_hourly_temp_peak_hour",
    "openmeteo_hourly_cloud_cover_mean_pct",
    "openmeteo_hourly_cloud_cover_max_pct",
    "openmeteo_hourly_visibility_min_m",
    "openmeteo_hourly_rain_sum_mm",
    "openmeteo_hourly_precipitation_sum_mm",
    "openmeteo_hourly_precipitation_probability_max_pct",
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


def load_openmeteo_json(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    daily = _daily_features_from_payload(payload)
    hourly = _hourly_features_from_payload(payload)
    if daily.empty:
        return hourly
    if hourly.empty:
        return daily
    return (
        daily.merge(hourly, on="local_date", how="outer")
        .sort_values("local_date")
        .drop_duplicates(subset=["local_date"], keep="last")
        .reset_index(drop=True)
    )


def fetch_openmeteo_forecast(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    *,
    historical: bool,
    timezone: str = "GMT",
    timeout_seconds: int = 60,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_VARIABLES),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": timezone,
        }
    )
    base_url = HISTORICAL_FORECAST_URL if historical else FORECAST_URL
    url = f"{base_url}?{query}"
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(url, timeout=timeout_seconds, context=context) as response:  # noqa: S310
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        reason = body
        try:
            payload = json.loads(body)
            reason = str(payload.get("reason", payload))
        except json.JSONDecodeError:
            pass
        raise ValueError(f"Open-Meteo API HTTP {exc.code}: {reason}; url={url}") from exc
    payload = json.loads(body)
    if payload.get("error"):
        raise ValueError(f"Open-Meteo API error: {payload.get('reason', payload)}")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dates = payload.get("daily", {}).get("time", [])
    return {
        "url": url,
        "output_path": str(output),
        "historical": historical,
        "start_date": start_date,
        "end_date": end_date,
        "days": len(dates),
    }


def ensure_openmeteo_training_data(
    history_json: str | Path | None,
    latitude: float | None,
    longitude: float | None,
    start_date: str,
    end_date: str,
    *,
    timezone: str = "GMT",
    force: bool = False,
) -> dict[str, object] | None:
    if not history_json or latitude is None or longitude is None:
        return None
    path = Path(history_json)
    if not force and _json_covers_date_range(path, start_date, end_date):
        return {
            "fetched": False,
            "output_path": str(path),
            "start_date": start_date,
            "end_date": end_date,
        }
    result = fetch_openmeteo_forecast(
        latitude,
        longitude,
        start_date,
        end_date,
        path,
        historical=True,
        timezone=timezone,
    )
    result["fetched"] = True
    return result


def ensure_openmeteo_live_data(
    live_json_pattern: str | None,
    latitude: float | None,
    longitude: float | None,
    local_date: str,
    *,
    timezone: str = "GMT",
    force: bool = False,
) -> dict[str, object] | None:
    if not live_json_pattern or latitude is None or longitude is None:
        return None
    path = Path(live_json_pattern.format(date=local_date))
    if path.exists() and not force:
        return {"fetched": False, "output_path": str(path), "date": local_date}
    result = fetch_openmeteo_forecast(
        latitude,
        longitude,
        local_date,
        local_date,
        path,
        historical=False,
        timezone=timezone,
    )
    result["fetched"] = True
    return result


def load_openmeteo_features_for_dates(
    history_csv: str | Path | None,
    live_csv_pattern: str | None,
    local_dates: Iterable[str],
    history_json: str | Path | None = None,
    live_json_pattern: str | None = None,
) -> pd.DataFrame | None:
    dates = sorted({str(date) for date in local_dates})
    if not history_csv and not live_csv_pattern and not history_json and not live_json_pattern:
        return None

    requested = pd.DataFrame({"local_date": dates})
    frames = []
    if history_csv and Path(history_csv).exists():
        frames.append(load_openmeteo_daily(history_csv))
    if history_json and Path(history_json).exists():
        frames.append(load_openmeteo_json(history_json))
    if live_csv_pattern:
        for local_date in dates:
            live_path = Path(live_csv_pattern.format(date=local_date))
            if live_path.exists():
                frames.append(load_openmeteo_daily(live_path))
    if live_json_pattern:
        for local_date in dates:
            live_path = Path(live_json_pattern.format(date=local_date))
            if live_path.exists():
                frames.append(load_openmeteo_json(live_path))

    if frames:
        forecast = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["local_date"], keep="last")
            .sort_values("local_date")
        )
        forecast = forecast[forecast["local_date"].isin(dates)]
        output = requested.merge(forecast, on="local_date", how="left")
    else:
        output = requested.copy()
        for column in OPENMETEO_BASE_FEATURE_COLUMNS:
            output[column] = np.nan
    return output


def openmeteo_date_range(path: str | Path) -> tuple[str, str] | None:
    frame = load_openmeteo_json(path)
    if frame.empty:
        return None
    dates = frame["local_date"].dropna().astype(str)
    if dates.empty:
        return None
    return str(dates.min()), str(dates.max())


def openmeteo_cache_has_date(
    local_date: str,
    history_json: str | Path | None = None,
    live_json_pattern: str | None = None,
    history_csv: str | Path | None = None,
    live_csv_pattern: str | None = None,
) -> bool:
    frames = []
    if history_json and Path(history_json).exists():
        frames.append(load_openmeteo_json(history_json))
    if live_json_pattern:
        live_json = Path(live_json_pattern.format(date=local_date))
        if live_json.exists():
            frames.append(load_openmeteo_json(live_json))
    if history_csv and Path(history_csv).exists():
        frames.append(load_openmeteo_daily(history_csv))
    if live_csv_pattern:
        live_csv = Path(live_csv_pattern.format(date=local_date))
        if live_csv.exists():
            frames.append(load_openmeteo_daily(live_csv))
    if not frames:
        return False
    available = pd.concat(frames, ignore_index=True)["local_date"].dropna().astype(str)
    return local_date in set(available)


def _json_covers_date_range(path: Path, start_date: str, end_date: str) -> bool:
    if not path.exists():
        return False
    try:
        date_range = openmeteo_date_range(path)
    except Exception:
        return False
    if date_range is None:
        return False
    available_start, available_end = date_range
    return available_start <= start_date and available_end >= end_date


def _daily_features_from_payload(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return pd.DataFrame()
    raw = pd.DataFrame(
        {"local_date": pd.Series(pd.to_datetime(times, errors="coerce")).dt.date.astype("string")}
    )
    output = pd.DataFrame()
    output["local_date"] = raw["local_date"]
    output["openmeteo_tmax_c"] = _numeric_payload_column(daily, "temperature_2m_max", raw.index)
    output["openmeteo_weather_code"] = _numeric_payload_column(daily, "weather_code", raw.index)
    output["openmeteo_precipitation_sum_mm"] = _numeric_payload_column(
        daily, "precipitation_sum", raw.index
    )
    output["openmeteo_precipitation_hours"] = _numeric_payload_column(
        daily, "precipitation_hours", raw.index
    )
    output["openmeteo_rain_sum_mm"] = _numeric_payload_column(daily, "rain_sum", raw.index)
    output["openmeteo_wind_speed_10m_max_kmh"] = _numeric_payload_column(
        daily, "wind_speed_10m_max", raw.index
    )
    output["openmeteo_wind_gusts_10m_max_kmh"] = _numeric_payload_column(
        daily, "wind_gusts_10m_max", raw.index
    )
    output["openmeteo_precipitation_flag"] = (
        output["openmeteo_precipitation_sum_mm"].fillna(0.0) > 0.0
    ).astype(int)
    output["openmeteo_rain_flag"] = (output["openmeteo_rain_sum_mm"].fillna(0.0) > 0.0).astype(int)
    return output.dropna(subset=["local_date"]).sort_values("local_date").reset_index(drop=True)


def _hourly_features_from_payload(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    frame = pd.DataFrame({"time": pd.to_datetime(times, errors="coerce")})
    frame = frame.dropna(subset=["time"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    frame["local_date"] = frame["time"].dt.date.astype("string")
    frame["hour"] = frame["time"].dt.hour
    for column in HOURLY_VARIABLES:
        frame[column] = _numeric_payload_column(hourly, column, frame.index)

    grouped = frame.groupby("local_date", as_index=False).agg(
        openmeteo_hourly_temp_max_c=("temperature_2m", "max"),
        openmeteo_hourly_temp_min_c=("temperature_2m", "min"),
        openmeteo_hourly_temp_mean_c=("temperature_2m", "mean"),
        openmeteo_hourly_weather_code_max=("weather_code", "max"),
        openmeteo_hourly_wind_speed_10m_max_kmh=("wind_speed_10m", "max"),
        openmeteo_hourly_wind_speed_10m_mean_kmh=("wind_speed_10m", "mean"),
        openmeteo_hourly_wind_gusts_10m_max_kmh=("wind_gusts_10m", "max"),
        openmeteo_hourly_cloud_cover_mean_pct=("cloud_cover", "mean"),
        openmeteo_hourly_cloud_cover_max_pct=("cloud_cover", "max"),
        openmeteo_hourly_visibility_min_m=("visibility", "min"),
        openmeteo_hourly_visibility_mean_m=("visibility", "mean"),
        openmeteo_hourly_rain_sum_mm=("rain", "sum"),
        openmeteo_hourly_rain_max_mm=("rain", "max"),
        openmeteo_hourly_precipitation_sum_mm=("precipitation", "sum"),
        openmeteo_hourly_precipitation_max_mm=("precipitation", "max"),
        openmeteo_hourly_precipitation_probability_max_pct=("precipitation_probability", "max"),
        openmeteo_hourly_precipitation_probability_mean_pct=("precipitation_probability", "mean"),
    )
    peak_rows = frame.dropna(subset=["temperature_2m"]).sort_values("temperature_2m")
    if not peak_rows.empty:
        peak = peak_rows.groupby("local_date").tail(1)[["local_date", "hour"]].rename(
            columns={"hour": "openmeteo_hourly_temp_peak_hour"}
        )
        grouped = grouped.merge(peak, on="local_date", how="left")
    for hour in (0, 3, 6, 9, 12, 15, 18, 21):
        at_hour = frame[frame["hour"] == hour][["local_date", "temperature_2m"]].rename(
            columns={"temperature_2m": f"openmeteo_hourly_temp_{hour:02d}z_c"}
        )
        grouped = grouped.merge(at_hour, on="local_date", how="left")
    return grouped.sort_values("local_date").reset_index(drop=True)


def _numeric_payload_column(payload_section: dict, key: str, index: Iterable) -> pd.Series:
    values = payload_section.get(key)
    if values is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(values), errors="coerce").reset_index(drop=True)


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
