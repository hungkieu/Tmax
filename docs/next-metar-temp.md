# Next-METAR 1h Integer Temperature Model

This workflow predicts the integer Celsius temperature reported by the METAR
nearest to one hour after the latest/current METAR observation.

It is separate from the daily `Tmax` heat-risk model. It does not use the old
`artifacts/next_metar_temp` bundle unless that bundle is rebuilt by the new
commands.

## Question Answered

```text
Given the latest METAR, the prior temperature evolution, cloud/wind/rain
signals, and Open-Meteo forecast context, what integer Celsius temperature is
most likely in the METAR around +60 minutes?
```

The prediction output includes:

- `predicted_temp_c`: most likely integer Celsius value, such as `25`, `26`,
  or `27`;
- `expected_temp_c`: probability-weighted mean;
- `probabilities_by_temp_c`: probability mass for each integer temperature;
- probability that the next METAR is exactly current `-1 C`, current, or
  current `+1 C`;
- probability that it decreases by at least `1 C` or increases by at least
  `1 C`.

## Feature Set

The model is designed around METAR nowcasting, not final daily Tmax:

- current METAR temperature/dewpoint/humidity/pressure/visibility;
- wind direction, wind speed, gust, and wind-regime flags;
- cloud cover from METAR layers, lowest ceiling, low-cloud flag;
- rain/precipitation, fog/mist, shower, and thunder flags from weather codes;
- recent temperature movement over 30, 60, 120, and 180 minutes;
- current day thermal trough observed so far;
- minutes since that trough and warming from that trough to now;
- same-local-minute comparison against the previous 3 days;
- Open-Meteo daily forecast features when configured;
- Open-Meteo hourly features at the current hour and target +1h hour, including
  forecast temperature, cloud, wind, rain, precipitation, visibility, and
  precipitation probability.

Only observations at or before the current METAR are used for current-day
thermal features. Future observations are used only as training targets.

## Build Dataset

```powershell
uv run rksi-build-next-metar-temp-dataset --config configs/default.yaml
```

This reads the configured station observations from DuckDB/CSV, prepares
Open-Meteo historical forecast cache when coordinates are configured, and
writes:

```text
artifacts/next_metar_temp/next_metar_temp_dataset.parquet
```

The dataset target is `target_temp_c_int`, the rounded integer Celsius
temperature at the observation nearest `current METAR + 60 minutes`, within the
training tolerance.

## Train

```powershell
uv run rksi-train-next-metar-temp --config configs/default.yaml
```

This writes:

```text
artifacts/next_metar_temp/next_metar_temp_model.joblib
artifacts/next_metar_temp/next_metar_temp_metrics.json
```

If an old parquet exists at the same path but does not contain the v3
METAR/weather/Open-Meteo feature set, training rebuilds it instead of reusing
it.

## Validate

```powershell
uv run rksi-validate-next-metar-temp --config configs/default.yaml
```

The validation report includes exact integer accuracy, within-1C accuracy,
MAE, bias, delta/direction accuracy, persistence baseline, and breakdowns by
station/hour.

## Predict

Use the latest imported observation:

```powershell
uv run rksi-predict-next-metar-temp --config configs/default.yaml
```

Use the latest observation at or before a station-local timestamp:

```powershell
uv run rksi-predict-next-metar-temp --config configs/default.yaml --as-of-local "2026-06-26 14:30"
```

Fetch/update Open-Meteo live cache for the prediction date before predicting:

```powershell
uv run rksi-predict-next-metar-temp --config configs/default.yaml --fetch-openmeteo
```

## UI Workflow

Run the dashboard:

```powershell
uv run rksi-ui
```

Open `Operations` -> `METAR` first and use `Update live data` when fresh data
is needed. That action fetches METAR, imports it into DuckDB, prepares the
daily Open-Meteo cache for selected locations, and displays coverage/missing
data by station.

Then open `Operations` -> `Next METAR`.

The main prediction action uses the latest updated database observation. The
tab also shows:

- latest database observation status;
- model artifact and validation metrics;
- whole-degree probability distribution;
- exact `-1/0/+1 C` and at-least-up/down probabilities;
- temperature-trend, METAR weather, and Open-Meteo context tables;
- an expander for build/train/validate controls when the model needs refresh.

Before live prediction, fetch/import METAR first, the same way as the existing
heat-risk workflow:

```powershell
uv run rksi-fetch-metar --stations RKSI --hours 6 --output data/shared/metar.txt
uv run rksi-import-metar --config configs/default.yaml --metar-file data/shared/metar.txt --reference-date 2026-06-26
uv run rksi-predict-next-metar-temp --config configs/default.yaml --fetch-openmeteo
```
