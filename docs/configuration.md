# Configuration Guide

The project is controlled by YAML files under `configs/`. Each station should
have its own config so models, metrics, and reports do not overwrite each
other.

Current configs:

| Station | Config |
|---|---|
| RKSI | `configs/default.yaml` |
| RKPK | `configs/rkpk.yaml` |
| RJTT | `configs/rjtt.yaml` |
| WSSS | `configs/wsss.yaml` |

Use a config with any command through `--config`:

```powershell
uv run rksi-predict-heat-risk --config configs/rjtt.yaml --date 2026-06-19 --cutoff-local 10:00 --plot --explain
uv run rksi-build-heat-risk-dataset --config configs/rjtt.yaml
uv run rksi-train-heat-risk --config configs/rjtt.yaml
uv run rksi-validate-heat-risk --config configs/rjtt.yaml
```

## Station And Time

```yaml
station: RJTT
timezone: Asia/Tokyo
cutoff_local: "12:00"
complete_day_min_local: "23:00"
target: tmax
```

- `station`: ICAO station ID to filter from CSV/DuckDB.
- `timezone`: local timezone used to group observations into local days.
- `cutoff_local`: default cutoff used by older/default dataset builders. For
  live prediction, `--cutoff-local` overrides this value.
- `complete_day_min_local`: latest local minute required before a day is
  treated as complete for training target `Tmax`. Keep this late, usually
  `23:00`.
- `target`: currently only `tmax` is supported.

## Data Sources

```yaml
input_csv: data/rjtt/RJTT.csv
input_db: artifacts/shared/observations.duckdb
prefer_duckdb: true
raw_csv_files:
  - data/rjtt/RJTT.csv
openmeteo_history_csv: data/rksi/openmeteo-rksi.csv
openmeteo_live_csv_pattern: data/rksi/openmeteo-rksi-{date}.csv
openmeteo_history_json: data/rksi/openmeteo-rksi-history.json
openmeteo_live_json_pattern: data/rksi/openmeteo-rksi-{date}.json
openmeteo_latitude: 37.4492
openmeteo_longitude: 126.451
openmeteo_timezone: GMT
openmeteo_training_start_date: 2023-01-01
```

- `input_csv`: fallback CSV for this station. Also used by METAR import when
  `--csv` is not passed.
- `input_db`: DuckDB file used for faster station filtering.
- `prefer_duckdb`: when `true`, commands read from DuckDB if the DB exists.
  If the DB is missing, code falls back to `input_csv`.
- `raw_csv_files`: files loaded by `rksi-sync-duckdb` when `--csv` is not
  passed. Sync updates rows from these CSV files and keeps existing rows for
  other stations in the same DuckDB database.
- `openmeteo_history_json`: Open-Meteo historical forecast API cache for
  training. Dataset build refreshes it when configured coordinates exist and
  the cache does not cover the completed observation date range.
- `openmeteo_live_json_pattern`: Open-Meteo forecast API cache pattern for
  daily prediction. `{date}` is replaced with the local forecast date.
- `openmeteo_latitude` / `openmeteo_longitude`: station coordinates used by
  the Open-Meteo API. Add these in the UI for each location that should train
  M3.
- `openmeteo_timezone`: timezone sent to Open-Meteo. `GMT` matches the current
  training cache convention.
- `openmeteo_training_start_date` / `openmeteo_training_end_date`: optional
  bounds for historical forecast training fetches. RKSI starts at `2023-01-01`
  to match available Open-Meteo historical forecast data.
- `openmeteo_history_csv` and `openmeteo_live_csv_pattern`: legacy CSV inputs.
  They still work, but JSON API cache is preferred.

If CSV and DuckDB disagree, resync:

```powershell
uv run rksi-sync-duckdb --config configs/rjtt.yaml
```

## Open-Meteo Forecast Features

Open-Meteo is optional per station. When coordinates and JSON cache paths are
configured, dataset build fetches historical forecast API data and joins both
daily and hourly-derived features by `local_date`. Training adds an M3
Open-Meteo enhanced regressor. Validation compares M3 against the ASOS/METAR-only
M0/M1 models and selects the lower-MAE method for `predicted_tmax_c`.
M3 also compares daily-only Open-Meteo features against daily+hourly features
and records `selected_openmeteo_variant`.

Historical forecast API cache:

```yaml
openmeteo_history_json: data/rksi/openmeteo-rksi-history.json
```

Daily forecast API cache files use `{date}` in the configured pattern:

```yaml
openmeteo_live_json_pattern: data/rksi/openmeteo-rksi-{date}.json
```

For example, prediction for `2026-06-21` will use
`data/rksi/openmeteo-rksi-2026-06-21.json`. The live row overrides the history
row for that date. Prediction output and Telegram reports show both the raw
Open-Meteo Tmax and the M3 corrected Tmax.

Prepare cache from CLI:

```powershell
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode training
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode daily --date 2026-06-21
```

## Heat-Risk Training

```yaml
heat_risk_cutoffs:
  - "09:00"
  - "10:00"
  - "11:00"
  - "12:00"
  - "13:00"
heat_risk_thresholds_c:
  - 28.0
  - 30.0
  - 32.0
  - 35.0
```

- `heat_risk_cutoffs`: cutoffs used to build the training/validation table.
  These are not the only cutoffs allowed at prediction time; prediction can use
  any local `HH:MM` as long as observations exist at or before that time.
- `heat_risk_thresholds_c`: final-Tmax thresholds used for probability fields
  like `prob_tmax_ge_30c`.

After changing either field, rebuild/train/validate:

```powershell
uv run rksi-build-heat-risk-dataset --config configs/rjtt.yaml
uv run rksi-train-heat-risk --config configs/rjtt.yaml
uv run rksi-validate-heat-risk --config configs/rjtt.yaml
```

Avoid dense half-degree thresholds unless the operation really needs them.
METAR temperatures are usually integer Celsius, so `28.5` and `29.0` can often
produce nearly identical probabilities.

## Artifact Paths

```yaml
heat_risk_dataset_parquet: artifacts/rjtt/rjtt_heat_risk_dataset.parquet
heat_risk_model_path: artifacts/rjtt/rjtt_heat_risk_model.joblib
heat_risk_metrics_path: artifacts/rjtt/rjtt_heat_risk_metrics.json
```

Use station-specific paths. If two configs share the same model or metrics
path, training one station can overwrite the other station's artifacts.

Validation also writes files next to `heat_risk_metrics_path`, using the same
stem:

```text
artifacts/rjtt/rjtt_heat_risk_validation_report.json
artifacts/rjtt/rjtt_heat_risk_top_errors.csv
artifacts/rjtt/rjtt_heat_risk_diagnostics.png
artifacts/rjtt/rjtt_heat_risk_thermal_curve_diagnostics.png
```

Prediction plots from `--plot` are separate and use this pattern by default:

```text
artifacts/{station}/{station}_{date}_{cutoff}_temperature_curve.png
```

## Train/Test And Model Stability

```yaml
test_fraction: 0.2
random_state: 42
feature_missing_threshold: 0.85
```

- `test_fraction`: last fraction of local dates held out for validation.
  Keep time-based split; do not random shuffle weather days.
- `random_state`: reproducibility seed for scikit-learn models.
- `feature_missing_threshold`: feature columns with missing rate above this
  value are excluded from training.

Change these only when you are intentionally revalidating the model behavior.

## Adding A New Location

Use this process when adding another ICAO station/location. The UI can create
the config, but the model still needs historical ASOS-style data before it can
train.

### 1. Prepare Historical Data

Create or obtain a CSV with ASOS-style columns. At minimum the training path
needs:

```text
station, valid, tmpf
```

Useful predictors include dewpoint, humidity, wind, pressure, visibility,
cloud layers, weather codes, and raw METAR text:

```text
dwpf, relh, drct, sknt, p01i, alti, mslp, vsby, gust,
skyc1, skyc2, skyc3, skyc4, skyl1, skyl2, skyl3, skyl4,
wxcodes, feel, metar
```

`valid` must be UTC in this format:

```text
YYYY-MM-DD HH:MM
```

Put the file under a station-specific folder, for example:

```text
data/eddm/EDDM.csv
```

A location created only from live METAR imports is not enough for training.
Those imports usually cover only the most recent day or two.

### 2. Create The Config

From the UI:

1. Run `uv run rksi-ui`.
2. Open `Locations`.
3. Choose `Create location`.
4. Enter station code, timezone, CSV path, artifact paths, cutoffs, and
   thresholds.
5. Enable Open-Meteo if the location should train M3, then enter latitude,
   longitude, and optional training date bounds.
6. Click `Create location`.

The UI can create an empty CSV header. For real training, replace or append
that file with historical ASOS-style rows before building the dataset.

From CLI/manual editing, copy the closest existing config:

```powershell
Copy-Item configs/rjtt.yaml configs/new_station.yaml
```

Edit station identity, timezone, source path, and artifact paths:

```yaml
station: XXXX
timezone: Region/City
input_csv: data/xxxx/YOUR_FILE.csv
raw_csv_files:
  - data/xxxx/YOUR_FILE.csv
heat_risk_dataset_parquet: artifacts/xxxx/xxxx_heat_risk_dataset.parquet
heat_risk_model_path: artifacts/xxxx/xxxx_heat_risk_model.joblib
heat_risk_metrics_path: artifacts/xxxx/xxxx_heat_risk_metrics.json
```

Use an IANA timezone such as `Europe/Berlin`, `Asia/Seoul`, `Asia/Tokyo`, or
`Asia/Singapore`. `--date`, `--cutoff-local`, `cutoff_local`, and
`complete_day_min_local` are interpreted in this station timezone.

Avoid setting `openmeteo_training_end_date` unless you intentionally want a
fixed cap. If the cap is earlier than the first completed observation day, the
Open-Meteo training range becomes empty.

### 3. Sync Historical CSV Into DuckDB

When `prefer_duckdb: true`, commands read DuckDB first if it exists. After
adding a new historical CSV, sync it:

```powershell
uv run rksi-sync-duckdb --config configs/new_station.yaml
```

This step prevents a common failure mode: the CSV has years of history, but
DuckDB only has recently imported METAR rows, so training sees only one or two
days.

Verify coverage:

```powershell
uv run python -c "from rksi_tmax.config import load_config; from rksi_tmax.features import load_observations; from rksi_tmax.heat_risk import _complete_observation_date_range; c=load_config('configs/new_station.yaml'); o=load_observations(c.input_csv,c); print(len(o), _complete_observation_date_range(o,c))"
```

The completed range should cover many historical days. If it only shows today
or yesterday, sync DuckDB again or set `prefer_duckdb: false` temporarily while
debugging the CSV.

### 4. Prepare Open-Meteo M3 Cache

If Open-Meteo coordinates are configured, prepare historical forecast cache:

```powershell
uv run rksi-fetch-openmeteo --config configs/new_station.yaml --mode training
```

In the UI, use `Train and Validate` -> `Open-Meteo M3` -> `Prepare training
data`.

The fetch range is computed from completed observation days after applying
optional `openmeteo_training_start_date` and `openmeteo_training_end_date`.
If the app reports an empty date range, sync/import historical observations or
remove/fix the Open-Meteo date bounds.

### 5. Build, Train, Validate

```powershell
uv run rksi-build-heat-risk-dataset --config configs/new_station.yaml
uv run rksi-train-heat-risk --config configs/new_station.yaml
uv run rksi-validate-heat-risk --config configs/new_station.yaml
```

In the UI, use `Train and Validate` in this order:

1. `Build dataset`.
2. `Train`.
3. `Validate`.

Review MAE, rounded Tmax win rates by cutoff, and the validation JSON/PNG files
next to `heat_risk_metrics_path` before using the model operationally.

### 6. Daily Operation And Predict

For live use, fetch/import METAR first so DuckDB has observations through the
latest cutoff:

1. In the UI, open `METAR`.
2. Select the location.
3. Click `Fetch METAR`.
4. Click `Import + DB`.
5. Verify the latest database observation.
6. Open `Predict` and run the forecast.

`Predict` can fetch the Open-Meteo daily forecast cache on demand for M3, but
it does not fetch/import METAR observations by itself.

```powershell
uv run rksi-predict-heat-risk --config configs/new_station.yaml --date 2026-06-19 --cutoff-local 12:00 --plot --explain
```

The cutoff is local time for the new station, based on `timezone` in the
config.

### 7. Add To Combined Telegram Report

Pass the config explicitly:

```powershell
uv run rksi-telegram-report --config configs/default.yaml --config configs/new_station.yaml
```

To make the new location part of the default automated report, update
`DEFAULT_CONFIG_PATHS` and `DEFAULT_STATIONS` in
`src/rksi_tmax/telegram_report.py`, then update tests.

### 8. Add A Shortcut Command

Shortcut commands such as `uv run wsss` are registered in two places:

1. Add a script entry in `pyproject.toml`.
2. Add the shortcut function and config mapping in `src/rksi_tmax/cli.py`.

If you do not need a shortcut, keep using `rksi-predict-heat-risk --config`.

### 9. Refresh Local Model Artifacts

After training a new operational model, keep the generated files under
`artifacts/` with station-specific names. If you still want a portable archive
for backup or manual transfer, rebuild the ZIP:

```powershell
.\scripts\create_model_release.ps1 -Force
```

## Retraining An Existing Location

Daily prediction does not require retraining. Retrain only when at least one of
these changed:

- you added enough completed historical days to improve the model;
- you changed `heat_risk_cutoffs`, `heat_risk_thresholds_c`, station timezone,
  source paths, or artifact paths;
- you enabled or changed Open-Meteo M3 coordinates/cache paths;
- you extended the historical Open-Meteo cache and want new validation metrics;
- model, feature, import, or validation code changed.

### 1. Refresh Historical Observations

Make sure the station CSV contains the completed historical days you want to
train on. If `prefer_duckdb: true`, sync DuckDB before rebuilding:

```powershell
uv run rksi-sync-duckdb --config configs/existing_station.yaml
```

Verify the completed observation range:

```powershell
uv run python -c "from rksi_tmax.config import load_config; from rksi_tmax.features import load_observations; from rksi_tmax.heat_risk import _complete_observation_date_range; c=load_config('configs/existing_station.yaml'); o=load_observations(c.input_csv,c); print(len(o), _complete_observation_date_range(o,c))"
```

If the range is unexpectedly short, DuckDB is probably stale or the CSV path is
wrong.

### 2. Refresh Open-Meteo Cache If M3 Is Enabled

If the location has Open-Meteo coordinates, refresh the historical forecast
cache before rebuilding:

```powershell
uv run rksi-fetch-openmeteo --config configs/existing_station.yaml --mode training
```

Use `--force` only when you intentionally want to re-download the cache:

```powershell
uv run rksi-fetch-openmeteo --config configs/existing_station.yaml --mode training --force
```

Avoid stale `openmeteo_training_end_date` values. If this field is set earlier
than the latest completed observation day, remove it or update it.

### 3. Rebuild, Train, Validate

```powershell
uv run rksi-build-heat-risk-dataset --config configs/existing_station.yaml
uv run rksi-train-heat-risk --config configs/existing_station.yaml
uv run rksi-validate-heat-risk --config configs/existing_station.yaml
```

In the UI, use `Train and Validate` in this order:

1. `Prepare training data` under `Open-Meteo M3`, if M3 is configured.
2. `Build dataset`.
3. `Train`.
4. `Validate`.

Validation overwrites the station validation report and diagnostics next to
`heat_risk_metrics_path`. Compare MAE, selected method, Open-Meteo variant,
interval coverage, and rounded Tmax win rates before using the new model
operationally.
