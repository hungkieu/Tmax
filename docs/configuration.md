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
input_csv: RJTT.csv
input_db: artifacts/observations.duckdb
prefer_duckdb: true
raw_csv_files:
  - RJTT.csv
openmeteo_history_csv: openmeteo-rksi.csv
openmeteo_live_csv_pattern: openmeteo-rksi-{date}.csv
```

- `input_csv`: fallback CSV for this station. Also used by METAR import when
  `--csv` is not passed.
- `input_db`: DuckDB file used for faster station filtering.
- `prefer_duckdb`: when `true`, commands read from DuckDB if the DB exists.
  If the DB is missing, code falls back to `input_csv`.
- `raw_csv_files`: files loaded by `rksi-sync-duckdb` when `--csv` is not
  passed. Sync updates rows from these CSV files and keeps existing rows for
  other stations in the same DuckDB database.
- `openmeteo_history_csv`: optional Open-Meteo daily forecast/history CSV.
  When present, dataset build joins it by `local_date` and trains the M3
  Open-Meteo enhanced regressor.
- `openmeteo_live_csv_pattern`: optional live forecast file pattern. `{date}`
  is replaced with the local forecast date, for example
  `openmeteo-rksi-2026-06-21.csv`. Live rows override history rows for the
  same date.

If CSV and DuckDB disagree, resync:

```powershell
uv run rksi-sync-duckdb --config configs/rjtt.yaml
```

## Open-Meteo Forecast Features

Open-Meteo is optional per station. When configured, dataset build joins the
daily forecast by `local_date` and training adds an M3 Open-Meteo enhanced
regressor. Validation compares M3 against the ASOS/METAR-only M0 model and
selects the lower-MAE method for `predicted_tmax_c`.

Historical forecast file:

```yaml
openmeteo_history_csv: openmeteo-rksi.csv
```

Live forecast files use `{date}` in the configured pattern:

```yaml
openmeteo_live_csv_pattern: openmeteo-rksi-{date}.csv
```

For example, prediction for `2026-06-21` will use
`openmeteo-rksi-2026-06-21.csv` if it exists. The live row overrides the
history row for that date. Prediction output and Telegram reports show both
the raw Open-Meteo Tmax and the M3 corrected Tmax.

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
heat_risk_dataset_parquet: artifacts/rjtt_heat_risk_dataset.parquet
heat_risk_model_path: artifacts/rjtt_heat_risk_model.joblib
heat_risk_metrics_path: artifacts/rjtt_heat_risk_metrics.json
```

Use station-specific paths. If two configs share the same model or metrics
path, training one station can overwrite the other station's artifacts.

Validation also writes files next to `heat_risk_metrics_path`, using the same
stem:

```text
artifacts/rjtt_heat_risk_validation_report.json
artifacts/rjtt_heat_risk_top_errors.csv
artifacts/rjtt_heat_risk_diagnostics.png
artifacts/rjtt_heat_risk_thermal_curve_diagnostics.png
```

Prediction plots from `--plot` are separate and use this pattern by default:

```text
artifacts/{station}_{date}_{cutoff}_temperature_curve.png
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

Use this process when adding another ICAO station/location.

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

### 2. Create A Config

Copy the closest existing config:

```powershell
Copy-Item configs/rjtt.yaml configs/new_station.yaml
```

Edit station identity, timezone, source path, and artifact paths:

```yaml
station: XXXX
timezone: Region/City
input_csv: YOUR_FILE.csv
heat_risk_dataset_parquet: artifacts/xxxx_heat_risk_dataset.parquet
heat_risk_model_path: artifacts/xxxx_heat_risk_model.joblib
heat_risk_metrics_path: artifacts/xxxx_heat_risk_metrics.json
```

Use an IANA timezone such as `Asia/Seoul`, `Asia/Tokyo`, or
`Asia/Singapore`. `--date`, `--cutoff-local`, `cutoff_local`, and
`complete_day_min_local` are interpreted in this station timezone.

### 3. Add The CSV To DuckDB Sync

Add the new CSV to `raw_csv_files` so `rksi-sync-duckdb` loads it. It can be
only this station's CSV even when several station configs share the same
`input_db`:

```yaml
raw_csv_files:
  - NEW_STATION.csv
```

Then sync:

```powershell
uv run rksi-sync-duckdb --config configs/new_station.yaml
```

### 4. Build, Train, Validate

```powershell
uv run rksi-build-heat-risk-dataset --config configs/new_station.yaml
uv run rksi-train-heat-risk --config configs/new_station.yaml
uv run rksi-validate-heat-risk --config configs/new_station.yaml
```

Review the validation JSON/PNG files next to `heat_risk_metrics_path` before
using the model operationally.

### 5. Predict

```powershell
uv run rksi-predict-heat-risk --config configs/new_station.yaml --date 2026-06-19 --cutoff-local 12:00 --plot --explain
```

The cutoff is local time for the new station, based on `timezone` in the
config.

### 6. Add To Combined Telegram Report

Pass the config explicitly:

```powershell
uv run rksi-telegram-report --config configs/default.yaml --config configs/new_station.yaml
```

To make the new location part of the default automated report, update
`DEFAULT_CONFIG_PATHS` and `DEFAULT_STATIONS` in
`src/rksi_tmax/telegram_report.py`, then update tests.

### 7. Add A Shortcut Command

Shortcut commands such as `uv run wsss` are registered in two places:

1. Add a script entry in `pyproject.toml`.
2. Add the shortcut function and config mapping in `src/rksi_tmax/cli.py`.

If you do not need a shortcut, keep using `rksi-predict-heat-risk --config`.

### 8. Refresh Local Model Artifacts

After training a new operational model, keep the generated files under
`artifacts/` with station-specific names. If you still want a portable archive
for backup or manual transfer, rebuild the ZIP:

```powershell
.\scripts\create_model_release.ps1 -Force
```
