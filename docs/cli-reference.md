# CLI Reference

All commands are run through `uv`:

```powershell
uv run <command> [options]
```

## Table Of Contents

- [Cutoff And Date Rules](#cutoff-and-date-rules)
- [Station Shortcuts](#station-shortcuts)
- [METAR Commands](#metar-commands)
- [DuckDB Commands](#duckdb-commands)
- [Heat-Risk Model Commands](#heat-risk-model-commands)
- [Prediction Command](#prediction-command)
- [Telegram Report Command](#telegram-report-command)
- [Common Daily Recipes](#common-daily-recipes)

## Cutoff And Date Rules

`--date` is the forecast date in the station's local timezone.

`--cutoff-local` is also in the station's local timezone. It is not the device
timezone and not UTC.

If you set:

```powershell
$CUTOFF = "11:00"
```

then pass:

```powershell
--cutoff-local $CUTOFF
```

the model interprets that as `11:00` in the timezone from the selected config:

| Config | Station | Timezone |
|---|---|---|
| `configs/default.yaml` | RKSI | `Asia/Seoul` |
| `configs/rkpk.yaml` | RKPK | `Asia/Seoul` |
| `configs/rjtt.yaml` | RJTT | `Asia/Tokyo` |
| `configs/wsss.yaml` | WSSS | `Asia/Singapore` |

The shortcut commands derive today's date and the default cutoff from the
station config timezone. For example, `uv run wsss` rounds the current
Singapore local hour down to `HH:00`.

## Station Shortcuts

These commands fetch recent METAR data for all built-in stations, import it,
repair DuckDB if needed, then predict one station. They default to:

- date: today in the station timezone;
- cutoff: current station local hour rounded down;
- `--fetch`;
- `--plot`;
- `--explain`;
- `--sync-duckdb`.

### `rksi`

```powershell
uv run rksi
uv run rksi --date 2026-06-20 --cutoff-local 11:00
uv run rksi --no-fetch --no-plot --no-explain
```

Uses `configs/default.yaml`.

### `rkpk`

```powershell
uv run rkpk
uv run rkpk --date 2026-06-20 --cutoff-local 11:00
```

Uses `configs/rkpk.yaml`.

### `rjtt`

```powershell
uv run rjtt
uv run rjtt --date 2026-06-20 --cutoff-local 11:00
```

Uses `configs/rjtt.yaml`.

### `wsss`

```powershell
uv run wsss
uv run wsss --date 2026-06-20 --cutoff-local 10:00
```

Uses `configs/wsss.yaml`.

Shortcut options:

| Option | Meaning |
|---|---|
| `--date YYYY-MM-DD` | Forecast date in station local time. |
| `--cutoff-local HH:MM` | Cutoff in station local time. |
| `--hours N` | METAR fetch lookback window. Default `4`. |
| `--metar-file PATH` | METAR text file. Default `metar.txt`. |
| `--fetch` / `--no-fetch` | Fetch fresh METAR lines or use existing file. |
| `--plot` / `--no-plot` | Write a temperature-curve PNG. |
| `--explain` / `--no-explain` | Print Vietnamese explanation after JSON. |
| `--sync-duckdb` / `--no-sync-duckdb` | Rebuild DuckDB if missing/too short. |

## METAR Commands

### `rksi-fetch-metar`

Fetch recent METAR text from Aviation Weather.

```powershell
uv run rksi-fetch-metar --stations RKSI,RKPK,RJTT,WSSS --hours 48 --output metar.txt
```

Options:

| Option | Meaning |
|---|---|
| `--stations RKSI,RKPK` | Comma-separated ICAO station list. |
| `--hours N` | Lookback window. Default `48`. |
| `--output PATH` | Output text file. Default `metar.txt`. |

This command overwrites the output file.

### `rksi-import-metar`

Parse `metar.txt`, append new rows to CSV, and upsert new rows into DuckDB when
the selected config has `prefer_duckdb: true`.

```powershell
uv run rksi-import-metar --metar-file metar.txt --reference-date 2026-06-20
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Config used for default CSV/DB paths. Default `configs/default.yaml`. |
| `--metar-file PATH` | METAR text input. Default `metar.txt`. |
| `--csv PATH` | Target ASOS CSV. Defaults to config `input_csv`. |
| `--reference-date YYYY-MM-DD` | UTC date used to infer month/year from METAR `DDHHMMZ` tokens. |

Re-running import is safe. Existing `(station, valid)` rows are skipped.

## DuckDB Commands

### `rksi-sync-duckdb`

Rebuild DuckDB from one or more raw CSV files.

```powershell
uv run rksi-sync-duckdb
uv run rksi-sync-duckdb --config configs/wsss.yaml
uv run rksi-sync-duckdb --csv asos.csv --csv RJTT.csv --csv WSSS.csv --db artifacts/observations.duckdb
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Config for default `raw_csv_files` and DB path. |
| `--csv PATH` | CSV to load. Repeat for multiple files. |
| `--db PATH` | Output DuckDB path. |

## Heat-Risk Model Commands

### `rksi-build-heat-risk-dataset`

Build the multi-cutoff training/validation table.

```powershell
uv run rksi-build-heat-risk-dataset --config configs/default.yaml
uv run rksi-build-heat-risk-dataset --config configs/rjtt.yaml --output artifacts/rjtt_heat_risk_dataset.parquet
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Station config. Default `configs/default.yaml`. |
| `--input-csv PATH` | Override config `input_csv`. |
| `--output PATH` | Override config dataset parquet path. |

### `rksi-train-heat-risk`

Train the model for one config.

```powershell
uv run rksi-train-heat-risk --config configs/default.yaml
uv run rksi-train-heat-risk --config configs/wsss.yaml
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Station config. Default `configs/default.yaml`. |

### `rksi-validate-heat-risk`

Validate the trained model and write validation artifacts.

```powershell
uv run rksi-validate-heat-risk --config configs/default.yaml
uv run rksi-validate-heat-risk --config configs/rkpk.yaml
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Station config. Default `configs/default.yaml`. |

## Prediction Command

### `rksi-predict-heat-risk`

Predict remaining heat and final `Tmax` for one date/cutoff.

```powershell
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local 11:00
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local 11:00 --plot --explain
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local 11:00 --plot artifacts/rksi_2026-06-20_1100.png
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | Station config. Default `configs/default.yaml`. |
| `--date YYYY-MM-DD` | Required. Forecast date in station local time. |
| `--cutoff-local HH:MM` | Required. Cutoff in station local time. |
| `--dataset PATH` | Optional existing heat-risk parquet to search before live feature build. |
| `--plot [PATH]` | Write chart. Omit path for default `artifacts/{station}_{date}_{cutoff}_temperature_curve.png`. |
| `--explain` | Print Vietnamese explanation after JSON. |

Prediction does not require rebuilding the full dataset. It builds one feature
row directly from the configured data source when `--dataset` is omitted or
does not contain the requested row.

## Telegram Report Command

### `rksi-telegram-report`

Build a combined report for configured stations.

```powershell
uv run rksi-telegram-report --output artifacts/telegram_report.md --hours 4
uv run rksi-telegram-report --no-fetch --no-sync-duckdb --output artifacts/telegram_report.md
uv run rksi-telegram-report --config configs/default.yaml --config configs/wsss.yaml
```

Options:

| Option | Meaning |
|---|---|
| `--output PATH` | Report markdown path. Default `artifacts/telegram_report.md`. |
| `--metar-file PATH` | METAR text file. Default `metar.txt`. |
| `--hours N` | Fetch lookback window. Default `4`. |
| `--fetch` / `--no-fetch` | Fetch new METAR lines before importing. |
| `--sync-duckdb` / `--no-sync-duckdb` | Sync DuckDB when missing/too short. |
| `--config PATH` | Add a config to the report. Repeat for multiple stations. |
| `--reference-date YYYY-MM-DD` | UTC date for METAR date inference. |

The report selects the latest configured heat-risk cutoff that is not later
than each station's local current time.

Send the generated report:

```powershell
node scripts/send_telegram_report.mjs artifacts/telegram_report.md
```

## Common Daily Recipes

### One Station, Fully Automatic

```powershell
uv run rksi
```

### One Station, Existing METAR File

```powershell
uv run rksi --no-fetch --date 2026-06-20 --cutoff-local 11:00
```

### All Built-In Stations Manually

```powershell
$DATE = "2026-06-20"

uv run rksi-fetch-metar --stations RKSI,RKPK,RJTT,WSSS --hours 48 --output metar.txt
uv run rksi-import-metar --metar-file metar.txt --reference-date $DATE

uv run rksi-predict-heat-risk --config configs/default.yaml --date $DATE --cutoff-local 11:00 --plot --explain
uv run rksi-predict-heat-risk --config configs/rkpk.yaml --date $DATE --cutoff-local 11:00 --plot --explain
uv run rksi-predict-heat-risk --config configs/rjtt.yaml --date $DATE --cutoff-local 11:00 --plot --explain
uv run rksi-predict-heat-risk --config configs/wsss.yaml --date $DATE --cutoff-local 10:00 --plot --explain
```

### Rebuild A Station Model

```powershell
uv run rksi-build-heat-risk-dataset --config configs/rjtt.yaml
uv run rksi-train-heat-risk --config configs/rjtt.yaml
uv run rksi-validate-heat-risk --config configs/rjtt.yaml
```
