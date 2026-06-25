# METAR/ASOS Tmax Heat Risk Forecast

Forecast final daily maximum temperature risk from historical and live
METAR/ASOS observations.

The operational forecast answers:

- how much higher final `Tmax` can still go after a local cutoff;
- likely final `Tmax`;
- probability of crossing configured hot thresholds;
- current thermal phase and late-warming risk;
- short future temperature curve through the next 3 hours;
- whether the next cutoff update is worth running.

`predicted_tmax_c` is the operational forecast from the method selected by
validation. Locations with Open-Meteo coordinates can use M3 API forecast
features; reports show both the raw Open-Meteo Tmax and the M3 corrected Tmax when available. Thermal
phase, late-warming risk, and the future curve are supporting context.

## Table Of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Supported Stations](#supported-stations)
- [Daily Operation](#daily-operation)
- [Local UI Dashboard](#local-ui-dashboard)
- [Training Workflow](#training-workflow)
- [Documentation](#documentation)
- [Development Checks](#development-checks)

## Quick Start

Install `uv` if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install dependencies:

```powershell
uv sync --dev
```

Run one station with fetch, import, prediction, chart, and Vietnamese
explanation enabled:

```powershell
uv run rksi
uv run rkpk
uv run rjtt
uv run wsss
```

Run a specific date/cutoff:

```powershell
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local 11:00 --plot --explain
```

Open the local dashboard:

```powershell
uv run rksi-ui
```

## Core Concepts

### Data Files

Historical source CSV files are expected at:

```text
data/rksi/asos.csv
data/rkpk/asos.csv
data/rjtt/RJTT.csv
data/wsss/WSSS.csv
```

Live/manual METAR lines go in:

```text
data/shared/metar.txt
```

Runtime reads prefer DuckDB when available:

```text
artifacts/shared/observations.duckdb
```

Rebuild DuckDB from configured raw CSV files:

```powershell
uv run rksi-sync-duckdb
```

Prepare Open-Meteo M3 API cache when a location has coordinates configured:

```powershell
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode training
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode daily --date 2026-06-21
```

### Cutoff Time

`--cutoff-local` and variables such as `$CUTOFF` mean local time for the
station/config being predicted, not the timezone of the computer running the
command.

Examples:

- `configs/default.yaml` and `configs/rkpk.yaml` use `Asia/Seoul`.
- `configs/rjtt.yaml` uses `Asia/Tokyo`.
- `configs/wsss.yaml` uses `Asia/Singapore`.

So this command means `11:00` in RKSI local time:

```powershell
$CUTOFF = "11:00"
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local $CUTOFF
```

The shortcut commands (`uv run rksi`, `uv run wsss`, etc.) compute today's date
and default cutoff from the station timezone in the config, not from the device
timezone.

Prediction can use any local `HH:MM` cutoff as long as the configured data
source has at least one observation for that local date at or before that time.

## Supported Stations

| Station | Shortcut | Config |
|---|---|---|
| RKSI | `uv run rksi` | `configs/default.yaml` |
| RKPK | `uv run rkpk` | `configs/rkpk.yaml` |
| RJTT | `uv run rjtt` | `configs/rjtt.yaml` |
| WSSS | `uv run wsss` | `configs/wsss.yaml` |

## Daily Operation

Single-station shortcut:

```powershell
uv run wsss
```

Override date/cutoff:

```powershell
uv run wsss --date 2026-06-20 --cutoff-local 10:00
```

Use existing `metar.txt` without fetching:

```powershell
uv run wsss --no-fetch
```

Manual all-station sequence:

```powershell
$DATE = "2026-06-20"
$CUTOFF = "11:00"

uv run rksi-fetch-metar --stations RKSI,RKPK,RJTT,WSSS --hours 48 --output metar.txt
uv run rksi-import-metar --metar-file metar.txt --reference-date $DATE

uv run rksi-predict-heat-risk --config configs/default.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
uv run rksi-predict-heat-risk --config configs/rkpk.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
uv run rksi-predict-heat-risk --config configs/rjtt.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
```

For WSSS, use the WSSS local cutoff you want, for example:

```powershell
uv run rksi-predict-heat-risk --config configs/wsss.yaml --date 2026-06-20 --cutoff-local 10:00 --plot --explain
```

Daily prediction does not require rebuilding the dataset or retraining. Rebuild
and retrain only after adding enough completed historical days, changing
thresholds/cutoffs, changing station config, or changing model code.

## Local UI Dashboard

Run the Streamlit dashboard locally:

```powershell
uv run rksi-ui
```

The UI lets you choose a YAML config/location, then use three tabs:

- `Locations`: create a new station YAML config, optional empty ASOS CSV header, and Open-Meteo API coordinates/cache paths.
- `METAR`: fetch station METAR, import station-scoped rows, sync DuckDB, and verify latest observations.
- `Train / Validate`: prepare Open-Meteo M3 cache, build the heat-risk dataset, train, validate, and inspect metrics/artifacts.
- `Predict`: choose latest database observation, config default cutoff, or a custom cutoff; choose Auto/M1/M3 when supported; then generate the explanation and structured JSON output.

The sidebar also has a `Delete location config` expander. It deletes only the selected YAML config after station-code confirmation; CSV, DuckDB rows, and model artifacts are kept.

Implementation context for future UI edits is in `docs/ui-dashboard-context.md`.

## Training Workflow

Build, train, validate, and predict for one config:

```powershell
uv run rksi-build-heat-risk-dataset --config configs/default.yaml
uv run rksi-train-heat-risk --config configs/default.yaml
uv run rksi-validate-heat-risk --config configs/default.yaml
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-20 --cutoff-local 11:00
```

Use station-specific config paths for other stations.

For retraining an existing station after adding historical data, changing
config, or updating Open-Meteo M3 cache, see
[Retraining An Existing Location](docs/configuration.md#retraining-an-existing-location).

## Documentation

- [CLI Reference](docs/cli-reference.md)
- [Configuration Guide](docs/configuration.md)
- [Heat-Risk Prediction Guide](docs/predict-heat-risk.md)
- [Daily Tmax Expert Report](docs/daily-tmax-expert-report.md)

For adding a new location, start with
[Adding A New Location](docs/configuration.md#adding-a-new-location).

## Development Checks

```powershell
uv run pytest
uv run ruff check .
```
