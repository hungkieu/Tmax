# METAR/ASOS Tmax Heat Risk Forecast

Python project for forecasting final daily maximum temperature risk from
historical and newly entered METAR/ASOS observations.

The active workflow is:

- Given observations up to any local cutoff, predict how much higher final
  `Tmax` can still go.
- Estimate final `Tmax`.
- Classify the current thermal phase and late-warming risk.
- Predict a 30-minute future temperature curve through the next 3 hours.
- Estimate probability of crossing configured hot thresholds.
- Recommend whether the forecast should be updated at the next cutoff.

`predicted_tmax_c` remains the operational M0 heat-risk forecast. Thermal phase,
late-warming risk, and future curve are additional expert-context outputs; the
curve model does not override the official forecast in v1.

## Setup

Install `uv` if it is not already available:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install dependencies:

```powershell
uv sync --dev
```

## Source Data

The historical source file is expected at:

```text
asos.csv
RJTT.csv
WSSS.csv
```

New live/manual METAR observations can be placed in:

```text
metar.txt
```

Import `metar.txt` into `asos.csv`:

```powershell
uv run rksi-import-metar --metar-file metar.txt --reference-date 2026-06-18
```

`--reference-date` is a UTC date used to infer month/year from METAR time
tokens like `172300Z`. Re-running the same import is safe: rows already present
for the same `(station, valid)` are skipped.

## DuckDB Storage

Raw CSV files remain the backup/source files. Runtime reads can use DuckDB for
cleaner and faster station filtering.

One-time sync from the available CSV files:

```powershell
uv run rksi-sync-duckdb
```

This creates:

```text
artifacts/observations.duckdb
```

The default sync loads:

```text
asos.csv
RJTT.csv
WSSS.csv
```

Current supported station configs:

| Station | Config |
|---|---|
| RKSI | `configs/default.yaml` |
| RKPK | `configs/rkpk.yaml` |
| RJTT | `configs/rjtt.yaml` |
| WSSS | `configs/wsss.yaml` |

## Daily Run

After setup and training have been done once, the daily operation only needs
new METAR data and prediction. You do not need to retrain every day.

Change only `$DATE` and `$CUTOFF`, then run this whole block once in
PowerShell:

```powershell
$DATE = "2026-06-19"
$CUTOFF = "12:30"

uv run rksi-fetch-metar --stations RKSI,RKPK,RJTT,WSSS --hours 48 --output metar.txt
uv run rksi-import-metar --metar-file metar.txt --reference-date $DATE

uv run rksi-predict-heat-risk --config configs/default.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
uv run rksi-predict-heat-risk --config configs/rkpk.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
uv run rksi-predict-heat-risk --config configs/rjtt.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
uv run rksi-predict-heat-risk --config configs/wsss.yaml --date $DATE --cutoff-local $CUTOFF --plot --explain
```

The fetch command overwrites `metar.txt` with recent METAR lines. The import
command appends new rows to the configured CSV and upserts the same rows into
`artifacts/observations.duckdb`; duplicate `(station, valid)` rows are skipped.
The `--plot` option writes a temperature-curve PNG under `artifacts/`. The
`--explain` option prints a Vietnamese explanation after the JSON output for
non-technical readers.

Daily prediction does not require rebuilding the dataset or retraining the
model. Rebuild and retrain only after adding enough completed historical days,
changing thresholds/cutoffs, changing station config, or changing model code.

If DuckDB ever gets out of sync with raw CSV files, rebuild it:

```powershell
uv run rksi-sync-duckdb
```

## Heat-Risk Workflow

Build the multi-cutoff dataset:

```powershell
uv run rksi-build-heat-risk-dataset
```

Train:

```powershell
uv run rksi-train-heat-risk
```

Validate:

```powershell
uv run rksi-validate-heat-risk
```

Predict for a local date and cutoff:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:00
```

Predict and write a daily temperature-curve chart:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --plot
```

This writes a PNG under `artifacts/` showing observed temperatures, forecast
curve, and the cutoff marker.

Predict with a human-readable explanation:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --explain
```

You can combine both:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --plot --explain
```

The prediction cutoff can be any local `HH:MM` value as long as `asos.csv`
contains at least one observation for that local date at or before the cutoff.

Examples:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 09:00
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:30
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:00
```

Changing `--cutoff-local` for one prediction does not require rebuilding or
retraining. To change which cutoffs are used during training and validation,
edit `heat_risk_cutoffs` in `configs/default.yaml`, then rebuild/train/validate.

## Output Artifacts

Active artifacts:

- `artifacts/heat_risk_dataset.parquet`
- `artifacts/heat_risk_model.joblib`
- `artifacts/heat_risk_metrics.json`
- `artifacts/heat_risk_validation_report.json`
- `artifacts/heat_risk_top_errors.csv`
- `artifacts/heat_risk_top_error_days.csv`
- `artifacts/heat_risk_diagnostics.png`
- `artifacts/heat_risk_thermal_curve_diagnostics.png`

Other station configs write the same artifact types with a station prefix, for
example:

```text
artifacts/rjtt_heat_risk_model.joblib
artifacts/rjtt_heat_risk_metrics.json
artifacts/rjtt_heat_risk_validation_report.json
artifacts/rjtt_heat_risk_thermal_curve_diagnostics.png
artifacts/wsss_heat_risk_model.joblib
artifacts/wsss_heat_risk_metrics.json
artifacts/wsss_heat_risk_validation_report.json
artifacts/wsss_heat_risk_thermal_curve_diagnostics.png
```

## Station Configs

RKSI uses:

```text
configs/default.yaml
```

RKPK uses:

```text
configs/rkpk.yaml
```

RJTT uses:

```text
configs/rjtt.yaml
```

WSSS uses:

```text
configs/wsss.yaml
```

Example RKPK commands:

```powershell
uv run rksi-build-heat-risk-dataset --config configs/rkpk.yaml
uv run rksi-train-heat-risk --config configs/rkpk.yaml
uv run rksi-validate-heat-risk --config configs/rkpk.yaml
uv run rksi-predict-heat-risk --config configs/rkpk.yaml --date 2026-06-18 --cutoff-local 12:00
```

Example RJTT/WSSS commands:

```powershell
uv run rksi-build-heat-risk-dataset --config configs/rjtt.yaml
uv run rksi-train-heat-risk --config configs/rjtt.yaml
uv run rksi-validate-heat-risk --config configs/rjtt.yaml

uv run rksi-build-heat-risk-dataset --config configs/wsss.yaml
uv run rksi-train-heat-risk --config configs/wsss.yaml
uv run rksi-validate-heat-risk --config configs/wsss.yaml
```

## Documentation

- [docs/predict-heat-risk.md](docs/predict-heat-risk.md)
- [docs/configuration.md](docs/configuration.md)
- [docs/daily-tmax-expert-report.md](docs/daily-tmax-expert-report.md)

## Development Checks

```powershell
uv run pytest
uv run ruff check .
```
