# RKSI METAR/ASOS Tmax Heat Risk Forecast

Python project for forecasting final daily maximum temperature risk from
historical and newly entered METAR/ASOS observations.

The active workflow is:

- Given observations up to any local cutoff, predict how much higher final
  `Tmax` can still go.
- Estimate final `Tmax`.
- Estimate probability of crossing configured hot thresholds.
- Recommend whether the forecast should be updated at the next cutoff.

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
- `artifacts/heat_risk_diagnostics.png`

## Station Configs

RKSI uses:

```text
configs/default.yaml
```

RKPK uses:

```text
configs/rkpk.yaml
```

Example RKPK commands:

```powershell
uv run rksi-build-heat-risk-dataset --config configs/rkpk.yaml
uv run rksi-train-heat-risk --config configs/rkpk.yaml
uv run rksi-validate-heat-risk --config configs/rkpk.yaml
uv run rksi-predict-heat-risk --config configs/rkpk.yaml --date 2026-06-18 --cutoff-local 12:00
```

## Documentation

- [docs/predict-heat-risk.md](docs/predict-heat-risk.md)
- [docs/daily-tmax-expert-report.md](docs/daily-tmax-expert-report.md)

## Development Checks

```powershell
uv run pytest
uv run ruff check .
```
