# Multi-Cutoff Tmax Remaining Heat And Heat Risk

This workflow answers the operational Tmax heat-risk question:

```text
Given observations up to a local cutoff, how much higher can final Tmax still
go, what is the probability of crossing hot thresholds, and is the next update
worth running?
```

## Model

- Target: `remaining_heat_target_c = final_tmax_c - max_temp_observed_to_cutoff_c`.
- Final prediction: `observed_max_to_cutoff_c + predicted_remaining_heat_c`.
- Algorithm:
  - `HistGradientBoostingRegressor` for remaining heat.
  - `HistGradientBoostingClassifier` per hot threshold.
- Default cutoffs: `09:00` through `13:00` local, every 30 minutes.
- Default thresholds: whole-degree values configured in `configs/default.yaml`.
- Feature set includes temperature, dewpoint, humidity, wind, pressure,
  visibility, cloud, weather-code flags, wind direction regime, and cloud/fog
  clearing signals up to the cutoff.

The model must not use observations after the requested cutoff in features.

## Build Dataset

```powershell
uv run rksi-build-heat-risk-dataset
```

This reads `asos.csv`, builds one row per configured cutoff per local day, and
writes:

```text
artifacts/heat_risk_dataset.parquet
```

## Train

```powershell
uv run rksi-train-heat-risk
```

This writes:

```text
artifacts/heat_risk_model.joblib
artifacts/heat_risk_metrics.json
```

Recent RKSI result on the current dataset:

```text
Test period: 2024-05-16 to 2026-06-17
Tmax MAE: 0.86 C
Tmax RMSE: 1.14 C
Remaining heat MAE: 0.86 C
Observed-max baseline MAE: 1.97 C
```

## Validate

```powershell
uv run rksi-validate-heat-risk
```

This writes:

```text
artifacts/heat_risk_validation_report.json
artifacts/heat_risk_top_errors.csv
artifacts/heat_risk_diagnostics.png
```

The diagnostics image includes:

- error by cutoff;
- actual vs predicted remaining heat;
- threshold probability Brier score;
- expected value of updating at the next cutoff.

## Predict

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:00
```

The cutoff can be any local `HH:MM` value as long as `asos.csv` contains at
least one observation for that local date at or before the cutoff:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 09:00
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:30
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:00
```

Changing `--cutoff-local` for prediction does not require rebuilding the full
dataset or retraining. The command builds one feature row directly from
`asos.csv`.

Example output:

```json
{
  "station": "RKSI",
  "local_date": "2026-06-18",
  "cutoff_local": "12:30",
  "observed_max_to_cutoff_c": 28.0,
  "last_temp_to_cutoff_c": 28.0,
  "predicted_remaining_heat_c": 0.51,
  "predicted_tmax_c": 28.51,
  "prediction_interval_80_low_c": 27.18,
  "prediction_interval_80_high_c": 29.88,
  "next_update_local": "13:00",
  "recommend_update_next_cutoff": true,
  "update_reason": "historical median improvement 0.08C; 80% interval width 2.69C",
  "prob_tmax_ge_28c": 1.0,
  "prob_tmax_ge_29c": 0.114,
  "prob_tmax_ge_30c": 0.0036,
  "prob_tmax_ge_31c": 0.0036
}
```

## Field Meanings

- `observed_max_to_cutoff_c`: highest temperature already observed by cutoff.
- `predicted_remaining_heat_c`: how much higher final Tmax may still go.
- `predicted_tmax_c`: final Tmax forecast.
- `prediction_interval_80_low_c` and `prediction_interval_80_high_c`: rough
  uncertainty interval from historical residuals.
- `prob_tmax_ge_*`: probability final Tmax reaches or exceeds that threshold.
  If the observed max has already crossed the threshold, this is forced to `1`.
- `recommend_update_next_cutoff`: whether the next cutoff is worth updating,
  based on historical improvement and current interval width.

## Live-Day Sequence

1. Add new METAR observations into `metar.txt`.
2. Import:

```powershell
uv run rksi-import-metar --metar-file metar.txt --reference-date 2026-06-18
```

3. Predict directly:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:00
```

The predict command can build a single cutoff feature row directly from
`asos.csv`, so rebuilding the full heat-risk dataset is not required for every
live prediction. Rebuild/retrain when adding enough completed days to improve
the model.

## Change Training Cutoffs Or Thresholds

There are two separate cutoff concepts:

- Prediction cutoff: passed with `--cutoff-local`; use this for day-to-day
  forecasts and live updates.
- Training cutoffs: configured in `heat_risk_cutoffs`; use this when you want
  the model and validation report to learn/evaluate a different set of cutoffs.

Edit `configs/default.yaml`:

```yaml
heat_risk_cutoffs:
  - "06:00"
  - "07:00"
  - "08:00"
  - "09:00"
  - "10:00"
heat_risk_thresholds_c:
  - 28.0
  - 29.0
  - 30.0
  - 31.0
```

Then rebuild, train, and validate the heat-risk workflow.

```powershell
uv run rksi-build-heat-risk-dataset
uv run rksi-train-heat-risk
uv run rksi-validate-heat-risk
```

Avoid very dense `0.5 C` threshold spacing unless there is a clear operational
reason. METAR temperatures are usually reported at integer Celsius precision,
so neighboring half-degree thresholds often carry little extra information.
Prediction output is also forced to be monotonic: a higher threshold cannot
have a higher probability than a lower threshold.
