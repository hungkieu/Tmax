# RKSI Tmax Heat-Risk Expert Evaluation Report

## Executive Summary

The active model answers this operational question:

```text
Given observations up to any local cutoff, how much higher can final Tmax still
go, what is the probability of crossing configured hot thresholds, and should
the forecast be updated at the next cutoff?
```

The project exposes a single Tmax-facing workflow:

```powershell
uv run rksi-predict-heat-risk --date YYYY-MM-DD --cutoff-local HH:MM
```

## Model Setup

Station: `RKSI`

Training data:

```text
Train period: 2016-01-02 to 2024-05-15
Test period: 2024-05-16 to 2026-06-17
Train rows: 27441
Test rows: 6867
Feature count: 98
```

Cutoffs used for training and validation:

```text
09:00, 09:30, 10:00, 10:30, 11:00,
11:30, 12:00, 12:30, 13:00 local
```

Target:

```text
remaining_heat_target_c = final_tmax_c - observed_max_to_cutoff_c
```

Final Tmax prediction:

```text
predicted_tmax_c = observed_max_to_cutoff_c + predicted_remaining_heat_c
```

Algorithms:

- `HistGradientBoostingRegressor` for remaining heat.
- `HistGradientBoostingClassifier` for each hot threshold.

Recent feature additions:

- wind direction circular encoding and last-observation wind regime flags;
- cloud clearing/increasing signal from first to last pre-cutoff observation;
- fog cleared/developed signal from first to last pre-cutoff observation.

## Overall Accuracy

| Metric | Value |
|---|---:|
| Remaining heat MAE | 0.857 C |
| Final Tmax MAE | 0.857 C |
| Final Tmax RMSE | 1.135 C |
| Final Tmax bias | +0.044 C |
| Baseline MAE using only observed max so far | 1.974 C |

Interpretation:

- The model is materially better than assuming the current observed maximum
  remains the final Tmax.
- Overall bias is small.
- The target is operationally interpretable: expected additional warming after
  the cutoff.

## Accuracy By Cutoff

| Cutoff | N | Tmax MAE (C) | Observed-max baseline MAE (C) | Bias (C) |
|---|---:|---:|---:|---:|
| 09:00 | 763 | 1.181 | 3.611 | +0.016 |
| 09:30 | 763 | 1.082 | 3.212 | +0.043 |
| 10:00 | 763 | 1.006 | 2.746 | +0.016 |
| 10:30 | 763 | 0.926 | 2.280 | +0.064 |
| 11:00 | 763 | 0.855 | 1.856 | +0.036 |
| 11:30 | 763 | 0.787 | 1.471 | +0.045 |
| 12:00 | 763 | 0.700 | 1.119 | +0.063 |
| 12:30 | 763 | 0.622 | 0.855 | +0.040 |
| 13:00 | 763 | 0.555 | 0.619 | +0.071 |

Interpretation:

- Accuracy improves as the cutoff gets later.
- ML adds clear value from `09:00` through `12:30`.
- By `13:00`, the observed maximum baseline is already strong, but ML still
  improves MAE slightly.

## Feature Experiment: Wind Regime And Cloud/Fog Clearing

The model was retrained after adding wind direction regime and cloud/fog
clearing features.

Comparison on the shared `10:00` to `13:00` cutoff set:

| Cutoff | MAE before (C) | MAE after (C) | Delta (C) |
|---|---:|---:|---:|
| 10:00 | 1.025 | 1.006 | -0.019 |
| 10:30 | 0.933 | 0.926 | -0.007 |
| 11:00 | 0.866 | 0.855 | -0.011 |
| 11:30 | 0.790 | 0.787 | -0.003 |
| 12:00 | 0.701 | 0.700 | -0.002 |
| 12:30 | 0.616 | 0.622 | +0.006 |
| 13:00 | 0.544 | 0.555 | +0.011 |
| Average | 0.782 | 0.779 | -0.003 |

Interpretation:

- The new features improve earlier cutoffs slightly.
- They do not materially improve later cutoffs, and `12:30`/`13:00` became a
  little worse.
- Overall impact is positive but very small. These features are worth keeping
  as meteorologically meaningful context, but they are not a major accuracy
  breakthrough.

## Hot-Threshold Probability Quality

| Threshold | Event rate | Brier score | ROC AUC |
|---|---:|---:|---:|
| Tmax >= 28 C | 20.7% | 0.025 | 0.995 |
| Tmax >= 29 C | 18.0% | 0.029 | 0.994 |
| Tmax >= 30 C | 14.8% | 0.027 | 0.994 |
| Tmax >= 31 C | 10.4% | 0.025 | 0.992 |

Interpretation:

- Threshold ranking skill is strong on the test period.
- Half-degree thresholds were removed because METAR temperature precision is
  usually integer Celsius and adjacent 0.5 C probabilities added little value.
- Threshold probabilities are forced monotonic, so a higher threshold cannot
  have higher probability than a lower threshold.
- If the threshold has already been crossed by the observed maximum before the
  cutoff, prediction output forces that probability to `1.0`.

## Forecast Update Value

| Current cutoff | Next cutoff | Median abs-error improvement (C) | Update helped rate |
|---|---|---:|---:|
| 09:00 | 09:30 | 0.112 | 57.8% |
| 09:30 | 10:00 | 0.056 | 53.6% |
| 10:00 | 10:30 | 0.059 | 55.0% |
| 10:30 | 11:00 | 0.046 | 52.9% |
| 11:00 | 11:30 | 0.082 | 56.5% |
| 11:30 | 12:00 | 0.115 | 60.4% |
| 12:00 | 12:30 | 0.113 | 59.0% |
| 12:30 | 13:00 | 0.079 | 58.8% |

Interpretation:

- Half-hour updates from `09:00` to `13:00` are usually modest but positive.
- The prediction command also considers interval width. A wide interval can
  trigger `recommend_update_next_cutoff = true` even when historical median
  improvement is modest.

## Live Example: 2026-06-18 At 12:30

Command:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30
```

Key output:

| Field | Value | Meaning |
|---|---:|---|
| `observed_max_to_cutoff_c` | 28.0 C | Highest observed temperature by 12:30 |
| `last_temp_to_cutoff_c` | 28.0 C | Latest temperature at/before 12:30 |
| `predicted_remaining_heat_c` | 0.51 C | Expected additional warming |
| `predicted_tmax_c` | 28.51 C | Final Tmax forecast |
| `prediction_interval_80_low_c` | 27.18 C | Raw lower interval bound |
| `prediction_interval_80_high_c` | 29.88 C | Upper interval bound |
| `next_update_local` | 13:00 | Suggested next forecast time |
| `recommend_update_next_cutoff` | true | Updating at 13:00 is worthwhile |

Operational reading:

```text
Observed max so far: 28.0 C
Expected final Tmax: about 28.5 C
Practical 80% interval: 28.0 C to 29.9 C
```

The raw lower interval is below the already observed maximum. Operationally,
clip the lower bound to the observed maximum because final Tmax cannot be lower
than a temperature already observed earlier in the day.

Hot-threshold probabilities:

| Threshold | Probability |
|---|---:|
| Tmax >= 28 C | 100.0% |
| Tmax >= 29 C | 11.4% |
| Tmax >= 30 C | 0.36% |
| Tmax >= 31 C | 0.36% |

## Diagnostics For Expert Review

Primary diagnostics:

```text
artifacts/heat_risk_diagnostics.png
artifacts/heat_risk_validation_report.json
artifacts/heat_risk_top_errors.csv
```

The largest current heat-risk errors include early-day cases where post-cutoff
warming behaved very differently from the morning signal:

- sharp late warming after a cool morning;
- over-predicted warming on days where Tmax was already reached early;
- precipitation, fog, cloud, or wind-regime cases.

Largest current test-set error:

| Date | Cutoff | Actual Tmax (C) | Predicted Tmax (C) | Error (C) |
|---|---|---:|---:|---:|
| 2025-03-26 | 09:00 | 22.0 | 15.19 | -6.81 |

This should be reviewed meteorologically before changing the model, because
extreme early-cutoff errors may represent regime shifts that are not visible in
METAR-only features.

## Final Recommendation

Use this as the single operational interface:

```powershell
uv run rksi-predict-heat-risk --date YYYY-MM-DD --cutoff-local HH:MM
```

For new observations:

```powershell
uv run rksi-import-metar --metar-file metar.txt --reference-date YYYY-MM-DD
uv run rksi-predict-heat-risk --date YYYY-MM-DD --cutoff-local HH:MM
```

Retraining is not required for every live prediction. Rebuild and retrain when
new completed historical days have accumulated or when changing feature/model
settings.
