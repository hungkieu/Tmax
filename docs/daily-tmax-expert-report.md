# RKSI Thermal Phase / Tmax Heat-Risk Expert Report

## Executive Summary

The current operational question is broader than direct Tmax regression:

```text
Given observations up to a local cutoff, what thermal phase is today in, how
much additional warming remains, what future temperature curve is likely, and
what is the final Tmax risk distribution?
```

Operational forecast is now selected by validation MAE:

```text
predicted_tmax_c = selected method output
```

For RKSI, the selected method is currently `openmeteo` because M3, trained from
the M1 feature set plus Open-Meteo daily and hourly API forecast features, has the lowest
validation MAE. M1 remains available as the METAR/ASOS-only feature-expanded
fallback.

## Model Setup

Station: `RKSI`

Training and validation:

```text
Train period: 2016-01-02 to 2024-05-17
Test period: 2024-05-19 to 2026-06-21
Train rows: 15280
Test rows: 3820
Cutoffs: 09:00, 10:00, 11:00, 12:00, 13:00 local
```

Model layers:

- `M0 heat-risk`: ASOS/METAR-only direct/two-stage remaining-heat model.
- `M1 phase features`: expanded remaining-heat model using phase/plateau,
  climatology, and last-3-days regime features.
- `M3 Open-Meteo`: M1 features plus daily and hourly Open-Meteo API forecast features
  including forecast Tmax, hourly temperature shape, WMO weather code,
  precipitation/rain, wind, gusts, cloud, visibility, and precipitation
  probability.
- `Thermal phase classifier`: predicts `pre_peak_ramp`, `peak_plateau`,
  `post_peak_decline`, or `uncertain_transition`.
- `Late-warming classifiers`: estimate remaining heat `>= 0.5/1/2/3 C`.
- `Future-curve models`: predict `T+30m` through `T+180m`.

## Overall Accuracy

| Model / Metric | Value |
|---|---:|
| Selected operational method | openmeteo |
| Selected Tmax MAE | 0.764 C |
| Selected Tmax RMSE | 1.038 C |
| Selected bias | +0.004 C |
| M0 two-stage Tmax MAE | 0.851 C |
| M1 phase-feature Tmax MAE | 0.818 C |
| M3 Open-Meteo Tmax MAE | 0.764 C |
| M3 daily-only Tmax MAE | 0.767 C |
| M3 daily+hourly Tmax MAE | 0.764 C |
| Curve-derived Tmax MAE | 0.945 C |
| Observed-max baseline MAE | 1.987 C |

Interpretation:

- `M3 Open-Meteo` improves RKSI backtest MAE by about `0.086 C` versus the
  ASOS/METAR-only M0 two-stage model and about `0.053 C` versus M1-only
  features on this validation window.
- The hourly Open-Meteo features improve M3 slightly versus daily-only
  Open-Meteo: `0.764 C` vs `0.767 C` MAE. The gain is small, but it is
  positive and the selected M3 variant is `hourly`.
- Curve-derived Tmax is worse than `M0`, so curve output should be treated as
  explanatory/diagnostic, not the final Tmax forecast.

## Accuracy By Cutoff

| Cutoff | N | Selected Tmax MAE (C) | Observed-max baseline MAE (C) | Bias (C) |
|---|---:|---:|---:|---:|
| 09:00 | 764 | 1.039 | 3.607 | +0.020 |
| 10:00 | 764 | 0.892 | 2.742 | +0.015 |
| 11:00 | 764 | 0.775 | 1.853 | -0.020 |
| 12:00 | 764 | 0.630 | 1.118 | -0.019 |
| 13:00 | 764 | 0.486 | 0.616 | +0.022 |

Interpretation:

- The model continues to add most value before noon.
- By 13:00, observed maximum is already a strong baseline, but ML still helps
  slightly.
- Bias remains small and positive across cutoffs.

## Thermal Phase Classifier

Current RKSI validation:

```text
Accuracy: 74.5%
```

Confusion matrix order:

```text
pre_peak_ramp, peak_plateau, post_peak_decline, uncertain_transition
```

```text
[[1534,  60,  72,   0],
 [ 194, 198,  67,  81],
 [ 140, 118, 600, 124],
 [   0,  33,  84, 515]]
```

Interpretation:

- `pre_peak_ramp` is detected well.
- `peak_plateau` remains the hardest class, often confused with ramp or
  post-peak states.
- This is operationally acceptable for v1 because phase is used for explanation
  and risk context, not to override the main forecast.

## Late-Warming Risk

Probability model quality:

| Event | Event rate | Brier | Brier Skill Score | ROC AUC |
|---|---:|---:|---:|---:|
| Remaining heat >= 0.5 C | 74.4% | 0.109 | 0.426 | 0.895 |
| Remaining heat >= 1.0 C | 64.9% | 0.121 | 0.470 | 0.899 |
| Remaining heat >= 2.0 C | 42.7% | 0.120 | 0.512 | 0.911 |
| Remaining heat >= 3.0 C | 27.1% | 0.104 | 0.472 | 0.913 |

Operational detection at probability threshold `30%`:

| Event | Recall | Precision | False alarm rate |
|---|---:|---:|---:|
| Remaining heat >= 2 C | 90.2% | 71.3% | 28.7% |
| Remaining heat >= 3 C | 79.6% | 66.8% | 33.2% |

Interpretation:

- The late-warming layer is useful: it captures most `>=2 C` and `>=3 C`
  events.
- False alarms are not trivial, so the risk label should be read as a warning,
  not as a deterministic correction.

Risk label policy:

```text
low:      P(remaining_heat >= 2 C) < 10%
moderate: 10% to <30%
elevated: 30% to <50%
high:     >=50%
```

## Future Curve Model

Horizon MAE:

| Horizon | MAE (C) |
|---|---:|
| T+30m | 0.497 |
| T+60m | 0.626 |
| T+90m | 0.742 |
| T+120m | 0.831 |
| T+150m | 0.922 |
| T+180m | 0.978 |

Curve-derived Tmax:

```text
MAE: 0.945 C
```

Interpretation:

- The curve is useful for near-term shape and expert interpretation.
- Error grows with horizon, as expected.
- The curve-derived Tmax is worse than the selected M3 forecast, so it remains
  diagnostic rather than operational.

## Hot-Threshold Probability

| Threshold | Event rate | Brier | Brier Skill Score | ROC AUC |
|---|---:|---:|---:|---:|
| Tmax >= 28 C | 20.9% | 0.025 | 0.853 | 0.995 |
| Tmax >= 29 C | 18.1% | 0.026 | 0.830 | 0.994 |
| Tmax >= 30 C | 14.9% | 0.025 | 0.814 | 0.995 |
| Tmax >= 31 C | 10.5% | 0.022 | 0.768 | 0.993 |

Interpretation:

- Hot-threshold probabilities remain strong.
- Output includes raw and monotonic operational probabilities.
- If the observed maximum has already crossed a threshold, operational
  probability for that threshold is forced to `1.0`.

## Live RKSI Example: 2026-06-18 12:30

Command:

```powershell
uv run rksi-predict-heat-risk --config configs/default.yaml --date 2026-06-18 --cutoff-local 12:30
```

Key output:

| Field | Value |
|---|---:|
| Last observation | 2026-06-18 12:30 |
| Data fresh enough | true |
| Observed max to cutoff | 28.0 C |
| Last temp to cutoff | 28.0 C |
| Predicted remaining heat | +0.16 C |
| Prediction method | openmeteo |
| Open-Meteo raw Tmax | 23.1 C |
| M3 corrected Tmax | 28.16 C |
| Practical 80% interval | 28.0 to 29.45 C |
| Thermal phase | post_peak_decline |
| Probability post-peak decline | 51.3% |
| Probability Tmax already reached | 78.2% |
| Late-warming risk | low |
| P(remaining heat >= 2 C) | 1.9% |
| P(remaining heat >= 3 C) | 0.23% |
| P(Tmax >= 29 C) | 15.5% |
| P(Tmax >= 30 C) | 0.05% |

Future curve:

| Local time | Predicted temp |
|---|---:|
| 13:00 | 28.01 C |
| 13:30 | 27.98 C |
| 14:00 | 27.78 C |
| 14:30 | 27.41 C |
| 15:00 | 27.25 C |
| 15:30 | 27.16 C |

Operational reading:

```text
The selected Open-Meteo-enhanced model sees 28 C as likely near or past the day's peak.
Expected additional warming is small after combining Open-Meteo and observed METAR/ASOS state.
The risk of a late +2 C or +3 C jump is low.
The future curve does not support a strong late-warming scenario.
```

## Multi-Station Snapshot

Latest validation MAE. Open-Meteo can be configured per location with API coordinates.

| Station | Selected / M0 MAE | M1 MAE | M3 Open-Meteo MAE | Curve Tmax MAE |
|---|---:|---:|---:|---:|
| RKSI | 0.764 C | 0.818 C | 0.764 C | 0.945 C |
| RKPK | 0.957 C | 0.961 C | n/a | 1.608 C |
| RJTT | 0.785 C | 0.762 C | n/a | 0.864 C |
| WSSS | 0.609 C | 0.609 C | n/a | 0.903 C |

Interpretation:

- M3 currently changes RKSI from ASOS/METAR-only M0/M1 to the selected
  operational method; the selected M3 variant is daily+hourly.
- Open-Meteo remains useful as both raw external forecast and M3-corrected
  operational signal in reports.
- Curve-derived Tmax is not yet good enough to become the operational forecast.
- Future-curve output is still valuable for expert review of phase and late
  warming.

## Diagnostics For Expert Review

Primary files:

```text
artifacts/rksi/heat_risk_diagnostics.png
artifacts/rksi/heat_risk_thermal_curve_diagnostics.png
artifacts/rksi/heat_risk_validation_report.json
artifacts/rksi/heat_risk_top_errors.csv
artifacts/rksi/heat_risk_top_error_days.csv
```

Expert review should focus on:

- cases where phase classifier predicts post-peak but actual later warming is
  large;
- late-warming false alarms;
- plateau vs post-peak confusion;
- whether M3 Open-Meteo should be added for other stations after coordinates
  are entered and API cache is prepared;
- days where raw Open-Meteo Tmax strongly disagrees with observed cutoff state;
- whether curve model should be used only for near-term curve shape, not Tmax.

## Daily Operation

Daily prediction does not require rebuilding the full dataset or retraining the
model. Normal daily flow for RKSI:

1. Fetch/import recent METAR and sync DuckDB.
2. Fetch Open-Meteo daily forecast cache for the forecast date.
3. Run prediction at the selected local cutoff.
4. Read `predicted_tmax_c`, `openmeteo_forecast_tmax_c`,
   `openmeteo_predicted_tmax_c`, `thermal_phase`, `late_warming_risk`, and
   threshold probabilities.

CLI form:

```powershell
uv run rksi-fetch-metar --stations RKSI,RKPK,RJTT,WSSS --hours 48 --output data/shared/metar.txt
uv run rksi-import-metar --config configs/default.yaml --metar-file data/shared/metar.txt --reference-date YYYY-MM-DD
uv run rksi-sync-duckdb --config configs/default.yaml
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode daily --date YYYY-MM-DD
uv run rksi-predict-heat-risk --config configs/default.yaml --date YYYY-MM-DD --cutoff-local HH:MM --plot --explain
```

Shortcut form:

```powershell
uv run rksi --date YYYY-MM-DD --cutoff-local HH:MM
```

The shortcut handles METAR fetch/import/sync and prediction. It now also fetches
Open-Meteo daily forecast on demand if the date is not already cached.

Retrain only when at least one of these changes:

- new completed historical days are worth incorporating into model calibration;
- station config, thresholds, cutoffs, or feature/model code changed;
- Open-Meteo API cache range was extended and you want new validation metrics;
- adding M3 for another location after entering coordinates in the UI.

Retraining sequence:

```powershell
uv run rksi-fetch-openmeteo --config configs/default.yaml --mode training --force
uv run rksi-build-heat-risk-dataset --config configs/default.yaml
uv run rksi-train-heat-risk --config configs/default.yaml
uv run rksi-validate-heat-risk --config configs/default.yaml
```

## Recommendation

Use this operational command:

```powershell
uv run rksi-predict-heat-risk --config configs/default.yaml --date YYYY-MM-DD --cutoff-local HH:MM
```

For now:

- Use `predicted_tmax_c` from the selected validation method. For RKSI this is
  currently M3/Open-Meteo.
- Read `openmeteo_forecast_tmax_c` as the raw external forecast and
  `openmeteo_predicted_tmax_c` as the model-corrected forecast after combining
  Open-Meteo with observed cutoff state.
- Use `thermal_phase`, `future_curve`, and `late_warming_risk` as expert
  context.
- Do not switch to curve-derived Tmax until backtest improves over the selected
  method.
