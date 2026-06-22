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
the M1 feature set plus Open-Meteo daily forecast features, has the lowest
validation MAE. M1 remains available as the METAR/ASOS-only feature-expanded
fallback.

## Model Setup

Station: `RKSI`

Training and validation:

```text
Train period: 2016-01-02 to 2024-05-17
Test period: 2024-05-18 to 2026-06-20
Train rows: 15275
Test rows: 3820
Cutoffs: 09:00, 10:00, 11:00, 12:00, 13:00 local
```

Model layers:

- `M0 heat-risk`: ASOS/METAR-only direct/two-stage remaining-heat model.
- `M1 phase features`: expanded remaining-heat model using phase/plateau,
  climatology, and last-3-days regime features.
- `M3 Open-Meteo`: M1 features plus daily Open-Meteo forecast features
  including forecast Tmax, WMO weather code, precipitation/rain, max wind, and
  gusts.
- `Thermal phase classifier`: predicts `pre_peak_ramp`, `peak_plateau`,
  `post_peak_decline`, or `uncertain_transition`.
- `Late-warming classifiers`: estimate remaining heat `>= 0.5/1/2/3 C`.
- `Future-curve models`: predict `T+30m` through `T+180m`.

## Overall Accuracy

| Model / Metric | Value |
|---|---:|
| Selected operational method | openmeteo |
| Selected Tmax MAE | 0.772 C |
| Selected Tmax RMSE | 1.044 C |
| Selected bias | +0.026 C |
| M0 two-stage Tmax MAE | 0.857 C |
| M1 phase-feature Tmax MAE | 0.819 C |
| M3 Open-Meteo Tmax MAE | 0.772 C |
| Curve-derived Tmax MAE | 0.948 C |
| Observed-max baseline MAE | 1.986 C |

Interpretation:

- `M3 Open-Meteo` improves RKSI backtest MAE by about `0.085 C` versus the
  ASOS/METAR-only M0 two-stage model and about `0.047 C` versus M1-only
  features on this validation window.
- Curve-derived Tmax is worse than `M0`, so curve output should be treated as
  explanatory/diagnostic, not the final Tmax forecast.

## Accuracy By Cutoff

| Cutoff | N | Selected Tmax MAE (C) | Observed-max baseline MAE (C) | Bias (C) |
|---|---:|---:|---:|---:|
| 09:00 | 764 | 1.026 | 3.605 | +0.045 |
| 10:00 | 764 | 0.908 | 2.740 | +0.049 |
| 11:00 | 764 | 0.791 | 1.852 | +0.025 |
| 12:00 | 764 | 0.656 | 1.116 | +0.032 |
| 13:00 | 764 | 0.510 | 0.616 | +0.015 |

Interpretation:

- The model continues to add most value before noon.
- By 13:00, observed maximum is already a strong baseline, but ML still helps
  slightly.
- Bias remains small and positive across cutoffs.

## Thermal Phase Classifier

Current RKSI validation:

```text
Accuracy: 75.3%
```

Confusion matrix order:

```text
pre_peak_ramp, peak_plateau, post_peak_decline, uncertain_transition
```

```text
[[1546,  55,  67,   0],
 [ 185, 201,  72,  83],
 [ 124, 125, 614, 113],
 [   0,  28,  90, 512]]
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
| T+30m | 0.496 |
| T+60m | 0.622 |
| T+90m | 0.734 |
| T+120m | 0.817 |
| T+150m | 0.886 |
| T+180m | 0.938 |

Curve-derived Tmax:

```text
MAE: 0.918 C
```

Interpretation:

- The curve is useful for near-term shape and expert interpretation.
- Error grows with horizon, as expected.
- The curve-derived Tmax is worse than the selected M3 forecast, so it remains
  diagnostic rather than operational.

## Hot-Threshold Probability

| Threshold | Event rate | Brier | Brier Skill Score | ROC AUC |
|---|---:|---:|---:|---:|
| Tmax >= 28 C | 20.7% | 0.024 | 0.857 | 0.995 |
| Tmax >= 29 C | 18.0% | 0.027 | 0.821 | 0.994 |
| Tmax >= 30 C | 14.8% | 0.025 | 0.808 | 0.995 |
| Tmax >= 31 C | 10.4% | 0.024 | 0.751 | 0.993 |

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
| Predicted remaining heat | +0.25 C |
| Prediction method | openmeteo |
| Open-Meteo raw Tmax | 23.1 C |
| M3 corrected Tmax | 28.22 C |
| Practical 80% interval | 28.0 to 29.47 C |
| Thermal phase | post_peak_decline |
| Probability post-peak decline | 77.1% |
| Probability Tmax already reached | 68.2% |
| Late-warming risk | low |
| P(remaining heat >= 2 C) | 1.1% |
| P(remaining heat >= 3 C) | 0.15% |
| P(Tmax >= 29 C) | 9.4% |
| P(Tmax >= 30 C) | 0.09% |

Future curve:

| Local time | Predicted temp |
|---|---:|
| 13:00 | 27.90 C |
| 13:30 | 27.95 C |
| 14:00 | 27.70 C |
| 14:30 | 27.07 C |
| 15:00 | 27.02 C |
| 15:30 | 26.49 C |

Operational reading:

```text
The selected Open-Meteo-enhanced model sees 28 C as likely near or past the day's peak.
Expected additional warming is small after combining Open-Meteo and observed METAR/ASOS state.
The risk of a late +2 C or +3 C jump is low.
The future curve does not support a strong late-warming scenario.
```

## Multi-Station Snapshot

Latest validation MAE. Open-Meteo is currently configured only for RKSI.

| Station | Selected / M0 MAE | M1 MAE | M3 Open-Meteo MAE | Curve Tmax MAE |
|---|---:|---:|---:|---:|
| RKSI | 0.772 C | 0.819 C | 0.772 C | 0.948 C |
| RKPK | 0.957 C | 0.961 C | n/a | 1.608 C |
| RJTT | 0.785 C | 0.762 C | n/a | 0.864 C |
| WSSS | 0.609 C | 0.609 C | n/a | 0.903 C |

Interpretation:

- M3 currently changes RKSI from ASOS/METAR-only M0/M1 to the selected
  operational method.
- Open-Meteo remains useful as both raw external forecast and M3-corrected
  operational signal in reports.
- Curve-derived Tmax is not yet good enough to become the operational forecast.
- Future-curve output is still valuable for expert review of phase and late
  warming.

## Diagnostics For Expert Review

Primary files:

```text
artifacts/heat_risk_diagnostics.png
artifacts/heat_risk_thermal_curve_diagnostics.png
artifacts/heat_risk_validation_report.json
artifacts/heat_risk_top_errors.csv
artifacts/heat_risk_top_error_days.csv
```

Expert review should focus on:

- cases where phase classifier predicts post-peak but actual later warming is
  large;
- late-warming false alarms;
- plateau vs post-peak confusion;
- whether M3 Open-Meteo should be added for other stations after matching
  forecast-history files are available;
- days where raw Open-Meteo Tmax strongly disagrees with observed cutoff state;
- whether curve model should be used only for near-term curve shape, not Tmax.

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
