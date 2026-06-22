# Multi-Cutoff Tmax Remaining Heat And Heat Risk

This workflow answers the operational Tmax heat-risk question:

```text
Given observations up to a local cutoff, what thermal phase is today in, how
much higher can final Tmax still go, what is the likely future temperature
curve, and is the next update worth running?
```

## Model

- Target: `remaining_heat_target_c = final_tmax_c - max_temp_observed_to_cutoff_c`.
- Final prediction: `observed_max_to_cutoff_c + predicted_remaining_heat_c`.
- Algorithm:
  - `M0` direct/two-stage `HistGradientBoostingRegressor` for ASOS/METAR-only
    remaining heat.
  - `M1` feature-expanded regressor using phase/plateau, weather-suppression,
    climatology, and last-3-day regime features.
  - `HistGradientBoostingClassifier` for thermal phase and hot thresholds.
  - horizon-specific regressors for `T+30m` through `T+180m`.
  - `M3` Open-Meteo enhanced regressor, trained from the M1 feature set plus
    daily Open-Meteo forecast features such as forecast Tmax,
    rain/precipitation, weather code, wind speed, and gusts.
- Default training/validation cutoffs: `09:00`, `10:00`, `11:00`,
  `12:00`, and `13:00` local.
- Default thresholds: whole-degree values configured in `configs/default.yaml`.
- Feature set includes temperature, dewpoint, humidity, wind, pressure,
  visibility, cloud, weather-code flags, wind direction regime, and cloud/fog
  clearing signals up to the cutoff.
- Feature-expanded models also use phase/plateau signals, last-3-day regime
  prior, and prior-only monthly climatology.
- Open-Meteo features are optional. If `openmeteo_history_csv` or the live
  `{date}` file is missing, prediction still runs with imputed values and
  marks `openmeteo_features_available: false`.
- Remaining heat prediction selects the operational method by validation MAE:
  - classify whether meaningful warming remains;
  - predict remaining heat conditional on warming continuing;
  - compare direct, two-stage, M1, and Open-Meteo enhanced regressors.

The model must not use observations after the requested cutoff in features.
Future temperatures after cutoff are targets only.

## Build Dataset

```powershell
uv run rksi-build-heat-risk-dataset
```

This reads DuckDB when available, otherwise the configured CSV source, builds
one row per configured cutoff per local day, and writes:

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

Recent RKSI result on the current dataset after training M3 from M1 plus
Open-Meteo:

```text
Test period: 2024-05-18 to 2026-06-20
Selected method: openmeteo
Tmax MAE: 0.772 C
Tmax RMSE: 1.044 C
Remaining heat MAE: 0.772 C
Direct remaining heat MAE: 0.86 C
Two-stage remaining heat MAE: 0.857 C
M0 two-stage Tmax MAE: 0.857 C
M1 phase-feature Tmax MAE: 0.819 C
M3 Open-Meteo Tmax MAE: 0.772 C
Curve-derived Tmax MAE: 0.948 C
Observed-max baseline MAE: 1.99 C
```

## Validate

```powershell
uv run rksi-validate-heat-risk
```

This writes:

```text
artifacts/heat_risk_validation_report.json
artifacts/heat_risk_top_errors.csv
artifacts/heat_risk_top_error_days.csv
artifacts/heat_risk_diagnostics.png
artifacts/heat_risk_thermal_curve_diagnostics.png
```

The diagnostics image includes:

- error by cutoff;
- actual vs predicted remaining heat;
- threshold probability Brier score;
- remaining-heat probability metrics;
- interval coverage by cutoff;
- expected value of updating at the next cutoff;
- thermal phase confusion matrix;
- future-curve MAE by horizon;
- late-warming precision/recall.

## Predict

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:00
```

`--date` and `--cutoff-local` are interpreted in the timezone configured for
the selected station. They are not device-local time and not UTC. For example,
`--config configs/wsss.yaml --cutoff-local 10:00` means `10:00` in
`Asia/Singapore`.

To also create a daily temperature-curve PNG:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --plot
```

To print a plain-language Vietnamese explanation after the JSON output:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --explain
```

To create both the chart and the explanation:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --plot --explain
```

You can provide a custom path:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --plot artifacts/rksi_2026-06-18_1230_curve.png
```

To ask "I bet X C will not be today's highest temperature; what is my win
probability?", pass `--bet-temp-c`. The bet wins when final `Tmax` is hotter
than that value:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:30 --bet-temp-c 29 --explain
```

The cutoff can be any local `HH:MM` value as long as the configured data source
contains at least one observation for that local date at or before the cutoff:

```powershell
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 09:00
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 10:30
uv run rksi-predict-heat-risk --date 2026-06-18 --cutoff-local 12:00
```

Changing `--cutoff-local` for prediction does not require rebuilding the full
dataset or retraining. The command builds one feature row directly from the
configured data source.

Example output:

```json
{
  "station": "RKSI",
  "local_date": "2026-06-18",
  "cutoff_local": "12:30",
  "observed_max_to_cutoff_c": 28.0,
  "last_temp_to_cutoff_c": 28.0,
  "predicted_remaining_heat_c": 0.28,
  "prediction_method": "openmeteo",
  "openmeteo_forecast_tmax_c": 28.4,
  "openmeteo_predicted_remaining_heat_c": 0.31,
  "openmeteo_predicted_tmax_c": 28.31,
  "openmeteo_features_available": true,
  "predicted_tmax_c": 28.28,
  "thermal_phase": "post_peak_decline",
  "prob_pre_peak_ramp": 0.25,
  "prob_peak_plateau": 0.20,
  "prob_post_peak_decline": 0.55,
  "future_curve": {
    "2026-06-18 13:00": 27.9,
    "2026-06-18 13:30": 28.1,
    "2026-06-18 14:00": 28.0
  },
  "curve_predicted_tmax_c": 28.1,
  "late_warming_risk": "low",
  "warming_strength": "no_or_weak_warming",
  "prob_no_or_weak_warming": 0.766,
  "prob_mild_warming": 0.224,
  "prob_strong_warming": 0.009,
  "prob_extreme_warming": 0.001,
  "late_warming_warning": "low",
  "tail_risk_upper_c": 29.62,
  "regime_break_score": 2.8,
  "regime_break_type": "similar_to_recent",
  "last3_weight_hint": 0.53,
  "prediction_interval_80_low_raw_c": 26.91,
  "prediction_interval_80_low_practical_c": 28.0,
  "prediction_interval_80_high_c": 29.62,
  "next_update_local": "13:00",
  "recommend_update_next_cutoff": true,
  "prob_tmax_already_reached": 0.766,
  "prob_remaining_heat_ge_0_5": 0.234,
  "prob_remaining_heat_ge_1_0": 0.234,
  "prob_remaining_heat_ge_2_0": 0.010,
  "prob_remaining_heat_ge_3_0": 0.001,
  "prob_tmax_ge_28c": 1.0,
  "prob_tmax_ge_29c": 0.075,
  "prob_tmax_ge_30c": 0.002,
  "prob_tmax_ge_31c": 0.0003,
  "not_highest_bet": {
    "bet_temp_c": 29.0,
    "question": "x_c_will_not_be_today_highest_temperature",
    "win_condition": "final_tmax_c > bet_temp_c",
    "win_probability": 0.075,
    "lose_probability": 0.925,
    "required_remaining_heat_c": 1.0,
    "probability_basis": "final_tmax_threshold_classifier_interpolated"
  }
}
```

## Field Meanings

- `observed_max_to_cutoff_c`: highest temperature already observed by cutoff.
- `predicted_remaining_heat_c`: how much higher final Tmax may still go.
- `predicted_tmax_c`: operational final Tmax forecast from the selected
  validation method. The future-curve model still does not override this field.
- `prediction_method`: selected operational method from validation. It can be
  `direct`, `two_stage`, `m1`, or `openmeteo`.
- `openmeteo_forecast_tmax_c`: raw daily Tmax forecast from the configured
  Open-Meteo CSV, when available for the forecast date.
- `openmeteo_predicted_tmax_c`: M3 output after combining Open-Meteo features
  with the observed pre-cutoff ASOS/METAR features. This equals
  `predicted_tmax_c` when `prediction_method` is `openmeteo`.
- `openmeteo_features_available`: whether the prediction row had an
  Open-Meteo forecast for that local date. If the live file is missing, the
  model can still run with imputed values but this field is `false`.
- `thermal_phase`: estimated current phase of the daily temperature curve.
- `future_curve`: predicted temperatures every 30 minutes through 180 minutes
  after cutoff.
- `curve_predicted_tmax_c`: max of observed max and the future-curve forecast;
  this is reported for comparison, not used as the operational forecast yet.
- `late_warming_risk`: operational bucket from probability of remaining heat
  `>= 2 C`.
- `warming_strength`: class derived from the binary remaining-heat
  classifiers: `no_or_weak_warming`, `mild_warming`, `strong_warming`, or
  `extreme_warming`.
- `prob_no_or_weak_warming`, `prob_mild_warming`,
  `prob_strong_warming`, `prob_extreme_warming`: approximate class
  probabilities derived from cumulative probabilities for remaining heat
  `>= 0.5 C`, `>= 2 C`, and `>= 4 C`.
- `late_warming_warning`: warning layer from the binary classifiers. The
  starting operational thresholds are `P(RH>=4C) >= 15%`,
  `P(RH>=3C) >= 30%`, and `P(RH>=2C) >= 50%`.
- `tail_risk_upper_c`: widened upper bound for late-warming tail risk. This
  does not change `predicted_tmax_c`; it only exposes a practical upper value
  when the classifiers see meaningful risk of strong late warming.
- `regime_break_score`, `regime_break_type`, `last3_weight_hint`: how much
  today's pre-cutoff pattern agrees with the last-3-days prior.
- `prediction_interval_80_low_c` and `prediction_interval_80_high_c`: rough
  uncertainty interval from historical residuals.
- `prediction_interval_80_low_raw_c`: statistical lower bound before clipping.
- `prediction_interval_80_low_practical_c`: lower bound clipped to the observed
  max, because final Tmax cannot be lower than a value already observed.
- `prob_tmax_already_reached`: probability that the current observed max is
  already final Tmax.
- `prob_remaining_heat_ge_*`: probability of at least that much additional
  warming after cutoff.
- `prob_tmax_ge_*`: probability final Tmax reaches or exceeds that threshold.
  If the observed max has already crossed the threshold, this is forced to `1`.
- `not_highest_bet`: present only when `--bet-temp-c` is passed. It estimates
  the probability that final `Tmax` is hotter than the bet temperature. If the
  bet temperature is outside trained final-Tmax thresholds, the estimate falls
  back to remaining-heat classifiers based on `bet_temp_c -
  observed_max_to_cutoff_c`.
- `raw_threshold_probabilities`: raw classifier output before monotonic forcing.
- `monotonic_threshold_probabilities`: probability after enforcing that higher
  thresholds cannot be more likely than lower thresholds.
- `recommend_update_next_cutoff`: whether the next cutoff is worth updating,
  based on historical improvement and current interval width.
- `plot_path`: present only when `--plot` is used; points to the PNG chart.
- `--explain`: CLI option that prints a short interpretation for
  non-technical readers after the JSON output. This does not change the JSON
  fields.

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

The predict command can build a single cutoff feature row directly from the
configured data source, so rebuilding the full heat-risk dataset is not
required for every live prediction. Rebuild/retrain when adding enough
completed days to improve the model.

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
