# Next-METAR Integer Temperature Model

> Hướng dẫn vận hành / sử dụng (tiếng Việt): [next-metar-temp-usage.md](next-metar-temp-usage.md).

## Purpose

This subsystem predicts the integer temperature of the **next 1-4 regular
METARs** (multi-horizon) for Korea stations `RKSI` and `RKPK`.

Because the observed source is highly trusted, horizon 1 (the immediate next
METAR) is essentially the rounded current temperature; the value of the model is
in the further horizons (2-4) where the temperature actually changes. Each
horizon is predicted and verified independently.

It is intentionally separate from the existing Tmax/heat-risk model. It has its
own dataset, model artifact, metrics file, prediction log, verification flow,
and promotion checks.

The operational question is:

```text
Given the latest live temperature for one station, what integer Celsius
temperature will the next regular METAR report?
```

Example input:

```json
{
  "station": "RKSI",
  "observed_at": "2026-06-23T09:30:00+09:00",
  "temp_c": 27.9
}
```

Example output:

```json
{
  "station": "RKSI",
  "observed_at": "2026-06-23T09:30:00+09:00",
  "next_metar_at": "2026-06-23T10:00:00+09:00",
  "input_temp_c": 27.9,
  "tmax_signal_c": null,
  "predicted_temp_c": 28,
  "model": "next_metar_integer_tmax",
  "model_version": 1,
  "status": "ok",
  "fallback": false,
  "fallback_reason": null,
  "verification_status": "pending"
}
```

`predicted_temp_c` is always an integer because operational METAR temperature
values are reported as whole Celsius degrees.

## Design

The implementation lives in `src/rksi_tmax/next_metar_temp.py`.

The model is trained from historical station observations. Each row represents
one current observation and the target is the integer Celsius temperature in the
next regular METAR.

Regular METAR schedule:

| Station | Next regular METAR slot |
| --- | --- |
| `RKSI` | next `:00` or `:30` |
| `RKPK` | next `:00` |

The feature set is deliberately compact:

- station flags;
- current live temperature;
- rounded current temperature and fractional position;
- minutes until the next METAR;
- local time and seasonal cyclic features;
- optional current-day Tmax signal only.

It does not use:

- future curve model outputs;
- curve metrics;
- full M3 feature columns;
- cloud, wind, weather, or other heat-risk subfeatures;
- existing heat-risk model internals.

If the active model cannot run, prediction falls back to half-up rounding of
the latest live `temp_c` and records `status: "fallback"` in the prediction log.

## Artifacts

Default artifact paths:

| Artifact | Path |
| --- | --- |
| Dataset | `artifacts/next_metar_temp/next_metar_temp_dataset.parquet` |
| Active model | `artifacts/next_metar_temp/next_metar_temp_model.joblib` |
| Active metrics | `artifacts/next_metar_temp/next_metar_temp_metrics.json` |
| Candidate model | `artifacts/next_metar_temp/next_metar_temp_candidate.joblib` |
| Candidate metrics | `artifacts/next_metar_temp/next_metar_temp_candidate_metrics.json` |
| Prediction log | `artifacts/next_metar_temp/next_metar_temp_predictions.jsonl` |

Current built artifact summary (model_version 2, multi-horizon):

```json
{
  "model_version": 2,
  "n_rows": 1136793,
  "stations": ["RKPK", "RKSI"],
  "validation_mae_c": 0.750,
  "validation_exact_accuracy": 0.432,
  "validation_within_1c_accuracy": 0.866
}
```

Validation by horizon (how accuracy degrades the further ahead you forecast):

| Horizon | MAE C | Exact | Within 1C |
| --- | ---: | ---: | ---: |
| 1 (next) | `0.512` | `54.5%` | `95.1%` |
| 2 | `0.684` | `45.2%` | `89.3%` |
| 3 | `0.839` | `38.7%` | `83.4%` |
| 4 | `0.967` | `34.4%` | `78.5%` |

Validation by station (averaged over horizons 1-4):

| Station | MAE C | Exact | Within 1C |
| --- | ---: | ---: | ---: |
| `RKPK` | `0.977` | `34.4%` | `78.0%` |
| `RKSI` | `0.625` | `48.1%` | `91.3%` |

> The aggregate MAE (0.75) looks higher than the previous single-horizon model
> (0.48) only because it now includes the harder horizons 2-4. Horizon 1 alone
> (`0.51` MAE) is comparable. Promotion across a `model_version` bump is therefore
> not gated on the previous version's metrics.

## Live Data Source (MongoDB)

Live observed temperatures are read **directly from MongoDB** instead of an HTTP
API. The only configuration is a single environment variable:

```text
MONGODB_URI=mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/<dbName>?retryWrites=true&w=majority
```

The database name is embedded in the URI. For local development put it in a
`.env` file at the repository root (gitignored; see `.env.example`); it is loaded
automatically by `rksi_tmax.mongo_source`. For scheduled/CI runs set it as a real
environment variable / secret. Since this project only reads, use a **read-only**
Atlas user.

Collections (KMA observed airport temperatures):

| Collection | Use |
| --- | --- |
| `airporttemperaturecurrents` | latest observation per ICAO (predict input) |
| `airporttemperaturehistories` | append-only history (verification actuals) |

Access helpers live in `src/rksi_tmax/mongo_source.py`:
`get_current_temperature(icao)` and `get_temperature_history(icao, since=...)`.
ICAO is always upper-cased before querying.

## Commands

Build the combined Korea dataset:

```powershell
uv run rksi-build-next-metar-dataset
```

Build one station only:

```powershell
uv run rksi-build-next-metar-dataset --station RKSI
uv run rksi-build-next-metar-dataset --station RKPK
```

Train and promote if validation is not degraded:

```powershell
uv run rksi-train-next-metar-temp
```

Train without promoting the active model:

```powershell
uv run rksi-train-next-metar-temp --no-promote
```

Predict one station using the latest live temperature from MongoDB (omit
`--temp-c` / `--observed-at` and they are read from `airporttemperaturecurrents`):

```powershell
uv run rksi-predict-next-metar-temp --station RKSI
```

Predict with explicit values (still supported):

```powershell
uv run rksi-predict-next-metar-temp --station RKSI --observed-at 2026-06-23T09:30:00+09:00 --temp-c 27.9
```

Optionally pass an external/current-day Tmax signal:

```powershell
uv run rksi-predict-next-metar-temp --station RKSI --observed-at 2026-06-23T09:30:00+09:00 --temp-c 27.9 --tmax-signal-c 30.0
```

Verify pending predictions against observed temperatures from MongoDB
(`airporttemperaturehistories`):

```powershell
uv run rksi-verify-next-metar-temp --from-db --hours 48
```

The MongoDB history is a continuous observation stream, not aligned to METAR
slots, so each pending prediction is matched to the observation **nearest** its
`next_metar_at` within `--tolerance-seconds` (default 300s).

Verify using existing METAR import/parser data instead:

```powershell
uv run rksi-verify-next-metar-temp
```

Fetch recent METAR first, then verify:

```powershell
uv run rksi-verify-next-metar-temp --fetch --hours 4
```

## Learning And Promotion

The subsystem is designed for controlled self-learning:

1. Every prediction is appended to the JSONL prediction log.
2. The verifier fills in official actual METAR temperature when the next METAR
   becomes available.
3. Verified predictions create a monitoring trail of recent model quality.
4. New datasets can be rebuilt from historical and newly imported observations.
5. Retraining writes a candidate model unless promotion checks pass.

Promotion rejects a candidate when:

- validation MAE is worse than the current model by more than `0.05 C`; or
- exact-hit accuracy drops by more than `0.02`.

This keeps the model from automatically replacing itself when recent data or
newly imported observations make validation worse.

## Verification And Health

Verification source of truth is the existing METAR parser/import path. The
verifier joins pending predictions to the official next regular METAR time and
records:

- `actual_temp_c`;
- `verification_error_c`;
- `exact_hit`;
- `within_1c`;
- `verified_at`;
- updated `verification_status`.

Rolling health is calculated from recent predictions and tracks:

- MAE;
- bias;
- exact accuracy;
- within-1C accuracy;
- verification coverage;
- consecutive exact misses.

The health status becomes `unhealthy` if:

- rolling MAE is above `1.0 C`;
- absolute rolling bias is above `0.75 C`;
- verification coverage is below `80%`; or
- consecutive exact misses exceed `5`.

## Notes

`RKPK` now has Open-Meteo cache configuration for the compact Tmax signal:

```yaml
openmeteo_history_csv: data/rkpk/openmeteo-rkpk.csv
openmeteo_live_csv_pattern: data/rkpk/openmeteo-rkpk-{date}.csv
openmeteo_history_json: data/rkpk/openmeteo-rkpk-history.json
openmeteo_live_json_pattern: data/rkpk/openmeteo-rkpk-{date}.json
openmeteo_latitude: 35.179444
openmeteo_longitude: 128.938333
openmeteo_timezone: GMT
```

The new model can still run when the Tmax signal is missing because the training
pipeline imputes missing `tmax_signal_c` values.
