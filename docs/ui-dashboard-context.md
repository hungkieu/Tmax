# UI Dashboard Context

## Purpose

This document is the short-read context for future LLM or engineer work on the local UI dashboard. Read it before changing UI-related files so the repo does not need to be scanned from scratch.

The UI is a local Streamlit dashboard for common station workflows:

- Fetch METAR for selected locations, import observations, sync/read DuckDB, and verify recent rows.
- Build datasets, train models, and validate results.
- Run heat-risk predictions with a selected cut-off time or the latest observation in the database, then read the explanation and inspect structured output.

Streamlit should handle rendering and user interaction only. Business logic belongs in service modules that call the existing CLI/source-of-truth functions.

## Current Forecasting Context

Existing source-of-truth code should remain the authority for forecasting behavior:

- Config loading: `src/rksi_tmax/config.py`
- METAR fetch/import: `src/rksi_tmax/metar_import.py`
- DuckDB sync/read: `src/rksi_tmax/storage.py`
- Dataset build/train/validate/predict/explain/plot: `src/rksi_tmax/heat_risk.py`
- CLI wiring and current workflow behavior: `src/rksi_tmax/cli.py`

`predicted_tmax_c` is produced by the model selected in `selected_prediction_method`. M3/Open-Meteo trains from the M1 feature set plus daily and hourly Open-Meteo API forecast features when a location has coordinates and JSON cache paths configured. M1 remains available as a selectable METAR/ASOS-only fallback.

## UI File Map

Update this section whenever module boundaries change.

- `src/rksi_tmax/ui_app.py`: thin Streamlit entrypoint; page setup, sidebar config selector, and tab routing.
- `src/rksi_tmax/ui_state.py`: session state helpers and lightweight dataclasses for UI selections/results.
- `src/rksi_tmax/ui_components.py`: reusable Streamlit blocks for metrics, JSON expanders, status panels, config summaries, and artifact links.
- `src/rksi_tmax/ui_tabs/metar_tab.py`: METAR fetch/import/sync/verify tab.
- `src/rksi_tmax/ui_tabs/location_tab.py`: create new location config, optional empty ASOS CSV header, and Open-Meteo API config.
- `src/rksi_tmax/ui_tabs/train_tab.py`: Open-Meteo cache preparation, dataset build, train, validate, and metrics display tab.
- `src/rksi_tmax/ui_tabs/predict_tab.py`: prediction controls, cut-off resolution, bet temperature input, explain text, JSON output, and optional plot.
- `src/rksi_tmax/services/config_service.py`: discover and load station configs.
- `src/rksi_tmax/services/db_service.py`: latest observation lookup, row counts, station coverage, and DuckDB health checks.
- `src/rksi_tmax/services/metar_service.py`: fetch METAR, import station-scoped observations, sync database, and verify import results.
- `src/rksi_tmax/services/training_service.py`: build dataset, train model, validate model, and return concise summaries.
- `src/rksi_tmax/services/prediction_service.py`: resolve date/cut-off/latest observation, call prediction, format explanation, and return plot/output paths.
- `src/rksi_tmax/services/artifact_service.py`: inspect model, metrics, diagnostic, plot, and dataset artifacts.

## Service Ownership

Service modules are the boundary between Streamlit UI code and existing project logic.

- `config_service` owns config discovery, display names, safe config loading, config-only location deletion, and new location config creation.
- `db_service` owns read-only database status and latest-observation queries.
- `metar_service` owns fetch/import/sync/verify workflow. It must avoid importing multi-station METAR text into the wrong station CSV.
- `training_service` owns Open-Meteo cache preparation, dataset build, model train, validation calls, and compact metric summaries.
- `prediction_service` owns prediction input resolution, including latest cut-off from DuckDB, custom cut-off, model-method options, bet temperature, explanation text, JSON result, and plots.
- `artifact_service` owns filesystem inspection for generated artifacts and should not know Streamlit UI layout.

## Read This First Guide

- Modifying UI shell/layout: read `src/rksi_tmax/ui_app.py`, then `src/rksi_tmax/ui_components.py`.
- Modifying one tab: read the target file under `src/rksi_tmax/ui_tabs/`, then the service it calls.
- Modifying METAR workflow: read `src/rksi_tmax/ui_tabs/metar_tab.py`, `src/rksi_tmax/services/metar_service.py`, and `src/rksi_tmax/metar_import.py`.
- Modifying prediction controls or display: read `src/rksi_tmax/ui_tabs/predict_tab.py`, `src/rksi_tmax/services/prediction_service.py`, then `src/rksi_tmax/heat_risk.py`.
- Modifying train/validate display: read `src/rksi_tmax/ui_tabs/train_tab.py`, `src/rksi_tmax/services/training_service.py`, and `src/rksi_tmax/services/artifact_service.py`.
- Modifying config/location discovery: read `src/rksi_tmax/services/config_service.py`, then `src/rksi_tmax/config.py`.
- Modifying add-location flow: read `src/rksi_tmax/ui_tabs/location_tab.py` and `src/rksi_tmax/services/config_service.py`.
- Modifying location deletion: read `src/rksi_tmax/ui_app.py` and `src/rksi_tmax/services/config_service.py`.
- Modifying database status or latest cut-off behavior: read `src/rksi_tmax/services/db_service.py`, then `src/rksi_tmax/storage.py`.

## Invariants

- Service modules must not import Streamlit.
- Existing CLI/functions remain the source of truth for model, training, import, and prediction behavior.
- UI code should return/display structured results and place raw JSON in expanders.
- Long tasks should use Streamlit spinners in v1; no async queue or background worker is planned.
- Do not duplicate forecast, feature, training, validation, or METAR parsing logic inside UI modules.
- Keep UI modules small enough that an LLM can read only the entrypoint, one tab, and one service for most edits.
- When Open-Meteo availability changes for a location, update config docs and this context file.
- Deleting a location from the UI deletes only its YAML config; raw CSV, DuckDB rows, and artifacts remain unless an explicit cleanup tool is added.
- Adding a location from the UI creates a YAML config and, by default, an empty ASOS CSV header. It can also save Open-Meteo coordinates/cache paths. It does not train a model; after adding historical CSV rows, sync DuckDB before training when `prefer_duckdb` is enabled.
- M3/Open-Meteo should appear as a prediction option only when the model bundle contains an Open-Meteo regressor and Open-Meteo feature columns.

## Operation

Run the dashboard from the repo root:

```powershell
uv run rksi-ui
```

Keep this document concise and update it whenever UI module boundaries or ownership rules change.
