from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services import db_service, next_metar_service, training_service
from rksi_tmax.ui_components import render_artifact_links, render_json, render_status_metrics


def render(config: ProjectConfig) -> None:
    st.subheader("Next METAR")
    render_artifact_links(next_metar_service.artifact_status(config))
    _render_latest_observation(config)
    _render_metrics_artifact(config)

    st.divider()
    with st.form("next-metar-live-form"):
        st.caption("Use the METAR tab's live update first when fresh METAR/Open-Meteo data is needed.")
        as_of_local = st.text_input(
            "As-of local time",
            value="",
            placeholder="latest, or 2026-06-26 14:30",
        )
        submitted = st.form_submit_button(
            "Predict from latest data",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            with st.spinner("Updating live data and predicting next METAR temperature..."):
                result = next_metar_service.run_live_nowcast(
                    config,
                    update_metar=False,
                    update_openmeteo=False,
                    metar_hours=1,
                    metar_file="data/shared/metar-ui.txt",
                    as_of_local=as_of_local.strip() or None,
                )
        except Exception as exc:
            _render_workflow_error("Next-METAR prediction failed", exc)
        else:
            for warning in result.get("warnings", []):
                st.warning(str(warning))
            prediction = result["prediction"]
            _render_prediction(prediction)
            render_json("Workflow steps", result.get("steps", {}))
            render_json("Full next-METAR prediction JSON", prediction)

    _render_training_controls(config)


def _render_latest_observation(config: ProjectConfig) -> None:
    status = db_service.database_status(config)
    latest = db_service.latest_observation(config) if status["exists"] else None
    render_status_metrics(
        {
            "DB rows": status["row_count"],
            "Latest valid UTC": status["latest_valid"],
            "Model exists": next_metar_service.artifact_status(config)["model"]["exists"],
            "Open-Meteo": "configured" if _openmeteo_configured(config) else "off",
        }
    )
    if latest:
        with st.expander("Latest database observation"):
            st.json(latest)


def _render_metrics_artifact(config: ProjectConfig) -> None:
    metrics = next_metar_service.read_metrics(config)
    if not metrics:
        return
    render_status_metrics(
        {
            "MAE C": metrics.get("mae_c"),
            "Baseline MAE C": metrics.get("baseline_persistence_mae_c"),
            "Exact": _format_percent(metrics.get("exact_accuracy")),
            "Within 1C": _format_percent(metrics.get("within_1c_accuracy")),
        }
    )
    with st.expander("Latest next-METAR metrics"):
        st.json(metrics)


def _render_prediction(prediction: dict[str, Any]) -> None:
    st.success(
        f"{prediction['station']} next METAR: {prediction['predicted_temp_c']} C "
        f"near {prediction['target_time_local']}"
    )
    render_status_metrics(
        {
            "Current C": prediction.get("current_temp_c_int"),
            "Predicted C": prediction.get("predicted_temp_c"),
            "Expected C": prediction.get("expected_temp_c"),
            "Delta C": prediction.get("predicted_delta_c"),
        }
    )
    render_status_metrics(
        {
            "P(-1C exact)": _format_percent(prediction.get("prob_next_temp_eq_current_minus_1c")),
            "P(same exact)": _format_percent(prediction.get("prob_next_temp_eq_current_c")),
            "P(+1C exact)": _format_percent(prediction.get("prob_next_temp_eq_current_plus_1c")),
            "P(up >=1C)": _format_percent(prediction.get("prob_next_temp_ge_current_plus_1c")),
        }
    )
    probability_rows = [
        {"Temp C": int(temp), "Probability": float(probability), "Percent": _format_percent(probability)}
        for temp, probability in prediction.get("probabilities_by_temp_c", {}).items()
    ]
    probability_rows = sorted(probability_rows, key=lambda row: row["Probability"], reverse=True)
    if probability_rows:
        st.caption("Temperature probability distribution")
        st.dataframe(probability_rows, hide_index=True, use_container_width=True)

    context_columns = st.columns(3)
    with context_columns[0]:
        st.caption("Temperature trend")
        st.dataframe(
            _context_table(prediction.get("temperature_context", {})),
            hide_index=True,
            use_container_width=True,
        )
    with context_columns[1]:
        st.caption("METAR weather")
        st.dataframe(
            _context_table(prediction.get("weather_context", {})),
            hide_index=True,
            use_container_width=True,
        )
    with context_columns[2]:
        st.caption("Open-Meteo")
        st.dataframe(
            _context_table(prediction.get("openmeteo_context", {})),
            hide_index=True,
            use_container_width=True,
        )


def _render_training_controls(config: ProjectConfig) -> None:
    with st.expander("Train / Validate next-METAR model", expanded=not config.next_metar_temp_model_path.exists()):
        st.caption("Use these after changing data, station config, or model code.")
        columns = st.columns(4)
        key_prefix = f"next-metar-{config.station.lower()}"
        if columns[0].button(
            "Build dataset",
            key=f"{key_prefix}-build-dataset",
            use_container_width=True,
        ):
            try:
                with st.spinner("Building next-METAR dataset..."):
                    result = next_metar_service.build_dataset(config)
            except Exception as exc:
                _render_workflow_error("Build next-METAR dataset failed", exc)
            else:
                st.success(f"Built {result['rows']} rows.")
                render_json("Dataset result", result)
        if columns[1].button(
            "Train",
            key=f"{key_prefix}-train",
            use_container_width=True,
        ):
            try:
                with st.spinner("Training next-METAR model..."):
                    result = next_metar_service.train_model(config)
            except Exception as exc:
                _render_workflow_error("Train next-METAR model failed", exc)
            else:
                st.success("Training finished.")
                _render_training_metrics(result)
                render_json("Training metrics", result)
        if columns[2].button(
            "Validate",
            key=f"{key_prefix}-validate",
            use_container_width=True,
        ):
            try:
                with st.spinner("Validating next-METAR model..."):
                    result = next_metar_service.validate_model(config)
            except Exception as exc:
                _render_workflow_error("Validate next-METAR model failed", exc)
            else:
                st.success("Validation finished.")
                _render_training_metrics(result.get("summary", result))
                render_json("Validation report", result)
        if columns[3].button(
            "Prepare Open-Meteo",
            key=f"{key_prefix}-prepare-openmeteo",
            use_container_width=True,
        ):
            try:
                with st.spinner("Preparing Open-Meteo training cache..."):
                    result = training_service.prepare_openmeteo_training_data(config)
            except Exception as exc:
                _render_workflow_error("Prepare Open-Meteo failed", exc)
            else:
                st.success("Open-Meteo training cache ready.")
                render_json("Open-Meteo result", result)


def _render_training_metrics(metrics: dict[str, Any]) -> None:
    render_status_metrics(
        {
            "MAE C": metrics.get("mae_c"),
            "Baseline MAE C": metrics.get("baseline_persistence_mae_c"),
            "Exact": _format_percent(metrics.get("exact_accuracy")),
            "Within 1C": _format_percent(metrics.get("within_1c_accuracy")),
        }
    )


def _context_table(context: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Signal": key, "Value": _display_value(value)}
            for key, value in context.items()
        ]
    )


def _openmeteo_configured(config: ProjectConfig) -> bool:
    return (
        config.openmeteo_live_json_pattern is not None
        and config.openmeteo_latitude is not None
        and config.openmeteo_longitude is not None
    )


def _format_percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _display_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _render_workflow_error(title: str, exc: Exception) -> None:
    st.error(f"{title}: {exc}")
    message = str(exc)
    if "not the v3" in message or "No such file" in message or "No file" in message:
        st.info("Build and train the next-METAR model for this location first.")
    elif "Open-Meteo API is not configured" in message:
        st.info("Open-Meteo is optional; configure coordinates/cache paths or turn off Open-Meteo update.")
