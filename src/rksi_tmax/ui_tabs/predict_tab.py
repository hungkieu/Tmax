from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services import prediction_service
from rksi_tmax.ui_components import render_json, render_status_metrics


def render(config: ProjectConfig) -> None:
    st.subheader("Predict")
    method_options = _available_prediction_methods(config)
    method_labels = [option["label"] for option in method_options]

    with st.form("predict-form"):
        selected_method_label = st.selectbox("Prediction model", method_labels)
        selected_method = method_options[method_labels.index(selected_method_label)]["value"]
        st.caption(method_options[method_labels.index(selected_method_label)]["description"])
        mode = st.segmented_control(
            "Cut-off mode",
            ["Latest in database", "Config default cutoff", "Custom"],
            default="Latest in database",
        )
        selected_date = st.date_input("Local date", value=date.today())
        cutoff_options = sorted(set(config.heat_risk_cutoffs + (config.cutoff_local,)))
        cutoff_local = st.selectbox(
            "Cut-off local",
            options=cutoff_options,
            index=cutoff_options.index(config.cutoff_local),
        )
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Running prediction..."):
            resolved_date, resolved_cutoff = prediction_service.resolve_prediction_time(
                config,
                str(mode),
                selected_date.isoformat(),
                cutoff_local,
            )
            result = prediction_service.run_prediction(
                config,
                resolved_date,
                resolved_cutoff,
                bet_temp_c=None,
                make_plot=False,
                prediction_method=selected_method,
            )
        render_status_metrics(_prediction_status_metrics(result))
        if result.get("prediction_method") == "m4":
            from rksi_tmax.heat_risk import format_m4_brief_explanation

            st.text(format_m4_brief_explanation(result))
            with st.expander("Giải thích đầy đủ (mọi layer)"):
                st.text(result["explanation"])
        else:
            st.text(result["explanation"])
        render_json("Selected Model Report", _selected_model_report(result))
        render_json("Full Prediction JSON (debug)", result)


def _available_prediction_methods(config: ProjectConfig) -> list[dict[str, str]]:
    try:
        model_path = config.heat_risk_model_path
        cached = _prediction_methods_for_path(
            str(model_path),
            model_path.stat().st_mtime_ns if model_path.exists() else 0,
        )
        return cached
    except Exception as exc:
        st.warning(f"Could not inspect model methods: {exc}")
        return [{"label": "Auto", "value": "auto", "description": str(exc)}]


@st.cache_data(show_spinner=False)
def _prediction_methods_for_path(
    model_path: str,
    model_mtime_ns: int,
) -> list[dict[str, str]]:
    del model_mtime_ns
    from pathlib import Path

    config = ProjectConfig(heat_risk_model_path=Path(model_path))
    return prediction_service.available_prediction_methods(config)


def _prediction_status_metrics(result: dict[str, Any]) -> dict[str, Any]:
    items: dict[str, Any] = {
        "Predicted Tmax C": result.get("predicted_tmax_c"),
        "Observed max C": result.get("observed_max_to_cutoff_c"),
        "Remaining heat C": result.get("predicted_remaining_heat_c"),
        "Method": result.get("prediction_method"),
    }
    method = result.get("prediction_method")
    if method == "m4":
        items["M4 top expert"] = result.get("m4_top_expert")
    return items


def _selected_model_report(result: dict[str, Any]) -> dict[str, Any]:
    base = {
        "station": result.get("station"),
        "local_date": result.get("local_date"),
        "cutoff_local": result.get("cutoff_local"),
        "last_observation_local": result.get("last_observation_local"),
        "observed_max_to_cutoff_c": result.get("observed_max_to_cutoff_c"),
        "predicted_remaining_heat_c": result.get("predicted_remaining_heat_c"),
        "predicted_tmax_c": result.get("predicted_tmax_c"),
        "prediction_method": result.get("prediction_method"),
    }
    method = result.get("prediction_method")
    if method == "m4":
        return {
            **base,
            "m4_predicted_remaining_heat_c": result.get("m4_predicted_remaining_heat_c"),
            "m4_predicted_tmax_c": result.get("m4_predicted_tmax_c"),
            "m4_top_expert": result.get("m4_top_expert"),
            "m4_expert_weights": result.get("m4_expert_weights"),
            "openmeteo_forecast_tmax_c": result.get("openmeteo_forecast_tmax_c"),
            "openmeteo_expected_remaining_heat_c": result.get("openmeteo_expected_remaining_heat_c"),
        }
    if method == "openmeteo":
        return {
            **base,
            "openmeteo_predicted_remaining_heat_c": result.get("openmeteo_predicted_remaining_heat_c"),
            "openmeteo_predicted_tmax_c": result.get("openmeteo_predicted_tmax_c"),
            "openmeteo_forecast_tmax_c": result.get("openmeteo_forecast_tmax_c"),
            "openmeteo_expected_remaining_heat_c": result.get("openmeteo_expected_remaining_heat_c"),
        }
    return base
