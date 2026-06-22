from __future__ import annotations

from datetime import date

import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services import prediction_service
from rksi_tmax.ui_components import render_json, render_plot, render_status_metrics


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
        bet_enabled = st.checkbox("Estimate bet probability")
        bet_temp_c = st.number_input(
            "Bet temperature C",
            value=30.0,
            step=0.1,
            help="Used only when Estimate bet probability is checked.",
        )
        make_plot = st.checkbox("Generate plot", value=True)
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
                bet_temp_c=float(bet_temp_c) if bet_enabled else None,
                make_plot=make_plot,
                prediction_method=selected_method,
            )
        render_status_metrics(
            {
                "Predicted Tmax C": result.get("predicted_tmax_c"),
                "Observed max C": result.get("observed_max_to_cutoff_c"),
                "Remaining heat C": result.get("predicted_remaining_heat_c"),
                "Method": result.get("prediction_method"),
                "Selected": result.get("selected_prediction_method"),
            }
        )
        st.text(result["explanation"])
        render_plot(result.get("plot_path"))
        render_json("Prediction JSON", result)


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
