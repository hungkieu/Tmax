from __future__ import annotations

import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services import artifact_service, training_service
from rksi_tmax.ui_components import render_artifact_links, render_json, render_status_metrics


def render(config: ProjectConfig) -> None:
    st.subheader("Train and Validate")
    render_artifact_links(artifact_service.artifact_status(config))

    columns = st.columns(3)
    if columns[0].button("Build dataset", use_container_width=True):
        try:
            with st.spinner("Building heat-risk dataset..."):
                result = training_service.build_dataset(config)
        except Exception as exc:
            _render_workflow_error("Build dataset failed", exc)
        else:
            st.success(f"Built {result['rows']} rows.")
            render_json("Dataset result", result)

    if columns[1].button("Train", use_container_width=True):
        try:
            with st.spinner("Training model..."):
                result = training_service.train_model(config)
        except Exception as exc:
            _render_workflow_error("Train failed", exc)
        else:
            st.success("Training finished.")
            _render_metric_summary(result)
            render_json("Training metrics", result)

    if columns[2].button("Validate", use_container_width=True):
        try:
            with st.spinner("Validating model..."):
                result = training_service.validate_model(config)
        except Exception as exc:
            _render_workflow_error("Validate failed", exc)
        else:
            st.success("Validation finished.")
            _render_metric_summary(result.get("summary", result))
            render_json("Validation report", result)

    metrics = artifact_service.read_metrics(config)
    if metrics:
        st.divider()
        st.caption("Latest metrics artifact")
        _render_metric_summary(metrics)
        render_json("Metrics artifact", metrics)


def _render_metric_summary(metrics: dict[str, object]) -> None:
    keys = [
        "selected_prediction_method",
        "prediction_method",
        "tmax_mae_c",
        "m1_phase_feature_tmax_mae_c",
        "openmeteo_tmax_mae_c",
        "m0_heat_risk_tmax_mae_c",
    ]
    summary = {key: metrics.get(key) for key in keys if key in metrics}
    if summary:
        render_status_metrics(summary)


def _render_workflow_error(title: str, exc: Exception) -> None:
    st.error(f"{title}: {exc}")
    message = str(exc)
    if "No " in message and "observations found before cutoff" in message:
        st.info(
            "Import observations for this station first in the METAR tab, then sync DuckDB. "
            "For training, you need enough completed historical days before the configured cutoffs."
        )
    elif "Need at least" in message:
        st.info("Add more completed historical observations before training this location.")
