from __future__ import annotations

from datetime import date

import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services import artifact_service, training_service
from rksi_tmax.ui_components import render_artifact_links, render_json, render_status_metrics


def render(config: ProjectConfig) -> None:
    st.subheader("Train and Validate")
    render_artifact_links(artifact_service.artifact_status(config))
    _render_openmeteo_controls(config)

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
            _render_integer_win_rate_table(result)
            render_json("Validation report", result)

    metrics = artifact_service.read_metrics(config)
    if metrics:
        st.divider()
        st.caption("Latest metrics artifact")
        _render_metric_summary(metrics)
        render_json("Metrics artifact", metrics)


def _render_metric_summary(metrics: dict[str, object]) -> None:
    integer_tmax_win_rates = metrics.get("integer_tmax_win_rates")
    keys = [
        "selected_prediction_method",
        "prediction_method",
        "tmax_mae_c",
        "m1_phase_feature_tmax_mae_c",
        "openmeteo_tmax_mae_c",
        "openmeteo_daily_tmax_mae_c",
        "openmeteo_hourly_tmax_mae_c",
        "selected_openmeteo_variant",
        "m0_heat_risk_tmax_mae_c",
    ]
    summary = {key: metrics.get(key) for key in keys if key in metrics}
    if summary:
        render_status_metrics(summary)
    if isinstance(integer_tmax_win_rates, dict):
        render_status_metrics(
            {
                "Tmax win": integer_tmax_win_rates.get("tmax_win_rate"),
                "Tmax +1 win": integer_tmax_win_rates.get("tmax_plus_1_win_rate"),
                "Tmax -1 win": integer_tmax_win_rates.get("tmax_minus_1_win_rate"),
                "Tmax +/-1 win": integer_tmax_win_rates.get(
                    "combined_tmax_minus_1_to_plus_1_win_rate"
                ),
            }
        )


def _render_integer_win_rate_table(report: dict[str, object]) -> None:
    metrics_by_cutoff = report.get("metrics_by_cutoff")
    if not isinstance(metrics_by_cutoff, list) or not metrics_by_cutoff:
        return

    rows = []
    for row in metrics_by_cutoff:
        if not isinstance(row, dict) or "tmax_win_rate" not in row:
            continue
        rows.append(
            {
                "Cutoff": row.get("cutoff_local"),
                "N": row.get("n"),
                "Exact": _format_percent(row.get("tmax_win_rate")),
                "Tmax +1": _format_percent(row.get("tmax_plus_1_win_rate")),
                "Tmax -1": _format_percent(row.get("tmax_minus_1_win_rate")),
                "+/-1": _format_percent(row.get("combined_tmax_minus_1_to_plus_1_win_rate")),
                "Exact count": row.get("tmax_win_count"),
                "+1 count": row.get("tmax_plus_1_win_count"),
                "-1 count": row.get("tmax_minus_1_win_count"),
                "+/-1 count": row.get("combined_tmax_minus_1_to_plus_1_win_count"),
            }
        )
    if rows:
        st.caption("Rounded Tmax win rate by cutoff")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _format_percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _render_openmeteo_controls(config: ProjectConfig) -> None:
    with st.expander("Open-Meteo M3", expanded=False):
        status = training_service.openmeteo_status(config)
        render_status_metrics(
            {
                "Configured": status["configured"],
                "Latitude": status["latitude"],
                "Longitude": status["longitude"],
                "History cache": "yes" if status["history_json_exists"] else "no",
            }
        )
        render_json("Open-Meteo config", status)
        force = st.checkbox("Refresh Open-Meteo cache", value=False)
        daily_date = st.date_input("Forecast date", value=date.today(), key=f"{config.station}-openmeteo-date")
        columns = st.columns(2)
        if columns[0].button("Prepare training data", use_container_width=True):
            try:
                with st.spinner("Fetching Open-Meteo historical forecast..."):
                    result = training_service.prepare_openmeteo_training_data(config, force=force)
            except Exception as exc:
                _render_workflow_error("Open-Meteo training data failed", exc)
            else:
                st.success("Open-Meteo training cache ready.")
                render_json("Open-Meteo training result", result)
        if columns[1].button("Prepare daily forecast", use_container_width=True):
            try:
                with st.spinner("Fetching Open-Meteo daily forecast..."):
                    result = training_service.prepare_openmeteo_daily_data(
                        config,
                        daily_date.isoformat(),
                        force=force,
                    )
            except Exception as exc:
                _render_workflow_error("Open-Meteo daily forecast failed", exc)
            else:
                st.success("Open-Meteo daily cache ready.")
                render_json("Open-Meteo daily result", result)


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
