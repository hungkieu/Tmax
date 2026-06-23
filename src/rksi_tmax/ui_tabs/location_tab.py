from __future__ import annotations

import streamlit as st

from rksi_tmax.services.config_service import (
    ConfigOption,
    update_location_openmeteo_config,
)
from rksi_tmax.config import ProjectConfig
from rksi_tmax.ui_components import render_json


def render(
    config_options: list[ConfigOption],
    selected_config: ProjectConfig,
    selected_config_path: str | None,
) -> None:
    st.subheader("Locations")
    if config_options:
        render_json(
            "Existing location configs",
            [
                {
                    "station": option.station,
                    "config_path": str(option.path),
                }
                for option in config_options
            ],
        )

    st.caption("Enable or update Open-Meteo M3 for the selected location.")
    _render_openmeteo_config_form(selected_config, selected_config_path)


def _render_openmeteo_config_form(config: ProjectConfig, config_path: str | None) -> None:
    if not config_path:
        st.warning("Select a location config before updating Open-Meteo.")
        return

    station_lower = config.station.lower()
    default_history_json = f"data/{station_lower}/openmeteo-{station_lower}-history.json"
    default_live_json_pattern = f"data/{station_lower}/openmeteo-{station_lower}-{{date}}.json"
    openmeteo_latitude = getattr(config, "openmeteo_latitude", None)
    openmeteo_longitude = getattr(config, "openmeteo_longitude", None)
    openmeteo_timezone = getattr(config, "openmeteo_timezone", "GMT")
    openmeteo_history_json = getattr(config, "openmeteo_history_json", None)
    openmeteo_live_json_pattern = getattr(config, "openmeteo_live_json_pattern", None)
    openmeteo_training_start_date = getattr(config, "openmeteo_training_start_date", None)
    openmeteo_training_end_date = getattr(config, "openmeteo_training_end_date", None)
    has_openmeteo = openmeteo_latitude is not None and openmeteo_longitude is not None
    status = "configured" if has_openmeteo else "not configured"
    st.caption(f"Selected: {config.station} ({status})")
    with st.form(f"openmeteo-config-{config_path}"):
        latitude = st.number_input(
            "Open-Meteo latitude",
            value=float(openmeteo_latitude) if openmeteo_latitude is not None else 0.0,
            min_value=-90.0,
            max_value=90.0,
            format="%.6f",
            key=f"openmeteo-latitude-{config_path}",
        )
        longitude = st.number_input(
            "Open-Meteo longitude",
            value=float(openmeteo_longitude) if openmeteo_longitude is not None else 0.0,
            min_value=-180.0,
            max_value=180.0,
            format="%.6f",
            key=f"openmeteo-longitude-{config_path}",
        )
        timezone = st.text_input(
            "Open-Meteo timezone",
            value=openmeteo_timezone or "GMT",
            key=f"openmeteo-timezone-{config_path}",
        )
        history_json = st.text_input(
            "Open-Meteo training cache JSON",
            value=str(openmeteo_history_json) if openmeteo_history_json else default_history_json,
            key=f"openmeteo-history-json-{config_path}",
        )
        live_json_pattern = st.text_input(
            "Open-Meteo daily cache JSON pattern",
            value=openmeteo_live_json_pattern or default_live_json_pattern,
            key=f"openmeteo-live-json-pattern-{config_path}",
        )
        training_start_date = st.text_input(
            "Open-Meteo training start date",
            value=openmeteo_training_start_date or "2023-01-01",
            key=f"openmeteo-training-start-{config_path}",
        )
        training_end_date = st.text_input(
            "Open-Meteo training end date",
            value=openmeteo_training_end_date or "",
            key=f"openmeteo-training-end-{config_path}",
        )
        submitted = st.form_submit_button("Save Open-Meteo M3 config", type="primary")

    if submitted:
        try:
            result = update_location_openmeteo_config(
                config_path,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                history_json=history_json,
                live_json_pattern=live_json_pattern,
                training_start_date=training_start_date,
                training_end_date=training_end_date or None,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Updated Open-Meteo M3 config for {result['station']}.")
            render_json("Open-Meteo config result", result)
            st.rerun()
