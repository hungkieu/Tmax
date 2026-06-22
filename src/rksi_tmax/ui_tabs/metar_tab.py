from __future__ import annotations

from datetime import date

import streamlit as st

from rksi_tmax.config import ProjectConfig
from rksi_tmax.services.config_service import ConfigOption, load_selected_config
from rksi_tmax.services import db_service, metar_service
from rksi_tmax.ui_components import render_json, render_status_metrics


def render(config: ProjectConfig, config_options: list[ConfigOption]) -> None:
    st.subheader("METAR")
    option_by_label = {option.label: option for option in config_options}
    default_labels = [
        option.label for option in config_options if option.station == config.station
    ] or [config_options[0].label]
    selected_labels = st.multiselect(
        "Locations",
        options=list(option_by_label),
        default=default_labels,
    )
    selected_configs = [
        load_selected_config(option_by_label[label].path)
        for label in selected_labels
    ]
    selected_stations = [selected_config.station for selected_config in selected_configs]
    hours = st.number_input("Lookback hours", min_value=1, max_value=168, value=24, step=1)
    metar_file = st.text_input("METAR file", value="metar-ui.txt")
    reference_date = st.date_input("Reference date", value=date.today())

    selected_actions = st.columns(3)
    if selected_actions[0].button("Fetch METAR", use_container_width=True):
        if not selected_stations:
            st.warning("Select at least one location.")
            return
        with st.spinner("Fetching METAR..."):
            result = metar_service.fetch_metar_for_stations(selected_stations, int(hours), metar_file)
        st.success(f"Fetched {result['lines']} METAR lines.")
        render_json("Fetch result", result)

    if selected_actions[1].button("Import + DB", use_container_width=True):
        if not selected_configs:
            st.warning("Select at least one location.")
            return
        with st.spinner("Importing station-scoped METAR for selected locations..."):
            result = metar_service.import_many_station_metars(
                selected_configs,
                metar_file,
                reference_date,
            )
        st.success(f"Inserted {result['inserted']} rows across {len(selected_configs)} locations.")
        render_json("Import result", result)

    if selected_actions[2].button("Sync DuckDB", use_container_width=True):
        if not selected_configs:
            st.warning("Select at least one location.")
            return
        with st.spinner("Syncing DuckDB for selected locations..."):
            result = metar_service.sync_many_databases(selected_configs)
        st.success(f"Synced {len(selected_configs)} location configs.")
        render_json("Sync result", result)

    st.divider()
    st.caption("Verification")
    for selected_config in selected_configs or [config]:
        with st.expander(f"{selected_config.station} database status", expanded=False):
            status = db_service.database_status(selected_config)
            render_status_metrics(
                {
                    "DB exists": status["exists"],
                    "Station rows": status["row_count"],
                    "Latest valid": status["latest_valid"],
                }
            )
            latest = db_service.latest_observation(selected_config) if status["exists"] else None
            render_json("Database status", status)
            if latest:
                render_json("Latest observation", latest)
