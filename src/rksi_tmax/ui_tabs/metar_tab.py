from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
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
    with st.form("live-data-update-form"):
        columns = st.columns([1, 1, 2, 1])
        hours = columns[0].number_input("Lookback hours", min_value=1, max_value=168, value=24, step=1)
        reference_date = columns[1].date_input(
            "Reference UTC date",
            value=datetime.now(timezone.utc).date(),
        )
        metar_file = columns[2].text_input("METAR file", value="data/shared/metar-ui.txt")
        update_openmeteo = columns[3].checkbox("Open-Meteo", value=True)
        submitted = st.form_submit_button("Update live data", type="primary", use_container_width=True)

    if submitted:
        if not selected_configs:
            st.warning("Select at least one location.")
            return
        try:
            with st.spinner("Fetching METAR, importing DB, and updating Open-Meteo..."):
                result = metar_service.update_live_data(
                    selected_configs,
                    hours=int(hours),
                    metar_path=metar_file,
                    reference_date=reference_date,
                    update_openmeteo=bool(update_openmeteo),
                )
        except Exception as exc:
            _render_workflow_error("Update live data failed", exc)
        else:
            _render_live_update_result(result)

    with st.expander("Manual actions", expanded=False):
        selected_actions = st.columns(3)
        if selected_actions[0].button("Fetch METAR", use_container_width=True):
            if not selected_stations:
                st.warning("Select at least one location.")
                return
            try:
                with st.spinner("Fetching METAR..."):
                    result = metar_service.fetch_metar_for_stations(selected_stations, int(hours), metar_file)
            except Exception as exc:
                _render_workflow_error("Fetch METAR failed", exc)
            else:
                st.success(f"Fetched {result['lines']} METAR lines.")
                render_json("Fetch result", result)

        if selected_actions[1].button("Import + DB", use_container_width=True):
            if not selected_configs:
                st.warning("Select at least one location.")
                return
            try:
                with st.spinner("Importing station-scoped METAR for selected locations..."):
                    result = metar_service.import_many_station_metars(
                        selected_configs,
                        metar_file,
                        reference_date,
                    )
            except Exception as exc:
                _render_workflow_error("Import METAR failed", exc)
            else:
                st.success(f"Inserted {result['inserted']} rows across {len(selected_configs)} locations.")
                render_json("Import result", result)

        if selected_actions[2].button("Sync DuckDB", use_container_width=True):
            if not selected_configs:
                st.warning("Select at least one location.")
                return
            try:
                with st.spinner("Syncing DuckDB for selected locations..."):
                    result = metar_service.sync_many_databases(selected_configs)
            except Exception as exc:
                _render_workflow_error("Sync DuckDB failed", exc)
            else:
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


def _render_live_update_result(result: dict[str, object]) -> None:
    fetch = result.get("fetch", {})
    import_result = result.get("import", {})
    station_rows = result.get("station_rows", [])
    st.success(
        "Live data update finished: "
        f"{fetch.get('lines', 0)} METAR lines fetched, "
        f"{import_result.get('inserted', 0)} CSV rows inserted, "
        f"{import_result.get('db_inserted', 0)} DB rows inserted."
    )
    render_status_metrics(
        {
            "Fetched lines": fetch.get("lines", 0),
            "CSV inserted": import_result.get("inserted", 0),
            "DB inserted": import_result.get("db_inserted", 0),
            "Stations": len(result.get("stations", [])),
        }
    )
    warnings = result.get("warnings", [])
    for warning in warnings:
        st.warning(str(warning))
    if station_rows:
        st.caption("Update coverage by location")
        st.dataframe(
            pd.DataFrame(station_rows),
            hide_index=True,
            use_container_width=True,
        )
    render_json("Live update details", result)


def _render_workflow_error(title: str, exc: Exception) -> None:
    st.error(f"{title}: {exc}")
    message = str(exc)
    if "HTTP Error 504" in message or "Gateway Time-out" in message:
        st.info(
            "The AviationWeather METAR API timed out. Retry in a few minutes, reduce lookback hours, "
            "or use an existing METAR file and run Import + DB."
        )
    elif "No such file" in message or "cannot find the file" in message.lower():
        st.info("Fetch METAR first, or enter a METAR file path that already exists.")
