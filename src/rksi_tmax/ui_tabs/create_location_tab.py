from __future__ import annotations

import streamlit as st

from rksi_tmax.services.config_service import (
    LocationConfigDraft,
    create_location_config,
    default_location_draft,
)
from rksi_tmax.ui_components import render_json


def render() -> None:
    st.subheader("Create location")
    st.caption("Create a YAML config and an empty ASOS CSV header for a new station.")
    station = st.text_input("Station ICAO", value="").strip().upper()
    defaults = default_location_draft(station or "NEW")
    with st.form("create-location-form"):
        timezone = st.text_input("Timezone", value=defaults.timezone)
        cutoff_local = st.text_input("Default cutoff local", value=defaults.cutoff_local)
        complete_day_min_local = st.text_input(
            "Complete day minimum local",
            value=defaults.complete_day_min_local,
        )
        input_csv = st.text_input("Input CSV", value=defaults.input_csv)
        input_db = st.text_input("DuckDB path", value=defaults.input_db)
        raw_csv_files = st.text_input("Raw CSV files", value=input_csv)
        heat_risk_cutoffs = st.text_input(
            "Heat-risk cutoffs",
            value=", ".join(defaults.heat_risk_cutoffs),
        )
        heat_risk_thresholds = st.text_input(
            "Heat-risk thresholds C",
            value=", ".join(str(value) for value in defaults.heat_risk_thresholds_c),
        )
        use_openmeteo = st.checkbox("Configure Open-Meteo API")
        openmeteo_history_csv = None
        openmeteo_live_csv_pattern = None
        openmeteo_history_json = None
        openmeteo_live_json_pattern = None
        openmeteo_latitude = None
        openmeteo_longitude = None
        openmeteo_timezone = defaults.openmeteo_timezone
        openmeteo_training_start_date = defaults.openmeteo_training_start_date
        openmeteo_training_end_date = defaults.openmeteo_training_end_date
        if use_openmeteo:
            station_lower = (station or "station").lower()
            openmeteo_latitude = st.number_input("Open-Meteo latitude", value=0.0, format="%.6f")
            openmeteo_longitude = st.number_input("Open-Meteo longitude", value=0.0, format="%.6f")
            openmeteo_timezone = st.text_input("Open-Meteo timezone", value=defaults.openmeteo_timezone)
            openmeteo_history_json = st.text_input(
                "Open-Meteo training cache JSON",
                value=f"data/{station_lower}/openmeteo-{station_lower}-history.json",
            )
            openmeteo_live_json_pattern = st.text_input(
                "Open-Meteo daily cache JSON pattern",
                value=f"data/{station_lower}/openmeteo-{station_lower}-{{date}}.json",
            )
            openmeteo_training_start_date = st.text_input(
                "Open-Meteo training start date",
                value=defaults.openmeteo_training_start_date or "",
            )
            openmeteo_training_end_date = st.text_input(
                "Open-Meteo training end date",
                value=defaults.openmeteo_training_end_date or "",
            )
        create_csv = st.checkbox("Create empty input CSV if missing", value=True)
        submitted = st.form_submit_button("Create location", type="primary")

    if submitted:
        try:
            draft = LocationConfigDraft(
                station=station,
                timezone=timezone,
                cutoff_local=cutoff_local,
                complete_day_min_local=complete_day_min_local,
                input_csv=input_csv,
                input_db=input_db,
                raw_csv_files=tuple(_split_csv(raw_csv_files)),
                heat_risk_cutoffs=tuple(_split_csv(heat_risk_cutoffs)),
                heat_risk_thresholds_c=tuple(float(value) for value in _split_csv(heat_risk_thresholds)),
                openmeteo_history_csv=openmeteo_history_csv,
                openmeteo_live_csv_pattern=openmeteo_live_csv_pattern,
                openmeteo_history_json=openmeteo_history_json,
                openmeteo_live_json_pattern=openmeteo_live_json_pattern,
                openmeteo_latitude=openmeteo_latitude,
                openmeteo_longitude=openmeteo_longitude,
                openmeteo_timezone=openmeteo_timezone,
                openmeteo_training_start_date=openmeteo_training_start_date,
                openmeteo_training_end_date=openmeteo_training_end_date or None,
            )
            result = create_location_config(draft, create_input_csv=create_csv)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Created {result['station']} config.")
            render_json("Create result", result)
            st.rerun()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
