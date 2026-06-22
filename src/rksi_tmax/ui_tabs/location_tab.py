from __future__ import annotations

import streamlit as st

from rksi_tmax.services.config_service import (
    ConfigOption,
    LocationConfigDraft,
    create_location_config,
    default_location_draft,
)
from rksi_tmax.ui_components import render_json


def render(config_options: list[ConfigOption]) -> None:
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
        use_openmeteo = st.checkbox("Configure Open-Meteo files")
        openmeteo_history_csv = None
        openmeteo_live_csv_pattern = None
        if use_openmeteo:
            station_lower = (station or "station").lower()
            openmeteo_history_csv = st.text_input(
                "Open-Meteo history CSV",
                value=f"openmeteo-{station_lower}.csv",
            )
            openmeteo_live_csv_pattern = st.text_input(
                "Open-Meteo live CSV pattern",
                value=f"openmeteo-{station_lower}-{{date}}.csv",
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
