from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from rksi_tmax.services.config_service import (
    delete_location_config,
    discover_config_options,
    load_selected_config,
    summarize_config,
)
from rksi_tmax.ui_components import render_json
from rksi_tmax.ui_tabs import location_tab, metar_tab, predict_tab, train_tab


def main() -> None:
    st.set_page_config(page_title="RKSI Tmax Dashboard", layout="wide")
    st.title("Tmax Operations")

    options = discover_config_options()
    if not options:
        st.error("No YAML configs found in configs/.")
        return

    labels = [option.label for option in options]
    selected_label = st.sidebar.selectbox("Location config", labels)
    selected = options[labels.index(selected_label)]
    config = load_selected_config(selected.path)

    st.sidebar.caption(f"Station: {config.station}")
    st.sidebar.caption(f"Timezone: {config.timezone}")
    st.sidebar.caption(f"Config: {selected.path}")
    with st.sidebar.expander("Config summary"):
        st.json(summarize_config(config, selected.path))
    with st.sidebar.expander("Delete location config"):
        st.caption("Deletes only the YAML config. CSV, DuckDB, and model artifacts are kept.")
        confirmation = st.text_input(
            "Type station code to confirm",
            key=f"delete-confirm-{selected.path}",
        )
        if st.button("Delete selected config", type="secondary", use_container_width=True):
            try:
                result = delete_location_config(selected.path, confirmation)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Deleted {result['station']} config.")
                st.rerun()

    locations, metar, train, predict = st.tabs(
        ["Locations", "METAR", "Train / Validate", "Predict"]
    )
    with locations:
        location_tab.render(options)
    with metar:
        metar_tab.render(config, options)
    with train:
        train_tab.render(config)
    with predict:
        predict_tab.render(config)

    render_json("Active config", summarize_config(config, selected.path))


def run() -> None:
    app_path = Path(__file__).resolve()
    env = {
        **os.environ,
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    args = sys.argv[1:]
    default_args = [
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
    ]
    raise SystemExit(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                *default_args,
                *args,
            ],
            check=False,
            env=env,
        ).returncode
    )


if __name__ == "__main__":
    main()
