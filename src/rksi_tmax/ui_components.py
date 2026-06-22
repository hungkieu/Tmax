from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


def render_json(label: str, value: Any) -> None:
    with st.expander(label):
        st.json(value)


def render_status_metrics(items: dict[str, Any]) -> None:
    columns = st.columns(min(4, max(1, len(items))))
    for column, (label, value) in zip(columns, items.items(), strict=False):
        column.metric(label, _display_value(value))


def render_artifact_links(artifacts: dict[str, dict[str, Any]]) -> None:
    for name, info in artifacts.items():
        exists = "available" if info.get("exists") else "missing"
        st.caption(f"{name}: {exists} - {info.get('path')}")


def render_plot(path: str | Path | None) -> None:
    if path and Path(path).exists():
        st.image(str(path))


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
