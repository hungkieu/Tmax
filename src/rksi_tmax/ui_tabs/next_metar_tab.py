from __future__ import annotations

import importlib

import streamlit as st

from rksi_tmax.services import next_metar_service
from rksi_tmax.ui_components import render_json, render_status_metrics

_STATE_KEY = "next_metar_last_run"
_STATUS_ICON = {
    "ok": "✅",
    "warning": "⚠️",
    "skipped": "⏭️",
    "error": "❌",
}
_HEALTH_ICON = {"healthy": "🟢", "unhealthy": "🔴", "unknown": "⚪"}


def render(station: str) -> None:
    station = station.upper()
    state_key = f"{_STATE_KEY}_{station}"
    st.subheader(f"Next-METAR — Dự báo nhiệt METAR kế tiếp ({station})")
    st.caption(
        f"Bấm **Run** một lần: hệ thống tự đọc nhiệt live của **{station}** từ "
        "MongoDB, dự báo METAR kế tiếp, rồi chấm điểm các dự báo đang chờ. "
        "Bạn không cần thao tác gì thêm."
    )

    if st.button("▶  Run", type="primary", use_container_width=True):
        importlib.reload(next_metar_service)
        with st.spinner("Đang chạy: đọc MongoDB → dự báo → chấm điểm..."):
            st.session_state[state_key] = next_metar_service.run_cycle(stations=[station])

    result = st.session_state.get(state_key)
    if result is None:
        st.info("Chưa chạy lần nào. Bấm **Run** để bắt đầu.")
        return

    _render_result(result)


def _render_result(result: dict[str, object]) -> None:
    st.caption(f"Lần chạy gần nhất: {result.get('ran_at')}")

    if not result.get("ok"):
        st.error(result.get("error") or "Chạy thất bại.")
        st.markdown(
            "Kiểm tra biến môi trường **`MONGODB_URI`** trong file `.env` ở gốc dự án. "
            "Xem hướng dẫn: `docs/next-metar-temp-usage.md`."
        )

    st.markdown("#### Các việc đã làm")
    for step in result.get("steps", []):
        icon = _STATUS_ICON.get(str(step.get("status")), "•")
        st.markdown(f"{icon} **{step.get('title')}** — {step.get('summary')}")
        detail = step.get("detail")
        if detail:
            render_json(f"Chi tiết: {step.get('title')}", detail)

    health = result.get("health") or {}
    if health:
        status = str(health.get("status", "unknown"))
        st.markdown(f"#### Sức khỏe model {_HEALTH_ICON.get(status, '')} `{status}`")
        render_status_metrics(
            {
                "MAE °C": health.get("mae_c"),
                "Sai lệch (bias)": health.get("bias_c"),
                "Đúng tuyệt đối": _percent(health.get("exact_accuracy")),
                "Trong ±1°C": _percent(health.get("within_1c_accuracy")),
            }
        )
        reasons = health.get("reasons") or []
        if status == "unhealthy" and reasons:
            st.warning("Lý do: " + ", ".join(str(reason) for reason in reasons))

    recent = result.get("recent") or []
    if recent:
        st.markdown("#### Dự báo gần đây")
        st.dataframe(recent, use_container_width=True, hide_index=True)


def _percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"
