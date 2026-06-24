from __future__ import annotations

from pathlib import Path

import joblib

from rksi_tmax.config import ProjectConfig
from rksi_tmax.heat_risk import (
    format_heat_risk_explanation,
    _m4_bundle_available,
    plot_prediction_curve,
    predict_heat_risk,
)
from rksi_tmax.services.db_service import latest_local_date_cutoff


def resolve_prediction_time(
    config: ProjectConfig,
    mode: str,
    local_date: str,
    cutoff_local: str,
) -> tuple[str, str]:
    if mode == "Latest in database":
        latest = latest_local_date_cutoff(config)
        if latest is None:
            raise ValueError("No latest observation found in DuckDB for this station.")
        return latest
    if mode == "Config default cutoff":
        return local_date, config.cutoff_local
    return local_date, cutoff_local


def run_prediction(
    config: ProjectConfig,
    local_date: str,
    cutoff_local: str,
    bet_temp_c: float | None = None,
    make_plot: bool = True,
    prediction_method: str | None = None,
) -> dict[str, object]:
    result = predict_heat_risk(
        config,
        local_date,
        cutoff_local,
        dataset_path=None,
        bet_temp_c=bet_temp_c,
        prediction_method_override=prediction_method,
    )
    if make_plot:
        result["plot_path"] = str(plot_prediction_curve(config, result, _plot_path(config, result)))
    result["explanation"] = format_heat_risk_explanation(result)
    return result


def available_prediction_methods(config: ProjectConfig) -> list[dict[str, str]]:
    bundle = joblib.load(config.heat_risk_model_path)
    selected = bundle.get("metrics", {}).get("selected_prediction_method", "direct")
    methods = [
        {
            "label": f"Auto ({_method_label(selected)})",
            "value": "auto",
            "description": "Use selected_prediction_method from validation.",
        }
    ]
    if bundle.get("m1_regressor") is not None:
        methods.append(
            {
                "label": "M1",
                "value": "m1",
                "description": "Use phase/history feature model.",
            }
        )
    if bundle.get("openmeteo_regressor") is not None and bundle.get("openmeteo_feature_columns", []):
        methods.append(
            {
                "label": "M3 Open-Meteo",
                "value": "openmeteo",
                "description": "Use Open-Meteo corrected model.",
            }
        )
    if _m4_bundle_available(bundle):
        methods.append(
            {
                "label": "M4 MoE",
                "value": "m4",
                "description": "Use mixture-of-experts remaining-heat model.",
            }
        )
    return methods


def _method_label(method: str) -> str:
    labels = {
        "direct": "M0 direct",
        "two_stage": "M0 two-stage",
        "m1": "M1",
        "openmeteo": "M3 Open-Meteo",
        "m4": "M4 MoE",
    }
    return labels.get(method, method)


def _plot_path(config: ProjectConfig, prediction: dict[str, object]) -> Path:
    cutoff = str(prediction["cutoff_local"]).replace(":", "")
    station = config.station.lower()
    return Path("artifacts") / station / f"{station}_{prediction['local_date']}_{cutoff}_ui_curve.png"
