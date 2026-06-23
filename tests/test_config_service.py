from __future__ import annotations

from pathlib import Path

import pytest

from rksi_tmax.services.config_service import (
    LocationConfigDraft,
    create_location_config,
    delete_location_config,
    update_location_openmeteo_config,
)
from rksi_tmax.config import load_config


def test_create_location_config_writes_yaml_and_empty_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    draft = LocationConfigDraft(
        station="VTBS",
        timezone="Asia/Bangkok",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv="data/vtbs/VTBS.csv",
        input_db="artifacts/shared/observations.duckdb",
        raw_csv_files=("data/vtbs/VTBS.csv",),
        heat_risk_cutoffs=("09:00", "10:00"),
        heat_risk_thresholds_c=(30.0, 31.0),
        openmeteo_history_json="data/vtbs/openmeteo-vtbs-history.json",
        openmeteo_live_json_pattern="data/vtbs/openmeteo-vtbs-{date}.json",
        openmeteo_latitude=13.69,
        openmeteo_longitude=100.75,
    )

    result = create_location_config(draft)

    assert result["station"] == "VTBS"
    assert Path("configs/vtbs.yaml").exists()
    config_text = Path("configs/vtbs.yaml").read_text(encoding="utf-8")
    assert "openmeteo_latitude: 13.69" in config_text
    assert "openmeteo_history_json: data/vtbs/openmeteo-vtbs-history.json" in config_text
    assert "openmeteo_training_start_date: '2023-01-01'" in config_text
    assert Path("data/vtbs/VTBS.csv").read_text(encoding="utf-8").startswith("station,valid,tmpf")


def test_create_location_config_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/vtbs.yaml").write_text("station: VTBS\n", encoding="utf-8")
    draft = LocationConfigDraft(
        station="VTBS",
        timezone="Asia/Bangkok",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv="data/vtbs/VTBS.csv",
        input_db="artifacts/shared/observations.duckdb",
        raw_csv_files=("data/vtbs/VTBS.csv",),
        heat_risk_cutoffs=("09:00",),
        heat_risk_thresholds_c=(30.0,),
    )

    with pytest.raises(FileExistsError):
        create_location_config(draft)


def test_delete_location_config_requires_station_confirmation(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "wihh.yaml"
    path.write_text("station: WIHH\n", encoding="utf-8")

    with pytest.raises(ValueError):
        delete_location_config(path, "RKSI")

    assert path.exists()


def test_delete_location_config_deletes_matching_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "wihh.yaml"
    path.write_text("station: WIHH\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = delete_location_config(Path("configs/wihh.yaml"), "WIHH")

    assert result["deleted"] is True
    assert result["station"] == "WIHH"
    assert not path.exists()


def test_update_location_openmeteo_config_adds_m3_to_existing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "rkpk.yaml"
    path.write_text(
        "\n".join(
            [
                "station: RKPK",
                "timezone: Asia/Seoul",
                'cutoff_local: "09:00"',
                'complete_day_min_local: "23:00"',
                "input_csv: data/rkpk/asos.csv",
                "raw_csv_files:",
                "  - data/rkpk/asos.csv",
                "heat_risk_dataset_parquet: artifacts/rkpk/rkpk_heat_risk_dataset.parquet",
                "heat_risk_model_path: artifacts/rkpk/rkpk_heat_risk_model.joblib",
                "heat_risk_metrics_path: artifacts/rkpk/rkpk_heat_risk_metrics.json",
            ]
        ),
        encoding="utf-8",
    )

    result = update_location_openmeteo_config(
        Path("configs/rkpk.yaml"),
        latitude=35.1796,
        longitude=129.0756,
        training_start_date="2023-01-01",
    )

    config = load_config(Path("configs/rkpk.yaml"))
    assert result["updated"] is True
    assert config.openmeteo_latitude == 35.1796
    assert config.openmeteo_longitude == 129.0756
    assert config.openmeteo_history_json == Path("data/rkpk/openmeteo-rkpk-history.json")
    assert config.openmeteo_live_json_pattern == "data/rkpk/openmeteo-rkpk-{date}.json"
    assert config.openmeteo_timezone == "GMT"
    assert config.openmeteo_training_start_date == "2023-01-01"
