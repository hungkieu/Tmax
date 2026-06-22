from __future__ import annotations

from pathlib import Path

import pytest

from rksi_tmax.services.config_service import (
    LocationConfigDraft,
    create_location_config,
    delete_location_config,
)


def test_create_location_config_writes_yaml_and_empty_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    draft = LocationConfigDraft(
        station="VTBS",
        timezone="Asia/Bangkok",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv="VTBS.csv",
        input_db="artifacts/observations.duckdb",
        raw_csv_files=("VTBS.csv",),
        heat_risk_cutoffs=("09:00", "10:00"),
        heat_risk_thresholds_c=(30.0, 31.0),
    )

    result = create_location_config(draft)

    assert result["station"] == "VTBS"
    assert Path("configs/vtbs.yaml").exists()
    assert Path("VTBS.csv").read_text(encoding="utf-8").startswith("station,valid,tmpf")


def test_create_location_config_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/vtbs.yaml").write_text("station: VTBS\n", encoding="utf-8")
    draft = LocationConfigDraft(
        station="VTBS",
        timezone="Asia/Bangkok",
        cutoff_local="09:00",
        complete_day_min_local="23:00",
        input_csv="VTBS.csv",
        input_db="artifacts/observations.duckdb",
        raw_csv_files=("VTBS.csv",),
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
