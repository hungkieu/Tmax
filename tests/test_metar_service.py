from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rksi_tmax.config import ProjectConfig
from rksi_tmax.metar_import import ASOS_COLUMNS
from rksi_tmax.services.metar_service import import_many_station_metars


def test_import_many_station_metars_scopes_rows_per_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    metar_path = tmp_path / "metar.txt"
    rksi_csv = tmp_path / "rksi.csv"
    rkpk_csv = tmp_path / "rkpk.csv"
    db_path = tmp_path / "observations.duckdb"
    _write_empty_asos_csv(rksi_csv)
    _write_empty_asos_csv(rkpk_csv)
    metar_path.write_text(
        "\n".join(
            [
                "METAR RKSI 210000Z 30005KT CAVOK 24/15 Q1010 NOSIG",
                "METAR RKPK 210000Z 18006KT CAVOK 27/22 Q1008 NOSIG",
            ]
        ),
        encoding="utf-8",
    )
    configs = [
        ProjectConfig(station="RKSI", input_csv=rksi_csv, input_db=db_path),
        ProjectConfig(station="RKPK", input_csv=rkpk_csv, input_db=db_path),
    ]

    result = import_many_station_metars(configs, metar_path, reference_date="2026-06-21")

    assert result["inserted"] == 2
    assert _csv_stations(rksi_csv) == ["RKSI"]
    assert _csv_stations(rkpk_csv) == ["RKPK"]


def _write_empty_asos_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ASOS_COLUMNS, lineterminator="\n")
        writer.writeheader()


def _csv_stations(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [row["station"] for row in csv.DictReader(file)]
