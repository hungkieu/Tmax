from __future__ import annotations

import csv
from pathlib import Path

from rksi_tmax.storage import ASOS_COLUMNS, read_station_observations_from_duckdb
from rksi_tmax.storage import sync_duckdb_from_csv, upsert_observation_rows


def test_sync_duckdb_from_csv_dedupes_station_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "asos.csv"
    db_path = tmp_path / "observations.duckdb"
    rows = [
        {"station": "RKSI", "valid": "2026-06-18 00:00", "tmpf": "75.20", "metar": "old"},
        {"station": "RKSI", "valid": "2026-06-18 00:00", "tmpf": "77.00", "metar": "new"},
        {"station": "RJTT", "valid": "2026-06-18 00:00", "tmpf": "80.60", "metar": "tokyo"},
    ]
    _write_asos_csv(csv_path, rows)

    result = sync_duckdb_from_csv([csv_path], db_path)
    observations = read_station_observations_from_duckdb(
        db_path,
        "RKSI",
        ["station", "valid", "tmpf", "metar"],
    )

    assert result["rows"] == 2
    assert result["stations"] == ["RJTT", "RKSI"]
    assert len(observations) == 1


def test_upsert_observation_rows_skips_existing_key(tmp_path: Path) -> None:
    db_path = tmp_path / "observations.duckdb"
    row = {"station": "RKSI", "valid": "2026-06-18 00:00", "tmpf": "75.20"}

    first = upsert_observation_rows([row], db_path)
    second = upsert_observation_rows([row], db_path)
    observations = read_station_observations_from_duckdb(
        db_path,
        "RKSI",
        ["station", "valid", "tmpf"],
    )

    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert len(observations) == 1


def _write_asos_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ASOS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            full_row = {column: "null" for column in ASOS_COLUMNS}
            full_row.update(row)
            writer.writerow(full_row)
