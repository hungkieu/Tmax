from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rksi_tmax.metar_import import ASOS_COLUMNS, import_metar_file, parse_metar


def test_parse_basic_rksi_metar_to_asos_row() -> None:
    parsed = parse_metar(
        "METAR RKSI 172300Z 30005KT 260V330 CAVOK 24/15 Q1010 NOSIG",
        reference_date="2026-06-18",
    )

    assert parsed.station == "RKSI"
    assert parsed.row["valid"] == "2026-06-17 23:00"
    assert parsed.row["tmpf"] == "75.20"
    assert parsed.row["dwpf"] == "59.00"
    assert parsed.row["drct"] == "300.00"
    assert parsed.row["sknt"] == "5.00"
    assert parsed.row["vsby"] == "6.21"
    assert parsed.row["alti"] == "29.83"
    assert parsed.row["metar"] == "METAR RKSI 172300Z 30005KT 260V330 CAVOK 24/15 Q1010 NOSIG"


def test_parse_negative_temperature_and_cloud_layers() -> None:
    parsed = parse_metar(
        "RKSI 010000Z 12005KT 9999 -SN SCT030 BKN100 M02/M08 Q1020",
        reference_date="2026-01-01",
    )

    assert parsed.row["valid"] == "2026-01-01 00:00"
    assert parsed.row["tmpf"] == "28.40"
    assert parsed.row["dwpf"] == "17.60"
    assert parsed.row["wxcodes"] == "-SN"
    assert parsed.row["skyc1"] == "SCT"
    assert parsed.row["skyl1"] == "3000.00"
    assert parsed.row["skyc2"] == "BKN"
    assert parsed.row["skyl2"] == "10000.00"


def test_parse_speci_prefix_to_asos_row() -> None:
    parsed = parse_metar(
        "SPECI RKPK 180941Z 18006KT CAVOK 25/22 Q1008 RMK CIG250",
        reference_date="2026-06-19",
    )

    assert parsed.station == "RKPK"
    assert parsed.row["valid"] == "2026-06-18 09:41"
    assert parsed.row["tmpf"] == "77.00"
    assert parsed.row["dwpf"] == "71.60"
    assert parsed.row["metar"] == "SPECI RKPK 180941Z 18006KT CAVOK 25/22 Q1008 RMK CIG250"


def test_import_metar_file_skips_existing_station_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "asos.csv"
    metar_path = tmp_path / "metar.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ASOS_COLUMNS, lineterminator="\n")
        writer.writeheader()
    metar_path.write_text(
        "\n".join(
            [
                "METAR RKSI 172300Z 30005KT CAVOK 24/15 Q1010 NOSIG",
                "METAR RKSI 180000Z 31006KT CAVOK 25/15 Q1010 NOSIG",
            ]
        ),
        encoding="utf-8",
    )

    first = import_metar_file(metar_path, csv_path, reference_date="2026-06-18")
    second = import_metar_file(metar_path, csv_path, reference_date="2026-06-18")

    assert first["inserted"] == 2
    assert second["inserted"] == 0
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2


def test_parse_rejects_missing_time_token() -> None:
    with pytest.raises(ValueError):
        parse_metar("METAR RKSI CAVOK 24/15 Q1010", reference_date="2026-06-18")
