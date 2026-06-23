from __future__ import annotations

import csv
import math
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


ASOS_COLUMNS = [
    "station",
    "valid",
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "wxcodes",
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust",
    "peak_wind_drct",
    "peak_wind_time",
    "feel",
    "metar",
    "snowdepth",
]

TIME_RE = re.compile(r"^(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})Z$")
WIND_RE = re.compile(r"^(?P<dir>\d{3}|VRB)(?P<speed>\d{2,3})(G(?P<gust>\d{2,3}))?KT$")
TEMP_RE = re.compile(r"^(?P<temp>M?\d{2})/(?P<dew>M?\d{2}|//)$")
QNH_RE = re.compile(r"^Q(?P<qnh>\d{4})$")
VIS_SM_RE = re.compile(r"^(?P<vis>\d+)(SM)$")
CLOUD_RE = re.compile(r"^(?P<cover>FEW|SCT|BKN|OVC|VV)(?P<height>\d{3}|///)?")
WEATHER_RE = re.compile(r"^[-+]?([A-Z]{2})+$")


@dataclass(frozen=True)
class ParsedMetar:
    station: str
    valid: datetime
    row: dict[str, str]


def import_metar_file(
    metar_path: str | Path = "metar.txt",
    csv_path: str | Path = "data/rksi/asos.csv",
    reference_date: str | date | None = None,
    db_path: str | Path | None = None,
) -> dict:
    metar_file = Path(metar_path)
    target_csv = Path(csv_path)
    reference = _reference_date(reference_date)

    lines = [
        line.strip()
        for line in metar_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    rows = [parse_metar(line, reference).row for line in lines]
    existing_keys = _existing_station_valid_keys(target_csv)
    new_rows = [
        row
        for row in rows
        if (row["station"], row["valid"]) not in existing_keys
    ]

    if new_rows:
        with target_csv.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=ASOS_COLUMNS, lineterminator="\n")
            writer.writerows(new_rows)

    db_result = None
    if db_path is not None:
        from rksi_tmax.storage import upsert_observation_rows

        db_result = upsert_observation_rows(new_rows, db_path)

    return {
        "metar_file": str(metar_file),
        "csv_path": str(target_csv),
        "db_path": str(db_path) if db_path is not None else None,
        "read": len(rows),
        "inserted": len(new_rows),
        "skipped_existing": len(rows) - len(new_rows),
        "db_inserted": db_result["inserted"] if db_result else None,
    }


def fetch_metar_text(
    stations: list[str],
    hours: int = 48,
    output_path: str | Path = "metar.txt",
) -> dict:
    if not stations:
        raise ValueError("At least one station is required.")
    normalized = [station.strip().upper() for station in stations if station.strip()]
    query = urlencode({"ids": ",".join(normalized), "hours": int(hours), "sep": "true"})
    url = f"https://aviationweather.gov/api/data/metar?{query}"

    with urlopen(url, timeout=30) as response:
        content = response.read().decode("utf-8")

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    output = Path(output_path)
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "url": url,
        "output_path": str(output),
        "stations": normalized,
        "hours": int(hours),
        "lines": len(lines),
    }


def parse_metar(metar: str, reference_date: str | date | None = None) -> ParsedMetar:
    reference = _reference_date(reference_date)
    tokens = _clean_metar(metar).split()
    if tokens and tokens[0] in {"METAR", "SPECI"}:
        tokens = tokens[1:]
    if len(tokens) < 3:
        raise ValueError(f"METAR line is too short: {metar}")

    station = tokens[0]
    time_token = tokens[1]
    time_match = TIME_RE.match(time_token)
    if time_match is None:
        raise ValueError(f"Could not parse METAR time token {time_token!r}: {metar}")
    valid = _infer_valid_datetime(
        day=int(time_match.group("day")),
        hour=int(time_match.group("hour")),
        minute=int(time_match.group("minute")),
        reference=reference,
    )

    fields: dict[str, float | str | None] = {column: None for column in ASOS_COLUMNS}
    fields["station"] = station
    fields["valid"] = valid.strftime("%Y-%m-%d %H:%M")
    fields["p01i"] = 0.0
    fields["metar"] = metar.strip()

    clouds: list[tuple[str, float | None]] = []
    weather_codes: list[str] = []

    for token in tokens[2:]:
        if token in {"AUTO", "COR", "NOSIG", "RMK"}:
            continue
        if token == "CAVOK":
            fields["vsby"] = 6.21
            continue
        if wind_match := WIND_RE.match(token):
            direction = wind_match.group("dir")
            fields["drct"] = None if direction == "VRB" else float(direction)
            fields["sknt"] = float(wind_match.group("speed"))
            if wind_match.group("gust"):
                fields["gust"] = float(wind_match.group("gust"))
            continue
        if TEMP_RE.match(token):
            temp_text, dew_text = token.split("/", maxsplit=1)
            fields["tmpf"] = _celsius_to_fahrenheit(_parse_signed_temperature(temp_text))
            if dew_text != "//":
                fields["dwpf"] = _celsius_to_fahrenheit(_parse_signed_temperature(dew_text))
            if fields["tmpf"] is not None and fields["dwpf"] is not None:
                fields["relh"] = _relative_humidity_f(float(fields["tmpf"]), float(fields["dwpf"]))
            fields["feel"] = fields["tmpf"]
            continue
        if qnh_match := QNH_RE.match(token):
            fields["alti"] = round(float(qnh_match.group("qnh")) * 0.029529983071445, 2)
            continue
        if token.isdigit() and len(token) == 4:
            fields["vsby"] = 6.21 if token == "9999" else round(float(token) / 1609.344, 2)
            continue
        if vis_match := VIS_SM_RE.match(token):
            fields["vsby"] = float(vis_match.group("vis"))
            continue
        if cloud_match := CLOUD_RE.match(token):
            cover = cloud_match.group("cover")
            height_text = cloud_match.group("height")
            height = None if height_text in {None, "///"} else float(height_text) * 100.0
            clouds.append((cover, height))
            continue
        if _looks_like_weather_code(token):
            weather_codes.append(token)

    for index, (cover, height) in enumerate(clouds[:4], start=1):
        fields[f"skyc{index}"] = cover
        fields[f"skyl{index}"] = height
    if weather_codes:
        fields["wxcodes"] = " ".join(weather_codes)

    row = {column: _format_csv_value(fields[column]) for column in ASOS_COLUMNS}
    return ParsedMetar(station=station, valid=valid, row=row)


def _existing_station_valid_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys.add((row["station"], row["valid"]))
    return keys


def _clean_metar(metar: str) -> str:
    return metar.strip().replace("=", "")


def _reference_date(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.now(timezone.utc).date()


def _infer_valid_datetime(day: int, hour: int, minute: int, reference: date) -> datetime:
    candidates: list[datetime] = []
    for year, month in _candidate_year_months(reference):
        if day <= monthrange(year, month)[1]:
            candidates.append(datetime(year, month, day, hour, minute))
    if not candidates:
        raise ValueError(f"Could not infer date for METAR day {day} near {reference}.")
    reference_dt = datetime(reference.year, reference.month, reference.day)
    return min(candidates, key=lambda candidate: abs(candidate - reference_dt))


def _candidate_year_months(reference: date) -> list[tuple[int, int]]:
    months = []
    for offset in (-1, 0, 1):
        month_index = reference.month + offset
        year = reference.year + (month_index - 1) // 12
        month = (month_index - 1) % 12 + 1
        months.append((year, month))
    return months


def _parse_signed_temperature(value: str) -> float:
    return -float(value[1:]) if value.startswith("M") else float(value)


def _celsius_to_fahrenheit(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def _relative_humidity_f(tmpf: float, dwpf: float) -> float:
    tmpc = (tmpf - 32.0) * 5.0 / 9.0
    dwpc = (dwpf - 32.0) * 5.0 / 9.0
    saturation = 6.112 * math.exp((17.67 * tmpc) / (tmpc + 243.5))
    actual = 6.112 * math.exp((17.67 * dwpc) / (dwpc + 243.5))
    return round(max(0.0, min(100.0, actual / saturation * 100.0)), 2)


def _looks_like_weather_code(token: str) -> bool:
    if token in {"BECMG", "TEMPO"}:
        return False
    return bool(WEATHER_RE.match(token)) and any(
        code in token for code in ("RA", "SN", "DZ", "TS", "SH", "BR", "FG", "HZ", "GR", "PL")
    )


def _format_csv_value(value: float | str | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.2f}"
    return value
