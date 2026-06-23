"""Read airport observed temperatures directly from MongoDB.

This replaces the previous live-temperature HTTP API. The only required
configuration is the ``MONGODB_URI`` environment variable (the database name is
embedded in the URI). A ``.env`` file at the repository root is loaded
automatically for local development.

Collections (written by the upstream KMA ingestion service, Mongoose-pluralized):

- ``airporttemperaturecurrents`` -- latest observation, one document per ICAO.
- ``airporttemperaturehistories`` -- append-only observation history.

ICAO codes are stored upper-cased, so all queries upper-case the station first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

CURRENT_COLLECTION = "airporttemperaturecurrents"
HISTORY_COLLECTION = "airporttemperaturehistories"
DEFAULT_TIMEZONE = "Asia/Seoul"


@dataclass(frozen=True)
class CurrentTemperature:
    icao: str
    temp_c: float
    observed_at_utc: datetime
    observed_at_local: datetime
    trigger: str | None = None

    @property
    def observed_at_local_iso(self) -> str:
        return self.observed_at_local.isoformat()


def mongodb_uri() -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Add it to a .env file at the repository root "
            "or export it as an environment variable."
        )
    return uri


@lru_cache(maxsize=4)
def _client(uri: str) -> MongoClient:
    # tz_aware so datetimes come back as timezone-aware UTC rather than naive.
    return MongoClient(uri, tz_aware=True)


def get_client(uri: str | None = None) -> MongoClient:
    """Return a cached MongoClient (one pool per URI)."""
    return _client(uri or mongodb_uri())


def _database(uri: str | None = None):
    # Database name comes from the connection string.
    return get_client(uri).get_default_database()


def get_current_temperature(
    icao: str,
    *,
    tz: str = DEFAULT_TIMEZONE,
    uri: str | None = None,
) -> CurrentTemperature | None:
    """Latest observed temperature for one airport, or None if absent."""
    station = icao.strip().upper()
    document = _database(uri)[CURRENT_COLLECTION].find_one({"icao": station})
    if document is None:
        return None
    return _to_current_temperature(document, tz)


def get_temperature_history(
    icao: str,
    *,
    since: datetime | None = None,
    tz: str = DEFAULT_TIMEZONE,
    uri: str | None = None,
) -> pd.DataFrame:
    """Observation history for one airport as a tidy frame.

    Columns: ``valid_local`` (tz-aware station-local), ``temp_c``,
    ``observed_at_utc``. Sorted ascending by observation time.
    """
    station = icao.strip().upper()
    query: dict[str, object] = {"icao": station}
    if since is not None:
        query["observedAtUtc"] = {"$gte": _ensure_utc(since)}
    cursor = (
        _database(uri)[HISTORY_COLLECTION]
        .find(query, {"temperatureC": 1, "observedAtUtc": 1})
        .sort("observedAtUtc", 1)
    )
    zone = ZoneInfo(tz)
    rows = []
    for document in cursor:
        observed_utc = _ensure_utc(document["observedAtUtc"])
        rows.append(
            {
                "observed_at_utc": observed_utc,
                "valid_local": observed_utc.astimezone(zone),
                "temp_c": float(document["temperatureC"]),
            }
        )
    return pd.DataFrame(rows, columns=["observed_at_utc", "valid_local", "temp_c"])


def _to_current_temperature(document: dict[str, object], tz: str) -> CurrentTemperature:
    observed_utc = _ensure_utc(document["observedAtUtc"])
    return CurrentTemperature(
        icao=str(document["icao"]).upper(),
        temp_c=float(document["temperatureC"]),
        observed_at_utc=observed_utc,
        observed_at_local=observed_utc.astimezone(ZoneInfo(tz)),
        trigger=document.get("trigger"),
    )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
