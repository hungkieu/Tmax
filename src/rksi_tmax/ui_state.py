from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UiConfigSelection:
    label: str
    path: Path
    station: str
