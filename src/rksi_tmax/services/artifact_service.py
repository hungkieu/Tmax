from __future__ import annotations

import json
from pathlib import Path

from rksi_tmax.config import ProjectConfig


def artifact_status(config: ProjectConfig) -> dict[str, object]:
    artifacts = {
        "dataset": config.heat_risk_dataset_parquet,
        "model": config.heat_risk_model_path,
        "metrics": config.heat_risk_metrics_path,
    }
    return {
        name: {
            "path": str(path),
            "exists": Path(path).exists(),
            "size_bytes": Path(path).stat().st_size if Path(path).exists() else None,
        }
        for name, path in artifacts.items()
    }


def read_metrics(config: ProjectConfig) -> dict[str, object] | None:
    path = Path(config.heat_risk_metrics_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
