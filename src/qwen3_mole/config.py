from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def base_model(self) -> str:
        return str(self.raw["base_model"])

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["domains"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {name!r} is missing")
        return value


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    required = {"base_model", "domains", "data", "lora", "training", "router"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"missing configuration keys: {', '.join(missing)}")
    domains = raw["domains"]
    if not isinstance(domains, list) or not domains or len(set(domains)) != len(domains):
        raise ValueError("domains must be a non-empty list of unique names")
    return ProjectConfig(raw=raw, path=config_path)
