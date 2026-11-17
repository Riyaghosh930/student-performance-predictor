"""Central configuration loader."""
from pathlib import Path
import yaml  # type: ignore[reportMissingModuleSource]
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        with open(path) as f:
            return cls(raw=yaml.safe_load(f))

    @property
    def data(self) -> dict: return self.raw["data"]
    @property
    def features(self) -> dict: return self.raw["features"]
    @property
    def model(self) -> dict: return self.raw["model"]
    @property
    def paths(self) -> dict: return self.raw["paths"]

    def ensure_dirs(self):
        for p in self.paths.values():
            Path(p).mkdir(parents=True, exist_ok=True)

CONFIG = Config.load()