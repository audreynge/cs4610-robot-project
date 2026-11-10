"""Label -> bin rules for sorting litter (no ROS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from clean_sweep_perception.labels import LITTER_LABELS


@dataclass(frozen=True)
class BinRules:
    bins: dict[str, str]
    fallback: str
    min_confidence: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> BinRules:
        data = yaml.safe_load(Path(path).read_text())
        rules = cls(dict(data["bins"]), data["fallback"], float(data.get("min_confidence", 0.0)))
        missing = set(LITTER_LABELS) - set(rules.bins)
        if missing:
            raise ValueError(f"{path}: no bin for labels {sorted(missing)}")
        return rules

    def bin_for(self, label: str, confidence: float) -> str:
        if confidence < self.min_confidence:
            return self.fallback
        return self.bins.get(label, self.fallback)
