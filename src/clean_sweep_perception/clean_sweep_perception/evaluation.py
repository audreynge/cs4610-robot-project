"""Score perception against tape-measured ground truth (no ROS).

Ground truth uses the same YAML format as the sim stub (see stub_scene.py):
place litter, measure where it is in the map frame, list it, and compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clean_sweep_perception.stub_scene import StubObject


@dataclass(frozen=True)
class Detected:
    id: int
    label: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class EvaluationReport:
    matches: list[tuple[StubObject, Detected, float]]  # (truth, detection, horizontal error m)
    missed: list[StubObject]
    false_positives: list[Detected]

    @property
    def recall(self) -> float:
        total = len(self.matches) + len(self.missed)
        return len(self.matches) / total if total else 1.0

    @property
    def precision(self) -> float:
        total = len(self.matches) + len(self.false_positives)
        return len(self.matches) / total if total else 1.0

    @property
    def mean_error(self) -> float:
        return float(np.mean([e for _, _, e in self.matches])) if self.matches else float("nan")

    @property
    def label_accuracy(self) -> float:
        if not self.matches:
            return float("nan")
        return sum(t.label == d.label for t, d, _ in self.matches) / len(self.matches)

    def summary(self) -> str:
        lines = [
            f"recall     {self.recall:6.1%}  ({len(self.matches)} of {len(self.matches) + len(self.missed)} found)",
            f"precision  {self.precision:6.1%}  ({len(self.false_positives)} false positives)",
            f"position   {self.mean_error * 100:5.1f} cm mean horizontal error",
            f"labels     {self.label_accuracy:6.1%} correct",
        ]
        lines += [f"  matched  {t.label:<6} -> #{d.id} {d.label:<6} {e * 100:4.1f} cm" for t, d, e in self.matches]
        lines += [f"  MISSED   {t.label:<6} at {t.position[:2]}" for t in self.missed]
        lines += [f"  FALSE +  #{d.id} {d.label:<6} at ({d.position[0]:.2f}, {d.position[1]:.2f})"
                  for d in self.false_positives]
        return "\n".join(lines)


def evaluate(truth: list[StubObject], detected: list[Detected], radius: float = 0.25) -> EvaluationReport:
    """Match closest pairs first (one-to-one) within `radius` meters, horizontally."""
    pairs = sorted(
        (float(np.linalg.norm(np.subtract(t.position[:2], d.position[:2]))), ti, di)
        for ti, t in enumerate(truth) for di, d in enumerate(detected)
    )
    used_truth, used_detected, matches = set(), set(), []
    for distance, ti, di in pairs:
        if distance > radius:
            break
        if ti in used_truth or di in used_detected:
            continue
        used_truth.add(ti)
        used_detected.add(di)
        matches.append((truth[ti], detected[di], distance))
    return EvaluationReport(
        matches=matches,
        missed=[t for i, t in enumerate(truth) if i not in used_truth],
        false_positives=[d for i, d in enumerate(detected) if i not in used_detected],
    )
