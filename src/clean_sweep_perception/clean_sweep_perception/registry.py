"""Turns per-frame detections into stable, numbered litter objects (no ROS).

Each frame's detections are matched to known objects by distance in the map
frame. Repeated sightings confirm an object (filtering one-frame false
positives) and refine its position, size and label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class State(IntEnum):
    """Mirrors the STATE_* constants in clean_sweep_interfaces/LitterObject.msg."""

    SEEN = 0
    CONFIRMED = 1
    TARGETED = 2
    PICKED = 3
    FAILED = 4


@dataclass(frozen=True)
class Observation:
    position: np.ndarray  # (3,) map frame
    width: float
    height: float
    depth_valid: bool
    detection_confidence: float
    label: str | None
    label_confidence: float


@dataclass
class TrackedObject:
    id: int
    position: np.ndarray
    width: float
    height: float
    depth_valid: bool
    detection_confidence: float
    state: State
    sightings: int
    first_seen: float
    last_seen: float
    weight: float  # total observation weight behind the current estimate (capped)
    label_votes: Counter = field(default_factory=Counter)  # label -> summed classifier confidence
    label_counts: Counter = field(default_factory=Counter)  # label -> number of votes

    @property
    def label(self) -> str:
        return self.label_votes.most_common(1)[0][0] if self.label_votes else "other"

    @property
    def label_confidence(self) -> float:
        """Mean classifier confidence of the winning label, scaled by its share of the votes.

        Consistent confident votes give a high value; consistent *unsure* votes stay
        low (so sorting can fall back to landfill), and so do split votes.
        """
        if not self.label_votes:
            return 0.0
        label = self.label
        mean_confidence = self.label_votes[label] / self.label_counts[label]
        share = self.label_counts[label] / sum(self.label_counts.values())
        return mean_confidence * share


@dataclass(frozen=True)
class RegistryParams:
    match_radius: float = 0.15  # m; observations this close to an object are the same object
    confirm_sightings: int = 3  # sightings needed before an object is CONFIRMED
    forget_after: float = 10.0  # s; drop unconfirmed objects not seen for this long
    depth_weight: float = 1.0  # how much a depth-based observation counts...
    floor_plane_weight: float = 0.3  # ...versus a floor-plane (no depth) one
    max_weight: float = 10.0  # cap, so estimates keep adapting to new observations


class LitterRegistry:
    def __init__(self, params: RegistryParams = RegistryParams()):
        self._params = params
        self._objects: dict[int, TrackedObject] = {}
        self._next_id = 1

    def objects(self) -> list[TrackedObject]:
        return sorted(self._objects.values(), key=lambda o: o.id)

    def get(self, object_id: int) -> TrackedObject | None:
        return self._objects.get(object_id)

    def update(self, observations: list[Observation], now: float) -> None:
        """Merge one frame's observations, then forget stale unconfirmed objects."""
        matched: set[int] = set()
        for obs in observations:
            target = self.nearest(obs.position, self._params.match_radius, exclude=matched)
            if target is None:
                target = self._create(obs, now)
            else:
                self._merge(target, obs, now)
            matched.add(target.id)

        for obj in list(self._objects.values()):
            if obj.state == State.SEEN and now - obj.last_seen > self._params.forget_after:
                del self._objects[obj.id]

    def nearest(self, position: np.ndarray, radius: float,
                exclude: set[int] = frozenset()) -> TrackedObject | None:
        """Closest object (horizontally) within `radius`, ignoring picked ones."""
        best, best_distance = None, radius
        for obj in self._objects.values():
            if obj.id in exclude or obj.state == State.PICKED:
                continue
            distance = float(np.linalg.norm(obj.position[:2] - position[:2]))
            if distance <= best_distance:
                best, best_distance = obj, distance
        return best

    def set_state(self, object_id: int, state: State) -> bool:
        obj = self._objects.get(object_id)
        if obj is None:
            return False
        obj.state = state
        return True

    def refine(self, object_id: int, observations: list[Observation], radius: float,
               since: float, now: float) -> int:
        """Re-estimate an object from close-range observations made since `since`.

        Uses observations within `radius` that aren't closer to another object that
        already existed, restarts the estimate from them, and absorbs objects created
        since `since` around the new position (the far estimate was off, so the close
        detections spawned duplicates meanwhile). Returns how many observations were used.
        """
        obj = self._objects.get(object_id)
        if obj is None:
            return 0
        neighbours = [o for o in self._objects.values()
                      if o.id != object_id and o.state != State.PICKED and o.first_seen < since]

        def distance(o: TrackedObject, obs: Observation) -> float:
            return float(np.linalg.norm(o.position[:2] - obs.position[:2]))

        used = [obs for obs in observations
                if distance(obj, obs) <= radius
                and all(distance(n, obs) > distance(obj, obs) for n in neighbours)]
        if not used:
            return 0
        obj.weight = 0.0
        obj.depth_valid = False
        for obs in used:
            self._merge(obj, obs, now)
        for other in list(self._objects.values()):
            if (other.id != object_id and other.first_seen >= since and other.state != State.PICKED
                    and np.linalg.norm(other.position[:2] - obj.position[:2]) <= self._params.match_radius):
                del self._objects[other.id]
        return len(used)

    def _weight(self, obs: Observation) -> float:
        return self._params.depth_weight if obs.depth_valid else self._params.floor_plane_weight

    def _create(self, obs: Observation, now: float) -> TrackedObject:
        obj = TrackedObject(
            id=self._next_id, position=obs.position.astype(float).copy(), width=obs.width,
            height=obs.height, depth_valid=obs.depth_valid,
            detection_confidence=obs.detection_confidence, state=State.SEEN, sightings=1,
            first_seen=now, last_seen=now, weight=self._weight(obs),
        )
        if obs.label:
            obj.label_votes[obs.label] += obs.label_confidence
            obj.label_counts[obs.label] += 1
        self._objects[obj.id] = obj
        self._next_id += 1
        self._maybe_confirm(obj)
        return obj

    def _merge(self, obj: TrackedObject, obs: Observation, now: float) -> None:
        w = self._weight(obs)
        total = obj.weight + w
        obj.position = (obj.position * obj.weight + obs.position * w) / total
        obj.width = (obj.width * obj.weight + obs.width * w) / total
        obj.height = (obj.height * obj.weight + obs.height * w) / total
        obj.weight = min(total, self._params.max_weight)
        obj.depth_valid = obj.depth_valid or obs.depth_valid
        obj.detection_confidence = max(obj.detection_confidence, obs.detection_confidence)
        if obs.label:
            obj.label_votes[obs.label] += obs.label_confidence
            obj.label_counts[obs.label] += 1
        obj.sightings += 1
        obj.last_seen = now
        self._maybe_confirm(obj)

    def _maybe_confirm(self, obj: TrackedObject) -> None:
        if obj.state == State.SEEN and obj.sightings >= self._params.confirm_sightings:
            obj.state = State.CONFIRMED
