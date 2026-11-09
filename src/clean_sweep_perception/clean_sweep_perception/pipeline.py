"""One camera frame in, located and checked litter out (no ROS).

    raw color -> upright -> detector -> crops -> classifier
              -> boxes back to raw pixels -> 3D localization -> sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from clean_sweep_perception.checks import rejection_reason
from clean_sweep_perception.geometry import (
    Box, CameraModel, Localization, LocalizerParams, Transform, localize,
)
from clean_sweep_perception.image_rotation import box_to_raw, rotate_upright


class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[tuple[Box, float]]: ...


class Classifier(Protocol):
    def classify(self, crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class Detection:
    upright_box: Box  # in the upright image (for drawing)
    raw_box: Box  # in the raw image (for geometry)
    confidence: float
    label: str | None  # None when there is no classifier
    label_confidence: float


@dataclass(frozen=True)
class FrameResult:
    accepted: list[tuple[Detection, Localization]]
    rejected: list[tuple[Detection, str]]  # with the reason
    upright_image: np.ndarray


class LitterPerception:
    def __init__(self, detector: Detector, classifier: Classifier | None, rotation: str,
                 crop_padding: float = 0.1, localizer_params: LocalizerParams = LocalizerParams()):
        """`crop_padding` must match the classifier dataset build (--pad, default 0.1)."""
        self._detector = detector
        self._classifier = classifier
        self._rotation = rotation
        self._crop_padding = crop_padding
        self._localizer_params = localizer_params

    def process(self, color_bgr: np.ndarray, depth_m: np.ndarray, camera: CameraModel,
                camera_to_base: Transform) -> FrameResult:
        upright = rotate_upright(color_bgr, self._rotation)
        raw_detections = self._detector.detect(upright)
        labels = self._classify(upright, [box for box, _ in raw_detections])

        accepted, rejected = [], []
        for (box, confidence), (label, label_confidence) in zip(raw_detections, labels):
            detection = Detection(box, box_to_raw(box, self._rotation, color_bgr.shape),
                                  confidence, label, label_confidence)
            localization = localize(detection.raw_box, depth_m, camera, camera_to_base,
                                    self._localizer_params)
            if localization is None:
                rejected.append((detection, "cannot place on the floor"))
            elif reason := rejection_reason(localization, label):
                rejected.append((detection, reason))
            else:
                accepted.append((detection, localization))
        return FrameResult(accepted, rejected, upright)

    def _classify(self, image: np.ndarray, boxes: list[Box]) -> list[tuple[str | None, float]]:
        if self._classifier is None or not boxes:
            return [(None, 0.0)] * len(boxes)
        return self._classifier.classify([self._crop(image, box) for box in boxes])

    def _crop(self, image: np.ndarray, box: Box) -> np.ndarray:
        pad_x = (box.x2 - box.x1) * self._crop_padding
        pad_y = (box.y2 - box.y1) * self._crop_padding
        h, w = image.shape[:2]
        x1, y1 = max(0, round(box.x1 - pad_x)), max(0, round(box.y1 - pad_y))
        x2, y2 = min(w, round(box.x2 + pad_x)), min(h, round(box.y2 + pad_y))
        return image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
