"""Thin wrappers around the trained Ultralytics models (detector and crop classifier).

Both accept a .pt file or an exported OpenVINO model directory. Images are
numpy arrays in BGR order (the OpenCV convention Ultralytics expects).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from clean_sweep_perception.geometry import Box

DEFAULT_CONFIDENCE = 0.35  # the robot's detection threshold; the evaluation numbers use it too


class YoloDetector:
    def __init__(self, model_path: str, confidence: float = DEFAULT_CONFIDENCE, imgsz: int = 640,
                 class_names: Iterable[str] | None = None):
        """`class_names` keeps only those classes; use it to run a stock COCO model
        (e.g. ["bottle", "cup"]) before the litter detector is trained."""
        from ultralytics import YOLO  # heavy import, only when a model is used

        self._model = YOLO(model_path, task="detect")
        _require_task(self._model, "detect", model_path)
        self._confidence = confidence
        self._imgsz = imgsz
        self._class_ids = None
        if class_names:
            names = self._model.names
            wanted = set(class_names)
            self._class_ids = [i for i, n in names.items() if n in wanted]
            missing = wanted - {names[i] for i in self._class_ids}
            if missing:
                raise ValueError(f"model {model_path} has no classes {sorted(missing)}")

    def detect(self, image_bgr: np.ndarray) -> list[tuple[Box, float]]:
        # agnostic_nms: one object seen as two classes (e.g. bottle and cup) must stay one box,
        # since every class means "litter" here.
        result = self._model.predict(image_bgr, conf=self._confidence, imgsz=self._imgsz,
                                     classes=self._class_ids, agnostic_nms=True, verbose=False)[0]
        boxes = result.boxes
        return [(Box(*map(float, xyxy)), float(conf))
                for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist())]


class YoloClassifier:
    def __init__(self, model_path: str, imgsz: int = 224):
        from ultralytics import YOLO

        self._model = YOLO(model_path, task="classify")
        _require_task(self._model, "classify", model_path)
        self._imgsz = imgsz

    def classify(self, crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops_bgr:
            return []
        results = self._model.predict(crops_bgr, imgsz=self._imgsz, verbose=False)
        return [(r.names[r.probs.top1], float(r.probs.top1conf)) for r in results]


def _require_task(model, task: str, path: str) -> None:
    if model.task != task:
        raise ValueError(f"{path} is a {model.task!r} model, expected a {task!r} model")
