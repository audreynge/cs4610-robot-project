"""Annotated upright image for RViz / rqt_image_view: green = accepted, red = rejected (with reason)."""

from __future__ import annotations

import cv2
import numpy as np

from clean_sweep_perception.pipeline import FrameResult

GREEN, RED = (0, 200, 0), (0, 0, 230)


def draw_debug(result: FrameResult) -> np.ndarray:
    image = result.upright_image.copy()
    for detection, loc in result.accepted:
        label = detection.label or "litter"
        text = (f"{label} {detection.confidence:.2f} | {loc.width * 100:.0f}cm"
                f"{'' if loc.depth_valid else ' (no depth)'}")
        _draw(image, detection.upright_box, text, GREEN)
    for detection, reason in result.rejected:
        _draw(image, detection.upright_box, f"{detection.confidence:.2f} {reason}", RED)
    return image


def _draw(image: np.ndarray, box, text: str, color) -> None:
    p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, text, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                cv2.LINE_AA)
