"""Single-class detection metrics at a fixed confidence threshold (no ML dependencies).

Boxes are (x1, y1, x2, y2) in pixels. Predictions carry a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def match(pred_boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray,
          iou_threshold: float = 0.5) -> np.ndarray:
    """True-positive flag per prediction: highest score first, each ground truth matched once."""
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp
    ious = iou_matrix(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-scores):
        candidates = np.where(~taken & (ious[i] >= iou_threshold))[0]
        if len(candidates):
            best = candidates[np.argmax(ious[i, candidates])]
            taken[best] = True
            tp[i] = True
    return tp


def average_precision(scores: np.ndarray, tp: np.ndarray, num_gt: int) -> float:
    """All-point interpolated AP (area under the precision-recall curve)."""
    if num_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp_sorted = tp[order]
    cum_tp = np.cumsum(tp_sorted)
    recall = cum_tp / num_gt
    precision = cum_tp / np.arange(1, len(tp_sorted) + 1)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # make it monotonically decreasing
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


@dataclass
class DetectionStats:
    """Accumulates predictions over a dataset."""

    scores: list[float]
    true_positive: list[bool]
    num_gt: int = 0
    negative_images: int = 0
    negative_images_with_detections: int = 0  # at the reporting threshold

    @classmethod
    def empty(cls) -> DetectionStats:
        return cls([], [])

    def add_image(self, pred_boxes, scores, gt_boxes, threshold: float, iou_threshold: float = 0.5):
        pred_boxes, scores, gt_boxes = (np.asarray(x, dtype=float).reshape(-1, n)
                                        if n else np.asarray(x, dtype=float)
                                        for x, n in ((pred_boxes, 4), (scores, 0), (gt_boxes, 4)))
        self.scores += scores.tolist()
        self.true_positive += match(pred_boxes, scores, gt_boxes, iou_threshold).tolist()
        self.num_gt += len(gt_boxes)
        if len(gt_boxes) == 0:
            self.negative_images += 1
            self.negative_images_with_detections += int(np.any(scores >= threshold))

    def at_threshold(self, threshold: float) -> tuple[float, float]:
        """(precision, recall) counting only predictions with score >= threshold."""
        scores, tp = np.array(self.scores), np.array(self.true_positive, dtype=bool)
        keep = scores >= threshold
        hits = int(tp[keep].sum())
        precision = hits / keep.sum() if keep.any() else 1.0
        recall = hits / self.num_gt if self.num_gt else float("nan")
        return precision, recall

    def ap50(self) -> float:
        return average_precision(np.array(self.scores), np.array(self.true_positive, dtype=bool), self.num_gt)
