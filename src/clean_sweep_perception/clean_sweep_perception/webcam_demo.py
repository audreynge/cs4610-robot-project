"""Live litter detection on a laptop webcam (no ROS, no depth).

Runs the robot's detector and classifier on each webcam frame and draws every
detection with its label and bin. It checks what the models see; 3D position,
size checks and tracking need the robot's depth camera and are not run here.

    PYTHONPATH=src/clean_sweep_perception python -m clean_sweep_perception.webcam_demo \
        --detector detector.pt --classifier classifier.pt

Press q in the window to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from clean_sweep_perception.pipeline import LitterPerception
from clean_sweep_perception.sorting import BinRules

DEFAULT_BINS = Path(__file__).resolve().parents[1] / "config" / "bins.yaml"
BIN_COLORS = {"recycling": (230, 120, 0), "landfill": (60, 60, 60)}  # BGR
NO_LABEL_COLOR = (0, 200, 0)


def annotate(frame_bgr: np.ndarray, perception: LitterPerception, rules: BinRules) -> np.ndarray:
    """Copy of the frame with each detection boxed: 'label conf -> bin', colored by bin."""
    image = frame_bgr.copy()
    for box, confidence, label, label_confidence in perception.detect_and_label(frame_bgr):
        if label is None:
            text, color = f"litter {confidence:.2f}", NO_LABEL_COLOR
        else:
            bin_name = rules.bin_for(label, label_confidence)
            text, color = f"{label} {label_confidence:.2f} -> {bin_name}", BIN_COLORS.get(bin_name, NO_LABEL_COLOR)
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(image, p1, p2, color, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(th + 8, p1[1])
        cv2.rectangle(image, (p1[0], top - th - 8), (p1[0] + tw + 6, top), color, -1)
        cv2.putText(image, text, (p1[0] + 3, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return image


def main() -> None:
    from clean_sweep_perception.models import YoloClassifier, YoloDetector

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True, help=".pt file (OpenVINO INT8 is slow on Apple Silicon)")
    parser.add_argument("--classifier", help="optional; without it boxes say just 'litter'")
    parser.add_argument("--confidence", type=float, default=0.35, help="the robot's detector_confidence")
    parser.add_argument("--camera", type=int, default=0, help="try 1 if 0 opens an iPhone or the wrong camera")
    parser.add_argument("--bins", type=Path, default=DEFAULT_BINS)
    args = parser.parse_args()

    detector = YoloDetector(args.detector, args.confidence)
    classifier = YoloClassifier(args.classifier) if args.classifier else None
    perception = LitterPerception(detector, classifier, rotation="none")
    rules = BinRules.from_yaml(args.bins)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}; on macOS allow camera access for your "
                         "terminal app in System Settings > Privacy & Security > Camera")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("camera stopped sending frames")
            start = time.monotonic()
            image = annotate(frame, perception, rules)
            cv2.putText(image, f"{1000 * (time.monotonic() - start):.0f} ms/frame  (q to quit)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Clean Sweep litter detector", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
