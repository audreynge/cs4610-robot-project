"""Measure detector + classifier speed on this machine (run it on the Stretch NUC).

    ros2 run clean_sweep_perception benchmark_perception \
        --detector detector_int8_openvino_model --classifier classifier_int8_openvino_model \
        --images /path/to/photos

No ROS needed. Uses random images if --images isn't given.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from clean_sweep_perception.models import YoloClassifier, YoloDetector


def load_images(directory: str | None, count: int) -> list[np.ndarray]:
    if directory:
        import cv2

        paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        images = [img for p in paths[:count] if (img := cv2.imread(str(p))) is not None]
        if images:
            return images
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(count)]


def percentile_ms(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier")
    parser.add_argument("--images", help="folder of test photos (default: random images)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    detector = YoloDetector(args.detector, confidence=0.25)
    classifier = YoloClassifier(args.classifier) if args.classifier else None
    images = load_images(args.images, max(args.runs, 1))

    detect_times, classify_times, crops_per_frame = [], [], []
    for i in range(args.warmup + args.runs):
        image = images[i % len(images)]
        start = time.perf_counter()
        boxes = detector.detect(image)
        detected = time.perf_counter()
        if classifier:
            crops = [image[int(b.y1):int(b.y2) + 1, int(b.x1):int(b.x2) + 1] for b, _ in boxes] or [image[:64, :64]]
            classifier.classify(crops)
        done = time.perf_counter()
        if i >= args.warmup:
            detect_times.append(detected - start)
            classify_times.append(done - detected)
            crops_per_frame.append(len(boxes))

    total = [d + c for d, c in zip(detect_times, classify_times)]
    print(f"{args.runs} frames ({'photos' if args.images else 'random images'})")
    print(f"detector    mean {np.mean(detect_times) * 1000:6.1f} ms   p95 {percentile_ms(detect_times, 95):6.1f} ms")
    if classifier:
        print(f"classifier  mean {np.mean(classify_times) * 1000:6.1f} ms   p95 {percentile_ms(classify_times, 95):6.1f} ms"
              f"   ({np.mean(crops_per_frame):.1f} crops/frame)")
    print(f"total       mean {np.mean(total) * 1000:6.1f} ms   -> up to {1 / np.mean(total):.1f} frames/s")


if __name__ == "__main__":
    main()
