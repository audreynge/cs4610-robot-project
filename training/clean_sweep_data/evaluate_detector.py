"""Evaluate a detector on a built litter dataset split at the deployment threshold.

Reports AP50, precision/recall at `--conf` (the robot's detector_confidence),
and how often clean background images trigger a false alarm. All of a model's
classes count as "litter", so a zero-shot YOLOE model can be compared directly:

    uv run evaluate-detector --model runs/detector/weights/best.pt
    uv run evaluate-detector --model yoloe-26s-seg.pt \
        --prompts "plastic bottle" "drink can" "paper cup" "crumpled paper" "trash"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from clean_sweep_data.detection_metrics import DetectionStats
from clean_sweep_data.records import image_size

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_labels(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                _, cx, cy, w, h = map(float, line.split()[:5])
                boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                              (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(boxes).reshape(-1, 4)


def load_model(path: str, prompts: list[str] | None):
    from ultralytics import YOLO, YOLOE

    if prompts:
        model = YOLOE(path)
        model.set_classes(prompts, model.get_text_pe(prompts))
        return model
    return YOLO(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", nargs="+", help="text prompts for a YOLOE (zero-shot) model")
    parser.add_argument("--data", type=Path, default=Path("data/detector"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=0.35, help="deployment confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. 0, mps, cpu")
    parser.add_argument("--limit", type=int, help="evaluate N images spread evenly over the split")
    args = parser.parse_args()

    model = load_model(args.model, args.prompts)
    images = sorted(p for p in (args.data / "images" / args.split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit and args.limit < len(images):  # spread evenly so every source is represented
        images = [images[i] for i in np.linspace(0, len(images) - 1, args.limit).round().astype(int)]

    by_source: dict[str, DetectionStats] = defaultdict(DetectionStats.empty)
    overall = DetectionStats.empty()
    for image_path in images:
        result = model.predict(str(image_path), conf=0.01, imgsz=args.imgsz, device=args.device,
                               verbose=False)[0]
        width, height = image_size(image_path)
        gt = read_labels(args.data / "labels" / args.split / f"{image_path.stem}.txt", width, height)
        pred = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        source = image_path.stem.split("__", 1)[0]
        for stats in (overall, by_source[source]):
            stats.add_image(pred, scores, gt, args.conf)

    name = args.model + (f" (zero-shot: {', '.join(args.prompts)})" if args.prompts else "")
    print(f"\n{name}\n{len(images)} {args.split} images, threshold {args.conf}\n")
    print(f"{'source':<16}{'AP50':>8}{'precision':>11}{'recall':>9}{'false alarms on clean images':>31}")
    for source, stats in [*sorted(by_source.items()), ("ALL", overall)]:
        precision, recall = stats.at_threshold(args.conf)
        alarms = (f"{stats.negative_images_with_detections}/{stats.negative_images}"
                  f" ({stats.negative_images_with_detections / stats.negative_images:.0%})"
                  if stats.negative_images else "-")
        ap = f"{stats.ap50():.3f}" if stats.num_gt else "-"
        recall_text = f"{recall:.3f}" if stats.num_gt else "-"
        print(f"{source:<16}{ap:>8}{precision:>11.3f}{recall_text:>9}{alarms:>31}")


if __name__ == "__main__":
    main()
