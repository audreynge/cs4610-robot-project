"""Train, evaluate and export the litter detector and classifier on a Kaggle GPU.

Runs as a Kaggle script kernel; push it with `kaggle/push.sh` (see training/README.md).
Inputs are two private Kaggle datasets: the built data (detector/, classifier/) and
this repo's code (clean_sweep_data/, configs/). Everything lands in /kaggle/working/runs,
which `kaggle kernels output` downloads.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

INPUT = Path("/kaggle/input")
RUNS = Path("/kaggle/working/runs")


def find(name: str, is_dir: bool = False) -> Path:
    """Locate a file or folder in the attached datasets (Kaggle's mount layout varies)."""
    for path in sorted(INPUT.rglob(name)):
        if path.is_dir() == is_dir:
            return path
    raise FileNotFoundError(f"{name} not found under {INPUT}")


def run(*args, env=None) -> None:
    print("\n+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, env=env)


def check_environment() -> None:
    """Kaggle silently disables GPU and internet on accounts without a verified phone number."""
    hint = "Verify your phone number in Kaggle settings, then push again."
    if shutil.which("nvidia-smi") is None:
        sys.exit(f"No GPU attached. {hint}")
    try:
        urllib.request.urlopen("https://pypi.org", timeout=15)
    except OSError:
        sys.exit(f"No internet access. {hint}")


# One detector per entry: (name, sources to train on, sources whose val images pick best.pt).
# None means every source. Validation leaves out indoor_waste: its low-angle images are easy,
# so they dominate the score that picks best.pt, while TACO is closer to Stretch's view.
DETECTOR_RUNS = [
    ("detector_all", None, ["taco", "coco_negatives"]),
    ("detector_no_indoor", ["taco", "coco_negatives"], ["taco", "coco_negatives"]),
]
TRAIN_CLASSIFIER = False  # the current classifier is already trained on all sources


def image_list(detector_data: Path, split: str, sources: list[str] | None) -> list[str]:
    """Images of one split, limited to the given sources (file names start with '<source>__')."""
    images = sorted((detector_data / "images" / split).iterdir())
    return [str(p) for p in images if sources is None or p.name.split("__", 1)[0] in sources]


def write_data_yaml(name: str, detector_data: Path, train_sources, val_sources) -> Path:
    """Ultralytics data file whose train/val are image lists (the input mount is read-only)."""
    out = Path("/kaggle/working") / name
    out.mkdir(parents=True, exist_ok=True)
    for split, sources in (("train", train_sources), ("val", val_sources)):
        (out / f"{split}.txt").write_text("\n".join(image_list(detector_data, split, sources)) + "\n")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(f"path: {detector_data}\ntrain: {out / 'train.txt'}\nval: {out / 'val.txt'}\n"
                         "test: images/test\nnames:\n  0: litter\n")
    return data_yaml


def main() -> None:
    check_environment()
    run(sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.208", "openvino", "nncf")

    detector_data = find("data.yaml").parent
    classifier_data = next(p for p in sorted(INPUT.rglob("classifier")) if (p / "train").is_dir())
    configs = find("detector.yaml").parent
    code = find("clean_sweep_data", is_dir=True).parent
    common = [f"project={RUNS}", "device=0", "workers=4", "exist_ok=True"]

    for name, train_sources, val_sources in DETECTOR_RUNS:
        data_yaml = write_data_yaml(name, detector_data, train_sources, val_sources)
        # Kaggle gives the data loader only 2 CPU cores, which limit training speed. Caching the
        # resized images in RAM (about 7 GB of Kaggle's ~30) skips decoding every JPEG each epoch:
        # 1.4x faster image loading, measured on this dataset.
        run("yolo", f"cfg={configs / 'detector.yaml'}", f"data={data_yaml}", f"name={name}", "cache=ram",
            *common)
        detector = RUNS / name / "weights" / "best.pt"
        run(sys.executable, "-m", "clean_sweep_data.evaluate_detector", "--model", detector,
            "--data", detector_data, "--device", "0",
            env={**os.environ, "PYTHONPATH": str(code)})
        run("yolo", "export", f"model={detector}", "format=openvino", "int8=True", f"data={data_yaml}")

    if TRAIN_CLASSIFIER:
        run("yolo", f"cfg={configs / 'classifier.yaml'}", f"data={classifier_data}", *common)
        classifier = RUNS / "classifier" / "weights" / "best.pt"
        run("yolo", "val", f"model={classifier}", f"data={classifier_data}", "split=test", *common[:2],
            "name=classifier_test")
        run("yolo", "export", f"model={classifier}", "format=openvino", "int8=True",
            f"data={classifier_data}")


if __name__ == "__main__":
    main()
