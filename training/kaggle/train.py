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


def main() -> None:
    check_environment()
    run(sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.208", "openvino", "nncf")

    detector_data = find("data.yaml").parent
    classifier_data = next(p for p in sorted(INPUT.rglob("classifier")) if (p / "train").is_dir())
    configs = find("detector.yaml").parent
    code = find("clean_sweep_data", is_dir=True).parent

    # The input mount is read-only and the uploaded data.yaml has a placeholder path.
    data_yaml = Path("/kaggle/working/detector_data.yaml")
    lines = (detector_data / "data.yaml").read_text().splitlines()
    data_yaml.write_text("\n".join([f"path: {detector_data}", *lines[1:]]) + "\n")

    common = [f"project={RUNS}", "device=0", "workers=4", "exist_ok=True"]
    run("yolo", f"cfg={configs / 'detector.yaml'}", f"data={data_yaml}", *common)
    run("yolo", f"cfg={configs / 'classifier.yaml'}", f"data={classifier_data}", *common)

    detector = RUNS / "detector" / "weights" / "best.pt"
    classifier = RUNS / "classifier" / "weights" / "best.pt"
    run("yolo", "val", f"model={detector}", f"data={data_yaml}", "split=test", *common[:2],
        "name=detector_test")
    run("yolo", "val", f"model={classifier}", f"data={classifier_data}", "split=test", *common[:2],
        "name=classifier_test")
    run(sys.executable, "-m", "clean_sweep_data.evaluate_detector", "--model", detector,
        "--data", detector_data, "--device", "0",
        env={**os.environ, "PYTHONPATH": str(code)})

    run("yolo", "export", f"model={detector}", "format=openvino", "int8=True", f"data={data_yaml}")
    run("yolo", "export", f"model={classifier}", "format=openvino", "int8=True", f"data={classifier_data}")


if __name__ == "__main__":
    main()
