#!/usr/bin/env bash
# One entry point for running and checking Clean Sweep on a laptop.
#
#   ./run.sh --perception        live litter detection on this laptop's webcam
#   ./run.sh --test              every test suite (no ROS or robot needed)
#   ./run.sh --all               --test, then --perception
#
# Other options are passed to the webcam demo, e.g. `./run.sh --perception --camera 1`.
# Override the models with DETECTOR=... CLASSIFIER=... ./run.sh --perception
#
# Adding a subsystem (e.g. --navigation): write a run_<name> function, add it to
# the case below and to `--all`.
set -euo pipefail
cd "$(dirname "$0")"

DETECTOR="${DETECTOR:-src/clean_sweep_perception/models/detector.pt}"
CLASSIFIER="${CLASSIFIER:-src/clean_sweep_perception/models/classifier.pt}"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

need_uv() {
  command -v uv >/dev/null || { echo "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
}

run_test() {
  echo "== perception tests"
  uv run --with pytest --with pyyaml --with numpy --with opencv-python-headless \
    env PYTHONPATH=src/clean_sweep_perception pytest -q src/clean_sweep_perception/test
  echo "== manipulation tests"
  uv run --with pytest --with numpy \
    env PYTHONPATH=src/clean_sweep_manipulation pytest -q src/clean_sweep_manipulation/test
  echo "== training tests"
  (cd training && uv run --group dev pytest -q)
}

run_perception() {
  for model in "$DETECTOR" "$CLASSIFIER"; do
    if [ ! -e "$model" ]; then
      echo "Model not found: $model" >&2
      echo "Set DETECTOR=... CLASSIFIER=... to a .pt file, or restore the bundled ones with git." >&2
      exit 1
    fi
  done
  echo "== perception webcam demo (press q in the window to quit)"
  uv run --with ultralytics --with pyyaml env PYTHONPATH=src/clean_sweep_perception \
    python -m clean_sweep_perception.webcam_demo --detector "$DETECTOR" --classifier "$CLASSIFIER" \
    ${demo_args[@]+"${demo_args[@]}"}
}

steps=()
demo_args=()
for arg in "$@"; do
  case "$arg" in
    --perception) steps+=(perception) ;;
    --test) steps+=(test) ;;
    --all) steps+=(test perception) ;;
    -h|--help) usage; exit 0 ;;
    *) demo_args+=("$arg") ;;
  esac
done
if [ ${#steps[@]} -eq 0 ]; then usage; exit 1; fi

need_uv
for step in "${steps[@]}"; do "run_$step"; done
