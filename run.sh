#!/usr/bin/env bash
# One entry point for running and checking Clean Sweep.
#
#   ./run.sh --perception            live litter detection on this laptop's webcam
#   ./run.sh --test                  every test suite (no ROS or robot needed)
#   ./run.sh --all                   --test, then --perception
#   ./run.sh --manipulation [mode]   on the robot: pick (default), joints, or calibrate
#
# Options after --perception go to the webcam demo, e.g. `./run.sh --perception --camera 1`.
# Options after the manipulation mode are ROS parameters, e.g.
# `./run.sh --manipulation pick -p x:=0.0 -p y:=-0.55 -p width:=0.066 -p height:=0.12`.
# Override the models with DETECTOR=... CLASSIFIER=... ./run.sh --perception
#
# Adding a subsystem (e.g. --navigation): write a run_<name> function and add it to the case below.
set -euo pipefail
cd "$(dirname "$0")"

DETECTOR="${DETECTOR:-src/clean_sweep_perception/models/detector.pt}"
CLASSIFIER="${CLASSIFIER:-src/clean_sweep_perception/models/classifier.pt}"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

need_uv() {
  command -v uv >/dev/null || { echo "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
}

run_test() {
  need_uv
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
  need_uv
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

# --- robot helpers -------------------------------------------------------------------------

started_pids=()
stop_started() {
  # Each was started in its own process group, so this also stops the nodes a launch file started.
  for pid in ${started_pids[@]+"${started_pids[@]}"}; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap stop_started EXIT

node_running() { ros2 node list 2>/dev/null | grep -qx "/$1"; }
topic_exists() { ros2 topic list 2>/dev/null | grep -qx "$1"; }

# start_unless_running <node name> <log name> <command...>: start it in the background, stop it on exit.
start_unless_running() {
  local node=$1 log=/tmp/clean_sweep_$2.log
  shift 2
  if node_running "$node"; then
    echo "== $node is already running"
    return
  fi
  echo "== starting $node (log: $log)"
  setsid "$@" > "$log" 2>&1 &  # own process group (the robot is Linux, which has setsid)
  started_pids+=($!)
  for _ in $(seq 1 30); do
    node_running "$node" && return
    sleep 1
  done
  echo "$node didn't start within 30 s. End of $log:" >&2
  tail -n 20 "$log" >&2
  exit 1
}

# start_camera <name> <image topic> <launch file>: launch a stretch_core camera unless its
# image topic already exists. Only warns if it doesn't come up (perception then sees nothing).
start_camera() {
  local name=$1 topic=$2 launch=$3 log=/tmp/clean_sweep_$1.log
  topic_exists "$topic" && return
  echo "== starting the $name (log: $log)"
  setsid ros2 launch stretch_core "$launch" > "$log" 2>&1 &
  started_pids+=($!)
  for _ in $(seq 1 30); do
    topic_exists "$topic" && return
    sleep 1
  done
  echo "WARNING: still no $topic after 30 s. End of $log:" >&2
  tail -n 20 "$log" >&2
}

# Stretch 3 gripper camera (D405): topics as stretch_core's d405_basic launch names them.
WRIST_COLOR=/gripper_camera/color/image_rect_raw
WRIST_DEPTH=/gripper_camera/aligned_depth_to_color/image_raw
WRIST_INFO=/gripper_camera/color/camera_info

run_manipulation() {
  if ! command -v ros2 >/dev/null; then
    echo "--manipulation runs on the robot: it needs ROS 2 and stretch_driver." >&2
    echo "On a Mac, build and test in Docker instead (docker/README.md)." >&2
    exit 1
  fi
  local script args=(${manip_args[@]+"${manip_args[@]}"})
  case "$manip_mode" in
    pick) script=demo_pick ;;
    joints) script=demo_arm_gripper ;;
    calibrate) script=calibrate_gripper ;;
  esac

  echo "== building the workspace"
  colcon build
  set +u; source install/setup.bash; set -u  # ROS setup scripts read unset variables

  # broadcast_odom_tf: perception reports litter in odom, so base_link -> odom must exist.
  start_unless_running stretch_driver stretch_driver \
    ros2 launch stretch_core stretch_driver.launch.py mode:=position broadcast_odom_tf:=True

  if [ "$manip_mode" = pick ] && [ ${#args[@]} -eq 0 ]; then
    # No coordinates given: pick the nearest litter that perception has confirmed.
    start_camera camera /camera/color/image_raw d435i_high_resolution.launch.py
    # No map on the robot: track litter in odom (survives the base turning to line up).
    start_unless_running litter_registry perception \
      ros2 launch clean_sweep_perception perception.launch.py ${PERCEPTION_ARGS:-output_frame:=odom}
    args=(-p source:=perception)

    if [ "${CLEAN_SWEEP_DEX_WRIST:-0}" = 1 ] && ! lsusb -d 8086:0b5b >/dev/null 2>&1; then
      echo "== no D405 wrist camera on USB: skipping the wrist camera checks"
    elif [ "${CLEAN_SWEEP_DEX_WRIST:-0}" = 1 ]; then
      # Second detector on the wrist camera, in base_link: the top grasp centers with it
      # and checks the object left the floor. No classifier: the label isn't needed there.
      start_camera wrist_camera "$WRIST_COLOR" d405_basic.launch.py
      local models
      models=$(ros2 pkg prefix clean_sweep_perception)/share/clean_sweep_perception/models
      start_unless_running wrist/litter_detector wrist_detector \
        ros2 run clean_sweep_perception litter_detector --ros-args -r __ns:=/wrist \
          -p color_topic:="$WRIST_COLOR" -p depth_topic:="$WRIST_DEPTH" -p info_topic:="$WRIST_INFO" \
          -p output_frame:=base_link -p image_rotation:=none \
          -p detector_model:="$models/detector_int8_openvino_model"
    fi
  fi

  echo "== manipulation: $script"
  ros2 run clean_sweep_manipulation "$script" ${args[@]+--ros-args "${args[@]}"}
}

# --- arguments -----------------------------------------------------------------------------

steps=()
demo_args=()
manip_mode=pick
manip_args=()
current=""
for arg in "$@"; do
  case "$arg" in
    --perception) steps+=(perception); current=perception ;;
    --test) steps+=(test); current="" ;;
    --all) steps+=(test perception); current=perception ;;
    --manipulation) steps+=(manipulation); current=manipulation ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [ "$current" = manipulation ]; then
        case "$arg" in
          pick|joints|calibrate) manip_mode=$arg ;;
          *) manip_args+=("$arg") ;;
        esac
      else
        demo_args+=("$arg")
      fi ;;
  esac
done
if [ ${#steps[@]} -eq 0 ]; then usage; exit 1; fi

for step in "${steps[@]}"; do "run_$step"; done
