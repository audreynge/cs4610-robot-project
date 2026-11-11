# Agent guide: Clean Sweep

Instructions for AI coding agents (and people) working in this repo. Read the root `README.md` for
what the project is. This file covers how to build, run and check changes.

## Where ROS runs

- **Dev machines (Macs) have no ROS installed.** For anything that needs ROS 2, use the Docker
  setup in `docker/`. That means building the packages, running nodes or launch files, and
  checking topics, services or parameters. Don't install ROS on the host, and don't skip a ROS
  check because ROS is missing; run it in the container instead.
- **The robot (Stretch 2) runs ROS natively.** Docker isn't used there.
- **The robot's ROS distro isn't confirmed** (Humble or Jazzy). Code must work on both, and
  `ROS_DISTRO=jazzy` builds the Jazzy image.

## Commands (from the repo root)

```bash
./run.sh --test          # every test suite, no ROS needed (CI runs this on every push)
./run.sh --perception    # live webcam demo of the models, no ROS or Docker
./run.sh --manipulation  # robot only: builds, starts stretch_driver (and perception for a pick), runs a demo

# ROS 2 in Docker
docker compose -f docker/compose.yaml build      # REQUIRED after any change in src/: the image copies src/ at build time
docker compose -f docker/compose.yaml run --rm -T ros <command>
docker compose -f docker/compose.yaml run --rm ros bash            # interactive shell
ROS_DISTRO=jazzy docker compose -f docker/compose.yaml build        # Jazzy instead of Humble
```

Run the full perception stack without the robot. `fake_head_camera` publishes a photo as the
head camera's color, depth, camera info and TF:

```bash
docker compose -f docker/compose.yaml run --rm -T -v /path/to/photos:/photos ros bash -c '
  ros2 run clean_sweep_perception fake_head_camera --ros-args -p image:=/photos/photo.jpg &
  ros2 launch clean_sweep_perception perception.launch.py &
  sleep 60; ros2 topic echo --once /litter/objects'
```

## Gotchas

- **Rebuild the image before testing ROS changes.** Without `docker compose ... build`, the
  container runs the old code.
- **INT8 models are very slow in Docker on a Mac.** Docker there is ARM, and OpenVINO INT8 is
  built for Intel. The launch files default to the INT8 models (for the robot's Intel NUC), at
  about 6 s/frame on a Mac. For speed in Docker, pass the `.pt` models:
  `detector_model:=/ws/install/clean_sweep_perception/share/clean_sweep_perception/models/detector.pt`
  (and `classifier.pt`).
- **`ros2 param get` on `litter_detector` can say "Node not found"** while the node is busy on a
  slow frame. Retry it.
- **Don't run two copies of a node at once.** Both answer to the same name and you'll read the
  wrong one's state. Use a fresh container per check.
- **macOS Docker can't reach the robot over DDS, and can't use the webcam.** Use it for building
  and testing only.

## Checking changes

- **Core logic is ROS-free** (`src/clean_sweep_perception/clean_sweep_perception/*.py`, apart from
  the `*_node.py` files and `head_controller.py`). It's covered by `./run.sh --test`. Add or update
  tests there for any behaviour change.
- **The unit tests don't cover ROS wiring.** After changing launch files, `setup.py`, node
  parameters or messages in `src/clean_sweep_interfaces`, rebuild the image. Then run the affected
  launch file or node in Docker, with `fake_head_camera` if it needs camera input.
- **Keep each default in one place.** Detector node parameters live in `PARAMETER_DEFAULTS`
  (`litter_detector_node.py`), and `litter_detector.launch.py` builds its arguments from it. The
  registry's tracking defaults come from `RegistryParams`. The detection threshold is
  `models.DEFAULT_CONFIDENCE`.

## Models and training

- **The trained models are committed** in `src/clean_sweep_perception/models/`; see its README.
  Keep the `_openvino_model` folder suffix, because Ultralytics needs it to load the folder.
- **Training happens on Kaggle GPUs** (`training/kaggle/push.sh`, see `training/README.md`). A Mac
  can train with `device=mps`, but slowly. Docker and VMs on a Mac can't use its GPU.
- **Never commit `training/data/` or `training/runs/`.** They're gitignored: downloads, built
  datasets and training outputs.
