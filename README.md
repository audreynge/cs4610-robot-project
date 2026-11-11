# Clean Sweep

[![Tests](https://github.com/audreynge/cs4610-robot-project/actions/workflows/test.yml/badge.svg)](https://github.com/audreynge/cs4610-robot-project/actions/workflows/test.yml)

Autonomous litter pickup with the Hello Robot Stretch.

**Team:** Paarth, Weiyi, Audrey, Amy

## Overview

Clean Sweep enables Stretch to search a room, detect common litter (cups, plastic bottles, drink cans, crumpled paper), pick each item up, and drop it into a trash can, all without human help. As a stretch goal, the robot sorts items into recycling and landfill bins based on the detected class.

## Approach

- **Perception (Audrey):** A single-class YOLO "litter" detector, trained on public datasets (TACO, an indoor-floor waste dataset, Drinking Waste and indoor negatives), finds litter in the head camera feed. A crop classifier then labels each item as bottle, can, cup, paper or other. RealSense depth, with a floor-plane fallback for clear or shiny objects, gives each item's 3D position and size.
- **Manipulation (Paarth):** Stretch rotates so the object is beside it, lowers the lift, extends the arm, and opens the gripper to fit the object's estimated size. Grasps are verified using gripper position and effort, and retried on failure.
- **Navigation (Oscar):** The room is mapped with SLAM, bin locations are marked, and Nav2 moves the robot between items and bins.
- **Integration (Amy):** A state machine coordinates the full loop: search → approach → grasp → verify → deliver.

Development starts in simulation (a single object at a known location) and incrementally adds detection, navigation, and multiple objects before moving to the real robot.

## Success Metrics

| Metric                          | Target                                       |
| ------------------------------- | -------------------------------------------- |
| Detection accuracy              | ≥ 85% (and low false positives on non-trash) |
| First-try grasp success         | ≥ 70%                                        |
| Task completion                 | ≥ 4 of 5 items binned within 10 minutes      |
| Sorting accuracy (stretch goal) | Recycling vs. landfill                       |

## Getting Started

### Quick start

You only need [uv](https://docs.astral.sh/uv/getting-started/installation/), not ROS. From the repo root:

```bash
./run.sh --perception   # live litter detection on your laptop's webcam (press q to quit)
./run.sh --test         # run every test suite (GitHub also runs this on every push)
./run.sh --all          # tests, then the webcam demo
```

- **Wrong camera:** add `--camera 1`.
- **Camera permission:** on macOS, allow camera access for your terminal in System Settings → Privacy & Security → Camera.
- **Models:** the trained models are in [`src/clean_sweep_perception/models`](src/clean_sweep_perception/models/README.md), so it runs straight after cloning. To try other models: `DETECTOR=path/to/detector.pt CLASSIFIER=path/to/classifier.pt ./run.sh --perception`.

To add your subsystem (e.g. `--navigation`), see the comment at the top of [`run.sh`](run.sh).

### Packages

ROS 2 packages live in `src/` (build with `colcon build` from the repo root).

- [`src/clean_sweep_interfaces`](src/clean_sweep_interfaces): shared messages (`LitterObject`, `LitterObjectArray`, `DetectLitter`)
- [`src/clean_sweep_perception`](src/clean_sweep_perception/README.md): perception nodes, including a ground-truth stub for simulation
- [`src/clean_sweep_manipulation`](src/clean_sweep_manipulation/README.md): lift, arm and gripper control, litter pick with grasp check and retry, and bin drop
- [`training`](training/README.md): dataset download and build tools, plus GPU training for the perception models
- [`docker`](docker/README.md): ROS 2 container (Humble or Jazzy) for building and running the packages

## Demo

Final demo: **December 9**
