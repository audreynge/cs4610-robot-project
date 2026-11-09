# Clean Sweep

Autonomous litter pickup with the Hello Robot Stretch.

**Team:** Paarth, Weiyi, Audrey, Amy

## Overview

Clean Sweep enables Stretch to search a room, detect common litter (cups, plastic bottles, drink cans, crumpled paper), pick each item up, and drop it into a trash can, all without human help. As a stretch goal, the robot sorts items into recycling and landfill bins based on the detected class.

## Approach

- **Perception (Audrey):** A YOLO model fine-tuned on TACO and TrashNet (plus our own images) detects litter in the head camera feed. RealSense depth data is used to estimate each item's 3D position in the robot's base frame.
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

_Setup and run instructions coming soon._

## Demo

Final demo: **December 9**
