# clean_sweep_perception

Litter detection, 3D localization and tracking for Stretch 2. Messages and services
are defined in [`clean_sweep_interfaces`](../clean_sweep_interfaces).

```
head camera ──> litter_detector ──litter/detections──> litter_registry ──litter/objects──> state machine
                  (per frame)                           (ids, states,     <──services────
                                                         head control)
```

## Interface for the rest of the team

| Name | Type | What it does |
|---|---|---|
| `litter/objects` | topic, `LitterObjectArray` | Every tracked object with a stable `id`, `label`, `position` (in `map`), `width`/`height` (m), `state` |
| `litter/detect` | `DetectLitter` | Same list, on request |
| `litter/scan` | `ScanForLitter` | Sweeps the head across `scan_pans`, then returns the **confirmed** objects |
| `litter/set_state` | `SetLitterState` | Set `TARGETED` when going for an object, `FAILED` when giving up |
| `litter/refine` | `RefineLitter` | After driving up to an object: points the head at it and re-estimates it from close-range detections only. Grasp using the returned position. |
| `litter/verify_pickup` | `VerifyPickup` | After a grasp (arm out of view): looks where the object was. If it's gone, marks it `PICKED` and returns `picked_up: true`. |

Each object also has a `bin` (`recycling` or `landfill`) for sorting, set from
[`config/bins.yaml`](config/bins.yaml). Items with low label confidence go to landfill, so a misread never contaminates recycling.

States: `SEEN` (detected, not yet trusted) → `CONFIRMED` (seen 3+ times) → `TARGETED` → `PICKED` or `FAILED`.
Only act on `CONFIRMED` objects. `width` is the gripper aperture to open to (add a margin).

A typical loop for the state machine: `scan` → choose the nearest `CONFIRMED` object → `set_state TARGETED` → navigate next to it → `refine` → grasp → `verify_pickup` → deliver, or `set_state FAILED`.

## Running

```bash
ros2 launch clean_sweep_perception perception.launch.py \
    detector_model:=/path/to/detector_int8_openvino_model \
    classifier_model:=/path/to/classifier_int8_openvino_model
```

Before the litter models are trained, a stock COCO model finds bottles and cups:
`detector_model:=yolo26n.pt detector_classes:="['bottle', 'cup']"`.

Useful detector arguments: `detector_confidence` (default 0.35), `image_rotation` (default `cw90`),
`output_frame` (default `map`; use `odom` if no map is running), and the camera topic names.
View `litter/debug_image` in rqt_image_view: green boxes are accepted, red are rejected (the reason is shown).
Add `litter/markers` in RViz to see the tracked objects.

### Without the robot

`fake_head_camera` publishes any photo as if the head camera saw it (sideways raw image,
flat-floor depth, camera info, TF), so the whole stack runs on a laptop:

```bash
ros2 run clean_sweep_perception fake_head_camera --ros-args -p image:=/path/to/photo.jpg
```

### In simulation: ground-truth stub

`ground_truth_stub` publishes `litter/objects` and serves `litter/detect` from a YAML list of objects.
Use it in place of the whole stack while the sim has no realistic litter:

```bash
ros2 launch clean_sweep_perception ground_truth_stub.launch.py objects_file:=/path/to/objects.yaml
```

## Settings to check on the real robot

These depend on the robot and are set as parameters. They're tested in software, but not yet on hardware:

| Parameter (node) | Default | How to check |
|---|---|---|
| `image_rotation` (detector) | `cw90` | `litter/debug_image` should look upright |
| camera topics (detector) | `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info` | `ros2 topic list`; enable aligned depth in the RealSense launch if it's missing |
| `head_action` (registry) | `/stretch_controller/follow_joint_trajectory` | `ros2 action list` with stretch_driver running |
| `pan_zero_heading` (registry) | `0.0` | `refine` should center the object in the image |
| `scan_pans`, `scan_tilt` (registry) | `[-2.4 … 0.8]`, `-0.75` | The sweep should cover the floor around the robot |

## Validating on the robot

1. **Speed.** On the NUC, check the exported models are fast enough (target: about 2+ frames/s):
   ```bash
   ros2 run clean_sweep_perception benchmark_perception \
       --detector detector_int8_openvino_model --classifier classifier_int8_openvino_model
   ```
2. **Record a bag** of the head camera to tune offline without tying up the robot:
   ```bash
   ros2 bag record /camera/color/image_raw /camera/aligned_depth_to_color/image_raw \
       /camera/color/camera_info /tf /tf_static /joint_states
   # later: ros2 bag play <bag> --clock, and launch perception with use_sim_time:=true
   ```
3. **Score detection against ground truth** (the ≥85% target). Place litter, tape-measure each
   item's position in the `map` frame, write them in the sim stub's YAML format, run `litter/scan`, then:
   ```bash
   ros2 run clean_sweep_perception evaluate_perception --ros-args -p truth_file:=truth.yaml
   ```
   This prints recall (items found), precision (false positives), mean position error and label accuracy.
   Include some non-litter items (shoes, bags, cables) in the scene to check for false positives.

## Tests (no ROS needed)

The core logic (geometry, pipeline, registry, head angles) has no ROS dependency:

```bash
uv run --with pytest --with pyyaml --with numpy env PYTHONPATH=src/clean_sweep_perception pytest src/clean_sweep_perception/test
```
