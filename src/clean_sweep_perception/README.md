# clean_sweep_perception

Litter detection and 3D localization for Stretch 2. The interface is defined in
[`clean_sweep_interfaces`](../clean_sweep_interfaces):

| Name | Type | Description |
|---|---|---|
| `litter/objects` | `clean_sweep_interfaces/LitterObjectArray` (topic) | All known litter, published periodically |
| `litter/detect` | `clean_sweep_interfaces/DetectLitter` (service) | Same list, on request |

Each `LitterObject` carries an id, a label (`bottle/can/cup/paper/other`), a position
(normally in `map`), width/height in meters, and a state (`SEEN`, `CONFIRMED`,
`TARGETED`, `PICKED`, `FAILED`).

## Ground-truth stub (for simulation)

Publishes the objects listed in a YAML file through the same interface, so the rest of
the team can develop against it before the real detector is ready:

```bash
colcon build --packages-select clean_sweep_interfaces clean_sweep_perception
source install/setup.bash
ros2 launch clean_sweep_perception ground_truth_stub.launch.py objects_file:=/path/to/objects.yaml
```

The default file is [`config/sim_objects.yaml`](config/sim_objects.yaml). Set its positions to
where the objects are placed in the sim.

## Tests (no ROS needed)

The core logic has no ROS dependency, so it can be tested on any machine:

```bash
uv run --with pytest --with pyyaml env PYTHONPATH=src/clean_sweep_perception pytest src/clean_sweep_perception/test
```
