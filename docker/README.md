# ROS 2 in Docker

Builds the packages in `src/` into a ROS 2 image, so they run the same on any machine
and on whichever distro the robot uses. Tested with Humble and Jazzy.

```bash
# from the repo root
docker compose -f docker/compose.yaml build                     # Humble (default)
ROS_DISTRO=jazzy docker compose -f docker/compose.yaml build    # or match the robot

docker compose -f docker/compose.yaml run --rm ros \
  ros2 launch clean_sweep_perception ground_truth_stub.launch.py

docker compose -f docker/compose.yaml run --rm ros bash         # interactive shell
```

Rebuild the image after changing code in `src/`. Only the source layer and `colcon build` re-run,
so rebuilds are quick.

## Networking

The container uses host networking, so on an **Ubuntu host** it can talk to the robot and other ROS
nodes like any native process. Set the same `ROS_DOMAIN_ID` as the robot:

```bash
ROS_DOMAIN_ID=<robot's id> docker compose -f docker/compose.yaml run --rm ros ros2 topic list
```

On **macOS**, Docker runs inside a VM, so DDS discovery with the robot doesn't work reliably.
Use a Mac for building and testing inside the container, and an Ubuntu machine to talk to the robot.

## Native ROS instead

If ROS 2 is installed natively and matches the robot's distro, Docker is optional:

```bash
colcon build && source install/setup.bash
ros2 launch clean_sweep_perception ground_truth_stub.launch.py
```
