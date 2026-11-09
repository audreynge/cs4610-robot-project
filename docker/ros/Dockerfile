# ROS 2 workspace image for the Clean Sweep packages in src/.
# Build from the repo root (see docker/README.md):
#   docker compose -f docker/compose.yaml build
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

WORKDIR /ws

# Install dependencies declared in package.xml before copying all sources,
# so editing code doesn't invalidate this (slow) layer.
COPY src/clean_sweep_interfaces/package.xml src/clean_sweep_interfaces/package.xml
COPY src/clean_sweep_perception/package.xml src/clean_sweep_perception/package.xml
RUN apt-get update \
    && rosdep update --rosdistro "$ROS_DISTRO" \
    && rosdep install --from-paths src --ignore-src -y --rosdistro "$ROS_DISTRO" \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
RUN . "/opt/ros/$ROS_DISTRO/setup.sh" && colcon build

COPY docker/ros/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
