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

# ML packages for the detector node (libgl1/libglib2.0-0 are OpenCV runtime libs). CPU-only PyTorch: the robot has no GPU, and the
# default CUDA wheels would add gigabytes. --break-system-packages is needed on
# Ubuntu 24.04 (Jazzy) and ignored by older pip.
COPY src/clean_sweep_perception/requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir -r /tmp/requirements.txt \
    # torch pulls in a new setuptools that needs a newer `packaging` than Ubuntu ships;
    # without this, colcon's Python package builds fail.
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir "packaging>=24.2"

COPY src/ src/
RUN . "/opt/ros/$ROS_DISTRO/setup.sh" && colcon build

COPY docker/ros/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
