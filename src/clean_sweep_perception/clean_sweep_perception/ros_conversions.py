"""Conversions between ROS message fields and numpy (no ROS import needed).

Kept free of cv_bridge, whose compiled numpy ABI often breaks when ML packages
upgrade numpy.
"""

from __future__ import annotations

import numpy as np

from clean_sweep_perception.geometry import Transform


def image_to_numpy(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """Color -> HxWx3 BGR uint8; depth -> HxW float32 meters (0 where there is no data)."""
    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        image = buffer.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
        return image[:, :, ::-1].copy() if encoding == "rgb8" else image.copy()
    if encoding in ("16UC1", "mono16"):  # RealSense depth: millimeters
        rows = buffer.reshape(height, step)[:, :width * 2]
        return rows.copy().view("<u2").reshape(height, width).astype(np.float32) / 1000.0
    if encoding == "32FC1":  # meters
        rows = buffer.reshape(height, step)[:, :width * 4]
        depth = rows.copy().view("<f4").reshape(height, width)
        return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"unsupported image encoding {encoding!r}")


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_from_tf(translation, rotation) -> Transform:
    """From geometry_msgs Vector3 + Quaternion-like objects (anything with x/y/z[/w])."""
    return Transform(
        quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w),
        np.array([translation.x, translation.y, translation.z]),
    )


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    w = (m[k, j] - m[j, k]) / s
    return (q[0], q[1], q[2], w)
