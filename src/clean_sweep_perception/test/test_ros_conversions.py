import numpy as np
import pytest

from clean_sweep_perception.ros_conversions import (
    image_to_numpy, matrix_to_quaternion, quaternion_to_matrix,
)
from synthetic_scene import head_camera


def test_rgb8_becomes_bgr_and_respects_row_padding():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)  # 1x2: red, green
    padded = np.concatenate([rgb.reshape(1, 6), np.zeros((1, 2), np.uint8)], axis=1)  # step = 8
    bgr = image_to_numpy("rgb8", 1, 2, 8, padded.tobytes())
    np.testing.assert_array_equal(bgr[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(bgr[0, 1], [0, 255, 0])


def test_16uc1_depth_is_millimeters():
    mm = np.array([[0, 1500]], dtype="<u2")
    depth = image_to_numpy("16UC1", 1, 2, 4, mm.tobytes())
    np.testing.assert_allclose(depth, [[0.0, 1.5]])


def test_32fc1_nan_becomes_zero():
    m = np.array([[np.nan, 2.0]], dtype="<f4")
    np.testing.assert_allclose(image_to_numpy("32FC1", 1, 2, 8, m.tobytes()), [[0.0, 2.0]])


def test_unsupported_encoding():
    with pytest.raises(ValueError):
        image_to_numpy("yuv422", 1, 1, 2, b"\x00\x00")


def test_quaternion_rotation_about_z():
    s = np.sqrt(0.5)
    rotation = quaternion_to_matrix(0, 0, s, s)  # +90 deg about z
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(quaternion_to_matrix(0, 0, 0, 1), np.eye(3))


def test_matrix_quaternion_round_trip():
    for rotation in (np.eye(3), head_camera().rotation, np.diag([1.0, -1.0, -1.0]),
                     np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])):
        np.testing.assert_allclose(quaternion_to_matrix(*matrix_to_quaternion(rotation)), rotation,
                                   atol=1e-12)
