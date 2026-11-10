import numpy as np
import pytest

from clean_sweep_perception.fake_camera import fake_frame
from clean_sweep_perception.image_rotation import rotate_upright


def test_raw_frame_is_sideways_and_turns_back_upright():
    upright = np.random.default_rng(0).integers(0, 255, (480, 360, 3), dtype=np.uint8)
    raw, depth, camera, pose = fake_frame(upright, "cw90")
    assert raw.shape == (360, 480, 3)
    np.testing.assert_array_equal(rotate_upright(raw, "cw90"), upright)
    assert depth.shape == raw.shape[:2]


def test_floor_depth_along_optical_axis():
    upright = np.zeros((480, 640, 3), dtype=np.uint8)
    _, depth, camera, _ = fake_frame(upright, "none", height=1.2, pitch_deg=45.0)
    # The optical axis hits the floor 1.2 / sin(45 deg) away.
    assert depth[240, 320] == pytest.approx(1.2 / np.sin(np.radians(45)), rel=0.01)
    assert depth[0].min() > depth[-1].max()  # top of the image sees farther floor
