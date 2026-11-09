import numpy as np
import pytest

from clean_sweep_perception.geometry import Box
from clean_sweep_perception.image_rotation import ROTATIONS, box_to_raw, rotate_upright


def bounding_box(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    return Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


@pytest.mark.parametrize("rotation", sorted(ROTATIONS))
def test_box_found_in_upright_image_maps_back_to_raw(rotation):
    raw = np.zeros((48, 64), dtype=np.uint8)
    raw[5:12, 40:58] = 1  # an off-center, non-square object
    upright = rotate_upright(raw, rotation)
    assert box_to_raw(bounding_box(upright), rotation, raw.shape) == bounding_box(raw)


def test_cw90_turns_clockwise():
    raw = np.array([[1, 2],
                    [3, 4]])
    np.testing.assert_array_equal(rotate_upright(raw, "cw90"), [[3, 1],
                                                                [4, 2]])


def test_unknown_rotation_is_an_error():
    with pytest.raises(ValueError, match="sideways"):
        rotate_upright(np.zeros((2, 2)), "sideways")
