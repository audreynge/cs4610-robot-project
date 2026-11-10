"""End-to-end frame processing with stand-in models on the synthetic camera."""

import numpy as np

from clean_sweep_perception.geometry import Box, Transform
from clean_sweep_perception.image_rotation import rotate_upright
from clean_sweep_perception.pipeline import LitterPerception
from synthetic_scene import CAMERA, HEIGHT, WIDTH, can_at, head_camera, project_box, render_depth


class MaskDetector:
    """'Detects' every white blob: returns the upright box of the white pixels in channel 0."""

    def detect(self, image):
        ys, xs = np.nonzero(image[:, :, 0])
        if not len(xs):
            return []
        return [(Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), 0.9)]


class RecordingClassifier:
    def __init__(self, label):
        self.label, self.crops = label, []

    def classify(self, crops):
        self.crops += crops
        return [(self.label, 0.8)] * len(crops)


def sideways_camera() -> Transform:
    """Head camera rolled 90 deg like Stretch's: its raw image needs a cw90 turn to be upright."""
    return head_camera(rotation="cw90")


def raw_frame(pose, objects, obstacles=()):
    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for low, high in objects:
        box = project_box(pose, low, high)
        color[int(box.y1):int(np.ceil(box.y2)), int(box.x1):int(np.ceil(box.x2))] = 255
    return color, render_depth(pose, [*objects, *obstacles])


def test_sideways_camera_frame_is_detected_upright_and_located():
    pose = sideways_camera()
    can = can_at(1.1, 0.1)
    color, depth = raw_frame(pose, [can])
    classifier = RecordingClassifier("can")

    result = LitterPerception(MaskDetector(), classifier, "cw90").process(color, depth, CAMERA, pose)

    [(detection, loc)] = result.accepted
    assert detection.label == "can"
    assert loc.depth_valid
    np.testing.assert_allclose(loc.position[:2], [1.1, 0.1], atol=0.02)
    # The classifier saw an upright, non-empty crop of the white object.
    [crop] = classifier.crops
    assert crop.size and crop[:, :, 0].mean() > 128
    assert result.upright_image.shape == rotate_upright(color, "cw90").shape


def test_detection_on_a_table_is_rejected_with_reason():
    pose = sideways_camera()
    cup = can_at(1.05, 0.0, base_z=0.45)
    table = ((0.9, -0.5, 0.0), (1.5, 0.5, 0.45))
    color, depth = raw_frame(pose, [cup], obstacles=[table])

    result = LitterPerception(MaskDetector(), None, "cw90").process(color, depth, CAMERA, pose)

    assert result.accepted == []
    [(detection, reason)] = result.rejected
    assert detection.label is None
    assert "not on the floor" in reason


def test_no_detections_gives_empty_result():
    pose = head_camera()
    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result = LitterPerception(MaskDetector(), None, "none").process(
        color, np.zeros((HEIGHT, WIDTH)), CAMERA, pose)
    assert result.accepted == [] and result.rejected == []


def test_upright_image_is_really_upright():
    """In the upright view, nearer floor objects appear lower and robot-left objects appear left."""
    pose = sideways_camera()
    detections = {}
    for name, can in {"near": can_at(0.9, 0.0), "far": can_at(1.5, 0.0), "left": can_at(1.2, 0.3)}.items():
        color, depth = raw_frame(pose, [can])
        [(detection, _)] = LitterPerception(MaskDetector(), None, "cw90").process(
            color, depth, CAMERA, pose).accepted
        detections[name] = detection.upright_box
    assert detections["near"].y1 > detections["far"].y1
    assert detections["left"].x2 < WIDTH and detections["left"].x1 < detections["near"].x1
