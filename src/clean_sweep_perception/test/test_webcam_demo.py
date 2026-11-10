import numpy as np
import pytest

pytest.importorskip("cv2")

from clean_sweep_perception.geometry import Box  # noqa: E402
from clean_sweep_perception.pipeline import LitterPerception  # noqa: E402
from clean_sweep_perception.sorting import BinRules  # noqa: E402
from clean_sweep_perception.webcam_demo import BIN_COLORS, annotate  # noqa: E402

RULES = BinRules({"bottle": "recycling", "can": "recycling", "cup": "landfill", "paper": "recycling",
                  "other": "landfill"}, fallback="landfill", min_confidence=0.6)


class FakeDetector:
    def detect(self, image):
        return [(Box(10, 40, 60, 90), 0.9), (Box(100, 40, 150, 90), 0.8)]


class FakeClassifier:
    def classify(self, crops):
        return [("bottle", 0.95), ("bottle", 0.4)]  # the second is too unsure to recycle


def test_boxes_are_colored_by_bin_and_unsure_labels_fall_back():
    frame = np.full((120, 200, 3), 255, np.uint8)
    out = annotate(frame, LitterPerception(FakeDetector(), FakeClassifier(), "none"), RULES)
    assert tuple(out[89, 10]) == BIN_COLORS["recycling"]  # bottom-left corner of each box outline
    assert tuple(out[89, 100]) == BIN_COLORS["landfill"]
    assert (frame == 255).all()  # input untouched
