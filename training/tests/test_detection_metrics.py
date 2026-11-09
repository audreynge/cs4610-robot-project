import numpy as np
import pytest

from clean_sweep_data.detection_metrics import DetectionStats, average_precision, iou_matrix, match


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_each_ground_truth_matches_only_the_best_scoring_prediction():
    gt = np.array([[0, 0, 10, 10]])
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10]])
    assert match(preds, np.array([0.6, 0.9]), gt).tolist() == [False, True]


def test_perfect_detector_has_ap_one():
    assert average_precision(np.array([0.9, 0.8]), np.array([True, True]), num_gt=2) == pytest.approx(1.0)


def test_ap_with_a_false_positive_ranked_first():
    # FP at 0.9 then TP at 0.8 -> precision at full recall is 0.5.
    assert average_precision(np.array([0.9, 0.8]), np.array([False, True]), num_gt=1) == pytest.approx(0.5)


def test_stats_threshold_and_false_alarms():
    stats = DetectionStats.empty()
    stats.add_image([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.2], [[0, 0, 10, 10]], threshold=0.35)
    stats.add_image([[5, 5, 9, 9]], [0.5], [], threshold=0.35)  # clean image with a false alarm
    stats.add_image([], [], [], threshold=0.35)  # clean image, nothing detected
    precision, recall = stats.at_threshold(0.35)
    assert (precision, recall) == (0.5, 1.0)  # the 0.9 TP and the 0.5 false alarm
    assert (stats.negative_images, stats.negative_images_with_detections) == (2, 1)
