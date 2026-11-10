import pytest

from clean_sweep_perception.evaluation import Detected, evaluate
from clean_sweep_perception.stub_scene import StubObject


def truth(label, x, y):
    return StubObject(label, (x, y, 0.0), 0.07, 0.12)


def test_scores_recall_precision_error_and_labels():
    report = evaluate(
        [truth("can", 1.0, 0.0), truth("bottle", 2.0, 0.0), truth("cup", 3.0, 1.0)],
        [Detected(1, "can", (1.03, 0.04, 0.0)), Detected(2, "cup", (2.0, 0.1, 0.0)),
         Detected(3, "other", (5.0, 5.0, 0.0))],
    )
    assert report.recall == pytest.approx(2 / 3)
    assert report.precision == pytest.approx(2 / 3)
    assert report.mean_error == pytest.approx((0.05 + 0.1) / 2)
    assert report.label_accuracy == pytest.approx(0.5)
    assert [t.label for t in report.missed] == ["cup"]
    assert [d.id for d in report.false_positives] == [3]
    assert "MISSED" in report.summary() and "FALSE +" in report.summary()


def test_matching_is_one_to_one_closest_first():
    # Two detections near one object: the closer one matches, the other is a false positive.
    report = evaluate([truth("can", 1.0, 0.0)],
                      [Detected(1, "can", (1.15, 0.0, 0.0)), Detected(2, "can", (1.02, 0.0, 0.0))])
    assert [d.id for _, d, _ in report.matches] == [2]
    assert [d.id for d in report.false_positives] == [1]


def test_nothing_to_find_and_nothing_found_is_perfect():
    report = evaluate([], [])
    assert report.recall == report.precision == 1.0
