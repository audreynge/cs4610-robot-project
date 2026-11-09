from pathlib import Path

import pytest

from clean_sweep_data.config import IGNORE, LITTER, CategoryMapping, load_config

CLASSES = ("bottle", "can", "cup", "paper", "other")
TRAINING_DIR = Path(__file__).resolve().parents[1]


def test_mapping_resolves_listed_then_default():
    mapping = CategoryMapping("s", {"Drink can": "can", "Cigarette": IGNORE}, "other", CLASSES)
    assert mapping.resolve("Drink can") == "can"
    assert mapping.resolve("Cigarette") == IGNORE
    assert mapping.resolve("Pizza box") == "other"


def test_unmapped_category_without_default_is_an_error():
    mapping = CategoryMapping("s", {"paper": "paper"}, None, CLASSES)
    with pytest.raises(KeyError, match="plastic"):
        mapping.resolve("plastic")


def test_rejects_unknown_target():
    with pytest.raises(ValueError, match="bottel"):
        CategoryMapping("s", {"x": "bottel"}, None, CLASSES)
    with pytest.raises(ValueError):
        CategoryMapping("s", {}, "trash", CLASSES)


def test_bundled_config_loads():
    config = load_config(TRAINING_DIR / "datasets.yaml")
    assert config.classes == CLASSES
    assert {s.name for s in config.detector_sources} >= {"taco", "coco_negatives"}
    taco = next(s for s in config.detector_sources if s.name == "taco")
    assert taco.mapping.resolve("Clear plastic bottle") == "bottle"
    assert taco.mapping.resolve("Unlabeled litter") == LITTER
    assert taco.mapping.resolve("Cigarette") == IGNORE
    assert taco.root == TRAINING_DIR / "data/raw/taco"


def test_classes_match_perception_labels():
    """The runtime label list and the training classes must not drift apart."""
    labels_file = TRAINING_DIR.parent / "src/clean_sweep_perception/clean_sweep_perception/labels.py"
    namespace: dict = {}
    exec(labels_file.read_text(), namespace)
    assert load_config(TRAINING_DIR / "datasets.yaml").classes == namespace["LITTER_LABELS"]


def test_rejects_classification_format_for_detector(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("""
classes: [bottle]
detector_sources:
  - {name: t, format: folders, root: x}
""")
    with pytest.raises(ValueError, match="folders"):
        load_config(path)
