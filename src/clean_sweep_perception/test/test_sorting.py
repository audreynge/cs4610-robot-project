from pathlib import Path

import pytest

from clean_sweep_perception.sorting import BinRules

CONFIG = Path(__file__).resolve().parents[1] / "config" / "bins.yaml"


def test_bundled_rules_sort_confident_labels():
    rules = BinRules.from_yaml(CONFIG)
    assert rules.bin_for("can", 0.9) == "recycling"
    assert rules.bin_for("other", 0.9) == "landfill"


def test_unsure_labels_go_to_the_fallback_bin():
    rules = BinRules.from_yaml(CONFIG)
    assert rules.bin_for("can", 0.3) == rules.fallback == "landfill"


def test_every_label_needs_a_bin(tmp_path):
    path = tmp_path / "bins.yaml"
    path.write_text("fallback: landfill\nbins: {can: recycling}\n")
    with pytest.raises(ValueError, match="bottle"):
        BinRules.from_yaml(path)
