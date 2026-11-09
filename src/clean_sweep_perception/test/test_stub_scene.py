from pathlib import Path

import pytest

from clean_sweep_perception.stub_scene import StubObject, load_stub_scene

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def write(tmp_path, text):
    path = tmp_path / "scene.yaml"
    path.write_text(text)
    return path


def test_bundled_sim_config_is_valid():
    scene = load_stub_scene(PACKAGE_DIR / "config" / "sim_objects.yaml")
    assert scene.frame_id == "map"
    assert len(scene.objects) >= 1


def test_loads_objects(tmp_path):
    path = write(tmp_path, """
frame_id: odom
objects:
  - {label: bottle, position: [1, 2, 0.1], width: 0.07, height: 0.2}
""")
    scene = load_stub_scene(path)
    assert scene.frame_id == "odom"
    assert scene.objects == (StubObject("bottle", (1.0, 2.0, 0.1), 0.07, 0.2),)


def test_empty_file_gives_empty_scene_in_map_frame(tmp_path):
    scene = load_stub_scene(write(tmp_path, ""))
    assert scene.frame_id == "map"
    assert scene.objects == ()


@pytest.mark.parametrize("obj, message", [
    ("{label: shoe, position: [0, 0, 0], width: 0.1, height: 0.1}", "label"),
    ("{label: can, position: [0, 0], width: 0.1, height: 0.1}", "position"),
    ("{label: can, position: [0, 0, 0], width: 0, height: 0.1}", "positive"),
])
def test_rejects_invalid_objects(tmp_path, obj, message):
    path = write(tmp_path, f"objects:\n  - {obj}\n")
    with pytest.raises(ValueError, match=message):
        load_stub_scene(path)
