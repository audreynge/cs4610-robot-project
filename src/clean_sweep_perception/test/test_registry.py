import numpy as np
import pytest

from clean_sweep_perception.registry import LitterRegistry, Observation, RegistryParams, State


def obs(x, y, depth_valid=True, label="can", label_confidence=0.9, width=0.07):
    return Observation(np.array([x, y, 0.06]), width, 0.12, depth_valid, 0.8, label, label_confidence)


def test_repeated_sightings_confirm_one_object():
    registry = LitterRegistry(RegistryParams(confirm_sightings=3))
    for t in range(3):
        registry.update([obs(1.0 + 0.01 * t, 0.5)], now=t)
    [obj] = registry.objects()
    assert obj.id == 1
    assert obj.state == State.CONFIRMED
    assert obj.sightings == 3
    np.testing.assert_allclose(obj.position[:2], [1.01, 0.5], atol=1e-9)


def test_distant_observations_become_separate_objects():
    registry = LitterRegistry()
    registry.update([obs(1.0, 0.0), obs(2.0, 0.0)], now=0)
    assert [o.id for o in registry.objects()] == [1, 2]


def test_two_detections_in_one_frame_never_merge_into_one_object():
    registry = LitterRegistry(RegistryParams(match_radius=0.15))
    registry.update([obs(1.0, 0.0)], now=0)
    registry.update([obs(1.0, 0.05), obs(1.05, 0.0)], now=1)  # two cans side by side
    assert len(registry.objects()) == 2


def test_one_off_false_positive_is_forgotten():
    registry = LitterRegistry(RegistryParams(forget_after=5.0))
    registry.update([obs(3.0, 3.0)], now=0)
    registry.update([], now=6)
    assert registry.objects() == []


def test_confirmed_objects_are_kept_when_out_of_view():
    registry = LitterRegistry(RegistryParams(confirm_sightings=1, forget_after=5.0))
    registry.update([obs(1.0, 0.0)], now=0)
    registry.update([], now=100)
    assert len(registry.objects()) == 1


def test_depth_observations_outweigh_floor_plane_ones():
    registry = LitterRegistry(RegistryParams(depth_weight=1.0, floor_plane_weight=0.25))
    registry.update([obs(1.0, 0.0, depth_valid=True)], now=0)
    registry.update([obs(1.1, 0.0, depth_valid=False)], now=1)
    [obj] = registry.objects()
    assert obj.position[0] == pytest.approx(1.0 + 0.1 * 0.25 / 1.25)
    assert obj.depth_valid


def test_label_is_decided_by_accumulated_confidence():
    registry = LitterRegistry()
    registry.update([obs(1.0, 0.0, label="cup", label_confidence=0.5)], now=0)
    registry.update([obs(1.0, 0.0, label="can", label_confidence=0.9)], now=1)
    registry.update([obs(1.0, 0.0, label="can", label_confidence=0.8)], now=2)
    [obj] = registry.objects()
    assert obj.label == "can"
    assert obj.label_confidence == pytest.approx(0.85 * 2 / 3)  # mean confidence x vote share


def test_consistently_unsure_label_keeps_low_confidence():
    """Regression: three unsure 'bottle' votes must not look certain (or sorting trusts them)."""
    registry = LitterRegistry()
    for t in range(3):
        registry.update([obs(1.0, 0.0, label="bottle", label_confidence=0.3)], now=t)
    assert registry.objects()[0].label_confidence == pytest.approx(0.3)


def test_unknown_label_defaults_to_other():
    registry = LitterRegistry()
    registry.update([obs(1.0, 0.0, label=None, label_confidence=0.0)], now=0)
    assert registry.objects()[0].label == "other"


def test_picked_objects_no_longer_absorb_observations():
    registry = LitterRegistry()
    registry.update([obs(1.0, 0.0)], now=0)
    assert registry.set_state(1, State.PICKED)
    registry.update([obs(1.0, 0.0)], now=1)  # something new where the old one was
    assert [(o.id, o.state) for o in registry.objects()] == [(1, State.PICKED), (2, State.SEEN)]


def test_set_state_unknown_id():
    assert not LitterRegistry().set_state(42, State.TARGETED)


def test_refine_discards_old_far_away_estimate():
    registry = LitterRegistry()
    registry.update([obs(1.0, 0.0, width=0.2)], now=0)
    close_up = [obs(1.08, 0.02, width=0.07), obs(1.1, 0.0, width=0.07)]
    assert registry.refine(1, close_up, radius=0.25, since=5, now=6) == 2
    obj = registry.get(1)
    np.testing.assert_allclose(obj.position[:2], [1.09, 0.01])
    assert obj.width == pytest.approx(0.07)
    assert registry.refine(1, [], radius=0.25, since=5, now=6) == 0
    assert registry.refine(99, close_up, radius=0.25, since=5, now=6) == 0


def test_refine_absorbs_duplicate_created_during_the_close_look():
    """Regression: far estimate 0.2 m off -> close detections spawn a new object meanwhile."""
    registry = LitterRegistry(RegistryParams(match_radius=0.15, confirm_sightings=3))
    for t in range(3):
        registry.update([obs(1.0, 0.0)], now=t)  # far-away estimate
    close_up = []
    for t in range(10, 13):  # during the refine dwell, the real object is at 1.2
        registry.update([obs(1.2, 0.0)], now=t)
        close_up.append(obs(1.2, 0.0))
    assert len(registry.objects()) == 2  # the duplicate exists before refine
    assert registry.refine(1, close_up, radius=0.25, since=10, now=13) == 3
    [obj] = registry.objects()
    assert obj.id == 1 and obj.position[0] == pytest.approx(1.2)


def test_refine_ignores_detections_of_a_neighbouring_object():
    registry = LitterRegistry(RegistryParams(confirm_sightings=1))
    registry.update([obs(1.0, 0.0), obs(1.0, 0.2)], now=0)  # two items 20 cm apart
    seen = [obs(1.02, 0.0), obs(1.0, 0.19)]  # close look sees both
    assert registry.refine(1, seen, radius=0.25, since=5, now=6) == 1
    np.testing.assert_allclose(registry.get(1).position[:2], [1.02, 0.0])
    assert registry.get(2) is not None
