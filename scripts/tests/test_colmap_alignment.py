import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colmap2nerf import rotmat


@pytest.mark.parametrize("source", [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])
@pytest.mark.parametrize("offset", [0., 1e-12, 1e-6, 0.01])
def test_antipodal_alignment_is_a_proper_rotation(source, offset):
    source = np.asarray(source)
    source /= np.linalg.norm(source)
    target = -source + offset * np.array([0.2, -0.3, 0.5])
    target /= np.linalg.norm(target)
    rotation = rotmat(source, target)
    np.testing.assert_allclose(rotation @ source, target, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)


def test_alignment_does_not_consume_random_state():
    before = np.random.get_state()
    source = np.array([1., 2., 3.])
    first = rotmat(source, -source)
    second = rotmat(source, -source)
    after = np.random.get_state()
    np.testing.assert_array_equal(first, second)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_general_and_parallel_alignment():
    rng = np.random.default_rng(42)
    for source, target in rng.normal(size=(100, 2, 3)):
        rotation = rotmat(source, target)
        np.testing.assert_allclose(rotation @ (source / np.linalg.norm(source)),
                                   target / np.linalg.norm(target), atol=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(rotmat([1., 2., 3.], [2., 4., 6.]), np.eye(3))


@pytest.mark.parametrize("offset", [1e-8, 1e-10, 1e-12, 1e-14])
def test_nearly_antipodal_general_directions(offset):
    rng = np.random.default_rng(29)
    for _ in range(100):
        source = rng.normal(size=3)
        source /= np.linalg.norm(source)
        target = -source + offset * rng.normal(size=3)
        target /= np.linalg.norm(target)
        rotation = rotmat(source, target)
        np.testing.assert_allclose(rotation @ source, target, atol=1e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-12)


def test_nearly_antipodal_axis_is_perpendicular():
    source = np.array([0.578317995089271, 0.718725786623855, 0.3859812692318722])
    target = np.array([-0.5783179950892747, -0.7187257866238511, -0.38598126923187387])
    rotation = rotmat(source, target)
    np.testing.assert_allclose(rotation @ source, target, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
