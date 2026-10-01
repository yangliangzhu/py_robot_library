"""Tests for the rotation and transform helpers in :mod:`tools.geometry`."""

from __future__ import annotations

import numpy as np
import pytest

from tools.geometry import (
    nervCartToAffine,
    nervRpy,
    quat2rot,
    rot2quat,
    rot2rpy,
    rot_x,
    rot_y,
    rot_z,
    rpy2rot,
)


def _is_rotation(matrix: np.ndarray) -> bool:
    """Whether ``matrix`` is a proper 3x3 rotation matrix."""
    if matrix.shape != (3, 3):
        return False
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12):
        return False
    return np.linalg.det(matrix) == pytest.approx(1.0)


class TestAxisRotations:
    """The three elementary rotations."""

    @pytest.mark.parametrize("builder", (rot_x, rot_y, rot_z))
    @pytest.mark.parametrize("theta", (-np.pi, -1.2, 0.0, 0.4, np.pi / 3))
    def test_are_proper_rotations(self, builder, theta):
        assert _is_rotation(builder(theta))

    def test_zero_angle_is_the_identity(self):
        for builder in (rot_x, rot_y, rot_z):
            assert np.allclose(builder(0.0), np.eye(3))

    def test_right_handed_axis_convention(self):
        assert np.allclose(rot_x(np.pi / 2) @ [0, 1, 0], [0, 0, 1])
        assert np.allclose(rot_y(np.pi / 2) @ [0, 0, 1], [1, 0, 0])
        assert np.allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0])

    def test_composition_order_is_reversed(self):
        # Rz(a) @ Rz(b) == Rz(a + b) because they share an axis.
        assert np.allclose(rot_z(0.3) @ rot_z(0.4), rot_z(0.7))


class TestRpy:
    """RPY angles and rotation matrices."""

    @pytest.mark.parametrize(
        "rpy",
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.7],
            [-1.0, 0.5, 2.0],
            [0.1, 1.2, -0.4],  # near, but not at, gimbal lock
        ],
    )
    def test_round_trip(self, rpy):
        recovered = rot2rpy(rpy2rot(rpy))
        assert np.allclose(recovered, rpy, atol=1e-9)

    def test_round_trip_over_random_angles(self, rng):
        for _ in range(50):
            rpy = rng.uniform(-1.4, 1.4, 3)
            assert np.allclose(rot2rpy(rpy2rot(rpy)), rpy, atol=1e-9)

    def test_gimbal_lock_is_handled(self):
        # pitch = +pi/2 is singular: yaw is pinned to zero and roll absorbed.
        rot = rpy2rot([0.6, np.pi / 2, 0.0])
        roll, pitch, yaw = rot2rpy(rot)

        assert pitch == pytest.approx(np.pi / 2)
        assert yaw == pytest.approx(0.0)
        assert np.allclose(rpy2rot([roll, pitch, yaw]), rot, atol=1e-9)

    def test_matches_the_explicit_zyx_composition(self):
        rpy = [0.2, 0.3, 0.4]
        assert np.allclose(rpy2rot(rpy), rot_z(0.4) @ rot_y(0.3) @ rot_x(0.2))


class TestQuaternions:
    """Quaternion conversions."""

    def test_identity_rotates_to_the_unit_quaternion(self):
        assert np.allclose(rot2quat(np.eye(3)), [1, 0, 0, 0])

    @pytest.mark.parametrize(
        "rot",
        [
            np.eye(3),
            rot_x(0.4),
            rot_y(-1.1),
            rot_z(2.7),
            rot_z(0.3) @ rot_y(0.5) @ rot_x(-0.2),
        ],
    )
    def test_round_trip_through_the_matrix(self, rot):
        quat = rot2quat(rot)
        assert np.allclose(quat2rot(quat), rot, atol=1e-12)

    def test_round_trip_over_random_rotations(self, rng):
        for _ in range(50):
            rot = rot_z(rng.uniform(-np.pi, np.pi)) @ rot_y(rng.uniform(-1.5, 1.5))
            assert np.allclose(quat2rot(rot2quat(rot)), rot, atol=1e-12)

    def test_quaternion_is_normalised_on_input(self):
        unit = quat2rot([1, 0, 0, 0])
        scaled = quat2rot([10, 0, 0, 0])
        assert np.allclose(unit, scaled)

    def test_rejects_a_wrong_length_quaternion(self):
        with pytest.raises(ValueError, match="4 elements"):
            quat2rot([1, 0, 0])

    def test_rejects_a_wrong_shape_matrix(self):
        with pytest.raises(ValueError, match="3x3"):
            rot2quat(np.eye(4))

    def test_is_normalised(self):
        quat = rot2quat(rot_z(0.9))
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-12)


class TestNervPoseHelpers:
    """The nerv-format pose conveniences."""

    def test_nervRpy_ignores_the_translation(self):
        assert np.allclose(nervRpy([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
                           rpy2rot(np.radians([10.0, 20.0, 30.0])))

    def test_nervRpy_accepts_radians(self):
        assert np.allclose(nervRpy([0, 0, 0, 0.1, 0.2, 0.3], rad=True),
                           rpy2rot([0.1, 0.2, 0.3]))

    def test_cart_to_affine_converts_millimetres_to_metres(self):
        pose = nervCartToAffine([1000.0, -500.0, 250.0, 0.0, 0.0, 0.0])
        assert pose.shape == (4, 4)
        assert np.allclose(pose[:3, 3], [1.0, -0.5, 0.25])
        assert np.allclose(pose[3, :], [0, 0, 0, 1])

    def test_cart_to_affine_applies_orientation_in_degrees(self):
        pose = nervCartToAffine([0, 0, 0, 0, 0, 90.0])
        assert np.allclose(pose[:3, :3], rot_z(np.pi / 2), atol=1e-12)

    def test_cart_to_affine_radians_flag(self):
        pose = nervCartToAffine([0, 0, 0, 0, 0, np.pi / 2], rad=True)
        assert np.allclose(pose[:3, :3], rot_z(np.pi / 2), atol=1e-12)
