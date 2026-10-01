"""Tests for forward kinematics, the Jacobian and the manipulability measure.

The Jacobian is checked against a numerical derivative of the forward
kinematics rather than against a stored reference value, so the test verifies the
analytic implementation instead of freezing today's output.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ROBOTS, sample_positions


def _skew_to_vector(skew: np.ndarray) -> np.ndarray:
    """Extract the axis vector from a 3x3 skew-symmetric matrix."""
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def _numeric_jacobian(robot, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return d(pose)/dq as a 6xn matrix, in the same layout as ``jacobian``."""
    n = robot.num_dof
    base_t = robot.fk(q)
    base_rot = base_t[:3, :3]
    out = np.zeros((6, n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = eps
        moved = robot.fk(q + delta)
        out[:3, i] = (moved[:3, 3] - base_t[:3, 3]) / eps
        # dR/dq @ R^T is the skew matrix of the angular velocity.
        out[3:, i] = _skew_to_vector(((moved[:3, :3] - base_rot) / eps) @ base_rot.T)
    return out


class TestForwardKinematics:
    """Properties every FK result must satisfy."""

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_returns_a_homogeneous_transform(self, backend, make_robot, rng, name, dof):
        robot = make_robot(name, backend)
        assert robot.num_dof == dof

        pose = robot.fk(sample_positions(robot, rng))
        assert pose.shape == (4, 4)
        assert np.allclose(pose[3, :], [0, 0, 0, 1])

        rot = pose[:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(rot) == pytest.approx(1.0)

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_zero_pose_is_reproducible(self, backend, make_robot, name, dof):
        robot = make_robot(name, backend)
        first = robot.fk(np.zeros(dof))
        second = robot.fk(np.zeros(dof))
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_backends_agree(self, make_robot, rng, name, dof):
        numpy_robot = make_robot(name, "numpy")
        casadi_robot = make_robot(name, "casadi")
        q = sample_positions(numpy_robot, rng)

        assert np.allclose(
            numpy_robot.fk(q), casadi_robot.fk(q), atol=1e-12
        )

    def test_moving_a_joint_moves_the_tool(self, make_robot):
        robot = make_robot("sr5", "numpy")
        q = np.zeros(robot.num_dof)
        q[0] = 0.4
        assert not np.allclose(robot.fk(q), robot.fk(np.zeros(robot.num_dof)))


class TestJacobian:
    """The analytic Jacobian against a numerical derivative of the FK."""

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_matches_finite_difference(self, backend, make_robot, rng, name, dof):
        robot = make_robot(name, backend)
        q = sample_positions(robot, rng)

        analytic = robot.jacobian(q)
        numeric = _numeric_jacobian(robot, q)

        assert analytic.shape == (6, dof)
        assert np.allclose(analytic, numeric, atol=1e-4)

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_backends_agree(self, make_robot, rng, name, dof):
        numpy_robot = make_robot(name, "numpy")
        casadi_robot = make_robot(name, "casadi")
        q = sample_positions(numpy_robot, rng)

        assert np.allclose(
            numpy_robot.jacobian(q), casadi_robot.jacobian(q), atol=1e-12
        )

    def test_columns_are_the_joint_screws(self, make_robot, rng):
        # Each column must be the screw axis of that joint expressed in the
        # world frame, so its angular part is a unit vector.
        robot = make_robot("er3", "numpy")
        q = sample_positions(robot, rng)
        jac = robot.jacobian(q)
        for i in range(robot.num_dof):
            assert np.linalg.norm(jac[3:, i]) == pytest.approx(1.0, abs=1e-9)


class TestNormalVariants:
    """The relaxed-yaw Jacobian and pose error stay consistent with each other."""

    def test_normal_diff_uses_the_virtual_link(self, make_robot):
        from tools.geometry import rot_x

        robot = make_robot("er3", "numpy")
        current = np.eye(4)
        target = np.eye(4)
        target[:3, :3] = rot_x(np.pi / 2)  # tool Z goes from +Z to -Y
        target[:3, 3] = [0.1, 0.2, 0.3]

        diff = robot.normal_diff(current, target)
        assert diff.shape == (6,)
        assert np.allclose(diff[:3], [0.1, 0.2, 0.3])
        # The orientation term is the change of the tool Z axis, scaled by the
        # virtual link length.
        expected = (target[:3, 2] - current[:3, 2]) * robot.virtual_link
        assert np.allclose(diff[3:], expected)
        assert not np.allclose(expected, 0.0)

    def test_jac_normal_has_unit_tool_axis_column_scale(self, make_robot, rng):
        robot = make_robot("er3", "numpy")
        q = sample_positions(robot, rng)
        full = robot.jacobian(q)
        normal = robot.jac_normal(q)

        assert normal.shape == full.shape
        # The linear block is shared.
        assert np.allclose(normal[:3, :], full[:3, :])
        # The angular block is the full one projected off the tool Z axis.
        assert normal[3:, :].shape == (3, robot.num_dof)


class TestManipulability:
    """The manipulability measure and its gradient."""

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_is_the_sqrt_of_the_determinant(self, backend, make_robot, rng, name, dof):
        # Regression: the NumPy backend used to return det(J J^T) - the square
        # of the measure - while the CasADi backend returned its square root.
        robot = make_robot(name, backend)
        q = sample_positions(robot, rng)
        jac = robot.jacobian(q)
        expected = np.sqrt(np.linalg.det(jac @ jac.T) + 1e-10)
        assert robot.manip(q) == pytest.approx(expected, rel=1e-6, abs=1e-9)

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_backends_agree(self, make_robot, rng, name, dof):
        numpy_robot = make_robot(name, "numpy")
        casadi_robot = make_robot(name, "casadi")
        q = sample_positions(numpy_robot, rng)

        assert numpy_robot.manip(q) == pytest.approx(casadi_robot.manip(q), rel=1e-4)

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_gradient_matches_a_finite_difference(self, backend, make_robot, rng, name, dof):
        robot = make_robot(name, backend)
        q = sample_positions(robot, rng)

        analytic = robot.derivative_manip(q)
        # The NumPy backend differentiates with a forward difference of 1e-4, so
        # the same step keeps the truncation error of the two sides comparable.
        eps = 1e-4
        numeric = np.zeros(dof)
        for i in range(dof):
            delta = np.zeros(dof)
            delta[i] = eps
            numeric[i] = (robot.manip(q + delta) - robot.manip(q)) / eps

        assert analytic.shape == (dof,)
        assert np.allclose(analytic, numeric, rtol=1e-2, atol=1e-6)

    def test_numpy_gradient_uses_the_model_dof(self, make_robot, rng):
        # Regression: the NumPy backend hard-coded 7 joints.
        robot = make_robot("sr5", "numpy")
        assert robot.derivative_manip(sample_positions(robot, rng)).shape == (6,)

    def test_backends_agree_on_the_gradient(self, make_robot, rng):
        numpy_robot = make_robot("er3", "numpy")
        casadi_robot = make_robot("er3", "casadi")
        q = sample_positions(numpy_robot, rng)
        assert np.allclose(
            numpy_robot.derivative_manip(q),
            casadi_robot.derivative_manip(q),
            rtol=1e-3,
            atol=1e-5,
        )


class TestHessian:
    """The Jacobian derivative."""

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_backends_agree_on_shape_and_values(self, make_robot, rng, name, dof):
        # Regression: the NumPy backend returned (6, n, n) while the CasADi
        # backend returned (6n, n).
        numpy_robot = make_robot(name, "numpy")
        casadi_robot = make_robot(name, "casadi")
        q = sample_positions(numpy_robot, rng)

        numpy_hess = numpy_robot.hessian(q)
        casadi_hess = casadi_robot.hessian(q)

        assert numpy_hess.shape == (6, dof, dof)
        assert casadi_hess.shape == (6, dof, dof)
        # The NumPy backend uses a forward difference, so it is only approximate.
        assert np.allclose(numpy_hess, casadi_hess, atol=1e-3)

    def test_numpy_hessian_uses_the_model_dof(self, make_robot, rng):
        # Regression: the NumPy backend hard-coded 7 joints, so a 6-DOF robot
        # produced the wrong shape and dropped the last joint.
        robot = make_robot("sr5", "numpy")
        assert robot.num_dof == 6
        assert robot.hessian(sample_positions(robot, rng)).shape == (6, 6, 6)


class TestSe3Diff:
    """The SE(3) pose error."""

    def test_is_zero_for_identical_poses(self, backend, make_robot):
        robot = make_robot("er3", backend)
        diff = robot.se3_diff(np.eye(4), np.eye(4))
        assert np.allclose(np.asarray(diff).flatten()[:3], 0.0)

    def test_translation_part_is_the_position_error(self, backend, make_robot):
        robot = make_robot("er3", backend)
        target = np.eye(4)
        target[:3, 3] = [0.1, -0.2, 0.3]
        diff = np.asarray(robot.se3_diff(np.eye(4), target)).flatten()
        assert np.allclose(diff[:3], [0.1, -0.2, 0.3])
