"""Tests for the inverse-kinematics solvers.

IK is iterative and does not converge from every seed, so the tests always
generate the target from a known joint configuration and seed the solver nearby.
The assertions then either check an exact round trip or a convergence rate over
several random cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import ROBOTS, sample_positions

from model import IkNaive, IkNullSpace, IkStandard, IkType, make_ik_solver
from model.robot_model_base import RobotModelBase

#: Solvers that iterate to a tight tolerance over the full pose.
CONVERGING_SOLVERS = (IkType.IK_STANDARD,)
#: Solvers that deliberately relax the tool yaw, so only the position (and the
#: relaxed orientation error) is expected to converge tightly.
NORMAL_SOLVERS = (IkType.IK_NORMAL, IkType.IK_NULL_NORMAL)
#: The redundant-arm variants of both families.
REDUNDANT_SOLVERS = (IkType.IK_NULL, IkType.IK_NULL_NORMAL)


def _target_and_seed(robot: RobotModelBase, rng: np.random.Generator, spread: float = 0.1):
    """Return a reachable target pose and a nearby seed."""
    q_true = sample_positions(robot, rng, margin=0.2)
    target = robot.fk(q_true)
    seed = q_true + rng.uniform(-spread, spread, robot.num_dof)
    seed = np.clip(seed, robot.get_lower_bounds(), robot.get_upper_bounds())
    return target, seed, q_true


class TestSolverInterface:
    """The uniform contract every solver satisfies."""

    def test_returns_a_solution_and_a_flag(self, make_robot, rng):
        robot = make_robot("er3", "numpy")
        target, seed, _ = _target_and_seed(robot, rng)
        solution, success = robot.ik(seed, target)
        assert isinstance(solution, np.ndarray)
        assert solution.shape == (robot.num_dof,)
        assert isinstance(success, (bool, np.bool_))

    def test_already_solved_pose_reports_success_immediately(self, make_robot, rng, backend):
        robot = make_robot("er3", backend)
        q = sample_positions(robot, rng, margin=0.2)
        solution, success = robot.ik(q, robot.fk(q))
        assert success
        assert np.allclose(solution, q)

    def test_solution_respects_the_joint_limits(self, make_robot, rng, backend):
        robot = make_robot("sr5", backend)
        target, seed, _ = _target_and_seed(robot, rng)
        solution, _ = robot.ik(seed, target)
        assert np.all(solution <= robot.get_upper_bounds() + 1e-9)
        assert np.all(solution >= robot.get_lower_bounds() - 1e-9)

    def test_make_ik_solver_rejects_an_unknown_type(self, make_robot):
        robot = make_robot("sr5", "numpy")
        with pytest.raises(NotImplementedError):
            make_ik_solver(robot, "not-a-solver")

    def test_ik_before_build_raises(self):
        from model.robot_model_numpy import RobotModelNumpy

        with pytest.raises(RuntimeError, match="build"):
            RobotModelNumpy().ik(np.zeros(6), np.eye(4))


class TestConvergence:
    """The iterative solvers actually reach the requested pose."""

    @pytest.mark.parametrize("solver_type", CONVERGING_SOLVERS)
    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_converges_from_a_nearby_seed(
        self, backend, make_robot, rng, solver_type, name, dof
    ):
        robot = make_robot(name, backend, solver_type)
        target, seed, _ = _target_and_seed(robot, rng)
        solution, success = robot.ik(seed, target)

        assert success, f"{solver_type} failed to converge"
        assert np.allclose(robot.fk(solution), target, atol=1e-4)

    @pytest.mark.parametrize("solver_type", NORMAL_SOLVERS)
    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_normal_variants_converge_in_position(
        self, backend, make_robot, rng, solver_type, name, dof
    ):
        # These solvers trade tool yaw for reachability, so the full pose is not
        # expected to match. The relaxed residual they actually minimise must.
        robot = make_robot(name, backend, solver_type)
        target, seed, _ = _target_and_seed(robot, rng)
        solution, success = robot.ik(seed, target)

        assert success, f"{solver_type} failed to converge"
        achieved = robot.fk(solution)
        assert np.linalg.norm(achieved[:3, 3] - target[:3, 3]) < 5e-3
        residual = np.asarray(robot.ik_solver.diff_func(achieved, target)).flatten()
        assert np.linalg.norm(residual) < 5e-3

    @pytest.mark.parametrize("name,dof", ROBOTS)
    def test_convergence_rate_over_random_cases(self, backend, make_robot, name, dof):
        robot = make_robot(name, backend)
        rng = np.random.default_rng(4242)

        successes = 0
        trials = 20
        for _ in range(trials):
            target, seed, _ = _target_and_seed(robot, rng)
            solution, success = robot.ik(seed, target)
            if success and np.allclose(robot.fk(solution), target, atol=1e-4):
                successes += 1

        assert successes >= trials * 0.75

    def test_backends_reach_the_same_pose(self, make_robot):
        rng = np.random.default_rng(11)
        numpy_robot = make_robot("sr5", "numpy")
        casadi_robot = make_robot("sr5", "casadi")
        target, seed, _ = _target_and_seed(numpy_robot, rng)

        numpy_solution, numpy_ok = numpy_robot.ik(seed, target)
        casadi_solution, casadi_ok = casadi_robot.ik(seed, target)

        assert numpy_ok and casadi_ok
        assert np.allclose(
            numpy_robot.fk(numpy_solution), casadi_robot.fk(casadi_solution), atol=1e-4
        )


class TestNullSpaceSolvers:
    """The redundant-arm solvers, which additionally manage joint limits."""

    @pytest.mark.parametrize("solver_type", REDUNDANT_SOLVERS)
    def test_converges_on_a_redundant_arm(self, backend, make_robot, rng, solver_type):
        robot = make_robot("er3", backend, solver_type)
        target, seed, _ = _target_and_seed(robot, rng)
        solution, success = robot.ik(seed, target)

        assert success
        # IK_NULL_NORMAL relaxes the yaw, so only the position is checked tightly.
        assert np.linalg.norm(robot.fk(solution)[:3, 3] - target[:3, 3]) < 5e-3

    def test_soft_limits_require_a_usable_margin(self, make_robot):
        robot = make_robot("er3", "numpy", IkType.IK_NULL)
        assert isinstance(robot.ik_solver, IkNullSpace)

        # A range smaller than the 20-degree soft margin on both sides cannot
        # host a soft interval.
        robot.set_bounds(np.zeros(robot.num_dof), np.radians(10) * np.ones(robot.num_dof))
        with pytest.raises(ValueError, match="soft upper"):
            robot.ik_solver.set_bounds()

    def test_limit_subtask_is_inactive_away_from_the_limits(self, make_robot):
        robot = make_robot("er3", "numpy", IkType.IK_NULL)
        robot.ik_solver.set_bounds()
        centre = 0.5 * (robot.get_lower_bounds() + robot.get_upper_bounds())

        command, active = robot.ik_solver.subtask_avoid_limit(centre)
        assert not active
        assert np.allclose(command, 0.0)

    def test_limit_subtask_pushes_back_inside(self, make_robot):
        robot = make_robot("er3", "numpy", IkType.IK_NULL)
        robot.ik_solver.set_bounds()
        outside = robot.get_upper_bounds().copy()  # at the hard limit
        command, active = robot.ik_solver.subtask_avoid_limit(outside)

        assert active
        assert np.all(command <= 0.0)
        assert np.any(command < 0.0)


class TestNaiveSolver:
    """The single-step solver used inside a servo loop."""

    def test_takes_exactly_one_step(self, make_robot, rng):
        robot = make_robot("sr5", "numpy", IkType.IK_NAIVE)
        assert isinstance(robot.ik_solver, IkNaive)

        target, seed, _ = _target_and_seed(robot, rng)
        solution, success = robot.ik(seed, target)

        assert success
        assert not np.allclose(solution, seed)  # it did move
        assert not np.allclose(robot.fk(solution), target, atol=1e-9)  # but not all the way

    def test_reports_convergence_for_a_matching_pose(self, make_robot, rng):
        robot = make_robot("sr5", "numpy", IkType.IK_NAIVE)
        q = sample_positions(robot, rng)
        converged, diff = robot.ik_solver.solution_converge(robot.fk(q), robot.fk(q))
        assert converged
        assert np.allclose(np.asarray(diff).flatten(), 0.0, atol=1e-9)

    def test_converges_over_repeated_steps(self, make_robot, rng):
        # One step is not enough, but iterating the single step must converge.
        robot = make_robot("sr5", "numpy", IkType.IK_NAIVE)
        target, seed, _ = _target_and_seed(robot, rng, spread=0.02)

        q = seed
        for _ in range(60):
            q, _ = robot.ik(q, target)
            if np.allclose(robot.fk(q), target, atol=1e-4):
                break
        assert np.allclose(robot.fk(q), target, atol=1e-4)


class TestQpSolver:
    """The experimental QP solver, which currently mirrors the pseudo-inverse step."""

    def test_first_step_returns_a_solution(self, make_robot, rng):
        robot = make_robot("er3", "numpy", IkType.IK_QP)
        target, seed, _ = _target_and_seed(robot, rng)

        solution, success = robot.ik(seed, target)
        assert success
        assert solution.shape == (robot.num_dof,)

    def test_qp_coefficients_have_the_right_shapes(self, make_robot, rng):
        robot = make_robot("er3", "numpy", IkType.IK_QP)
        q = sample_positions(robot, rng)
        hess, grad, lbx, ubx = robot.ik_solver.get_qp_coefficients(
            robot.jacobian(q), q, np.zeros(6)
        )

        n = robot.num_dof
        assert hess.shape == (n, n)
        assert grad.shape == (n,)
        assert lbx.shape == (n,)
        assert ubx.shape == (n,)
        assert np.all(ubx >= lbx)
        # The Hessian of the least-squares cost is symmetric positive definite.
        assert np.allclose(hess, hess.T)


class TestSolverTypes:
    """Solver selection and the classes it instantiates."""

    @pytest.mark.parametrize(
        "solver_type,expected",
        [
            (IkType.IK_STANDARD, IkStandard),
            (IkType.IK_NORMAL, IkStandard),
            (IkType.IK_NULL, IkNullSpace),
            (IkType.IK_NULL_NORMAL, IkNullSpace),
            (IkType.IK_NAIVE, IkNaive),
        ],
    )
    def test_factory_selects_the_matching_class(self, make_robot, solver_type, expected):
        robot = make_robot("er3", "numpy", solver_type)
        assert isinstance(robot.ik_solver, expected)
        assert robot.ik_solver.name is solver_type

    def test_unsupported_variant_raises(self, make_robot):
        from model.ik_type import IkType as _IkType

        robot = make_robot("er3", "numpy")
        # IkNaive only implements the full SE(3) error, so asking it for the
        # normal variant must fail loudly instead of silently doing the wrong thing.
        with pytest.raises(NotImplementedError):
            IkNaive(robot, _IkType.IK_NORMAL)
