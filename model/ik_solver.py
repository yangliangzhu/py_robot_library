"""Inverse-kinematics solvers.

Every solver is a callable that maps ``(seed_joint_positions, target_pose)`` to
``(joint_positions, success)``, so the robot models can expose a uniform
:meth:`~model.robot_model_base.RobotModelBase.ik` regardless of the strategy
behind it.

Two pose-error conventions are in play, and they must be paired with the
matching Jacobian:

``standard``
    :meth:`~model.robot_model_numpy.RobotModelNumpy.se3_diff` with
    :meth:`~model.robot_model_numpy.RobotModelNumpy.jacobian` - full SE(3)
    error, all six degrees of freedom constrained.
``normal``
    :meth:`~model.robot_model_numpy.RobotModelNumpy.normal_diff` with
    :meth:`~model.robot_model_numpy.RobotModelNumpy.jac_normal` - the tool yaw
    is relaxed, which helps a 7-DOF arm converge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import casadi as ca
import numpy as np

from .ik_type import IkType

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .robot_model_base import RobotModelBase

__all__ = [
    "IkStandard",
    "IkNullSpace",
    "IkNaive",
    "IkQp",
    "make_ik_solver",
]

#: Solver types that use the full SE(3) error.
_STANDARD_FAMILY = (IkType.IK_STANDARD, IkType.IK_NULL)
#: Solver types that relax the tool yaw.
_NORMAL_FAMILY = (IkType.IK_NORMAL, IkType.IK_NULL_NORMAL)


class _IkSolverBase:
    """Shared solver plumbing: pose error and Jacobian selection.

    Args:
        model: The robot model the solver operates on. It must already be built.
        solver_type: The :class:`~model.ik_type.IkType` this instance implements.
    """

    def __init__(self, model: RobotModelBase, solver_type: IkType) -> None:
        self.model = model
        self.name = solver_type
        self.configure(solver_type)

    def configure(self, solver_type: IkType) -> None:
        """Bind the pose-error and Jacobian functions for ``solver_type``.

        Args:
            solver_type: The solver variant being configured.

        Raises:
            NotImplementedError: If this solver does not support ``solver_type``.
        """
        if solver_type in _STANDARD_FAMILY:
            self.jac_func: Callable[[np.ndarray], np.ndarray] = self.model.jacobian
            self.diff_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = self.model.se3_diff
        elif solver_type in _NORMAL_FAMILY:
            self.jac_func = self.model.jac_normal
            self.diff_func = self.model.normal_diff
        else:
            raise NotImplementedError(
                f"[ik] {solver_type} is not implemented by {type(self).__name__}"
            )


def _damped_pinv(jac: np.ndarray, singular_threshold: float) -> np.ndarray:
    """Return a singular-value-truncated pseudo-inverse of ``jac``.

    Singular values below ``singular_threshold`` are treated as zero, which
    discards the corresponding Cartesian directions instead of producing the
    huge joint velocities a plain inverse would.

    Args:
        jac: The Jacobian to invert.
        singular_threshold: Relative cut-off on the singular values.

    Returns:
        The pseudo-inverse, shape ``(n, 6)``.
    """
    u_mat, s_vec, v_mat = np.linalg.svd(jac)
    s_vec = s_vec.copy()
    below = s_vec < singular_threshold
    s_vec[below] = 0.0
    s_vec[~below] = 1.0 / s_vec[~below]

    s_mat = np.zeros((v_mat.shape[0], u_mat.shape[0]))
    for i in range(s_vec.shape[0]):
        s_mat[i, i] = s_vec[i]
    return v_mat.T @ s_mat @ u_mat.T


class IkStandard(_IkSolverBase):
    """Damped least-squares IK with adaptive step-size back-off.

    Each iteration takes a truncated pseudo-inverse step, capped at
    ``pos_step_`` in translation and ``rot_step_`` in rotation. When an
    iteration makes the error worse the step size is scaled down and the
    iteration budget extended; if it falls below ``1e-4`` the solve fails.
    """

    def __call__(self, angle: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Solve IK from ``angle`` towards ``target``.

        Args:
            angle: Seed joint positions in radians.
            target: Desired 4x4 end-effector pose.

        Returns:
            A ``(joint_positions, success)`` tuple. On failure the returned
            positions are the best estimate reached so far.
        """
        tar_t = target
        # A unit gain makes convergence hard, so the budget is adaptive.
        max_iter = 20.0
        max_iter_sup = 50
        pre_t = self.model.fk(angle)
        t = 0
        diff = self.diff_func(pre_t, tar_t)
        dis1 = np.linalg.norm(diff[:3])
        dis2 = np.linalg.norm(diff[3:])
        eps = 1e-5
        err_norm = max(dis1, dis2)
        if err_norm < eps:
            return angle, True

        last_err_norm = 1e5
        step_size = 1.0
        pos_step_ = 1e-1
        rot_step_ = 5e-2
        q_best = np.array(angle)

        while t < max_iter:
            if err_norm > last_err_norm:
                factor = min(0.2, last_err_norm / err_norm)
                max_iter = t + 1 + (max_iter - t - 1) / factor
                max_iter = min(max_iter_sup, max_iter)
                step_size *= factor
                if step_size < 1e-4:
                    return angle, False
                angle = np.array(q_best)
            else:
                q_best = np.array(angle)
                jac = self.jac_func(angle)
                # A transmission ratio below 1% drops that direction.
                inv_jac = _damped_pinv(jac, singular_threshold=0.01)

                if dis1 > pos_step_:
                    diff[:3] = diff[:3] * pos_step_ / dis1
                if dis2 > rot_step_:
                    diff[3:] = diff[3:] * rot_step_ / dis2
                dq = inv_jac @ diff.flatten()

            angle = angle + dq * step_size
            angle = np.clip(angle, self.model.lower_bounds, self.model.upper_bounds)
            pre_t = self.model.fk(angle)
            diff = self.diff_func(pre_t, tar_t)
            dis1 = np.linalg.norm(diff[:3])
            dis2 = np.linalg.norm(diff[3:])
            last_err_norm = err_norm
            err_norm = max(dis1, dis2)
            if err_norm < eps:
                return angle, True
            t = t + 1
        return angle, False


class IkNullSpace(_IkSolverBase):
    """IK that projects a subtask through the Jacobian null space.

    Built for 7-DOF arms: the primary task tracks the end-effector pose while a
    redundancy subtask uses the remaining degrees of freedom. The subtask first
    pushes joints away from their soft limits; once no limit is violated it
    maximises manipulability instead.
    """

    def __init__(self, model: RobotModelBase, solver_type: IkType) -> None:
        self.set_bounds_flag = False
        super().__init__(model, solver_type)

    def set_bounds(self) -> None:
        """Compute the soft joint limits used by the limit-avoidance subtask.

        Raises:
            ValueError: If a 20-degree margin does not fit inside the limits.
        """
        # TODO: scale the velocity by how far the joint is outside its limit.
        soft_limit = np.radians(20)
        self.soft_upper = self.model.upper_bounds - soft_limit
        self.soft_lower = self.model.lower_bounds + soft_limit
        if np.any(self.soft_upper - self.soft_lower < 0):
            raise ValueError("soft upper must be >= soft lower")
        self.set_bounds_flag = True

    def subtask_avoid_limit(self, q: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return a joint command pushing joints back inside their soft limits.

        Args:
            q: Current joint positions in radians.

        Returns:
            A ``(command, active)`` tuple; ``active`` is ``False`` when every
            joint is already inside its soft limits.
        """
        velocity = np.radians(100)
        dt = 2e-2
        v = np.zeros_like(q)
        for i in range(self.model.num_dof):
            if q[i] > self.soft_upper[i]:
                v[i] = -1.0
            elif q[i] < self.soft_lower[i]:
                v[i] = 1.0
        if np.allclose(v, np.zeros_like(v)):
            return np.zeros_like(v), False
        return velocity * dt * v, True

    def sub_task_manip(self, q: np.ndarray) -> np.ndarray:
        """Return a joint command that increases manipulability."""
        velocity = np.radians(100)
        dt = 2e-2
        grad = self.model.derivative_manip(q)
        grad_vec = grad / (np.linalg.norm(grad) + 1e-10)
        return velocity * dt * grad_vec

    def __call__(self, angle: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Solve IK from ``angle`` towards ``target``.

        Args:
            angle: Seed joint positions in radians.
            target: Desired 4x4 end-effector pose.

        Returns:
            A ``(joint_positions, success)`` tuple.
        """
        if not self.set_bounds_flag:
            self.set_bounds()

        tar_t = target
        max_iter = 20.0
        max_iter_sup = 50
        pre_t = self.model.fk(angle)
        t = 0
        diff = self.diff_func(pre_t, tar_t)
        dis1 = np.linalg.norm(diff[:3])
        dis2 = np.linalg.norm(diff[3:])
        eps = 1e-5
        err_norm = max(dis1, dis2)
        if err_norm < eps:
            return angle, True

        last_err_norm = 1e5
        step_size = 1.0
        pos_step_ = 1e-1
        rot_step_ = 5e-2
        q_best = np.array(angle)

        while t < max_iter:
            if err_norm > last_err_norm:
                factor = min(0.2, last_err_norm / err_norm)
                max_iter = t + 1 + (max_iter - t - 1) / factor
                max_iter = min(max_iter_sup, max_iter)
                step_size *= factor
                if step_size < 1e-4:
                    return angle, False
                angle = np.array(q_best)
            else:
                q_best = np.array(angle)
                jac = self.jac_func(angle)
                u_mat, s_vec, v_mat = np.linalg.svd(jac)
                dim_v = v_mat.shape[0]
                dim_u = u_mat.shape[0]

                # Identity minus the row space of J: the null-space projector.
                s_null = np.eye(dim_v)
                singular_threshold = 0.01
                s_vec = s_vec.copy()
                for i in range(s_vec.shape[0]):
                    if s_vec[i] < singular_threshold:
                        s_vec[i] = 0.0
                    else:
                        s_vec[i] = 1.0 / s_vec[i]
                        s_null[i, i] = 0.0

                s_mat = np.zeros((dim_v, dim_u))
                for i in range(s_vec.shape[0]):
                    s_mat[i, i] = s_vec[i]
                inv_jac = v_mat.T @ s_mat @ u_mat.T
                null_jac = v_mat.T @ s_null @ v_mat
                null_space_exist = not np.allclose(s_null, np.zeros_like(s_null))

                if dis1 > pos_step_:
                    diff[:3] = diff[:3] * pos_step_ / dis1
                if dis2 > rot_step_:
                    diff[3:] = diff[3:] * rot_step_ / dis2
                dq = inv_jac @ diff.flatten()

                dq_null, activate_limit_task = self.subtask_avoid_limit(angle)
                if activate_limit_task and null_space_exist:
                    dq += null_jac @ dq_null
                elif null_space_exist:
                    dq += null_jac @ self.sub_task_manip(angle)

            angle = angle + dq * step_size
            angle = np.clip(angle, self.model.lower_bounds, self.model.upper_bounds)
            pre_t = self.model.fk(angle)
            diff = self.diff_func(pre_t, tar_t)
            dis1 = np.linalg.norm(diff[:3])
            dis2 = np.linalg.norm(diff[3:])
            last_err_norm = err_norm
            err_norm = max(dis1, dis2)
            if err_norm < eps:
                return angle, True
            t = t + 1
        return angle, False


class IkNaive(_IkSolverBase):
    """A single damped least-squares step.

    Not an iterative solver: it applies one truncated pseudo-inverse step and
    always reports success. Useful as a building block for a real-time servo
    loop that runs one step per control cycle.
    """

    def configure(self, solver_type: IkType) -> None:
        """Bind the full SE(3) error and Jacobian.

        Raises:
            NotImplementedError: If ``solver_type`` is not :attr:`IkType.IK_NAIVE`.
                This solver has a single variant, so an unsupported request must
                fail loudly rather than silently solve a different problem.
        """
        if solver_type is not IkType.IK_NAIVE:
            raise NotImplementedError(
                f"[ik] {solver_type} is not implemented by {type(self).__name__}"
            )
        self.jac_func = self.model.jacobian
        self.diff_func = self.model.se3_diff

    def solution_converge(
        self, pre_t: np.ndarray, tar_t: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        """Report whether two poses are within tolerance of each other.

        Args:
            pre_t: The current 4x4 pose.
            tar_t: The target 4x4 pose.

        Returns:
            A ``(converged, diff)`` tuple.
        """
        diff = self.diff_func(pre_t, tar_t)
        eps_trans = 1e-4
        eps_rot = np.deg2rad(0.5)
        trans_converge = np.linalg.norm(diff[:3]) < eps_trans
        rot_converge = np.linalg.norm(diff[3:]) < eps_rot
        return (trans_converge and rot_converge), diff

    def __call__(self, angle: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Take one IK step from ``angle`` towards ``target``."""
        pre_t = self.model.fk(angle)
        _, diff = self.solution_converge(pre_t, target)

        jac = self.jac_func(angle)
        inv_jac = _damped_pinv(jac, singular_threshold=2e-10)

        angle = angle + inv_jac @ diff.flatten()
        angle = np.clip(angle, self.model.lower_bounds, self.model.upper_bounds)
        return angle, True


class IkQp(_IkSolverBase):
    """One-step IK formulated as a quadratic program.

    Minimises ``||J dq - dx||^2 + w ||dq - dq_last||^2`` subject to
    acceleration, velocity and joint-position limits, solved with qpOASES
    through CasADi.

    Warning:
        Experimental. The QP result is currently computed for comparison but the
        step actually applied comes from the pseudo-inverse solution, so this
        solver does not yet enforce its constraints.
    """

    #: TODO: one-step IK-QP is still not available.
    def __init__(self, model: RobotModelBase, solver_type: IkType) -> None:
        self.last_dq = np.zeros(model.num_dof)
        self.ddq_max = np.ones(model.num_dof) * 100.0
        self.dq_max = np.ones(model.num_dof) * np.deg2rad(80)
        self.dt = 1e-3
        self.penalty_dq = 1e-4
        self.initialized = False
        super().__init__(model, solver_type)

    def configure(self, solver_type: IkType) -> None:
        """Bind the full SE(3) error and Jacobian, and build the QP solver.

        Raises:
            NotImplementedError: If ``solver_type`` is not :attr:`IkType.IK_QP`.
        """
        if solver_type is not IkType.IK_QP:
            raise NotImplementedError(
                f"[ik] {solver_type} is not implemented by {type(self).__name__}"
            )
        self.jac_func = self.model.jacobian
        self.diff_func = self.model.se3_diff

        n = self.model.num_dof
        qp = {
            "h": ca.DM.zeros(n, n).sparsity(),
            "a": ca.DM.zeros(n, n).sparsity(),
        }
        opts = {
            "printLevel": "none",
            "error_on_fail": False,
        }
        self.solver = ca.conic("solver", "qpoases", qp, opts)
        self.initialized = False

    def get_qp_coefficients(
        self,
        jac: np.ndarray,
        q_ref: np.ndarray,
        dx_des: np.ndarray,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the QP cost and box constraints.

        The program is::

            min ||J dq - dx||^2 + w ||dq - dq_last||^2
            s.t. -ddq * dt^2    < dq - dq_last < ddq * dt^2
                 - dq_max * dt  <      dq      < dq_max * dt
                         q_min  <  q + dq      < q_max

        Args:
            jac: The Jacobian at ``q_ref``.
            q_ref: Current joint positions in radians.
            dx_des: Desired Cartesian step.
            verbose: Print the individual and combined constraint bounds.

        Returns:
            The ``(H, g, lbx, ubx)`` tuple.
        """
        # TODO: add joint-limit avoidance.
        n = self.model.num_dof
        hess = jac.T @ jac + self.penalty_dq * np.eye(n)
        grad = -(self.penalty_dq * q_ref + jac.T @ dx_des)

        lbx_acc = self.last_dq - self.ddq_max * self.dt**2
        ubx_acc = self.last_dq + self.ddq_max * self.dt**2
        lbx_vel = -self.dq_max * self.dt
        ubx_vel = self.dq_max * self.dt
        lbx_pos = self.model.get_lower_bounds() - q_ref
        ubx_pos = self.model.get_upper_bounds() - q_ref

        if verbose:
            print(f"lbx_acc: {lbx_acc}")
            print(f"ubx_acc: {ubx_acc}")
            print(f"lbx_vel: {lbx_vel}")
            print(f"ubx_vel: {ubx_vel}")
            print(f"lbx_pos: {lbx_pos}")
            print(f"ubx_pos: {ubx_pos}")

        lbx = np.maximum(np.maximum(lbx_acc, lbx_vel), lbx_pos)
        ubx = np.minimum(np.minimum(ubx_acc, ubx_vel), ubx_pos)
        return hess, grad, lbx, ubx

    def initial_call(self, q_ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        """First-step IK: step towards ``target`` and seed the QP warm start."""
        tar_t = target
        pre_t = self.model.fk(q_ref)
        dx_des = self.diff_func(pre_t, tar_t).flatten()
        jac = self.model.jacobian(q_ref)
        inv_jac = _damped_pinv(jac, singular_threshold=0.01)
        dq_pinv = inv_jac @ dx_des

        if not self.initialized:
            self.last_dq = dq_pinv
        self.initialized = True

        return dq_pinv

    def __call__(self, q_ref: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Take one IK step from ``q_ref`` towards ``target``."""
        if not self.initialized:
            return q_ref + self.initial_call(q_ref, target), True

        pre_t = self.model.fk(q_ref)
        dx_des = self.diff_func(pre_t, target).flatten()
        jac = self.model.jacobian(q_ref)
        hess, grad, lbx, ubx = self.get_qp_coefficients(jac, q_ref, dx_des)
        res = self.solver(h=hess, g=grad, lbx=lbx, ubx=ubx)
        opti_dq = res["x"].full().flatten()
        self.last_dq = opti_dq

        return q_ref + opti_dq, True


#: Solver types a robot model can be built with.
_SOLVER_REGISTRY = {
    IkType.IK_STANDARD: IkStandard,
    IkType.IK_NORMAL: IkStandard,
    IkType.IK_NULL: IkNullSpace,
    IkType.IK_NULL_NORMAL: IkNullSpace,
    IkType.IK_NAIVE: IkNaive,
    IkType.IK_QP: IkQp,
}


def make_ik_solver(model: RobotModelBase, solver_type: IkType) -> _IkSolverBase:
    """Instantiate the solver implementing ``solver_type``.

    Args:
        model: The robot model to solve for. It must already be built.
        solver_type: One of the :class:`~model.ik_type.IkType` values.

    Returns:
        The solver instance.

    Raises:
        NotImplementedError: If no solver is registered for ``solver_type``.

    Example:
        >>> solver = make_ik_solver(robot, IkType.IK_NULL)  # doctest: +SKIP
    """
    try:
        solver_class = _SOLVER_REGISTRY[solver_type]
    except KeyError:
        raise NotImplementedError(f"[ik] {solver_type} not implemented") from None
    return solver_class(model, solver_type)
