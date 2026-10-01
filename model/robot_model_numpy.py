"""NumPy implementation of the robot model.

This backend evaluates the kinematics with dense NumPy arrays on every call. It
is the reference implementation: straightforward to read, and fast enough for
trajectory generation and offline analysis. Use
:class:`~model.robot_model_casadi.RobotModelCasadi` when symbolic derivatives or
compiled expressions are needed.
"""

from __future__ import annotations

import numpy as np

from .robot_model_base import RobotModelBase

__all__ = ["RobotModelNumpy"]


class RobotModelNumpy(RobotModelBase):
    """A serial robot arm evaluated with NumPy.

    The arm is described by an ordered transform list ``Ms`` of length
    ``num_dof + 2``: a base transform, one transform per joint, and a tool
    transform. :meth:`fk` walks that list, inserting a rotation about the local
    Z axis for each joint.

    Example:
        >>> import numpy as np
        >>> from model.robot_model_numpy import RobotModelNumpy
        >>> robot = RobotModelNumpy()
        >>> robot.build("dh", IkType.IK_STANDARD, config)  # doctest: +SKIP
        >>> pose = robot.fk(np.zeros(robot.num_dof))
    """

    ########################## generate expressions ###################################

    @staticmethod
    def rot_z(theta: float) -> np.ndarray:
        """Return the 4x4 transform for a rotation of ``theta`` about Z."""
        matrix = np.eye(4)
        matrix[0, 0] = np.cos(theta)
        matrix[0, 1] = -np.sin(theta)
        matrix[1, 0] = np.sin(theta)
        matrix[1, 1] = np.cos(theta)
        return matrix

    ########################## kinematics #############################################

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Compute the forward kinematics.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            The 4x4 end-effector pose.
        """
        transform = self.Ms[0]
        for i in range(self.num_dof):
            transform = transform @ self.Ms[i + 1] @ self.rot_z(q[i])
        return transform @ self.Ms[self.num_dof + 1]

    def se3_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the SE(3) error from pose ``a`` to pose ``b``.

        Args:
            a: The current 4x4 pose.
            b: The target 4x4 pose.

        Returns:
            A ``(6, 1)`` error vector ``[dx, dy, dz, wx, wy, wz]``, the
            translational part in metres and the rotational part as a
            rotation-vector in radians.
        """
        dx = (b - a)[0:3, 3]
        dr = b[0:3, 0:3] @ a[0:3, 0:3].T
        if np.allclose(a[:3, :3], b[:3, :3], atol=1e-5):
            w1 = w2 = w3 = 0.0
        else:
            phi = np.arccos(np.clip((dr[0, 0] + dr[1, 1] + dr[2, 2] - 1) / 2, -1.0, 1.0))
            omega = phi / (2 * np.sin(phi)) * (dr - dr.T)
            w1 = omega[2, 1]
            w2 = omega[0, 2]
            w3 = omega[1, 0]
        return np.array([dx[0], dx[1], dx[2], w1, w2, w3]).reshape(6, 1)

    def normal_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the "normal" pose error used by the IK-normal variants.

        The orientation error is reduced to the tilt of the tool Z axis scaled
        by :attr:`virtual_link`, which loosens the yaw constraint and makes a
        7-DOF arm easier to converge.

        Args:
            a: The current 4x4 pose.
            b: The target 4x4 pose.

        Returns:
            A length-6 error vector.
        """
        diff = np.zeros(6)
        diff[:3] = (b - a)[:3, 3]
        diff[3:] = (b - a)[:3, 2] * self.virtual_link
        return diff

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 6xn geometric Jacobian.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            A ``(6, num_dof)`` Jacobian: the first three rows map joint
            velocities to linear velocity, the last three to angular velocity.
        """
        frame_list = []
        pose = self.Ms[0]
        for i in range(self.num_dof):
            pose = pose @ self.Ms[i + 1] @ self.rot_z(q[i])
            frame_list.append(pose.copy())
        tool_pose = pose @ self.Ms[-1]

        jac = np.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            offset = tool_pose[:3, 3] - frame_list[i][:3, 3]
            angular_v = frame_list[i][:3, 2]
            jac[:3, i] = np.cross(angular_v, offset)
            jac[3:, i] = angular_v
        return jac

    def jac_normal(self, q: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matching :meth:`normal_diff`.

        The angular block is replaced by the derivative of the virtual-link
        displacement, dropping the tool yaw direction.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            A ``(6, num_dof)`` Jacobian.
        """
        jac = self.jacobian(q)
        z_axis = self.fk(q)[:3, 2]
        z_axis_hat = np.array(
            [
                [0.0, -z_axis[2], z_axis[1]],
                [z_axis[2], 0.0, -z_axis[0]],
                [-z_axis[1], z_axis[0], 0.0],
            ]
        )
        jac[3:, :] = -z_axis_hat @ jac[3:, :] * self.virtual_link
        return jac

    ########################## analysis ###############################################

    def manip(self, q: np.ndarray) -> float:
        """Return the Yoshikawa manipulability measure at ``q``.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            ``sqrt(det(J @ J.T))``, which vanishes at kinematic singularities.
            This matches both the CasADi backend and the C++ implementation; an
            earlier version returned ``det(J @ J.T)``, i.e. the square of the
            measure, which disagreed with :meth:`derivative_manip` below.
        """
        jac = self.jacobian(q)
        manip_square = np.linalg.det(jac @ jac.T)
        return float(np.sqrt(manip_square + 1e-10))

    def is_not_singular(self, q: np.ndarray) -> bool:
        """Report whether ``q`` is away from a kinematic singularity."""
        return bool(np.abs(self.manip(q)) > 0.005)

    def hessian(self, q: np.ndarray) -> np.ndarray:
        """Estimate the Jacobian derivative with forward differences.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            A ``(6, num_dof, num_dof)`` array.
        """
        result = np.zeros((6, self.num_dof, self.num_dof))
        eps = 1e-4
        jac_base = self.jacobian(q)
        for i in range(self.num_dof):
            delta_q = np.zeros(self.num_dof)
            delta_q[i] = eps
            result[:, :, i] = (self.jacobian(q + delta_q) - jac_base) / eps
        return result

    def derivative_manip(self, q: np.ndarray) -> np.ndarray:
        """Return the gradient of the manipulability measure.

        A forward difference is used instead of the analytic derivative: it is
        faster, and the null-space subtask tolerates the lower accuracy.

        Args:
            q: Joint positions in radians, length ``num_dof``.

        Returns:
            The gradient, length ``num_dof``.
        """
        grad_manip = np.zeros(self.num_dof)
        delta = 1e-4
        jac = self.jacobian(q)
        manip_square = np.linalg.det(jac @ jac.T)
        manip = np.sqrt(manip_square + 1e-10)
        coeff = 1.0 / (2.0 * manip) if manip > delta else 0.0
        for i in range(self.num_dof):
            dq = np.zeros(self.num_dof)
            dq[i] = delta
            jac_new = self.jacobian(q + dq)
            manip_square_new = np.linalg.det(jac_new @ jac_new.T)
            # * f(q + dq) - f(q) rather than a central difference, for speed.
            grad_manip[i] = (manip_square_new - manip_square) / delta
        return grad_manip * coeff


if __name__ == "__main__":
    pass
