"""CasADi implementation of the robot model.

This backend builds the forward kinematics symbolically once, at
:meth:`RobotModelCasadi.set_tool` time, and derives the Jacobian, the Hessian,
the pseudo-inverse and the manipulability gradient from those expressions. Every
subsequent evaluation is a compiled CasADi function call, which is markedly
faster than re-walking the transform list in Python and makes the derivatives
exact rather than finite-difference approximations.

The price is that the tool transform is baked into the compiled expressions, so
:meth:`~model.robot_model_base.RobotModelBase.set_tool` has to rebuild them.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .robot_model_base import RobotModelBase

__all__ = ["RobotModelCasadi"]


class RobotModelCasadi(RobotModelBase):
    """A serial robot arm whose kinematics are compiled CasADi functions.

    Example:
        >>> import numpy as np
        >>> from model.robot_model_casadi import RobotModelCasadi
        >>> robot = RobotModelCasadi()
        >>> robot.build("dh", IkType.IK_STANDARD, config)  # doctest: +SKIP
        >>> jac = robot.jacobian(np.zeros(robot.num_dof))
    """

    def __init__(self) -> None:
        super().__init__()
        # Compiled CasADi functions, populated by build_kinematics_functions().
        self.__fk = None
        self.__jacobian = None
        self.__pinv_jac = None
        self.__jac_normal = None
        self.__hessian = None
        self.__manip = None
        self.__d_manip = None
        self.build_se3_diff_functions()

    ########################## generate casadi expressions ############################

    @staticmethod
    def rot_z(theta: ca.SX) -> ca.SX:
        """Return the symbolic 4x4 transform for a rotation about Z."""
        matrix = ca.SX_eye(4)
        matrix[0, 0] = ca.cos(theta)
        matrix[0, 1] = -ca.sin(theta)
        matrix[1, 0] = ca.sin(theta)
        matrix[1, 1] = ca.cos(theta)
        return matrix

    def fk_sym(self, joints: ca.SX) -> ca.SX:
        """Build the symbolic forward kinematics.

        Args:
            joints: Symbolic joint positions, length ``num_dof``.

        Returns:
            The symbolic 4x4 end-effector pose.
        """
        last_transform = self.Ms[0]
        for i in range(self.num_dof + 1):
            theta = joints[i] if i < self.num_dof else 0
            last_transform = last_transform @ self.Ms[i + 1] @ self.rot_z(theta)
        return last_transform

    def build_kinematics_functions(self) -> None:
        """Compile the FK, Jacobian, Hessian and manipulability functions.

        Called from :meth:`_on_tool_changed`, so it must not be invoked until
        ``Ms`` and ``num_dof`` are set by
        :meth:`~model.robot_model_base.RobotModelBase.build`.
        """
        q = ca.SX.sym("q", self.num_dof)
        last_frame = self.fk_sym(q)
        self.__fk = ca.Function("forward", [q], [last_frame])

        # Linear velocity from the position Jacobian, angular velocity from the
        # derivatives of the rotation-matrix rows.
        jacx = ca.jacobian(last_frame[:3, 3], q)
        row1 = last_frame[0, :3]
        row2 = last_frame[1, :3]
        row3 = last_frame[2, :3]
        omegax = row2 @ ca.jacobian(row3, q)
        omegay = row3 @ ca.jacobian(row1, q)
        omegaz = row1 @ ca.jacobian(row2, q)
        jac = ca.vertcat(jacx, omegax, omegay, omegaz)
        self.__jacobian = ca.Function("jacobian", [q], [jac])

        # The "normal" Jacobian constrains the tilt of the tool Z axis scaled by
        # the virtual link instead of the full orientation.
        normal_vec = last_frame[:3, 2] * self.virtual_link
        jac_normal = ca.jacobian(normal_vec, q)
        self.__jac_normal = ca.Function(
            "jac_normal", [q], [ca.vertcat(jacx, jac_normal)]
        )

        self.__hessian = ca.Function("hessian", [q], [ca.jacobian(jac, q)])
        self.__pinv_jac = ca.Function("pinv_jacobian", [q], [ca.pinv(jac)])

        manip = ca.sqrt(ca.det(jac @ jac.T))
        self.__manip = ca.Function("manip", [q], [manip])
        self.__d_manip = ca.Function("d_manip", [q], [ca.jacobian(manip, q)])

    def build_se3_diff_functions(self) -> None:
        """Compile the symbolic SE(3) pose-error functions.

        Two variants are built: the general one, and one that reports a zero
        rotation error, used when the two orientations already match to avoid the
        ``sin(phi)`` division by zero at ``phi = 0``.
        """
        a = ca.SX.sym("A", 4, 4)
        b = ca.SX.sym("B", 4, 4)
        dx = (b - a)[0:3, 3]
        dr = b[0:3, 0:3] @ a[0:3, 0:3].T

        phi = ca.acos((dr[0, 0] + dr[1, 1] + dr[2, 2] - 1) / 2)
        omega = phi / (2 * ca.sin(phi)) * (dr - dr.T)
        rhs = ca.vertcat(dx[0], dx[1], dx[2], omega[2, 1], omega[0, 2], omega[1, 0])
        self.__se3_diff = ca.Function("diff_matrix", [a, b], [rhs])

        rhs_singular = ca.vertcat(dx[0], dx[1], dx[2], 0, 0, 0)
        self.__se3_diff_singular = ca.Function(
            "diff_matrix_singular", [a, b], [rhs_singular]
        )

    def _on_tool_changed(self) -> None:
        """Recompile the kinematics after the tool transform changed."""
        self.build_kinematics_functions()

    ########################## kinematics #############################################

    def fk(self, angle: np.ndarray) -> np.ndarray:
        """Compute the forward kinematics.

        Args:
            angle: Joint positions in radians, length ``num_dof``.

        Returns:
            The 4x4 end-effector pose.
        """
        return np.array(self.__fk(angle))

    def jacobian(self, angle: np.ndarray) -> np.ndarray:
        """Return the 6xn geometric Jacobian at ``angle``."""
        return np.array(self.__jacobian(angle))

    def pinv_jac(self, angle: np.ndarray) -> np.ndarray:
        """Return the pseudo-inverse of the Jacobian at ``angle``."""
        return np.array(self.__pinv_jac(angle))

    def jac_normal(self, angle: np.ndarray) -> np.ndarray:
        """Return the Jacobian matching :meth:`normal_diff`."""
        return np.array(self.__jac_normal(angle))

    def hessian(self, angle: np.ndarray) -> np.ndarray:
        """Return the exact Jacobian derivative.

        Args:
            angle: Joint positions in radians, length ``num_dof``.

        Returns:
            A ``(6, num_dof, num_dof)`` array, matching the shape the NumPy
            backend returns. CasADi evaluates ``dJ/dq`` as a ``(6 * num_dof,
            num_dof)`` matrix whose rows are ``vec(J)`` in column-major order, so
            the reshape below uses Fortran ordering to recover ``dJ[r, i]/dq[j]``
            at index ``[r, i, j]``.
        """
        return np.array(self.__hessian(angle)).reshape(
            6, self.num_dof, self.num_dof, order="F"
        )

    def se3_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the SE(3) error from pose ``a`` to pose ``b``.

        Args:
            a: The current 4x4 pose.
            b: The target 4x4 pose.

        Returns:
            A length-6 error vector ``[dx, dy, dz, wx, wy, wz]``.
        """
        if np.allclose(a[:3, :3], b[:3, :3], atol=1e-5):
            return np.array(self.__se3_diff_singular(a, b))
        return np.array(self.__se3_diff(a, b))

    def normal_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the "normal" pose error used by the IK-normal variants.

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

    ########################## analysis ###############################################

    def manip(self, q: np.ndarray) -> np.ndarray:
        """Return the manipulability measure at ``q``."""
        return np.array(self.__manip(q))[0, 0]

    def is_not_singular(self, q: np.ndarray) -> bool:
        """Report whether ``q`` is away from a kinematic singularity."""
        return bool(np.abs(self.__manip(q))[0, 0] > 0.005)

    def derivative_manip(self, q: np.ndarray) -> np.ndarray:
        """Return the exact gradient of the manipulability measure."""
        return np.array(self.__d_manip(q)).flatten()


if __name__ == "__main__":
    pass
