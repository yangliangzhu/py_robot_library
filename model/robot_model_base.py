"""Shared configuration and parameter handling for robot models.

Both backends - :class:`~model.robot_model_numpy.RobotModelNumpy` and
:class:`~model.robot_model_casadi.RobotModelCasadi` - describe a serial arm with
the same ordered transform list::

    Ms = [base, link_1, ..., link_n, tool]

Kinematics differ between the backends, but building that list from a robot
configuration, tracking joint limits and managing the tool transform do not.
Those live here so the two backends only implement what actually differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from .dh_param import Dh, get_matrix_list
from .ik_solver import make_ik_solver

__all__ = ["RobotModelBase", "DEFAULT_VIRTUAL_LINK"]

#: Length of the virtual link used by the "normal" IK variants, in metres.
DEFAULT_VIRTUAL_LINK = 0.15


class RobotModelBase(ABC):
    """Common behaviour of the NumPy and CasADi robot models.

    Subclasses implement the kinematics (:meth:`fk`, :meth:`jacobian`, ...) and
    override :meth:`_on_tool_changed` when the tool transform is baked into
    pre-compiled expressions.

    Attributes:
        num_dof: Number of actuated joints.
        Ms: Ordered transform list ``[base, link_1, ..., link_n, tool]``.
        virtual_link: Length of the virtual link for the "normal" IK variants.
        upper_bounds: Upper joint limits in radians.
        lower_bounds: Lower joint limits in radians.
        ik_solver: The IK strategy selected by :meth:`build`.
    """

    def __init__(self) -> None:
        self.num_dof: int = 0
        self.Ms: list[np.ndarray] = []
        self.virtual_link: float = DEFAULT_VIRTUAL_LINK
        self.upper_bounds: np.ndarray = np.zeros(0)
        self.lower_bounds: np.ndarray = np.zeros(0)
        self.fixed_offset: np.ndarray | None = None
        self.ik_solver: Any = None

    ########################## configure robot ########################################

    def build(
        self,
        param_type: str,
        solver_type: Any,
        config: Mapping[str, Any],
        base: np.ndarray | None = None,
        ee: np.ndarray | None = None,
        tool: np.ndarray | None = None,
    ) -> None:
        """Configure the model from a robot configuration mapping.

        Args:
            param_type: ``"dh"`` when ``config["param"]`` is a :class:`~model.dh_param.Dh`
                table, ``"mat"`` when it is a list of link transforms.
            solver_type: An :class:`~model.ik_type.IkType` value selecting the IK strategy.
            config: Mapping produced by :func:`model.configs.loader.load_robot_config`,
                holding ``param``, ``lower`` and ``upper``.
            base: Transform from the world frame to the robot base. Defaults to
                the identity.
            ee: Fixed transform from the flange to the end effector. Defaults to
                the identity.
            tool: Transform of the currently mounted, interchangeable tool.
                Defaults to the identity.

        Raises:
            KeyError: If ``config`` is missing a required entry.
            NotImplementedError: If ``param_type`` is not ``"dh"`` or ``"mat"``.
            TypeError: If ``param_type`` is ``"dh"`` but ``config["param"]`` is
                not a :class:`~model.dh_param.Dh`.
        """
        base = np.eye(4) if base is None else base
        ee = np.eye(4) if ee is None else ee
        tool = np.eye(4) if tool is None else tool

        param = config["param"]
        if param_type == "dh":
            if not isinstance(param, Dh):
                raise TypeError(
                    f"param_type='dh' expects a Dh instance, got {type(param).__name__}"
                )
            self.num_dof = len(param.dh_list)
            self.Ms = get_matrix_list(param, base, ee)
        elif param_type == "mat":
            self.num_dof = len(param)
            self.Ms = [base, *param, ee]
        else:
            raise NotImplementedError(f"param type not supported: {param_type!r}")

        # Remember the flange-to-EE transform: set_tool() always composes onto
        # this rather than onto whatever the previous tool left behind.
        self.fixed_offset = self.Ms[-1].copy()
        self.set_tool(tool)
        self.set_bounds(config["lower"], config["upper"])

        # Must run last: the solvers bind to the finished model.
        self.ik_solver = make_ik_solver(self, solver_type)

    def get_upper_bounds(self) -> np.ndarray:
        """Return the upper joint limits in radians."""
        return self.upper_bounds

    def get_lower_bounds(self) -> np.ndarray:
        """Return the lower joint limits in radians."""
        return self.lower_bounds

    def set_bounds(
        self,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> None:
        """Set the joint limits.

        Args:
            lower_bounds: Lower joint limits in radians.
            upper_bounds: Upper joint limits in radians.

        Raises:
            ValueError: If the limits are not one-dimensional and of equal
                length, or if any lower limit exceeds its upper limit.
        """
        if lower_bounds is None or upper_bounds is None:
            self.lower_bounds = np.zeros(0)
            self.upper_bounds = np.zeros(0)
            return

        lower = np.asarray(lower_bounds, dtype=float).ravel()
        upper = np.asarray(upper_bounds, dtype=float).ravel()
        if lower.shape != upper.shape:
            raise ValueError(
                f"bounds must have the same length, got {lower.shape} and {upper.shape}"
            )
        if np.any(lower > upper):
            raise ValueError("every lower bound must be <= its upper bound")

        self.lower_bounds = lower
        self.upper_bounds = upper

    def set_tool(self, matrix: np.ndarray) -> None:
        """Mount an interchangeable tool on the flange.

        The new transform replaces the previous tool rather than accumulating on
        top of it, and is composed onto the fixed flange-to-EE transform.

        Args:
            matrix: 4x4 transform from the flange to the tool tip.
        """
        if self.fixed_offset is None:
            raise RuntimeError("build() must be called before set_tool()")
        self.Ms[-1] = self.fixed_offset @ matrix
        self._on_tool_changed()

    def _on_tool_changed(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Hook invoked after :meth:`set_tool` updates the transform list.

        Backends that bake the tool transform into compiled expressions override
        this to regenerate them. The NumPy backend reads ``Ms`` on every call
        and therefore does nothing, so the hook is deliberately optional and
        not abstract.
        """

    ########################## kinematics #############################################

    @abstractmethod
    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the end-effector pose for joint positions ``q``."""

    @abstractmethod
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the 6xn geometric Jacobian at joint positions ``q``."""

    @abstractmethod
    def se3_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return a 6x1 pose error between two homogeneous transforms."""

    ########################## user interfaces ########################################

    def ik(self, angle: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Solve inverse kinematics with the configured solver.

        Args:
            angle: Seed joint positions in radians.
            target: Desired 4x4 end-effector pose.

        Returns:
            A ``(joint_positions, success)`` tuple.
        """
        if self.ik_solver is None:
            raise RuntimeError("build() must be called before ik()")
        return self.ik_solver(angle, target)
