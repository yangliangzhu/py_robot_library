"""Denavit-Hartenberg parameter handling and link-transform construction.

Two conventions are supported:

* **MDH** (modified / Craig) - ``M = Tx(a) @ Rx(alpha) @ Tz(d) @ Rz(theta)``
* **SDH** (standard / Denavit-Hartenberg) - ``M = Rz(theta) @ Tz(d) @ Rx(alpha) @ Tx(a)``

An SDH chain implicitly carries one extra flange offset: the last link frame of
an SDH description stops at the flange rather than at the tool. It is emitted as
a separate identity-plus-offset entry by :func:`get_matrix_list` so that the
tool transform can be composed on top of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = [
    "Rx",
    "Ry",
    "Rz",
    "Tx",
    "Ty",
    "Tz",
    "Dh",
    "mdh_to_matrix_list",
    "sdh_to_matrix_list",
    "get_matrix_list",
]

#: Default ordering of the four DH entries inside a parameter row.
DEFAULT_ORDER_MAP: tuple[str, ...] = ("d", "alpha", "a", "theta")

#: Recognised DH conventions.
DH_TYPES: tuple[str, ...] = ("mdh", "sdh")

Matrix4 = np.ndarray


def Rx(x: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a rotation about X.

    Args:
        x: Rotation angle in radians.

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    res = np.eye(4)
    res[1, 1] = np.cos(x)
    res[1, 2] = -np.sin(x)
    res[2, 1] = np.sin(x)
    res[2, 2] = np.cos(x)
    return res


def Ry(x: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a rotation about Y.

    Args:
        x: Rotation angle in radians.

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    res = np.eye(4)
    res[0, 0] = np.cos(x)
    res[0, 2] = np.sin(x)
    res[2, 0] = -np.sin(x)
    res[2, 2] = np.cos(x)
    return res


def Rz(x: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a rotation about Z.

    Args:
        x: Rotation angle in radians.

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    res = np.eye(4)
    res[0, 0] = np.cos(x)
    res[0, 1] = -np.sin(x)
    res[1, 0] = np.sin(x)
    res[1, 1] = np.cos(x)
    return res


def Tx(p: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a translation along X."""
    res = np.eye(4)
    res[0, 3] = p
    return res


def Ty(p: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a translation along Y."""
    res = np.eye(4)
    res[1, 3] = p
    return res


def Tz(p: float) -> Matrix4:
    """Return the 4x4 homogeneous transform for a translation along Z."""
    res = np.eye(4)
    res[2, 3] = p
    return res


class Dh:
    """A Denavit-Hartenberg table.

    Attributes:
        dh_list: One ``[d, alpha, a, theta]`` row per joint, in the order given
            by ``order_map``.
        order_map: Names describing the column order of each row.
        dh_type: Either ``"mdh"`` or ``"sdh"``.
        d_idx: Column index of the ``d`` entry.
        alpha_idx: Column index of the ``alpha`` entry.
        a_idx: Column index of the ``a`` entry.
        theta_idx: Column index of the ``theta`` entry.
    """

    def __init__(
        self,
        dh_list: Sequence[Sequence[float]],
        dh_type: str = "mdh",
        order_map: Iterable[str] = DEFAULT_ORDER_MAP,
    ) -> None:
        """Build a DH table.

        Args:
            dh_list: Per-joint parameter rows.
            dh_type: ``"mdh"`` or ``"sdh"``.
            order_map: Column names in the order they appear in each row.

        Raises:
            ValueError: If ``dh_type`` is not a supported convention, or if
                ``order_map`` is missing one of ``d``, ``alpha``, ``a``,
                ``theta``.
        """
        if dh_type not in DH_TYPES:
            raise ValueError(f"dh_type must be one of {DH_TYPES}, got {dh_type!r}")

        order = list(order_map)
        missing = {"d", "alpha", "a", "theta"} - set(order)
        if missing:
            raise ValueError(f"order_map is missing entries: {sorted(missing)}")

        self.dh_list = [list(row) for row in dh_list]
        self.order_map = order
        self.dh_type = dh_type
        self.d_idx = order.index("d")
        self.alpha_idx = order.index("alpha")
        self.a_idx = order.index("a")
        self.theta_idx = order.index("theta")

    def millimeter_to_meter(self) -> None:
        """Convert the ``a`` and ``d`` columns from millimetres to metres.

        The table is modified in place.
        """
        for row in self.dh_list:
            row[self.a_idx] /= 1000
            row[self.d_idx] /= 1000

    def print_order_map(self) -> None:
        """Print the column order of the DH rows."""
        print(self.order_map)


def mdh_to_matrix_list(dh_params: Dh) -> list[Matrix4]:
    """Convert a modified-DH table to per-joint link transforms.

    Args:
        dh_params: The DH table.

    Returns:
        One 4x4 transform per joint, using
        ``M = Tx(a) @ Rx(alpha) @ Tz(d) @ Rz(theta)``.
    """
    matrices = []
    for dh in dh_params.dh_list:
        trans = (
            Tx(dh[dh_params.a_idx])
            @ Rx(dh[dh_params.alpha_idx])
            @ Tz(dh[dh_params.d_idx])
            @ Rz(dh[dh_params.theta_idx])
        )
        matrices.append(trans)
    return matrices


def sdh_to_matrix_list(dh_params: Dh) -> list[Matrix4]:
    """Convert a standard-DH table to per-joint link transforms.

    Args:
        dh_params: The DH table.

    Returns:
        One 4x4 transform per joint, using
        ``M = Rz(theta) @ Tz(d) @ Rx(alpha) @ Tx(a)``.
    """
    matrices = []
    for dh in dh_params.dh_list:
        trans = (
            Rz(dh[dh_params.theta_idx])
            @ Tz(dh[dh_params.d_idx])
            @ Rx(dh[dh_params.alpha_idx])
            @ Tx(dh[dh_params.a_idx])
        )
        matrices.append(trans)
    return matrices


def get_matrix_list(
    dh_params: Dh,
    base: Matrix4,
    ee: Matrix4,
) -> list[Matrix4]:
    """Build the full ordered transform list for a DH robot.

    The returned list is laid out as ``[base, link_1 ... link_n, tool]``, which
    is the form :class:`~model.robot_model_numpy.RobotModelNumpy` and
    :class:`~model.robot_model_casadi.RobotModelCasadi` expect. For an SDH
    table an extra identity frame is inserted before the flange so that the
    SDH flange offset and the end-effector transform compose correctly.

    Args:
        dh_params: The DH table describing the arm.
        base: Transform from the world frame to the robot base.
        ee: Transform from the flange to the end effector.

    Returns:
        The ordered transform list.

    Raises:
        ValueError: If ``dh_params.dh_type`` is not supported.

    Note:
        For SDH robots the last link transform produced by the DH table is
        folded into the end-effector entry, so calling this again with a
        different ``ee`` is not equivalent to re-running it from scratch.
    """
    res = [base]
    if dh_params.dh_type == "mdh":
        res.extend(mdh_to_matrix_list(dh_params))
        res.append(ee)
    elif dh_params.dh_type == "sdh":
        matrices = sdh_to_matrix_list(dh_params)
        res.append(np.eye(4))
        last_link_trans = matrices.pop()
        res.extend(matrices)
        res.append(last_link_trans @ ee)
    else:
        raise ValueError(f"unsupported dh type: {dh_params.dh_type!r}")

    return res
