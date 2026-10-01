"""Rotation and rigid-transform helpers.

Conventions used throughout this module:

* Rotation matrices are 3x3 and act on column vectors.
* Homogeneous transforms are 4x4.
* Angles are in radians unless a function documents otherwise.
* Quaternions are ``[w, x, y, z]``, scalar part first.
* RPY angles are ``[roll, pitch, yaw]`` for a ZYX composition, i.e.
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = [
    "rot_x",
    "rot_y",
    "rot_z",
    "rpy2rot",
    "rot2rpy",
    "quat2rot",
    "rot2quat",
    "nervRpy",
    "nervCartToAffine",
]

VectorLike = Union[np.ndarray, list, tuple]
Matrix3 = np.ndarray
Matrix4 = np.ndarray


def rot_x(theta: float) -> Matrix3:
    """Return the 3x3 rotation matrix for a rotation about X.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rot_y(theta: float) -> Matrix3:
    """Return the 3x3 rotation matrix for a rotation about Y.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rot_z(theta: float) -> Matrix3:
    """Return the 3x3 rotation matrix for a rotation about Z.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rpy2rot(rpy: VectorLike) -> Matrix3:
    """Convert RPY angles to a rotation matrix.

    Args:
        rpy: ``[roll, pitch, yaw]`` in radians.

    Returns:
        A 3x3 rotation matrix, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    return rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])


def rot2rpy(rot: Matrix3) -> np.ndarray:
    """Convert a rotation matrix to RPY angles.

    The inverse of :func:`rpy2rot`. The gimbal-lock branch fixes ``yaw`` at zero
    and recovers ``roll`` from the remaining terms.

    Args:
        rot: A 3x3 rotation matrix.

    Returns:
        ``[roll, pitch, yaw]`` in radians.
    """
    r11, _, _ = rot[0, :]
    r21, _, _ = rot[1, :]
    r31, r32, r33 = rot[2, :]

    if abs(r31) < 0.9999999:
        yaw = np.arctan2(r21, r11)
        pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))
        roll = np.arctan2(r32, r33)
    else:
        yaw = 0.0
        if r31 < 0:
            pitch = np.pi / 2
            roll = np.arctan2(rot[0, 1], rot[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-rot[0, 1], -rot[0, 2])

    return np.array([roll, pitch, yaw])


def quat2rot(q: VectorLike) -> Matrix3:
    """Convert a quaternion to a rotation matrix.

    The quaternion is normalised first, so a non-unit input is accepted.

    Args:
        q: Quaternion ``[w, x, y, z]``.

    Returns:
        A 3x3 rotation matrix.

    Raises:
        ValueError: If ``q`` does not have exactly 4 elements.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Input must be a quaternion with 4 elements [w, x, y, z]")

    q = q / np.linalg.norm(q)
    w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
        ]
    )


def rot2quat(rot: Matrix3) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    Uses the branch on the largest diagonal element, which keeps the division
    well conditioned. The returned quaternion has a non-negative scalar part only
    when the trace is positive; the sign is otherwise whatever the branch yields,
    and ``q`` and ``-q`` describe the same rotation.

    Args:
        rot: A 3x3 rotation matrix.

    Returns:
        Quaternion ``[w, x, y, z]``.

    Raises:
        ValueError: If ``rot`` is not 3x3.
    """
    rot = np.asarray(rot, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix")

    trace = np.trace(rot)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def nervRpy(p: VectorLike, rad: bool = False) -> Matrix3:
    """Return the rotation block of a nerv-format Cartesian pose.

    Args:
        p: Six-element pose ``[x, y, z, rx, ry, rz]``. The translation part is
            ignored.
        rad: When ``False`` (the default) the orientation is converted from
            degrees to radians first.

    Returns:
        A 3x3 rotation matrix.
    """
    omega = np.asarray(p[3:], dtype=float)
    if not rad:
        omega = np.deg2rad(omega)
    return rpy2rot(omega)


def nervCartToAffine(p: VectorLike, rad: bool = False) -> Matrix4:
    """Convert a nerv-format Cartesian pose to a 4x4 homogeneous transform.

    Args:
        p: Six-element pose ``[x, y, z, rx, ry, rz]`` with the translation in
            millimetres.
        rad: When ``False`` (the default) the orientation is converted from
            degrees to radians first.

    Returns:
        A 4x4 homogeneous transform with the translation in metres.
    """
    omega = np.asarray(p[3:], dtype=float)
    if not rad:
        omega = np.deg2rad(omega)

    result = np.eye(4)
    result[:3, 3] = np.asarray(p[:3], dtype=float) * 1e-3  # mm -> m
    result[:3, :3] = rpy2rot(omega)
    return result
