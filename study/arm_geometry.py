"""Joint-axis geometry: the structure that decides closed-form solvability.

Pieper's condition is a statement about *axes*, not about DH tables: a 6R has a closed form when
the last three joint axes meet at a point (a spherical wrist), and more generally when enough of
the chain's axes intersect or are parallel.  The DH tables in ``model/configs`` are in three
different notations, so reading solvability off them is error-prone; measuring it off the axes is
not.

What this module provides:

* :func:`joint_frames` -- the frame after each joint, on either backend (the library's two
  backends share the ``Ms`` layout, so one implementation covers both);
* :func:`joint_axes` -- each joint's axis as a point and a direction in the base frame;
* :func:`line_distance` / :func:`line_angle` -- the skew distance and angle between two axes;
* :func:`axis_fingerprint` -- the pairwise table, which is what tells you whether a wrist is
  spherical and which parameters would have to move to make it so.

For a spherical wrist every pair among the last three axes has distance ~0.  For the SR5 the
numbers are not zero -- which is the measured form of "the engineers were right, there is no
Pieper closed form" -- and the size of those distances is exactly what a homotopy has to travel
(see ``study/NOTES.md``, Finding 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot_z(theta: float) -> np.ndarray:
    """Homogeneous rotation about z, matching the library's own convention."""
    cosine, sine = np.cos(theta), np.sin(theta)
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return transform


def joint_frames(model, q: np.ndarray) -> list[np.ndarray]:
    """The frame after each joint, plus the base frame first.

    Args:
        model: Robot model.  Both backends share the ``Ms = [base, link_1, ..., link_n, tool]``
            layout, so this works for either.
        q: Joint positions in radians, length ``num_dof``.

    Returns:
        ``[T_base, T_1, ..., T_n]``, where ``T_i`` is the pose of the frame that joint ``i``
        rotates in (equivalently: after joint ``i`` has been applied).
    """
    q = np.asarray(q, dtype=float)
    frames = [np.asarray(model.Ms[0], dtype=float).copy()]
    pose = frames[0]
    for index in range(model.num_dof):
        pose = pose @ np.asarray(model.Ms[index + 1], dtype=float) @ rot_z(q[index])
        frames.append(pose)
    return frames


def joint_axes(model, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each joint's axis in the base frame.

    Args:
        model: Robot model.
        q: Joint positions in radians.

    Returns:
        ``(origins, directions)``, each ``(num_dof, 3)``.  Row ``i`` is joint ``i + 1``, the
        axis it rotates about, as a point and a unit direction.
    """
    frames = joint_frames(model, q)
    origins = np.array([frame[:3, 3] for frame in frames[1:]])
    directions = np.array([frame[:3, 2] for frame in frames[1:]])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions / np.where(norms > 0, norms, 1.0)


def line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """Shortest distance between two lines (0 when they intersect).

    Args:
        p1: A point on the first line.
        d1: The first line's unit direction.
        p2: A point on the second line.
        d2: The second line's unit direction.

    Returns:
        The distance; ``0.0`` for intersecting lines, and the common-perpendicular length
        otherwise.  Parallel lines return the point-to-line distance.
    """
    cross = np.cross(d1, d2)
    norm = float(np.linalg.norm(cross))
    if norm < 1e-12:  # parallel: distance from p2 to the first line
        delta = p2 - p1
        return float(np.linalg.norm(delta - np.dot(delta, d1) * d1))
    return float(abs(np.dot(p2 - p1, cross / norm)))


def line_angle(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle between two axis directions, in degrees, folded into ``[0, 90]``."""
    cosine = float(np.clip(abs(np.dot(d1, d2)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class AxisPair:
    """One entry of the pairwise axis table."""

    first: int
    second: int
    distance: float
    angle_deg: float

    @property
    def intersects(self) -> bool:
        """Whether the two axes meet, within a micrometre."""
        return self.distance < 1e-6

    @property
    def parallel(self) -> bool:
        """Whether the two axes are parallel to within a hundredth of a degree."""
        return self.angle_deg < 0.01


def axis_fingerprint(
    model, q: np.ndarray | None = None, *, samples: int = 5, seed: int = 0
) -> list[AxisPair]:
    """The pairwise axis table, averaged over a few configurations.

    Averaging over configurations is safe for these quantities: the *distance* between two joint
    axes is a property of the arm's structure and does not depend on ``q`` at all for a serial
    chain (each axis is fixed in its parent link), so several samples are a check rather than a
    necessity -- the spread is reported by the caller if it wants it.

    Args:
        model: Robot model.
        q: Configuration to measure at; a random regular one when omitted.
        samples: How many configurations to average over.
        seed: Random seed for those configurations.

    Returns:
        One :class:`AxisPair` per pair ``i < j``, in row-major order.
    """
    rng = np.random.default_rng(seed)
    accumulators = np.zeros((model.num_dof, model.num_dof))
    angles = np.zeros_like(accumulators)
    for _ in range(samples):
        configuration = (
            rng.uniform(model.lower_bounds, model.upper_bounds) if q is None else np.asarray(q, float)
        )
        origins, directions = joint_axes(model, configuration)
        for i in range(model.num_dof):
            for j in range(i + 1, model.num_dof):
                accumulators[i, j] += line_distance(
                    origins[i], directions[i], origins[j], directions[j]
                )
                angles[i, j] += line_angle(directions[i], directions[j])
    pairs = []
    for i in range(model.num_dof):
        for j in range(i + 1, model.num_dof):
            pairs.append(
                AxisPair(i + 1, j + 1, accumulators[i, j] / samples, angles[i, j] / samples)
            )
    return pairs


def describe(model, q: np.ndarray | None = None) -> str:
    """A printable fingerprint: intersecting, parallel and skew axis pairs.

    Args:
        model: Robot model.
        q: Configuration to measure at; random when omitted.

    Returns:
        A multi-line report.
    """
    pairs = axis_fingerprint(model, q)
    intersecting = [pair for pair in pairs if pair.intersects]
    parallel = [pair for pair in pairs if pair.parallel]
    lines = [
        f"axes: {model.num_dof}",
        "  intersecting (distance < 1e-6 m): "
        + (", ".join(f"{p.first}-{p.second}" for p in intersecting) or "none"),
        "  parallel (angle < 0.01 deg):      "
        + (", ".join(f"{p.first}-{p.second}" for p in parallel) or "none"),
        "  all pairs (distance m / angle deg):",
    ]
    for pair in pairs:
        lines.append(
            f"    {pair.first}-{pair.second}: {pair.distance:11.6f} / {pair.angle_deg:7.2f}"
        )
    return "\n".join(lines)
