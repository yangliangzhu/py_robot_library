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
    """The pairwise axis table **at one configuration**.

    Only *adjacent* pairs are structural invariants.  Axis ``i`` is fixed in link ``i-1`` and axis
    ``i+1`` in link ``i``, and the two links differ by a rotation about axis ``i`` -- which leaves
    axis ``i`` where it is -- so the ``i, i+1`` distance is the same at every configuration.  For
    ``j >= i+2`` the intervening joints move one axis relative to the other, so the distance is a
    property of the *configuration*, not of the arm.

    An earlier version of this function averaged the distances over random configurations, which for
    non-adjacent pairs averages a varying quantity and reports a number that no configuration has;
    it also produced a self-contradictory table (axes 5 and 7 reported at 0 mm while the printed
    lines were 88 mm apart), which is how the defect was found.  Use :func:`structural_report` for
    the invariants.

    Args:
        model: Robot model.
        q: Configuration to measure at; zeros when omitted.
        samples: Accepted and ignored; kept for call compatibility.
        seed: Accepted and ignored; kept for call compatibility.

    Returns:
        One :class:`AxisPair` per pair ``i < j``, in row-major order, measured at ``q``.
    """
    del samples, seed
    configuration = np.zeros(model.num_dof) if q is None else np.asarray(q, dtype=float)
    origins, directions = joint_axes(model, configuration)
    pairs = []
    for i in range(model.num_dof):
        for j in range(i + 1, model.num_dof):
            pairs.append(
                AxisPair(
                    i + 1,
                    j + 1,
                    line_distance(origins[i], directions[i], origins[j], directions[j]),
                    line_angle(directions[i], directions[j]),
                )
            )
    return pairs


def axis_triple_is_concurrent(
    model, triple: tuple[int, int, int], q: np.ndarray | None = None
) -> tuple[bool, float]:
    """Whether three joint axes meet at one point -- the invariant test for a spherical wrist.

    The first two axes of the triple must be adjacent (or already intersect); the test is the
    distance from *their* intersection point to the third axis.  That distance is configuration
    independent when the first two are adjacent, which is what makes the answer structural rather
    than a snapshot.

    Args:
        model: Robot model.
        triple: Three 1-based joint numbers, e.g. ``(4, 5, 6)``.
        q: Configuration to measure at; zeros when omitted.

    Returns:
        ``(concurrent, distance in metres)``.
    """
    first, second, third = (index - 1 for index in triple)
    configuration = np.zeros(model.num_dof) if q is None else np.asarray(q, dtype=float)
    origins, directions = joint_axes(model, configuration)
    point = _intersection_point(
        origins[first], directions[first], origins[second], directions[second]
    )
    if point is None:
        return False, float("nan")
    return (
        line_distance(point, directions[third], origins[third], directions[third]) < 1e-9,
        line_distance(point, directions[third], origins[third], directions[third]),
    )


def _intersection_point(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    """Intersection point of two lines, or ``None`` when they are parallel."""
    cross = np.cross(d1, d2)
    if float(np.linalg.norm(cross)) < 1e-12:
        return None
    matrix = np.array([[d1 @ d1, -(d1 @ d2)], [d1 @ d2, -(d2 @ d2)]])
    right = np.array([(p2 - p1) @ d1, (p2 - p1) @ d2])
    parameters = np.linalg.solve(matrix, right)
    return p1 + parameters[0] * d1


def structural_report(model, *, wrist: tuple[int, int, int] | None = None) -> str:
    """The *invariant* structural facts: adjacent-axis distances and axis-triple concurrency.

    Args:
        model: Robot model.
        wrist: Triple to test for concurrency; the last three axes when omitted.

    Returns:
        A multi-line report.
    """
    n = model.num_dof
    wrist = wrist or (n - 2, n - 1, n)
    lines = ["adjacent-axis geometry (configuration independent), in mm:"]
    for index in range(n - 1):
        pair = [
            item for item in axis_fingerprint(model)
            if item.first == index + 1 and item.second == index + 2
        ][0]
        lines.append(
            f"  axes {pair.first}-{pair.second}: distance {1000 * pair.distance:9.4f} mm, "
            f"angle {pair.angle_deg:6.2f} deg"
        )
    for triple in ((1, 2, 3), (n - 2, n - 1, n)):
        concurrent, distance = axis_triple_is_concurrent(model, triple)
        lines.append(
            f"  axes {triple[0]}-{triple[1]}-{triple[2]}: "
            f"{'CONCURRENT (spherical)' if concurrent else 'not concurrent'}"
            f" | offset {1000 * distance:9.4f} mm"
        )
    return "\n".join(lines)


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
