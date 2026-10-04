"""A deterministic, complete inverse-kinematics solver for the SR0 arm.

``SR0`` is the ROKAE SR5 with one number changed: link 5's y offset, ``+0.136 m``, set to zero
(`study/exp10_sr0_geometry.py`).  That single change makes the last three joint axes meet at a
point -- a spherical wrist -- so the arm is Pieper-solvable, and it is the starting point of the
homotopy that Finding 2 is built on.

The structure this module exploits is *measured*, not assumed:

* the shoulder centre ``S = (0, 0, 0.328)`` is the intersection of axes 1 and 2, constant in the
  base frame;
* joint 1 turns the whole arm about the base z axis, so the wrist centre's azimuth gives two
  candidates for ``q1``, a half turn apart;
* the arm is planar: the wrist centre lies in the plane spanned by the base z axis and that
  azimuth (measured: the offset of the wrist centre from that plane is zero);
* in the plane, ``q3`` alone decides the wrist centre's distance from the shoulder, so a scalar
  root-find over ``q3`` is complete; ``q2`` then follows in closed form from an angle
  ``psi(q3)`` that is *evaluated from the model* rather than fitted to an assumed link geometry;
* the wrist centre sits at a fixed point of the flange, ``(0, 0, -0.1035)``, so the target pose
  gives the wrist centre directly;
* the wrist is spherical with perpendicular axes and satisfies ``n4 . n6 = cos(q5)`` exactly
  (measured), giving ``q5 = +-acos(n4 . n6)``; ``q4`` and ``q6`` then follow from a two-variable
  Newton step that is deterministic (no random restarts) and whose result is verified against the
  forward kinematics before it is returned.

So: two closed-form branches for ``q1``, two roots for ``q3``, closed-form ``q2``, two branches
for ``q5`` and a deterministic finish for ``(q4, q6)`` -- eight solutions, all verified.  Nothing
here needs a random seed, which is the property a homotopy needs from its starting fibre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import IkType
from model.configs.loader import load_robot_config
from model.robot_model_numpy import RobotModelNumpy
from study import arm_geometry as ag
from study.exp11_dh_continuation import link_transforms

#: Shoulder centre in the base frame: the intersection of axes 1 and 2, measured constant.
SHOULDER_CENTRE = np.array([0.0, 0.0, 0.328])

#: The wrist centre expressed in the flange frame, measured constant on SR0.
WRIST_IN_FLANGE = np.array([0.0, 0.0, -0.1035])

#: Acceptance for a solution to count, on ``max(position error, rotation error)``.
TOLERANCE = 1e-9


def sr0_links() -> list[np.ndarray]:
    """The SR5 link transforms with the one offset that blocks Pieper set to zero."""
    links = link_transforms()
    links[4] = links[4].copy()
    links[4][1, 3] = 0.0
    return links


def robot(links: list[np.ndarray] | None = None) -> RobotModelNumpy:
    """A numpy model of the (possibly relaxed) arm."""
    config = load_robot_config("rokae_sr5")
    config["param"] = [
        np.asarray(link, dtype=float) for link in (sr0_links() if links is None else links)
    ]
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config)
    return model


@dataclass
class Sr0Solution:
    """One IK solution, with the residual it was verified at and whether the robot can reach it.

    The pose check and the limit check are different questions (exp45 measured a pose where all eight
    closed-form solutions lie outside the joint limits while every one of them satisfies the pose), so
    both flags travel with the solution instead of being silently conflated.
    """

    q: np.ndarray
    position_error: float
    rotation_error: float
    within_limits: bool = True

    @property
    def error(self) -> float:
        """The worst of the two errors."""
        return max(self.position_error, self.rotation_error)


def _frames(model: RobotModelNumpy, q: np.ndarray) -> list[np.ndarray]:
    """Joint frames via :mod:`study.arm_geometry` (the library layout, unrolled here)."""
    return ag.joint_frames(model, q)


def _planar_point(model: RobotModelNumpy, q2: float, q3: float) -> np.ndarray:
    """Wrist centre at ``(q1=0, q2, q3)`` in the plane's coordinates: (radial, vertical)."""
    point = _frames(model, np.array([0.0, q2, q3, 0.0, 0.0, 0.0]))[4][:3, 3] - SHOULDER_CENTRE
    return np.array([point[0], point[2]])


def _planar_rotation_sign(model: RobotModelNumpy) -> float:
    """Whether increasing ``q2`` rotates the wrist centre counter-clockwise in the plane.

    The arm plane's normal is axis 2's direction, and axis 2's sense is a property of the URDF,
    not of the mathematics: assuming one sign silently rejects every candidate, which is how a
    first version of this solver returned no solutions at all.  Measuring the sign costs one
    forward-kinematics call.
    """
    before = _planar_point(model, 0.0, 0.7)
    after = _planar_point(model, 0.2, 0.7)
    turn = float(
        np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0])
    )
    return 1.0 if turn >= 0.0 else -1.0


def position_solutions(model: RobotModelNumpy, target: np.ndarray) -> list[np.ndarray]:
    """All ``(q1, q2, q3)`` that put the wrist centre where the target asks.

    Args:
        model: The (SR0) model.
        target: Desired 4x4 tool pose.

    Returns:
        Up to four configurations, each ``[q1, q2, q3, 0, 0, 0]``.
    """
    wrist = (target @ np.append(WRIST_IN_FLANGE, 1.0))[:3]
    radial = wrist - SHOULDER_CENTRE
    azimuth = float(np.arctan2(radial[1], radial[0]))
    turn_sign = _planar_rotation_sign(model)
    solutions: list[np.ndarray] = []

    def reach(q3: float, direction: float) -> float:
        """Distance the wrist centre reaches for ``q3`` with ``q2`` matched to ``direction``."""
        base = _planar_point(model, 0.0, q3)
        psi = float(np.arctan2(base[1], base[0]))
        q2 = turn_sign * (direction - psi)
        return float(np.linalg.norm(_planar_point(model, q2, q3)))

    grid = np.linspace(-np.pi, np.pi, 361)
    for q1 in (azimuth, azimuth + np.pi):
        # q1 rotates the whole chain about the base z axis through the shoulder, so the planar
        # target is the wrist centre rotated back by q1 -- keeping its sign, which is what
        # separates the two shoulder branches.
        back = np.array(
            [
                np.cos(-q1) * radial[0] - np.sin(-q1) * radial[1],
                radial[2],
            ]
        )
        wanted = float(np.linalg.norm(back))
        direction = float(np.arctan2(back[1], back[0]))
        values = np.array([reach(q3, direction) - wanted for q3 in grid])
        for index in range(len(grid) - 1):
            low, high = values[index], values[index + 1]
            if low * high > 0.0 and low != 0.0:
                continue
            a, b, fa = float(grid[index]), float(grid[index + 1]), low
            for _ in range(80):
                mid = 0.5 * (a + b)
                fm = reach(mid, direction) - wanted
                if (fa < 0.0) != (fm < 0.0):
                    b = mid
                else:
                    a, fa = mid, fm
            q3 = 0.5 * (a + b)
            base = _planar_point(model, 0.0, q3)
            psi = float(np.arctan2(base[1], base[0]))
            q2 = turn_sign * (direction - psi)
            candidate = np.array([q1, q2, q3, 0.0, 0.0, 0.0])
            reached = _frames(model, candidate)[4][:3, 3]
            if float(np.linalg.norm(reached - wrist)) < 1e-9:
                solutions.append(candidate)
    return _unique(solutions)


def _unique(points: list[np.ndarray], tol: float = 1e-6) -> list[np.ndarray]:
    """Drop duplicates (angles compared on the torus)."""
    out: list[np.ndarray] = []
    for point in points:
        if not any(
            np.linalg.norm((point - other + np.pi) % (2 * np.pi) - np.pi) < tol for other in out
        ):
            out.append(point)
    return out


def _wrist_solutions(
    model: RobotModelNumpy, target: np.ndarray, q123: np.ndarray
) -> list[np.ndarray]:
    """Complete ``(q4, q5, q6)`` for a fixed position part, by the measured wrist relations."""
    frames = _frames(model, q123)
    n4 = frames[4][:3, 2]
    n6_target = target[:3, 2]
    cosine = float(np.clip(n4 @ n6_target, -1.0, 1.0))
    candidates: list[np.ndarray] = []
    for q5 in (np.arccos(cosine), -np.arccos(cosine)):
        for q4_start, q6_start in ((0.0, 0.0), (np.pi, np.pi), (0.0, np.pi), (np.pi, 0.0)):
            q = q123.copy()
            q[3], q[4], q[5] = q4_start, q5, q6_start
            for _ in range(80):  # deterministic two-variable Newton on the wrist
                error = _rotation_error(_frames(model, q)[6][:3, :3], target[:3, :3])
                if np.linalg.norm(error) < 1e-14:
                    break
                jacobian = np.zeros((3, 2))
                for column, index in enumerate((3, 5)):
                    step = np.zeros(6)
                    step[index] = 1e-7
                    plus = _rotation_error(_frames(model, q + step)[6][:3, :3], target[:3, :3])
                    minus = _rotation_error(_frames(model, q - step)[6][:3, :3], target[:3, :3])
                    jacobian[:, column] = (plus - minus) / 2e-7
                step_q = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
                q[3] += step_q[0]
                q[5] += step_q[1]
            candidates.append(q)
    # keep only the ones that actually solve it (the Newton starts are redundant by design)
    return [q for q in _unique(candidates, tol=1e-7) if _pose_error(model, q, target) < 1e-9]


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector taking ``current`` to ``target``."""
    spin = target @ current.T
    cosine = float(np.clip(0.5 * (np.trace(spin) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0], spin[1, 0] - spin[0, 1]]
    )
    return axis / (2.0 * np.sin(angle)) * angle


def _pose_error(model: RobotModelNumpy, q: np.ndarray, target: np.ndarray) -> float:
    """``max(position error, rotation error)`` of ``FK(q)`` against ``target``."""
    pose = model.fk(q)
    position = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
    rotation = float(np.linalg.norm(_rotation_error(pose[:3, :3], target[:3, :3])))
    return max(position, rotation)


def solve(model: RobotModelNumpy, target: np.ndarray) -> list[Sr0Solution]:
    """Every IK solution of the SR0 arm for one pose.

    Args:
        model: The SR0 model.
        target: Desired 4x4 tool pose.

    Returns:
        The solutions, each verified to :data:`TOLERANCE` against the forward kinematics.
        Empty when the pose is outside the reachable workspace.
    """
    solutions: list[Sr0Solution] = []
    for q123 in position_solutions(model, target):
        for q in _wrist_solutions(model, target, q123):
            pose = model.fk(q)
            position = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            rotation = float(np.linalg.norm(_rotation_error(pose[:3, :3], target[:3, :3])))
            if max(position, rotation) < TOLERANCE:
                values = np.asarray(q, dtype=float)
                admissible = bool(
                    np.all(values >= np.asarray(model.lower_bounds))
                    and np.all(values <= np.asarray(model.upper_bounds))
                )
                solutions.append(Sr0Solution(values, position, rotation, admissible))
    unique: list[Sr0Solution] = []
    for solution in solutions:
        if not any(
            np.linalg.norm(
                (solution.q - other.q + np.pi) % (2 * np.pi) - np.pi
            )
            < 1e-6
            for other in unique
        ):
            unique.append(solution)
    return unique


__all__ = [
    "SHOULDER_CENTRE",
    "TOLERANCE",
    "WRIST_IN_FLANGE",
    "Sr0Solution",
    "position_solutions",
    "robot",
    "solve",
    "sr0_links",
]
