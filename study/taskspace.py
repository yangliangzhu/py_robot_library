"""Continuation along a *task* path: can a prescribed Cartesian path be tracked, and by how many
postures at once?

``study/homotopy.py`` moves the *robot* (the DH parameters) while the task stays fixed.  This module
does the dual: the robot is fixed and the **task** (a 6-DOF pose for SR5, a 3-vector position for
the 3R of :mod:`study.cuspidal3r`) walks along a prescribed path.  That is the continuation a robot
controller actually performs, and it is the instrument the region-of-feasible-paths theory is
written in terms of:

* a *lift* of the task path from a given posture exists exactly while the posture's branch survives
  the crossing of the discriminant image -- so tracking one solution answers "can this path be
  tracked from here?";
* tracking **all** solutions in lockstep answers the finer question, because the discriminant image
  is crossed exactly when two of the tracked branches *collide*: at a real fold of the task map two
  real solutions coalesce (one in each of the two aspects bounded by that piece of the singular
  set) and one of them ceases to be real.  A collision that involves the lifted branch is a
  characteristic-surface crossing -- the event that separates two *uniqueness domains* inside one
  aspect (`study/collab/QUESTIONS.md` Q18).

The tracker is deliberately observable: every step is corrected to a tolerance, the interval is
bisected (in task space) when the corrector fails, the smallest clearance seen is reported, and a
track that cannot be continued says why instead of returning a wrong configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: A task map: configuration -> task vector, and its Jacobian (task x dof).
TaskMap = Callable[[np.ndarray], np.ndarray]
TaskJacobian = Callable[[np.ndarray], np.ndarray]
#: A residual: (configuration, target) -> error vector whose norm the corrector drives to zero.
TaskResidual = Callable[[np.ndarray, np.ndarray], np.ndarray]
#: A target interpolation: (target_a, target_b, fraction) -> target.
TaskInterpolation = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def se3_interpolate(target_a: np.ndarray, target_b: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate two 4x4 poses along a rigid motion (slerp in rotation, linear in position).

    Needed because a corrector that bisects a pose interval must *produce poses*: a linear
    combination of two rotation matrices is not a rotation, and feeding it to a pose residual
    makes the residual meaningless.  The rotation is composed from the world-frame rotation vector
    that :func:`study.homotopy.se3_error` reports, so interpolation and residual agree, and at
    ``fraction`` 1 the result is ``target_b`` exactly.
    """
    target_a = np.asarray(target_a, dtype=float)
    target_b = np.asarray(target_b, dtype=float)
    out = np.eye(4)
    spin = target_b[:3, :3] @ target_a[:3, :3].T
    cosine = float(np.clip(0.5 * (np.trace(spin) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        out[:3, :3] = target_a[:3, :3]
    else:
        axis = np.array([spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0],
                         spin[1, 0] - spin[0, 1]]) / (2.0 * np.sin(angle))
        out[:3, :3] = _exp_so3(axis * (fraction * angle))[:3, :3] @ target_a[:3, :3]
    out[:3, 3] = target_a[:3, 3] + fraction * (target_b[:3, 3] - target_a[:3, 3])
    return out


def _exp_so3(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues' formula as a 4x4 rotation."""
    theta = float(np.linalg.norm(rotation_vector))
    out = np.eye(4)
    if theta < 1e-12:
        return out
    axis = np.asarray(rotation_vector, dtype=float) / theta
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    out[:3, :3] = rotation
    return out


def newton(
    task: TaskMap, jacobian: TaskJacobian, target: np.ndarray, q0: np.ndarray, *,
    iterations: int = 40, tolerance: float = 1e-10, damping: float = 1e-12,
    residual: TaskResidual | None = None,
) -> tuple[np.ndarray, float]:
    """Damped Gauss-Newton on ``task(q) = target`` from a seed.

    Args:
        task: Task map.
        jacobian: Its Jacobian, ``(m, n)``.
        target: Task vector to reach.
        q0: Starting configuration.
        iterations: Maximum iterations.
        tolerance: Residual that counts as converged.
        damping: Tikhonov term for the normal equations.
        residual: Error function ``(q, target) -> vector``.  When omitted the error is
            ``target - task(q)``, which is only correct if the task map *is* the identity chart on
            the task space (true for a position).  For poses the error must be measured *relative
            to the current pose* (e.g. ``se3_error(FK(q), target)``), otherwise the correction
            direction is wrong by a pose-dependent linear map and the tracker stalls.

    Returns:
        ``(q, residual)``, residual being the Euclidean task error.
    """
    target = np.asarray(target, dtype=float)
    q = np.asarray(q0, dtype=float).copy()

    def error_at(configuration: np.ndarray) -> np.ndarray:
        if residual is not None:
            return np.asarray(residual(configuration, target), dtype=float)
        return target - task(configuration)

    error = error_at(q)
    for _ in range(iterations):
        if float(np.linalg.norm(error)) < tolerance:
            break
        matrix = jacobian(q)
        normal = matrix @ matrix.T + damping * np.eye(matrix.shape[0])
        q = q + matrix.T @ np.linalg.solve(normal, error)
        error = error_at(q)
    return q, float(np.linalg.norm(error_at(q)))


@dataclass
class Lift:
    """One posture's journey along a task path."""

    start: np.ndarray
    q: np.ndarray
    finished: bool
    #: Index of the last target sample reached (``len(targets) - 1`` when finished).
    index: int
    residual: float
    #: Smallest clearance seen and where, in samples.
    min_sigma: float
    min_sigma_at: int
    #: Configurations at every *sample* of the path that the lift reached (not every substep).
    samples: list[np.ndarray] = field(default_factory=list)
    #: Corrected steps rejected as basin jumps (the instrument fault this tracker guards against).
    jumps: int = 0
    reason: str = ""
    #: Sample at which the track entered the lift set; 0 for a track started at ``targets[0]``, and
    #: the birth sample for one admitted mid-path by :func:`lift_fiber`'s ``admit`` hook.
    born: int = 0
    #: Set when the track was dropped as a duplicate: two live tracks coincided away from the
    #: discriminant, which the fibre cannot do, so one of them is a jump artefact.
    retired: bool = False

    @property
    def progress(self) -> float:
        """Fraction of the path reached, in ``[0, 1]`` (by sample index)."""
        return float(self.index)


def lift(
    task: TaskMap, jacobian: TaskJacobian, targets: np.ndarray, q0: np.ndarray, *,
    sigma: Callable[[np.ndarray], float] | None = None, sigma_floor: float = 0.0,
    tolerance: float = 1e-9, min_division: float = 1e-4, jump_tolerance: float = 0.3,
    max_step: float = 0.5, residual: TaskResidual | None = None,
    interpolate: TaskInterpolation | None = None,
    feasible: Callable[[np.ndarray], bool] | None = None,
) -> Lift:
    """Carry one solution along ``targets``, bisecting the task interval when a step fails.

    Args:
        task: Task map.
        jacobian: Its Jacobian.
        targets: ``(K, m)`` array of task vectors; the path is the polyline through them.
        q0: A solution at ``targets[0]``.
        sigma: Clearance function; a step is rejected below ``sigma_floor``.
        sigma_floor: Rejection threshold for the clearance.
        tolerance: Task residual a step must reach.
        min_division: Smallest task-space fraction of an interval before the lift is declared dead.
        jump_tolerance: Largest distance between the predicted and corrected configuration that
            still counts as following the branch rather than jumping to a neighbour.
        max_step: Largest joint-space step the corrector may take; longer proposals are bisected.
        residual: Error function ``(q, target) -> vector``; see :func:`newton`.
        interpolate: Target interpolation ``(a, b, fraction)``; see :func:`se3_interpolate`.
        feasible: Predicate a corrected configuration must satisfy, e.g. "inside the joint box".
            Without it the tracker can walk a limited joint past its stop, and the configuration it
            reports is a solution of the pose but not one the arm can be in.

    Returns:
        The :class:`Lift`; ``finished`` is the certificate that the whole path was tracked.
    """
    targets = np.asarray(targets, dtype=float)
    sigma = (lambda q: 0.0) if sigma is None else sigma
    result = Lift(start=np.asarray(q0, dtype=float).copy(), q=np.asarray(q0, dtype=float).copy(),
                  finished=False, index=0, residual=float("nan"),
                  min_sigma=float(sigma(q0)), min_sigma_at=0, samples=[np.asarray(q0, float)])
    for index in range(1, targets.shape[0]):
        q, ok, clearance, jumps = _interval(task, jacobian, targets[index - 1], targets[index],
                                            result.q, sigma, sigma_floor, tolerance, min_division,
                                            jump_tolerance, max_step, residual, interpolate,
                                            feasible)
        result.jumps += jumps
        if not ok:
            result.reason = (
                f"died between samples {index - 1} and {index} "
                f"(best residual {clearance:.1e} clearance, step limit or no admissible fraction)"
            )
            return result
        result.q = q
        result.index = index
        result.residual = float(np.linalg.norm(
            targets[index] - task(q) if residual is None else residual(q, targets[index])))
        if clearance < result.min_sigma:
            result.min_sigma = clearance
            result.min_sigma_at = index
        result.samples.append(q.copy())
    result.finished = True
    result.reason = "arrived"
    return result


def _interval(task, jacobian, target_a, target_b, q_start, sigma, sigma_floor, tolerance,
              min_division, jump_tolerance, max_step, residual=None, interpolate=None,
              feasible=None) -> tuple[np.ndarray, bool, float, int]:
    """Advance from ``target_a`` to ``target_b``, bisecting when the corrector cannot follow.

    The predictor is the **tangent** step ``dq = pinv(J) (target_b - target_a)``, not the previous
    configuration: seeding the corrector with the previous configuration alone lets Newton fall
    into a neighbouring basin near a fold, which is how a fibre tracker silently permutes a fibre
    it was supposed to follow (`study/collab/QUESTIONS.md` Q17, and measured again while building
    this module: two tracks merged to ``6e-15`` rad).  A correction that lands further from the
    prediction than ``jump_tolerance`` is rejected as a jump, and the interval is bisected.

    Returns:
        ``(q, ok, clearance, jumps)`` -- ``jumps`` counts the rejected basin jumps.
    """
    lo, hi = 0.0, 1.0
    q = np.asarray(q_start, dtype=float).copy()
    best_residual = float("inf")
    best_clearance = float("inf")
    jumps = 0
    while hi - lo > min_division:
        mid = 0.5 * (lo + hi)
        trial_target = (interpolate(target_a, target_b, mid) if interpolate is not None
                        else target_a + mid * (target_b - target_a))
        matrix = jacobian(q)
        if residual is None:
            error = trial_target - task(q)
        else:
            error = np.asarray(residual(q, trial_target), dtype=float)
        predicted = q + matrix.T @ np.linalg.solve(
            matrix @ matrix.T + 1e-12 * np.eye(matrix.shape[0]), error
        )
        step_length = float(np.linalg.norm(predicted - q))
        trial, residual_value = newton(task, jacobian, trial_target, predicted,
                                       tolerance=tolerance, residual=residual)
        clearance = float(sigma(trial))
        drifted = float(np.linalg.norm(trial - predicted))
        if (residual_value <= tolerance and clearance >= sigma_floor
                and drifted <= jump_tolerance and step_length <= max_step
                and (feasible is None or feasible(trial))):
            lo = mid
            q = trial
            best_residual = residual_value
            best_clearance = clearance
            continue
        if residual_value <= tolerance and clearance >= sigma_floor:
            jumps += 1
        best_residual = min(best_residual, residual_value)
        best_clearance = min(best_clearance, clearance)
        hi = mid
    if lo <= 0.0:
        return q, False, best_clearance, jumps
    # finish the interval with the last accepted configuration carried to the far end
    final, final_residual = newton(task, jacobian, target_b, q, tolerance=tolerance,
                                   residual=residual)
    clearance = float(sigma(final))
    if (final_residual <= tolerance and clearance >= sigma_floor
            and (feasible is None or feasible(final))):
        return final, True, clearance, jumps
    if lo < 1.0:
        return q, False, best_clearance, jumps
    return final, False, clearance, jumps


@dataclass
class FiberLift:
    """The whole fibre tracked along one task path, with the collisions that mark the discriminant."""

    lifts: list[Lift]
    #: Minimum pairwise distance between live tracks at each sample index.
    gaps: list[float]
    #: Sample indices where two live tracks came closer than ``collision``.
    collisions: list[int]
    #: Sample indices where two live tracks came closer than ``merge``: a tracker fault, not a fold.
    merged: list[int]
    #: Number of live tracks at each sample index.
    counts: list[int]
    collision_threshold: float
    #: Indices of the two tracks attaining the minimum gap at each sample (``None`` if fewer than
    #: two are live).  A gap that stays tiny for many samples is a tracker fault, not a fold, and
    #: saying *which* pair it is decides between the two readings.
    closest: list[tuple[int, int] | None]
    #: Tracks dropped as duplicates, as ``(sample, kept track, retired track)``.
    duplicates: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def arrivals(self) -> list[int]:
        """Indices of the tracks that reached the end."""
        return [i for i, item in enumerate(self.lifts) if item.finished]


def lift_fiber(
    task: TaskMap, jacobian: TaskJacobian, targets: np.ndarray, starts, *,
    sigma: Callable[[np.ndarray], float] | None = None, sigma_floor: float = 0.0,
    tolerance: float = 1e-9, min_division: float = 1e-4, collision: float = 5e-3,
    merge: float = 1e-6, jump_tolerance: float = 0.3, max_step: float = 0.5,
    residual: TaskResidual | None = None, interpolate: TaskInterpolation | None = None,
    distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
    admit: Callable[[int, list[np.ndarray]], list[np.ndarray]] | None = None,
    feasible: Callable[[np.ndarray], bool] | None = None,
    prune_merged: bool = False,
) -> FiberLift:
    """Track every solution in ``starts`` along ``targets`` in lockstep.

    Lockstep matters: the collision report compares positions at the *same* sample, so a lift that
    cannot keep up must not silently shift the comparison.  A track that dies stays dead (the
    configuration is not advanced), which is what a real fold does to a real branch.

    ``admit`` lifts the tracker's structural blindness: a forward lift can only follow what it
    already carries, so a pair of branches that becomes real *between* two samples (a birth at a
    fold) is invisible to it, and a branch the corrector dropped is not recovered either.  When
    given, ``admit(index, alive)`` is called at every sample with the configurations alive there and
    returns configurations at that same target to be *added* as new tracks (each gets
    ``Lift.born = index``); it is the caller's job to verify that the returned configurations are
    real solutions there.  Counts, gaps and collisions from that sample on include the new tracks.

    Args:
        task: Task map.
        jacobian: Its Jacobian.
        targets: ``(K, m)`` task path.
        starts: Solutions at ``targets[0]``.
        sigma: Clearance function.
        sigma_floor: Rejection threshold for the clearance.
        tolerance: Task residual a step must reach.
        min_division: Smallest interval fraction before a track is declared dead.
        collision: Distance below which two live tracks count as colliding.
        merge: Distance below which two live tracks count as *merged* -- a tracker fault, since
            two distinct branches cannot coincide away from the discriminant.
        jump_tolerance: Passed to the corrector's jump guard.
        max_step: Largest joint-space step the corrector may take.
        residual: Error function ``(q, target) -> vector``; see :func:`newton`.
        interpolate: Target interpolation ``(a, b, fraction)``; see :func:`se3_interpolate`.
        distance: Distance between configurations; Euclidean when omitted.
        admit: Optional hook admitting solutions found mid-path; see above.
        feasible: Predicate every corrected configuration must satisfy (e.g. inside the joint box);
            see :func:`lift`.  Without it a track can walk a limited joint past its stop.
        prune_merged: Drop a live track that coincides with an older live track.  Two distinct
            branches cannot be the same configuration away from the discriminant, so a coincidence
            means the newer track is a jump artefact -- the fault a fold crossing produces when the
            corrector follows the *other* real solution instead of dying.  Retired tracks stop being
            advanced, leave the live counts, and are listed in :attr:`FiberLift.duplicates`.

    Returns:
        The :class:`FiberLift`.
    """
    targets = np.asarray(targets, dtype=float)
    distance = (lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))) \
        if distance is None else distance
    lifts = [
        Lift(start=np.asarray(q, dtype=float).copy(), q=np.asarray(q, dtype=float).copy(),
             finished=False, index=0, residual=0.0,
             min_sigma=0.0 if sigma is None else float(sigma(q)), min_sigma_at=0,
             samples=[np.asarray(q, float)])
        for q in starts
    ]
    gaps: list[float] = []
    collisions: list[int] = []
    merged: list[int] = []
    counts: list[int] = []
    closest: list[tuple[int, int] | None] = []
    duplicates: list[tuple[int, int, int]] = []
    for index in range(1, targets.shape[0]):
        for item in lifts:
            if (not item.finished and not item.retired and item.index == index - 1):
                q, ok, clearance, jumps = _interval(
                    task, jacobian, targets[index - 1], targets[index], item.q,
                    (lambda q: 0.0) if sigma is None else sigma, sigma_floor, tolerance,
                    min_division, jump_tolerance, max_step, residual, interpolate, feasible)
                item.jumps += jumps
                if ok:
                    item.q = q
                    item.index = index
                    item.samples.append(q.copy())
                    item.residual = float(np.linalg.norm(
                        targets[index] - task(q) if residual is None
                        else residual(q, targets[index])))
                    item.min_sigma = min(item.min_sigma, clearance)
                else:
                    item.reason = f"died between samples {index - 1} and {index}"
        alive = [i for i, item in enumerate(lifts) if item.index == index and not item.retired]
        if admit is not None:
            for q in admit(index, [lifts[i].q for i in alive]):
                q = np.asarray(q, dtype=float)
                lifts.append(Lift(
                    start=q.copy(), q=q.copy(), finished=False, index=index, residual=0.0,
                    min_sigma=0.0 if sigma is None else float(sigma(q)), min_sigma_at=index,
                    samples=[q.copy()], born=index))
            alive = [i for i, item in enumerate(lifts) if item.index == index and not item.retired]
        if prune_merged:
            kept: list[int] = []
            for i in alive:
                holder = next((k for k in kept if distance(lifts[k].q, lifts[i].q) < merge), None)
                if holder is None:
                    kept.append(i)
                else:
                    lifts[i].retired = True
                    duplicates.append((index, holder, i))
            alive = kept
        counts.append(len(alive))
        best = float("inf")
        nearest = None
        for a in range(len(alive)):
            for b in range(a + 1, len(alive)):
                gap = distance(lifts[alive[a]].q, lifts[alive[b]].q)
                if gap < best:
                    best = gap
                    nearest = (alive[a], alive[b])
        gaps.append(best)
        closest.append(nearest)
        if best < collision:
            collisions.append(index)
        if best < merge:
            merged.append(index)
    for item in lifts:
        if item.retired:
            continue
        if item.index == targets.shape[0] - 1:
            item.finished = True
            item.reason = item.reason or "arrived"
    return FiberLift(lifts=lifts, gaps=gaps, collisions=collisions, merged=merged, counts=counts,
                     collision_threshold=collision, closest=closest, duplicates=duplicates)


def circle_targets(centre: np.ndarray, radius: float, samples: int, *, axis_a: int = 0,
                   axis_b: int = 2) -> np.ndarray:
    """A closed polyline circle in the plane of two task coordinates.

    Args:
        centre: Task vector at the circle's centre.
        radius: Circle radius in task units.
        samples: Number of samples; the first is repeated at the end to close the loop.
        axis_a: First task coordinate of the plane.
        axis_b: Second task coordinate of the plane.

    Returns:
        A ``(samples + 1, m)`` array of task vectors.
    """
    angles = np.linspace(0.0, 2 * np.pi, samples + 1)
    out = np.tile(np.asarray(centre, dtype=float), (angles.size, 1))
    out[:, axis_a] += radius * np.cos(angles)
    out[:, axis_b] += radius * np.sin(angles)
    return out


__all__ = [
    "FiberLift",
    "Lift",
    "TaskJacobian",
    "TaskMap",
    "circle_targets",
    "lift",
    "lift_fiber",
    "newton",
]
