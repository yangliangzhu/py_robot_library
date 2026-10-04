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


def newton(
    task: TaskMap, jacobian: TaskJacobian, target: np.ndarray, q0: np.ndarray, *,
    iterations: int = 40, tolerance: float = 1e-10, damping: float = 1e-12,
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

    Returns:
        ``(q, residual)``, residual being the Euclidean task error.
    """
    target = np.asarray(target, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    error = target - task(q)
    for _ in range(iterations):
        residual = float(np.linalg.norm(error))
        if residual < tolerance:
            break
        matrix = jacobian(q)
        normal = matrix @ matrix.T + damping * np.eye(matrix.shape[0])
        q = q + matrix.T @ np.linalg.solve(normal, error)
        error = target - task(q)
    return q, float(np.linalg.norm(target - task(q)))


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

    @property
    def progress(self) -> float:
        """Fraction of the path reached, in ``[0, 1]`` (by sample index)."""
        return float(self.index)


def lift(
    task: TaskMap, jacobian: TaskJacobian, targets: np.ndarray, q0: np.ndarray, *,
    sigma: Callable[[np.ndarray], float] | None = None, sigma_floor: float = 0.0,
    tolerance: float = 1e-9, min_division: float = 1e-4, jump_tolerance: float = 0.3,
    max_step: float = 0.5,
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
                                            jump_tolerance, max_step)
        result.jumps += jumps
        if not ok:
            result.reason = (
                f"died between samples {index - 1} and {index} "
                f"(best residual {clearance:.1e} clearance, step limit or no admissible fraction)"
            )
            return result
        result.q = q
        result.index = index
        result.residual = float(np.linalg.norm(targets[index] - task(q)))
        if clearance < result.min_sigma:
            result.min_sigma = clearance
            result.min_sigma_at = index
        result.samples.append(q.copy())
    result.finished = True
    result.reason = "arrived"
    return result


def _interval(task, jacobian, target_a, target_b, q_start, sigma, sigma_floor, tolerance,
              min_division, jump_tolerance, max_step) -> tuple[np.ndarray, bool, float, int]:
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
        trial_target = target_a + mid * (target_b - target_a)
        matrix = jacobian(q)
        error = trial_target - task(q)
        predicted = q + matrix.T @ np.linalg.solve(
            matrix @ matrix.T + 1e-12 * np.eye(matrix.shape[0]), error
        )
        step_length = float(np.linalg.norm(predicted - q))
        trial, residual = newton(task, jacobian, trial_target, predicted, tolerance=tolerance)
        clearance = float(sigma(trial))
        drifted = float(np.linalg.norm(trial - predicted))
        if (residual <= tolerance and clearance >= sigma_floor and drifted <= jump_tolerance
                and step_length <= max_step):
            lo = mid
            q = trial
            best_residual = residual
            best_clearance = clearance
            continue
        if residual <= tolerance and clearance >= sigma_floor:
            jumps += 1
        best_residual = min(best_residual, residual)
        best_clearance = min(best_clearance, clearance)
        hi = mid
    if lo <= 0.0:
        return q, False, best_clearance, jumps
    # finish the interval with the last accepted configuration carried to the far end
    final, residual = newton(task, jacobian, target_b, q, tolerance=tolerance)
    clearance = float(sigma(final))
    if residual <= tolerance and clearance >= sigma_floor:
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

    @property
    def arrivals(self) -> list[int]:
        """Indices of the tracks that reached the end."""
        return [i for i, item in enumerate(self.lifts) if item.finished]


def lift_fiber(
    task: TaskMap, jacobian: TaskJacobian, targets: np.ndarray, starts, *,
    sigma: Callable[[np.ndarray], float] | None = None, sigma_floor: float = 0.0,
    tolerance: float = 1e-9, min_division: float = 1e-4, collision: float = 5e-3,
    merge: float = 1e-6, jump_tolerance: float = 0.3, max_step: float = 0.5,
    distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> FiberLift:
    """Track every solution in ``starts`` along ``targets`` in lockstep.

    Lockstep matters: the collision report compares positions at the *same* sample, so a lift that
    cannot keep up must not silently shift the comparison.  A track that dies stays dead (the
    configuration is not advanced), which is what a real fold does to a real branch.

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
        distance: Distance between configurations; Euclidean when omitted.

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
    for index in range(1, targets.shape[0]):
        for item in lifts:
            if not item.finished and item.index == index - 1:
                q, ok, clearance, jumps = _interval(
                    task, jacobian, targets[index - 1], targets[index], item.q,
                    (lambda q: 0.0) if sigma is None else sigma, sigma_floor, tolerance,
                    min_division, jump_tolerance, max_step)
                item.jumps += jumps
                if ok:
                    item.q = q
                    item.index = index
                    item.samples.append(q.copy())
                    item.residual = float(np.linalg.norm(targets[index] - task(q)))
                    item.min_sigma = min(item.min_sigma, clearance)
                else:
                    item.reason = f"died between samples {index - 1} and {index}"
        alive = [item.q for item in lifts if item.index == index]
        counts.append(len(alive))
        best = float("inf")
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                best = min(best, distance(alive[i], alive[j]))
        gaps.append(best)
        if best < collision:
            collisions.append(index)
        if best < merge:
            merged.append(index)
    for item in lifts:
        if item.index == targets.shape[0] - 1:
            item.finished = True
            item.reason = item.reason or "arrived"
    return FiberLift(lifts=lifts, gaps=gaps, collisions=collisions, merged=merged, counts=counts,
                     collision_threshold=collision)


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
