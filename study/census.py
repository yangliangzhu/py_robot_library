"""Tools for enumerating the solutions of an inverse-kinematics problem.

Enumerating a fiber is the prerequisite for every branch question: you cannot ask whether two
solutions are in the same branch before you have both of them.  This module holds two things:

* :func:`solve_lm` -- a Levenberg-Marquardt solver written for *census* use rather than for
  servo use.  The library's :class:`~model.ik_solver.IkStandard` is deliberately conservative
  (adaptive step back-off, a 20-iteration budget, ``1e-5`` acceptance) because it runs inside a
  control loop; a census wants the opposite: converge from a bad seed if a solution is
  reachable at all, and converge *tightly* (``1e-9``) so that deduplication and analytic
  comparison are not fighting the solver's tolerance.  :mod:`study.exp00_solver_baselines`
  measures both, so the choice is a measurement rather than a preference.
* :func:`census` -- multi-start enumeration with deduplication on the configuration space
  (circles for the continuous joints, intervals for the limited ones), returning the
  :class:`~study.ik_structure.Fiber` used by the later experiments.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from study.ik_structure import (
    Fiber,
    circular_joints,
    configuration_distance,
    pose_error,
    random_reachable_target,
)


def solve_lm(
    model,
    seed: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 200,
    damping: float = 1e-3,
    step_cap: float = 0.5,
    tol: float = 1e-9,
) -> tuple[np.ndarray, bool]:
    """Solve one pose from one seed with Levenberg-Marquardt.

    Args:
        model: Robot model exposing ``fk``, ``jacobian``, ``se3_diff`` and the joint bounds.
        seed: Starting joint positions in radians, length ``num_dof``.
        target: Desired 4x4 end-effector pose.
        iterations: Maximum iterations.
        damping: Initial damping factor (added to ``J J^T`` as ``damping**2``).
        step_cap: Largest allowed joint step per iteration, in radians.
        tol: Convergence tolerance on ``max(position error, rotation error)``.

    Returns:
        ``(q, converged)``; ``q`` is clipped to the joint bounds either way.
    """
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    q = np.clip(np.asarray(seed, dtype=float), lower, upper)
    error = np.asarray(model.se3_diff(model.fk(q), target), dtype=float).ravel()
    cost = float(np.linalg.norm(error))
    lam = float(damping)
    for _ in range(iterations):
        if max(np.linalg.norm(error[:3]), np.linalg.norm(error[3:])) < tol:
            return q, True
        jac = np.asarray(model.jacobian(q), dtype=float)
        normal = jac @ jac.T + (lam**2) * np.eye(6)
        try:
            dq = jac.T @ np.linalg.solve(normal, error)
        except np.linalg.LinAlgError:  # pragma: no cover - the damping makes this singular-free
            break
        biggest = float(np.abs(dq).max())
        if biggest > step_cap:
            dq *= step_cap / biggest
        candidate = np.clip(q + dq, lower, upper)
        candidate_error = np.asarray(
            model.se3_diff(model.fk(candidate), target), dtype=float
        ).ravel()
        candidate_cost = float(np.linalg.norm(candidate_error))
        if candidate_cost < cost:
            q, error, cost = candidate, candidate_error, candidate_cost
            lam = max(lam * 0.7, 1e-9)
        else:
            lam = min(lam * 2.5, 1e3)
    converged = max(np.linalg.norm(error[:3]), np.linalg.norm(error[3:])) < tol
    return q, bool(converged)


def library_solver(model, accept: float = 1e-5) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, bool]]:
    """Wrap the model's configured solver, rejecting solutions above ``accept``.

    The library solvers report success against their own tolerance, so a census that wants a
    comparable success criterion has to re-check the pose error itself.
    """

    def solve(seed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        q, ok = model.ik(np.asarray(seed, dtype=float), target)
        if not ok:
            return q, False
        position, rotation = pose_error(model, q, target)
        return q, bool(max(position, rotation) <= accept)

    return solve


def make_lm_solver(model, **kwargs) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, bool]]:
    """Bind :func:`solve_lm` to a model, for use as a census solver."""

    def solve(seed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        return solve_lm(model, seed, target, **kwargs)

    return solve


def census(
    model,
    target: np.ndarray,
    *,
    solver: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, bool]],
    seeds: int = 512,
    rng: np.random.Generator | None = None,
    dedupe_tol: float = 1e-4,
    seed_scale: float = 1.0,
    seed_center: np.ndarray | None = None,
    circular: np.ndarray | None = None,
) -> Fiber:
    """Enumerate the solutions of one pose by multi-start.

    Args:
        model: Robot model; its joint bounds are the seed distribution's support.
        target: The 4x4 pose to solve for.
        solver: ``(seed, target) -> (q, converged)``, e.g. :func:`make_lm_solver` or
            :func:`library_solver`.
        seeds: How many restarts.
        rng: Random generator (pass a seeded one for reproducibility).
        dedupe_tol: Solutions closer than this in the configuration space are one solution.
        seed_scale: Seeds are drawn uniformly over ``center +- scale * (bound - center)``.
        seed_center: Centre of the seed box; defaults to the middle of the joint ranges.
        circular: Boolean mask of continuous joints; computed from the model when omitted.

    Returns:
        The :class:`~study.ik_structure.Fiber`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    circular = circular_joints(model) if circular is None else circular
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    center = 0.5 * (lower + upper) if seed_center is None else np.asarray(seed_center, dtype=float)
    half = 0.5 * (upper - lower) * seed_scale
    fiber = Fiber(target=np.asarray(target, dtype=float), seeds=seeds)
    for _ in range(seeds):
        seed = np.clip(center + rng.uniform(-half, half), lower, upper)
        q, ok = solver(seed, target)
        if not ok:
            continue
        fiber.converged += 1
        position, rotation = pose_error(model, q, target)
        for index, known in enumerate(fiber.solutions):
            if configuration_distance(q, known, circular) < dedupe_tol:
                fiber.hits[index] += 1
                break
        else:
            fiber.solutions.append(np.asarray(q, dtype=float))
            fiber.hits.append(1)
            fiber.errors.append((position, rotation))
    return fiber


def evaluate_solvers(
    model,
    *,
    solver: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, bool]],
    rng: np.random.Generator,
    seeds: int = 400,
    perturbations: Sequence[float] = (0.05, 0.2, 0.5, 1.0),
    tolerance: float = 1e-6,
    perturbed_samples: int = 200,
) -> dict[str, object]:
    """Measure how a solver behaves as a census tool.

    Two numbers matter and they are different: how often it converges from a seed anywhere in
    the joint range (can it *find* solutions?), and how far a known solution's basin of
    attraction reaches (how close must a seed be?).  The second is what a continuation-based
    census exploits and what a random-restart census has to pay for.

    Args:
        model: Robot model.
        solver: ``(seed, target) -> (q, converged)``.
        rng: Random generator.
        seeds: Random restarts for the first number.
        perturbations: Standard deviations, in radians, of the seeds around a known solution.
        tolerance: Pose error a solve must reach to count as converged.
        perturbed_samples: Restarts per perturbation radius.

    Returns:
        ``{"random_rate", "seconds", "perturbed"}`` with ``perturbed`` a list of
        ``(radius, rate-of-returning-to-the-known-solution)``.
    """
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    circular = circular_joints(model)
    target, known = random_reachable_target(model, rng)

    started = time.perf_counter()
    converged = 0
    for _ in range(seeds):
        q, ok = solver(rng.uniform(lower, upper), target)
        if ok:
            position, rotation = pose_error(model, q, target)
            converged += int(max(position, rotation) <= tolerance)
    seconds = time.perf_counter() - started

    perturbed: list[tuple[float, float]] = []
    for radius in perturbations:
        hits = 0
        for _ in range(perturbed_samples):
            seed = known + rng.normal(0.0, radius, size=known.size)
            q, ok = solver(seed, target)
            if ok and configuration_distance(q, known, circular) < 1e-3:
                hits += 1
        perturbed.append((float(radius), hits / perturbed_samples))
    return {"random_rate": converged / seeds, "seconds": seconds, "perturbed": perturbed}


def perturbation_sweep(
    model,
    *,
    solver: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, bool]],
    rng: np.random.Generator,
    radii: Sequence[float] = (1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1),
    samples: int = 200,
    tolerance: float = 1e-6,
    same_solution_tol: float = 1e-3,
) -> list[tuple[float, float, float, int, float]]:
    """How a solver behaves when it starts near a known solution.

    Two success rates, because they are different questions and a control loop cares about the
    second one:

    * **any**: did it converge to *some* solution of the pose?
    * **same**: did it converge to *the* solution the seed was perturbed around?  A solver that
      scores high on the first and low on the second is jumping between branches, which is
      invisible if you only measure pose error -- every returned solution is "correct" -- and
      fatal if you are tracking a branch, e.g. keeping a configuration stable across a
      trajectory.

    Args:
        model: Robot model.
        solver: ``(seed, target) -> (q, converged)``.
        rng: Random generator.
        radii: Seed perturbation sizes, in radians.
        samples: Restarts per radius.
        tolerance: Pose error a solve must reach to count.
        same_solution_tol: Configuration-space distance within which two solutions are one.

    Returns:
        ``[(radius, any rate, same rate, distinct solutions seen, median pose error)]``.
    """
    circular = circular_joints(model)
    target, known = random_reachable_target(model, rng)
    rows: list[tuple[float, float, float, int, float]] = []
    for radius in radii:
        any_hits, same_hits, errors = 0, 0, []
        reached: list[np.ndarray] = []
        for _ in range(samples):
            seed = known + rng.normal(0.0, radius, size=known.size)
            q, ok = solver(seed, target)
            position, rotation = pose_error(model, q, target)
            errors.append(max(position, rotation))
            if not (ok and max(position, rotation) <= tolerance):
                continue
            any_hits += 1
            if configuration_distance(q, known, circular) < same_solution_tol:
                same_hits += 1
            elif not any(configuration_distance(q, other, circular) < same_solution_tol for other in reached):
                reached.append(np.asarray(q, dtype=float))
        rows.append(
            (float(radius), any_hits / samples, same_hits / samples, len(reached), float(np.median(errors)))
        )
    return rows
