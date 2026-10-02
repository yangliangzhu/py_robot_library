"""Solve a pose completely: every IK solution of the SR5, with the residuals to prove it.

The study's recipe, packaged as one call.  Three modes, all returning verified solutions:

* ``"census"`` -- multi-start Levenberg-Marquardt (about 2 s for 300 seeds).  The workhorse: complete
  on most poses, and the cheapest of the three that is reliable.
* ``"homotopy"`` -- SR0's closed-form fibre (eight solutions, no random seed) carried along
  ``link 5 y: 0 -> 0.136 m`` (about 0.5 s).  Fast, and complete when the straight path carries its
  branches; it returns few or no solutions when branches stall at a discriminant, which is a
  *result* of the method rather than a failure of the call (see NOTES 3.10).
* ``"scan"`` -- solve the intermediate arm on a grid of parameters and carry everything forward
  (about 30 s).  The complete version: it found solutions a 1000-seed census missed (NOTES 3.11), at
  15x the cost.

Every returned solution is checked against the forward kinematics, so a caller never has to trust the
solver: ``max(position_error, rotation_error) <= tolerance``, and duplicates are removed on the torus.

This module is deliberately in ``study/`` and not in the library: it hard-codes the SR5's structure
(link 5's y offset is what separates the arm from its Pieper-solvable neighbour), and the library's
four solvers are not modified from this branch (NOTES 3.4).

Usage::

    from study.complete_ik import solve_all

    solutions = solve_all(model, target)                 # census mode
    solutions = solve_all(model, target, mode="homotopy")  # fast, complete when unobstructed
    for solution in solutions:
        print(solution.source, solution.q, solution.error)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import IkType, ModelFactory
from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at

#: Acceptance for a returned solution, on ``max(position error, rotation error)``.
TOLERANCE = 1e-6


@dataclass
class Solution:
    """One IK solution and where it came from."""

    q: np.ndarray
    source: str
    position_error: float
    rotation_error: float

    @property
    def error(self) -> float:
        """The worst of the two pose errors."""
        return max(self.position_error, self.rotation_error)


def _verify(model, target: np.ndarray, q: np.ndarray, source: str) -> Solution | None:
    """Check a candidate against the forward kinematics."""
    pose = np.asarray(model.fk(q), dtype=float)
    position = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
    spin = target[:3, :3] @ pose[:3, :3].T
    cosine = float(np.clip(0.5 * (np.trace(spin) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if max(position, angle) > TOLERANCE:
        return None
    return Solution(np.asarray(q, dtype=float), source, position, angle)


def _deduplicate(solutions: list[Solution], circular: np.ndarray, match: float) -> list[Solution]:
    """Keep one representative per distinct configuration."""
    out: list[Solution] = []
    for solution in solutions:
        if not any(
            iks.configuration_distance(solution.q, other.q, circular) < match for other in out
        ):
            out.append(solution)
    return out


def solve_all(
    model,
    target: np.ndarray,
    *,
    mode: str = "census",
    seeds: int = 300,
    grid: int = 9,
    sigma_floor: float = 2e-3,
    match: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> list[Solution]:
    """Every IK solution of the SR5 for one pose.

    Args:
        model: The SR5 model (any backend; the J_dh path is built from the config on the numpy side).
        target: Desired 4x4 tool pose.
        mode: ``"census"``, ``"homotopy"`` or ``"scan"``.
        seeds: Multi-start seeds (census mode) or seeds per grid point (scan mode).
        grid: Parameter samples along the path (scan mode).
        sigma_floor: Clearance below which the tracker halves its step.
        match: Configuration distance below which two solutions count as one, in radians.
        rng: Random generator.

    Returns:
        The solutions, deduplicated, each verified to :data:`TOLERANCE`.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode not in {"census", "homotopy", "scan"}:
        raise ValueError(f"mode must be 'census', 'homotopy' or 'scan', got {mode!r}")
    rng = np.random.default_rng(0) if rng is None else rng
    circular = iks.circular_joints(model)
    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    sr0_model = sr0.robot()
    found: list[Solution] = []

    if mode == "census":
        fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
        for q in fiber.solutions:
            solution = _verify(model, target, q, "census")
            if solution is not None:
                found.append(solution)

    elif mode == "homotopy":
        for index, seed in enumerate(sr0.solve(sr0_model, target)):
            track = homotopy.track(path, target, seed.q, sigma_floor=sigma_floor)
            if not track.finished:
                continue
            solution = _verify(model, target, track.q, f"homotopy/seed{index}")
            if solution is not None:
                found.append(solution)

    else:  # scan
        for s in np.linspace(0.0, 1.0, grid):
            intermediate = sr0.robot(path(s))
            local = census(intermediate, target, solver=make_lm_solver(intermediate),
                           seeds=seeds, rng=rng)
            for q in local.solutions:
                shifted = path if s == 0.0 else _from(path, float(s))
                track = homotopy.track(shifted, target, q, sigma_floor=sigma_floor)
                if not track.finished:
                    continue
                solution = _verify(model, target, track.q, f"scan/s={s:.2f}")
                if solution is not None:
                    found.append(solution)

    return _deduplicate(found, circular, match)


def _from(path, s0: float):
    """The path restricted to ``[s0, 1]``, reparameterised to ``[0, 1]``."""
    def make(t: float) -> list[np.ndarray]:
        return path(s0 + t * (1.0 - s0))

    return make


def _demo(poses: int = 2) -> None:
    """Print a small table of the three modes, for a sanity check."""
    import time

    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    rng = np.random.default_rng(0)
    for pose in range(poses):
        target, _ = iks.random_reachable_target(model, rng)
        for mode in ("census", "homotopy", "scan"):
            started = time.perf_counter()
            solutions = solve_all(model, target, mode=mode, rng=np.random.default_rng(pose))
            seconds = time.perf_counter() - started
            worst = max((s.error for s in solutions), default=float("nan"))
            print(f"pose {pose} {mode:9s}: {len(solutions):2d} solutions in {seconds:5.1f} s "
                  f"| worst residual {worst:.1e}")


if __name__ == "__main__":
    _demo()
