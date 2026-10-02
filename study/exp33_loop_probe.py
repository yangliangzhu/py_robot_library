#!/usr/bin/env python3
"""exp33: why do the monodromy loops fail -- a per-sample probe of the loop tracker.

Task A-3 (hypotheses H1/H2) of the collaboration (``study/collab/day02-status.md``).  The loop
tracker of ``study/monodromy.py`` currently fails on all five loops.  This probe rebuilds its
base pose and fibre exactly (master seed 0, 600 census seeds), then tracks with per-sample
logging instead of a bare "failed":

* **H2 (position circle xy)** -- track until the first failure and dump the residual and
  clearance curves approaching it; at the failing pose, run a census.  If the real-solution
  count dropped or nothing sits near the last good configuration, the loop crossed the
  discriminant image and the branch died -- that is geometry (a fold), not a tracker bug.
* **H1 (orientation sweep x)** -- track every solution; for the one whose endpoint matches no
  known solution, verify the endpoint's pose error, and re-solve the base pose with the
  complete solver in scan mode: a genuine ninth solution the census missed, or an artifact?

Run::

    python3 -m study.exp33_loop_probe
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study import monodromy
from study.census import census, make_lm_solver
from study.complete_ik import solve_all


@dataclass
class Probe:
    """One track with its per-sample history."""

    ok: bool
    q: np.ndarray
    failed_at: int | None = None
    reason: str = ""
    history: list[tuple[int, float, float, float]] = field(default_factory=list)


def probe_track(model, q0: np.ndarray, poses: list[np.ndarray], *,
                sigma_floor: float, iterations: int = 40, tolerance: float = 1e-11,
                accept: float = 1e-6) -> Probe:
    """``monodromy.track_pose_path`` with per-sample logging and a last-good memory."""
    q = np.asarray(q0, dtype=float).copy()
    last_good = q.copy()
    history: list[tuple[int, float, float, float]] = []
    for index, target in enumerate(poses):
        before = q.copy()
        for _ in range(iterations):
            error = monodromy.pose_error(model, q, target)
            if float(np.linalg.norm(error)) < tolerance:
                break
            jacobian = monodromy.numeric_jacobian(model, q, target)
            normal = jacobian.T @ jacobian + 1e-8 * np.eye(len(q))
            q = q - np.linalg.solve(normal, jacobian.T @ error)
        residual = float(np.linalg.norm(monodromy.pose_error(model, q, target)))
        clearance = iks.sigma_min(model, q)
        history.append((index, residual, clearance, float(np.linalg.norm(q - before))))
        if residual > accept:
            return Probe(False, last_good, index, f"lost the pose (error {residual:.1e})",
                         history)
        if clearance < sigma_floor:
            return Probe(False, last_good, index, f"met the singular set ({clearance:.1e})",
                         history)
        last_good = q.copy()
    return Probe(True, q, history=history)


def _tail(history: list[tuple[int, float, float, float]], count: int = 6) -> str:
    """Format the last ``count`` samples as (pose, residual, sigma, |dq|)."""
    return "; ".join(f"p{i}: r={r:.1e} s={s:.1e} |dq|={d:.1e}"
                     for i, r, s, d in history[-count:])


def main() -> int:
    """Run the two probes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--probe-seeds", type=int, default=200,
                        help="census seeds at the failure pose")
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--amplitude", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.sigma_floor]
    print(f"base fibre reproduced: {len(solutions)} solutions "
          f"({fiber.converged}/{fiber.seeds} seeds)\n")

    # --- H2: position circle xy, first failure -------------------------------------------
    print("=== H2: position circle xy ===")
    poses = monodromy.position_loop(target, args.radius, first=0, second=1)
    for track_index, q in enumerate(solutions):
        probe = probe_track(model, q, poses, sigma_floor=args.sigma_floor)
        if probe.ok:
            print(f"  track {track_index}: survived, min clearance "
                  f"{min(s for _, _, s, _ in probe.history):.2e}")
            continue
        print(f"  track {track_index}: FAILED at pose {probe.failed_at}: {probe.reason}")
        print(f"    tail: {_tail(probe.history)}")
        at_failure = poses[probe.failed_at]
        recensus = census(model, at_failure, solver=make_lm_solver(model),
                          seeds=args.probe_seeds, rng=np.random.default_rng(7))
        distances = sorted(
            iks.configuration_distance(probe.q, other, circular)
            for other in recensus.solutions
        )
        print(f"    census at the failing pose: {len(recensus.solutions)} solutions "
              f"({recensus.converged}/{recensus.seeds} seeds); last good config is "
              f"{distances[0]:.3e} from the nearest one, clearance there "
              f"{iks.sigma_min(model, probe.q):.2e}")
        break

    # --- H1: orientation sweep x, the endpoint that matches nothing ----------------------
    print("\n=== H1: orientation sweep x ===")
    poses = monodromy.orientation_loop(target, args.amplitude, axis=0)
    for track_index, q in enumerate(solutions):
        probe = probe_track(model, q, poses, sigma_floor=args.sigma_floor)
        if not probe.ok:
            print(f"  track {track_index}: FAILED at pose {probe.failed_at}: {probe.reason}")
            print(f"    tail: {_tail(probe.history)}")
            continue
        distances = [iks.configuration_distance(probe.q, other, circular)
                     for other in solutions]
        nearest = int(np.argmin(distances))
        print(f"  track {track_index}: endpoint matches solution {nearest} "
              f"(distance {distances[nearest]:.1e})")
        if distances[nearest] > 1e-3:
            residual = float(np.linalg.norm(monodromy.pose_error(model, probe.q, target)))
            print(f"    MISMATCH: endpoint is a root of the base pose with residual "
                  f"{residual:.1e}, {distances[nearest]:.1e} away from every known solution")
            complete = solve_all(model, target, mode="scan", seeds=80, grid=5,
                                 rng=np.random.default_rng(11))
            matched = min(
                (iks.configuration_distance(probe.q, solution.q, circular)
                 for solution in complete),
                default=float("inf"),
            )
            print(f"    complete solver (scan mode): {len(complete)} solutions; "
                  f"the endpoint is {matched:.1e} from the nearest of them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
