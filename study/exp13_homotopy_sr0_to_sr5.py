#!/usr/bin/env python3
"""exp13: the SR0 -> SR5 homotopy -- how many of the arm's solutions does it reach?

Finding 2's method, end to end:

1. solve the Pieper-solvable neighbour (**SR0** = the SR5 with link 5's y offset set to zero) with
   the deterministic complete solver of `study/sr0.py` -- eight solutions, no random seeds;
2. carry each of those eight along the parameter path ``link 5 y: 0 -> 0.136 m`` with the
   predictor-corrector tracker of `study/homotopy.py`, which shrinks its step where the clearance
   drops (the measured failure mode of the predictor);
3. compare what arrives at the real arm (**SR5**) with a multi-start numerical census of the same
   pose, which knows nothing about SR0.

The interesting numbers are the ones that disagree: solutions the homotopy reaches that a
400-seed census misses, census solutions the homotopy never reaches (those live across a
discriminant, where branches meet and exchange -- the case a tracker cannot cross without
branch switching), and how close to the singular set each path had to travel.

Run::

    python3 -m study.exp13_homotopy_sr0_to_sr5 --poses 3 --seeds 800
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms, parameter_vector


def path_at(offset: float):
    """A parameter path: SR0 at ``offset = 0``, the SR5 at ``offset = 0.136``."""

    def make(s: float) -> list[np.ndarray]:
        links = link_transforms()
        links[4] = links[4].copy()
        links[4][1, 3] = float(offset) * s
        return links

    return make


def links_from_values(values: np.ndarray) -> list[np.ndarray]:
    """Link transforms from a vector of the 18 translation components."""
    links = link_transforms()
    for index, value in enumerate(np.asarray(values, dtype=float)):
        link_index, row = divmod(index, 3)
        links[link_index] = links[link_index].copy()
        links[link_index][row, 3] = float(value)
    return links


def curved_path(offset: float, direction: np.ndarray, amplitude: float):
    """A generic path: the straight SR0 -> SR5 path plus a bump that vanishes at both ends.

    A straight path in parameter space meets the discriminant for some branches, and there the
    solution genuinely stops existing along *that* path.  The discriminant has codimension one, so
    a detour -- a bump that is zero at ``s = 0`` and ``s = 1`` and perturbs the other parameters in
    between -- reaches the same endpoints by a different route.  This is the standard "generic
    path" trick, and with it the question stops being "does the straight path work?" and becomes
    "is the solution reachable by *some* path?".
    """
    base = parameter_vector(link_transforms())
    delta = np.zeros_like(base)
    delta[4 * 3 + 1] = offset

    def make(s: float) -> list[np.ndarray]:
        values = base + delta * s + amplitude * np.sin(np.pi * s) * direction
        return links_from_values(values)

    return make


def main() -> int:
    """Run the homotopy and the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=800, help="census seeds per pose")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match", type=float, default=1e-4, help="matching tolerance, radians")
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--retries", type=int, default=0,
                        help="generic curved-path retries per stopped branch")
    parser.add_argument("--amplitude", type=float, default=0.02,
                        help="size of the bump in the curved path, metres")
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(path_at(offset)(1.0))
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.seed)
    print(f"SR0 -> SR5 along link 5 y: 0 -> {offset:.4f} m, sigma_floor {args.sigma_floor:g}")

    total_arrived = total_census = 0
    unreached_total = unreached_poses = 0
    for pose in range(args.poses):
        q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
        target = sr5_model.fk(q_true)

        # the seeds: SR0's fibre, solved deterministically
        seeds = sr0.solve(sr0_model, target)
        started = time.perf_counter()
        straight = path_at(offset)
        tracks = [homotopy.track(straight, target, s.q, sigma_floor=args.sigma_floor)
                  for s in seeds]
        retried = 0
        if args.retries:
            for seed, first in zip(seeds, tracks):
                if first.finished:
                    continue
                for _ in range(args.retries):
                    direction = rng.normal(size=18)
                    direction /= np.linalg.norm(direction)
                    attempt = homotopy.track(
                        curved_path(offset, direction, args.amplitude),
                        target,
                        seed.q,
                        sigma_floor=args.sigma_floor,
                    )
                    retried += 1
                    if attempt.finished:
                        attempt.reason = "arrived on a curved path"
                        tracks.append(attempt)
                        break
        seconds = time.perf_counter() - started
        arrived = [t for t in tracks if t.finished]

        # the reference: a multi-start census of the real arm
        fiber = census(sr5_model, target, solver=make_lm_solver(sr5_model), seeds=args.seeds, rng=rng)
        numerical = fiber.solutions

        def matches(point: np.ndarray, others: list[np.ndarray]) -> bool:
            return any(
                iks.configuration_distance(point, other, circular) < args.match for other in others
            )

        reached = [t.q for t in arrived]
        missed_by_homotopy = [q for q in numerical if not matches(q, reached)]
        extra_from_homotopy = [q for q in reached if not matches(q, numerical)]
        total_arrived += len(arrived)
        total_census += len(numerical)
        unreached_total += len(missed_by_homotopy)
        unreached_poses += int(bool(missed_by_homotopy))

        worst_sigma = min((t.min_sigma for t in tracks), default=float("nan"))
        print(
            f"pose {pose}: {len(seeds)} seeds -> {len(arrived)} arrived"
            f" | census {len(numerical)} (from {fiber.converged}/{fiber.seeds})"
            f" | census solutions the homotopy missed: {len(missed_by_homotopy)}"
            f" | homotopy solutions the census missed: {len(extra_from_homotopy)}"
            f" | worst clearance on any path {worst_sigma:.1e}"
            f" | {seconds:.1f} s"
            + (f" | curved-path retries {retried}" if args.retries else "")
        )
        for index, t in enumerate(tracks):
            flag = "arrived" if t.finished else "stopped"
            print(
                f"    seed {index}: {flag} at s = {t.progress:.3f}, steps {t.steps:4d}, "
                f"min clearance {t.min_sigma:.2e} at s = {t.min_sigma_at:.3f}, "
                f"residual {t.residual:.1e}"
                + ("" if t.finished else f"  ({t.reason})")
            )

    print(f"\ntotals: {total_arrived} solutions carried to SR5, {total_census} found by census, "
          f"{unreached_total} census solutions never reached "
          f"(on {unreached_poses} of {args.poses} poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
