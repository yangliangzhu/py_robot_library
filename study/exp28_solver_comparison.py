#!/usr/bin/env python3
"""exp28: is the SR0-homotopy method worth using?  A four-way comparison on the same poses

The same pose, four ways to solve it, no privileged reference -- the methods are compared with each
other, and what matters is what each one *misses* that the others find, and what it costs:

* **the library's own solver**, `IkStandard`, from random seeds (how the IK is normally used);
* **a multi-start census** with the study's Levenberg-Marquardt solver (the thorough numerical way);
* **the SR0 homotopy on the straight parameter path** (Finding 2's method, cheap: eight closed-form
  seeds carried 0.136 m);
* **the SR0 scan** (the complete version: solve the intermediate arm on a grid, continue everything),
  which is the expensive one.

Measured per pose: solutions found, wall time, and which method's solutions the others miss.  The
comparison is deliberately without a reference set, because a reference would have to be one of the
methods or a more expensive one, and either choice would beg the question.

Run::

    python3 -m study.exp28_solver_comparison --poses 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from model import IkType, ModelFactory
from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, library_solver, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def distinct(points: list[np.ndarray], circular, match: float = 1e-3) -> list[np.ndarray]:
    """Deduplicate a solution list on the torus."""
    out: list[np.ndarray] = []
    for point in points:
        if not any(iks.configuration_distance(point, other, circular) < match for other in out):
            out.append(point)
    return out


def missing(reference: list[np.ndarray], other: list[np.ndarray], circular) -> int:
    """How many of ``reference`` have no match in ``other``."""
    return sum(
        1 for q in reference
        if not any(iks.configuration_distance(q, o, circular) < 1e-3 for o in other)
    )


def main() -> int:
    """Run the four-way comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--library-seeds", type=int, default=20)
    parser.add_argument("--census-seeds", type=int, default=300)
    parser.add_argument("--scan-grid", type=int, default=9)
    parser.add_argument("--scan-seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    library = library_solver(model, accept=1e-5)
    sr0_model = sr0.robot()
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    lower, upper = np.asarray(model.lower_bounds), np.asarray(model.upper_bounds)

    totals = {"library": 0, "census": 0, "homotopy": 0, "scan": 0}
    seconds = dict.fromkeys(totals, 0.0)
    for pose in range(args.poses):
        target, _ = iks.random_reachable_target(model, rng)

        started = time.perf_counter()
        library_solutions = []
        for _ in range(args.library_seeds):
            seed = rng.uniform(lower, upper)
            solution, converged = library(seed, target)
            if converged:
                library_solutions.append(solution)
        seconds["library"] += time.perf_counter() - started
        found_library = distinct(library_solutions, circular)

        started = time.perf_counter()
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.census_seeds, rng=rng)
        seconds["census"] += time.perf_counter() - started
        found_census = distinct(fiber.solutions, circular)

        started = time.perf_counter()
        path = path_at(offset)
        found_homotopy = []
        for seed in sr0.solve(sr0_model, target):
            track = homotopy.track(path, target, seed.q, sigma_floor=2e-3)
            if track.finished:
                found_homotopy.append(track.q)
        seconds["homotopy"] += time.perf_counter() - started
        found_homotopy = distinct(found_homotopy, circular)

        started = time.perf_counter()
        found_scan = []
        for s in np.linspace(0.0, 1.0, args.scan_grid):
            intermediate = sr0.robot(path(s))
            local = census(intermediate, target, solver=make_lm_solver(intermediate),
                           seeds=args.scan_seeds, rng=rng)
            for q in local.solutions:
                track = homotopy.track(
                    path if s == 0.0 else _shifted(path, s), target, q, sigma_floor=2e-3
                )
                if track.finished:
                    found_scan.append(track.q)
        seconds["scan"] += time.perf_counter() - started
        found_scan = distinct(found_scan, circular)

        for name, found in (("library", found_library), ("census", found_census),
                            ("homotopy", found_homotopy), ("scan", found_scan)):
            totals[name] += len(found)
        union = distinct(found_library + found_census + found_homotopy + found_scan, circular)
        print(
            f"pose {pose}: library {len(found_library)} ({seconds['library'] / (pose + 1):.1f} s avg), "
            f"census {len(found_census)} ({seconds['census'] / (pose + 1):.1f} s), "
            f"homotopy {len(found_homotopy)} ({seconds['homotopy'] / (pose + 1):.1f} s), "
            f"scan {len(found_scan)} ({seconds['scan'] / (pose + 1):.1f} s) | union {len(union)}"
        )
        for name, found in (("library", found_library), ("census", found_census),
                            ("homotopy", found_homotopy), ("scan", found_scan)):
            print(f"    {name:9s} misses {missing(union, found, circular):2d} of the union "
                  f"({len(found)} of {len(union)})")
    print("\ntotals over poses:", {name: totals[name] for name in totals})
    return 0


def _shifted(path, s0: float):
    """The path restricted to ``[s0, 1]``, reparameterised to ``[0, 1]``."""
    def make(t: float) -> list[np.ndarray]:
        return path(s0 + t * (1.0 - s0))

    return make


if __name__ == "__main__":
    raise SystemExit(main())
