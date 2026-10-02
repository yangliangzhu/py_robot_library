#!/usr/bin/env python3
"""exp16: where do the solutions the homotopy loses actually come from?

exp13/14 lost solutions at the real arm and recovered most of them by reseeding at the points where
a tracked branch stalled.  The hypothesis to test here is sharper: at a discriminant a **multiple
root** may split into several real roots, and a tracker that follows one branch past that point
follows only one of them -- so the lost solutions should be traceable to a *birth event* somewhere
along the path, not to numerical bad luck.

The experiment measures, along the parameter path ``link 5 y: 0 -> 0.136 m`` at a fixed pose:

* the **fibre size** on a grid of ``s`` (a multi-start census at the intermediate arm, which is
  legitimate here because it is only used to *count* and to *locate* events, not to define them);
* the **coalescence** structure at each stall of the tracker: how far the stalled configuration is
  from the nearest *other* real solution of the same intermediate arm (a double root shows up as a
  distance going to zero);
* whether the fibre size **changes** across each stall -- a death, a birth, or nothing;
* which of the arm's final solutions are traceable to a birth that the tracker never visited.

Run::

    python3 -m study.exp16_root_splitting --census-seeds 400 --grid 25
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


@dataclass
class FibrePoint:
    """One sample of the parameter sweep."""

    s: float
    count: int
    converged: int


def fibre_count(model, target: np.ndarray, seeds: int, rng: np.random.Generator) -> tuple[int, int]:
    """``(distinct solutions, seeds converged)`` for one intermediate arm."""
    fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
    return fiber.count, fiber.converged


def main() -> int:
    """Run the sweep and the stall analysis."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0, help="seed for the target pose")
    parser.add_argument("--census-seeds", type=int, default=400)
    parser.add_argument("--grid", type=int, default=25, help="samples along the parameter path")
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.pose_seed)
    q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    print(f"pose from seed {args.pose_seed}: {np.round(q_true, 3).tolist()}")
    print(f"path: link 5 y 0 -> {offset:.4f} m, sigma_floor {args.sigma_floor:g}\n")

    # 1. the fibre size along the path, and where it changes
    rng_sweep = np.random.default_rng(args.seed)
    sweep: list[FibrePoint] = []
    for s in np.linspace(0.0, 1.0, args.grid):
        model = sr0.robot(path_at(offset)(s))
        count, converged = fibre_count(model, target, args.census_seeds, rng_sweep)
        sweep.append(FibrePoint(float(s), count, converged))
    print("fibre size along the path (s, distinct solutions, seeds converged):")
    for point in sweep:
        marker = ""
        if point is not sweep[0] and point.count != sweep[sweep.index(point) - 1].count:
            marker = "  <-- size changes here"
        print(f"  s = {point.s:5.3f}: {point.count:3d} solutions from {point.converged:4d} seeds"
              f"{marker}")

    # 2. track the eight SR0 solutions and look at every stall
    seeds = sr0.solve(sr0_model, target)
    tracks = [
        homotopy.track(path_at(offset), target, seed.q, sigma_floor=args.sigma_floor)
        for seed in seeds
    ]
    print(f"\ntracking: {sum(1 for t in tracks if t.finished)} of {len(tracks)} arrived")
    for index, track in enumerate(tracks):
        if track.finished:
            print(f"  seed {index}: arrived, steps {track.steps}, "
                  f"min clearance {track.min_sigma:.2e}")
            continue
        s_star = track.progress
        model = sr0.robot(path_at(offset)(s_star))
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.census_seeds,
                       rng=rng_sweep)
        distances = sorted(
            iks.configuration_distance(track.q, other, circular) for other in fiber.solutions
        )
        nearest = distances[0] if distances else float("nan")
        # a double root: the stalled configuration coincides with another solution
        double = nearest < 1e-3
        before = fibre_count(sr0.robot(path_at(offset)(max(s_star - 0.02, 0.0))), target,
                             args.census_seeds, rng_sweep)[0]
        after = fibre_count(sr0.robot(path_at(offset)(min(s_star + 0.02, 1.0))), target,
                            args.census_seeds, rng_sweep)[0]
        print(
            f"  seed {index}: stalled at s = {s_star:.3f} | fibre there {fiber.count} "
            f"| nearest other solution {nearest:.2e} rad "
            f"({'DOUBLE ROOT' if double else 'isolated'}) "
            f"| fibre at s-0.02: {before}, at s+0.02: {after} "
            f"| clearance {track.min_sigma:.2e}"
        )

    # 3. which final solutions are reachable from a track, and which are not
    fiber_final = census(sr5_model, target, solver=make_lm_solver(sr5_model),
                         seeds=max(args.census_seeds, 800), rng=rng_sweep)
    reached = [t.q for t in tracks if t.finished]
    missed = [
        q for q in fiber_final.solutions
        if not any(iks.configuration_distance(q, r, circular) < 1e-4 for r in reached)
    ]
    print(f"\nfinal arm: {len(fiber_final.solutions)} solutions "
          f"({fiber_final.converged} seeds converged), traces reached {len(reached)}, "
          f"lost {len(missed)}")
    for index, q in enumerate(missed):
        clearance = iks.sigma_min(sr5_model, q)
        nearest_track = min(
            (iks.configuration_distance(q, r, circular) for r in reached), default=float("nan")
        )
        print(f"  lost solution {index}: clearance at the real arm {clearance:.3f}, "
              f"nearest reached solution {nearest_track:.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
