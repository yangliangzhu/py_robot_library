#!/usr/bin/env python3
"""exp32: the [4,3,1] residue -- is the singleton degenerate, or a real same-sign chamber?

Task A-4 of the collaboration (``study/collab/day02-status.md``).  exp19's partition of one fibre
(pose 0, master seed 0, 600 census seeds) leaves one sign class split as ``[3, 1]``: three
solutions witnessed connected, one left over.  This experiment reproduces that fibre and that
partition exactly (same rng consumption as exp19), then measures the singleton:

* its clearance and its distance to every other solution of the fibre -- is it near a double
  root (degenerate), or a healthy isolated point?
* what the straight line to each mate of its sign class does (det J sign changes, refined
  minima of ``sigma_min``);
* whether stronger walks connect it anyway: more waypoints with fresh waypoint seeds, and
  longer step budgets for the clearance walk.

Run::

    python3 -m study.exp32_singleton
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair
from study.exp19_partition import walked


def _sign_changes(model, q_a: np.ndarray, q_b: np.ndarray, circular, samples: int = 401) -> int:
    """How often ``det J`` changes sign on the straight line from ``q_a`` to ``q_b``."""
    end = iks.shortest_representative(q_a, q_b, circular)
    signs = [
        np.sign(iks.det_jacobian(model, q_a + t * (end - q_a)))
        for t in np.linspace(0.0, 1.0, samples)
    ]
    return sum(1 for first, second in zip(signs, signs[1:]) if first != second)


def main() -> int:
    """Reproduce the [4,3,1] fibre of exp19 and interrogate the singleton."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--waypoints", type=int, default=6, help="as in the exp19 run")
    parser.add_argument("--retry-waypoints", type=int, default=48)
    parser.add_argument("--budget", type=int, default=2400, help="steps for the long walks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)

    # --- replicate exp19's pose 0 exactly: same rng, same consumption order ---------------
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
    print(f"pose 0 reproduced: {len(solutions)} solutions ({fiber.converged}/{fiber.seeds} "
          f"seeds), sign groups {signs.count(1)}+/{signs.count(-1)}-")

    same, undetermined = [], []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            if signs[i] != signs[j]:
                continue
            if walked(model, solutions[i], solutions[j], delta=args.delta,
                      circular=circular, rng=rng, waypoints=args.waypoints):
                same.append((i, j))
            else:
                undetermined.append((i, j))

    parent = list(range(len(solutions)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for i, j in same:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for index in range(len(solutions)):
        groups.setdefault(find(index), []).append(index)
    partition = sorted((sorted(g) for g in groups.values()), key=len, reverse=True)
    print(f"partition reproduced: {[len(g) for g in partition]} -> {partition}, "
          f"undetermined pairs: {undetermined}")

    singletons = [g[0] for g in partition if len(g) == 1]
    print(f"\nsingletons: {singletons}")
    for s in singletons:
        mates = [j for j in range(len(solutions)) if j != s and signs[j] == signs[s]]
        print(f"\n--- singleton solution {s} (det sign {signs[s]:+d}) ---")
        print(f"  sigma_min at the solution : {iks.sigma_min(model, solutions[s]):.4e}")
        distances = sorted(
            (iks.configuration_distance(solutions[s], solutions[j], circular), j)
            for j in range(len(solutions)) if j != s
        )
        print(f"  distance to nearest solution: {distances[0][0]:.4e} (solution "
              f"{distances[0][1]}, sign {signs[distances[0][1]]:+d})")
        print(f"  all distances             : "
              f"{[f'{d:.3f}(#{j}{signs[j]:+d})' for d, j in distances]}")
        all_pairs = sorted(
            iks.configuration_distance(solutions[i], solutions[j], circular)
            for i in range(len(solutions)) for j in range(i + 1, len(solutions))
        )
        print(f"  closest pair in the fibre : {all_pairs[0]:.4e} "
              f"(singleton's rank by nearest-neighbour distance: "
              f"{1 + sum(1 for d in all_pairs if d < distances[0][0])} of {len(all_pairs)})")

        for m in mates:
            group = "same group" if find(m) == find(s) else "the [3] group"
            scan = iks.scan_path(model, solutions[s], solutions[m], circular=circular)
            crossings = _sign_changes(model, solutions[s], solutions[m], circular)
            print(f"\n  pair ({s},{m}) [{group}]: straight line det J sign changes "
                  f"= {crossings}, refined min sigma_min = {scan.min_sigma:.3e} "
                  f"at t = {scan.min_at:.2f}; local minima "
                  f"{[(round(t, 2), f'{v:.1e}') for t, v in scan.local_minima]}")
            walk = resolve_pair(model, solutions[s], solutions[m], delta=args.delta,
                                steps=args.budget, circular=circular)
            print(f"    long walk ({args.budget} steps): connected={walk.connected}, "
                  f"progress {walk.progress:.0%}, reason '{walk.reason}', "
                  f"min sigma {walk.min_sigma:.2e}")
            for trial in range(3):
                retry_rng = np.random.default_rng(1000 + 97 * trial + m)
                connected = walked(model, solutions[s], solutions[m], delta=args.delta,
                                   circular=circular, rng=retry_rng,
                                   waypoints=args.retry_waypoints)
                print(f"    waypoint retry seed {1000 + 97 * trial + m} "
                      f"({args.retry_waypoints} waypoints): connected={connected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
