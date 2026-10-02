#!/usr/bin/env python3
"""exp34: are exp19's other same-sign residuals also walk-budget artefacts?

Follow-up to exp32 (which closed the `[4,3,1]` residue as a walk-budget artefact: the singleton
connected at 48 waypoints, restoring the ``det J`` sign split).  Kimi's stronger conjecture
(`study/collab/day02-kimi.md` section 3): **every** same-sign residue left by exp19's greedy walk
is a budget artefact, i.e. the partition equals the sign split generically.  This experiment replays
the two residual poses found by the A-1.2 stability runs -- ``[2,2,1]`` (master seed 1, pose 2) and
``[2,1,1]`` (master seed 2, pose 0) -- with exp19's exact rng consumption, then interrogates each
singleton the way exp32 did: clearance, nearest-neighbour distance, and stronger walks
(48 waypoints x 3 fresh seeds, plus a long-budget walk).

Run::

    python3 -m study.exp34_residuals --seed 1 --poses 3
    python3 -m study.exp34_residuals --seed 2 --poses 1
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


def _partition(model, solutions, signs, args, circular, rng) -> tuple[list[list[int]], list]:
    """exp19's witnessed-pair partition, replayed with the same rng consumption."""
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
    return sorted((sorted(g) for g in groups.values()), key=len, reverse=True), undetermined


def main() -> int:
    """Replay the residual poses and interrogate their singletons."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--poses", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--waypoints", type=int, default=6, help="as in the exp19 runs")
    parser.add_argument("--retry-waypoints", type=int, default=48)
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    for pose in range(args.poses):
        target, _ = iks.random_reachable_target(model, rng)
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
        signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
        groups, undetermined = _partition(model, solutions, signs, args, circular, rng)
        print(f"\npose {pose}: {len(solutions)} solutions ({fiber.converged}/{fiber.seeds}), "
              f"sign groups {signs.count(1)}+/{signs.count(-1)}-, "
              f"partition {[len(g) for g in groups]} -> {groups}")
        by_sign: dict[int, list[list[int]]] = {}
        for group in groups:
            by_sign.setdefault(signs[group[0]], []).append(group)
        for sign, sign_groups in sorted(by_sign.items()):
            if len(sign_groups) < 2:
                continue
            print(f"  sign class {sign:+d} splits as {[len(g) for g in sign_groups]} "
                  f"(residue against the sign split)")
            singletons = [g[0] for g in sign_groups if len(g) == 1]
            for s in singletons:
                mates = [j for g in sign_groups for j in g if j != s]
                nearest = min(
                    iks.configuration_distance(solutions[s], solutions[j], circular)
                    for j in range(len(solutions)) if j != s
                )
                print(f"  singleton {s}: sigma_min {iks.sigma_min(model, solutions[s]):.3e}, "
                      f"nearest solution {nearest:.3f} rad")
                for m in mates:
                    crossings = _sign_changes(model, solutions[s], solutions[m], circular)
                    walk = resolve_pair(model, solutions[s], solutions[m], delta=args.delta,
                                        steps=args.budget, circular=circular)
                    retries = []
                    for trial in range(3):
                        retry_rng = np.random.default_rng(
                            1000 + 97 * trial + 13 * m + pose)
                        retries.append(
                            walked(model, solutions[s], solutions[m], delta=args.delta,
                                   circular=circular, rng=retry_rng,
                                   waypoints=args.retry_waypoints))
                    print(f"    pair ({s},{m}): line crosses det J {crossings}x | "
                          f"long walk connected={walk.connected} ({walk.progress:.0%}, "
                          f"'{walk.reason}') | 48-waypoint retries {retries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
