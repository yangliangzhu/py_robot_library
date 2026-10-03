#!/usr/bin/env python3
"""exp19: how completely can one fibre's branch partition be determined?

The two certificates this study has established, applied together to a single pose of the SR5:

* **sign det J** -- rigorous for "different": two solutions whose determinants differ are in
  different chambers, at the cost of one Jacobian each;
* **a witnessed clearance path** -- constructive for "same": a verified route in
  ``{sigma_min >= delta}`` between two solutions proves they share a chamber (the route *is* the
  lift of a workspace loop), tested in both directions and through random admissible waypoints.

Everything the two certify is reported, and so is the **residue**: pairs that neither certificate
resolves.  For those the experiment also reports what the *dip heuristic* of the original question
would say (a path whose clearance dips below a threshold is suspected to cross the singular set),
because that is the criterion under study -- with the error rates measured for it in
`study/exp02_pair_census.py` kept in view (at a 0.01 threshold: no false negatives, some near-miss
false positives).

Run::

    python3 -m study.exp19_partition --poses 1 --seeds 1000 --waypoints 12
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair


def walked(
    model, q_a: np.ndarray, q_b: np.ndarray, *, delta: float, circular, rng, waypoints: int
) -> bool:
    """Whether a verified clearance route joins the two configurations."""
    for first, second in ((q_a, q_b), (q_b, q_a)):
        if resolve_pair(model, first, second, delta=delta, circular=circular).connected:
            return True
    middle = 0.5 * (np.asarray(q_a) + np.asarray(q_b))
    for _ in range(waypoints):
        for _ in range(300):
            candidate = middle + rng.normal(0.0, 0.35, size=middle.size)
            if iks.sigma_min(model, candidate) >= delta:
                break
        else:
            continue
        if (
            resolve_pair(model, q_a, candidate, delta=delta, circular=circular).connected
            and resolve_pair(model, candidate, q_b, delta=delta, circular=circular).connected
        ):
            return True
    return False


def main() -> int:
    """Run the partition for one pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--poses", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--waypoints", type=int, default=12)
    parser.add_argument("--dip", type=float, default=0.01, help="dip threshold for the heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-seed", type=int, default=None,
                        help="decouple the poses from the census stream: pose k uses "
                             "default_rng(target_seed + k), so re-running with more waypoints or more "
                             "seeds chases the *same* pose (a coupled stream silently changes the pose, "
                             "which made one comparison in this study invalid)")
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    for pose in range(args.poses):
        if args.target_seed is None:
            target, _ = iks.random_reachable_target(model, rng)
        else:
            target, _ = iks.random_reachable_target(
                model, np.random.default_rng(args.target_seed + pose)
            )
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
        signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
        print(f"pose {pose}: {len(solutions)} solutions "
              f"({fiber.converged}/{fiber.seeds} seeds), sign groups "
              f"{signs.count(1)}+/{signs.count(-1)}-")

        same, different, undetermined = [], [], []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                if signs[i] != signs[j]:
                    different.append((i, j))
                    continue
                if walked(model, solutions[i], solutions[j], delta=args.delta,
                          circular=circular, rng=rng, waypoints=args.waypoints):
                    same.append((i, j))
                else:
                    undetermined.append((i, j))
        total = len(solutions) * (len(solutions) - 1) // 2
        print(f"  pairs {total}: {len(same)} certified same branch, "
              f"{len(different)} certified different (sign), {len(undetermined)} undetermined")

        # group the certified-same pairs
        parent = list(range(len(solutions)))

        def find(item: int, parent: list[int] = parent) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for i, j in same:
            parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for index in range(len(solutions)):
            groups.setdefault(find(index), []).append(index)
        print(f"  after merging the witnessed pairs: {len(groups)} groups "
              f"(sizes {sorted((len(g) for g in groups.values()), reverse=True)}), "
              f"each inside one sign group")

        # what the dip heuristic says about the residue
        dipped = 0
        for i, j in undetermined:
            scan = iks.scan_path(model, solutions[i], solutions[j])
            if scan.manipulability_ratio < args.dip:
                dipped += 1
        print(f"  of the undetermined pairs, {dipped} show a dip below {args.dip:g} "
              f"(the heuristic would call them different branches; its measured error rates are in "
              f"exp02)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
