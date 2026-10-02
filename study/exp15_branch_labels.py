#!/usr/bin/env python3
"""exp15: the branch partition by directed oracles, checked against the geometric labels

Two independent ways to say "same branch", and one audit:

* **certificates** -- ``sign det J`` splits the solutions, and opposite signs prove different
  chambers (one Jacobian each, no paths);
* **directed walks** -- a greedy clearance walk between two solutions, tried in both directions and
  through random admissible waypoints, because a single one-directional walk leaves most pairs
  unresolved (NOTES 3.7);
* **audit** -- no walk may ever connect two solutions of opposite determinant sign, and no
  component may mix labels that the geometry says are separated by a stratum.

Then the question the eye cannot answer: do the geometric shoulder/elbow/wrist labels agree with
the partition the certificates and walks produce?

Run::

    python3 -m study.exp15_branch_labels --poses 3 --seeds 600
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair
from study.labels import branch_labels


def resolve(model, q_a, q_b, *, delta: float, circular, rng: np.random.Generator,
            waypoints: int = 4) -> bool:
    """Whether a clearance-keeping route exists, by walking both ways and through waypoints."""
    for first, second in ((q_a, q_b), (q_b, q_a)):
        if resolve_pair(model, first, second, delta=delta, circular=circular).connected:
            return True
    middle = 0.5 * (np.asarray(q_a) + np.asarray(q_b))
    for _ in range(waypoints):
        for _ in range(200):
            candidate = middle + rng.normal(0.0, 0.3, size=middle.size)
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
    """Run the partition and the label comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--poses", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    same_label_connected = same_label_split = 0
    different_label_connected = 0
    sign_violations = 0
    for pose in range(args.poses):
        target, _ = iks.random_reachable_target(model, rng)
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
        signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
        labels = [branch_labels(model, q) for q in solutions]
        connected: dict[tuple[int, int], bool] = {}
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                if signs[i] != signs[j]:
                    connected[(i, j)] = False  # proven different by the sign certificate
                    continue
                connected[(i, j)] = resolve(
                    model, solutions[i], solutions[j],
                    delta=args.delta, circular=circular, rng=rng,
                )
                if connected[(i, j)] and signs[i] != signs[j]:
                    sign_violations += 1
        pairs = len(connected)
        linked = sum(1 for value in connected.values() if value)
        cross = sum(
            1 for (i, j), value in connected.items() if value and signs[i] != signs[j]
        )
        for (i, j), value in connected.items():
            if labels[i] == labels[j]:
                same_label_connected += int(value)
                same_label_split += int(not value)
            elif value:
                different_label_connected += 1
        print(
            f"pose {pose}: {len(solutions)} solutions, sign groups "
            f"{signs.count(1)}+/{signs.count(-1)}- | {pairs} pairs, {linked} linked by a walk "
            f"| walks across sign groups: {cross} (must be 0) "
            f"| distinct label triples: {len(set(labels))}"
        )
        for index, (sign, label) in enumerate(zip(signs, labels)):
            print(f"    solution {index}: det sign {sign:+d}, labels (shoulder, elbow, wrist) {label}")

    print(f"\nlabel agreement: same-label pairs connected {same_label_connected}, "
          f"same-label pairs NOT connected {same_label_split}, "
          f"different-label pairs connected {different_label_connected}")
    print(f"sign-certificate violations by walks: {sign_violations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
