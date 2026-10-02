#!/usr/bin/env python3
"""exp20: do the classical labels survive on a redundant arm?

`study/labels.py` computes the textbook shoulder/elbow/wrist signs from the axes.  On the SR5 they
were refuted as branch labels (NOTES 3.9).  On a 7-DOF arm the test is sharper and cheaper: follow a
**self-motion** path -- the pose never changes, and if the path keeps clearance then every
configuration on it is in one chamber by construction (the path is a witness) -- and watch the three
signs.  Any flip along such a path proves that sign is not an invariant of the chamber, so no
classifier built on the triple can be sound.

Run::

    python3 -m study.exp20_labels_redundant --robot franka --seeds 150
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.exp18_franka_redundant import correct_pose, null_direction
from study.labels import branch_labels


def main() -> int:
    """Trace self-motion paths and watch the labels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    rng = np.random.default_rng(args.seed)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
    print(f"{args.robot}: {len(solutions)} solutions at one pose "
          f"({fiber.converged}/{fiber.seeds} seeds converged)")
    triples = [branch_labels(model, q) for q in solutions]
    print(f"distinct label triples among them: {sorted(set(triples))}")

    flips = 0
    for index, q in enumerate(solutions[:6]):
        start_labels = branch_labels(model, q)
        q_current = np.asarray(q, dtype=float).copy()
        seen = [start_labels]
        min_clearance = iks.sigma_min(model, q_current)
        worst_drift = 0.0
        travelled = 0.0
        for _ in range(args.steps):
            direction = null_direction(model, q_current)
            if direction is None:
                break
            moved, drift = correct_pose(model, q_current + args.step * direction, target)
            worst_drift = max(worst_drift, drift)
            clearance = iks.sigma_min(model, moved)
            if clearance < args.delta or drift > 1e-9:
                break
            min_clearance = min(min_clearance, clearance)
            travelled += float(np.linalg.norm(moved - q_current))
            q_current = moved
            current = branch_labels(model, q_current)
            if current != seen[-1]:
                seen.append(current)
        changed = len(seen) > 1
        flips += int(changed)
        print(
            f"  solution {index}: labels along the self-motion path "
            f"{' -> '.join(str(t) for t in seen)}"
            f" | travelled {travelled:.2f} rad, min clearance {min_clearance:.2e}, "
            f"pose drift {worst_drift:.1e}"
            f" | {'LABELS CHANGE' if changed else 'labels constant'}"
        )
    print(f'\nsolutions whose labels change along a pose-invariant, clearance-keeping path: '
          f'{flips} of {min(6, len(solutions))}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
