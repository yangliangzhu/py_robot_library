#!/usr/bin/env python3
"""exp12: the SR0 analytic solver, checked against the numerical census

`study/sr0.py` solves the Pieper-solvable neighbour of the SR5 deterministically: two shoulder
branches for ``q1``, a scalar root-find for ``q3``, closed-form ``q2``, two branches for ``q5``
and a deterministic finish for ``(q4, q6)``.  A solver like that is only useful if it is
*complete* -- a homotopy seeded from a missing solution starts by losing it -- so this
experiment compares it against multi-start Levenberg-Marquardt, which knows nothing about the
structure.

Measured per pose:

* how many solutions each method finds, and whether the analytic set contains the numerical one
  (and vice versa);
* the worst pose residual of the analytic solutions;
* wall time per solve, because the homotopy will call this thousands of times.

Run::

    python3 -m study.exp12_sr0_solver --poses 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver


def main() -> int:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=400, help="numerical census seeds per pose")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match", type=float, default=1e-6, help="matching tolerance, radians")
    args = parser.parse_args()

    model = sr0.robot()
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    print(f"SR0: {model.num_dof} axes, turn sign {sr0._planar_rotation_sign(model):+.0f}, "
          f"wrist centre in the flange {sr0.WRIST_IN_FLANGE.tolist()}")

    analytic_counts, numeric_counts = [], []
    worst_residual = 0.0
    analytic_seconds = 0.0
    missing_numeric = missing_analytic = 0
    for pose in range(args.poses):
        q_true = rng.uniform(-2.0, 2.0, model.num_dof)
        target = model.fk(q_true)

        started = time.perf_counter()
        analytic = sr0.solve(model, target)
        analytic_seconds += time.perf_counter() - started
        worst_residual = max(worst_residual, max((s.error for s in analytic), default=0.0))

        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        numeric = fiber.solutions

        def matches(point: np.ndarray, others: list[np.ndarray]) -> bool:
            return any(
                iks.configuration_distance(point, other, circular) < args.match for other in others
            )

        lost = [q for q in numeric if not matches(q, [s.q for s in analytic])]
        extra = [s.q for s in analytic if not matches(s.q, numeric)]
        missing_numeric += len(lost)
        missing_analytic += len(extra)
        analytic_counts.append(len(analytic))
        numeric_counts.append(len(numeric))
        contains_true = any(
            iks.configuration_distance(q_true, s.q, circular) < args.match for s in analytic
        )
        print(
            f"pose {pose}: analytic {len(analytic)}, numerical {len(numeric)} "
            f"(from {fiber.converged}/{fiber.seeds} converged) | numerical solutions the "
            f"analytic solver missed: {len(lost)} | analytic solutions the census missed: "
            f"{len(extra)} | generated configuration recovered: {contains_true}"
        )

    print(f"\nanalytic counts {analytic_counts}")
    print(f"numerical counts {numeric_counts}")
    print(f"worst analytic pose residual: {worst_residual:.2e}")
    print(f"analytic time per solve: {1e3 * analytic_seconds / args.poses:.1f} ms")
    print(f"totals: census-only solutions {missing_numeric}, analytic-only solutions "
          f"{missing_analytic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
