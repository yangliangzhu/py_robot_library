#!/usr/bin/env python3
"""exp00: what does it take to enumerate the IK solutions of a 6R, and how many are there?

Every later experiment stands on a census: the pair tests need two solutions of one pose, and
"two" has to mean "two of the k the arm has".  This experiment measures the tools before using
them.

Measured, per robot:

1. **Convergence from arbitrary seeds.**  The library's IK is built for a control loop (a
   nearby seed, a small correction); a census needs the opposite.  For each solver the table
   reports how often it converges from seeds drawn over the whole joint range, and from seeds
   perturbed around a known solution at several radii -- the second number is the basin size.
2. **Cost.**  Wall time per solve, on both backends.
3. **Coverage.**  Distinct solutions found as a function of the seed budget, which is what says
   whether a census is finished or merely stopped.
4. **The count itself.**  How many solutions a 6R with an offset wrist has at a generic pose --
   the number the whole branch discussion is about.

Run::

    python3 -m study.exp00_solver_baselines --robot sr5 --targets 3
    python3 -m study.exp00_solver_baselines --robot er3 --seeds 200
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, evaluate_solvers, make_lm_solver, perturbation_sweep

#: Seed perturbations for the sweep, in radians (2.9 degrees at 0.05).
RADII = (1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1)


def _bound(robot) -> object:
    """The model's own IK as a ``(seed, target) -> (q, converged)`` callable."""

    def solve(seed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        return robot.ik(seed, target)

    return solve


def main() -> int:
    """Run the baselines and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5", help="robot name from ModelFactory")
    parser.add_argument("--backend", default="casadi", help="numpy or casadi")
    parser.add_argument("--targets", type=int, default=3, help="poses to census")
    parser.add_argument("--seeds", type=int, default=400, help="seeds per pose")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend=args.backend, ik_type=IkType.IK_STANDARD)
    rng = np.random.default_rng(args.seed)
    print(f"{args.robot} on {args.backend}: dof {model.num_dof}, circular joints "
          f"{iks.circular_joints(model).astype(int).tolist()}")
    print(f"  bounds (deg): {np.round(np.degrees(model.lower_bounds), 1).tolist()} .. "
          f"{np.round(np.degrees(model.upper_bounds), 1).tolist()}")

    solvers = {
        "library": lambda seed, target: model.ik(seed, target),
        "lm": make_lm_solver(model),
    }
    for name, solver in solvers.items():
        report = evaluate_solvers(
            model, solver=solver, rng=rng, seeds=args.seeds, perturbations=(0.05, 0.2, 0.5, 1.0)
        )
        rates = " ".join(f"{radius:g}:{rate:5.1%}" for radius, rate in report["perturbed"])
        per_attempt = 1e3 * float(report["seconds"]) / args.seeds
        print(
            f"  {name:8s} random seeds: {report['random_rate']:6.1%} converged"
            f" | perturbed {rates}"
            f" | {per_attempt:6.2f} ms/attempt, {float(report['seconds']):5.2f} s for {args.seeds}"
        )

    print("\nseed-distance sweep (success rate / median final pose error):")
    print("  " + f"{'solver':14s}" + "".join(f"{r:>16g}" for r in RADII))
    print("  " + f"{'(radius rad)':14s}" + "".join(f"{'any/same/other':>16s}" for _ in RADII))

    def sweep_row(label: str, robot) -> None:
        sweep = perturbation_sweep(
            robot, solver=_bound(robot), rng=rng, radii=RADII
        )
        cells = "".join(f"{any_:>5.0%}/{same:>4.0%}/{other:>5d}" for _, any_, same, other, _ in sweep)
        print(f"  {label:14s}{cells}")

    for solver_name in ("IK_STANDARD", "IK_NULL"):
        sweep_row(
            solver_name,
            ModelFactory.create(args.robot, backend=args.backend, ik_type=IkType[solver_name]),
        )
    sweep_row("study lm", model)
    print("  (any = converged to some solution, same = to the perturbed one, other = distinct")

    print("\ncoverage and count (lm solver, increasing seed budget):")
    for trial in range(args.targets):
        target, q_true = iks.random_reachable_target(model, rng)
        row = []
        for budget in (25, 50, 100, 200, 400, 800):
            fiber = census(
                model, target, solver=make_lm_solver(model), seeds=budget, rng=rng
            )
            row.append(f"{budget}:{fiber.count}")
        fiber = census(model, target, solver=make_lm_solver(model), seeds=800, rng=rng)
        hits = sorted(fiber.hits, reverse=True)
        sigma = sorted(round(iks.sigma_min(model, q), 4) for q in fiber.solutions)
        print(
            f"  pose {trial}: solutions by budget {' '.join(row)}"
            f" | final {fiber.count} from {fiber.converged} converged"
            f" | hits {hits[:8]}{'...' if len(hits) > 8 else ''}"
        )
        print(f"     sigma_min at each solution: {sigma}")
        print(f"     worst pose error: {fiber.summary()['worst_error']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
