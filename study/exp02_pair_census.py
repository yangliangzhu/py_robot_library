#!/usr/bin/env python3
"""exp02: are two solutions of one pose in different chambers?  Certificates, not dips.

For every pair of distinct solutions of one pose the shortest path in configuration space is
scanned, and the verdict comes from ``det J`` (continuous, signed, and zero exactly on the
singular set of a 6-DOF arm) rather than from the manipulability dip:

* an **odd number of transversal crossings** of ``det J = 0`` *proves* the two solutions are in
  different chambers -- the path leaves one component of ``Q \\ Sigma`` and cannot come back;
* **no crossing** (and no grazing touch) with a positive clearance means the path stays inside
  ``Q \\ Sigma``, i.e. the two solutions are in the same chamber: a counterexample to "distinct
  solutions are always separated by the singular set", and the case worth studying;
* a **dip without a crossing** is the near miss the original experiments kept hitting; the
  refined minimum of ``sigma_min`` measures how near.

The same pairs are then judged by the manipulability criterion at several thresholds, which
gives that criterion's confusion matrix against certificates.

Run::

    python3 -m study.exp02_pair_census --robot sr5 --poses 6 --seeds 600
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair

#: Manipulability-dip thresholds to score the empirical criterion at.
DIP_THRESHOLDS = (0.2, 0.1, 0.05, 0.01, 0.005)


def main() -> int:
    """Run the pair census."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--backend", default="casadi")
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma-floor", type=float, default=1e-3,
                        help="solutions closer to the singular set than this are excluded")
    parser.add_argument("--clearance", type=float, default=5e-3,
                        help="clearance a connecting walk must keep to certify same chamber")
    parser.add_argument("--resolve", action="store_true",
                        help="try to connect the not-certified-different pairs at that clearance")
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend=args.backend, ik_type=IkType.IK_STANDARD)
    if model.num_dof != 6:
        raise SystemExit("exp02 needs a 6-DOF arm: det J is the certificate it uses")
    rng = np.random.default_rng(args.seed)
    circular = iks.circular_joints(model)

    different = same = even = grazing = 0
    sign_certificate = sign_disagreements = 0
    pairs_total = 0
    clearances: list[float] = []
    solution_sigmas: list[float] = []
    best_same: list[tuple[int, int, int, float, float]] = []
    dip_says: dict[float, int] = {eps: 0 for eps in DIP_THRESHOLDS}
    solutions_of: dict[int, list[np.ndarray]] = {}
    crossing_of: dict[tuple[int, int, int], bool] = {}
    dip_true_positive: dict[float, int] = {eps: 0 for eps in DIP_THRESHOLDS}

    for pose in range(args.poses):
        target, _q_true = iks.random_reachable_target(model, rng)
        fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
        solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.sigma_floor]
        excluded = fiber.count - len(solutions)
        solution_sigmas.extend(iks.sigma_min(model, q) for q in solutions)
        solutions_of[pose] = solutions
        hits = sorted(fiber.hits, reverse=True)
        sigmas = [iks.sigma_min(model, q) for q in solutions]
        print(f"pose {pose}: {len(solutions)} solutions"
              f"{f' ({excluded} excluded as near-singular)' if excluded else ''}"
              f", {fiber.converged}/{fiber.seeds} seeds converged"
              f" | hits {hits[:6]}{'...' if len(hits) > 6 else ''}"
              f" | sigma_min {min(sigmas):.3f}..{max(sigmas):.3f}")
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                scan = iks.scan_path(
                    model, solutions[i], solutions[j], samples=args.samples, circular=circular
                )
                pairs_total += 1
                odd = scan.crossings % 2 == 1
                crossing_of[(pose, i, j)] = odd
                if odd:
                    different += 1
                elif scan.crossings == 0:
                    if scan.det_touches:
                        grazing += 1
                    else:
                        same += 1
                        clearances.append(scan.min_sigma)
                        best_same.append(
                            (pose, i, j, scan.min_sigma, float(scan.manipulability_ratio))
                        )
                else:
                    even += 1
                det_i = iks.det_jacobian(model, solutions[i])
                det_j = iks.det_jacobian(model, solutions[j])
                if (det_i < 0.0) != (det_j < 0.0):
                    sign_certificate += 1
                    if not odd:
                        sign_disagreements += 1
                ratio = scan.manipulability_ratio
                for eps in DIP_THRESHOLDS:
                    if ratio < eps:
                        dip_says[eps] += 1
                        dip_true_positive[eps] += int(odd)

    print(f"\n{pairs_total} pairs over {args.poses} poses, judged by det J crossings:")
    print(f"  different chambers (odd crossings) : {different}")
    print(f"  same chamber      (no crossing)    : {same}")
    print(f"  even crossings (left and came back): {even}")
    print(f"  grazing touches of the singular set: {grazing}")
    print(f"  the one-Jacobian sign certificate fired for {sign_certificate} pairs; "
          f"path scans disagreed with it {sign_disagreements} times")
    print(f"  sigma_min at the census solutions: median "
          f"{np.median(solution_sigmas):.3f} (scale for the clearances below)")
    if clearances:
        values = np.asarray(clearances)
        print(f"  clearance of same-chamber paths: min {values.min():.3e}, "
              f"median {np.median(values):.3e}")
        for pose, i, j, clearance, ratio in sorted(best_same, key=lambda row: row[3])[:5]:
            print(f"    closest call: pose {pose} pair ({i},{j}) clearance {clearance:.3e}, "
                  f"dip ratio {ratio:.3f}")

    touches = different + even + grazing
    print(f"\n  paths that touch the singular set at all: {touches}/{pairs_total} "
          f"({different} crossing once, {even} crossing twice, {grazing} grazing)")
    print("\nmanipulability-dip criterion as a *crossing detector* "
          "(truth = the refined minimum of sigma_min reaches zero):")
    print(f"  {'threshold':>10s} {'dips':>6s} {'true pos':>9s} {'false pos':>10s} {'false neg':>10s}")
    for eps in DIP_THRESHOLDS:
        # a dip on a pair whose path does not touch Sigma is the near-miss counterexample
        true_positive = dip_says[eps] - max(0, dip_says[eps] - touches)
        false_positive = max(0, dip_says[eps] - touches)
        false_negative = touches - true_positive
        print(f"  {eps:>10.3f} {dip_says[eps]:>6d} {max(true_positive, 0):>9d} "
              f"{false_positive:>10d} {max(false_negative, 0):>10d}")

    if args.resolve:
        print(f"\nclearance walks at delta = {args.clearance:g}, for every pair:")
        walk_same = walk_unresolved = walk_failed = 0
        controls_ok = controls_bad = 0
        for pose in solutions_of:
            group = solutions_of[pose]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    odd = crossing_of.get((pose, i, j), False)
                    walk = resolve_pair(
                        model, group[i], group[j], delta=args.clearance, circular=circular
                    )
                    if walk.connected:
                        walk_same += 1
                        if odd:
                            controls_bad += 1
                            print(f"  CONTROL FAILURE: pose {pose} pair ({i},{j}) has odd "
                                  f"crossings yet a clearance walk connected it "
                                  f"(min clearance {walk.min_sigma:.3e})")
                    elif odd:
                        controls_ok += 1
                        walk_failed += 1
                    else:
                        walk_unresolved += 1
                        if walk.progress > 0.5:
                            print(f"  pose {pose} pair ({i},{j}): unresolved at "
                                  f"{walk.progress:.0%} of the way "
                                  f"({walk.reason}, min clearance {walk.min_sigma:.3e})")
        print(f"  connected while keeping clearance (same chamber): {walk_same}")
        print(f"  not connected (unresolved or different)         : "
              f"{walk_unresolved + walk_failed}")
        print(f"  negative control -- certified-different pairs the walk could not connect: "
              f"{controls_ok}/{controls_ok + controls_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
