#!/usr/bin/env python3
"""exp49: uniqueness domains on the SR5 -- a posture change is *not* a trackable Cartesian loop
that stays in one feasible region.

Background (definitions pinned in `study/collab/day04-dsh.md`, from Wenger's review
`arXiv:1610.04080` section 11): for an aspect ``A_i`` of the joint space, the **characteristic
surfaces** are ``CS_i = f^{-1}(f(A_i*)) cap A_i`` with ``A_i*`` the *boundary* of ``A_i``.  The
components of ``A_i \\ CS_i`` are uniqueness domains, ``f`` is one-to-one on each of them, and
their images are the regions of feasible paths: a prescribed Cartesian path is trackable from a
posture exactly while it stays inside the image of that posture's uniqueness domain.  So "same
branch" (same aspect: a joint-space statement, which is what the 14 witness pairs of exp02/exp06
establish) and "same feasible path region" (same uniqueness domain: a *task-space* statement) are
different objects, and the second is the one offline programming needs.

How the two are told apart by measurement: a path inside one aspect can cross ``CS_i`` (that is
cuspidality), and *crossing* means the path's projection hits the discriminant image ``f(Sigma)``
-- a pose where two solutions of the fibre coalesce.  Tracking the **whole fibre** along the
projected loop therefore turns the abstract characteristic surface into an event list: a collision
between two branches is a crossing, and a loop that changes posture must show collisions while a
loop that does not change posture must not.

Phases:

1. Find a pose with a rich fibre (12 solutions measured elsewhere) and split it by ``sign det J``
   and by witnessed clearance walks -- the aspect partition.
2. Take a same-aspect pair, get the witness path (`study.chamber.clearance_path`), and project it
   to the workspace: a closed Cartesian loop.
3. Lift the whole fibre along that loop with `study.taskspace.lift_fiber` (lockstep, tangent
   predictor, jump guard).  Report arrivals, the permutation, collisions, and tracker faults.
4. Controls: the degenerate loop (all targets equal) and a small loop around the base pose.

Run::

    python3 -m study.exp49_sr5_uniqueness
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from model import ModelFactory
from study import homotopy
from study import ik_structure as iks
from study import taskspace as ts
from study.census import census, make_lm_solver
from study.chamber import clearance_path
from study.exp11_dh_continuation import link_transforms


def rich_pose(model, rng, *, seeds: int, least: int = 10, tries: int = 8):
    """A pose whose census finds at least ``least`` distinct solutions."""
    for _ in range(tries):
        target = model.fk(rng.uniform(-2.0, 2.0, model.num_dof))
        fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
        solutions: list[np.ndarray] = []
        for candidate in fiber.solutions:
            if iks.sigma_min(model, candidate) <= 1e-4:
                continue
            if any(iks.torus_distance(candidate, other) < 1e-3 for other in solutions):
                continue
            solutions.append(np.asarray(candidate, dtype=float))
        if len(solutions) >= least:
            return target, solutions
    return None, []


def pose_kinematics(links):
    """Residual and Jacobian for the 6-DOF pose the way ``study.homotopy`` measures them.

    The residual must be *relative to the current pose* -- ``se3_error(FK(q), T)`` -- because that
    is what :func:`study.homotopy.jacobian_q` differentiates.  The first version of this experiment
    used a fixed reference pose and differenced two such errors, which is only a first-order
    approximation: the correction direction was wrong by a pose-dependent linear map, and the
    tracker stalled on 9 of 10 branches at healthy clearance (NOTES 3.33).
    """

    def residual(q: np.ndarray, target: np.ndarray) -> np.ndarray:
        return homotopy.se3_error(homotopy.fk_links(links, q), target)

    def jacobian(q: np.ndarray) -> np.ndarray:
        return homotopy.jacobian_q(links, q)

    return residual, jacobian


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--least", type=int, default=10,
                        help="fewest solutions a pose must have to be used (8 is common on SR5)")
    parser.add_argument("--tries", type=int, default=12, help="poses to try before giving up")
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--waypoints", type=int, default=4)
    parser.add_argument("--max-step", type=float, default=0.5,
                        help="largest joint-space step the corrector may take (Q21 probe)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.pose_seed)
    links = link_transforms()
    model = ModelFactory.create("sr5")
    circular = iks.circular_joints(model)

    print("exp49: uniqueness domains on the SR5 -- whole-fibre lifts along witness loops\n")
    print("phase 1: a rich pose and its aspect partition")
    target, solutions = rich_pose(model, rng, seeds=args.seeds, least=args.least,
                                  tries=args.tries)
    if not solutions:
        print("  no pose with a rich fibre in this sample")
        return 0
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
    groups = {1: [i for i, s in enumerate(signs) if s > 0],
              -1: [i for i, s in enumerate(signs) if s < 0]}
    print(f"  {len(solutions)} solutions, det J signs {[signs.count(1), signs.count(-1)]} "
          f"(+/-)")

    print("\nphase 2: a same-aspect pair and its witness path")
    pair = None
    for i, j in itertools.combinations(groups[1] + groups[-1], 2):
        if signs[i] != signs[j]:
            continue
        walk = clearance_path(model, solutions[i], solutions[j], delta=args.delta,
                              circular=circular, steps=1500)
        if walk.connected:
            pair = (i, j, walk)
            break
    if pair is None:
        print("  no same-aspect pair resolved by the clearance walk at this pose")
        return 0
    i, j, walk = pair
    path = np.array(walk.path)
    print(f"  pair ({i}, {j}): both sign {signs[i]:+d}, {len(path)} configurations, smallest "
          f"clearance {walk.min_sigma:.2e} rad, drift reason {walk.reason!r}")
    poses = np.array([model.fk(q) for q in path])
    print(f"  projection: closed workspace loop, start/end agree to "
          f"{np.max(np.abs(poses[0] - poses[-1])):.1e}")

    print("\nphase 3: whole-fibre lift along the projected loop")
    residual_of, jacobian_of = pose_kinematics(links)
    targets = poses
    task_of = lambda q: homotopy.fk_links(links, q)  # noqa: E731 - only for the default error
    result = ts.lift_fiber(
        task_of, jacobian_of, targets, solutions, residual=residual_of,
        interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q),
        sigma_floor=1e-9, collision=1e-2, merge=1e-6, max_step=args.max_step,
        distance=lambda a, b: iks.configuration_distance(a, b, circular),
    )
    arrivals = result.arrivals
    lifted = [result.lifts[k].q for k in arrivals]
    mapping = []
    for q in lifted:
        best, index = float("inf"), -1
        for k, start in enumerate(solutions):
            gap = iks.configuration_distance(start, q, circular)
            if gap < best:
                best, index = gap, k
        # an arrival further than 1e-3 rad from every starting solution is not a match: printing
        # the closest one anyway produced a "track 2 arrived at solution 3 (distance 1.4e+01 rad)"
        # line, which reads like a permutation and is not one.
        mapping.append((index if best < 1e-3 else -1, round(best, 6)))
    print(f"  arrivals {len(arrivals)}/{len(solutions)} (tracks that reached the end), "
          f"live counts {result.counts[0]} -> {result.counts[-1]}")
    for member in (i, j):
        if member in arrivals:
            slot = arrivals.index(member)
            matched = mapping[slot][0]
            print(f"  track {member} arrived at "
                  f"{'solution ' + str(matched) if matched >= 0 else 'no starting solution'}"
                  f" (closest joint-space distance {mapping[slot][1]:.1e} rad)")
        else:
            item = result.lifts[member]
            print(f"  track {member} did NOT arrive: stopped at sample {item.index} "
                  f"({item.reason})")
    print(f"  shortest gap between live tracks {min(result.gaps):.2e}; "
          f"collision samples {len(result.collisions)} of {len(result.gaps)}; "
          f"merged (tracker fault) {len(result.merged)}")
    if result.collisions:
        first = result.collisions[:6]
        print(f"  first crossing samples {first}; at those samples the loop's image is on the "
              f"discriminant: the prescribed Cartesian path cannot be tracked past them without "
              f"the posture change, i.e. the loop left the feasible region of the starting "
              f"posture")
    print(f"  rejected basin jumps {sum(item.jumps for item in result.lifts)}")
    print("  fate of every track (a fold death sits on the discriminant with a partner nearby; a "
          "tracker death does not):")
    for k, item in enumerate(result.lifts):
        if item.finished:
            continue
        sigma_here = iks.sigma_min(model, item.q)
        partner = min(
            (iks.configuration_distance(item.q, other.samples[item.index], circular)
             for other in result.lifts
             if other is not item and len(other.samples) > item.index),
            default=float("nan"),
        )
        print(f"    track {k}: stopped at sample {item.index}, sigma_min {sigma_here:.2e}, "
              f"nearest other track {partner:.2e} rad, jumps rejected {item.jumps}")

    print("\nphase 4: controls")
    degenerate = np.tile(targets[0], (targets.shape[0], 1, 1))
    control = ts.lift_fiber(
        task_of, jacobian_of, degenerate, solutions, residual=residual_of,
        interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q),
        sigma_floor=1e-9, collision=1e-2, merge=1e-6,
        distance=lambda a, b: iks.configuration_distance(a, b, circular),
    )
    moved = 0
    for k in control.arrivals:
        best = min(range(len(solutions)),
                   key=lambda m: iks.configuration_distance(solutions[m], control.lifts[k].q,
                                                            circular))
        moved += int(best != k)
    print(f"  degenerate loop (all targets equal): arrivals {len(control.arrivals)}"
          f"/{len(solutions)}, permuted {moved}, collisions {len(control.collisions)}, "
          f"merged {len(control.merged)}")
    print("  (an identity control that permutes or collides would mean the tracker, not the arm, "
          "is producing the phase-3 events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
