#!/usr/bin/env python3
"""exp50: births and stops along a prescribed task path -- is a stopped branch gone, or dropped?

Two registered questions are the same question asked twice.  **Q20**: a forward whole-fibre tracker
sees only what it carries, so a crossing between two samples -- or one whose coalescing pair was
complex at the base pose -- leaves no trace, which is why two of three pose seeds reported zero
collisions while the third reported three.  **Q21's remainder**: six branches stop at *healthy*
clearance with no partner nearby, and the step bound has already been ruled out as the cause
(NOTES 3.35).  In both cases the thing to ask is the fibre itself.

The decisive question for every stop is: at the target where the track stopped, does the fibre
still have a real solution near the stopped configuration?  Perturbed seeds around that
configuration answer it locally, which is where the answer lives -- a global census would answer a
different question (whether the pose is reachable at all, which it is).

Phases:

1. Rebuild exp49's measured object: a pose from ``rich_pose``, a certified same-aspect pair
   (witnessed clearance path), and the loop its projection traces.
2. Lift the whole fibre in lockstep (:mod:`study.taskspace`) and record live counts, the smallest
   pairwise gap per sample -- the crossing candidates Q20 wants, where two branches come together --
   and every stop.
3. Classify every stop with a local census at the *next* target: at least one real solution within
   0.2 rad means the branch continued and the tracker dropped it; none means the branch ceased to be
   real between the two samples, i.e. a fold the samples straddle.
4. Controls: the degenerate loop (all targets equal) must be the identity with no collisions and no
   stops, or the events belong to the tracker.

Run::

    python3 -m study.exp50_births_and_stops --pose-seed 0
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from model import ModelFactory
from study import homotopy
from study import ik_structure as iks
from study import taskspace as ts
from study.census import solve_lm
from study.chamber import clearance_path
from study.exp11_dh_continuation import link_transforms
from study.exp49_sr5_uniqueness import pose_kinematics, rich_pose


def same_aspect_pair(model, solutions, signs, *, delta, circular, steps=1500):
    """The first same-sign pair a witnessed clearance path connects, with that path.

    Returns:
        ``(i, j, path)`` or ``None``.  A failure here is the planner's, not the arm's (Q22), which
        is why the caller reports it as "no pair resolved" rather than as a negative result.
    """
    for i, j in itertools.combinations(range(len(solutions)), 2):
        if signs[i] != signs[j]:
            continue
        walk = clearance_path(model, solutions[i], solutions[j], delta=delta, circular=circular,
                              steps=steps)
        if walk.connected:
            return i, j, np.array(walk.path)
    return None


def local_census(model, target, q_center, *, radius=0.2, seeds=200, spread=0.3, rng=None):
    """Real solutions of the fibre over ``target`` within ``radius`` of ``q_center``.

    Args:
        model: The arm.
        target: The pose (4x4).
        q_center: The configuration whose neighbourhood is being probed, radians.
        radius: Torus distance that counts as "near", radians.
        seeds: Random perturbations of ``q_center`` used as solver seeds.
        spread: Half-width of the perturbation box, radians.
        rng: Generator; seeded at 0 when omitted.

    Returns:
        A list of distinct solutions inside the radius.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    found: list[np.ndarray] = []
    for _ in range(seeds):
        seed = q_center + rng.uniform(-spread, spread, np.asarray(q_center).size)
        q, converged = solve_lm(model, seed, target)
        if not converged:
            continue
        if iks.torus_distance(q, q_center) > radius:
            continue
        if any(iks.torus_distance(q, other) < 1e-3 for other in found):
            continue
        found.append(np.asarray(q, dtype=float))
    return found


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--least", type=int, default=8)
    parser.add_argument("--tries", type=int, default=14)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--gap", type=float, default=1e-2,
                        help="pairwise gap below which a sample counts as a crossing candidate")
    parser.add_argument("--radius", type=float, default=0.2,
                        help="neighbourhood radius of the local-census stop test")
    parser.add_argument("--census-seeds", type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(args.pose_seed)
    links = link_transforms()
    model = ModelFactory.create("sr5")
    circular = iks.circular_joints(model)

    print("exp50: are the stops folds or tracker deaths, and where are the crossings?\n")
    target, solutions = rich_pose(model, rng, seeds=args.seeds, least=args.least, tries=args.tries)
    if not solutions:
        print("  no pose with a rich fibre in this sample")
        return 0
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
    resolved = same_aspect_pair(model, solutions, signs, delta=args.delta, circular=circular)
    if resolved is None:
        print(f"  no same-aspect pair resolved by the clearance walk at this pose "
              f"({len(solutions)} solutions, signs {[signs.count(1), signs.count(-1)]}) -- Q22, "
              f"a planner limit, not a result")
        return 0
    i, j, path = resolved
    poses = np.array([model.fk(q) for q in path])
    print(f"  pose: {len(solutions)} solutions, signs {[signs.count(1), signs.count(-1)]}; "
          f"pair ({i}, {j}), witness {len(path)} configurations")

    residual_of, jacobian_of = pose_kinematics(links)
    result = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, poses, solutions,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9,
        collision=args.gap, merge=1e-6,
        distance=lambda a, b: iks.configuration_distance(a, b, circular),
    )
    live = result.counts
    candidates = [k for k, value in enumerate(result.gaps) if value < args.gap]
    print(f"\n  whole-fibre lift: arrivals {len(result.arrivals)}/{len(solutions)}, live {live[0]} -> "
          f"{live[-1]}, shortest gap {min(result.gaps):.2e}")
    print(f"  crossing candidates (gap < {args.gap:g}): {len(candidates)} of {len(result.gaps)} "
          f"samples {candidates[:10]}")
    stopped = [k for k, item in enumerate(result.lifts) if not item.finished]
    print(f"  stopped tracks: {len(stopped)} {stopped}")

    print("\n  stop classification (local census of the *next* target's fibre near the stopped "
          "configuration):")
    folds = drops = 0
    for k in stopped:
        item = result.lifts[k]
        nxt = item.index + 1
        here = local_census(model, poses[item.index], item.q, radius=args.radius,
                            seeds=args.census_seeds, rng=rng)
        there = (local_census(model, poses[nxt], item.q, radius=args.radius,
                              seeds=args.census_seeds, rng=rng) if nxt < poses.shape[0] else [])
        sigma_here = iks.sigma_min(model, item.q)
        nearest = min((iks.configuration_distance(item.q, other.samples[item.index], circular)
                       for other in result.lifts
                       if other is not item and len(other.samples) > item.index),
                      default=float("nan"))
        verdict = "TRACKER DROP" if there else "FOLD (branch gone)"
        folds += int(not there)
        drops += int(bool(there))
        print(f"    track {k:>2}: stopped at sample {item.index:>3}, sigma_min {sigma_here:.2e}, "
              f"nearest other {nearest:.2e} rad | solutions near it now {len(here)}, at next "
              f"target {len(there)} => {verdict}")
    print(f"  verdict counts: {folds} fold(s), {drops} tracker drop(s)")

    print("\n  control: degenerate loop (all targets equal)")
    degenerate = np.tile(poses[0], (poses.shape[0], 1, 1))
    control = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, degenerate, solutions,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9, collision=args.gap,
        merge=1e-6, distance=lambda a, b: iks.configuration_distance(a, b, circular),
    )
    moved = 0
    for k in control.arrivals:
        best = min(range(len(solutions)),
                   key=lambda m: iks.configuration_distance(solutions[m], control.lifts[k].q,
                                                            circular))
        moved += int(best != k)
    print(f"    arrivals {len(control.arrivals)}/{len(solutions)}, permuted {moved}, "
          f"collisions {len(control.collisions)}, stops {len(solutions) - len(control.arrivals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
