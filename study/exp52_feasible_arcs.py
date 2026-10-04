#!/usr/bin/env python3
"""exp52: the feasible-path arcs of the loop's postures, with both ends located.

exp51 established that each discriminant crossing can be bracketed in task space and certified
(gap -> 0, ``sigma_min`` -> 0, square-root vanishing, and for one of them a singular configuration
from the augmented system).  This module turns that into the answer to the A->B planning question in
its operational form: **run the whole loop once per posture and say how much of it that posture can
track** -- not in sample counts but as a fraction of the loop's *Cartesian* length -- and which
located crossing ends its arc.  A posture's arc is exactly what Wenger's theory calls the image of
its uniqueness domain: the set of poses reachable from it without leaving the domain, so its end is
where the prescribed path leaves that image.

Track indices equal base solution indices here because nothing is admitted mid-path, which is what
makes the arc table a statement about *named* postures rather than about anonymous tracks.

Phases:

1. Build exp50's measured object and run the guarded, pruned whole-fibre lift over the whole loop.
2. For every interval in which a track dies, pair the dead roots (opposite ``det J`` sign), bracket
   the crossing in task space, and report the two zero certificates, the square-root fit and the
   augmented-system point where it converges.
3. The arc table: posture -> samples tracked -> fraction of the loop's Cartesian length -> the
   crossing that ends it (or "whole loop").
4. Control: on a degenerate (constant) path every posture must track everything.

Run::

    python3 -m study.exp52_feasible_arcs --pose-seed 0
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from model import ModelFactory
from study import homotopy
from study import ik_structure as iks
from study import taskspace as ts
from study.exp11_dh_continuation import link_transforms
from study.exp49_sr5_uniqueness import pose_kinematics, rich_pose
from study.exp50_births_and_stops import distinct_live, root_census, same_aspect_pair
from study.exp51_event_resolution import (
    dead_roots,
    guarded_lift,
    localise_crossing,
    pair_dead_roots,
    pair_roots,
    refine_fold_pose,
)


def path_lengths(poses: np.ndarray) -> np.ndarray:
    """Cumulative Cartesian length of the pose path, in the units of ``se3_error``.

    Args:
        poses: ``(K, 4, 4)`` pose samples.

    Returns:
        A ``(K,)`` array whose entry ``k`` is the length from sample 0 to sample ``k``.
    """
    steps = np.zeros(poses.shape[0])
    for k in range(1, poses.shape[0]):
        steps[k] = float(np.linalg.norm(homotopy.se3_error(poses[k - 1], poses[k])))
    return np.cumsum(steps)


def march_to_fold(links, pose_a, pose_b, seeds, *, steps=64):
    """March the pair's two seeds towards the crossing in small pose steps.

    The tracker's own last live sample can be up to a whole interval before the crossing, and a
    bisection seeded that far away fails *early* (the corrector's basin shrinks near the fold), which
    biases the bracket late -- measured: the same two folds came out at 45.3847/45.3948 from
    interval-level seeds against 45.3792/45.3894 from sub-interval-level ones in exp51.  Walking the
    seeds forward in ``steps`` substeps with the same corrector gives the bisection a seed adjacent to
    the fold without re-running the tracker.

    Args:
        links: Link transforms.
        pose_a: Pose where the pair is known real.
        pose_b: Pose where it is not.
        seeds: The pair's configurations at ``pose_a``.
        steps: Substeps of the march.

    Returns:
        ``(fraction_lo, fraction_hi, roots_lo)``: the last fraction of the interval where both seeds
        continued, the next one, and the pair's configurations at ``fraction_lo``.
    """
    current = [np.asarray(seed, dtype=float) for seed in seeds]
    fraction_lo, roots_lo = 0.0, list(current)
    for k in range(1, steps + 1):
        fraction = k / steps
        pose = ts.se3_interpolate(pose_a, pose_b, fraction)
        roots = pair_roots(links, pose, current)
        if len(roots) < 2:
            return fraction_lo, fraction, roots_lo
        current, fraction_lo, roots_lo = roots, fraction, roots
    return fraction_lo, 1.0, roots_lo


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--least", type=int, default=8)
    parser.add_argument("--tries", type=int, default=14)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--probe-seeds", type=int, default=300)
    parser.add_argument("--bisections", type=int, default=40)
    parser.add_argument("--subdivisions", type=int, default=64,
                        help="march substeps between an interval's ends, so the bisection is seeded "
                             "adjacent to the fold instead of a whole interval away")
    parser.add_argument("--ladder", type=int, default=10)
    parser.add_argument("--fold-iterations", type=int, default=60)
    args = parser.parse_args()

    rng = np.random.default_rng(args.pose_seed)
    links = link_transforms()
    model = ModelFactory.create("sr5")
    circular = iks.circular_joints(model)
    residual_of, jacobian_of = pose_kinematics(links)

    print("exp52: how much of the loop can each posture track, and where does its arc end?\n")
    target, solutions = rich_pose(model, rng, seeds=args.seeds, least=args.least, tries=args.tries)
    if not solutions:
        print("  no pose with a rich fibre in this sample")
        return 0
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
    resolved = same_aspect_pair(model, solutions, signs, delta=args.delta, circular=circular)
    if resolved is None:
        print(f"  no same-aspect pair resolved by the clearance walk at this pose ({len(solutions)} "
              f"solutions) -- Q22, a planner limit, not a result")
        return 0
    i, j, path = resolved
    poses = np.array([model.fk(q) for q in path])
    print(f"  pose: {len(solutions)} solutions (signs {signs.count(1)}/{signs.count(-1)}), witness "
          f"{len(path)} configurations; tracked posture indices equal base solution indices")

    whole = guarded_lift(links, poses, solutions, model, circular, jacobian_of, residual_of)
    counts = whole.counts
    print(f"  whole-fibre lift (guarded, pruned): live {counts[0]} -> {counts[-1]}, stops "
          f"{sum(1 for item in whole.lifts if not item.finished)}, collisions "
          f"{len(whole.collisions)}, merged {len(whole.merged)}, duplicates "
          f"{len(whole.duplicates)}")

    lengths = path_lengths(poses)
    total = float(lengths[-1])
    print(f"  the loop's Cartesian length is {total:.4f} (sum of se3_error steps); "
          f"{poses.shape[0] - 1} intervals")

    print("\n  the crossings that end the arcs (bracketed in task space, certificates measured)")
    crossings: dict[int, dict] = {}
    intervals = sorted({(item.index, item.index + 1) for item in whole.lifts if not item.finished
                        and item.index + 1 < poses.shape[0]})
    for before, after in intervals:
        roots_before = distinct_live(whole.lifts, before, circular)
        census_after = root_census(model, links, poses[after], seeds=args.probe_seeds, rng=rng,
                                   circular=circular)
        dead = dead_roots(roots_before, census_after, links, poses[before], circular)
        pairs, leftover = pair_dead_roots(roots_before, dead, model, circular)
        print(f"    interval {before}->{after}: {len(roots_before)} roots -> census "
              f"{len(census_after)}, dead {[list(roots_before[m][1]) for m in dead]}, leftover "
              f"{[list(roots_before[m][1]) for m in leftover]}")
        for a, b in pairs:
            seeds_pair = [roots_before[a][0], roots_before[b][0]]
            march_lo, march_hi, seeds_lo = march_to_fold(links, poses[before], poses[after],
                                                         seeds_pair, steps=args.subdivisions)
            pose_lo = ts.se3_interpolate(poses[before], poses[after], march_lo)
            pose_hi = ts.se3_interpolate(poses[before], poses[after], march_hi)
            # ``lo``/``hi`` are fractions of the *marched sub-interval*, whose width is
            # ``march_hi - march_lo``; the sample position needs both conversions.
            width = march_hi - march_lo
            base = before + march_lo
            lo, hi, gap, sigma, roots_at = localise_crossing(links, model, pose_lo, pose_hi,
                                                             seeds_lo, iterations=args.bisections)
            step = lo / 2.0 if lo > 0 else 0.0
            ladder = []
            for k in range(args.ladder):
                offset = step * 0.5 ** k
                roots_step = pair_roots(links, ts.se3_interpolate(pose_lo, pose_hi,
                                                                 max(lo - offset, 0.0)), seeds_lo)
                if len(roots_step) < 2:
                    continue
                ladder.append((offset, min(iks.torus_distance(x, y)
                                           for x, y in itertools.combinations(roots_step, 2))))
            exponent = float("nan")
            if len(ladder) >= 3:
                offsets = np.array([item[0] for item in ladder])
                gaps = np.array([item[1] for item in ladder])
                exponent = float(np.polyfit(np.log(offsets), np.log(gaps), 1)[0])
            tau = base + lo * width
            tau_star = float("nan")
            if len(ladder) >= 3:
                fractions = np.array([tau - item[0] * width for item in ladder])
                squares = np.array([item[1] for item in ladder]) ** 2
                line = np.polyfit(fractions, squares, 1)
                if line[0] < 0:
                    tau_star = float(-line[1] / line[0])
            seed_mid = (0.5 * (np.asarray(seeds_pair[0]) + np.asarray(seeds_pair[1]))
                        if len(roots_at) < 2
                        else 0.5 * (np.asarray(roots_at[0]) + np.asarray(roots_at[1])))
            folds = [refine_fold_pose(links, model, pose_lo, pose_hi, seed, lo,
                                      iterations=args.fold_iterations)
                     for seed in (seed_mid, 0.5 * (np.asarray(seeds_pair[0])
                                                   + np.asarray(seeds_pair[1])))]
            tau_fold, _q_fold, residual, det, sigma_fold = min(folds, key=lambda item: item[2])
            print(f"      pair (tracks {list(roots_before[a][1])} / {list(roots_before[b][1])}): "
                  f"bracket [{tau:.6f}, {base + hi * width:.6f}] samples, gap {gap:.3e} rad, "
                  f"sigma_min {sigma:.3e}, exponent {exponent:.3f}, gap^2 zero at "
                  f"{tau_star:.6f}, augmented tau {base + tau_fold * width:.6f} (residual "
                  f"{residual:.1e}, det J {det:.1e}, sigma_min {sigma_fold:.1e})")
            for m, partner in ((a, b), (b, a)):
                for holder in roots_before[m][1]:
                    crossings[holder] = {"tau": tau, "gap": gap, "sigma": sigma,
                                         "exponent": exponent, "tau_star": tau_star,
                                         "with": list(roots_before[partner][1])}

    print("\n  the feasible arcs (A->B: each posture owns the arc it can track without a branch change)")
    print("    posture | sign | arc (samples) | Cartesian fraction | ends at")
    for k, item in enumerate(whole.lifts):
        if k >= len(solutions):
            continue
        end = item.index
        fraction = float(lengths[end] / total)
        info = crossings.get(k)
        if info is None:
            where = "whole loop"
        else:
            where = (f"crossing at sample {info['tau']:.4f} (gap {info['gap']:.1e}, "
                     f"sigma_min {info['sigma']:.1e}), paired with posture {info['with']}")
        print(f"    {k:>7} | {int(np.sign(iks.det_jacobian(model, solutions[k]))):>4} | "
              f"0..{end:>3} | {100 * fraction:>17.1f}% | {where}")
    print("\n  control: degenerate loop (constant target)")
    degenerate = np.tile(poses[0], (poses.shape[0], 1, 1))
    control = guarded_lift(links, degenerate, solutions, model, circular, jacobian_of, residual_of)
    print(f"    live {control.counts[0]} -> {control.counts[-1]}, stops "
          f"{sum(1 for item in control.lifts if not item.finished)}, collisions "
          f"{len(control.collisions)}, merged {len(control.merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
