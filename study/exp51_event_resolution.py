#!/usr/bin/env python3
"""exp51: resolve the loop's fold events -- how many are there in one sample interval, and where?

``exp50``'s verified event table reports the loop as three death events and three birth events, but
one of them changes the fibre by four (samples 45 to 46), which two folds produce and one degenerate
event could too.  A sample interval hides that; this module removes the hiding by *subdividing the
witness polyline* in joint space: the new targets are images of ``path[k] + t (path[k+1] - path[k])``
for a ladder of ``t``, so the same tracker walks the same physical path with a finer sampling and the
events either separate or do not.

Each event that does resolve is then located in **task space** rather than in sample index.  Two
branches coalesce exactly when the pose reaches the discriminant image, so bisecting the pose
interval (``taskspace.se3_interpolate``, so every bisection point is a pose) between "the pair is
still two distinct real roots" and "it is not" brackets the crossing.  Three numbers are reported at
the bracket, none of them assumed: the pair's gap, the clearance ``sigma_min`` of the coalesced
configuration, and the exponent of the gap's vanishing along a geometric ladder towards the crossing
-- the task-path-side counterpart of the split exponent 1/2 measured on the DH side (exp27).

Phases:

1. Rebuild the measured object and track the guarded fibre over the first 47 samples: the live set
   the refinement starts from, cross-checked against an independent polished census at sample 45.
2. Subdivide every interval in which the live count changed, and re-track: the number of events
   inside the interval is read off the live counts of the refined path.
3. For every death event with exactly one pair of dead roots, bracket the crossing in task space and
   report the pair's gap, the coalesced clearance and the fitted exponent.
4. Control: subdividing a degenerate (constant) path must produce no event at all.

Run::

    python3 -m study.exp51_event_resolution --pose-seed 0
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


def guarded_lift(links, poses, starts, model, circular, jacobian_of, residual_of, *, gap=1e-2):
    """Track ``starts`` along ``poses`` with the box guard on (the repaired instrument of 3.40).

    Args:
        links: Link transforms of ``model``.
        poses: ``(K, 4, 4)`` task path.
        starts: Solutions at ``poses[0]``.
        model: The arm.
        circular: Boolean mask of continuous joints.
        jacobian_of: Pose Jacobian ``(q) -> (6, n)``.
        residual_of: Pose residual ``(q, T) -> (6,)``.
        gap: Pairwise distance below which two live tracks count as colliding.

    Returns:
        The :class:`study.taskspace.FiberLift`.
    """
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)

    def in_box(q: np.ndarray) -> bool:
        return bool(np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6))

    return ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, poses, starts,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9, collision=gap, merge=1e-6,
        distance=lambda a, b: iks.configuration_distance(a, b, circular), feasible=in_box,
        prune_merged=True,
    )


def dead_roots(before_roots, after_roots, links, pose_before, circular):
    """Which of ``before_roots`` have no continuation in the fibre at the next target.

    Args:
        before_roots: ``(configuration, [track indices])`` at the earlier sample.
        after_roots: The fibre at the later sample (polished configurations).
        links: Link transforms, for the backward corrector.
        pose_before: The earlier pose.
        circular: Boolean mask of continuous joints.

    Returns:
        The indices into ``before_roots`` of the roots that do not continue.
    """
    matched: set[int] = set()
    for q in after_roots:
        back, residual = homotopy.correct(links, q, pose_before, iterations=80, tolerance=1e-13)
        if residual > 1e-9:
            continue
        best = min(range(len(before_roots)),
                   key=lambda m: iks.configuration_distance(before_roots[m][0], back, circular))
        if iks.configuration_distance(before_roots[best][0], back, circular) < 1e-3:
            matched.add(best)
    return [m for m in range(len(before_roots)) if m not in matched]


def pair_roots(links, pose, seeds, *, dedupe=1e-6, tolerance=1e-9, jump_tolerance=0.3):
    """Real solutions near ``seeds`` at ``pose``: each seed corrected locally, then deduplicated.

    The local-correction check is what makes this a *continuation* test rather than a census: a seed
    whose corrector lands further than ``jump_tolerance`` from where it started is following another
    branch, so it is not evidence that this one is still real.
    """
    found: list[np.ndarray] = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        q, residual = homotopy.correct(links, seed, pose, iterations=80, tolerance=1e-13)
        if residual > tolerance:
            continue
        if float(np.linalg.norm(q - seed)) > jump_tolerance:
            continue
        if any(iks.torus_distance(q, other) < dedupe for other in found):
            continue
        found.append(np.asarray(q, dtype=float))
    return found


def localise_crossing(links, model, pose_a, pose_b, seeds, *, iterations=40, tolerance=1e-6):
    """Bracket the pose where the two branches through ``seeds`` stop being two real roots.

    The predicate is measured, not assumed: at a candidate pose the two seeds are continued locally
    and the answer is "two distinct roots" only if both converge and stay apart.  Bisection keeps a
    fraction where the pair is two roots and one where it is not, so the crossing is bracketed to
    ``tolerance`` of the interval.

    Args:
        links: Link transforms.
        model: The arm, for ``sigma_min``.
        pose_a: Pose where the pair is known to be two real roots.
        pose_b: Pose where it is not.
        seeds: The pair's configurations at ``pose_a``.
        iterations: Maximum bisection steps.
        tolerance: Width of the final bracket, in interval fraction.

    Returns:
        ``(lo, hi, gap, sigma, roots)``: the last fraction with two roots, the first without, the
        pair's gap at ``lo``, the smaller clearance of the two configurations there, and those
        configurations (the best available seed for the augmented refinement).
    """
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        if hi - lo <= tolerance:
            break
        mid = 0.5 * (lo + hi)
        pose = ts.se3_interpolate(pose_a, pose_b, mid)
        if len(pair_roots(links, pose, seeds)) >= 2:
            lo = mid
        else:
            hi = mid
    pose = ts.se3_interpolate(pose_a, pose_b, lo)
    roots = pair_roots(links, pose, seeds)
    if len(roots) >= 2:
        gap = min(iks.torus_distance(a, b) for a, b in itertools.combinations(roots, 2))
        sigma = min(iks.sigma_min(model, roots[0]), iks.sigma_min(model, roots[1]))
    else:
        gap = float("nan")
        sigma = float("nan")
    return lo, hi, gap, sigma, roots


def refine_fold_pose(links, model, pose_a, pose_b, q_seed, fraction, *, iterations=60,
                     tolerance=1e-12, h=1e-7):
    """Newton on the augmented system for a fold of the *task* map, in the pose parameter.

    The unknowns are ``(q, tau)`` and the equations are the six pose components plus ``det J = 0``,
    so a converged point is a singular configuration together with the pose on the discriminant image
    that it maps to.  This is the dual of exp21's refinement (there the robot moved and the pose was
    fixed); the pose here moves along the segment between two samples.

    Args:
        links: Link transforms.
        model: The arm, for ``det J``.
        pose_a: Pose at fraction 0 of the segment.
        pose_b: Pose at fraction 1.
        q_seed: Starting configuration, e.g. the midpoint of the two coalescing branches.
        fraction: Starting fraction along the segment.
        iterations: Maximum Newton steps.
        tolerance: Residual norm that counts as converged.
        h: Finite-difference step, in radians for the joints and in fraction for the pose.

    A seed close to the fold converges (measured: residual 4e-15, ``det J`` 2.6e-18, ``sigma_min``
    7.6e-17 from the bracket's own roots); a seed a little further away runs the parameter off the
    segment, and the caller is expected to retry rather than quote that.

    Returns:
        ``(fraction, q, residual, det, sigma_min)``.
    """
    q = np.asarray(q_seed, dtype=float).copy()
    tau = float(fraction)

    weight = 1e-3  # keeps tau on the event segment: without it the refinement walks the pose line

    def augmented(configuration, value):
        pose = ts.se3_interpolate(pose_a, pose_b, float(value))
        return np.concatenate([homotopy.se3_error(homotopy.fk_links(links, configuration), pose),
                               [iks.det_jacobian(model, configuration)],
                               [weight * (float(value) - float(fraction))]])

    for _ in range(iterations):
        residual = augmented(q, tau)
        size = float(np.linalg.norm(residual))
        if size < tolerance:
            break
        jacobian = np.zeros((8, 7))
        for column in range(6):
            step = np.zeros(6)
            step[column] = h
            jacobian[:, column] = (augmented(q + step, tau) - augmented(q - step, tau)) / (2 * h)
        jacobian[:7, 6] = (augmented(q, tau + h) - augmented(q, tau - h))[:7] / (2 * h)
        jacobian[7, 6] = weight
        update = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        q = q + update[:6]
        tau = float(tau + update[6])
    return tau, q, float(np.linalg.norm(augmented(q, tau))), float(iks.det_jacobian(model, q)), \
        float(iks.sigma_min(model, q))


def pair_dead_roots(roots_before, dead, model, circular):
    """Pair the dead roots of one event into fold candidates: opposite sign, closest first.

    A fold coalesces one branch from each of the two aspects it separates, so the two roots of a pair
    have opposite ``det J`` signs; among the admissible pairings the closest couple is taken first.

    Args:
        roots_before: ``(configuration, [track indices])`` at the sample before the event.
        dead: Indices into ``roots_before`` that do not continue.
        model: The arm, for ``det J``.
        circular: Boolean mask of continuous joints.

    Returns:
        ``(pairs, leftover)``: the index pairs found, and the dead roots left unpaired.
    """
    signs = {m: int(np.sign(iks.det_jacobian(model, roots_before[m][0]))) for m in dead}
    remaining = list(dead)
    pairs: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        options = [(iks.configuration_distance(roots_before[a][0], roots_before[b][0], circular),
                    a, b) for a, b in itertools.combinations(remaining, 2)
                   if signs[a] != signs[b]]
        if not options:
            break
        _, a, b = min(options)
        pairs.append((a, b))
        remaining = [m for m in remaining if m not in (a, b)]
    return pairs, remaining


def postures_of(roots_before, origin, group: int) -> list[int]:
    """Original posture labels of the track(s) holding the root at ``group`` in a subdivided run.

    The subdivided run numbers its tracks by position in the start list, and pruning can retire one,
    so a group's holders must be mapped one by one; reading the group index as a posture index names
    the wrong branch.
    """
    return sorted({p for holder in roots_before[group][1] for p in origin.get(holder, [])})


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--least", type=int, default=8)
    parser.add_argument("--tries", type=int, default=14)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--probe-seeds", type=int, default=300)
    parser.add_argument("--head", type=int, default=47,
                        help="samples of the witness path the head lift covers (past both early "
                             "death events and the -4 at 45-46)")
    parser.add_argument("--subdivisions", type=int, default=24,
                        help="sub-samples inserted into each interval the event table flags")
    parser.add_argument("--bisections", type=int, default=40)
    parser.add_argument("--fold-iterations", type=int, default=60)
    parser.add_argument("--ladder", type=int, default=10,
                        help="points on the geometric ladder towards the crossing, for the exponent")
    args = parser.parse_args()

    rng = np.random.default_rng(args.pose_seed)
    links = link_transforms()
    model = ModelFactory.create("sr5")
    circular = iks.circular_joints(model)
    residual_of, jacobian_of = pose_kinematics(links)

    print("exp51: how many folds hide inside one sample interval, and where is the crossing?\n")
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
          f"{len(path)} configurations")

    print(f"\n  phase 1: guarded lift over the first {args.head} samples (the live set the refinement "
          f"starts from)")
    head_poses = poses[:args.head]
    head = guarded_lift(links, head_poses, solutions, model, circular, jacobian_of, residual_of)
    counts = head.counts
    changes = [(k + 1, counts[k - 1], counts[k]) for k in range(1, len(counts))
               if counts[k] != counts[k - 1]]
    print(f"    live counts {counts[0]} -> {counts[-1]}, changes (sample: before -> after) {changes}")
    probe = args.head - 2
    census_probe = root_census(model, links, poses[probe], seeds=args.probe_seeds, rng=rng,
                               circular=circular)
    roots_probe = distinct_live(head.lifts, probe, circular)
    print(f"    sample {probe}: independent census {len(census_probe)} solutions, tracked distinct "
          f"roots {len(roots_probe)} (holding tracks {[holders for _, holders in roots_probe]})")

    print("\n  phase 2: subdivide every interval where the live count changed, and re-track")
    intervals = sorted({(k, k + 1) for k in range(1, len(counts)) if counts[k] != counts[k - 1]})
    intervals = [(a, b) for a, b in intervals if b < poses.shape[0]]
    for before, after in intervals:
        roots = distinct_live(head.lifts, before, circular)
        starts = [q for q, _ in roots]
        # Labels: the subdivided run numbers its tracks by position in ``starts``, which is a
        # permutation of the original (posture) indices; without this map a report would name the
        # wrong branch as the one that dies.
        origin = {index: holders for index, (_, holders) in enumerate(roots)}
        # The fine path must keep t = 0 as its first sample: the starts are the solutions there, and
        # a path that begins one sub-sample later would attribute them to the wrong target.
        ladder = np.linspace(0.0, 1.0, args.subdivisions + 1)
        fine_poses = np.array([model.fk(path[before] + t * (path[after] - path[before]))
                               for t in ladder])
        fine = guarded_lift(links, fine_poses, starts, model, circular, jacobian_of, residual_of)
        fine_counts = [len(starts)] + list(fine.counts)
        steps = [(k, fine_counts[k - 1], fine_counts[k]) for k in range(1, len(fine_counts))
                 if fine_counts[k] != fine_counts[k - 1]]
        print(f"    interval {before}->{after}: {len(starts)} starts, {args.subdivisions} "
              f"sub-samples, live {fine_counts[0]} -> {fine_counts[-1]}, changes (sub-sample: "
              f"before -> after) {steps}")
        if steps:
            verdict = ("resolved into separate folds" if len(steps) > 1
                       else "a single event at this resolution")
            print(f"      => {len(steps)} event(s) inside this interval: {verdict}")
        for sub, count_before, count_after in steps:
            pose_a = fine_poses[sub - 1]
            pose_b = fine_poses[sub]
            roots_before = distinct_live(fine.lifts, sub - 1, circular)
            census_after = root_census(model, links, pose_b, seeds=args.probe_seeds, rng=rng,
                                       circular=circular)
            dead = dead_roots(roots_before, census_after, links, pose_a, circular)
            detail = [(list(roots_before[m][1]),
                       int(np.sign(iks.det_jacobian(model, roots_before[m][0]))),
                       postures_of(roots_before, origin, m)) for m in dead]
            print(f"      sub-sample {sub}: live {count_before} -> {count_after}, census after "
                  f"{len(census_after)}, dead roots (subdivided-run track, det J sign, original "
                  f"postures) {[(h, s, o) for h, s, o in detail]}")
            if len(dead) not in (2, 4):
                print(f"        {len(dead)} dead root(s): the crossing test needs a pair, so this "
                      f"event is reported unresolved (Q24)")
                continue
            pairs, leftover = pair_dead_roots(roots_before, dead, model, circular)
            if len(dead) == 4:
                shape = [(list(roots_before[a][1]), list(roots_before[b][1])) for a, b in pairs]
                print(f"        four dead roots pair into {shape} by opposite sign; unpaired "
                      f"{[list(roots_before[m][1]) for m in leftover]}")
            folds = []
            for a, b in pairs:
                seeds_pair = [roots_before[a][0], roots_before[b][0]]
                lo, hi, gap, sigma, roots_at = localise_crossing(links, model, pose_a, pose_b,
                                                                 seeds_pair,
                                                                 iterations=args.bisections)
                print(f"        pair (postures {postures_of(roots_before, origin, a)} / {postures_of(roots_before, origin, b)}): "
                      f"crossing bracketed in [{lo:.6f}, {hi:.6f}] "
                      f"of this event interval; pair gap at the bracket {gap:.3e} rad, coalesced "
                      f"sigma_min {sigma:.3e}")
                step = lo / 2.0 if lo > 0 else 0.0
                ladders = []
                for k in range(args.ladder):
                    offset = step * 0.5 ** k
                    pose = ts.se3_interpolate(pose_a, pose_b, max(lo - offset, 0.0))
                    roots_step = pair_roots(links, pose, seeds_pair)
                    if len(roots_step) < 2:
                        continue
                    ladders.append((lo - offset, offset, min(
                        iks.torus_distance(x, y) for x, y in itertools.combinations(roots_step, 2))))
                if len(ladders) >= 3:
                    fractions = np.array([item[0] for item in ladders])
                    offsets = np.array([item[1] for item in ladders])
                    gaps = np.array([item[2] for item in ladders])
                    slope = float(np.polyfit(np.log(offsets), np.log(gaps), 1)[0])
                    # gap ~ c (f* - f)**(1/2)  =>  gap**2 is linear in f, so the crossing f* is the
                    # root of that line: an estimator independent of the bisection, and a test of the
                    # square-root law at the same time.
                    line = np.polyfit(fractions, gaps ** 2, 1)
                    tau_star = float(-line[1] / line[0]) if line[0] < 0 else float("nan")
                    print(f"          gap ladder {[(f'{f:.3f}', f'{g:.2e}') for f, _, g in ladders]}"
                          f" => exponent {slope:.3f}; squared-gap line crosses zero at tau* = "
                          f"{tau_star:.6f}")
                elif ladders:
                    print(f"          gap ladder {[(f'{f:.3f}', f'{g:.2e}') for f, _, g in ladders]}"
                          f" (too few points to fit)")
                candidates = [0.5 * (np.asarray(seeds_pair[0]) + np.asarray(seeds_pair[1]))]
                if len(roots_at) >= 2:
                    candidates.insert(0, 0.5 * (np.asarray(roots_at[0]) + np.asarray(roots_at[1])))
                refinements = [refine_fold_pose(links, model, pose_a, pose_b, seed, lo,
                                                iterations=args.fold_iterations)
                               for seed in candidates]
                tau, q_fold, residual, det, sigma_fold = min(refinements, key=lambda item: item[2])
                folds.append((tau, q_fold))
                print(f"          augmented system (pose + det J = 0): tau {tau:.6f}, residual "
                      f"{residual:.2e}, det J {det:.2e}, sigma_min {sigma_fold:.2e}, det of the two "
                      f"branches at the bracket "
                      f"{iks.det_jacobian(model, seeds_pair[0]):+.2e}/"
                      f"{iks.det_jacobian(model, seeds_pair[1]):+.2e}")
            if len(folds) == 2:
                separation = iks.torus_distance(folds[0][1], folds[1][1])
                print(f"        the two refined folds sit at tau {folds[0][0]:.6f} and "
                      f"{folds[1][0]:.6f} (separation {abs(folds[0][0] - folds[1][0]):.2e} of the "
                      f"event interval) with configurations {separation:.3e} rad apart")
    print("\n  phase 3: control (a degenerate path must not produce events)")
    starts_control = [q for q, _ in distinct_live(head.lifts, 45, circular)]
    degenerate = np.tile(poses[45], (args.subdivisions + 1, 1, 1))
    control = guarded_lift(links, degenerate, starts_control, model, circular, jacobian_of,
                           residual_of)
    print(f"    {len(starts_control)} starts, live {control.counts[0]} -> {control.counts[-1]}, "
          f"collisions {len(control.collisions)}, merged {len(control.merged)}, stops "
          f"{sum(1 for item in control.lifts if not item.finished)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
