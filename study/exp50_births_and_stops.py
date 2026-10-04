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
5. Carry the births forward: admit every real solution the carried tracks do not cover, classify it
   as a birth (no backward continuation to the previous sample) or as a dropped branch, and follow
   it like any other track to the end of the path -- which turns the blind spot into an event table
   and tests the loop's closure on the whole fibre at once.

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
from study.census import census, make_lm_solver, solve_lm
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


def sample_births(tracked, poses, model, *, seeds, rng, merge=1e-3):
    """Per-sample solutions that no tracked branch occupies -- the births a tracker cannot see.

    A forward lift can only follow what it already carries, so a pair of branches that becomes real
    *during* the path (a birth at a fold) leaves no trace in the live counts: this is exactly the
    blindness Q20 records, and the cure is to reseed every sample from scratch and see what is there
    that the carried tracks do not cover.

    Args:
        tracked: The :class:`study.taskspace.Lift` objects of the run.
        poses: The ``(K, 4, 4)`` task path.
        model: The arm.
        seeds: Random restarts per sample.
        rng: Generator.
        merge: Torus distance below which two configurations are one solution.

    Returns:
        A list of ``(sample index, number of new solutions)`` for the samples that have any.
    """
    births: list[tuple[int, int]] = []
    for index in range(poses.shape[0]):
        live = [item.samples[index] for item in tracked if len(item.samples) > index]
        fresh: list[np.ndarray] = []
        for _ in range(seeds):
            q, converged = solve_lm(model, rng.uniform(-np.pi, np.pi, model.num_dof),
                                    poses[index])
            if not converged:
                continue
            if any(iks.torus_distance(q, other) < merge for other in live):
                continue
            if any(iks.torus_distance(q, other) < merge for other in fresh):
                continue
            fresh.append(np.asarray(q, dtype=float))
        if fresh:
            births.append((index, len(fresh)))
    return births


def root_census(model, links, pose, *, seeds, rng, circular, dedupe=1e-6, tolerance=1e-9):
    """Distinct real solutions at ``pose`` from fresh seeds, every root Newton-polished first.

    ``census``'s own deduplication cannot simply be tightened here: two converged copies of one root
    differ by the solver's scatter, so a fine tolerance chains them (measured: 83 to 109 "solutions"
    for a 10-solution pose at ``dedupe_tol=1e-6``) while a loose one merges genuinely distinct roots.
    Strict Newton steps collapse the scatter first, and only then does a distance between two roots
    say something about the fibre.  This is the independent ground truth against which the tracker's
    live count is read -- a forward lift can carry duplicates after a fold jump, and the fibre cannot.

    Args:
        model: The arm.
        links: Its link transforms (for the polisher).
        pose: The 4x4 pose to solve for.
        seeds: Fresh multi-start seeds.
        rng: Generator for the seeds.
        circular: Boolean mask of continuous joints (the torus distance).
        dedupe: Torus distance below which two polished roots are one solution.
        tolerance: Pose residual above which a polished root is discarded.

    Returns:
        The distinct polished solutions.
    """
    fiber = census(model, pose, solver=make_lm_solver(model), seeds=seeds, rng=rng,
                   dedupe_tol=1e-3, circular=circular)
    found: list[np.ndarray] = []
    for candidate in fiber.solutions:
        q, residual = homotopy.correct(links, np.asarray(candidate, dtype=float), pose,
                                       iterations=80, tolerance=1e-13)
        if residual > tolerance:
            continue
        if any(iks.configuration_distance(q, other, circular) < dedupe for other in found):
            continue
        found.append(np.asarray(q, dtype=float))
    return found


def distinct_live(tracked, index, circular, *, dedupe=1e-3):
    """The live configurations at one sample, merged into distinct roots, with their track indices.

    Args:
        tracked: The :class:`study.taskspace.Lift` objects.
        index: Sample index.
        circular: Boolean mask of continuous joints.
        dedupe: Torus distance below which two tracks hold the same root.

    Returns:
        A list of ``(configuration, [track indices holding it])``.
    """
    groups: list[tuple[np.ndarray, list[int]]] = []
    for k, item in enumerate(tracked):
        if item.retired:
            continue
        offset = index - item.born
        if not 0 <= offset < len(item.samples):
            continue
        q = np.asarray(item.samples[offset], dtype=float)
        for known, holders in groups:
            if iks.configuration_distance(q, known, circular) < dedupe:
                holders.append(k)
                break
        else:
            groups.append((q, [k]))
    return groups


class FiberAdmitter:
    """Admit the solutions a lift does not carry, and say whether each is a birth or a drop.

    This is Q23's remainder.  Reseeding alone (``sample_births``) measures the blind spot; admitting
    the uncovered solutions into the tracker and carrying them forward turns the blind spot into an
    *event table*: the live count stops being monotone, a birth is dated to the sample where the
    pair becomes real, and the track that appeared can be followed to the end of the path like any
    other.  It also repairs the one class of error a forward tracker cannot notice in itself -- a
    branch that is still real but was dropped (Q21).

    Classification is structural, not statistical.  A candidate at sample ``k`` is continued
    *backwards* one sample with the same residual convention (``homotopy.correct``,
    ``se3_error(FK(q), target)``): if the corrector reaches the previous target without a basin jump
    the branch was already real there -- the candidate either continues a track that was dropped or
    one the previous census missed -- whereas a corrector that cannot get there has run into the
    discriminant *between* the samples, i.e. the pair was born there.  A sample whose birth count is
    odd is reported as such: real folds create solutions in pairs, so an odd count means the census
    found only one of the two.

    Args:
        model: The arm.
        poses: The ``(K, 4, 4)`` task path.
        links: The link transforms of ``model`` (for the backward corrector).
        restarts: Random restarts per sample, on top of the previous sample's solutions.
        merge: Torus distance below which two configurations are one solution.
        rng: Generator.
        jump_tolerance: Backward-step distance above which the step is a basin jump, not a
            continuation.
        tolerance: Backward residual above which the branch is not real at the previous sample.
    """

    def __init__(self, model, poses, links, *, restarts, merge, rng, jump_tolerance=0.3,
                 tolerance=1e-9):
        self.model = model
        self.poses = poses
        self.links = links
        self.restarts = restarts
        self.merge = merge
        self.rng = rng
        self.jump_tolerance = jump_tolerance
        self.tolerance = tolerance
        #: Every real solution known at the previous sample: the census seeds.
        self.known: list[np.ndarray] = []
        #: Configurations alive at the previous sample, for the classification.
        self.previous_alive: list[np.ndarray] = []
        #: One record per admission: sample, kind, configuration and the backward step's numbers.
        self.events: list[dict] = []
        #: Per-sample ``(index, live before admission, admitted, births, drops, odd)``.
        self.trace: list[tuple[int, int, int, int, int, bool]] = []

    def __call__(self, index: int, alive: list[np.ndarray]) -> list[np.ndarray]:
        """Hook for :func:`study.taskspace.lift_fiber`; returns configurations to admit."""
        found = self._uncovered(index, alive)
        admitted: list[np.ndarray] = []
        births = drops = 0
        for q in found:
            kind, residual, travelled = self._classify(index, q)
            births += int(kind == "birth")
            drops += int(kind == "existed")
            self.events.append({"sample": index, "kind": kind, "q": q, "residual": residual,
                                "travel": travelled})
            admitted.append(q)
        self.trace.append((index, len(alive), len(admitted), births, drops, bool(births % 2)))
        self.previous_alive = [np.asarray(q, dtype=float) for q in alive]
        self.known = self.previous_alive + [np.asarray(q, dtype=float) for q in admitted]
        return admitted

    def _uncovered(self, index: int, alive: list[np.ndarray]) -> list[np.ndarray]:
        """Real solutions at ``poses[index]`` that no alive track (and no earlier candidate) covers."""
        target = self.poses[index]
        seen = [np.asarray(q, dtype=float) for q in alive]
        found: list[np.ndarray] = []
        seeds = list(self.known) + [self.rng.uniform(-np.pi, np.pi, self.model.num_dof)
                                    for _ in range(self.restarts)]
        for seed in seeds:
            q, converged = solve_lm(self.model, seed, target)
            if not converged:
                continue
            q = np.asarray(q, dtype=float)
            if any(iks.torus_distance(q, other) < self.merge for other in seen):
                continue
            seen.append(q)
            found.append(q)
        return found

    def _classify(self, index: int, q: np.ndarray) -> tuple[str, float, float]:
        """Continue ``q`` backwards one sample: a reachable predecessor means it is not a birth."""
        if index == 0:
            return "seed", 0.0, 0.0
        back, residual = homotopy.correct(self.links, q, self.poses[index - 1], iterations=60,
                                          tolerance=1e-11)
        travelled = float(np.linalg.norm(np.asarray(back, dtype=float) - q))
        if residual <= self.tolerance and travelled <= self.jump_tolerance:
            return "existed", residual, travelled
        return "birth", residual, travelled


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
    parser.add_argument("--sample-seeds", type=int, default=120,
                        help="random restarts per sample for the per-sample birth census (Q23 step 2)")
    parser.add_argument("--merge", type=float, default=1e-3,
                        help="torus distance below which two configurations count as one in the "
                             "birth census; run 1e-2 against 1e-3 to separate genuine new solutions "
                             "from census twins")
    parser.add_argument("--control-stride", type=int, default=8,
                        help="the degenerate control keeps every stride-th target: it is constant, "
                             "so all samples ask the same question, and the hook must find nothing")
    parser.add_argument("--probe-seeds", type=int, default=300,
                        help="fresh multi-start seeds for the ground-truth census at the event "
                             "samples (the tracker's live count can carry duplicates)")
    parser.add_argument("--box-guard", action="store_true",
                        help="reject corrections outside the joint box (a track can otherwise walk "
                             "a limited joint past its stop; off by default so that the recorded "
                             "Q20/Q21 numbers stay reproducible with the instrument that made them)")
    parser.add_argument("--prune-merged", action="store_true",
                        help="retire a live track that coincides with an older one: two distinct "
                             "branches cannot be the same configuration, so the newer one is a jump "
                             "artefact left by a fold crossing")
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
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)

    def in_box(q: np.ndarray) -> bool:
        return bool(np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6))

    def outside_box(q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float)
        return float(max(np.max(lower - q, initial=0.0), np.max(q - upper, initial=0.0)))

    guard = in_box if args.box_guard else None
    result = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, poses, solutions,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9,
        collision=args.gap, merge=1e-6,
        distance=lambda a, b: iks.configuration_distance(a, b, circular), feasible=guard,
        prune_merged=args.prune_merged,
    )
    live = result.counts
    candidates = [k for k, value in enumerate(result.gaps) if value < args.gap]
    print(f"\n  whole-fibre lift: arrivals {len(result.arrivals)}/{len(solutions)}, live {live[0]} -> "
          f"{live[-1]}, shortest gap {min(result.gaps):.2e}, box guard {'on' if guard else 'off'}")
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

    print("\n  per-sample reseeding (Q23 step 2): solutions no carried track occupies")
    births = sample_births(result.lifts, poses, model, seeds=args.sample_seeds, rng=rng,
                           merge=args.merge)
    runs = []
    for index, _ in births:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    print(f"    samples with unoccupied solutions (merge {args.merge:g} rad): {len(births)} of "
          f"{poses.shape[0]}; total {sum(count for _, count in births)}")
    print(f"    contiguous runs of such samples: {len(runs)} "
          f"{[tuple(pair) for pair in runs[:8]]}")
    if births:
        print(f"    first samples {[index for index, _ in births[:12]]} "
              f"(a birth here is a pair becoming real, which the live counts cannot show)")
    print("\n  control: degenerate loop (all targets equal)")
    degenerate = np.tile(poses[0], (poses.shape[0], 1, 1))
    control = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, degenerate, solutions,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9, collision=args.gap,
        merge=1e-6, distance=lambda a, b: iks.configuration_distance(a, b, circular),
        feasible=guard,
    )
    moved = 0
    for k in control.arrivals:
        best = min(range(len(solutions)),
                   key=lambda m: iks.configuration_distance(solutions[m], control.lifts[k].q,
                                                            circular))
        moved += int(best != k)
    print(f"    arrivals {len(control.arrivals)}/{len(solutions)}, permuted {moved}, "
          f"collisions {len(control.collisions)}, stops {len(solutions) - len(control.arrivals)}")

    print("\n  births admitted and carried forward (Q23's remainder): every real solution the carried "
          "tracks do not cover is added as a track, classified, and followed to the end")
    admitter = FiberAdmitter(model, poses, links, restarts=args.sample_seeds, merge=args.merge,
                             rng=rng)
    admitter.known = [np.asarray(q, dtype=float) for q in solutions]
    carried = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, poses, solutions,
        residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9,
        collision=args.gap, merge=1e-6,
        distance=lambda a, b: iks.configuration_distance(a, b, circular), admit=admitter,
        feasible=guard, prune_merged=args.prune_merged,
    )
    counts = carried.counts
    runs = []
    for offset, count in enumerate(counts):
        index = offset + 1  # ``counts`` starts at sample 1: sample 0 is the starting fibre
        if runs and runs[-1][2] == count:
            runs[-1][1] = index
        else:
            runs.append([index, index, count])
    print(f"    live count is no longer monotone: {counts[0]} -> {min(counts)} -> {counts[-1]}; "
          f"runs (first..last: count) {[(a, b, c) for a, b, c in runs]}")
    admitted = [entry for entry in admitter.trace if entry[2]]
    births = [event for event in admitter.events if event["kind"] == "birth"]
    drops = [event for event in admitter.events if event["kind"] == "existed"]
    odd = [entry for entry in admitter.trace if entry[5]]
    print(f"    admissions at {len(admitted)} of {poses.shape[0] - 1} samples: "
          f"{[(entry[0], entry[2]) for entry in admitted]}")
    print(f"    classified: {len(births)} birth(s) (no backward continuation to the previous "
          f"sample), {len(drops)} dropped branch(es) re-admitted")
    if births:
        back = np.array([event["residual"] for event in births])
        born = np.array([event["residual"] for event in drops]) if drops else np.array([np.nan])
        print(f"    backward-continuation residual: births {back.min():.2e}..{back.max():.2e}, "
              f"re-admitted {born.min():.2e}..{born.max():.2e} (tolerance "
              f"{admitter.tolerance:.0e})")
    print(f"    samples with an ODD birth count (a pair minus a census miss): {len(odd)} "
          f"{[entry[0] for entry in odd]}")
    print(f"    lift faults under the hook: collisions {len(carried.collisions)}, merged "
          f"{len(carried.merged)} (a merge would mean two tracks became one)")
    fault_runs = []
    for index in carried.merged:
        pair = carried.closest[index - 1]
        if fault_runs and index == fault_runs[-1][1] + 1 and pair == fault_runs[-1][2]:
            fault_runs[-1][1] = index
        else:
            fault_runs.append([index, index, pair])
    if fault_runs:
        print(f"    merged samples by pair (first..last: tracks) "
              f"{[(a, b, list(pair) if pair else None) for a, b, pair in fault_runs]}")
        print("    a pair that merges and then stays merged is a tracker fault: a real fold takes "
              "the branch out between two samples")
    if carried.duplicates:
        print(f"    duplicates retired (sample, kept track, retired track): {carried.duplicates}")
    print("\n  fibre census at the event samples (fresh seeds, polished roots: the tracker's live "
          "count is not the fibre)")
    events = sorted({0, poses.shape[0] - 1}
                    | {item.index for item in carried.lifts if not item.finished}
                    | {entry["sample"] for entry in admitter.events}
                    | {item.index + 1 for item in carried.lifts if not item.finished}
                    | {entry["sample"] + 1 for entry in admitter.events})
    events = [index for index in events if 0 <= index < poses.shape[0]]
    real: dict[int, list[np.ndarray]] = {}
    live_distinct: dict[int, list[tuple[np.ndarray, list[int]]]] = {}
    for index in events:
        real[index] = root_census(model, links, poses[index], seeds=args.probe_seeds, rng=rng,
                                  circular=circular)
        live_distinct[index] = distinct_live(carried.lifts, index, circular)
        signs = [int(np.sign(iks.det_jacobian(model, q))) for q, _ in live_distinct[index]]
        escaped = sum(1 for q, _ in live_distinct[index] if outside_box(q) > 1e-6)
        print(f"    sample {index:>3}: fibre {len(real[index]):>2}, live distinct "
              f"{len(live_distinct[index]):>2} (tracks {[holders for _, holders in live_distinct[index]]})"
              f", signs {[signs.count(1), signs.count(-1)]}, live outside the joint box {escaped}")
    print("\n  which postures each fold kills (the fibre after the event, continued back one sample)")
    intervals = sorted({(item.index, item.index + 1) for item in carried.lifts if not item.finished
                        and item.index + 1 < poses.shape[0]})
    for before, after in intervals:
        groups = live_distinct[before]
        matched: set[int] = set()
        continued = 0
        for q in real[after]:
            back, residual = homotopy.correct(links, q, poses[before], iterations=80,
                                              tolerance=1e-13)
            if residual > 1e-9:
                continue
            continued += 1
            best = min(range(len(groups)),
                       key=lambda m: iks.configuration_distance(groups[m][0], back, circular))
            if iks.configuration_distance(groups[best][0], back, circular) < 1e-3:
                matched.add(best)
        dead = [m for m in range(len(groups)) if m not in matched]
        details = [(list(groups[m][1]),
                    int(np.sign(iks.det_jacobian(model, groups[m][0]))),
                    float(iks.sigma_min(model, groups[m][0]))) for m in dead]
        print(f"    between samples {before} and {after}: fibre {len(groups)} -> {len(real[after])}"
              f" ({continued} roots continued back); dead roots (holding tracks, det J sign, "
              f"sigma_min) {[(h, s, f'{v:.1e}') for h, s, v in details]}")
    print(f"    balance: the fibre size must return to its start value on a closed path; measured "
          f"{len(solutions)} at sample 0 and {len(real.get(poses.shape[0] - 1, []))} at the last "
          f"sample (the witness path's last configuration is {np.abs(poses[-1] - poses[0]).max():.1e} "
          f"away from the first pose, so the last sample is a *nearby* pose, not the same one)")
    ends = [item for item in carried.lifts if item.index == poses.shape[0] - 1]
    print(f"    closure: {len(ends)} live track(s) at the final sample, against {len(solutions)} "
          f"base solutions, {sum(1 for item in ends if outside_box(item.q) > 1e-6)} of them outside "
          f"the joint box")
    to_base = [min(iks.configuration_distance(item.q, q, circular) for q in solutions)
               for item in ends]
    pairwise = min((iks.configuration_distance(ends[a].q, ends[b].q, circular)
                    for a in range(len(ends)) for b in range(a + 1, len(ends))), default=float("nan"))
    print(f"    distance from each final track to the *nearest base solution*: "
          f"{min(to_base):.2e}..{max(to_base):.2e} rad (a track far from all of them is a fibre "
          f"solution the base census missed); smallest distance between two final tracks "
          f"{pairwise:.2e} rad")
    permutation = {}
    for k, item in enumerate(carried.lifts):
        if item.index != poses.shape[0] - 1:
            continue
        best = min(range(len(solutions)),
                   key=lambda m: iks.configuration_distance(solutions[m], item.q, circular))
        permutation[k] = best
    print(f"    start track -> fibre solution it ends on: "
          f"{[permutation.get(k, -1) for k in range(len(solutions))]} "
          f"(-1 = that track died; tracks > {len(solutions) - 1} were born mid-path: "
          f"{[permutation.get(k, -1) for k in range(len(solutions), len(carried.lifts))]})")
    print("\n  control: degenerate loop, with the hook armed (no admission may be reported)")
    admitter_control = FiberAdmitter(model, degenerate[::args.control_stride], links,
                                     restarts=args.sample_seeds, merge=args.merge, rng=rng)
    admitter_control.known = [np.asarray(q, dtype=float) for q in solutions]
    control_carry = ts.lift_fiber(
        lambda q: homotopy.fk_links(links, q), jacobian_of, degenerate[::args.control_stride],
        solutions, residual=residual_of, interpolate=ts.se3_interpolate,
        sigma=lambda q: iks.sigma_min(model, q), sigma_floor=1e-9, collision=args.gap,
        merge=1e-6, distance=lambda a, b: iks.configuration_distance(a, b, circular),
        admit=admitter_control, feasible=guard, prune_merged=args.prune_merged,
    )
    print(f"    samples {len(control_carry.counts)}, live {control_carry.counts[0]} -> "
          f"{control_carry.counts[-1]}, admissions {len(admitter_control.events)}, birth(s) "
          f"{sum(1 for event in admitter_control.events if event['kind'] == 'birth')}, collisions "
          f"{len(control_carry.collisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
