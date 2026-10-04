#!/usr/bin/env python3
"""exp53: a second closed-form neighbour -- which relaxation gives another Pieper arm, and does it
cover SR0's blind spot?

SR0 (exp10's finding) closes the *wrist*: zeroing the y-offset of link 5 makes axes 4, 5, 6 meet at a
point, which is one of Pieper's two sufficient conditions.  It leaves a seed gap, though: on 20-30% of
poses SR0's position sub-problem has no real solution, so a continuation from SR0 cannot even start
(exp13/exp25).  A second neighbour would have to be a *different* Pieper arm -- and Pieper's other
condition is the parallel one: **three consecutive axes parallel**.  SR5 already has axes 2 || 3, so a
single twist change (link 4's rotation from 90 degrees to 0 or 180) makes axes 2, 3, 4 parallel.  That
is a candidate, and this module measures it rather than assuming it.

Two things are measured for every candidate.  (1) The **structural certificate**, from the pairwise
axis fingerprint: the angle between axes 3 and 4 must vanish (or be 180 degrees), the arm must stay
non-degenerate (full-rank Jacobian at random configurations), and the wrist offset must stay what it
was -- the candidate is a *different* arm, not a repaired SR5.  (2) The **coverage**: on poses drawn
from the arm's own workspace, SR0's closed form is run (``study.sr0.solve``) and the candidate is
censused numerically (``solve_lm`` from many seeds), so the table separates "SR0 blind but the
candidate has real solutions" from "blind for both".

Phase 4 is the instrument's own control: the *unmodified* SR5 as candidate must be found solvable on
every pose SR0 solves (a census that misses them would be under-detecting), and SR0 as candidate must
reproduce its own blind set.

Run::

    python3 -m study.exp53_second_neighbour --poses 60 --seeds 200
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType
from model.configs.loader import load_robot_config
from model.robot_model_numpy import RobotModelNumpy
from study import arm_geometry as ag
from study import ik_structure as iks
from study import sr0
from study.census import solve_lm
from study.exp11_dh_continuation import link_transforms


def rot_x(angle: float) -> np.ndarray:
    """A 3x3 rotation about x, for building candidate link twists."""
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def build(links: list[np.ndarray]) -> RobotModelNumpy:
    """A model of the candidate arm, with SR5's joint limits and tool."""
    config = load_robot_config("rokae_sr5")
    config["param"] = [np.asarray(link, dtype=float) for link in links]
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config)
    return model


def candidates(baseline: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """The arms to measure: SR0 (reference), the parallel-twist candidates, and the bare SR5.

    Args:
        baseline: SR5's six link transforms.

    Returns:
        A mapping from name to a link list.
    """
    out: dict[str, list[np.ndarray]] = {"SR5 (unchanged)": [link.copy() for link in baseline]}
    sr0_links = [link.copy() for link in baseline]
    sr0_links[4][1, 3] = 0.0
    out["SR0 (wrist closed)"] = sr0_links
    for label, angle in (("parallel: link4 twist 0 deg", 0.0), ("parallel: link4 twist 180 deg",
                                                                 np.pi)):
        links = [link.copy() for link in baseline]
        links[3] = links[3].copy()
        links[3][:3, :3] = rot_x(angle)
        out[label] = links
    return out


def curved_path(twist0: float, direction: np.ndarray, amplitude: float,
                target_angle: float | None = None):
    """The straight twist path plus a bump in the link translations that vanishes at both ends.

    A straight path in parameter space meets the discriminant for some branches, and there the
    solution genuinely stops existing *along that path*; the discriminant has codimension one, so a
    detour reaches the same endpoints by a different route (the trick of exp13, moved from the wrist
    offset to the twist).  The bump is applied to the 18 translation components only, so the twist
    interpolation -- the thing that makes the start arm Pieper-parallel -- is untouched.

    Args:
        twist0: The twist at ``s = 0``.
        direction: An 18-vector, the bump's direction in link-translation space.
        amplitude: Size of the bump, in metres.
        target_angle: The twist at ``s = 1``; SR5's link 4 when omitted.

    Returns:
        A callable ``s -> link transforms`` for :func:`study.homotopy.track`.
    """
    end = twist_angle(link_transforms()[3]) if target_angle is None else float(target_angle)
    base = [np.asarray(link, dtype=float).copy() for link in link_transforms()]
    shaped = np.asarray(direction, dtype=float).reshape(6, 3)

    def make(s: float) -> list[np.ndarray]:
        links = [link.copy() for link in base]
        links[3][:3, :3] = rot_x((1.0 - float(s)) * twist0 + float(s) * end)
        bump = amplitude * float(np.sin(np.pi * float(s))) * shaped
        for index in range(6):
            links[index][:3, 3] += bump[index]
        return links

    return make


def twist_angle(link: np.ndarray) -> float:
    """The twist of an ``Rx(alpha)`` link, read off the matrix (SR5's link 4 is ``Rx(-90 deg)``)."""
    matrix = np.asarray(link, dtype=float)
    return float(np.arctan2(matrix[2, 1], matrix[1, 1]))


def parallel_path(twist0: float, target_angle: float | None = None):
    """A parameter path from the parallel arm to SR5: link 4's twist, ``twist0 -> SR5's twist``.

    The end must be the *measured* twist of SR5's link 4, sign included: an earlier version
    interpolated to ``+90 deg`` where the arm has ``-90 deg``, so ``s = 1`` was a different arm and
    every tracked endpoint missed the SR5 reference solutions (measured: 0 of 14 matched).

    Args:
        twist0: The twist at ``s = 0`` (0 or pi, both of which make axes 2, 3, 4 parallel).
        target_angle: The twist at ``s = 1``; SR5's link 4 when omitted.

    Returns:
        A callable ``s -> link transforms`` for :func:`study.homotopy.track`.
    """
    end = twist_angle(link_transforms()[3]) if target_angle is None else float(target_angle)

    def make(s: float) -> list[np.ndarray]:
        links = link_transforms()
        links[3] = links[3].copy()
        links[3][:3, :3] = rot_x((1.0 - float(s)) * twist0 + float(s) * end)
        return links

    return make


def rank_fraction(model, rng, *, samples: int = 200) -> tuple[int, float]:
    """How many random configurations have a full-rank Jacobian, and the worst ``sigma_min``."""
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    full = 0
    worst = float("inf")
    for _ in range(samples):
        q = lower + (upper - lower) * rng.random(model.num_dof)
        singular = np.linalg.svd(model.jacobian(q), compute_uv=False)
        worst = min(worst, float(singular[-1]))
        full += int(singular[-1] > 1e-6)
    return full, worst


def census(model, target, rng, *, seeds: int, dedupe: float = 1e-4) -> int:
    """Distinct real solutions of ``model`` at ``target``, from ``seeds`` multi-start solves."""
    found: list[np.ndarray] = []
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    for _ in range(seeds):
        seed = lower + (upper - lower) * rng.random(model.num_dof)
        q, converged = solve_lm(model, seed, target)
        if not converged:
            continue
        if any(iks.torus_distance(q, other) < dedupe for other in found):
            continue
        found.append(np.asarray(q, dtype=float))
    return len(found)


def homotopy_with_births(links_at, target, starts, *, grid, seeds, merge, rng, sigma_floor,
                         tolerance=1e-10):
    """Walk a parameter path in small steps, admitting solutions the carried tracks do not cover.

    A forward tracker can only follow what it carries, so a branch that appears along the path -- for
    the twist path, one of the SR5 solutions with no counterpart at ``s = 0`` -- is invisible to it;
    that is measured in 3.44/3.45 as the reason the second neighbour recovers only half the fibre.
    This is Q23's instrument in parameter space: at every grid point, census the *current* arm at the
    fixed target and add every verified solution no live track occupies as a new track.

    Args:
        links_at: ``s -> link transforms``; the arm at parameter ``s``.
        target: The pose being solved, fixed along the path.
        starts: Solutions of ``links_at(0)`` to carry.
        grid: Number of steps from ``s = 0`` to ``s = 1``.
        seeds: Census restarts per grid point.
        merge: Torus distance below which a census solution is one a track already occupies.
        rng: Generator for the census seeds.
        sigma_floor: Clearance passed to the tracker.
        tolerance: Pose residual the tracker must reach.

    Returns:
        ``(tracks, births, profile)``: the tracks (each a dict with ``q``, ``s``, ``born`` and
        ``alive``), the admitted births as ``(s, q)`` pairs, and ``(s, census size, live tracks)`` at
        every grid point.
    """
    tracks = [{"q": np.asarray(q, dtype=float), "s": 0.0, "born": 0.0, "alive": True,
               "reason": "carried"} for q in starts]
    births: list[tuple[float, np.ndarray]] = []
    #: ``(s, census size, live tracks)`` per grid point: says *where* the fibre grows, which is what
    #: separates a genuine birth along the path from a census of the target arm at the last step.
    profile: list[tuple[float, int, int]] = []
    for step in range(1, grid + 1):
        s_to = step / grid
        for track in tracks:
            if not track["alive"]:
                continue
            s_from = track["s"]
            from study.homotopy import track as homotopy_track

            def segment(u: float, a: float = s_from, b: float = s_to) -> list[np.ndarray]:
                """The path restricted to the sub-interval ``[a, b]``, re-parameterised to [0, 1]."""
                return links_at(a + float(u) * (b - a))

            result = homotopy_track(segment, target, track["q"], sigma_floor=sigma_floor,
                                    tolerance=tolerance)
            if result.finished:
                track["q"] = np.asarray(result.q, dtype=float)
                track["s"] = s_to
            else:
                track["alive"] = False
                track["reason"] = result.reason
        model = build(links_at(s_to))
        fiber = census_of(model, target, rng, seeds=seeds, dedupe=1e-3)
        profile.append((s_to, len(fiber), sum(1 for track in tracks if track["alive"])))
        occupied = [track["q"] for track in tracks if track["alive"]]
        for candidate in fiber:
            if any(iks.torus_distance(candidate, other) < merge for other in occupied):
                continue
            occupied.append(candidate)
            tracks.append({"q": candidate, "s": s_to, "born": s_to, "alive": True,
                           "reason": "born"})
            births.append((s_to, candidate))
    return tracks, births, profile


def census_count(model, target, rng, *, seeds: int, dedupe: float = 1e-4) -> tuple[int, int]:
    """``(distinct solutions, seeds that converged)`` -- the convergence count matters when a small
    number is being read as "the fibre really is that small"."""
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    found: list[np.ndarray] = []
    converged = 0
    for _ in range(seeds):
        seed = lower + (upper - lower) * rng.random(model.num_dof)
        q, ok = solve_lm(model, seed, target)
        if not ok:
            continue
        converged += 1
        q = np.asarray(q, dtype=float)
        if any(iks.torus_distance(q, other) < dedupe for other in found):
            continue
        found.append(q)
    return len(found), converged


def census_of(model, target, rng, *, seeds: int, dedupe: float = 1e-4) -> list[np.ndarray]:
    """Distinct real solutions of ``model`` at ``target`` (the module's census helper, as a list)."""
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    found: list[np.ndarray] = []
    for _ in range(seeds):
        seed = lower + (upper - lower) * rng.random(model.num_dof)
        q, converged = solve_lm(model, seed, target)
        if not converged:
            continue
        q = np.asarray(q, dtype=float)
        if any(iks.torus_distance(q, other) < dedupe for other in found):
            continue
        found.append(q)
    return found


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=200, help="census restarts per pose")
    parser.add_argument("--samples", type=int, default=200, help="random configurations per rank test")
    parser.add_argument("--homotopy", action="store_true",
                        help="phase 3: track each parallel-arm solution to SR5 along link 4's twist "
                             "and check it against an SR5 census at the same pose")
    parser.add_argument("--twist0", type=float, default=0.0,
                        help="link 4's twist at the start of the homotopy (0 or pi)")
    parser.add_argument("--reference-seeds", type=int, default=400)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--second-arm", action="store_true",
                        help="also use the 180 deg parallel arm's starts (union of the two)")
    parser.add_argument("--retries", type=int, default=0,
                        help="curved-path retries per stopped start")
    parser.add_argument("--amplitude", type=float, default=0.02,
                        help="size of the curved path's bump, metres")
    parser.add_argument("--births", action="store_true",
                        help="phase 4: admit uncovered solutions at every grid point of the twist "
                             "path (the DH-side counterpart of Q23)")
    parser.add_argument("--grid", type=int, default=40, help="homotopy steps for phase 4")
    parser.add_argument("--birth-seeds", type=int, default=40, help="census restarts per grid point")
    parser.add_argument("--deep-seeds", type=int, default=300,
                        help="seeds for the deep census at --deep-at, to test whether the "
                             "intermediate arms really have fewer solutions")
    parser.add_argument("--deep-at", type=float, nargs="*", default=[0.5, 0.95])
    parser.add_argument("--merge", type=float, default=1e-2,
                        help="torus distance below which a census solution is one a track occupies")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    baseline = link_transforms()
    arms = candidates(baseline)
    print("exp53: a second Pieper neighbour, and whether it covers SR0's seed gap\n")

    print("phase 1: structural certificates (pairwise axis fingerprint; adjacent pairs are "
          "configuration-independent)")
    for name, links in arms.items():
        model = build(links)
        pairs = {(pair.first, pair.second): pair for pair in ag.axis_fingerprint(model)}
        full, worst = rank_fraction(model, rng, samples=args.samples)
        angle = pairs[(3, 4)].angle_deg
        distance = pairs[(3, 4)].distance
        wrist = pairs[(4, 6)].distance
        print(f"    {name:<28} angle(3,4) {angle:8.3f} deg, dist(3,4) {distance:.4f} m, "
              f"wrist gap (4,6) {wrist:.4f} m, full rank {full}/{args.samples} "
              f"(worst sigma_min {worst:.1e})")

    print(f"\nphase 2: coverage of {args.poses} poses drawn from SR5's own workspace "
          f"(SR0 by its closed form, the others by a {args.seeds}-seed census)")
    sr0_model = sr0.robot()
    sr5_model = build(baseline)
    circular = iks.circular_joints(sr5_model)
    lower = np.asarray(sr5_model.lower_bounds, dtype=float)
    upper = np.asarray(sr5_model.upper_bounds, dtype=float)
    table: dict[str, dict[str, int]] = {name: {"both": 0, "sr0 only": 0, "candidate only": 0,
                                              "neither": 0} for name in arms}
    blind_poses = 0
    blind_targets: list[np.ndarray] = []
    for _ in range(args.poses):
        q = lower + (upper - lower) * rng.random(sr5_model.num_dof)
        target = sr5_model.fk(q)
        sr0_count = len(sr0.solve(sr0_model, target))
        blind_poses += int(sr0_count == 0)
        if sr0_count == 0:
            blind_targets.append(target)
        for name, links in arms.items():
            model = sr0_model if name.startswith("SR0") else build(links)
            count = (len(sr0.solve(model, target)) if name.startswith("SR0")
                     else census(model, target, rng, seeds=args.seeds))
            key = ("both" if (sr0_count and count) else
                   "sr0 only" if sr0_count else
                   "candidate only" if count else "neither")
            table[name][key] += 1
    print(f"    SR0 blind on {blind_poses} of {args.poses} poses "
          f"({100.0 * blind_poses / args.poses:.0f}%)")
    print("    candidate | both | SR0 only | candidate only (covers the gap) | neither")
    for name in arms:
        row = table[name]
        print(f"    {name:<28} | {row['both']:>4} | {row['sr0 only']:>8} | "
              f"{row['candidate only']:>27} | {row['neither']:>7}")
    print("\n  reading: 'candidate only' is the number of poses SR0 cannot seed and the candidate "
          "can; 'neither' is the gap no measured arm covers; the SR5 row is the census control "
          "(it must not be blind where SR0 solves).")

    if args.homotopy:
        from study.census import census as multi_start
        from study.census import make_lm_solver
        from study.homotopy import track

        angles = [args.twist0] + ([np.pi] if args.second_arm else [])
        print(f"\nphase 3: can the parallel arm *seed* SR5?  DH homotopy along link 4's twist "
              f"{' and '.join(f'{a:g}' for a in angles)} -> "
              f"{np.degrees(twist_angle(link_transforms()[3])):.0f} deg (SR5's own twist), on the "
              f"{len(blind_targets)} poses SR0 is blind on, {args.retries} curved retr"
              f"{'y' if args.retries == 1 else 'ies'} per stopped start")
        recovered = total_reference = total_starts = total_arrivals = total_retried = 0
        solved_poses = 0
        for index, target in enumerate(blind_targets):
            reference = multi_start(sr5_model, target, solver=make_lm_solver(sr5_model),
                                    seeds=args.reference_seeds, rng=rng, dedupe_tol=1e-4,
                                    circular=circular)
            known = [np.asarray(q, dtype=float) for q in reference.solutions]
            reached: list[int] = []
            starts_total = arrivals_total = retried_total = 0
            for angle in angles:
                path = parallel_path(angle)
                start_model = build(path(0.0))
                starts = multi_start(start_model, target, solver=make_lm_solver(start_model),
                                     seeds=args.seeds, rng=rng, dedupe_tol=1e-4, circular=circular)
                starts_total += len(starts.solutions)
                for solution in starts.solutions:
                    q_start = np.asarray(solution, dtype=float)
                    result = track(path, target, q_start, sigma_floor=args.sigma_floor)
                    if not result.finished and args.retries:
                        for _ in range(args.retries):
                            direction = rng.normal(size=18)
                            direction /= float(np.linalg.norm(direction))
                            attempt = track(curved_path(angle, direction, args.amplitude), target,
                                            q_start, sigma_floor=args.sigma_floor)
                            retried_total += 1
                            if attempt.finished:
                                result = attempt
                                break
                    if not result.finished:
                        continue
                    arrivals_total += 1
                    q_final = np.asarray(result.q, dtype=float)
                    for slot, candidate in enumerate(known):
                        if slot in reached:
                            continue
                        if iks.torus_distance(q_final, candidate) < 1e-3:
                            reached.append(slot)
                            break
            total_reference += len(known)
            recovered += len(reached)
            total_starts += starts_total
            total_arrivals += arrivals_total
            total_retried += retried_total
            solved_poses += int(bool(len(known)) and len(reached) == len(known))
            print(f"    pose {index + 1}: SR5 reference {len(known)} solutions, starts "
                  f"{starts_total}, arrived {arrivals_total} (after {retried_total} curved retries), "
                  f"distinct reference solutions reached {len(reached)}")
        print(f"    totals: {solved_poses}/{len(blind_targets)} blind poses fully recovered, "
              f"{recovered}/{total_reference} reference solutions reached from {total_arrivals} "
              f"arrivals of {total_starts} starts ({total_retried} curved retries)")

    if args.births:
        from study.census import census as multi_start
        from study.census import make_lm_solver

        print(f"\nphase 4: the same blind poses with births admitted along the twist path "
              f"({args.grid} grid points, {args.birth_seeds} census seeds each; Q23's instrument in "
              f"parameter space)")
        recovered = total_reference = total_births = 0
        solved_poses = 0
        for index, target in enumerate(blind_targets):
            reference = multi_start(sr5_model, target, solver=make_lm_solver(sr5_model),
                                    seeds=args.reference_seeds, rng=rng, dedupe_tol=1e-4,
                                    circular=circular)
            known = [np.asarray(q, dtype=float) for q in reference.solutions]
            starts = census_of(build(parallel_path(args.twist0)(0.0)), target, rng,
                               seeds=args.seeds, dedupe=1e-3)
            tracks, births, profile = homotopy_with_births(parallel_path(args.twist0), target,
                                                           starts, grid=args.grid,
                                                           seeds=args.birth_seeds,
                                                           merge=args.merge, rng=rng,
                                                           sigma_floor=args.sigma_floor)
            alive = [track["q"] for track in tracks if track["alive"]]
            reached: list[int] = []
            for q_final in alive:
                for slot, candidate in enumerate(known):
                    if slot in reached:
                        continue
                    if iks.torus_distance(q_final, candidate) < 1e-3:
                        reached.append(slot)
                        break
            total_reference += len(known)
            recovered += len(reached)
            total_births += len(births)
            solved_poses += int(bool(len(known)) and len(reached) == len(known))
            born_at = sorted({round(s, 3) for s, _q in births})
            print(f"    pose {index + 1}: SR5 reference {len(known)}, starts {len(starts)}, "
                  f"births admitted {len(births)} at s {born_at[:8]}, alive at s = 1 "
                  f"{len(alive)}, reference solutions reached {len(reached)}")
            print(f"      fibre along the path (s: census / live) "
                  f"{[(f'{s:.2f}', c, a) for s, c, a in profile]}")
            deep = {s: census_count(build(parallel_path(args.twist0)(s)), target, rng,
                                    seeds=args.deep_seeds, dedupe=1e-3)
                    for s in args.deep_at}
            print(f"      deep census ({args.deep_seeds} seeds) at s {args.deep_at} -> "
                  f"{{s: solutions/converged}} {deep} (rules out a weak census as the reason no "
                  f"birth appears before s = 1)")
        print(f"    totals: {solved_poses}/{len(blind_targets)} blind poses fully recovered, "
              f"{recovered}/{total_reference} reference solutions reached, {total_births} births "
              f"admitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
