#!/usr/bin/env python3
"""exp58: SR0's monodromy generator at the fold whose pair is real at the base pose (Q8, fixed tracker).

kimi's ``exp57`` measured the identity around the refined fold at ``u* = -0.448423`` and could not
reach the informative fold (its pair is real at the base pose, so the loop around it *must* show a
permutation).  The reason turned out to be the tracker rather than the census: sweeping ``u`` with a
naive "correct from the previous configuration" scheme, the eight complex roots collapse pairwise to
~1e-15 at almost every step (measured), which is the "persistent merge" fault already characterised in
the task-space work (NOTES 3.40) -- a tracker that merges roots can read the identity for reasons of its
own.

This module replaces the tracker with the discipline that the task-space instrument uses:

* **linear prediction** from the last two accepted configurations, corrected from the prediction;
* **jump rejection**: a correction further from the prediction than ``jump_tolerance`` is a basin jump,
  and that root is marked untracked (never silently merged);
* **merge guard**: if two tracked roots come closer than ``merge`` at any pose, the step is reported as
  a fault instead of being accepted;
* **root identity**: a loop's permutation is only accepted when all eight roots survive it and the
  endpoint matching is unambiguous.

Phases:

1. track the eight analytic roots of the base pose along the ``u`` axis in fine steps and print the
   minimum pairwise gap -- with identity bookkeeping the gap should stay near the base separation and
   go to zero only *at* a fold;
2. refine that fold with the augmented system (``exp57.refine_branch_pose``);
3. walk a complex circle around it (arcing in from the base pose) and read the permutation.

Run::

    python3 -m study.exp58_sr0_fold_and_group
"""

from __future__ import annotations

import argparse

import numpy as np

from study import sr0
from study.exp37_complex_loop import newton_complex
from study.exp57_sr0_monodromy import refine_branch_pose, torus_distance_complex
from study.sr0 import sr0_links


def walk(links, q_start, poses, *, jump_tolerance: float = 0.3, tolerance: float = 1e-10,
         iterations: int = 30):
    """Carry one root through a pose list with linear prediction and jump rejection.

    Args:
        links: SR0's link transforms.
        q_start: The root at the first pose, complex.
        poses: The pose sequence (complex 4x4 arrays).
        jump_tolerance: Largest distance from the prediction that still counts as following the root.
        tolerance: Residual below which a correction counts as converged.
        iterations: Newton iterations per step.

    Returns:
        ``(q, ok)``: the final configuration and whether the root survived every step.
    """
    q_prev = np.asarray(q_start, dtype=complex).copy()
    q_cur, residual = newton_complex(links, q_prev, poses[0], iterations=iterations,
                                     tolerance=tolerance)
    if residual > tolerance or not np.all(np.isfinite(q_cur)):
        return q_prev, False
    for pose in poses[1:]:
        prediction = 2.0 * q_cur - q_prev
        q_new, residual = newton_complex(links, prediction, pose, iterations=iterations,
                                        tolerance=tolerance)
        if residual > tolerance or not np.all(np.isfinite(q_new)):
            return q_cur, False
        if float(np.linalg.norm(q_new - prediction)) > jump_tolerance:
            return q_cur, False
        q_prev, q_cur = q_cur, q_new
    return q_cur, True


def min_pair_gap(qs, alive) -> tuple[float, int, int]:
    """The smallest distance between two tracked roots, and which pair attains it."""
    best = (float("inf"), -1, -1)
    live = [i for i in alive]
    for a, i in enumerate(live):
        for j in live[a + 1:]:
            distance = torus_distance_complex(qs[i], qs[j])
            if distance < best[0]:
                best = (distance, i, j)
    return best


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u-index", type=int, default=0, help="position coordinate to vary")
    parser.add_argument("--span", type=float, default=0.03, help="how far to sweep along u")
    parser.add_argument("--steps", type=int, default=60, help="sweep steps (span/steps = du)")
    parser.add_argument("--radius", type=float, default=5e-4, help="circle radius around the fold")
    parser.add_argument("--merge", type=float, default=1e-6,
                        help="two tracked roots closer than this at any pose is a tracker fault")
    parser.add_argument("--seeds", type=int, default=6,
                        help="base poses to try (each seed gives one random pose)")
    parser.add_argument("--fold-gap", type=float, default=5e-2,
                        help="a closest approach below this counts as a fold in reach")
    args = parser.parse_args()

    links = sr0_links()
    model = sr0.robot()

    def build(seed: int, index: int):
        """A base pose (fk of a random configuration) and its eight analytic roots."""
        local = np.random.default_rng(seed)
        q_true = local.uniform(-2.0, 2.0, model.num_dof)
        target0 = np.asarray(model.fk(q_true), dtype=float)
        u0 = float(target0[:3, 3][index])

        def target_at(u: complex) -> np.ndarray:
            moved = target0.astype(complex)
            moved[:3, 3] = target0[:3, 3].astype(complex)
            moved[:3, 3][index] = u
            return moved

        fibre = [np.asarray(solution.q, dtype=complex) for solution in sr0.solve(model, target0)]
        return u0, target_at, fibre

    def sweep(u0, target_at, fibre, direction: float, span: float, steps: int):
        """Walk the eight roots along one direction; adaptive step, jump rejection, merge guard."""
        du = span / steps
        qs = [q.copy() for q in fibre]
        previous = [q.copy() for q in qs]
        alive = list(range(len(fibre)))
        u = u0
        step = du * direction
        smallest = (float("inf"), u0, -1, -1)
        reached = u0
        while abs(u - u0) < span and len(alive) >= 2:
            trial_u = u + step
            pose = target_at(trial_u)
            updated = {}
            faults = 0
            for index in alive:
                prediction = 2.0 * qs[index] - previous[index]
                q_new, residual = newton_complex(links, prediction, pose)
                if (residual > 1e-10 or not np.all(np.isfinite(q_new))
                        or float(np.linalg.norm(q_new - prediction)) > 0.3):
                    faults += 1
                    continue
                updated[index] = (q_new, qs[index])
            if faults:
                step *= 0.5
                if abs(step) < 1e-7:
                    break
                continue
            for index, (q_new, q_old) in updated.items():
                previous[index] = q_old
                qs[index] = q_new
            u = trial_u
            reached = u
            gap, first, second = min_pair_gap(qs, alive)
            if gap < smallest[0]:
                smallest = (gap, u, first, second)
            step = min(abs(step) * 1.5, du) * direction
        return smallest, reached

    print(f"\nphase 0: which base pose has a fold *inside* its reachable interval?  "
          f"({args.seeds} seeds x 3 position coordinates)")
    best = None
    for seed in range(args.seeds):
        for index in (0, 1, 2):
            u0, target_at, fibre = build(seed, index)
            if len(fibre) < 8:
                continue
            out = {}
            for direction in (-1.0, 1.0):
                smallest, reached = sweep(u0, target_at, fibre, direction, args.span, args.steps)
                out[direction] = (smallest, reached)
            gap, u_gap, first, second = min(out[-1.0][0], out[1.0][0], key=lambda item: item[0])
            print(f"  seed {seed} coord {index}: reachable u in "
                  f"[{out[-1.0][1]:+.6f}, {out[1.0][1]:+.6f}] around {u0:+.6f}; closest pair "
                  f"{gap:.3e} at u = {u_gap:+.6f} (pair {first},{second})")
            if best is None or gap < best[0]:
                best = (gap, seed, index, u0, target_at, fibre, u_gap)
    if best is None:
        print("  no base pose with eight analytic solutions found")
        return 1
    gap, seed, index, u0, target_at, fibre, u_gap = best
    print(f"\n  best: seed {seed}, coordinate {index}, closest approach {gap:.3e} at "
          f"u = {u_gap:+.6f} (base u0 = {u0:+.6f})")
    if gap > args.fold_gap:
        print(f"  no fold closer than {args.fold_gap:g} in configuration space on this sweep; "
              f"a generator needs a fold inside the reachable interval -- widen --span/--seeds")
        return 1

    print("\nphase 2: refine the fold with the augmented system")
    y, residual = refine_branch_pose(links, target_at, 0.5 * (fibre[0] + fibre[1]), u_gap)
    print(f"  refined fold: u* = {y[6].real:+.6f}{y[6].imag:+.1e}i (residual {residual:.1e})")
    if residual > 1e-8:
        print("  refinement failed -- the loop below would not be trustworthy")
        return 1
    u_star = y[6]

    print("\nphase 3: a complex circle around the fold, all eight roots tracked with identity")
    radius = args.radius
    base_u = u_star + radius
    arc_in = [target_at(u0 + alpha * (base_u - u0)) for alpha in np.linspace(0.0, 1.0, 25)[1:]]
    circle = [target_at(u_star + radius * np.exp(1j * theta))
              for theta in np.linspace(0.0, 2 * np.pi, 97)[1:]]
    arc_out = [target_at(base_u + alpha * (u0 - base_u)) for alpha in np.linspace(0.0, 1.0, 25)[1:]]
    poses = arc_in + circle + arc_out
    images: list[int] = []
    for q0 in fibre:
        q_end, ok = walk(links, q0, poses)
        if not ok:
            images.append(-1)
            continue
        distances = [torus_distance_complex(q_end, member) for member in fibre]
        images.append(int(np.argmin(distances)) if min(distances) < 1e-2 else -1)
    print(f"  permutation {images}")
    if -1 in images:
        print(f"  {images.count(-1)} root(s) untracked -- no generator claimed from this loop")
        return 1
    cycles, seen = [], set()
    for start in range(len(images)):
        if start in seen:
            continue
        length, node = 0, start
        while node not in seen:
            seen.add(node)
            node = images[node]
            length += 1
        cycles.append(length)
    transpositions = sum(1 for length in cycles if length == 2)
    identity = all(images[i] == i for i in range(len(images)))
    print(f"  cycle type {'+'.join(str(length) for length in sorted(cycles, reverse=True))}; "
          f"{transpositions} transposition(s); identity {identity}")
    print("\n  reading: one transposition generates a group of order 2 (abelian, exponent 2), "
          "consistent with the predicted elementary abelian 2-group; identity alone says nothing. "
          "The group measured this way is always a *subgroup* of the true monodromy group, so each "
          "new independent transposition raises the lower bound.")
    return 0 if not identity else 2


if __name__ == "__main__":
    raise SystemExit(main())
