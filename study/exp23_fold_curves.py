#!/usr/bin/env python3
"""exp23: tracking the fold curves -- counting the events structurally instead of statistically

exp22 failed because it counted a noisy fibre size.  The well-posed object is the set of *fold
points*: configurations where the pose is right and the Jacobian is singular.  For a fixed pose and a
moving DH parameter that set is a union of curves in ``(q, s)``, and exp21's augmented system
(``FK - T = 0``, ``det J = 0``) is exactly the corrector needed to follow them.

What this buys, if it works:

* the number of events the path crosses is a *count of curves*, not an estimate;
* each crossing changes the real fibre size by two (exp21's exponent is 1/2), which is a checkable
  prediction: ``fibre(s + eps) - fibre(s - eps) = 2 x (curves crossing s)``;
* the result is *structural completeness* for the homotopy of Finding 2: not just "the scan covers
  every solution" but "this solution was born at this fold".

Run::

    python3 -m study.exp23_fold_curves --pose-seed 0 --s 0.70 --seeds 150
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import refine_fold


def seeds_at(offset: float, target, s: float, seeds: int, rng, threshold: float = 0.03):
    """Fold seeds at one ``s``: refine every near-singular solution the census finds."""
    model = sr0.robot(path_at(offset)(s))
    fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
    found = []
    for q in fiber.solutions:
        if iks.sigma_min(model, q) >= threshold:
            continue
        s_star, q_star, residual = refine_fold(path_at(offset), target, q, s, iterations=12)
        if residual < 1e-8:
            found.append((s_star, q_star))
    return found


def track_curve(offset: float, target, q0: np.ndarray, s0: float, *, direction: float,
                step: float = 0.04, max_steps: int = 60, tolerance: float = 1e-7):
    """Follow one fold curve away from ``s0``; returns the ``(s, q)`` path."""
    path = path_at(offset)
    trace = [(s0, np.asarray(q0, dtype=float).copy())]
    s, q = s0, np.asarray(q0, dtype=float).copy()
    for _ in range(max_steps):
        probe = step
        accepted = None
        for _attempt in range(4):
            s_try = s + direction * probe
            if not 0.0 <= s_try <= 1.0:
                return trace
            # two or three iterations is enough once the curve is being followed: measured 106 ms
            # per refine at three iterations against 1062 ms at thirty, at the same residual
            s_new, q_new, residual = refine_fold(path, target, q, s_try, iterations=3)
            if residual < tolerance and abs(s_new - s_try) < 2.0 * probe:
                accepted = (s_new, q_new)
                break
            probe *= 0.5
        if accepted is None:
            return trace
        s, q = accepted
        trace.append((s, q.copy()))
        step = min(step * 1.3, 0.05)
    return trace


def main() -> int:
    """Find, track and count the fold curves."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--s", type=float, default=0.70, help="where to look for fold seeds")
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--dedupe", type=float, default=1e-3, help="curve identity tolerance")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr5_model = sr0.robot(link_transforms())
    rng_pose = np.random.default_rng(args.pose_seed)
    q_true = rng_pose.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    rng = np.random.default_rng(args.seed)

    found = seeds_at(offset, target, args.s, args.seeds, rng)
    print(f"fold seeds at s = {args.s}: {len(found)} (refined to |augmented| < 1e-8)")
    curves = []
    for s_star, q_star in found:
        for direction in (1.0, -1.0):
            trace = track_curve(offset, target, q_star, s_star, direction=direction)
            if len(trace) > 1:
                curves.append(trace)
    # merge curves that meet at a common s (a fold curve tracked from both sides)
    merged: list[list[tuple[float, np.ndarray]]] = []
    for trace in curves:
        for existing in merged:
            common = min(
                (abs(a[0] - b[0]) for a in trace for b in existing), default=1.0
            )
            if common < 1e-6:
                match = [
                    (a, b)
                    for a in trace
                    for b in existing
                    if abs(a[0] - b[0]) < 1e-6
                ]
                if match and all(
                    float(np.linalg.norm(a[1] - b[1])) < args.dedupe for a, b in match
                ):
                    existing.extend(trace)
                    break
        else:
            merged.append(list(trace))

    print(f"fold curves found: {len(merged)}")
    for index, trace in enumerate(merged):
        values = [point[0] for point in trace]
        clearances = [
            iks.sigma_min(sr0.robot(path_at(offset)(min(max(s, 0.0), 1.0))), q) for s, q in trace
        ]
        print(
            f"  curve {index}: {len(trace)} points, s from {min(values):.3f} to {max(values):.3f}"
            f" | clearance at the ends {clearances[0]:.1e} .. {clearances[-1]:.1e}"
        )
    crossings = sum(
        1 for trace in merged if min(point[0] for point in trace) <= args.s
        <= max(point[0] for point in trace)
    )
    print(f"curves crossing s = {args.s}: {crossings} "
          f"(exp21 says each crossing changes the real fibre size by two)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
