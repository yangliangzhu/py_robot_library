#!/usr/bin/env python3
"""exp31: triggering the scan on the fold spectrum instead of on count differences

exp30's adaptive scan refines the intervals whose *counts* differ.  That trigger is sound but not
complete: an interval containing one birth and one death has equal counts at its ends and is never
refined.  The fold spectrum is the right trigger, because every event *is* a fold (3.18/3.20): find the
fold points along the path deterministically, then put the extra samples exactly at ``s* +- delta``.

So this experiment compares three scans on the same poses:

* **uniform** -- the module's default (grid x seeds);
* **count-triggered** -- exp30's two-pass version;
* **fold-triggered** -- a coarse grid plus two samples around every fold.

The claim is the same as exp30's: identical solution sets, less time -- with the trigger now resting on
the structure rather than on a count that can cancel.

Run::

    python3 -m study.exp31_fold_triggered_scan --pose-seeds 0 7
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study import sr0
from study.complete_ik import solve_all
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import approach, refine_fold
from study.exp30_adaptive_scan import adaptive_scan, carry_forward


def fold_spectrum(offset: float, target, seeds, *, near: float, iterations: int) -> list[float]:
    """The fold parameter values along the path, found deterministically."""
    path = path_at(offset)
    candidates = []
    for seed in seeds:
        for s, q in approach(path, target, seed, 1.0):
            if iks.sigma_min(sr0.robot(path(s)), q) < near:
                candidates.append((s, q))
    folds: list[tuple[float, np.ndarray]] = []
    for s, q in candidates:
        s_star, q_star, residual = refine_fold(path, target, q, s, iterations=iterations)
        if residual > 1e-8:
            continue
        if not any(
            abs(s_star - other) < 1e-4 and float(np.linalg.norm(q_star - q_other)) < 1e-3
            for other, q_other in folds
        ):
            folds.append((s_star, q_star))
    return sorted(s_star for s_star, _q in folds)


def fold_triggered_scan(offset: float, target, circular, folds: list[float], *, coarse: int,
                        coarse_seeds: int, fine_seeds: int, delta: float,
                        sigma_floor: float) -> tuple[list[np.ndarray], list[float]]:
    """Coarse grid plus two samples around every fold."""
    samples = list(np.linspace(0.0, 1.0, coarse))
    for s_star in folds:
        for offset_s in (-delta, +delta):
            s = float(np.clip(s_star + offset_s, 0.0, 1.0))
            if all(abs(s - existing) > 1e-3 for existing in samples):
                samples.append(s)
    samples.sort()
    found: list[np.ndarray] = []
    for s in samples:
        seeds = coarse_seeds if any(abs(s - g) < 1e-9 for g in np.linspace(0.0, 1.0, coarse)) \
            else fine_seeds
        found.extend(carry_forward(offset, target, float(s), seeds, sigma_floor=sigma_floor))
    unique: list[np.ndarray] = []
    for q in found:
        if not any(iks.configuration_distance(q, other, circular) < 1e-3 for other in unique):
            unique.append(q)
    return unique, samples


def main() -> int:
    """Compare the three scans."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seeds", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--coarse-seeds", type=int, default=60)
    parser.add_argument("--fine-seeds", type=int, default=120)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--uniform-grid", type=int, default=9)
    parser.add_argument("--uniform-seeds", type=int, default=300)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    sr0_model = sr0.robot()
    circular = iks.circular_joints(model)
    for pose_seed in args.pose_seeds:
        rng = np.random.default_rng(pose_seed)
        target, _ = iks.random_reachable_target(model, rng)

        started = time.perf_counter()
        folds = fold_spectrum(offset, target, [s.q for s in sr0.solve(sr0_model, target)],
                              near=args.near, iterations=args.iterations)
        spectrum_seconds = time.perf_counter() - started

        started = time.perf_counter()
        fold_found, samples = fold_triggered_scan(
            offset, target, circular, folds, coarse=args.coarse, coarse_seeds=args.coarse_seeds,
            fine_seeds=args.fine_seeds, delta=args.delta, sigma_floor=2e-3)
        fold_seconds = time.perf_counter() - started

        started = time.perf_counter()
        count_found, counts = adaptive_scan(offset, target, circular, coarse=args.coarse,
                                            coarse_seeds=args.coarse_seeds,
                                            fine_seeds=args.fine_seeds, sigma_floor=2e-3)
        count_seconds = time.perf_counter() - started

        started = time.perf_counter()
        uniform = solve_all(model, target, mode="scan", grid=args.uniform_grid,
                            seeds=args.uniform_seeds, rng=np.random.default_rng(pose_seed))
        uniform_seconds = time.perf_counter() - started
        uniform_points = [solution.q for solution in uniform]

        def misses(reference, other) -> int:
            return sum(
                1 for q in reference
                if not any(iks.configuration_distance(q, o, circular) < 1e-3 for o in other)
            )

        print(
            f"pose seed {pose_seed}: folds at {[round(s, 4) for s in folds]} "
            f"(found in {spectrum_seconds:.1f} s)"
        )
        print(
            f"  fold-triggered : {len(fold_found):2d} solutions in {fold_seconds:5.1f} s "
            f"({len(samples)} samples) | misses {misses(uniform_points, fold_found)} of uniform"
        )
        print(
            f"  count-triggered: {len(count_found):2d} solutions in {count_seconds:5.1f} s "
            f"(coarse counts {counts}) | misses {misses(uniform_points, count_found)} of uniform"
        )
        print(
            f"  uniform        : {len(uniform_points):2d} solutions in {uniform_seconds:5.1f} s "
            f"| fold-triggered misses {misses(fold_found, uniform_points)} of it"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
