#!/usr/bin/env python3
"""exp43: census-based fold seeding -- closing the sensitivity gap of tracked-branch collection.

DeepSeek's fourth-round handoff (``study/TO_DEEPSEEK.md``): tracked-branch candidate collection
misses folds whose birth happens on untracked pairs (the study's own section 4.4), so spectra
from tracking alone are a lower bound.  This experiment replaces the seed source: at a grid of
``s`` values, sample random configurations, keep the near-singular ones, and refine each with
the augmented Newton -- covering the whole joint space rather than only tracked branches.

Measured question: at the witness pose (exp19's pose 0), does the census-based source find
**more** folds than the two the tracked branches gave (0.9911, 0.9936 in exp41)?

Run::

    python3 -m study.exp43_fold_census
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import refine_fold
from study.exp35_cusp import fold_seeds


def census_fold_seeds(path, target, sr5_model, *, grid: int = 40, samples: int = 300,
                      threshold: float = 0.04, seed: int = 23,
                      verbose: bool = False) -> list[tuple[float, np.ndarray]]:
    """Fold seeds from random near-singular configurations on a grid of s (whole joint space).

    Unlike the tracked-branch collection (exp35's ``fold_seeds``), this sees folds whose birth
    happens on untracked pairs -- measured at the witness pose: 4 folds versus 2 by tracking.
    """
    lower = np.asarray(sr5_model.lower_bounds)
    upper = np.asarray(sr5_model.upper_bounds)
    rng = np.random.default_rng(seed)
    found: list[tuple[float, np.ndarray]] = []
    for s in np.linspace(0.05, 1.0, grid):
        links = path(s)
        for _ in range(samples):
            q = rng.uniform(lower, upper)
            if homotopy.sigma_min_links(links, q) > threshold:
                continue
            s_star, q_star, residual = refine_fold(path, target, q, s)
            if residual > 1e-8 or not (0.0 < s_star < 1.3):
                continue
            if any(abs(s_star - old_s) < 3e-4 for old_s, _ in found):
                continue
            found.append((s_star, q_star))
            if verbose:
                print(f"  fold at s* = {s_star:.6f} (residual {residual:.1e})")
    found.sort(key=lambda item: item[0])
    return found


def main() -> int:
    """Compare census-based and tracking-based fold spectra at the witness pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, default=40, help="s samples")
    parser.add_argument("--samples", type=int, default=300, help="random configs per s")
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--skip-track", action="store_true",
                        help="skip the (slow) tracking-based reference run")
    parser.add_argument("--save", default="", help="npz path to save the census fold spectrum")
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    sr5_model = sr0.robot(link_transforms())

    # the witness pose, exactly as exp19/exp39 draw it
    from model import IkType, ModelFactory  # noqa: PLC0415

    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    rng = np.random.default_rng(0)
    target, _ = iks.random_reachable_target(model, rng)

    tracked_s: list[float] = []
    if not args.skip_track:
        print("tracking-based spectrum (exp35 machinery, reference):")
        folds_tracked = fold_seeds(path, target, sr0.solve(sr0.robot(), target))
        tracked_s = [s for s, _ in folds_tracked]
        print(f"  {len(tracked_s)} folds: {[f'{s:.4f}' for s in tracked_s]}")

    print(f"census-based spectrum ({args.grid} s x {args.samples} configs, "
          f"threshold {args.threshold}):")
    folds_census = census_fold_seeds(path, target, sr5_model, grid=args.grid,
                                     samples=args.samples, threshold=args.threshold,
                                     seed=args.seed, verbose=True)
    census_s = [s for s, _ in folds_census]
    print(f"  {len(census_s)} folds: {[f'{s:.4f}' for s in census_s]}")
    if args.save:
        np.savez(args.save, s=census_s, q=np.array([q for _, q in folds_census]))
        print(f"  saved to {args.save}")

    only_census = [s for s in census_s if not any(abs(s - t) < 3e-4 for t in tracked_s)]
    only_tracked = [t for t in tracked_s if not any(abs(t - s) < 3e-4 for s in census_s)]
    print(f"\nonly in census: {[f'{s:.4f}' for s in only_census]}")
    print(f"only in tracking: {[f'{s:.4f}' for s in only_tracked]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
