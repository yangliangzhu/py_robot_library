#!/usr/bin/env python3
"""exp46: the two strata of the new real fold at s* = 1.047164 -- paired and looped (K-4).

exp40's complex branch-point census found a new *real* fold just beyond the SR5 (s = 1.047164)
where **two** strata coalesce at the same parameter value (two different q*).  Each stratum is
a simple fold and contributes one transposition to the monodromy group of the s-family.  This
experiment refines both strata (exp40's clip-free Newton), seeds each fold's coalescing pair
along the Jacobian kernel (exp38's machinery), tracks both members to the base fibre at s = 1,
and reads off the two transpositions -- two more edges for the transposition graph.

Run::

    python3 -m study.exp46_strata
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp35_cusp import fold_seeds
from study.exp38_monodromy_group import (
    make_s_path,
    pair_at_fold,
    torus_distance_complex,
    track_path,
)

TARGET_S = 1.047164


def find_strata(base_links, offset: float, target: np.ndarray, sr5_model, path, *,
                s_probe: float = 1.046, seeds: int = 300, pair_radius: float = 0.5
                ) -> list[tuple[np.ndarray, float]]:
    """The coalescing strata near ``TARGET_S``.

    Seed source: a census at ``s_probe`` -- near a fold the fibre contains close pairs, whose
    midpoints seed ``refine_fold`` (real Newton, clip now [-0.3, 1.3]).  Two measured pitfalls
    are fenced: (i) tracked-branch samples never reach s > 1 (``collect_sigma_samples`` stops
    at s = 1); (ii) dedup must be torus-aware -- exp40's branch-point census used a plain
    ``|q - q'|`` and double-counted this fold, which is one stratum at two 2pi representatives.
    """
    from study.census import census, make_lm_solver  # noqa: PLC0415
    from study.exp21_endgame import refine_fold  # noqa: PLC0415

    model_at = sr0.robot(path(s_probe))
    fibre = census(model_at, target, solver=make_lm_solver(model_at), seeds=seeds,
                   rng=np.random.default_rng(5))
    circular = iks.circular_joints(sr5_model)
    sols = fibre.solutions
    found: list[tuple[np.ndarray, float]] = []
    for i in range(len(sols)):
        for j in range(i + 1, len(sols)):
            if iks.configuration_distance(sols[i], sols[j], circular) > pair_radius:
                continue
            mid = 0.5 * (sols[i] + sols[j])
            s_star, q_star, residual = refine_fold(path, target, mid, s_probe)
            if residual > 1e-8 or abs(s_star - TARGET_S) > 1e-3:
                continue
            if any(iks.torus_distance(q_star, q_old) < 1e-3 for q_old, _ in found):
                continue
            found.append((q_star.copy(), float(s_star)))
            print(f"  stratum at s* = {s_star:.6f}, residual {residual:.1e}")
    return found


def main() -> int:
    """Pair both strata of the fold at 1.047164 and read the two transpositions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rho", type=float, default=1e-3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    base_links = link_transforms()
    sr5_model = sr0.robot(base_links)
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)

    print("phase 1: fold spectrum (branches 0 and 6) + the two strata at 1.047164")
    fibre0 = sr0.solve(sr0.robot(), target)
    folds = fold_seeds(path, target, [fibre0[0], fibre0[6]])
    all_fold_s = [s for s, _ in folds]
    print(f"  folds on record: {[f'{s:.4f}' for s in all_fold_s]}")
    strata = find_strata(base_links, offset, target, sr5_model, path)
    print(f"  strata found: {len(strata)}")
    if not strata:
        print("  no stratum found -- aborting")
        return 1
    if len(strata) == 1:
        print("  NOTE: exactly one stratum -- the 'two strata' reading was a torus artefact")
        print("        (the same fold point at two 2pi-shifted representatives).")

    print("phase 2: full fibre at s=1 (10 real + 6 complex, exp38's construction)")
    census1 = census(sr5_model, target, solver=make_lm_solver(sr5_model), seeds=400,
                     rng=np.random.default_rng(2))
    fibre: list[np.ndarray] = []
    for candidate in census1.solutions:
        if not iks.physically_admissible(sr5_model, candidate):
            continue
        if any(iks.torus_distance(candidate, kept) < 1e-3 for kept in fibre):
            continue
        fibre.append(candidate)
    for s_star, q_star in folds:
        if not any(abs(s_star - d) < 1e-4 for d in (0.718354, 0.799244, 0.961848)):
            continue
        s0, pair = pair_at_fold(base_links, offset, target, path, q_star, s_star, circular)
        for q0 in pair:
            q_end, _, ok = track_path(base_links, offset, target, q0,
                                      make_s_path(s0, 1.0, all_fold_s))
            if ok and not any(torus_distance_complex(q_end, old) < 1e-2 for old in fibre):
                fibre.append(q_end)
    print(f"  fibre: {len(fibre)} roots (expect 16)")
    if len(fibre) != 16:
        print("  incomplete fibre -- aborting")
        return 1

    print("phase 3: each stratum's pair, tracked to s=1, matched onto the fibre")
    edges: list[tuple[int, int]] = []
    for k, (q_star, s_star) in enumerate(strata):
        s0, pair = pair_at_fold(base_links, offset, target, path, q_star, s_star, circular,
                                rho=args.rho)
        if not pair:
            print(f"  stratum {k}: pair not found on either side")
            continue
        endpoints = []
        for q0 in pair:
            q_end, _, ok = track_path(base_links, offset, target, q0,
                                      make_s_path(s0, 1.0, all_fold_s))
            if not ok:
                endpoints.append(None)
                continue
            distances = [torus_distance_complex(q_end, member) for member in fibre]
            endpoints.append(int(np.argmin(distances)) if min(distances) < 1e-2 else None)
        print(f"  stratum {k} (s*={s_star:.6f}): endpoints {endpoints}")
        if endpoints[0] is not None and endpoints[1] is not None:
            edges.append((min(endpoints), max(endpoints)))
    print(f"\nedges from the two strata: {edges}")
    print("merge with the 10 known edges and recompute components downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
