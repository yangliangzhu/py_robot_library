#!/usr/bin/env python3
"""exp45: adversarial cross-check of the fold transposition edges behind the S6 verdict.

DeepSeek's `exp44_group_from_edges` turns kimi's measured edges into the verdict "no radical
solution": the five edges on the six-root component generate S6 (order 720, verified by closure).
Everything else in the chain is theorem; **the edges are the measurement**.  This experiment
re-measures four of the five critical edges by an independent method -- exp38 used pair tracking
to s=1 and endpoint matching; here the **whole 16-root fibre is carried around each fold** in the
complex s-plane (the exp37 loop machine), and the permutation is read off directly.

The four folds covered: 0.655086 (birth), 0.656878 (birth), 0.718354 (death), 0.799244 (death) --
four of the five edges of the S6 component; the fifth (branch point 0.9088+0.0629i) was itself
loop-measured in exp40.

Run::

    python3 -m study.exp45_fold_loops
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

#: the edges exp38 measured by pair tracking, for these four folds
EXPECTED = {0.655086: (0, 10), 0.656878: (7, 13), 0.718354: (10, 11), 0.799244: (12, 13)}


def main() -> int:
    """Loop the whole fibre around each fold and compare the transpositions with exp38's."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rho", type=float, default=1e-3, help="loop radius around each fold")
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    base_links = link_transforms()
    sr5_model = sr0.robot(base_links)
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)

    print("fold seeds (branches 0 and 6 carry the whole documented spectrum)")
    fibre0 = sr0.solve(sr0.robot(), target)
    folds = fold_seeds(path, target, [fibre0[0], fibre0[6]])
    print(f"  {len(folds)} folds: {[f'{s:.4f}' for s, _ in folds]}")

    print("full fibre at s=1 (10 real census + 6 complex from the death folds)")
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
                                      make_s_path(s0, 1.0, [s for s, _ in folds]))
            if ok and not any(torus_distance_complex(q_end, old) < 1e-2 for old in fibre):
                fibre.append(q_end)
    print(f"  fibre: {len(fibre)} roots (expect 16)")
    if len(fibre) != 16:
        print("  incomplete fibre -- aborting")
        return 1

    print("loops: whole fibre around each fold, permutation read on the base fibre")
    all_fold_s = [s for s, _ in folds]
    for fold_s in EXPECTED:
        base = fold_s + args.rho if fold_s < 0.7 else fold_s - args.rho
        # base side with the real pair (birth: right, death: left), matching exp38's pairing
        if fold_s in (0.655086, 0.656878):
            base = fold_s + args.rho
        out = make_s_path(1.0, base, all_fold_s)
        # the circle starts at the angle of the base point (the pair is real on one side only):
        # angle 0 for birth folds (base right of the fold), pi for death folds (base left)
        start = 0.0 if base > fold_s else float(np.pi)
        circle = [fold_s + args.rho * np.exp(1j * (start + t))
                  for t in np.linspace(0, 2 * np.pi, args.samples + 1)[1:]]
        # the return arcs run through the same (upper) half-plane: out and back then cancel
        # around every intervening fold, and the measured permutation is the target fold's
        back = make_s_path(base, 1.0, all_fold_s)
        full = out + circle + back[1:]
        images: list[int] = []
        for q0 in fibre:
            # max_step 0.02: at an intervening fold's arc the ramified pair is ~0.05 rad
            # apart, so a 0.1 cap lets Newton hop between the sheets (measured: the fold's
            # own transposition contaminates the readout); 0.02 stays under it
            q_end, worst, ok = track_path(base_links, offset, target, q0, full,
                                          max_step=0.02)
            if not ok:
                images.append(-1)
                continue
            distances = [torus_distance_complex(q_end, member) for member in fibre]
            images.append(int(np.argmin(distances)) if min(distances) < 1e-2 else -1)
        moved = sorted((i, j) for i, j in enumerate(images) if 0 <= j != i)
        untracked = images.count(-1)
        same = {tuple(sorted(edge)) for edge in moved} == {tuple(sorted(EXPECTED[fold_s]))}
        print(f"  fold {fold_s}: moved {moved}, untracked {untracked} | "
              f"exp38's edge {EXPECTED[fold_s]} -> {'MATCH' if same else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
