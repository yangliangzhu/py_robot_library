#!/usr/bin/env python3
"""exp27: classifying a fold by tracking its pair -- structure instead of sampling

exp26 tried to decide whether a fold is a birth, a death or an invisible turning point by asking a
census to find a real solution near the fold.  It failed for a structural reason: at gap ``eps`` the
coalescing pair sits at ``c sqrt(eps)`` from the fold with ``c`` unknown, so a fixed search radius
either misses it or catches a neighbour.

The two facts that make this decidable without sampling:

* at a fold the pair separates along the **null direction** of the Jacobian -- the right singular
  vector of its smallest singular value -- so the direction is known from the fold itself;
* the distance follows the measured square-root law (3.16), so the seed for a solve at gap ``eps`` is
  ``q* + c sqrt(eps) v`` for a small set of amplitudes ``c``.

Seeding a *local* solve that way asks a well-posed question ("is there a solution here?") instead of a
statistical one ("did the census happen to find it?"), and the classification then feeds the law of
3.20: the fibre changes by two per event, zero per invisible fold.

Run::

    python3 -m study.exp27_pair_tracking --pose-seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.census import solve_lm
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import approach, refine_fold

#: Amplitudes of the square-root seed, in the units of the fitted law (3.16 measured c ~ 1-2).
AMPLITUDES = (0.2, 0.5, 1.0, 2.0, 4.0)


def null_direction(model, q: np.ndarray) -> np.ndarray:
    """Right singular vector of the smallest singular value of the pose Jacobian."""
    _u, _s, vh = np.linalg.svd(np.asarray(model.jacobian(q), dtype=float))
    vector = vh[-1]
    return vector / np.linalg.norm(vector)


def pair_present(offset: float, target, s_star: float, q_star: np.ndarray, gap: float,
                 circular, *, radii: float = 8.0) -> tuple[bool, float]:
    """Whether a real solution sits near ``q*`` at ``s* +- gap``, seeding along the null direction."""
    s = s_star + gap
    if not 0.0 <= s <= 1.0:
        return False, float("nan")
    model = sr0.robot(path_at(offset)(s))
    direction = null_direction(model, q_star)
    best = float("nan")
    for amplitude in AMPLITUDES:
        seed = q_star + amplitude * np.sqrt(abs(gap)) * direction
        solution, converged = solve_lm(model, seed, target)
        if not converged:
            continue
        distance = iks.configuration_distance(solution, q_star, circular)
        if distance < radii * np.sqrt(abs(gap)) + 1e-3:
            best = distance if np.isnan(best) else min(best, distance)
    return (not np.isnan(best)), best


def classify(offset: float, target, s_star: float, q_star: np.ndarray, circular, *,
             gaps: tuple[float, ...] = (1e-4, 1e-3, 5e-3)) -> tuple[str, dict]:
    """Classify a fold by tracking its pair at several gaps on both sides."""
    evidence: dict[str, list] = {"before": [], "after": []}
    for gap in gaps:
        for label, signed in (("before", -gap), ("after", +gap)):
            present, distance = pair_present(offset, target, s_star, q_star, signed, circular)
            evidence[label].append((present, distance))
    before = sum(1 for present, _d in evidence["before"] if present)
    after = sum(1 for present, _d in evidence["after"] if present)
    if before and not after:
        kind = "death"
    elif after and not before:
        kind = "birth"
    else:
        kind = "invisible"
    return kind, evidence


def main() -> int:
    """Classify the folds of one pose by pair tracking."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--window", type=float, default=0.012)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    pose_rng = np.random.default_rng(args.pose_seed)
    q_true = pose_rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    seeds = [s.q for s in sr0.solve(sr0_model, target)]

    path = path_at(offset)
    candidates = []
    for seed in seeds:
        for s, q in approach(path, target, seed, 1.0):
            if iks.sigma_min(sr0.robot(path(s)), q) < args.near:
                candidates.append((s, q))
    folds = []
    for s, q in candidates:
        s_star, q_star, residual = refine_fold(path, target, q, s, iterations=args.iterations)
        if residual > 1e-8:
            continue
        if not any(
            abs(s_star - other) < 1e-4 and float(np.linalg.norm(q_star - q_other)) < 1e-3
            for other, q_other in folds
        ):
            folds.append((s_star, q_star))
    folds.sort()
    print(f"pose seed {args.pose_seed}: {len(folds)} folds, classified by tracking their pair\n")
    groups: list[list[tuple[str, float]]] = []
    for s_star, q_star in folds:
        kind, evidence = classify(offset, target, s_star, q_star, circular)
        model = sr0.robot(path(s_star))
        print(
            f"  fold {s_star:.6f}: {kind:9s} | pair found before {evidence['before']} "
            f"after {evidence['after']} | clearance {iks.sigma_min(model, q_star):.1e}"
        )
        if groups and s_star - groups[-1][-1][1] <= args.window:
            groups[-1].append((kind, s_star))
        else:
            groups.append([(kind, s_star)])
    print("\nprediction per group (two solutions per event, zero per invisible fold):")
    for group in groups:
        events = [kind for kind, _s in group if kind != "invisible"]
        predicted = 2 * sum(1 if kind == "birth" else -1 for kind in events)
        print(
            f"  group at s ~ {group[0][1]:.4f}: {len(group)} fold(s) "
            f"{[kind for kind, _s in group]} -> {len(events)} event(s), predicted delta "
            f"{predicted:+d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
