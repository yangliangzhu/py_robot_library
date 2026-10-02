#!/usr/bin/env python3
"""exp36: cusp hunt in a workspace slice at fixed s -- the classical Wenger picture.

exp35 swept (s, u) slices and found no cusp in range.  This experiment freezes ``s`` (default
``s = 1``, the real SR5) and frees **two** pose coordinates ``(u, v)`` -- the slice in which
Wenger's cusp criterion lives: the discriminant is a plane curve, and its cusp points are where
three IK solutions coalesce.

Pipeline:

1. **Seeds.**  Load exp35's saved ``(s, u)`` fold curves (``--save-curves`` npz), take every
   crossing of ``s = s_fixed``, and refine each to an exact fold point ``(q, u)`` of the
   fixed-parameter arm (7 equations -- 6 FK + det J -- in 7 unknowns ``(q, u)``).
2. **Curves.**  Track the fold locus in the ``(u, v)`` slice from each seed with exp35's
   pseudo-arclength continuation (augmented system is again 7 equations in 8 unknowns
   ``(q, u, v)``).
3. **Detection.**  Two-curve meetings and single-curve turnarounds, each verified by the
   triple-root test (root count 3 vs 1) and the split-exponent test (~1/3 vs ~1/2).
4. **The loop.**  Encircle the first verified cusp in the ``(u, v)`` plane and read the
   survivor map plus deaths (kimi's predicted cusp signature: a survivor that moved).

Run::

    python3 -m study.exp36_workspace_cusp --crossings-from /tmp/study_verify/curves/u0.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp35_cusp import (
    count_roots_near,
    find_meetings,
    find_turnarounds,
    target_at,
    track_fold_curve,
)


def make_augmented_fixed_s(links, target: np.ndarray, u_index: int, v_index: int,
                           u_kind: str = "pos"):
    """The 7-vector [se3 error; det J] as a function of y = (q, u, v) at fixed s."""

    def augmented(y: np.ndarray) -> np.ndarray:
        q, u, v = y[:6], y[6], y[7]
        moved = target_at(target_at(target, u_index, u, u_kind), v_index, v, u_kind)
        error = homotopy.se3_error(homotopy.fk_links(links, q), moved)
        determinant = float(np.linalg.det(homotopy.jacobian_q(links, q)))
        return np.concatenate([error, [determinant]])

    return augmented


def refine_fixed_s_fold(augmented, q: np.ndarray, u: float, v_fixed: float, *,
                        iterations: int = 40, tolerance: float = 1e-12) -> tuple[np.ndarray, float, float]:
    """Newton on the fixed-s augmented system: solve ``(q, u)`` at ``v = v_fixed``.

    The full point is ``y8 = (q, u, v_fixed)``; the Newton Jacobian is the 7x7 block over
    ``(q, u)`` (7 equations -- 6 FK + det J -- in 7 unknowns).
    """
    y8 = np.concatenate([np.asarray(q, dtype=float), [u, v_fixed]])
    for _ in range(iterations):
        value = augmented(y8)
        if float(np.linalg.norm(value)) < tolerance:
            break
        jacobian = np.zeros((7, 7))
        for column in range(7):
            step = np.zeros(8)
            step[column] = 1e-7
            jacobian[:, column] = (augmented(y8 + step) - augmented(y8 - step)) / 2e-7
        y8[:7] += np.linalg.lstsq(jacobian, -value, rcond=None)[0]
    return y8[:6].copy(), float(y8[6]), float(np.linalg.norm(augmented(y8)))


def seeds_from_crossings(curves: list[np.ndarray], augmented, s_fixed: float, v_fixed: float,
                         *, dedup_u: float = 2e-3, dedup_q: float = 0.1):
    """Exact fixed-s fold points from the (s, u) curves' crossings of ``s = s_fixed``."""
    seeds: list[tuple[np.ndarray, float]] = []
    for points in curves:
        s = points[:, 6]
        for i in range(len(points) - 1):
            if (s[i] - s_fixed) * (s[i + 1] - s_fixed) > 0:
                continue
            fraction = (s_fixed - s[i]) / (s[i + 1] - s[i]) if s[i + 1] != s[i] else 0.0
            guess = points[i] + fraction * (points[i + 1] - points[i])
            q, u, residual = refine_fixed_s_fold(augmented, guess[:6], guess[7], v_fixed)
            if residual > 1e-8:
                continue
            if any(abs(u - old_u) < dedup_u
                   and float(np.linalg.norm(q - old_q)) < dedup_q for old_q, old_u in seeds):
                continue
            seeds.append((q, u))
    return seeds


def main() -> int:
    """Run the fixed-s workspace-slice cusp hunt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crossings-from", required=True,
                        help="npz of exp35's saved (s, u) fold curves")
    parser.add_argument("--s-fixed", type=float, default=1.0)
    parser.add_argument("--u-index", type=int, default=0)
    parser.add_argument("--v-index", type=int, default=1)
    parser.add_argument("--v-range", type=float, default=0.15)
    parser.add_argument("--u-kind", default="pos", choices=["pos", "rot"])
    args = parser.parse_args()

    archive = np.load(args.crossings_from)
    target = archive["target"]
    curves = [archive[key] for key in sorted(k for k in archive if k.startswith("curve_"))]
    print(f"loaded {len(curves)} (s, u) curves from {args.crossings_from}")

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    links_fixed = path(args.s_fixed)
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)

    augmented = make_augmented_fixed_s(links_fixed, target, args.u_index, args.v_index,
                                       args.u_kind)
    v0 = 0.0 if args.u_kind == "rot" else float(target[:3, 3][args.v_index])
    seeds = seeds_from_crossings(curves, augmented, args.s_fixed, v0)
    print(f"{len(seeds)} exact fold seeds at s = {args.s_fixed}:")
    for q, u in seeds:
        print(f"  u = {u:+.4f}, det = {np.linalg.det(homotopy.jacobian_q(links_fixed, q)):.1e}")
    if not seeds:
        print("no crossings of the fixed s level in the loaded curves -- nothing to track")
        return 0

    print(f"phase 2: fold curves in the (u{args.u_index}, v{args.v_index}) slice "
          f"(+/- {args.v_range} in v)")
    slice_curves: list[np.ndarray] = []
    for k, (q, u) in enumerate(seeds):
        y0 = np.concatenate([q, [u, v0]])
        for direction in (1.0, -1.0):
            points = track_fold_curve(augmented, y0, direction=direction,
                                      u_range=args.v_range, s_bounds=(-10.0, 10.0))
            slice_curves.append(points)
            print(f"  seed {k} (u={u:+.4f}) dir {direction:+.0f}: {len(points)} points, "
                  f"v {points[-1][7]:.4f}, u {points[-1][6]:.4f}")
    meetings = find_meetings(slice_curves)
    # drop the trivial meetings at the seed points themselves (u_seed, v0)
    meetings = [m for m in meetings if abs(m[2][7] - v0) > 1e-3]
    turnarounds = find_turnarounds(slice_curves)
    print(f"phase 3: {len(meetings)} meeting(s), {len(turnarounds)} turnaround(s)")
    verified: list[np.ndarray] = []
    candidates = [(f"curves {a} x {b}", point) for a, b, point in meetings]
    candidates += [(f"turnaround on curve {ci}", point) for ci, point in turnarounds]
    for label, point in candidates:
        print(f"  {label}: candidate at u={point[6]:.6f} v={point[7]:.6f}")
        counts = []
        for side in (1.0, -1.0):
            moved = target_at(target, args.v_index, point[7] + side * 2e-3, args.u_kind)
            count = count_roots_near(sr0.robot(links_fixed), moved, point[:6], circular)
            counts.append(count)
            print(f"    roots near q_c at v{'+' if side > 0 else '-'}2e-3: {count}")
        if max(counts) >= 3:
            print("    -> VERIFIED cusp (three roots coalesce on one side)")
            verified.append(point)
        else:
            print("    -> not a cusp (fold)")
    if verified:
        print("phase 4: encircling loop in the (u, v) plane around the first verified cusp")

        cusp = verified[0]
        # the loop lives in the (u, v) slice at fixed s, so drive it directly
        angles = np.linspace(0.0, 2.0 * np.pi, 241)
        points = [(cusp[6] + 5e-3 * np.cos(a), cusp[7] + 5e-3 * np.sin(a)) for a in angles]
        from study.census import census, make_lm_solver  # noqa: PLC0415

        base_u, base_v = points[0]
        base_model = sr0.robot(links_fixed)
        base_target = target_at(target_at(target, args.u_index, base_u, args.u_kind),
                                args.v_index, base_v, args.u_kind)
        fibre = census(base_model, base_target, solver=make_lm_solver(base_model), seeds=200,
                       rng=np.random.default_rng(3))
        base_fibre: list[np.ndarray] = []
        for candidate in fibre.solutions:
            if not iks.physically_admissible(base_model, candidate):
                continue
            if any(iks.torus_distance(candidate, kept) < 1e-3 for kept in base_fibre):
                continue
            base_fibre.append(candidate)
        print(f"  loop base: u={base_u:.4f} v={base_v:.4f}, {len(base_fibre)} solutions")
        mapping, deaths = {}, {}
        for index, q0 in enumerate(base_fibre):
            q, alive = q0.copy(), True
            for k, (u, v) in enumerate(points[1:], start=1):
                before = q.copy()
                moved = target_at(target_at(target, args.u_index, u, args.u_kind),
                                  args.v_index, v, args.u_kind)
                q, residual = homotopy.correct(links_fixed, q, moved, tolerance=1e-11)
                step = float(np.linalg.norm(q - before))
                clearance = homotopy.sigma_min_links(links_fixed, q) if residual < 1e-8 else 0.0
                if residual > 1e-8 or clearance < 2e-3 or step > 0.5:
                    deaths[index] = (f"died at sample {k}: residual {residual:.1e}, "
                                     f"clearance {clearance:.1e}")
                    alive = False
                    break
            if alive:
                distances = [iks.configuration_distance(q, other, circular)
                             for other in base_fibre]
                mapping[index] = int(np.argmin(distances))
        print(f"  survivor map: {mapping}")
        moved = [i for i, j in mapping.items() if i != j]
        print(f"  survivors moved: {moved if moved else 'none (identity)'}")
        for index, why in sorted(deaths.items()):
            print(f"  death: solution {index}: {why}")
    else:
        print("phase 4: no verified cusp in this slice -- no loop to build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
