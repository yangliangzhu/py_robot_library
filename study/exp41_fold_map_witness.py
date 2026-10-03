#!/usr/bin/env python3
"""exp41: the fold-curve map at the witness pose, and what the witness loop encircles (K-1).

exp39 showed the witnessed path of pair (5, 3) at exp19's pose 0 winds around *nothing* visible
(zero winding around both straight-crossing fold images), but the fold-curve map only existed
at exp21's pose.  This experiment builds the map at the witness pose itself:

1. fold spectrum of the s-family *at the witness pose* (exp35 machinery, pose-swapped);
2. (s, x) fold curves, then their crossings of s = 1 refined to exact fold seeds
   (exp36 machinery);
3. the fold curves of the real arm (s = 1) in the (x, y) slice through the pose;
4. the witnessed path of pair (5, 3) projected into the same slice, and then the K-1 questions:
   how often does the loop cross the fold curves, does it enclose a turnaround (a cusp
   candidate on the curve), and what does it wind around.

Run::

    python3 -m study.exp41_fold_map_witness
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study import sr0
from study.census import census, make_lm_solver
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp35_cusp import fold_seeds, make_augmented, track_fold_curve
from study.exp36_workspace_cusp import make_augmented_fixed_s, seeds_from_crossings
from study.exp39_witness_projection import winding_number, witnessed_path


def segment_intersections(loop: np.ndarray, curve: np.ndarray) -> int:
    """Crossings between two 2D polylines (proper segment intersections only)."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    count = 0
    for i in range(len(loop) - 1):
        a, b = loop[i], loop[i + 1]
        for k in range(len(curve) - 1):
            c, d = curve[k], curve[k + 1]
            o1, o2 = orient(a, b, c), orient(a, b, d)
            o3, o4 = orient(c, d, a), orient(c, d, b)
            if o1 * o2 < 0 and o3 * o4 < 0:
                count += 1
    return count


def main() -> int:
    """Build the fold map at the witness pose and overlay the witness loop."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", type=int, nargs=2, default=[5, 3])
    parser.add_argument("--waypoint-seed", type=int, default=1003)
    parser.add_argument("--u-range", type=float, default=0.15)
    args = parser.parse_args()

    model = ModelFactory.create("sr5", backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(0)
    target, _ = iks.random_reachable_target(model, rng)
    fibre = census(model, target, solver=make_lm_solver(model), seeds=600, rng=rng)
    solutions = [q for q in fibre.solutions if iks.sigma_min(model, q) > 5e-3]
    q_a, q_b = solutions[args.pair[0]], solutions[args.pair[1]]
    print(f"witness pose reproduced: {len(solutions)} solutions, pair {tuple(args.pair)}")

    print("phase 1: fold spectrum at the witness pose")
    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    folds = fold_seeds(path, target, sr0.solve(sr0.robot(), target))
    print(f"  {len(folds)} folds: {[f'{s:.4f}' for s, _ in folds]}")

    print("phase 2: (s, x) fold curves and their s=1 crossings")
    augmented_su = make_augmented(path, target, 0, "pos")
    u0 = float(target[0, 3])
    curves_su: list[np.ndarray] = []
    for s_star, q_star in folds:
        y0 = np.concatenate([q_star, [s_star, u0]])
        for direction in (1.0, -1.0):
            curves_su.append(track_fold_curve(augmented_su, y0, direction=direction,
                                              u_range=args.u_range))
    links1 = path(1.0)
    augmented_xy = make_augmented_fixed_s(links1, target, 0, 1, "pos")
    v0 = float(target[1, 3])
    seeds = seeds_from_crossings(curves_su, augmented_xy, 1.0, v0)
    print(f"  {len(seeds)} exact fold seeds of the real arm near the witness pose")

    print("phase 3: fold curves of the (x, y) slice at s=1")
    slice_curves: list[np.ndarray] = []
    for q, u in seeds:
        y0 = np.concatenate([q, [u, v0]])
        for direction in (1.0, -1.0):
            slice_curves.append(track_fold_curve(augmented_xy, y0, direction=direction,
                                                 u_range=args.u_range,
                                                 s_bounds=(-10.0, 10.0)))
    print(f"  {len(slice_curves)} slice curves tracked")
    from study.exp35_cusp import find_turnarounds  # noqa: PLC0415

    turnarounds = find_turnarounds(slice_curves)
    print(f"  turnarounds (cusp candidates): {len(turnarounds)}")
    for ci, point in turnarounds:
        print(f"    curve {ci}: turnaround at (x, y) = ({point[6]:.4f}, {point[7]:.4f})")

    print("phase 4: the witness loop of pair (5, 3) overlaid")
    path_q = witnessed_path(model, q_a, q_b, delta=5e-3, circular=circular, waypoints=48,
                            seed=args.waypoint_seed)
    if path_q is None:
        print("  witness walk failed this run")
        return 1
    loop = np.array([model.fk(q)[:2, 3] for q in path_q])
    print(f"  witness loop: {len(loop)} points")
    for k, curve in enumerate(slice_curves):
        crossings = segment_intersections(loop, curve[:, 6:8])
        distance = min(
            float(np.linalg.norm(p - c)) for p in loop for c in curve[:, 6:8]
        ) if len(curve) else float("nan")
        if crossings or distance < 0.02:
            print(f"  slice curve {k}: crossings with loop = {crossings}, "
                  f"min distance = {distance:.4f} m")
    for ci, point in turnarounds:
        winding = winding_number(loop, point[6:8])
        print(f"  turnaround of curve {ci}: winding number of the loop around it = {winding:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
