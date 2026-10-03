#!/usr/bin/env python3
"""exp39: what do the double-crossing witness paths encircle?  (task K-1)

DeepSeek's K-1 (``study/collab/to_kimi.md``): project the witnessed paths of the pairs whose
straight line crosses the singular set **twice** onto a workspace slice, and answer what they
wind around.  The pair studied: (5, 3) of exp19's pose 0 -- straight path crosses Sigma twice,
yet 48-waypoint walks connected it three times out of three (exp32/exp34).

Measurements (all numeric, no figures):

1. The two fold points ``q_a, q_b`` where the *straight* path crosses Sigma (refined minima of
   ``sigma_min`` reaching zero), and their images ``FK(q_a), FK(q_b)`` in the (x, y) slice.
2. A witnessed path (from ``resolve_pair``, path recorded per step), projected to (x, y) --
   a loop based at the pose's image.  Its **winding numbers around the two fold images**:
   odd winding around a cusp endpoint is Wenger's mechanism; winding around nothing (or an even
   wiggle past both fold images) is the "detour" reading.
3. ``det J`` of the witness path at its closest approach to each fold image: must be *nonzero*
   (the projection crosses the discriminant image at regular preimages = characteristic surface
   points, which is how a path can cross Delta without meeting Sigma).

Run::

    python3 -m study.exp39_witness_projection
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair


def witnessed_path(model, q_a: np.ndarray, q_b: np.ndarray, *, delta: float, circular,
                   waypoints: int, seed: int) -> list[np.ndarray] | None:
    """exp32's successful retry, keeping the path: direct, then random waypoints (same rng use)."""
    middle = 0.5 * (np.asarray(q_a) + np.asarray(q_b))
    rng = np.random.default_rng(seed)
    for _ in range(waypoints):
        for _ in range(300):
            candidate = middle + rng.normal(0.0, 0.35, size=middle.size)
            if iks.sigma_min(model, candidate) >= delta:
                break
        else:
            continue
        first = resolve_pair(model, q_a, candidate, delta=delta, circular=circular)
        if not first.connected:
            continue
        second = resolve_pair(model, candidate, q_b, delta=delta, circular=circular)
        if second.connected:
            return first.path + second.path
    return None


def winding_number(points: np.ndarray, around: np.ndarray) -> float:
    """Winding number of the closed 2D polyline ``points`` around ``around``."""
    relative = points - np.asarray(around)
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    return float(np.sum(np.diff(np.unwrap(angles))) / (2.0 * np.pi))


def main() -> int:
    """Project pair (5,3)'s witness path and measure what it winds around."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--pair", type=int, nargs=2, default=[5, 3])
    parser.add_argument("--waypoint-seeds", type=int, nargs=3, default=[1003, 1100, 1197])
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(0)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
    print(f"exp19 pose 0 reproduced: {len(solutions)} solutions "
          f"({fiber.converged}/{fiber.seeds} seeds)")
    i, j = args.pair
    q_a, q_b = solutions[i], solutions[j]

    print(f"\n1. straight path {tuple(args.pair)}: refined Sigma crossings")
    scan = iks.scan_path(model, q_a, q_b, circular=circular)
    end = iks.shortest_representative(q_a, q_b, circular)
    crossings = [(t, q_a + t * (end - q_a)) for t, v in scan.local_minima if v < 1e-9]
    print(f"   local minima refined: {[(round(t, 3), f'{v:.1e}') for t, v in scan.local_minima]}"
          f" -> {len(crossings)} true crossings")
    fold_images = []
    for t, q in crossings:
        image = model.fk(q)
        fold_images.append(image[:2, 3].copy())
        print(f"   crossing at t={t:.3f}: FK = ({image[0, 3]:.4f}, {image[1, 3]:.4f}), "
              f"det J = {float(np.linalg.det(model.jacobian(q))):.1e}")

    print(f"\n2. witnessed path (waypoint seeds {args.waypoint_seeds})")
    path = None
    for seed in args.waypoint_seeds:
        path = witnessed_path(model, q_a, q_b, delta=args.delta, circular=circular,
                              waypoints=48, seed=seed)
        if path is not None:
            print(f"   seed {seed}: connected, {len(path)} waypoints' path "
                  f"({len(path)} points)")
            break
        print(f"   seed {seed}: failed")
    if path is None:
        print("   no witness found this run -- aborting")
        return 1

    projection = np.array([model.fk(q)[:2, 3] for q in path])
    base_xy = target[:2, 3]
    print(f"   projection: {len(projection)} points, based at ({base_xy[0]:.4f}, "
          f"{base_xy[1]:.4f}), loop extent x [{projection[:, 0].min():.4f}, "
          f"{projection[:, 0].max():.4f}] y [{projection[:, 1].min():.4f}, "
          f"{projection[:, 1].max():.4f}]")
    for k, image in enumerate(fold_images):
        w = winding_number(projection, image)
        closest = int(np.argmin(np.linalg.norm(projection - image, axis=1)))
        det_at = float(np.linalg.det(model.jacobian(path[closest])))
        print(f"   fold image {k}: winding {w:+.2f} | closest approach "
              f"{np.linalg.norm(projection[closest] - image):.4f} m at step {closest}, "
              f"det J there {det_at:+.4f}")
    print("\nreading: odd winding around a fold image's endpoint = cusp mechanism;")
    print("zero winding while the straight line crosses twice = the detour reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
