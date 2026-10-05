#!/usr/bin/env python3
"""exp55: kimi's cusp detector on DeepSeek's 3R benchmark (task K-6).

DeepSeek's `exp54_cusp_discriminant` settled the 3R analytically: the position IK is a quadratic
in ``t^2``, the discriminant image is the *smooth* conic
``C2 = 16 R^2 + 32 R Z - 264 R + 16 Z^2 - 136 Z + 289`` (``R = rho^2, Z = z^2``), with
``nabla C2`` never zero -- hence **no cusp**, while the arm *is* cuspidal (exp48's witness).
That makes the arm the negative control for my cusp detector: it must report "no cusp" here,
or the detector is broken.

The detector is the same machinery as exp35/exp36, dimensioned down to the position map:
folds satisfy ``position(q) = (rho, 0, z)`` and ``det J_position(q) = 0`` -- 4 equations in
``(q, rho, z)`` (5 unknowns), i.e. fold *curves* in the meridian; a cusp is a turnaround of
such a curve or a meeting of two.  Independently, the curve must lie on the conic: ``|C2|``
along it is a cross-check of both instruments.

Run::

    python3 -m study.exp55_3r_cusp_check
"""

from __future__ import annotations

import argparse

import numpy as np

from study import cuspidal3r


def make_augmented(model):
    """The 4-vector [position error; det J] as a function of y = (q, rho, z)."""

    def augmented(y: np.ndarray) -> np.ndarray:
        q, rho, z = y[:3], y[3], y[4]
        error = cuspidal3r.position(model, q) - np.array([rho, 0.0, z])
        return np.concatenate([error, [cuspidal3r.det_position(model, q)]])

    return augmented


def _jacobian5(augmented, y: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Central-difference 4x5 Jacobian."""
    jacobian = np.zeros((4, 5))
    for k in range(5):
        step = np.zeros(5)
        step[k] = h
        jacobian[:, k] = (augmented(y + step) - augmented(y - step)) / (2 * h)
    return jacobian


def refine_fold_3r(augmented, q0: np.ndarray, z: float, *, iterations: int = 60,
                   tolerance: float = 1e-12):
    """Newton on the 4x4 system at fixed z: unknowns (q, rho)."""
    y = np.concatenate([np.asarray(q0, dtype=float), [0.0, z]])
    for _ in range(iterations):
        value = augmented(y)
        if float(np.linalg.norm(value)) < tolerance:
            break
        jacobian = np.zeros((4, 4))
        for column in range(4):  # unknowns are q and rho, not z
            step = np.zeros(5)
            step[column] = 1e-7
            jacobian[:, column] = (augmented(y + step) - augmented(y - step)) / 2e-7
        y[:4] += np.linalg.lstsq(jacobian, -value, rcond=None)[0]
    return y[:3].copy(), float(y[3]), float(np.linalg.norm(augmented(y)))


def track_fold_3r(augmented, y0: np.ndarray, *, direction: float, h0: float = 2e-3,
                  max_steps: int = 600, z_bounds: tuple[float, float] = (-2.0, 4.0),
                  rho_bounds: tuple[float, float] = (1e-3, 4.0)):
    """Pseudo-arclength continuation of the fold curve (chord Newton), as in exp35."""
    jacobian = _jacobian5(augmented, y0)
    tangent = np.linalg.svd(jacobian)[2][-1]
    tangent = tangent / np.linalg.norm(tangent) * direction
    y, h = y0.copy(), h0
    points = [y.copy()]
    failures = 0
    for _ in range(max_steps):
        prediction = y + h * tangent
        z = prediction.copy()
        converged = False
        for _ in range(15):
            value = augmented(z)
            if float(np.linalg.norm(value)) < 1e-11:
                converged = True
                break
            big = np.vstack([jacobian, tangent])
            z = z + np.linalg.lstsq(big, np.concatenate([-value, [0.0]]), rcond=None)[0]
        if not converged:
            h *= 0.5
            failures += 1
            if h < 2e-5 or failures > 12:
                break
            continue
        new_jacobian = _jacobian5(augmented, z)
        new_tangent = np.linalg.svd(new_jacobian)[2][-1]
        if float(np.dot(new_tangent, tangent)) < 0:
            new_tangent = -new_tangent
        y, jacobian, tangent = z, new_jacobian, new_tangent
        points.append(y.copy())
        h = min(h * 1.3, 0.02) if failures == 0 else h
        if not (z_bounds[0] < y[4] < z_bounds[1]) or not (rho_bounds[0] < y[3] < rho_bounds[1]):
            break
    return np.array(points)


def find_turnarounds_3r(curves: list[np.ndarray], *, window: int = 5,
                        min_excursion: float = 4e-3) -> list[tuple[int, np.ndarray]]:
    """Candidate cusps: the (rho, z) projection of a fold curve turns around in rho."""
    candidates = []
    for ci, points in enumerate(curves):
        if len(points) < 2 * window + 1:
            continue
        rho = points[:, 3]
        for i in range(window, len(points) - window):
            before = rho[i] - rho[i - window]
            after = rho[i + window] - rho[i]
            if before * after < 0 and abs(before) > min_excursion and abs(after) > min_excursion:
                candidates.append((ci, points[i]))
                break
    return candidates


def conic(rho: float, z: float) -> float:
    """DeepSeek's discriminant conic C2 in (R, Z) = (rho^2, z^2)."""
    big_r, big_z = rho * rho, z * z
    return 16 * big_r**2 + 32 * big_r * big_z - 264 * big_r + 16 * big_z**2 - 136 * big_z + 289


def main() -> int:
    """Run the detector on the benchmark 3R."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.02, help="|det J| for fold seeds")
    args = parser.parse_args()

    model = cuspidal3r.build()
    print(cuspidal3r.rank_report(model))
    augmented = make_augmented(model)

    print(f"phase 1: fold seeds ({args.seeds} random configs, |det J| < {args.threshold})")
    rng = np.random.default_rng(11)
    folds: list[tuple[np.ndarray, float, float]] = []
    for _ in range(args.seeds):
        q0 = rng.uniform(-np.pi, np.pi, 3)
        if abs(cuspidal3r.det_position(model, q0)) > args.threshold:
            continue
        z0 = float(cuspidal3r.position(model, q0)[2])
        q_star, rho_star, residual = refine_fold_3r(augmented, q0, z0)
        if residual > 1e-9:
            continue
        if any(np.linalg.norm(q_star - old[0]) < 0.05 for old in folds):
            continue
        folds.append((q_star, rho_star, z0))
    print(f"  {len(folds)} distinct fold points")

    print("phase 2: fold curves in the (rho, z) meridian")
    curves: list[np.ndarray] = []
    for q_star, rho_star, z_star in folds:
        y0 = np.concatenate([q_star, [rho_star, z_star]])
        for direction in (1.0, -1.0):
            points = track_fold_3r(augmented, y0, direction=direction)
            curves.append(points)
    sizes = [len(c) for c in curves]
    print(f"  {len(curves)} half-curves, sizes {sizes}")

    print("phase 3: turnaround (cusp) candidates + conic cross-check")
    turnarounds = find_turnarounds_3r(curves)
    print(f"  turnarounds: {len(turnarounds)}")
    worst = 0.0
    for curve in curves:
        if len(curve) == 0:
            continue
        values = [abs(conic(float(p[3]), float(p[4]))) for p in curve]
        worst = max(worst, max(values))
    print(f"  max |C2| along all fold curves: {worst:.3e} (0 = the curves lie on the conic)")
    for ci, point in turnarounds:
        rho_c, z_c = float(point[3]), float(point[4])
        print(f"  candidate on curve {ci}: (rho, z) = ({rho_c:.4f}, {z_c:.4f}), "
              f"C2 = {conic(rho_c, z_c):.2e}")
        for side in (1.0, -1.0):
            target = np.array([rho_c + side * 2e-3, 0.0, z_c])
            fibre = cuspidal3r.census_position(model, target, seeds=200)
            near = 0
            for q in fibre.solutions:
                if cuspidal3r.torus_distance(q, point[:3]) < 0.3:
                    near += 1
            print(f"    roots near q_c at rho{'+' if side > 0 else '-'}2e-3: {near}")
    verdict = ("no cusp found -- agrees with exp54 (negative control PASSED)"
               if not turnarounds else "candidates found -- needs review against exp54")
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
