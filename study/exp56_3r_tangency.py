#!/usr/bin/env python3
"""exp56: the tangency test at the two component intersections (the decisive K-6 follow-up).

DeepSeek withdrew the "no cusp" verdict: the full discriminant of the 3R's IK factors as
``Disc = A0 . A1 . C2^2``, and component intersections are tangent by construction (all
gradients depend only on S = R + Z).  Three pairwise intersections exist; two are on the
workspace boundary with a higher-order root: ``(rho, z) = (0.7071, +-1.3229)`` (A1 = C2) and
``(3.5355, +-1.3229)`` (A0 = C2).  The decisive question: are the two boundary arcs *tangent*
there (a genuine cusp / tacnode-like contact) or do they cross transversally (a node)?

This instrument:

1. finds fold points near each candidate and splits them by stratum (elbow ``q3 = 0`` vs not);
2. tracks each stratum's curve through the intersection and measures the angle between the
   tangents at the closest approach (0 degrees = tangent contact);
3. measures the local root structure at the exact point (how many roots coalesce).

Run::

    python3 -m study.exp56_3r_tangency
"""

from __future__ import annotations

import argparse

import numpy as np

from study import cuspidal3r as c3
from study.exp54_cusp_discriminant import coefficients
from study.exp55_3r_cusp_check import make_augmented, refine_fold_3r, track_fold_3r

CANDIDATES = [(0.7071, 1.3229), (0.7071, -1.3229), (3.5355, 1.3229), (3.5355, -1.3229)]


def folds_near(model, augmented, rho_c: float, z_c: float, *, seeds: int = 400,
               window: float = 0.35) -> list[tuple[np.ndarray, float, float]]:
    """Fold points whose workspace image lies within ``window`` of the candidate."""
    rng = np.random.default_rng(19)
    found: list[tuple[np.ndarray, float, float]] = []
    for _ in range(seeds):
        q0 = rng.uniform(-np.pi, np.pi, 3)
        if abs(c3.det_position(model, q0)) > 0.05:
            continue
        z0 = float(c3.position(model, q0)[2])
        q_star, rho_star, residual = refine_fold_3r(augmented, q0, z0)
        if residual > 1e-9:
            continue
        if abs(rho_star - rho_c) > window or abs(z0 - z_c) > window:
            continue
        if any(np.linalg.norm(q_star - old[0]) < 0.05 for old in found):
            continue
        found.append((q_star, rho_star, z0))
    return found


def main() -> int:
    """Measure tangency and root structure at the two boundary candidates."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    model = c3.build()
    augmented = make_augmented(model)
    for rho_c, z_c in CANDIDATES:
        a0, a1, a2 = coefficients(rho_c, z_c)
        print(f"\n=== candidate (rho, z) = ({rho_c}, {z_c:+.4f}) "
              f"| A0={a0:.2e} A1={a1:.2e} A2={a2:.2e} ===")
        folds = folds_near(model, augmented, rho_c, z_c)
        is_elbow = [abs(f[0][2]) < 0.1 or abs(abs(f[0][2]) - np.pi) < 0.1 for f in folds]
        elbow = [f for f, flag in zip(folds, is_elbow, strict=True) if flag]
        other = [f for f, flag in zip(folds, is_elbow, strict=True) if not flag]
        print(f"  {len(folds)} fold points near the candidate, elbow-stratum: {len(elbow)}")

        # track curves from representatives of each stratum and measure tangents
        tangents: list[np.ndarray] = []
        for label, group in (("elbow", elbow), ("other", other)):
            if not group:
                continue
            q_star, rho_star, z_star = group[0]
            y0 = np.concatenate([q_star, [rho_star, z_star]])
            parts = [track_fold_3r(augmented, y0, direction=direction)
                     for direction in (1.0, -1.0)]
            curve = np.concatenate([parts[0][::-1], parts[1]])
            # tangent at the closest approach to the candidate
            distances = np.linalg.norm(curve[:, 3:5] - np.array([rho_c, z_c]), axis=1)
            k = int(np.argmin(distances))
            lo, hi = max(0, k - 4), min(len(curve), k + 5)
            direction_vec = curve[hi - 1, 3:5] - curve[lo, 3:5]
            norm = float(np.linalg.norm(direction_vec))
            tangent = direction_vec / norm if norm > 0 else np.array([np.nan, np.nan])
            tangents.append(tangent)
            print(f"  {label}: curve of {len(curve)} pts, closest {distances[k]:.2e} to the "
                  f"candidate, tangent {np.round(tangent, 4)}")
        if len(tangents) == 2 and np.all(np.isfinite(tangents)):
            cosine = float(abs(np.dot(tangents[0], tangents[1])))
            angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
            print(f"  angle between the two strata's tangents: {angle:.2f} deg "
                  f"({'TANGENT CONTACT (cusp-like)' if angle < 5 else 'transversal (node)'})")

        # local root structure at the exact candidate point
        fibre = c3.census_position(model, np.array([rho_c, 0.0, z_c]), seeds=300)
        sols = fibre.solutions
        dists = sorted(
            c3.torus_distance(sols[i], sols[j])
            for i in range(len(sols)) for j in range(i + 1, len(sols))
        ) if len(sols) >= 2 else []
        print(f"  fibre at the candidate: {len(sols)} solutions, "
              f"closest pairs {[f'{d:.1e}' for d in dists[:4]]}")


if __name__ == "__main__":
    raise SystemExit(main())
