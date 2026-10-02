#!/usr/bin/env python3
"""exp37: the transposition around one fold, measured with a complex loop.

The real-loop experiments could not produce a non-identity permutation (kimi's prediction:
avoiding the discriminant forces the identity; a real loop's swap needs sheet deaths), and the
cusp hunt has not located a cusp in range.  The clean way to see the branching itself is a
**complex** loop: encircle one fold ``s*`` in the complex parameter plane and track the two
coalescing roots with a complex Newton corrector.  Square-root branching says they must come
back *swapped* -- a transposition, the first non-identity monodromy measured in this study.

Controls (the study's convention):

* a **far root** carried around the same loop must return to itself;
* a loop of the same radius centred away from every fold must return every root to itself.

The tracker is algebraic (12-component FK residual, complex least-squares Newton), because
``homotopy.se3_error``'s arccos log-error is not complex-safe.

Run::

    python3 -m study.exp37_complex_loop
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import solve_lm
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp35_cusp import fold_seeds


def rot_z_complex(angle: complex) -> np.ndarray:
    """Rotation about z, complex-safe (no float() casts)."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=complex)


def fk_complex(links: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    """Forward kinematics from link transforms, complex joints."""
    pose = np.eye(4, dtype=complex)
    for index, link in enumerate(links):
        pose = pose @ link @ rot_z_complex(q[index])
    return pose


def residual12(links: list[np.ndarray], q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Algebraic 12-component pose residual (position + rotation matrix entries)."""
    current = fk_complex(links, q)
    return np.concatenate([current[:3, 3] - target[:3, 3],
                           (current[:3, :3] - target[:3, :3]).ravel()])


def newton_complex(links: list[np.ndarray], q: np.ndarray, target: np.ndarray,
                   *, iterations: int = 30, tolerance: float = 1e-12,
                   h: float = 1e-7) -> tuple[np.ndarray, float]:
    """Least-squares Newton on the 12-component residual, complex arithmetic."""
    q = np.asarray(q, dtype=complex)
    for _ in range(iterations):
        residual = residual12(links, q, target)
        if float(np.linalg.norm(residual)) < tolerance:
            break
        jacobian = np.zeros((12, len(q)), dtype=complex)
        for column in range(len(q)):
            step = np.zeros(len(q), dtype=complex)
            step[column] = h
            jacobian[:, column] = (residual12(links, q + step, target)
                                   - residual12(links, q - step, target)) / (2 * h)
        q = q + np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
    return q, float(np.linalg.norm(residual12(links, q, target)))


def links_at(base_links: list[np.ndarray], offset: float, s: complex) -> list[np.ndarray]:
    """The homotopy path at a complex parameter value.

    The copies must be *complex*: assigning ``offset * s`` into a real array silently drops
    the imaginary part and collapses the loop onto the real axis (measured, first version).
    """
    links = [link.astype(complex) for link in base_links]
    links[4][1, 3] = offset * s
    return links


def track_circle(base_links, offset: float, target: np.ndarray, q0: np.ndarray,
                 centre: float, radius: float, *, samples: int = 96,
                 orientation: float = 1.0, max_step: float = 0.1,
                 tolerance: float = 1e-10, start_angle: float = 0.0
                 ) -> tuple[np.ndarray, float, float]:
    """Carry one root around the circle ``s = centre + radius * exp(i * theta)``.

    The circle starts at ``start_angle`` -- it must be the angle of the base point whose
    fibre the roots were taken from (for a death fold the pair is real on one side only,
    so the loop starts at the angle of that side, not at angle 0).
    Adaptive: a Newton step larger than ``max_step`` (or a residual above tolerance) means
    the corrector may have hopped between nearby sheets -- halve the angular step instead.
    """
    q = np.asarray(q0, dtype=complex)
    worst = 0.0
    angle_prev = start_angle
    for k in range(1, samples + 1):
        angle = start_angle + orientation * 2.0 * np.pi * k / samples
        span, accepted = angle - angle_prev, False
        for _ in range(8):
            mid = angle_prev + span
            s = centre + radius * np.exp(1j * mid)
            candidate, residual = newton_complex(links_at(base_links, offset, s), q, target)
            step = float(np.linalg.norm(candidate - q))
            if step <= max_step and residual < tolerance:
                q, worst, accepted = candidate, max(worst, residual), True
                angle_prev = mid
                break
            span *= 0.5
        if not accepted:
            return q, worst, float("inf")
    return q, worst, float(np.linalg.norm(q - np.asarray(q0, dtype=complex)))


def roots_near(model, target: np.ndarray, q_c: np.ndarray, circular, *, radius: float,
               seeds: int = 30, span: float = 0.3) -> list[np.ndarray]:
    """Distinct real solutions within ``radius`` of ``q_c`` (perturbation seeds)."""
    rng = np.random.default_rng(9)
    found: list[np.ndarray] = []
    for _ in range(seeds):
        seed = q_c + rng.normal(0.0, span, size=q_c.size)
        q, converged = solve_lm(model, seed, target)
        if not converged or iks.configuration_distance(q, q_c, circular) > radius:
            continue
        if not any(iks.configuration_distance(q, old, circular) < 1e-3 for old in found):
            found.append(q)
    return found


def main() -> int:
    """Run the complex loop around the fold at s* = 0.718354."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=float, default=0.718354,
                        help="fold to encircle (from the documented spectrum)")
    parser.add_argument("--radius", type=float, default=1e-3)
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

    print("refining the fold (same machinery as exp35 phase 1; branch 0 carries this fold)")
    fibre0 = sr0.solve(sr0.robot(), target)
    folds = fold_seeds(path, target, fibre0[:1])
    s_star, q_star = min(folds, key=lambda item: abs(item[0] - args.fold))
    print(f"  fold at s* = {s_star:.6f} (target {args.fold})")

    # the split direction at the fold is the kernel of the Jacobian: the two coalescing
    # roots live at q* +- c sqrt(|s - s*|) . eta, so seed the base fibre along eta

    kernel = np.linalg.svd(homotopy.jacobian_q(path(s_star), q_star))[2][-1]
    kernel = kernel / np.linalg.norm(kernel)

    def targeted_pair(model_at, side_s: float) -> list[np.ndarray]:
        # homotopy.correct (pseudo-inverse Newton) converges from near-singular seeds where
        # the LM census solver does not -- measured: every LM seed failed at 1e-3 from the
        # fold, correct lands both split roots (0.067 / 0.072 rad, the square-root law)
        found: list[np.ndarray] = []
        links_side = path(side_s)
        for scale in (0.02, 0.05, 0.1):
            for sign in (1.0, -1.0):
                q, residual = homotopy.correct(links_side, q_star + sign * scale * kernel,
                                               target, tolerance=1e-12)
                if residual > 1e-9:
                    continue
                if iks.configuration_distance(q, q_star, circular) > 0.25:
                    continue
                if not any(iks.configuration_distance(q, old, circular) < 1e-3
                           for old in found):
                    found.append(q)
        return found

    s0, pair = None, []
    for candidate_s0 in (s_star + args.radius, s_star - args.radius):
        model0 = sr0.robot(path(candidate_s0))
        pair = targeted_pair(model0, candidate_s0)
        if len(pair) != 2:
            pair = roots_near(model0, target, q_star, circular, radius=0.25, seeds=60)
        if len(pair) == 2:
            s0 = candidate_s0
            break
    print(f"  roots within 0.25 rad of the fold at s0 = {s0}: {len(pair)} "
          f"(the pair is real on the {'s > s*' if s0 and s0 > s_star else 's < s*'} side)")
    if len(pair) != 2 or s0 is None:
        print("  expected exactly the square-root pair nearby; aborting")
        return 1
    from study.census import census, make_lm_solver  # noqa: PLC0415

    fibre0_census = census(model0, target, solver=make_lm_solver(model0), seeds=150,
                           rng=np.random.default_rng(4))
    controls = [
        q for q in fibre0_census.solutions
        if all(iks.configuration_distance(q, p, circular) > 0.5 for p in pair)
    ][:2]
    print(f"  control roots (far from the pair): {len(controls)}")

    start_angle = 0.0 if s0 > s_star else float(np.pi)
    for orientation in (1.0, -1.0):
        print(f"\nloop: centre s*={s_star:.6f}, radius {args.radius:g}, "
              f"orientation {'+' if orientation > 0 else '-'} "
              f"(base at angle {start_angle:.2f}, s0={s0:.6f})")
        images = []
        for index, q0 in enumerate(pair + controls):
            q_end, worst, gap = track_circle(base_links, offset, target, q0, s_star,
                                             args.radius, samples=args.samples,
                                             orientation=orientation,
                                             start_angle=start_angle)
            imag = float(np.abs(q_end.imag).max())
            distances = [iks.configuration_distance(q_end.real, other, circular)
                         for other in pair + controls]
            nearest = int(np.argmin(distances))
            if imag > 1e-6 or not np.isfinite(gap):
                # the track left the real fibre: a hop between sheets, not a permutation
                nearest = -1
            images.append(nearest)
            label = "pair" if index < 2 else "control"
            print(f"  root {index} ({label}): -> root {nearest} "
                  f"(distance {distances[nearest] if nearest >= 0 else float('nan'):.1e}, "
                  f"|imag| {imag:.1e}, worst residual {worst:.1e})")
        moved = [i for i, j in enumerate(images) if i != j]
        is_transposition = (
            len(moved) == 2
            and images[moved[0]] == moved[1]
            and images[moved[1]] == moved[0]
        )
        print(f"  permutation {images}: "
              f"{'TRANSPOSITION' if is_transposition else moved or 'identity'}")

    print("\nnegative control: same radius, centre away from every fold")
    centre = 0.85
    for index, q0 in enumerate(pair + controls):
        q_end, worst, _ = track_circle(base_links, offset, target, q0, centre,
                                       args.radius, samples=args.samples)
        distances = [iks.configuration_distance(q_end.real, other, circular)
                     for other in pair + controls]
        print(f"  root {index}: -> root {int(np.argmin(distances))} "
              f"(distance {min(distances):.1e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
