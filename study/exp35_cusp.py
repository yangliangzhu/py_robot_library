#!/usr/bin/env python3
"""exp35: hunt the cusp -- fold curves in (s, u), their meetings, and the encircling loop.

Task N2 of the collaboration (`study/collab/day02-status.md`, Q14): a non-trivial monodromy
permutation needs a loop that *encircles a cusp* of the singular image (Wenger's criterion), and
kimi's prediction says such a loop must show survivor motion *plus* sheet deaths at folds.

Phases:

1. **Fold seeds.**  At exp21's pose (master seed 0, ``q_true = rng.uniform(-2, 2, 6)``) track the
   eight SR0 solutions along ``s: 0 -> 1``, collect local minima of ``sigma_min`` below 1.2e-2 and
   refine each with exp21's augmented Newton.  Sanity: the seven documented folds
   (0.655086 .. 0.961848) must reappear.
2. **Fold curves.**  Free one pose coordinate ``u`` (position component): the augmented system
   ``FK_s(q) = T(u), det J_s(q) = 0`` is 7 equations in 8 unknowns ``(q, s, u)``, i.e. curves.
   Pseudo-arclength continuation (chord Newton) tracks each fold both ways.
3. **Meetings.**  A cusp is where two fold curves meet (Whitney).  Candidates: sample pairs with
   ``(q, s, u)`` all close.  Verification: at the candidate, count distinct roots near ``q_c`` at
   ``u_c +- delta`` -- three on one side, one on the other (A3) -- and fit the split exponent
   (cube root ~ 1/3 vs the fold's 1/2).
4. **The loop.**  An ellipse around the cusp in the ``(s, u)`` plane; carry every real solution of
   the base point around it with the trust-region corrector; report the survivor map and the
   deaths.  Cusp signature: a survivor that *moved*.

Run::

    python3 -m study.exp35_cusp --u-index 0
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.census import solve_lm
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at
from study.exp21_endgame import refine_fold

DOCUMENTED_FOLDS = [0.655086, 0.656878, 0.657375, 0.658119, 0.718354, 0.799244, 0.961848]


def collect_sigma_samples(path, target, q0: np.ndarray, *, grid: int = 801):
    """Chain the corrector along s from s=0, sampling sigma_min; stops when the branch dies."""
    links = path(0.0)
    q, residual = homotopy.correct(links, q0, target, tolerance=1e-11)
    samples = []
    for s in np.linspace(0.0, 1.0, grid)[1:]:
        links = path(s)
        q, residual = homotopy.correct(links, q, target, tolerance=1e-11)
        if residual > 1e-9:
            break
        samples.append((s, q.copy(), homotopy.sigma_min_links(links, q)))
    return samples


def fold_seeds(path, target, fibre0: list[np.ndarray], *, threshold: float = 3e-2,
               stride: int = 3):
    """Refine a fold from every low-clearance sample along all branches; deduplicate.

    Every sample below ``threshold`` (not only local minima, which are fragile on a grid) is
    refined by exp21's augmented Newton; duplicates are removed afterwards -- the same
    collection strategy as exp23 (threshold 0.03).
    """
    folds: list[tuple[float, np.ndarray]] = []
    refined = 0
    for index, q0 in enumerate(fibre0):
        samples = collect_sigma_samples(path, target, q0.q if hasattr(q0, "q") else q0)
        for s, q, sigma in samples[::stride]:
            if sigma > threshold:
                continue
            refined += 1
            s_star, q_star, residual = refine_fold(path, target, q, s)
            if residual > 1e-8 or not (0.0 < s_star < 1.3):
                continue
            if any(abs(s_star - old_s) < 3e-4 for old_s, _ in folds):
                continue
            folds.append((s_star, q_star))
            print(f"    branch {index}: fold at s* = {s_star:.6f} (residual {residual:.1e})")
    folds.sort(key=lambda item: item[0])
    print(f"    ({refined} low-clearance samples refined)")
    return folds


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    """Rotation about a base-frame coordinate axis."""
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(3)
    other = [index for index in range(3) if index != axis]
    matrix[other[0], other[0]] = cosine
    matrix[other[0], other[1]] = -sine
    matrix[other[1], other[0]] = sine
    matrix[other[1], other[1]] = cosine
    return matrix


def target_at(target: np.ndarray, u_index: int, u: float, u_kind: str = "pos") -> np.ndarray:
    """The base pose moved along one slice coordinate: a position component (metres) or a
    base-axis rotation (radians)."""
    moved = target.copy()
    if u_kind == "rot":
        moved[:3, :3] = _axis_rotation(u_index, u) @ target[:3, :3]
    else:
        moved[:3, 3][u_index] = u
    return moved


def make_augmented(path, target: np.ndarray, u_index: int, u_kind: str = "pos"):
    """The 7-vector [se3 error; det J] as a function of y = (q, s, u)."""

    def augmented(y: np.ndarray) -> np.ndarray:
        q, s, u = y[:6], y[6], y[7]
        links = path(s)
        moved = target_at(target, u_index, u, u_kind)
        error = homotopy.se3_error(homotopy.fk_links(links, q), moved)
        determinant = float(np.linalg.det(homotopy.jacobian_q(links, q)))
        return np.concatenate([error, [determinant]])

    return augmented


def _jacobian8(augmented, y: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Central-difference 7x8 Jacobian of the augmented system."""
    jacobian = np.zeros((7, 8))
    for k in range(8):
        step = np.zeros(8)
        step[k] = h
        jacobian[:, k] = (augmented(y + step) - augmented(y - step)) / (2 * h)
    return jacobian


def track_fold_curve(augmented, y0: np.ndarray, *, direction: float, h0: float = 2e-3,
                     max_steps: int = 500, u_range: float = 0.15,
                     s_bounds: tuple[float, float] = (0.3, 1.4)):
    """Pseudo-arclength continuation of the fold curve from ``y0`` in one direction.

    Chord Newton (one 7x8 Jacobian per step) on ``[F(y); nu . (y - y_pred)] = 0``.
    """
    jacobian = _jacobian8(augmented, y0)
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
            rhs = np.concatenate([-value, [0.0]])
            z = z + np.linalg.lstsq(big, rhs, rcond=None)[0]
        if not converged:
            h *= 0.5
            failures += 1
            if h < 2e-5 or failures > 12:
                break
            continue
        new_jacobian = _jacobian8(augmented, z)
        new_tangent = np.linalg.svd(new_jacobian)[2][-1]
        if float(np.dot(new_tangent, tangent)) < 0:
            new_tangent = -new_tangent
        y, jacobian, tangent = z, new_jacobian, new_tangent
        points.append(y.copy())
        h = min(h * 1.3, 0.02) if failures == 0 else h
        if abs(y[7] - y0[7]) > u_range or not (s_bounds[0] < y[6] < s_bounds[1]):
            break
    return np.array(points)


def find_turnarounds(curves: list[np.ndarray], *, window: int = 5,
                     min_excursion: float = 4e-3) -> list[tuple[int, np.ndarray]]:
    """Candidate A3 (cusp) points on a *single* fold curve: the (s, u) projection turns around.

    For a Whitney cusp the fold locus in source space is smooth, but its projection to the
    parameter plane has a turning point -- s reaches an extremum while u passes through
    (the local model is t -> (t^2, t^3)).  Two-curve meetings miss these, so scan each curve
    for a robust sign flip of the s-increment.
    """
    candidates = []
    for ci, points in enumerate(curves):
        if len(points) < 2 * window + 1:
            continue
        s = points[:, 6]
        for i in range(window, len(points) - window):
            before = s[i] - s[i - window]
            after = s[i + window] - s[i]
            if before * after < 0 and abs(before) > min_excursion and abs(after) > min_excursion:
                candidates.append((ci, points[i]))
                break
    return candidates


def find_meetings(curves: list[np.ndarray], *, ds: float = 4e-3, du: float = 4e-3,
                  dq: float = 0.15) -> list[tuple[int, int, np.ndarray]]:
    """Candidate cusp points: two fold curves passing through the same (q, s, u)."""
    meetings = []
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            best = None
            for ya in curves[a]:
                delta = curves[b] - ya
                close = np.where(
                    (np.abs(delta[:, 6]) < ds) & (np.abs(delta[:, 7]) < du))[0]
                if close.size == 0:
                    continue
                distances = np.linalg.norm(delta[close][:, :6], axis=1)
                index = int(np.argmin(distances))
                if distances[index] < dq and (best is None or distances[index] < best[0]):
                    best = (float(distances[index]), curves[b][close[index]])
            if best is not None:
                meetings.append((a, b, best[1]))
    return meetings


def count_roots_near(model, target: np.ndarray, q_c: np.ndarray, circular,
                     *, radius: float = 0.5, seeds: int = 12) -> int:
    """Distinct solutions within ``radius`` of ``q_c``, from small perturbation seeds."""
    rng = np.random.default_rng(5)
    found: list[np.ndarray] = []
    for _ in range(seeds):
        seed = q_c + rng.normal(0.0, 0.3, size=q_c.size)
        q, converged = solve_lm(model, seed, target)
        if not converged:
            continue
        if iks.configuration_distance(q, q_c, circular) > radius:
            continue
        if not any(iks.configuration_distance(q, old, circular) < 1e-3 for old in found):
            found.append(q)
    return len(found)


def split_exponent(path, target: np.ndarray, u_index: int, q_c: np.ndarray, s_c: float,
                   u_c: float, side: float, circular, u_kind: str = "pos") -> float:
    """Fit log(spread of the coalescing cluster) against log(delta) along u at fixed s."""
    gaps, spreads = [], []
    model = sr0.robot(path(s_c))
    for decade in (2.0, 2.5, 3.0, 3.5):
        delta = 10.0 ** (-decade)
        moved = target_at(target, u_index, u_c + side * delta, u_kind)
        rng = np.random.default_rng(7)
        roots: list[np.ndarray] = []
        for _ in range(24):
            seed = q_c + rng.normal(0.0, 0.25, size=q_c.size)
            q, converged = solve_lm(model, seed, moved)
            if converged and iks.configuration_distance(q, q_c, circular) < 0.6:
                if not any(iks.configuration_distance(q, old, circular) < 1e-3 for old in roots):
                    roots.append(q)
        if len(roots) < 2:
            continue
        spread = max(
            iks.configuration_distance(roots[i], roots[j], circular)
            for i in range(len(roots)) for j in range(i + 1, len(roots))
        )
        gaps.append(float(np.log(delta)))
        spreads.append(float(np.log(spread)))
    if len(gaps) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.array(gaps), np.array(spreads), 1)
    return float(slope)


def run_loop(path, target: np.ndarray, u_index: int, cusp: np.ndarray, *,
             radii: tuple[float, float], samples: int = 240, sigma_floor: float = 2e-3,
             u_kind: str = "pos"):
    """Carry every real solution of the loop's base point around the cusp in the (s, u) plane."""
    from study.census import census, make_lm_solver

    s_c, u_c = cusp[6], cusp[7]
    angles = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    points = [(s_c + radii[0] * np.cos(a), u_c + radii[1] * np.sin(a)) for a in angles]
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)

    base_s, base_u = points[0]
    base_model = sr0.robot(path(base_s))
    base_target = target_at(target, u_index, base_u, u_kind)
    fibre = census(base_model, base_target, solver=make_lm_solver(base_model), seeds=200,
                   rng=np.random.default_rng(3))
    # deduplicate on the torus and keep physically admissible representatives, exactly as
    # monodromy.main does -- a raw census can return 2 pi-shifted copies (18 "solutions"
    # for a 16-root fibre in the first version of this loop)
    base_fibre: list[np.ndarray] = []
    for candidate in fibre.solutions:
        if not iks.physically_admissible(base_model, candidate):
            continue
        if any(iks.torus_distance(candidate, kept) < 1e-3 for kept in base_fibre):
            continue
        base_fibre.append(candidate)
    if not base_fibre:
        print("  loop: empty base fibre, aborting")
        return
    print(f"  loop base: s={base_s:.4f} u={base_u:.4f}, {len(base_fibre)} solutions, "
          f"radii ds={radii[0]:.4f} du={radii[1]:.4f}")
    mapping: dict[int, int] = {}
    deaths: dict[int, str] = {}
    for index, q0 in enumerate(base_fibre):
        q, alive = q0.copy(), True
        for k, (s, u) in enumerate(points[1:], start=1):
            before = q.copy()
            q, residual = homotopy.correct(path(s), q, target_at(target, u_index, u, u_kind),
                                           tolerance=1e-11)
            step = float(np.linalg.norm(q - before))
            clearance = homotopy.sigma_min_links(path(s), q) if residual < 1e-8 else 0.0
            if residual > 1e-8 or clearance < sigma_floor or step > 0.5:
                deaths[index] = (f"died at sample {k}/{samples}: residual {residual:.1e}, "
                                 f"clearance {clearance:.1e}, step {step:.1e}")
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


def main() -> int:
    """Run the cusp hunt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--u-index", type=int, default=0, help="freed coordinate axis 0/1/2")
    parser.add_argument("--u-kind", default="pos", choices=["pos", "rot"],
                        help="slice coordinate: position component (m) or base-axis rotation (rad)")
    parser.add_argument("--u-range", type=float, default=0.15)
    parser.add_argument("--phase", default="all", choices=["seeds", "curves", "all"])
    parser.add_argument("--save-curves", default="", help="npz path to persist the fold curves")
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    path = path_at(offset)
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    u0 = float(target[:3, 3][args.u_index]) if args.u_kind == "pos" else 0.0
    print(f"exp21's pose reproduced; freed coordinate {args.u_kind}{args.u_index} = {u0:.4f}")

    print("phase 1: fold seeds (tracking 8 branches, refining low-clearance minima)")
    fibre0 = sr0.solve(sr0.robot(), target)
    folds = fold_seeds(path, target, fibre0)
    matched = [min(abs(s - doc) for doc in DOCUMENTED_FOLDS) for s, _ in folds]
    print(f"  {len(folds)} folds: {[f'{s:.6f}' for s, _ in folds]}")
    print(f"  distance to the documented 7-fold spectrum: "
          f"{[f'{d:.1e}' for d in matched]}")
    if args.phase == "seeds":
        return 0

    print(f"phase 2: fold curves along {args.u_kind}{args.u_index} (+/- {args.u_range})")
    augmented = make_augmented(path, target, args.u_index, args.u_kind)
    curves: list[np.ndarray] = []
    for k, (s_star, q_star) in enumerate(folds):
        y0 = np.concatenate([q_star, [s_star, u0]])
        for direction in (1.0, -1.0):
            points = track_fold_curve(augmented, y0, direction=direction,
                                      u_range=args.u_range)
            curves.append(points)
            print(f"  fold {k} (s*={s_star:.6f}) dir {direction:+.0f}: "
                  f"{len(points)} points, u {points[-1][7]:.4f}, s {points[-1][6]:.4f}")
    meetings = find_meetings(curves)
    # drop the trivial meetings: every fold's +dir/-dir branches trivially share their seed
    # point at u == u0; a genuine cusp meeting is away from the seeds
    meetings = [meeting for meeting in meetings if abs(meeting[2][7] - u0) > 1e-3]
    turnarounds = find_turnarounds(curves)
    print(f"phase 3: {len(meetings)} candidate meeting(s) (seed meetings excluded), "
          f"{len(turnarounds)} single-curve turnaround(s)")
    verified: list[np.ndarray] = []
    candidates = [(f"curves {a} x {b}", point) for a, b, point in meetings]
    candidates += [(f"turnaround on curve {ci}", point) for ci, point in turnarounds]
    for label, point in candidates:
        print(f"  {label}: candidate at s={point[6]:.6f} u={point[7]:.6f}")
        counts = []
        for side in (1.0, -1.0):
            moved = target_at(target, args.u_index, point[7] + side * 2e-3, args.u_kind)
            count = count_roots_near(sr0.robot(path(point[6])), moved, point[:6], circular)
            counts.append(count)
            print(f"    roots near q_c at u{'+' if side > 0 else '-'}2e-3: {count}")
        for side in (1.0, -1.0):
            exponent = split_exponent(path, target, args.u_index, point[:6], point[6],
                                      point[7], side, circular, args.u_kind)
            print(f"    split exponent on u{'+' if side > 0 else '-'} side: {exponent:.3f}")
        if max(counts) >= 3:
            print("    -> VERIFIED cusp (three roots coalesce on one side)")
            verified.append(point)
        else:
            print("    -> not a cusp (fold: at most two roots coalesce)")
    if args.save_curves:
        np.savez(args.save_curves, target=target, u_index=args.u_index, u_kind=args.u_kind,
                 u0=u0, fold_s=np.array([s for s, _ in folds]),
                 **{f"curve_{i}": curve for i, curve in enumerate(curves)})
        print(f"  curves saved to {args.save_curves}")
    if verified:
        print("phase 4: encircling loop around the first verified cusp")
        run_loop(path, target, args.u_index, verified[0], radii=(5e-3, 5e-3),
                 u_kind=args.u_kind)
    else:
        print("phase 4: no verified cusp in this slice -- no loop to build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
