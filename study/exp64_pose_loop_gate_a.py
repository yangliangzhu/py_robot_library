#!/usr/bin/env python3
"""exp64: Gate A -- one reliable complete complex-loop permutation of the **fixed** SR5.

Q27 / T4.  The earlier non-solvability evidence (exp37/38/40) moves the *mechanical* parameter
(``links[4][1,3] = offset * s``) with the target pose fixed, so it proves the **family** is
non-solvable uniformly in the parameter.  Transferring that to the fixed arm (parameters fixed,
pose varying) needs the monodromy measured by closed loops in **pose space**, which is exactly
what this module does for Gate A only (GPT ``GPT_NEXT_GATE_20261017.md`` section 3):

* the arm is the fixed SR5 (``links[4][1,3] = 0.136 m``), never swept;
* the pose is complexified along one legal coordinate (GPT C1: a straight line ``tau = tau0 + u d``
  with only the translation part of ``d`` non-zero -- here ``d = [0,0,-1]``, orientation fixed);
* a simple source fold of the pose-space family is found by a real sweep and refined with
  the augmented branch-point system ``[err6; det J] = 0`` over ``(q, u)`` (GPT C2);
* one closed complex loop around it (approaching via the upper half-plane so the complex roots'
  real-axis birth folds are avoided) is tracked with the full 16-root fibre and identity
  bookkeeping (linear prediction, jump rejection, endpoint-injective matching, unmatched ==
  failure state; merge detection lives in the sweep via ``min_pair_gap``, as in exp58/59);
* Gate A's acceptance (GPT section 3, first list): the loop, its reverse, an empty loop, and a
  step-refinement re-run must agree; each closed loop yields one complete permutation; shared
  source folds are tracked together (the permutation is read from the *whole* fibre, never from
  one pair); no unmatched root is silently treated as a fixed point.

Run::

    python3 -m study.exp64_pose_loop_gate_a

Exit code 0 iff every Gate-A self-check passes.
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
from study.exp37_complex_loop import fk_complex, newton_complex, residual12
from study.exp38_monodromy_group import (
    DEATH_FOLDS,
    make_s_path,
    pair_at_fold,
    torus_distance_complex,
    track_path,
)
from study.exp57_sr0_monodromy import refine_branch_pose

#: Which folds of the s-family contribute the three complex (non-real) pairs at s = 1.
_COMPLEX_PAIR_FOLDS = DEATH_FOLDS


def fingerprint(links: list[np.ndarray]) -> np.ndarray:
    """The 18 translation components that pin down the mechanical parameters.

    Args:
        links: The six 4x4 link transforms.

    Returns:
        A flat length-18 vector of the translation components, in link-major order.
    """
    return np.array([link[row, 3] for link in links for row in range(3)])


def fixed_sr5_links() -> list[np.ndarray]:
    """The fixed SR5 link transforms (link 5's y offset 0.136 m), never swept."""
    return link_transforms()


def base_pose(seed: int, model):
    """A base pose ``T0`` and the configuration that produced it.

    Args:
        seed: Seed of the random configuration.
        model: The fixed SR5 model.

    Returns:
        ``(target0, q_true)``.
    """
    rng = np.random.default_rng(seed)
    q_true = rng.uniform(-2.0, 2.0, model.num_dof)
    return np.asarray(model.fk(q_true), dtype=float), q_true


def build_full_fibre(
    links: list[np.ndarray],
    target: np.ndarray,
    *,
    seeds: int = 400,
    rng_seed: int = 2,
) -> list[np.ndarray]:
    """The full 16-root fibre of the fixed SR5 at ``target`` (exp38's deterministic construction).

    Ten real solutions from a torus-deduplicated, physically-admissible census, plus the six
    complex (non-real) ones obtained by seeding each of the three death folds of the s-family and
    tracking the coalescing pair back to ``s = 1``.  The parameter continuation is used only to
    *enumerate* the fibre; every root is polished against the fixed arm's equation below.

    Args:
        links: The fixed SR5 link transforms.
        target: The base pose.
        seeds: Census seeds for the real part.
        rng_seed: Seed of the census generator.

    Returns:
        The 16 roots as complex arrays (real roots first, then the three complex pairs).
    """
    model = sr0.robot(links)
    circular = iks.circular_joints(model)
    offset = float(links[4][1, 3])
    path = path_at(offset)

    real_fibre: list[np.ndarray] = []
    census1 = census(model, target, solver=make_lm_solver(model), seeds=seeds,
                     rng=np.random.default_rng(rng_seed))
    for candidate in census1.solutions:
        if not iks.physically_admissible(model, candidate):
            continue
        if any(iks.torus_distance(candidate, kept) < 1e-3 for kept in real_fibre):
            continue
        real_fibre.append(np.asarray(candidate, dtype=float))
    fibre: list[np.ndarray] = [np.asarray(q, dtype=complex) for q in real_fibre]

    fibre0 = sr0.solve(sr0.robot(), target)
    folds = fold_seeds(path, target, fibre0)
    real_fold_s = [s for s, _ in folds]
    for s_star, q_star in folds:
        if not any(abs(s_star - death) < 1e-4 for death in _COMPLEX_PAIR_FOLDS):
            continue
        s0, pair = pair_at_fold(links, offset, target, path, q_star, s_star, circular)
        if not pair:
            continue
        for q0 in pair:
            q_end, _, ok = track_path(links, offset, target, q0, make_s_path(s0, 1.0, real_fold_s))
            if ok and not any(torus_distance_complex(q_end, old) < 1e-2 for old in fibre):
                fibre.append(q_end)
    return fibre


def target_at(target0: np.ndarray, direction: np.ndarray, u: complex) -> np.ndarray:
    """The base pose displaced by ``u`` along ``direction`` (a legal pose coordinate).

    GPT C1: ``tau(u) = tau0 + u * d`` with ``d``'s translation part non-zero and its rotation part
    zero; ``u`` is complex so a loop in ``u`` is a loop in the complexified pose space.  The
    rotation block of ``target0`` is kept exactly.

    Args:
        target0: The base 4x4 pose.
        direction: Unit translation direction (length 3).
        u: Complex displacement along ``direction``, in metres.

    Returns:
        The complex 4x4 pose.
    """
    moved = target0.astype(complex)
    moved[:3, 3] = target0[:3, 3].astype(complex) + u * np.asarray(direction, dtype=complex)
    return moved


def min_pair_gap(qs: list[np.ndarray], alive: list[int]) -> tuple[float, int, int]:
    """Smallest complex torus distance among two tracked roots, and which pair attains it.

    Args:
        qs: All tracked configurations (complex).
        alive: Indices still being tracked.

    Returns:
        ``(gap, i, j)``.
    """
    best = (float("inf"), -1, -1)
    for a, i in enumerate(alive):
        for j in alive[a + 1:]:
            distance = torus_distance_complex(qs[i], qs[j])
            if distance < best[0]:
                best = (distance, i, j)
    return best


def walk_fibre(links: list[np.ndarray], roots: list[np.ndarray], poses: list[np.ndarray], *,
               jump_tolerance: float = 0.5, tolerance: float = 1e-9, iterations: int = 40,
               ) -> tuple[list[np.ndarray], list[str | None]]:
    """Track every root through a pose list with linear prediction and jump rejection.

    The loop discipline from exp58/exp59: predict ``2 q_cur - q_prev``, correct from the
    prediction, and refuse a correction further from the prediction than ``jump_tolerance`` (a
    basin jump).  A root that fails keeps a failure string; it is never silently carried or
    silently treated as fixed.  Merge detection is *not* applied per step here -- the approach
    leg legitimately crosses the birth folds of the complex roots, where a conjugate pair
    coalesces by construction -- and is instead enforced at the endpoint by
    :func:`read_permutation`'s injectivity check plus the loop/reverse/empty agreement.

    Args:
        links: The fixed SR5 link transforms (complex).
        roots: The root configurations at ``poses[0]`` (complex).
        poses: The pose sequence (complex 4x4 arrays), in order.
        jump_tolerance: Largest accepted distance from the prediction, radians.
        tolerance: Residual the corrector must reach for a step to count.
        iterations: Newton iterations per correction.

    Returns:
        ``(q_end, status)``: the final configuration per root and the per-root failure string
        (``None`` when the root survived every step).
    """
    n = len(roots)
    q_prev = [np.asarray(root, dtype=complex).copy() for root in roots]
    q_cur: list[np.ndarray] = []
    status: list[str | None] = [None] * n
    for index in range(n):
        q_new, residual = newton_complex(links, q_prev[index], poses[0], iterations=iterations,
                                         tolerance=tolerance)
        if residual > tolerance or not np.all(np.isfinite(q_new)):
            status[index] = f"start residual {residual:.1e}"
            q_cur.append(q_prev[index])
        else:
            q_cur.append(q_new)

    for pose_index, pose in enumerate(poses[1:], start=1):
        alive = [i for i in range(n) if status[i] is None]
        predictions = [2.0 * q_cur[index] - q_prev[index] for index in range(n)]
        q_new: list[np.ndarray] = [q_cur[index].copy() for index in range(n)]
        for index in alive:
            candidate, residual = newton_complex(links, predictions[index], pose,
                                                 iterations=iterations, tolerance=tolerance)
            if residual > tolerance or not np.all(np.isfinite(candidate)):
                status[index] = f"died at pose {pose_index} (residual {residual:.1e})"
                continue
            if float(np.linalg.norm(candidate - predictions[index])) > jump_tolerance:
                status[index] = f"jumped at pose {pose_index}"
                continue
            q_new[index] = candidate
        q_prev, q_cur = q_cur, q_new
    return q_cur, status


def read_permutation(q_end: list[np.ndarray], status: list[str | None],
                     fibre: list[np.ndarray], *, match_tol: float = 1e-2) -> tuple[list[int], bool]:
    """Match each surviving root's endpoint to the base fibre; unmatched == failure state.

    Args:
        q_end: Final configuration per root.
        status: Per-root failure string (``None`` = survived).
        fibre: The base fibre to match against (complex).
        match_tol: Largest torus distance that still counts as "the same root".

    Returns:
        ``(permutation, ok)``: ``permutation[i]`` is the index of the base root that root ``i``
        landed on, ``-1`` for a failure state; ``ok`` is True only when every root matched and the
        map is injective (a genuine permutation of the whole fibre).
    """
    permutation: list[int] = []
    for index in range(len(fibre)):
        if status[index] is not None:
            permutation.append(-1)
            continue
        distances = [torus_distance_complex(q_end[index], member) for member in fibre]
        nearest = int(np.argmin(distances))
        permutation.append(nearest if distances[nearest] < match_tol else -1)
    images = [image for image in permutation if image >= 0]
    injective = len(set(images)) == len(images) and len(images) == len(fibre)
    return permutation, injective


def cycle_type(images: list[int]) -> str:
    """Cycle type of a permutation in image notation, e.g. ``2+1+1...`` (``-`` for a failure)."""
    if any(image < 0 for image in images):
        return "-"
    seen = [False] * len(images)
    lengths: list[int] = []
    for start in range(len(images)):
        if seen[start]:
            continue
        length, node = 0, start
        while not seen[node]:
            seen[node] = True
            node = images[node]
            length += 1
        lengths.append(length)
    return "+".join(str(value) for value in sorted(lengths, reverse=True))


def inverse(images: list[int]) -> list[int]:
    """Inverse of a permutation given as images (``-1`` entries preserved as failures)."""
    out = [-1] * len(images)
    for index, image in enumerate(images):
        if image >= 0:
            out[image] = index
    return out


def compose(first: list[int], second: list[int]) -> list[int]:
    """Composition ``first . second`` (apply ``second``, then ``first``)."""
    return [first[second[index]] if second[index] >= 0 else -1 for index in range(len(first))]


def _complex_leg(u0: float, u1: float, offset: float, samples: int) -> list[complex]:
    """A complex path ``u0 -> u1`` arcing into the upper half-plane by ``offset`` (a trapezoid).

    This keeps the path off the real axis, where the complex roots' birth folds live, so a
    conjugate pair stays complex along the whole leg instead of coalescing.  ``offset`` must be
    positive and smaller than the distance to any off-axis branch point (the real-axis folds are
    exactly at ``Im = 0``).

    Args:
        u0: Start ``u`` (real).
        u1: End ``u`` (real).
        offset: Height of the arc above the real axis.
        samples: Total samples along the trapezoid.

    Returns:
        The ``u``-values of the path (first ``u0``, last ``u1``).
    """
    n = max(12, samples)
    third = n // 3
    points: list[complex] = []
    points += [complex(u0) + 1j * offset * t for t in np.linspace(0.0, 1.0, third)]
    points += [complex(u0) + 1j * offset + (u1 - u0) * t
               for t in np.linspace(0.0, 1.0, third)[1:]]
    points += [complex(u1) + 1j * offset * (1.0 - t)
               for t in np.linspace(0.0, 1.0, third)[1:]]
    points.append(complex(u1))
    return points


def build_loop(u0: float, u_star: complex, radius: float, *, samples: int,
               orientation: float = 1.0, arc: float = 0.05) -> list[complex]:
    """The closed loop in ``u``: out from ``u0`` to the circle, around ``u_star``, and back.

    The circle starts on the real side ``u_star - radius`` (inside a death fold, where the pair
    is real).  The approach leg arcs into the upper half-plane by ``arc`` so it never crosses the
    real-axis birth folds of the complex roots; the return leg retraces the same arc, so the two
    legs' net monodromy cancels and only the circle contributes.

    Args:
        u0: The base pose's ``u`` coordinate (0 for this construction).
        u_star: The branch point's ``u`` coordinate.
        radius: Circle radius around ``u_star``.
        samples: Angular samples on the circle.
        orientation: +1 counter-clockwise, -1 clockwise.
        arc: Height of the approach arc above the real axis, metres.

    Returns:
        The ``u``-values of the closed loop (first and last are ``u0``).
    """
    base_u = complex(u_star) - radius
    leg_out = _complex_leg(u0, base_u.real, arc, 3 * samples)
    start_angle = float(np.pi)
    thetas = [start_angle + orientation * 2.0 * np.pi * k / samples for k in range(samples + 1)]
    circle = [complex(u_star) + radius * np.exp(1j * theta) for theta in thetas[1:]]
    leg_back = _complex_leg(base_u.real, u0, arc, 3 * samples)[1:]
    return leg_out + circle + leg_back


def build_empty_loop(u0: float, radius: float, *, samples: int) -> list[complex]:
    """A closed loop based at ``u0`` that encloses no branch point (the negative control).

    Args:
        u0: The base pose's ``u`` coordinate.
        radius: Circle radius around ``u0`` (must be smaller than the distance to any fold).
        samples: Angular samples on the circle.

    Returns:
        The ``u``-values of the closed loop.
    """
    base = u0 + radius
    leg_out = list(make_s_path(u0, base, []))
    circle = [u0 + radius * np.exp(1j * theta)
              for theta in np.linspace(0.0, 2.0 * np.pi, samples + 1)[1:]]
    leg_back = list(make_s_path(base, u0, []))[1:]
    return leg_out + circle + leg_back


def run_loop(links: list[np.ndarray], fibre: list[np.ndarray], u_points: list[complex],
             target_at_fn, *, label: str, jump_tolerance: float,
             ) -> tuple[list[int], bool, list[str | None], list[np.ndarray]]:
    """Carry the full fibre around one closed loop and read the permutation.

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The base fibre.
        u_points: The ``u``-values of the loop.
        target_at_fn: ``u -> complex pose``.
        label: Human-readable loop name.
        jump_tolerance: Largest accepted prediction distance.

    Returns:
        ``(permutation, ok, status, q_end)``.
    """
    poses = [target_at_fn(u) for u in u_points]
    q_end, status = walk_fibre(links, fibre, poses, jump_tolerance=jump_tolerance)
    permutation, ok = read_permutation(q_end, status, fibre)
    _print_loop_result(label, links, q_end, permutation, ok, status, poses[-1], fibre)
    return permutation, ok, status, q_end


def _print_loop_result(label: str, links: list[np.ndarray], q_end: list[np.ndarray],
                       permutation: list[int], ok: bool, status: list[str | None],
                       final_pose: np.ndarray, fibre: list[np.ndarray]) -> None:
    """Print one loop's permutation, cycle type, per-root residual/gap, and failure states.

    Args:
        label: Human-readable loop name.
        links: The fixed SR5 link transforms (complex), for the endpoint residual.
        q_end: Final configuration per root.
        permutation: The permutation (``-1`` for failure states).
        ok: Whether the permutation is complete and injective.
        status: Per-root failure string.
        final_pose: The loop's closing pose.
        fibre: The base fibre, for the return gap.
    """
    moved = [(i, image) for i, image in enumerate(permutation) if image >= 0 and image != i]
    failures = [(i, status[i]) for i in range(len(status)) if status[i] is not None]
    print(f"  {label}: permutation {permutation}")
    print(f"    cycle type {cycle_type(permutation)} | complete {ok} | moved {moved} "
          f"| failures {len(failures)}")
    for index in range(len(q_end)):
        residual = float(np.linalg.norm(residual12(links, q_end[index], final_pose)))
        matched = permutation[index]
        gap = (torus_distance_complex(q_end[index], fibre[matched])
               if 0 <= matched < len(fibre) else float("nan"))
        state = "FAILED" if status[index] is not None else f"-> {matched}"
        print(f"    root {index:2d}: endpoint residual {residual:.1e}, gap to image {gap:.1e}, "
              f"{state}")
    for index, reason in failures:
        print(f"    FAILURE root {index}: {reason}")


def main() -> int:
    """Run Gate A: find the fold, loop it, and run the agreement self-checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="base-pose seed")
    parser.add_argument("--census-seeds", type=int, default=400, help="census seeds for the fibre")
    parser.add_argument("--direction", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                        help="unit translation direction for the pose slice (legal C1 coordinate)")
    parser.add_argument("--span", type=float, default=1.5, help="sweep length, metres")
    parser.add_argument("--dip", type=float, default=5e-2,
                        help="pair gap below which the sweep stops (a fold in reach)")
    parser.add_argument("--radius", type=float, default=5e-3, help="loop radius around the fold")
    parser.add_argument("--arc", type=float, default=0.05,
                        help="height of the approach arc above the real axis, metres")
    parser.add_argument("--samples", type=int, default=96, help="circle samples")
    parser.add_argument("--refined-samples", type=int, default=192,
                        help="circle samples for the step-refinement re-run")
    parser.add_argument("--jump-tolerance", type=float, default=0.5)
    parser.add_argument("--out", default="study/exp64_gate_a.npz",
                        help="npz path for the raw data (root table, trajectory, residuals)")
    args = parser.parse_args()

    links = fixed_sr5_links()
    model = sr0.robot(links)
    print("exp64: Gate A -- one fixed-SR5 pose-space complex loop and its agreement checks\n")

    print("phase 0: mechanical-parameter fingerprint (the arm is fixed, never swept)")
    print(f"  18 translation components: {np.array2string(fingerprint(links), precision=6)}")
    print(f"  link 5 y offset (the SR0<->SR5 parameter): {links[4][1, 3]:.6f} m")
    direction = np.asarray(args.direction, dtype=float)
    direction = direction / float(np.linalg.norm(direction))
    print(f"  target-path formula: p(u) = p0 + u * d, d = "
          f"{np.array2string(direction, precision=6)} (unit), orientation fixed, u complex")

    target0, q_true = base_pose(args.seed, model)
    position = target0[:3, 3]
    print(f"  base position p0 = {np.array2string(position, precision=6)} m, "
          f"|p0| = {float(np.linalg.norm(position)):.4f} m")
    print(f"  sigma_min at the producing configuration: {iks.sigma_min(model, q_true):.3e}\n")

    print("phase 1: full 16-root fibre at the base pose (exp38's deterministic construction)")
    fibre = build_full_fibre(links, target0, seeds=args.census_seeds)
    real_count = sum(1 for q in fibre if float(np.abs(q.imag).max()) < 1e-9)
    print(f"  {len(fibre)} roots: {real_count} real + {len(fibre) - real_count} complex")
    links_c = [link.astype(complex) for link in links]
    for index, q in enumerate(fibre):
        residual = float(np.linalg.norm(
            np.concatenate([fk_complex(links_c, q)[:3, 3] - target0[:3, 3],
                            (fk_complex(links_c, q)[:3, :3] - target0[:3, :3]).ravel()])))
        print(f"    root {index:2d}: residual {residual:.1e} | imag "
              f"{float(np.abs(q.imag).max()):.1e}")

    print("\nphase 2: sweep along d for a pose-space fold (real pair coalescence)")
    fold_info = _radial_sweep(links_c, fibre, target0, direction, span=args.span, dip=args.dip)
    if fold_info is None:
        print("  no fold found in the sweep -- nothing to loop")
        return 1
    u_gap, (q_a, q_b) = fold_info
    print(f"  closest real-pair approach at u = {u_gap:+.6f} m "
          f"(pair gap there {torus_distance_complex(q_a, q_b):.3e})")

    def target_at_fn(u: complex) -> np.ndarray:
        return target_at(target0, direction, u)

    print("phase 3: refine the fold with the augmented system [err6; det J] = 0 over (q, u)")
    # The two roots approach on different 2 pi sheets; align them before averaging so the seed is
    # a genuine midpoint of the coalescing pair, not the average of two 2 pi-shifted copies.
    shift = 2.0 * np.pi * np.round((q_a.real - q_b.real) / (2.0 * np.pi))
    q_b_aligned = q_b + shift
    seed_q = 0.5 * (q_a + q_b_aligned)
    y, residual = refine_branch_pose(links_c, target_at_fn, seed_q, u_gap)
    u_star = y[6]
    print(f"  refined fold: u* = {u_star.real:+.6f}{u_star.imag:+.1e}i "
          f"(augmented residual {residual:.1e})")
    if residual > 1e-8 or abs(u_star.imag) > 1e-3:
        print("  refinement failed or the branch point is not real -- no loop")
        return 1

    print("\nphase 4: one closed complex loop + its reverse, an empty loop, and a re-run")
    loop = build_loop(0.0, u_star, args.radius, samples=args.samples, orientation=1.0,
                      arc=args.arc)
    reverse_loop = list(reversed(loop))
    empty_loop = build_empty_loop(0.0, min(args.radius, 0.01 * abs(u_star.real)),
                                  samples=args.samples)
    refined_loop = build_loop(0.0, u_star, args.radius, samples=args.refined_samples,
                              orientation=1.0, arc=args.arc)
    print(f"  loop length {len(loop)}, reverse {len(reverse_loop)}, empty {len(empty_loop)}, "
          f"refined {len(refined_loop)}")
    print("  parameter trajectory (u, real+imag) -- first/last/circle samples:")
    for index in (0, 1, len(loop) // 2, len(loop) - 2, len(loop) - 1):
        u = loop[index]
        print(f"    u[{index}] = ({u.real:+.6f}, {u.imag:+.6f})")

    perm, ok_loop, _, q_end = run_loop(
        links_c, fibre, loop, target_at_fn, label="loop (+1)", jump_tolerance=args.jump_tolerance)
    perm_rev, ok_rev, _, _ = run_loop(
        links_c, fibre, reverse_loop, target_at_fn, label="reverse loop",
        jump_tolerance=args.jump_tolerance)
    perm_empty, ok_empty, _, _ = run_loop(
        links_c, fibre, empty_loop, target_at_fn, label="empty loop",
        jump_tolerance=args.jump_tolerance)
    perm_refined, ok_refined, _, _ = run_loop(
        links_c, fibre, refined_loop, target_at_fn, label="step-refinement re-run",
        jump_tolerance=args.jump_tolerance)

    print("\nphase 5: Gate-A agreement self-checks")
    identity = list(range(len(fibre)))
    checks: list[tuple[str, bool, str]] = []
    checks.append(("loop is a complete permutation", ok_loop, "all 16 matched, injective"))
    checks.append(("reverse loop is a complete permutation", ok_rev, "all 16 matched, injective"))
    checks.append(("empty loop is a complete permutation", ok_empty, "all 16 matched, injective"))
    checks.append(("refined re-run is a complete permutation", ok_refined,
                   "all 16 matched, injective"))
    if ok_loop and ok_rev:
        checks.append(("reverse == loop^-1", perm_rev == inverse(perm),
                       f"{perm_rev} vs {inverse(perm)}"))
    if ok_loop and ok_empty:
        checks.append(("empty loop is identity", perm_empty == identity, f"{perm_empty}"))
    if ok_loop and ok_refined:
        checks.append(("step-refinement agrees with loop", perm_refined == perm,
                       f"{perm_refined} vs {perm}"))
    if ok_loop and ok_rev:
        composed = compose(perm, perm_rev)
        checks.append(("loop . reverse == identity", composed == identity, f"{composed}"))
    all_ok = True
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
        all_ok = all_ok and passed

    print(f"\n  complete permutation: {perm} (cycle type {cycle_type(perm)})")
    _dump_raw_data(
        args.out, links_c, target0, direction, u_star, args.radius, fibre, loop, q_end, perm,
        perm_rev, perm_empty, perm_refined, all_ok,
    )
    if not all_ok:
        print("FAIL: a Gate-A self-check did not pass; see the failure states above")
        return 1
    print("PASS: loop, reverse, empty, and step-refinement all agree; "
          "one complete permutation delivered")
    return 0


def _dump_raw_data(path: str, links: list[np.ndarray], target0: np.ndarray, direction: np.ndarray,
                   u_star: complex, radius: float, fibre: list[np.ndarray], loop: list[complex],
                   q_end: list[np.ndarray], perm: list[int], perm_rev: list[int],
                   perm_empty: list[int], perm_refined: list[int], all_ok: bool) -> None:
    """Write the raw Gate-A data (root table, trajectory, permutation, residuals) to an npz.

    This is the machine-readable record the reviewer checks: the complete complex root table, the
    actual complex parameter path, the permutation, and the per-root endpoint residual and gap to
    its matched image -- complex values stored as separate real and imaginary arrays, never
    truncated to reals.

    Args:
        path: Output ``.npz`` path.
        links: The fixed SR5 link transforms (complex).
        target0: Base 4x4 pose.
        direction: Unit translation direction.
        u_star: The branch point's ``u`` coordinate.
        radius: Loop radius.
        fibre: The 16 base roots (complex).
        loop: The parameter trajectory ``u``-values (complex).
        q_end: Final configuration per root after the loop.
        perm: The loop's permutation.
        perm_rev, perm_empty, perm_refined: The other three loops' permutations.
        all_ok: Whether every self-check passed.
    """
    fibre_arr = np.asarray(fibre, dtype=complex)
    loop_u = np.asarray([(float(u.real), float(u.imag)) for u in loop], dtype=float)
    q_end_arr = np.asarray(q_end, dtype=complex)
    final_pose = target_at(target0, direction, loop[-1])
    residual = np.array([float(np.linalg.norm(residual12(links, q_end_arr[i], final_pose)))
                         for i in range(len(fibre))])
    gap = np.array([
        torus_distance_complex(q_end_arr[i], fibre_arr[perm[i]]) if perm[i] >= 0 else float("nan")
        for i in range(len(fibre))
    ])
    np.savez(
        path,
        target0=np.asarray(target0, dtype=complex),
        direction=np.asarray(direction, dtype=float),
        u_star=complex(u_star),
        radius=float(radius),
        fibre_real=fibre_arr.real,
        fibre_imag=fibre_arr.imag,
        loop_u_real=loop_u[:, 0],
        loop_u_imag=loop_u[:, 1],
        q_end_real=q_end_arr.real,
        q_end_imag=q_end_arr.imag,
        permutation=np.asarray(perm, dtype=int),
        permutation_reverse=np.asarray(perm_rev, dtype=int),
        permutation_empty=np.asarray(perm_empty, dtype=int),
        permutation_refined=np.asarray(perm_refined, dtype=int),
        endpoint_residual=residual,
        gap_to_image=gap,
        self_checks_passed=bool(all_ok),
    )
    print(f"  raw data written to {path}")


def _radial_sweep(links: list[np.ndarray], fibre: list[np.ndarray], target0: np.ndarray,
                  direction: np.ndarray, *, span: float, dip: float,
                  ) -> tuple[float, tuple[np.ndarray, np.ndarray]] | None:
    """Walk the real roots along ``direction`` and find the first real-pair coalescence.

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The base fibre (complex).
        target0: The base pose.
        direction: Unit translation direction.
        span: Sweep length in metres.
        dip: Pair gap below which the sweep stops.

    Returns:
        ``(u_gap, (q_a, q_b))``: the ``u`` value and the two coalescing configurations there,
        or ``None`` when no pair came within ``dip`` over the whole sweep.
    """
    real_indices = [i for i, q in enumerate(fibre) if float(np.abs(q.imag).max()) < 1e-9]
    qs = [np.asarray(fibre[i], dtype=complex).copy() for i in real_indices]
    previous = [q.copy() for q in qs]
    alive = list(range(len(real_indices)))
    step = 2e-3
    u = 0.0
    best = (float("inf"), 0.0, np.zeros(6, dtype=complex), np.zeros(6, dtype=complex))
    while u < span and len(alive) >= 2:
        trial = u + step
        pose = target_at(target0, direction, complex(trial))
        updated: dict[int, np.ndarray] = {}
        faults = 0
        for index in alive:
            prediction = 2.0 * qs[index] - previous[index]
            q_new, residual = newton_complex(links, prediction, pose)
            if (residual > 1e-10 or not np.all(np.isfinite(q_new))
                    or float(np.linalg.norm(q_new - prediction)) > 0.5):
                faults += 1
                continue
            updated[index] = q_new
        if faults:
            step *= 0.5
            if step < 1e-7:
                break
            continue
        for index, q_new in updated.items():
            previous[index] = qs[index]
            qs[index] = q_new
        u = trial
        gap, first, second = min_pair_gap(qs, alive)
        if gap < best[0]:
            best = (gap, u, qs[first].copy(), qs[second].copy())
        if gap < dip:
            return u, (qs[first].copy(), qs[second].copy())
        step = min(step * 1.5, 4e-3)
    if best[0] < float("inf"):
        return best[1], (best[2], best[3])
    return None


if __name__ == "__main__":
    raise SystemExit(main())
