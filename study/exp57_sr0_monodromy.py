#!/usr/bin/env python3
"""exp57: Q8 -- the monodromy group of SR0 (the Pieper-solvable neighbour), measured.

The paper's claim is that the Galois group sees the *shape* of the solution tree.  SR0's
closed form is three independent binary choices (shoulder/elbow/wrist, each a quadratic), so
its monodromy group must be an elementary abelian 2-group: every generator a product of
disjoint transpositions, orbits of size at most two, the whole group abelian of order 2^k.
This experiment measures it, the same machinery as exp37/exp38/exp40 but with the loop in a
workspace coordinate instead of the DH parameter:

1. the eight analytic solutions of the base pose (deterministic, no census);
2. branch points along one position coordinate ``u`` -- real tracking to spot pair approaches,
   then a clip-free complex Newton refinement of ``(q, u)``;
3. a complex circle around each branch point, all eight roots carried around, permutation read;
4. the group reading: cycle types, the generated group's order, abelianness.

Prediction (falsifiable): every generator is an involution and the group is abelian.
A single 3-cycle would break the closed form's story.

Run::

    python3 -m study.exp57_sr0_monodromy
"""

from __future__ import annotations

import argparse

import numpy as np

from study import sr0
from study.exp11_dh_continuation import link_transforms
from study.exp37_complex_loop import newton_complex
from study.exp38_monodromy_group import torus_distance_complex
from study.exp40_complex_branches import err6_complex, make_s_path_complex


def sr0_links() -> list[np.ndarray]:
    """The SR0's link transforms: the SR5's with the link-5 y offset zeroed."""
    links = [link.copy() for link in link_transforms()]
    links[4][1, 3] = 0.0
    return links


def det_jac_complex(links: list[np.ndarray], q: np.ndarray, target: np.ndarray) -> complex:
    """det of the 6x6 Jacobian of the complex-safe error, at q."""
    jacobian = np.zeros((6, 6), dtype=complex)
    for index in range(6):
        step = np.zeros(6, dtype=complex)
        step[index] = 1e-7
        jacobian[:, index] = (err6_complex(links, q + step, target)
                              - err6_complex(links, q - step, target)) / 2e-7
    return np.linalg.det(jacobian)


def refine_branch_pose(links: list[np.ndarray], target_at, q0: np.ndarray, u0: complex,
                       *, iterations: int = 60, tolerance: float = 1e-12,
                       max_step: float = 0.2):
    """Newton on [err6; det J] = 0 over complex (q, u) -- the pose-coordinate fold.

    Trust region as in the loop tracker: the first version oscillated from random complex
    seeds (measured: residual bouncing 0.07..0.5, |update| up to 1.7e3 on step one), because
    an uncapped Newton step can leave the basin entirely.
    """
    y = np.concatenate([np.asarray(q0, dtype=complex), [complex(u0)]])

    def augmented(z: np.ndarray) -> np.ndarray:
        return np.concatenate([err6_complex(links, z[:6], target_at(z[6])),
                               [det_jac_complex(links, z[:6], target_at(z[6]))]])

    for _ in range(iterations):
        value = augmented(y)
        if float(np.linalg.norm(value)) < tolerance:
            break
        jacobian = np.zeros((7, 7), dtype=complex)
        for column in range(7):
            step = np.zeros(7, dtype=complex)
            step[column] = 1e-7
            jacobian[:, column] = (augmented(y + step) - augmented(y - step)) / 2e-7
        update = np.linalg.lstsq(jacobian, -value, rcond=None)[0]
        size = float(np.linalg.norm(update))
        if size > max_step:
            update *= max_step / size
        y = y + update
    return y, float(np.linalg.norm(augmented(y)))


def track_pose(links: list[np.ndarray], q0: np.ndarray, targets: list[np.ndarray], *,
               max_step: float = 0.1, tolerance: float = 1e-10):
    """Carry one root along a sequence of (complex) target poses; adaptive halving."""
    q = np.asarray(q0, dtype=complex)
    for index, target in enumerate(targets):
        q_new, residual = newton_complex(links, q, target)
        tries = 0
        while (float(np.linalg.norm(q_new - q)) > max_step or residual > tolerance) \
                and tries < 8:
            mid = 0.5 * (np.asarray(targets[index - 1]) + target) if index else target
            q_mid, residual_mid = newton_complex(links, q, mid)
            if residual_mid > tolerance:
                tries += 1
                continue
            q_new, residual = newton_complex(links, q_mid, target)
            tries += 1
        if residual > tolerance:
            return q, False
        q = q_new
    return q, True


def cycle_type(images: list[int]) -> str:
    """Cycle type of a permutation in image notation, e.g. '2+2+1'."""
    seen = [False] * len(images)
    lengths = []
    for start in range(len(images)):
        if seen[start]:
            continue
        length, index = 0, start
        while not seen[index]:
            seen[index] = True
            index = images[index]
            length += 1
        lengths.append(length)
    return "+".join(str(v) for v in sorted(lengths, reverse=True))


def main() -> int:
    """Measure SR0's monodromy group along one position coordinate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u-index", type=int, default=0, help="position coordinate to vary")
    parser.add_argument("--radius", type=float, default=5e-4)
    args = parser.parse_args()

    links = sr0_links()
    model = sr0.robot()
    rng = np.random.default_rng(args.seed)
    q_true = rng.uniform(-2.0, 2.0, model.num_dof)
    target0 = np.asarray(model.fk(q_true), dtype=float)
    u0 = float(target0[:3, 3][args.u_index])

    def target_at(u: complex) -> np.ndarray:
        moved = target0.astype(complex)
        moved[:3, 3] = target0[:3, 3].astype(complex)
        moved[:3, 3][args.u_index] = u
        return moved

    fibre = [np.asarray(solution.q, dtype=float) for solution in sr0.solve(model, target0)]
    print(f"SR0 at the base pose: {len(fibre)} analytic solutions")

    branch_points: list[tuple[complex, np.ndarray, bool]] = []
    for sign in (1.0, -1.0):
        qs = [np.asarray(q, dtype=complex) for q in fibre]
        dead: set[int] = set()
        for step in range(1, 301):
            u = u0 + sign * step * 0.001
            alive = []
            for index, q in enumerate(qs):
                if index in dead:
                    continue
                q_new, residual = newton_complex(links, q, target_at(u))
                if residual < 1e-9:
                    qs[index] = q_new
                    alive.append(index)
            if len(alive) < 2:
                break
            close = None
            for a, i in enumerate(alive):
                for j in alive[a + 1:]:
                    distance = torus_distance_complex(qs[i], qs[j])
                    if close is None or distance < close[0]:
                        close = (distance, i, j)
            if close is not None and close[0] < 0.02:
                mid = 0.5 * (qs[close[1]] + qs[close[2]])
                y, residual = refine_branch_pose(links, target_at, mid, u)
                if residual > 1e-8:
                    # Refinement can fail at a genuine fold (trust-region limit).  The transposition
                    # is read from the *roots*, not from a refined point, so record the approximate
                    # location and keep scanning instead of aborting the whole sweep: kimi's day17
                    # lost the informative fold at u ~ -0.0072 here (its pair is real at the base
                    # pose, which is exactly what makes the loop around it informative).
                    if not any(abs(u - old_u) < 0.02 for old_u, _q, _r in branch_points):
                        branch_points.append((complex(u), np.asarray(mid, dtype=complex), False))
                        print(f"  approximate branch point u* ~ {u.real:.6f}{u.imag:+.1e}i "
                              f"(refinement failed, residual {residual:.1e})")
                    dead.add(close[1])
                    dead.add(close[2])
                    continue
                u_star = y[6]
                # the merged pair is dead for further tracking; the rest continue
                dead.add(close[1])
                dead.add(close[2])
                if not any(abs(u_star - old_u) < 1e-4 for old_u, _q, _r in branch_points):
                    branch_points.append((u_star, y[:6], True))
                    print(f"  branch point u* = {u_star.real:.6f}{u_star.imag:+.1e}i "
                          f"(residual {residual:.1e})")
    print(f"  {len(branch_points)} branch points")

    print("phase 2: a complex circle around each branch point, all eight roots tracked")
    generators: list[list[int]] = []
    for u_star, _q_star, refined in branch_points:
        # The circle has to *enclose* the fold.  An unrefined centre can sit closer to the fold than
        # the default radius (measured: the informative fold near u ~ -0.0052 was detected ~2e-3
        # away), so approximate centres get a radius that spans the detection distance -- and the
        # loop's cycle type says whether it enclosed exactly one fold (a single transposition) or
        # more.  The approach must not cross the fold: arc around it in complex u (upper half-plane),
        # and the circle starts on the side where the pair is real -- the exp45 lesson.
        radius = args.radius if refined else 1e-2
        base_u = u_star + radius
        leg_out = [target_at(u) for u in
                   np.fromiter(make_s_path_complex(u0, base_u, [u_star.real]), dtype=complex)]
        circle = [target_at(u_star + radius * np.exp(1j * t))
                  for t in np.linspace(0, 2 * np.pi, 97)[1:]]
        leg_back = [target_at(u) for u in
                    np.fromiter(make_s_path_complex(base_u, u0, [u_star.real]),
                                dtype=complex)[1:]]
        full = leg_out + circle + leg_back
        images: list[int] = []
        for q0 in fibre:
            try:
                q_end, ok = track_pose(links, q0, full)
            except np.linalg.LinAlgError:
                # A step can leave the finite-difference domain (the complex Newton builds a
                # Jacobian from a non-finite configuration and the SVD refuses).  That is an
                # untracked root for this generator, not a reason to abort the measurement --
                # with approximate (unrefined) branch points this happens for real.
                images.append(-1)
                continue
            if not np.all(np.isfinite(np.asarray(q_end, dtype=complex))):
                images.append(-1)
                continue
            if not ok:
                images.append(-1)
                continue
            distances = [torus_distance_complex(q_end, member) for member in fibre]
            images.append(int(np.argmin(distances)) if min(distances) < 1e-2 else -1)
        if -1 in images:
            print(f"  u*={u_star.real:.4f}{u_star.imag:+.1e}i: untracked roots, skipped")
            continue
        generators.append(images)
        print(f"  u*={u_star.real:.4f}{u_star.imag:+.1e}i: cycle type "
              f"{cycle_type(images)}")
    if not generators:
        print("no clean generator measured")
        return 1

    print("phase 3: the generated group")
    size = len(fibre)
    identity = tuple(range(size))
    seen = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            new = tuple(current[generator[i]] for i in range(size))
            if new not in seen:
                seen.add(new)
                frontier.append(new)
    abelian = all(
        all(tuple(g1[g2[i]] for i in range(size)) == tuple(g2[g1[i]] for i in range(size))
            for g2 in generators)
        for g1 in generators
    )
    involutions = all(cycle_type(g).replace("+1", "").strip("+").split("+")
                      and set(cycle_type(g).split("+")) <= {"2", "1"} for g in generators)
    print(f"  generated group order: {len(seen)} | abelian: {abelian} | "
          f"all generators involutions: {involutions}")
    print(f"  prediction (three binary choices): abelian 2-group, order a power of 2 -- "
          f"{'CONSISTENT' if abelian and len(seen) & (len(seen) - 1) == 0 else 'CHECK'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
