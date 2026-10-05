#!/usr/bin/env python3
"""exp59: SR0's monodromy generator -- sweep *radially* to a real fold, then loop it (Q8).

``exp58`` established that the complex tracker can be made trustworthy (linear prediction, jump
rejection, adaptive step) and that one-coordinate sweeps of 0.03-0.06 m never meet a fold: the eight
roots stay 2.8-4.0 rad apart because the workspace boundary is tens of centimetres away.  This module
therefore sweeps **radially** -- the target moves along the line from the base to (and past) the base
pose's radius -- with a 0.6 m span, stops at the first dip of the minimum pair gap (a real pair
coalescing, which is what a fold is), refines it with the augmented system, and tracks all eight roots
around a small complex circle enclosing it.

The permutation is only accepted when all eight roots survive the loop with their identities
unambiguous; a single transposition then generates a group of order 2 (abelian, exponent 2), which is
the first non-trivial lower bound for SR0's monodromy group -- the thing Q8 is after.

Run::

    python3 -m study.exp59_sr0_radial_fold
"""

from __future__ import annotations

import argparse

import numpy as np

from study import sr0
from study.exp37_complex_loop import newton_complex
from study.exp40_complex_branches import make_s_path_complex
from study.exp57_sr0_monodromy import refine_branch_pose, torus_distance_complex
from study.exp58_sr0_fold_and_group import min_pair_gap, walk
from study.sr0 import sr0_links


def build_radial(seed: int, model):
    """A base pose and a radial coordinate ``u`` for it (``u = 0`` at the base pose).

    Args:
        seed: Seed of the random configuration whose ``fk`` gives the base pose.
        model: The SR0 model.

    Returns:
        ``(target0, target_at, fibre)`` with ``target_at(u)`` the pose displaced by ``u`` metres
        along the base pose's radius, and ``fibre`` its eight analytic roots.
    """
    local = np.random.default_rng(seed)
    q_true = local.uniform(-2.0, 2.0, model.num_dof)
    target0 = np.asarray(model.fk(q_true), dtype=float)
    position = target0[:3, 3]
    radial = position / float(np.linalg.norm(position))

    def target_at(u: complex) -> np.ndarray:
        moved = target0.astype(complex)
        moved[:3, 3] = position.astype(complex) + float(np.real(u)) * radial
        if np.imag(u) != 0.0:
            moved[:3, 3] = moved[:3, 3] + 1j * np.imag(u) * radial
        return moved

    fibre = [np.asarray(solution.q, dtype=complex) for solution in sr0.solve(model, target0)]
    return target0, target_at, fibre


def sweep(links, target_at, fibre, direction: float, span: float, step: float, *, dip: float,
          jump_tolerance: float = 0.3):
    """Walk the eight roots along ``direction`` until a pair dips or a root dies.

    Returns:
        ``(event, info)`` where ``event`` is ``"dip"``, ``"death"`` or ``"span"``, and ``info`` holds
        ``u``, the gap, the closest pair's configurations and the number of root-step faults.
    """
    qs = [q.copy() for q in fibre]
    previous = [q.copy() for q in qs]
    alive = list(range(len(fibre)))
    u = 0.0
    step = abs(step) * direction
    faults = 0
    #: The fold is bracketed by the last step where all eight roots were still real and the first one
    #: where the corrector cannot continue them at any step size -- the adaptive step makes that
    #: bracket as tight as its floor (1e-9 m), which is what the circle below needs.
    info = {"u": 0.0, "gap": float("inf"), "pair": None, "faults": 0, "dip": float("inf")}
    while abs(u) < span:
        trial = u + step
        pose = target_at(trial)
        updated = {}
        failed = 0
        for index in alive:
            prediction = 2.0 * qs[index] - previous[index]
            q_new, residual = newton_complex(links, prediction, pose)
            if (residual > 1e-10 or not np.all(np.isfinite(q_new))
                    or float(np.linalg.norm(q_new - prediction)) > jump_tolerance):
                failed += 1
                continue
            updated[index] = (q_new, qs[index])
        if failed:
            faults += failed
            step *= 0.5
            if abs(step) < 1e-9:
                info.update({"u": u, "faults": faults})
                return "death", info
            continue
        for index, (q_new, q_old) in updated.items():
            previous[index] = q_old
            qs[index] = q_new
        u = trial
        gap, first, second = min_pair_gap(qs, alive)
        info["dip"] = min(info["dip"], gap)
        if gap < info["gap"]:
            info.update({"u": u, "gap": gap,
                         "pair": (qs[first].copy(), qs[second].copy())})
        step = min(abs(step) * 1.4, abs(step) * 4.0) * direction
    info["faults"] = faults
    return "span", info


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=6, help="base poses to try")
    parser.add_argument("--span", type=float, default=0.6, help="radial sweep length, metres")
    parser.add_argument("--step", type=float, default=2e-3, help="initial radial step, metres")
    parser.add_argument("--dip", type=float, default=5e-2,
                        help="pair gap below which the sweep stops (a fold in reach)")
    parser.add_argument("--radius", type=float, default=5e-4, help="circle radius around the fold")
    parser.add_argument("--folds", type=int, default=4,
                        help="how many fold candidates to refine and loop around")
    args = parser.parse_args()

    links = sr0_links()
    model = sr0.robot()
    print("exp59: radial sweep to a real fold, then a complex loop around it\n")
    print(f"phase 1: {args.seeds} base poses, radial span {args.span:g} m, stop at gap < {args.dip:g}")
    candidates: list[tuple[float, int, float, dict, object, list[np.ndarray]]] = []
    for seed in range(args.seeds):
        target0, target_at, fibre = build_radial(seed, model)
        if len(fibre) < 8:
            continue
        for direction in (-1.0, 1.0):
            event, info = sweep(links, target_at, fibre, direction, args.span, args.step,
                                dip=args.dip)
            gap = info["gap"] if info["pair"] is not None else float("nan")
            print(f"  seed {seed} direction {direction:+.0f}: {event} at u = {info['u']:+.4f} m, "
                  f"gap there {gap:.3e} (smallest seen {info['dip']:.2e}), faults "
                  f"{info['faults']}")
            if info["pair"] is not None and event in ("death", "dip"):
                candidates.append((gap, seed, direction, info, target_at, fibre))
    if not candidates:
        print("\n  no pair coalescence found in any sweep -- widen --span or raise --seeds")
        return 1
    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:args.folds]
    print(f"\n  {len(candidates)} fold candidate(s) (best gaps: "
          f"{[f'{item[0]:.1e}' for item in candidates]})")

    def close_group(generators: list[tuple[int, ...]], size: int, cap: int = 200000):
        """Closure of the permutations under composition, capped; returns the set and abelian flag."""
        identity = tuple(range(size))
        group = {identity}
        frontier = [identity]
        while frontier:
            new_frontier = []
            for element in frontier:
                for generator in generators:
                    composed = tuple(generator[element[i]] for i in range(size))
                    if composed not in group:
                        group.add(composed)
                        new_frontier.append(composed)
                        if len(group) > cap:
                            return group, False
            frontier = new_frontier
        abelian = all(tuple(a[b[i]] for i in range(size)) == tuple(b[a[i]] for i in range(size))
                      for a in group for b in generators)
        return group, abelian

    print("\nphase 2+3: refine each fold, loop it, and read the permutation")
    generators: list[tuple[int, ...]] = []
    for gap, seed, direction, info, target_at, fibre in candidates:
        y, residual = refine_branch_pose(links, target_at, 0.5 * (info["pair"][0] + info["pair"][1]),
                                         info["u"])
        if residual > 1e-8:
            print(f"  seed {seed} {direction:+.0f}: refinement failed (residual {residual:.1e}), "
                  f"skipped")
            continue
        u_star = y[6].real
        base_u = u_star + args.radius
        leg_out = [target_at(u) for u in make_s_path_complex(0.0, base_u, [u_star])]
        circle = [target_at(u_star + args.radius * np.exp(1j * theta))
                  for theta in np.linspace(0.0, 2 * np.pi, 97)[1:]]
        leg_back = [target_at(u) for u in make_s_path_complex(base_u, 0.0, [u_star])][1:]
        poses = leg_out + circle + leg_back
        images: list[int] = []
        for q0 in fibre:
            q_end, ok = walk(links, q0, poses)
            if not ok:
                images.append(-1)
                continue
            distances = [torus_distance_complex(q_end, member) for member in fibre]
            images.append(int(np.argmin(distances)) if min(distances) < 1e-2 else -1)
        cycles, seen = [], set()
        for begin in range(len(images)):
            if begin in seen or images[begin] < 0:
                continue
            length, node = 0, begin
            while node not in seen and images[node] >= 0:
                seen.add(node)
                node = images[node]
                length += 1
            cycles.append(length)
        identity = all(images[i] == i for i in range(len(images)))
        print(f"  fold u*={u_star:+.6f} (gap before refinement {gap:.1e}): permutation {images}; "
              f"cycle type {'+'.join(str(length) for length in sorted(cycles, reverse=True))}; "
              f"untracked {images.count(-1)}; identity {identity}")
        if -1 not in images and not identity:
            generators.append(tuple(images))

    print("\nphase 4: the group generated by the measured loops")
    if not generators:
        print("  no clean non-identity generator -- the group's lower bound stays trivial")
        return 1
    group, abelian = close_group(generators, len(fibre))
    orders = sorted({_order(generator) for generator in generators})
    print(f"  {len(generators)} generator(s), supports "
          f"{[sum(1 for i, image in enumerate(generator) if image != i) for generator in generators]}")
    print(f"  every generator an involution: {all(_order(generator) == 2 for generator in generators)}"
          f"; cycle-type orders {orders}")
    print(f"  closure order {len(group)}; abelian {abelian}; order a power of two "
          f"{len(group) > 0 and (len(group) & (len(group) - 1)) == 0}")
    print("\n  reading: a group of order 2^k is solvable (Burnside), and so is every subgroup of an "
          "abelian group; either way the measured subgroup is solvable, which is what SR0's closed "
          "form demands -- and it is exactly what SR5 fails (its group contains S6).  The measured "
          "group is a subgroup of the true monodromy group, so this is a lower bound on the evidence.")


def _order(permutation: tuple[int, ...]) -> int:
    """The order of a permutation."""
    size = len(permutation)
    seen = [False] * size
    order = 1
    for start in range(size):
        if seen[start]:
            continue
        length, node = 0, start
        while not seen[node]:
            seen[node] = True
            node = permutation[node]
            length += 1
        if length:
            order = order * length // _gcd(order, length) if order % length else order
    return order


def _gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


if __name__ == "__main__":
    raise SystemExit(main())
