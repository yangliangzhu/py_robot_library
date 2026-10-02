"""Loops and monodromy: which solutions can a real motion actually reach?

The theory this module measures (the Day 1 dictionary, ``study/collab/day01-dsh.md``):

* a **loop** in the workspace based at a pose ``T0`` lifts to a permutation of ``T0``'s solutions --
  move the tool around the loop, carry each solution continuously, and it may come back to a
  *different* solution;
* the permutations of all such loops generate the **monodromy group**, which for a generic base point
  equals the Galois group of the IK equation over the function field of the workspace (Harris 1979);
* the **orbits** of that action are the aspects / chambers -- a lift stays inside one connected
  component of ``Q \\ Sigma``, and a path inside a component projects to a loop -- so "same branch"
  means "same orbit", which is why distinct solutions can share a branch;
* with **real** loops only (what a robot can do) this is the *real* monodromy, i.e. cuspidality:
  a non-trivial permutation proves the arm can change posture without meeting a singularity.

Two certificates fall out, both cheap:

* a loop whose lift returns to a different solution **proves** those two solutions share an aspect
  (the tracker refuses to enter the singular set, so the lifted path is a witness);
* permutations that generate a non-solvable group **prove** the IK equation has no radical solution.
  Inside one orbit, a 5-cycle plus a double transposition generate ``A5``, so the group's order is
  never needed -- only two permutations of the right cycle type on the same orbit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from model import IkType, ModelFactory
from study import homotopy
from study import ik_structure as iks
from study.census import census, make_lm_solver


@dataclass
class Track:
    """One solution carried around a loop."""

    q: np.ndarray
    ok: bool
    min_sigma: float
    failed_at: int | None = None
    reason: str = ""


@dataclass
class LoopResult:
    """What one loop did to the whole fibre."""

    name: str
    permutation: tuple[int, ...] = ()
    ok: bool = False
    min_sigma: float = float("nan")
    reason: str = ""
    tracks: list[Track] = field(default_factory=list)
    #: Images of the solutions that survived the loop (partial map when some failed).
    partial: dict[int, int] = field(default_factory=dict)
    #: Which solutions failed and why.
    failures: dict[int, str] = field(default_factory=dict)

    @property
    def cycle_type(self) -> str:
        """Cycle type of the permutation, e.g. ``2+2+1``; ``-`` when the loop failed."""
        if not self.ok:
            return "-"
        seen = [False] * len(self.permutation)
        cycles = []
        for start in range(len(self.permutation)):
            if seen[start]:
                continue
            length, index = 0, start
            while not seen[index]:
                seen[index] = True
                index = self.permutation[index]
                length += 1
            cycles.append(length)
        return "+".join(str(length) for length in sorted(cycles, reverse=True))


def pose_error(model, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The library's SE(3) error of ``FK(q)`` against ``target``, as a 6-vector."""
    return homotopy.se3_error(np.asarray(model.fk(q), dtype=float), target)


def numeric_jacobian(model, q: np.ndarray, target: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Jacobian of the *error* with respect to ``q``, by central differences.

    The library's own Jacobian is a geometric Jacobian in its own frame convention; pairing it with
    the SE(3) error of :func:`pose_error` made every loop diverge (residuals up to 0.33 with healthy
    clearance, i.e. not a singularity).  Differentiating the error itself keeps the two conventions
    identical, which is the same lesson as the tracker's: the predictor and the corrector must agree
    on what they are measuring.
    """
    jacobian = np.zeros((6, len(q)))
    for index in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        # d(error)/dq, in the same order as the Newton step below uses it
        jacobian[:, index] = (
            pose_error(model, plus, target) - pose_error(model, minus, target)
        ) / (2 * h)
    return jacobian


def track_pose_path(
    model, q0: np.ndarray, poses: list[np.ndarray], *, sigma_floor: float = 2e-3,
    iterations: int = 40, tolerance: float = 1e-11, accept: float = 1e-6,
    damping: float = 1e-8, max_step: float = 0.2, check_limits: bool = True,
) -> Track:
    """Carry one configuration along a sequence of poses, refusing to enter the singular set.

    Args:
        model: Robot model.
        q0: Configuration that solves ``poses[0]``.
        poses: The loop, sampled finely enough that each step is a small correction.
        sigma_floor: Clearance below which the track is declared to have met the singular set.
        iterations: Gauss-Newton steps per pose.
        tolerance: Pose error at which the correction stops.
        accept: Pose error above which the track is declared lost.

    Returns:
        The :class:`Track` with the final configuration and why it stopped, if it did.
    """
    q = np.asarray(q0, dtype=float).copy()
    minimum = iks.sigma_min(model, q)
    for index, target in enumerate(poses):
        for _ in range(iterations):
            error = pose_error(model, q, target)
            if float(np.linalg.norm(error)) < tolerance:
                break
            jacobian = numeric_jacobian(model, q, target)
            # damped least squares: the loops pass close to the singular set, where a plain
            # Gauss-Newton step converges slowly or oscillates (measured: stalls at ~1e-3)
            normal = jacobian.T @ jacobian + damping * np.eye(len(q))
            step = -np.linalg.solve(normal, jacobian.T @ error)
            # trust region: with a small damping a near-singular Jacobian produces a giant step, and
            # because the pose is periodic in every joint that step can land on a 2 pi-shifted copy of
            # the same solution -- measured: a lift that wound joint 1 by 158 turns and left the limits
            size = float(np.linalg.norm(step))
            if size > max_step:
                step *= max_step / size
            q = q + step
        error = float(np.linalg.norm(pose_error(model, q, target)))
        if error > accept:
            return Track(q, False, minimum, index, f"lost the pose (error {error:.1e})")
        if check_limits and not iks.within_limits(model, q):
            return Track(q, False, minimum, index, "left the joint limits")
        clearance = iks.sigma_min(model, q)
        minimum = min(minimum, clearance)
        if clearance < sigma_floor:
            return Track(q, False, minimum, index, f"met the singular set ({clearance:.1e})")
    return Track(q, True, minimum)


def position_loop(
    target: np.ndarray, radius: float, *, samples: int = 48, first: int = 0, second: int = 1
) -> list[np.ndarray]:
    """A closed loop of the tool position (orientation fixed), starting and ending at ``target``."""
    poses = []
    for step in range(samples + 1):
        angle = 2.0 * np.pi * step / samples
        offset = np.zeros(3)
        offset[first] = radius * (np.cos(angle) - 1.0)
        offset[second] = radius * np.sin(angle)
        pose = target.copy()
        pose[:3, 3] = target[:3, 3] + offset
        poses.append(pose)
    return poses


def orientation_loop(
    target: np.ndarray, amplitude: float, *, samples: int = 48, axis: int = 2,
    full_turn: bool = False,
) -> list[np.ndarray]:
    """A closed loop of the tool orientation about one axis, position fixed.

    ``full_turn=False`` sweeps out and back (a contractible loop); ``full_turn=True`` rotates a full
    turn, which in ``SO(3)`` is *not* contractible -- the belt trick -- so the two cases are expected
    to behave differently, and that difference is part of what this module measures.
    """
    poses = []
    for step in range(samples + 1):
        angle = 2.0 * np.pi * step / samples if full_turn else amplitude * np.sin(
            2.0 * np.pi * step / samples
        )
        pose = target.copy()
        pose[:3, :3] = _rotation(axis, angle) @ target[:3, :3]
        poses.append(pose)
    return poses


def _rotation(axis: int, angle: float) -> np.ndarray:
    """Rotation about a coordinate axis."""
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(3)
    other = [index for index in range(3) if index != axis]
    matrix[other[0], other[0]] = cosine
    matrix[other[0], other[1]] = -sine
    matrix[other[1], other[0]] = sine
    matrix[other[1], other[1]] = cosine
    return matrix


def loop_permutation(
    model, target: np.ndarray, solutions: list[np.ndarray], poses: list[np.ndarray], circular,
    *, name: str = "", sigma_floor: float = 2e-3, match: float = 1e-3,
) -> LoopResult:
    """Carry every solution around the loop and read off the permutation it induces."""
    result = LoopResult(name=name)
    for index, q in enumerate(solutions):
        track = track_pose_path(model, q, poses, sigma_floor=sigma_floor)
        result.tracks.append(track)
        if not track.ok:
            # a solution that cannot follow the loop is a *result*, not a failure of the experiment:
            # it may hit a joint limit or run into the singular set, and both are physical
            result.failures[index] = track.reason
            continue
        # torus distance: every revolute joint is 2 pi periodic, and a lift can return to a different
        # representative of the same solution
        distances = [iks.torus_distance(track.q, other) for other in solutions]
        nearest = int(np.argmin(distances))
        if distances[nearest] > match:
            result.failures[index] = "ended away from every solution"
            continue
        result.partial[index] = nearest
    result.permutation = tuple(result.partial.get(index, -1) for index in range(len(solutions)))
    result.ok = len(result.partial) == len(solutions)
    result.min_sigma = min(
        (track.min_sigma for track in result.tracks if not np.isnan(track.min_sigma)),
        default=float("nan"),
    )
    result.reason = "; ".join(f"{index}: {why}" for index, why in sorted(result.failures.items()))
    return result


def orbits(permutations: list[tuple[int, ...]], size: int) -> list[list[int]]:
    """Orbits of the group generated by the permutations (union-find, no group enumeration).

    This is the aspect structure: two solutions are in one orbit exactly when some product of loops
    carries one to the other.  Entries equal to ``-1`` mark solutions that did not survive that loop
    and are ignored, so a partial map still contributes the connections it does establish.

    Args:
        permutations: Permutations of ``range(size)``, ``-1`` for "did not survive".
        size: Number of solutions.

    Returns:
        The orbits, each a sorted list of solution indices.
    """
    parent = list(range(size))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for permutation in permutations:
        for index, image in enumerate(permutation):
            if image >= 0:
                parent[find(index)] = find(image)
    groups: dict[int, list[int]] = {}
    for index in range(size):
        groups.setdefault(find(index), []).append(index)
    return [sorted(group) for group in groups.values()]


def non_solvable_certificate(
    permutations: list[tuple[int, ...]], orbit: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Find a 5-cycle and a double transposition inside one orbit, which generate ``A5``.

    ``A5`` is simple and non-solvable, and a subgroup of a solvable group is solvable, so exhibiting
    ``A5`` inside the monodromy group proves the group -- and therefore the Galois group of the IK
    equation -- is not solvable, hence no solution in radicals.  The cycle types are read off the
    permutations restricted to the orbit, so nothing has to be enumerated.

    Returns:
        The two permutations, or ``None`` when this orbit shows no such certificate.
    """
    five_cycle = double_transposition = None
    for permutation in permutations:
        restricted = tuple(orbit.index(permutation[point]) for point in orbit)
        cycles = _cycle_lengths(restricted)
        if len(orbit) == 5 and sorted(cycles, reverse=True) == [5]:
            five_cycle = restricted
        if sorted(cycles, reverse=True) == [2, 2] + [1] * (len(orbit) - 4):
            double_transposition = restricted
    if five_cycle is not None and double_transposition is not None:
        return five_cycle, double_transposition
    return None


def _cycle_lengths(permutation: tuple[int, ...]) -> list[int]:
    """Cycle lengths of a permutation given as images."""
    seen = [False] * len(permutation)
    lengths = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        length, index = 0, start
        while not seen[index]:
            seen[index] = True
            index = permutation[index]
            length += 1
        lengths.append(length)
    return lengths


def main() -> int:
    """Run the first monodromy measurement on one pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="sr5")
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--radius", type=float, default=0.05, help="position-loop radius, metres")
    parser.add_argument("--amplitude", type=float, default=0.4, help="orientation-loop amplitude, rad")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.sigma_floor]
    print(f"{args.robot}: {len(solutions)} solutions at the base pose "
          f"(clearance above {args.sigma_floor:g})\n")

    loops = [
        ("position circle xy", position_loop(target, args.radius, first=0, second=1)),
        ("position circle xz", position_loop(target, args.radius, first=0, second=2)),
        ("orientation sweep z", orientation_loop(target, args.amplitude, axis=2)),
        ("orientation sweep x", orientation_loop(target, args.amplitude, axis=0)),
        ("orientation full turn z", orientation_loop(target, 0.0, axis=2, full_turn=True)),
    ]
    results = []
    for name, poses in loops:
        result = loop_permutation(
            model, target, solutions, poses, circular, name=name, sigma_floor=args.sigma_floor
        )
        results.append(result)
        survived = len(result.partial)
        print(f"  {name:24s}: {survived}/{len(solutions)} solutions followed the loop | "
              f"map {result.partial if survived else '{}'} | min clearance {result.min_sigma:.2e}")
        if result.failures:
            print(f"      did not follow: {result.reason}")

    good = [result.permutation for result in results if result.partial]
    if good:
        decomposition = orbits(good, len(solutions))
        print(f"\norbits of the loops that succeeded: {decomposition}")
        print("  (each orbit is an aspect: solutions that real motions can carry into each other)")
        for orbit in decomposition:
            certificate = non_solvable_certificate(good, orbit) if len(orbit) == 5 else None
            print(f"  orbit of size {len(orbit)}: non-solvable certificate "
                  f"{'FOUND ' + str(certificate) if certificate else 'not found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
