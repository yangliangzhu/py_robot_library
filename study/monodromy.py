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
    images = list(result.partial.values())
    if len(set(images)) != len(images):
        # two tracks landed on the same solution: a permutation cannot do that, so something is wrong
        # with the tracker rather than with the geometry (kimi measured this under the old algorithm)
        result.reason = (result.reason + "; " if result.reason else "") + "not injective"
        result.partial = {}
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


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Composition ``first . second`` (apply ``second``, then ``first``)."""
    return tuple(first[second[index]] for index in range(len(first)))


def _inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of a permutation given as images."""
    out = [0] * len(permutation)
    for index, image in enumerate(permutation):
        out[image] = index
    return tuple(out)


def non_solvable_certificate(
    permutations: list[tuple[int, ...]], orbit: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Find a 5-cycle and a double transposition that **provably** generate a non-solvable group.

    A 5-cycle alone gives a solvable group, so the second permutation has to escape the normaliser of
    the first; otherwise the pair generates a dihedral group of order 10, which is solvable, and the
    certificate would be a false proof.  kimi produced exactly that counterexample
    (``<(1 2 3 4 5), (1 4)(2 3)> = D5``, order 10, solvable), which is why the test below is

        tau . sigma . tau^-1  is **not** a power of sigma.

    Under that condition the generated group contains a 5-cycle and is not contained in the
    normaliser of ``<sigma>`` (of order 20 in ``S5``), and the only subgroups of ``S5`` containing a
    5-cycle are ``C5``, ``D5``, the order-20 normaliser, ``A5`` and ``S5`` -- so the group is ``A5``
    or ``S5``, hence not solvable.

    Returns:
        ``(sigma, tau)`` when the certificate holds, else ``None``.
    """
    offsets = {point: index for index, point in enumerate(orbit)}
    five_cycles: list[tuple[int, ...]] = []
    double_transpositions: list[tuple[int, ...]] = []
    for permutation in permutations:
        if any(image < 0 for image in permutation):
            continue
        restricted = tuple(offsets[permutation[point]] for point in orbit)
        lengths = sorted(_cycle_lengths(restricted), reverse=True)
        if len(orbit) == 5 and lengths == [5]:
            five_cycles.append(restricted)
        if lengths == [2, 2] + [1] * (len(orbit) - 4):
            double_transpositions.append(restricted)
    for sigma in five_cycles:
        powers = set()
        current = tuple(range(len(orbit)))
        for _ in range(len(orbit)):
            powers.add(current)
            current = _compose(sigma, current)
        for tau in double_transpositions:
            # tau sigma tau^-1, with the inverse of *tau*: using sigma's inverse instead is invisible
            # whenever tau is an involution, which is exactly why the first version of this test
            # certified kimi's solvable counterexample
            conjugated = _compose(_compose(tau, sigma), _inverse(tau))
            if conjugated not in powers:
                return sigma, tau
    return None


def _self_test() -> str:
    """Check the certificate against kimi's counterexample and against a genuine one."""
    orbit = [0, 1, 2, 3, 4]
    sigma = (1, 2, 3, 4, 0)  # the 5-cycle 0->1->2->3->4->0, in image notation
    bad = (3, 2, 1, 0, 4)  # (1 4)(2 3), the pair kimi showed generates D5 (solvable)
    good = (1, 0, 3, 2, 4)  # (1 2)(3 4), which escapes the normaliser
    assert non_solvable_certificate([sigma, bad], orbit) is None, "D5 must be rejected"
    assert non_solvable_certificate([sigma, good], orbit) is not None, "A5 must be certified"
    return "certificate self-test passed: D5 rejected, A5 certified"


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
    parser.add_argument("--robot", default="sr5", help="'sr5', 'sr0' (control) or any config name")
    parser.add_argument("--seeds", type=int, default=600)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    parser.add_argument("--radius", type=float, default=0.05, help="position-loop radius, metres")
    parser.add_argument("--amplitude", type=float, default=0.4, help="orientation-loop amplitude, rad")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.robot == "sr0":
        # the Pieper-solvable neighbour of the SR5 (link 5's y offset zeroed): the control group for
        # "solvable arm", where the solutions come from +-acos choices and the real monodromy should be
        # simple -- run it to separate tracker weakness from physical obstruction
        from study import sr0 as sr0_module

        model = sr0_module.robot()
    else:
        model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    # deduplicate on the torus (every revolute joint is 2 pi periodic) and keep the solutions that
    # have a representative inside the joint limits: the first is a question about the pose equation,
    # the second about the robot
    solutions: list[np.ndarray] = []
    for candidate in fiber.solutions:
        if iks.sigma_min(model, candidate) <= args.sigma_floor:
            continue
        if not iks.physically_admissible(model, candidate):
            continue
        if any(iks.torus_distance(candidate, kept) < 1e-3 for kept in solutions):
            continue
        solutions.append(candidate)
    print(_self_test())
    print(f"{args.robot}: {len(solutions)} solutions at the base pose "
          f"(clearance above {args.sigma_floor:g})\n")

    loops = []
    for radius in (0.002, 0.005, 0.02, 0.05, 0.1):
        loops.append((f"position xy r={radius}", position_loop(target, radius, first=0, second=1)))
        loops.append((f"position xz r={radius}", position_loop(target, radius, first=0, second=2)))
    for amplitude in (0.02, 0.05, 0.2, 0.4, 0.8):
        loops.append((f"orientation z a={amplitude}", orientation_loop(target, amplitude, axis=2)))
        loops.append((f"orientation x a={amplitude}", orientation_loop(target, amplitude, axis=0)))
    loops.append(("orientation full turn z", orientation_loop(target, 0.0, axis=2, full_turn=True)))
    loops.append(("orientation full turn x", orientation_loop(target, 0.0, axis=0, full_turn=True)))
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

    # kimi's prediction (day01-kimi.md section 2): a loop whose lift never leaves
    # Q \ f^{-1}(Delta) stays inside one uniqueness domain, which holds at most one solution per pose,
    # so it must return the *identity*; a non-trivial permutation therefore requires some sheet to die
    # at a fold.  The tally below is the test: loops with no deaths must be identity, and the claim is
    # falsified by a single survival-only loop whose map is not the identity.
    clean = [r for r in results if not r.failures and r.partial]
    with_deaths = [r for r in results if r.failures and r.partial]
    non_identity = [
        r for r in clean
        if any(image != index for index, image in r.partial.items())
    ]
    print(f"\ntally: {len(results)} loops | all-survivors {len(clean)} (non-identity among them: "
          f"{len(non_identity)}) | some sheets died {len(with_deaths)} | "
          f"prediction 'no deaths => identity': "
          f"{'HELD' if not non_identity else 'FALSIFIED by ' + str([r.name for r in non_identity])}")
    for result in with_deaths:
        moved = [index for index, image in result.partial.items() if image != index]
        print(f"  (deaths present) {result.name}: survivors {len(result.partial)}, "
              f"of which moved {moved}")

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
