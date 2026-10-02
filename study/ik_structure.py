"""Tools for studying the branch structure of an inverse-kinematics map.

The objects here are the ones the branch question is actually about:

* ``Sigma`` -- the singular set, ``{q : rank J(q) < 6}``.  Measured by the smallest
  singular value of the geometric Jacobian, ``sigma_min``, which is zero exactly on
  ``Sigma`` and continuous everywhere (unlike ``det J``, which is signed and only exists
  for square Jacobians, and unlike the Yoshikawa measure, which is a *product* of singular
  values and therefore flattens out near the set instead of vanishing linearly).
* a **chamber** -- a connected component of ``Q \\ Sigma``.  Two configurations in
  different chambers cannot be joined by any path that avoids ``Sigma``: that direction is a
  tautology, and it is the *useful* direction, because it gives a certificate of "different
  branch" from a path scan.
* the **fiber** over a pose ``T`` -- ``{q : FK(q) = T}``.  For a 6-DOF arm it is a finite
  set; for a redundant arm it is a union of curves (the self-motion manifolds).  A *branch*
  is a connected component of ``fiber \\ Sigma``, i.e. what remains of one self-motion curve
  once the singular configurations on it are removed.

Every function here is deliberately backend-agnostic: it takes any object with the library's
``fk``, ``jacobian``, ``manip``, ``ik`` and ``lower_bounds``/``upper_bounds`` API, so the
same experiment runs on ``RobotModelNumpy`` and ``RobotModelCasadi``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# local measurements
# --------------------------------------------------------------------------- #


def sigma_min(model, q: np.ndarray) -> float:
    """Smallest singular value of the geometric Jacobian: 0 exactly on the singular set."""
    return float(np.linalg.svd(model.jacobian(np.asarray(q, dtype=float)), compute_uv=False)[-1])


def sigma_spectrum(model, q: np.ndarray) -> np.ndarray:
    """All singular values of the Jacobian, descending."""
    return np.linalg.svd(model.jacobian(np.asarray(q, dtype=float)), compute_uv=False)


def det_jacobian(model, q: np.ndarray) -> float:
    """Determinant of the geometric Jacobian; only defined for a square (6-DOF) Jacobian.

    The determinant is the cheap *signed* witness this study leans on: it is continuous, it
    vanishes exactly on the singular set for a 6-DOF arm, and a *sign change* between two
    samples is a certified crossing -- unlike a manipulability dip, which is only a symptom.

    Args:
        model: Robot model.
        q: Joint positions in radians, length ``num_dof``.

    Returns:
        ``det J(q)``.

    Raises:
        ValueError: If the robot is not 6-DOF, where ``det J`` is not defined.
    """
    jac = np.asarray(model.jacobian(np.asarray(q, dtype=float)), dtype=float)
    if jac.shape != (6, 6):
        raise ValueError(f"det J needs a 6x6 Jacobian, got {jac.shape}")
    return float(np.linalg.det(jac))


def manipulation_measure(model, q: np.ndarray) -> float:
    """The library's manipulability at ``q`` (Yoshikawa's measure)."""
    return float(model.manip(np.asarray(q, dtype=float)))


def pose_error(model, q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """``(position error, rotation error)`` of ``FK(q)`` against ``target``."""
    diff = model.se3_diff(model.fk(q), target)
    return float(np.linalg.norm(diff[:3])), float(np.linalg.norm(diff[3:]))


def is_regular(model, q: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether ``q`` is away from the singular set by ``tol``."""
    return sigma_min(model, q) > tol


# --------------------------------------------------------------------------- #
# the fiber: several solutions for one pose
# --------------------------------------------------------------------------- #


def torus_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Distance between two configurations **on the torus**: every revolute joint is 2 pi periodic.

    This is the right notion of "the same solution".  ``circular_joints`` (below) only marks the
    joints whose *limits* span a full turn, which is a fact about the physical box, not about the
    kinematics: FK depends on every revolute joint through its cosine and sine, so a configuration and
    the same configuration shifted by 2 pi in any joint are the same pose solution.  A census can
    return representatives that differ by hundreds of multiples of 2 pi (measured: 2 pi x
    [158, -30, -1, 125, 101, 6] between two representatives of *one* solution), which inflates counts
    unless they are compared this way.
    """
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return float(np.linalg.norm((delta + np.pi) % (2.0 * np.pi) - np.pi))


def canonical_representative(q: np.ndarray) -> np.ndarray:
    """The representative of a configuration with every joint wrapped into ``(-pi, pi]``."""
    values = np.asarray(q, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def within_limits(model, q: np.ndarray) -> bool:
    """Whether a configuration lies inside the joint limits (a physical filter, not an identity)."""
    values = np.asarray(q, dtype=float)
    return bool(np.all(values >= np.asarray(model.lower_bounds)) and
                np.all(values <= np.asarray(model.upper_bounds)))


def physically_admissible(model, q: np.ndarray) -> bool:
    """Whether some torus-equivalent representative of ``q`` lies inside the joint limits.

    A solution whose only representatives are outside the box is mathematically a solution of the pose
    equation and physically unreachable; the two questions have to be asked separately.
    """
    if within_limits(model, q):
        return True
    for shift in np.ndindex(*([3] * len(q))):
        candidate = canonical_representative(
            np.asarray(q, dtype=float) + (np.array(shift, dtype=float) - 1.0) * 2.0 * np.pi
        )
        if within_limits(model, candidate):
            return True
    return False


def circular_joints(model, tol: float = 1e-9) -> np.ndarray:
    """Which joints are continuous rotation, as opposed to limited intervals.

    The configuration space is a product of circles and intervals, and the distinction is not
    cosmetic: on a limited joint ``q`` and ``q + 2*pi`` are *different* configurations (the
    arm cannot get from one to the other without passing the limit), while on a continuous
    joint they are the same one.  Treating every joint as circular would let a "shortest path"
    leave the joint range; treating every joint as an interval would invent branch boundaries
    at the limits.  A joint is circular when its range covers a full turn.
    """
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    return (upper - lower) >= TWO_PI - tol


def configuration_distance(a: np.ndarray, b: np.ndarray, circular: np.ndarray) -> float:
    """Distance in the configuration space, wrapping the circular joints only."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta = np.where(circular, (delta + np.pi) % TWO_PI - np.pi, delta)
    return float(np.linalg.norm(delta))


def shortest_representative(q_from: np.ndarray, q_to: np.ndarray, circular: np.ndarray) -> np.ndarray:
    """``q_to`` shifted by whole turns on the circular joints, so the straight line is shortest."""
    delta = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    delta = np.where(circular, (delta + np.pi) % TWO_PI - np.pi, delta)
    return np.asarray(q_from, dtype=float) + delta


@dataclass
class Fiber:
    """The solutions found for one pose, with the bookkeeping a census needs."""

    target: np.ndarray
    solutions: list[np.ndarray] = field(default_factory=list)
    #: How many seeds converged, and how many hit each solution (index-aligned).
    hits: list[int] = field(default_factory=list)
    seeds: int = 0
    converged: int = 0
    #: Position/rotation error of each solution (they are all acceptance-level).
    errors: list[tuple[float, float]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many distinct solutions were found."""
        return len(self.solutions)

    def summary(self) -> dict[str, object]:
        """A compact, printable description of the census."""
        return {
            "solutions": self.count,
            "seeds": self.seeds,
            "converged": self.converged,
            "hits": sorted(self.hits, reverse=True),
            "worst_error": max((max(err) for err in self.errors), default=float("nan")),
        }


def find_fiber(
    model,
    target: np.ndarray,
    *,
    seeds: int = 512,
    rng: np.random.Generator | None = None,
    dedupe_tol: float = 1e-4,
    accept: float = 1e-5,
    circular: np.ndarray | None = None,
) -> Fiber:
    """Multi-start IK for one pose, deduplicated on the torus.

    Args:
        model: A robot model exposing ``ik``, ``fk`` and the joint bounds.
        target: The 4x4 pose to solve for.
        seeds: How many random restarts to run.
        rng: Random generator (pass one for reproducibility).
        dedupe_tol: Two solutions closer than this on the torus are the same one.
        accept: Position/rotation error a solution must reach to be kept.

    Returns:
        The :class:`Fiber`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    circular = circular_joints(model) if circular is None else circular
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    fiber = Fiber(target=np.asarray(target, dtype=float), seeds=seeds)
    for _ in range(seeds):
        seed = rng.uniform(lower, upper)
        q, ok = model.ik(seed, target)
        if not ok:
            continue
        position, rotation = pose_error(model, q, target)
        if max(position, rotation) > accept:
            continue
        fiber.converged += 1
        for index, known in enumerate(fiber.solutions):
            if configuration_distance(q, known, circular) < dedupe_tol:
                fiber.hits[index] += 1
                break
        else:
            fiber.solutions.append(np.asarray(q, dtype=float))
            fiber.hits.append(1)
            fiber.errors.append((position, rotation))
    return fiber


# --------------------------------------------------------------------------- #
# paths between two configurations
# --------------------------------------------------------------------------- #


def straight_path(
    q_from: np.ndarray, q_to: np.ndarray, samples: int, circular: np.ndarray | None = None
) -> np.ndarray:
    """``samples`` points along the shortest straight line between two configurations."""
    if circular is None:
        circular = np.zeros_like(np.asarray(q_from, dtype=float), dtype=bool)
    end = shortest_representative(q_from, q_to, circular)
    s = np.linspace(0.0, 1.0, samples)[:, None]
    return np.asarray(q_from, dtype=float) + s * (end - np.asarray(q_from, dtype=float))


def _golden_minimum(function, low: float, high: float, iterations: int = 60) -> tuple[float, float]:
    """Golden-section minimisation of a scalar function on ``[low, high]``.

    Written out rather than imported: the study code is meant to need nothing beyond what
    the library already depends on (numpy, casadi, yaml).
    """
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(low), float(high)
    c, d = b - inverse_phi * (b - a), a + inverse_phi * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inverse_phi * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + inverse_phi * (b - a)
            fd = function(d)
    s = 0.5 * (a + b)
    return s, float(function(s))


@dataclass
class PathScan:
    """What a path between two configurations says about the singular set."""

    #: Smallest ``sigma_min`` found on the path, after refining every local minimum.
    min_sigma: float
    #: Path parameter of that minimum, in ``[0, 1]``.
    min_at: float
    #: The configuration at that minimum.
    min_q: np.ndarray
    #: Smallest manipulability seen on the coarse sample, and the value at each end.
    min_manip: float
    end_manip: tuple[float, float]
    #: ``sigma_min`` at each end (a pair whose ends are singular is not a branch question).
    end_sigma: tuple[float, float]
    #: Local minima found on the coarse sample, refined.
    local_minima: list[tuple[float, float]] = field(default_factory=list)
    samples: int = 0
    #: ``det J`` at the two ends (NaN when the Jacobian is not square).
    end_det: tuple[float, float] = (float("nan"), float("nan"))
    #: Path parameters where ``det J`` changes sign, bisected to the crossing.
    det_sign_changes: list[float] = field(default_factory=list)
    #: Path parameters where ``|det J|`` touches zero *without* changing sign (a grazing hit).
    det_touches: list[float] = field(default_factory=list)

    @property
    def crossings(self) -> int:
        """Certified transversal crossings of the singular set along this path."""
        return len(self.det_sign_changes)

    @property
    def different_chambers(self) -> bool:
        """Whether an odd number of transversal crossings *proves* different chambers."""
        return self.crossings % 2 == 1

    @property
    def end_singular(self) -> bool:
        """Whether either end sits on the singular set (the user's excluded case)."""
        return min(self.end_sigma) < 1e-6

    @property
    def touches_singular_set(self) -> bool:
        """Whether the path passes through ``Sigma``, judged by the refined minimum."""
        return self.min_sigma < 1e-7

    @property
    def manipulability_ratio(self) -> float:
        """``min manip / max end manip`` -- the statistic the empirical criterion watches."""
        reference = max(self.end_manip)
        return float(self.min_manip / reference) if reference > 0 else float("nan")


def scan_path(
    model,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    samples: int = 201,
    refine: bool = True,
    circular: np.ndarray | None = None,
) -> PathScan:
    """Walk the shortest straight line between two configurations and look for ``Sigma``.

    The coarse sample is only used to bracket local minima of ``sigma_min``; each bracket is
    then refined by golden-section search, so "crossed the singular set" is decided by a
    refined value that is actually zero, not by a threshold on a sampled dip.  That
    distinction is the whole point of the experiment: a *dip* can be a near miss.
    """
    path = straight_path(q_from, q_to, samples, circular)
    sigma = np.array([sigma_min(model, q) for q in path])
    manip = np.array([manipulation_measure(model, q) for q in path])

    brackets = []
    for index in range(1, samples - 1):
        if sigma[index] <= sigma[index - 1] and sigma[index] <= sigma[index + 1]:
            brackets.append((index - 1, index + 1))
    # the ends are candidates too, but they are reported separately
    minima: list[tuple[float, float]] = []
    best_s, best_value = float(np.argmin(sigma)) / (samples - 1), float(np.min(sigma))
    if refine and brackets:
        span = 1.0 / (samples - 1)
        for low, high in brackets:
            s, value = _golden_minimum(
                lambda s_: sigma_min(model, path[0] + s_ * (path[-1] - path[0])),
                low * span,
                high * span,
            )
            minima.append((s, value))
            if value < best_value:
                best_s, best_value = s, value
    else:
        minima.append((best_s, best_value))

    end_det = (float("nan"), float("nan"))
    sign_changes: list[float] = []
    touches: list[float] = []
    if model.num_dof == 6:
        dets = np.array([det_jacobian(model, q) for q in path])
        end_det = (float(dets[0]), float(dets[-1]))
        span = 1.0 / (samples - 1)

        def det_at(s_: float) -> float:
            return det_jacobian(model, path[0] + s_ * (path[-1] - path[0]))

        for index in range(samples - 1):
            low, high = dets[index], dets[index + 1]
            if low == 0.0 or high == 0.0 or (low < 0.0) != (high < 0.0):
                # bisect the sign change (or the exact zero) to a path parameter
                a, b = index * span, (index + 1) * span
                fa = det_at(a)
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    fm = det_at(mid)
                    if fm == 0.0:
                        a = b = mid
                        break
                    if (fa < 0.0) != (fm < 0.0):
                        b = mid
                    else:
                        a, fa = mid, fm
                sign_changes.append(0.5 * (a + b))
        # a grazing hit: a local minimum of |det J| that refinement brings to zero without a
        # sign change (the path touches the singular set instead of crossing it)
        for index in range(1, samples - 1):
            if abs(dets[index]) <= abs(dets[index - 1]) and abs(dets[index]) <= abs(dets[index + 1]):
                if abs(dets[index]) > 1e-9 * max(1.0, float(np.max(np.abs(dets)))):
                    continue
                s_, value = _golden_minimum(
                    lambda s_: abs(det_at(s_)), (index - 1) * span, (index + 1) * span
                )
                if value < 1e-12 * max(1.0, float(np.max(np.abs(dets)))):
                    touches.append(s_)

    best_q = path[0] + best_s * (path[-1] - path[0])
    return PathScan(
        min_sigma=best_value,
        min_at=best_s,
        min_q=best_q,
        min_manip=float(np.min(manip)),
        end_manip=(float(manip[0]), float(manip[-1])),
        end_sigma=(float(sigma[0]), float(sigma[-1])),
        local_minima=sorted(minima, key=lambda item: item[1]),
        samples=samples,
        end_det=end_det,
        det_sign_changes=sign_changes,
        det_touches=touches,
    )


def random_regular_configuration(
    model,
    rng: np.random.Generator,
    *,
    sigma_floor: float = 1e-3,
    margin: float = 1e-9,
    tries: int = 500,
) -> np.ndarray:
    """A random configuration well inside the joint bounds and away from ``Sigma``."""
    lower = np.asarray(model.lower_bounds, dtype=float) + margin
    upper = np.asarray(model.upper_bounds, dtype=float) - margin
    for _ in range(tries):
        q = rng.uniform(lower, upper)
        if sigma_min(model, q) > sigma_floor:
            return q
    raise RuntimeError(
        f"no regular configuration found in {tries} tries (sigma_floor={sigma_floor})"
    )


def random_reachable_target(model, rng: np.random.Generator, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """``(target pose, configuration that produced it)``, the pose being regular."""
    q = random_regular_configuration(model, rng, **kwargs)
    return model.fk(q), q
