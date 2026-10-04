"""A 3R orthogonal arm in *position-task* form -- the cuspidality witness and the test bed.

Why this module exists
----------------------
The study claims two things are orthogonal: **solvability in radicals** (complex monodromy,
`study/exp44_group_from_edges.py`) and **branch structure** (real cuspidality,
`study/exp45_sr0_cuspidality.py`).  A witness for the claim needs one arm that is *both*
cuspidal and solvable by radicals.  SR0 is Pieper-solvable but measured non-cuspidal (exp45), so
the witness has to come from a family whose IK is elementary by construction.

The arm is Wenger's 3-R orthogonal robot (`arXiv:1610.04080`, Fig. 1 and eq. (1)): three revolute
joints with **mutually orthogonal** axes, geometric parameters ``d2, d3, r2`` and a tool offset
``d4``.  Its IK is a quartic in ``t = tan(theta_3 / 2)`` (the review cites Kholi & Spanos 1985 for
the derivation), hence solvable by radicals, while the review's classification (its section 4.1)
lists the geometric conditions under which such an arm is *not* cuspidal -- so the family contains
both kinds, and cuspidality must be measured, not assumed.

Two instrument decisions matter and were both forced by measurement (exp46's failure is recorded
in its docstring):

* **The task is the position only.**  exp46 aimed a 6-DOF pose solver at a 3-joint arm and got no
  solutions at all: a 3-joint arm reaches a 3-parameter family of poses, so a pose drawn from its
  own forward kinematics has a one-element fibre and the "IK census" measures nothing.  The
  position map ``q -> P`` is the map the theory is about, and the singular set is where its 3x3
  Jacobian is singular.
* **The link matrices come from an MDH table.**  exp46 composed ``Rx(-90)`` with ``Rx(90)`` and
  left joint 3 parallel to joint 2, so three joints could not control three coordinates
  (``rank J = 2`` everywhere).  :func:`build` takes explicit MDH rows in the library's convention
  and :func:`rank_report` asserts ``rank J = 3`` before anything else is measured.

Library conventions used here (``model/dh_param.py``): an MDH row is ``[d, alpha, a]`` with link
matrix ``Tx(a) @ Rx(alpha) @ Tz(d)``, and the chain is ``M_1 Rz(q1) M_2 Rz(q2) M_3 Rz(q3) @ tool``
(the joint rotation follows its link matrix), with the tool transform passed to
``RobotModelNumpy.build(..., ee=...)`` -- it is *not* read from the config mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import IkType
from model.dh_param import Rx, Tx, Tz
from model.robot_model_numpy import RobotModelNumpy

#: The review's example geometry, in the library's MDH convention ``[d, alpha, a]``:
#: a base rotation, a shoulder offset ``r2`` at height ``d2``, then two more mutually
#: orthogonal joints with offsets ``d3`` and ``d4``.  Row 4 is the tool, not a joint.
DEFAULT_ROWS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, np.pi / 2, 1.0),
    (2.0, np.pi / 2, 0.0),
)
#: Tool row ``[d4, alpha, 0]``: the tool point sits at distance ``d4`` from the last joint,
#: perpendicular to its axis (``alpha = 90 deg``), which is what lets joint 3 move it.
DEFAULT_TOOL: tuple[tuple[float, float, float], ...] = ((1.5, np.pi / 2, 0.0),)


def link_matrices(rows) -> list[np.ndarray]:
    """Turn MDH rows ``[d, alpha, a]`` into the library's per-joint link matrices."""
    return [Tx(a) @ Rx(alpha) @ Tz(d) for d, alpha, a in rows]


def build(
    rows=DEFAULT_ROWS,
    tool=DEFAULT_TOOL,
    *,
    limits: float = np.pi,
) -> RobotModelNumpy:
    """Build the position-task 3R model.

    Args:
        rows: Three MDH rows ``[d, alpha, a]``.
        tool: One or more MDH rows composing the tool transform.
        limits: Symmetric joint bound, radians; the joints are revolute, so this only fixes the
            branch of ``q`` that is reported.

    Returns:
        The model, with the tool transform installed as the end-effector offset.
    """
    matrices = link_matrices(rows)
    tool_matrix = np.eye(4)
    for d, alpha, a in tool:
        tool_matrix = tool_matrix @ (Tx(a) @ Rx(alpha) @ Tz(d))
    config = {
        "name": "cuspidal3r",
        "param_type": "mat",
        "param": matrices,
        "lower": [-limits] * 3,
        "upper": [limits] * 3,
        "joint_unit": "radian",
        "dh_unit": "meter",
    }
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config, base=np.eye(4), ee=tool_matrix)
    return model


class PositionTask:
    """A model seen as a position-only task: ``3`` outputs instead of six.

    ``study.chamber`` and ``study.ik_structure.sigma_min`` only need ``jacobian`` and the joint
    bounds, so wrapping the model this way reuses the whole clearance-walk machinery for the map
    the 3R theory is about.  The bounds are widened to several turns so a walk can wind the
    joints instead of being clipped at a limit.
    """

    def __init__(self, model: RobotModelNumpy, *, turns: float = 3.0):
        self.model = model
        self.num_dof = model.num_dof
        self.lower_bounds = np.full(model.num_dof, -turns * np.pi)
        self.upper_bounds = np.full(model.num_dof, turns * np.pi)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """The ``(3, 3)`` position Jacobian."""
        return np.asarray(self.model.jacobian(np.asarray(q, dtype=float)), dtype=float)[:3, :]

    def fk(self, q: np.ndarray) -> np.ndarray:
        """The 4x4 pose; only its translation is the task."""
        return self.model.fk(np.asarray(q, dtype=float))


def position(model: RobotModelNumpy, q: np.ndarray) -> np.ndarray:
    """The tool point, metres, in the base frame."""
    return np.asarray(model.fk(np.asarray(q, dtype=float)), dtype=float)[:3, 3]


def position_jacobian(model: RobotModelNumpy, q: np.ndarray) -> np.ndarray:
    """The ``(3, 3)`` Jacobian of :func:`position`."""
    return np.asarray(model.jacobian(np.asarray(q, dtype=float)), dtype=float)[:3, :]


def det_position(model: RobotModelNumpy, q: np.ndarray) -> float:
    """``det`` of the position Jacobian: zero exactly on the 3R's singular set."""
    return float(np.linalg.det(position_jacobian(model, q)))


def sigma_min_position(model: RobotModelNumpy, q: np.ndarray) -> float:
    """Smallest singular value of the position Jacobian."""
    return float(np.linalg.svd(position_jacobian(model, q), compute_uv=False)[-1])


def rank_report(model: RobotModelNumpy, *, samples: int = 200, seed: int = 0) -> dict[str, float]:
    """Assert-worthy summary of the position Jacobian's rank over random configurations.

    exp46 shipped a family whose Jacobian had rank 2 everywhere; charging an arm with
    "controlling" three coordinates needs this check first.
    """
    rng = np.random.default_rng(seed)
    ranks = []
    worst_sigma = np.inf
    for _ in range(samples):
        q = rng.uniform(-np.pi, np.pi, model.num_dof)
        jacobian = position_jacobian(model, q)
        ranks.append(int(np.linalg.matrix_rank(jacobian, tol=1e-9)))
        worst_sigma = min(worst_sigma, float(np.linalg.svd(jacobian, compute_uv=False)[-1]))
    return {
        "samples": samples,
        "rank3_fraction": ranks.count(3) / len(ranks),
        "min_rank": min(ranks),
        "worst_sigma_min": worst_sigma,
    }


def solve_position(
    model: RobotModelNumpy, target: np.ndarray, q0: np.ndarray, *, iterations: int = 80,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Gauss-Newton on the three position equations from a seed.

    Args:
        model: The arm.
        target: Tool point, metres.
        q0: Starting configuration, radians.
        iterations: Maximum iterations.
        tolerance: Position residual that counts as converged.

    Returns:
        ``(q, residual)`` -- the residual is the Euclidean position error in metres.
    """
    target = np.asarray(target, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    error = target - position(model, q)
    for _ in range(iterations):
        residual = float(np.linalg.norm(error))
        if residual < tolerance:
            break
        jacobian = position_jacobian(model, q)
        normal = jacobian @ jacobian.T + 1e-12 * np.eye(3)
        q = q + jacobian.T @ np.linalg.solve(normal, error)
        error = target - position(model, q)
    return q, float(np.linalg.norm(target - position(model, q)))


def canonical(q: np.ndarray) -> np.ndarray:
    """Wrap every joint into ``[-pi, pi)``."""
    return (np.asarray(q, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def torus_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two configurations on the joint torus, radians."""
    delta = canonical(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return float(np.linalg.norm(delta))


@dataclass
class PositionFiber:
    """A deduplicated set of position-IK solutions at one target."""

    target: np.ndarray
    solutions: list[np.ndarray]
    worst_residual: float

    def __len__(self) -> int:
        return len(self.solutions)


def census_position(
    model: RobotModelNumpy, target: np.ndarray, *, seeds: int = 300,
    rng: np.random.Generator | None = None, tolerance: float = 1e-9, merge: float = 1e-3,
) -> PositionFiber:
    """Multi-start census of the position-IK fibre.

    Args:
        model: The arm.
        target: Tool point, metres.
        seeds: Random restarts.
        rng: Generator; a fresh one seeded at 0 when omitted.
        tolerance: Position residual below which a restart counts as converged.
        merge: Torus distance below which two solutions are one.

    Returns:
        The :class:`PositionFiber`; solutions are canonical and pairwise ``merge`` apart.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    solutions: list[np.ndarray] = []
    worst = 0.0
    for _ in range(seeds):
        q, residual = solve_position(model, target, rng.uniform(-np.pi, np.pi, model.num_dof))
        if residual > tolerance:
            continue
        if any(torus_distance(q, other) < merge for other in solutions):
            continue
        solutions.append(canonical(q))
        worst = max(worst, residual)
    return PositionFiber(target=np.asarray(target, dtype=float), solutions=solutions,
                         worst_residual=worst)


def aspect_labels(
    model: RobotModelNumpy, *, grid: int = 400, threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Label the connected components of ``T^2 \\ Sigma`` on the ``(q2, q3)`` torus.

    The 3R's singular set does not depend on ``q1`` (the base rotation is a symmetry of the
    determinant: rotating the whole arm about the base axis cannot change whether it is
    singular), so the aspects of this arm are read off a two-dimensional torus -- which makes the
    partition *exact* rather than a budget-limited walk: no planner, no random seeds, no waypoints.

    Args:
        model: The arm.
        grid: Samples per joint; the grid is treated as a torus (neighbours wrap).
        threshold: ``|det J|`` below which a sample is declared singular and excluded.

    Returns:
        ``(labels, clearance, singular_fraction)``.  ``labels[i, j]`` is the component index of the
        sample ``(q2, q3) = (angles[i], angles[j])``, or ``-1`` where the sample is singular;
        ``clearance[i, j]`` is ``sigma_min`` there, for clearance-aware routing.
    """
    angles = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    values = np.empty((grid, grid))
    clearance = np.empty((grid, grid))
    for i, q2 in enumerate(angles):
        for j, q3 in enumerate(angles):
            point = np.array([0.0, q2, q3])
            clearance[i, j] = sigma_min_position(model, point)
            values[i, j] = abs(det_position(model, point))
    free = values > threshold
    labels = np.full((grid, grid), -1, dtype=int)
    component = 0
    for start in zip(*np.nonzero(free)):
        if labels[start] >= 0:
            continue
        component += 1
        stack = [start]
        labels[start] = component
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = (i + di) % grid, (j + dj) % grid
                if free[ni, nj] and labels[ni, nj] < 0:
                    labels[ni, nj] = component
                    stack.append((ni, nj))
    return labels, clearance, float(1.0 - free.mean())


def label_of(labels: np.ndarray, q: np.ndarray, *, grid: int | None = None) -> int:
    """Component label at a configuration, read from an :func:`aspect_labels` grid."""
    grid = labels.shape[0] if grid is None else grid
    angles = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    q2, q3 = canonical(np.asarray(q, dtype=float))[1:]
    i = int(np.argmin(np.abs(((angles - q2 + np.pi) % (2 * np.pi)) - np.pi)))
    j = int(np.argmin(np.abs(((angles - q3 + np.pi) % (2 * np.pi)) - np.pi)))
    return int(labels[i, j])


def aspect_path(
    model: RobotModelNumpy, labels: np.ndarray, q_from: np.ndarray, q_to: np.ndarray, *,
    clearance: np.ndarray | None = None, delta: float = 0.0, substeps: int = 24,
) -> list[np.ndarray] | None:
    """A witnessed path between two configurations inside one aspect, from the label grid.

    The greedy clearance walk used elsewhere in this study is a potential-field planner: it stalls
    on the level set, needs a waypoint budget, and its failures prove nothing.  On the 3R the
    aspects are *labelled* (:func:`aspect_labels`), so a path inside an aspect is a shortest path
    on a grid graph -- exact, budget-free, and reproducible.  The returned path is the grid path
    subdivided, and the caller is expected to verify ``sigma_min`` along it densely (the grid only
    guarantees that the nodes are clear).

    Args:
        model: The arm.
        labels: Grid from :func:`aspect_labels`.
        q_from: Start configuration, radians.
        q_to: End configuration, radians; must carry the same label.
        clearance: ``sigma_min`` grid from :func:`aspect_labels`; when given, the route is chosen by
            Dijkstra with an edge cost that grows as the clearance shrinks, and nodes below
            ``delta`` are excluded, so the path keeps a margin instead of hugging the singular set.
        delta: Clearance the route must keep at every node.
        substeps: Samples per grid edge in the returned path.

    Returns:
        The path as a list of configurations from ``q_from`` to ``q_to``, or ``None`` when the
        endpoints are in different components (or one of them is labelled singular).
    """
    import heapq

    grid = labels.shape[0]
    angles = np.linspace(-np.pi, np.pi, grid, endpoint=False)

    def cell(q: np.ndarray) -> tuple[int, int]:
        q2, q3 = canonical(np.asarray(q, dtype=float))[1:]
        i = int(np.argmin(np.abs(((angles - q2 + np.pi) % (2 * np.pi)) - np.pi)))
        j = int(np.argmin(np.abs(((angles - q3 + np.pi) % (2 * np.pi)) - np.pi)))
        return i, j

    start, goal = cell(q_from), cell(q_to)
    if labels[start] < 0 or labels[start] != labels[goal]:
        return None
    if clearance is not None and (clearance[start] < delta or clearance[goal] < delta):
        return None
    previous: dict[tuple[int, int], tuple[int, int]] = {start: start}
    distance = {start: 0.0}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost > distance.get(node, float("inf")):
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = ((node[0] + di) % grid, (node[1] + dj) % grid)
            if labels[neighbour] != labels[start]:
                continue
            step = 1.0
            if clearance is not None:
                if clearance[neighbour] < delta:
                    continue
                step += (0.1 / max(clearance[neighbour], 1e-12)) ** 2
            if cost + step < distance.get(neighbour, float("inf")):
                distance[neighbour] = cost + step
                previous[neighbour] = node
                heapq.heappush(queue, (cost + step, neighbour))
    if goal not in previous:
        return None
    chain = [goal]
    while chain[-1] != start:
        chain.append(previous[chain[-1]])
    chain.reverse()
    start_q = canonical(q_from)
    end_q = canonical(q_to)
    # the aspect structure does not depend on q1 (the determinant factorises as -rho * det Dg with
    # rho the tool's distance from the base axis), so q1 is carried along the shortest way
    # independently of the (q2, q3) route.
    turn = shortest_step(np.array([start_q[0]]), np.array([end_q[0]]))[0]
    path: list[np.ndarray] = [start_q]
    travelled = 0
    total = len(chain) - 1
    for a, b in zip(chain, chain[1:]):
        node_a = np.array([0.0, angles[a[0]], angles[a[1]]])
        node_b = np.array([0.0, angles[b[0]], angles[b[1]]])
        step = shortest_step(node_a, node_b)
        for k in range(1, substeps + 1):
            travelled += 1
            point = node_a + (k / substeps) * step
            point[0] = start_q[0] + (travelled / (total * substeps)) * turn
            path.append(point)
    path[0] = start_q
    path[-1] = end_q
    return path


def shortest_step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest displacement from ``a`` to ``b`` on the joint torus."""
    return canonical(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))


__all__ = [
    "DEFAULT_ROWS",
    "DEFAULT_TOOL",
    "PositionFiber",
    "PositionTask",
    "aspect_labels",
    "aspect_path",
    "build",
    "canonical",
    "census_position",
    "det_position",
    "label_of",
    "link_matrices",
    "position",
    "position_jacobian",
    "rank_report",
    "shortest_step",
    "sigma_min_position",
    "solve_position",
    "torus_distance",
]
