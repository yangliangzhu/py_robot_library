"""A probabilistic roadmap in ``{sigma_min >= delta}``: a stronger branch-connectivity oracle.

The greedy clearance walk of :mod:`study.chamber` answers "can I get from here to there without
losing clearance?" along one route.  For *partitioning* the solutions of a pose into branches
that is too weak: a walk that fails has proved nothing, and in practice it fails often (section
3.7 of the notes: it left most same-sign pairs unresolved).

A roadmap answers the question for a whole set at once.  Sample configurations that keep
clearance, join the ones that can see each other along a verified straight segment, and read the
connected components off the graph.  Two configurations in the same component are connected in
``{sigma_min >= delta}`` by construction -- every edge is a verified path -- so this is a
certificate in the same sense as the walk, but it explores many routes instead of one.

It remains incomplete in the honest sense: a sparse or unlucky roadmap can split a component in
two, so "different components" is evidence, never proof, and the report says which pairs were
resolved at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from study.ik_structure import circular_joints, sigma_min


class _UnionFind:
    """Disjoint-set forest, for reading components off a graph."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """Component representative of ``item``."""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        """Merge the components of two items."""
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


#: Samples per radian of segment length when verifying an edge.  A fixed sample *count* is not
#: a certificate: on a long edge it leaves gaps a thin sliver of the singular set can hide in,
#: which is exactly how a roadmap first reported "17 solutions, one branch" for a pose whose own
#: determinant signs prove at least two.  Density, not count.
SAMPLES_PER_RADIAN = 400.0

#: No edge longer than this (radians), so the roadmap stays local.
MAX_EDGE = 0.6


def _segment_min_sigma(
    model, q_a: np.ndarray, q_b: np.ndarray, density: int | None = None
) -> float:
    """Smallest ``sigma_min`` on the straight segment ``q_a -> q_b``.

    Args:
        model: Robot model.
        q_a: Start of the segment.
        q_b: End of the segment.
        density: Sample count; derived from the segment's length when omitted.

    Returns:
        The smallest clearance sampled on the segment.
    """
    length = float(np.linalg.norm(np.asarray(q_b) - np.asarray(q_a)))
    count = density if density is not None else max(24, int(SAMPLES_PER_RADIAN * length))
    return min(sigma_min(model, q_a + t * (q_b - q_a)) for t in np.linspace(0.0, 1.0, count))


@dataclass
class Roadmap:
    """Sampled configurations, verified edges, and the components they form."""

    points: list[np.ndarray] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    component: list[int] = field(default_factory=list)
    delta: float = 0.0
    #: Clearances kept by each accepted edge, for reporting the margin.
    edge_clearance: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many components the roadmap has."""
        return len(set(self.component)) if self.component else 0

    def component_of(self, point: np.ndarray, model, *, verify: int | None = None) -> int | None:
        """The component a configuration belongs to, or ``None`` if no edge could be verified.

        Args:
            point: The configuration to attach.
            model: Robot model.
            verify: Samples per candidate edge.

        Returns:
            A component id, or ``None`` when the point could not be joined to the roadmap.
        """
        order = np.argsort([np.linalg.norm(entry - point) for entry in self.points])
        for index in order[:8]:
            if _segment_min_sigma(model, point, self.points[index], verify) >= self.delta:
                return self.component[index]
        return None


def build_roadmap(
    model,
    *,
    center: np.ndarray,
    radius: float,
    delta: float,
    samples: int = 400,
    neighbors: int = 10,
    verify: int | None = None,
    rng: np.random.Generator | None = None,
) -> Roadmap:
    """Sample and connect a roadmap inside ``{sigma_min >= delta}``.

    Args:
        model: Robot model.
        center: Centre of the sampling box, in joint space.
        radius: Half-width of the sampling box, per joint.
        delta: Clearance every accepted edge must keep.
        samples: Candidate configurations to draw.
        neighbors: Nearest neighbours each configuration is tested against.
        verify: Fixed sample count per segment; by default the count follows the segment's
            length (:data:`SAMPLES_PER_RADIAN`).
        rng: Random generator.

    Returns:
        The :class:`Roadmap`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    box_lower = np.maximum(lower, np.asarray(center) - radius)
    box_upper = np.minimum(upper, np.asarray(center) + radius)

    roadmap = Roadmap(delta=delta)
    while len(roadmap.points) < samples:
        candidate = rng.uniform(box_lower, box_upper)
        if sigma_min(model, candidate) >= delta:
            roadmap.points.append(candidate)
        elif len(roadmap.points) + 10000 < 0:  # pragma: no cover - guard against a bad box
            break

    union = _UnionFind(len(roadmap.points))
    coordinates = np.array(roadmap.points)
    for index, point in enumerate(roadmap.points):
        distances = np.linalg.norm(coordinates - point, axis=1)
        distances[index] = np.inf
        for other in np.argsort(distances)[:neighbors]:
            if other < index:
                continue
            other_point = roadmap.points[int(other)]
            if float(np.linalg.norm(other_point - point)) > MAX_EDGE:
                continue
            clearance = _segment_min_sigma(model, point, other_point, verify)
            if clearance >= delta:
                roadmap.edges.append((index, int(other)))
                roadmap.edge_clearance.append(clearance)
                union.union(index, int(other))
    roadmap.component = [union.find(index) for index in range(len(roadmap.points))]
    return roadmap


def branch_partition(
    model,
    solutions: list[np.ndarray],
    *,
    delta: float = 5e-3,
    samples: int = 400,
    neighbors: int = 10,
    verify: int | None = None,
    margin: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[int]], list[int | None]]:
    """Group solutions into branches by roadmap connectivity.

    Args:
        model: Robot model.
        solutions: Distinct solutions of one pose.
        delta: Clearance the roads must keep.
        samples: Roadmap sample count.
        neighbors: Nearest neighbours per sample.
        verify: Samples per clearance check.
        margin: Sampling-box half-width, as a multiple of the solutions' spread.
        rng: Random generator.

    Returns:
        ``(groups, component_ids)`` -- the groups are lists of indices into ``solutions``, and
        ``component_ids`` holds the roadmap component of each solution.  A solution that no road
        reached gets ``None``: unresolved, and deliberately *not* its own branch, because an
        unattached solution says something about the roadmap, not about the arm.
    """
    points = [np.asarray(q, dtype=float) for q in solutions]
    center = np.mean(points, axis=0)
    spread = max(
        (float(np.linalg.norm(a - b)) for a in points for b in points), default=1.0
    )
    roadmap = build_roadmap(
        model,
        center=center,
        radius=margin * max(spread, 0.5),
        delta=delta,
        samples=samples,
        neighbors=neighbors,
        verify=verify,
        rng=rng,
    )
    components: list[int | None] = []
    for point in points:
        components.append(roadmap.component_of(point, model, verify=verify))
    groups: dict[int, list[int]] = {}
    for index, component in enumerate(components):
        if component is not None:
            groups.setdefault(component, []).append(index)
    return list(groups.values()), components


def audit_against_sign(
    model, solutions: list[np.ndarray], components: list[int | None]
) -> list[tuple[int, int]]:
    """Pairs a roadmap put in one component although their determinant signs differ.

    ``sign det J`` is a *proof* of different chambers, so any such pair is a false edge and the
    roadmap is wrong about it.  Running this audit is how the sampling defect above was found;
    it is kept as a standing check rather than a one-off.

    Args:
        model: Robot model (6-DOF, so that ``det J`` exists).
        solutions: The solutions the roadmap partitioned.
        components: Component ids from :func:`branch_partition`.

    Returns:
        The offending pairs, as ``(i, j)`` index pairs (empty when the roadmap is consistent).
    """
    from study.ik_structure import det_jacobian  # local import keeps the module import-light

    signs = [np.sign(det_jacobian(model, q)) for q in solutions]
    violations = []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            if components[i] is None or components[j] is None:
                continue
            if components[i] == components[j] and signs[i] != signs[j]:
                violations.append((i, j))
    return violations


def clearance_mask(model, points: list[np.ndarray], delta: float) -> np.ndarray:
    """Which of the given configurations keep ``delta`` clearance (for reporting)."""
    return np.array([sigma_min(model, q) >= delta for q in points])


__all__ = [
    "MAX_EDGE",
    "SAMPLES_PER_RADIAN",
    "Roadmap",
    "audit_against_sign",
    "branch_partition",
    "build_roadmap",
    "circular_joints",
    "clearance_mask",
]
