"""Constructive chamber membership: can two configurations be joined while keeping clearance?

A crossing count gives a one-sided certificate.  An *odd* number of transversal crossings of
``det J = 0`` proves the endpoints are in different chambers, but an even number proves nothing:
the path may have left one chamber, wandered into another and come back, or it may have gone
somewhere else entirely.  Deciding "same chamber" needs a *path that is witnessed to stay inside
``Q \\ Sigma``*, and that is what this module builds:

* :func:`grad_sigma_min` -- the direction that most increases the clearance (numerically, since
  the library exposes no derivative of ``sigma_min``);
* :func:`clearance_path` -- a greedy path from ``q_from`` towards ``q_to`` that is allowed to
  keep moving only while ``sigma_min >= delta``, backing off and sliding along the level set
  when it is not.  A path that arrives with the clearance verified at every step is a
  certificate that the two configurations are in the same chamber.
* :func:`resolve_pair` -- the pair-level answer, with "unresolved" as an honest third outcome:
  a greedy planner that fails has not proved anything, because the chamber may be connected by
  a route this planner is too crude to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from study.ik_structure import circular_joints, shortest_representative, sigma_min


def grad_sigma_min(model, q: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of ``sigma_min`` at ``q``, as a unit vector.

    Args:
        model: Robot model.
        q: Joint positions in radians.
        h: Finite-difference step.

    Returns:
        A unit vector pointing towards increasing clearance; zeros if the gradient vanishes.
    """
    q = np.asarray(q, dtype=float)
    gradient = np.zeros_like(q)
    for index in range(q.size):
        step = np.zeros_like(q)
        step[index] = h
        gradient[index] = (sigma_min(model, q + step) - sigma_min(model, q - step)) / (2.0 * h)
    norm = float(np.linalg.norm(gradient))
    return gradient / norm if norm > 0 else gradient


@dataclass
class ClearanceWalk:
    """The outcome of trying to walk from one configuration to another without losing clearance."""

    connected: bool
    #: Smallest clearance actually visited (the certificate, when ``connected``).
    min_sigma: float
    steps: int
    reason: str = ""
    #: The path, sampled every step, for inspection.
    path: list[np.ndarray] = field(default_factory=list)
    #: Largest fraction of the way to the target the walk reached, in ``[0, 1]``.
    progress: float = 0.0


def clearance_path(
    model,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    delta: float = 5e-3,
    steps: int = 600,
    step_size: float = 0.05,
    circular: np.ndarray | None = None,
    repulsion: float = 2.0,
    verify_step: int = 16,
) -> ClearanceWalk:
    """Walk from ``q_from`` to ``q_to`` keeping ``sigma_min >= delta``.

    The rule is a potential-field one: aim at the target, and when the clearance would drop
    below ``delta``, add the clearance-increasing direction and shorten the step until the walk
    is admissible again.  Steps that cannot be made admissible are reported, not forced.

    Args:
        model: Robot model.
        q_from: Starting configuration, radians.
        q_to: Target configuration, radians.
        delta: Clearance the walk must keep, in the units of ``sigma_min``.
        steps: Maximum number of steps.
        step_size: Nominal step length in the configuration space.
        circular: Boolean mask of continuous joints; taken from the model when omitted.
        repulsion: Weight of the clearance gradient relative to the target direction.
        verify_step: Samples per accepted step used to verify that the *whole* step keeps the
            clearance.  Checking only the endpoints is not enough: a step can jump a thin
            sliver of the singular set, which would turn a walk into a false certificate.  The
            measured failure mode this closes: 2 of 16 certified-different pairs were
            "connected" by endpoint-only checks.

    Returns:
        The :class:`ClearanceWalk`.  ``connected`` means the walk arrived with the clearance
        held; anything else is unproven, not disproven.
    """
    circular = circular_joints(model) if circular is None else circular
    start = np.asarray(q_from, dtype=float)
    end = shortest_representative(start, np.asarray(q_to, dtype=float), circular)
    lower = np.asarray(model.lower_bounds, dtype=float)
    upper = np.asarray(model.upper_bounds, dtype=float)
    span = float(np.linalg.norm(end - start))
    # Arrival has to be much tighter than any deduplication tolerance used upstream: two
    # "solutions" a millimetre apart in joint space are one solution with numerical noise, and
    # certifying them as connected would be a false certificate.
    if span < 1e-4:
        walk = ClearanceWalk(connected=False, min_sigma=sigma_min(model, start), steps=0)
        walk.reason = f"endpoints coincide within 1e-4 (distance {span:.2e})"
        return walk
    # Two tolerances: the walk aims for ``near``, and the last stretch is finished by a
    # *verified* straight segment.  A certificate needs a witnessed path, not a step size; the
    # greedy walk alone stalls on the delta level set about a centimetre out, which is a
    # property of the planner and not of the chamber.
    near = 1e-2 * span
    tolerance = max(1e-6, 1e-6 * span)

    q = start.copy()
    walk = ClearanceWalk(connected=False, min_sigma=sigma_min(model, q), steps=0, path=[q.copy()])
    if walk.min_sigma < delta:
        walk.reason = "start below the clearance"
        return walk
    for step in range(steps):  # noqa: B007 - the step index is reported through walk.steps
        to_go = end - q
        distance = float(np.linalg.norm(to_go))
        walk.progress = max(walk.progress, 1.0 - distance / span if span > 0 else 1.0)
        if distance < near and _segment_clears(model, q, end, delta, circular):
            walk.path.append(end.copy())
            walk.min_sigma = min(walk.min_sigma, _segment_min_sigma(model, q, end, circular))
            walk.connected = True
            walk.steps = step
            walk.reason = "arrived (final segment verified)"
            walk.progress = 1.0
            return walk
        if distance < tolerance:
            walk.connected = True
            walk.steps = step
            walk.reason = "arrived"
            return walk
        direction = to_go / distance
        trial_step = step_size
        moved = False
        for _ in range(12):
            proposal = q + trial_step * direction
            if sigma_min(model, proposal) < delta:
                # add clearance-seeking motion and try a shorter step
                proposal = q + trial_step * (direction + repulsion * grad_sigma_min(model, q))
                trial_step *= 0.5
            proposal = np.clip(proposal, lower, upper)
            if sigma_min(model, proposal) < delta:
                trial_step *= 0.5
                continue
            step_min = _segment_min_sigma(model, q, proposal, circular, density=verify_step)
            if step_min < delta:
                trial_step *= 0.5
                continue
            q = proposal
            walk.min_sigma = min(walk.min_sigma, step_min)
            walk.path.append(q.copy())
            moved = True
            break
        if not moved:
            walk.reason = "no admissible step"
            walk.steps = step
            return walk
    walk.reason = "step budget exhausted"
    walk.steps = steps
    return walk


def resolve_pair(
    model,
    q_a: np.ndarray,
    q_b: np.ndarray,
    *,
    delta: float = 5e-3,
    **kwargs: object,
) -> ClearanceWalk:
    """Decide whether two configurations are connected inside ``{sigma_min >= delta}``.

    Args:
        model: Robot model.
        q_a: First configuration, radians.
        q_b: Second configuration, radians.
        delta: Clearance the walk must keep.
        **kwargs: Forwarded to :func:`clearance_path`.

    Returns:
        The :class:`ClearanceWalk`; ``connected`` is the certificate, anything else is open.
    """
    return clearance_path(model, q_a, q_b, delta=delta, **kwargs)


def _segment_min_sigma(
    model, q_a: np.ndarray, q_b: np.ndarray, circular: np.ndarray, density: int = 2000
) -> float:
    """Smallest ``sigma_min`` on the straight segment from ``q_a`` to ``q_b``, densely sampled."""
    end = shortest_representative(q_a, q_b, circular)
    return min(
        sigma_min(model, q_a + t * (end - q_a)) for t in np.linspace(0.0, 1.0, density)
    )


def _segment_clears(
    model, q_a: np.ndarray, q_b: np.ndarray, delta: float, circular: np.ndarray, density: int = 200
) -> bool:
    """Whether the straight segment keeps at least ``delta`` clearance (sampled)."""
    end = shortest_representative(q_a, q_b, circular)
    return all(
        sigma_min(model, q_a + t * (end - q_a)) >= delta
        for t in np.linspace(0.0, 1.0, density)
    )
