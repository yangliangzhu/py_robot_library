#!/usr/bin/env python3
"""exp18: the redundant case -- chamber branches versus self-motion components

On a 6-DOF arm the two notions people mean by "branch" coincide, so nobody has to distinguish them:
two solutions of a pose are on the same sheet of ``FK: Q_reg -> W`` exactly when a path between them
can avoid the singular set.  On a 7-DOF arm they come apart, and this experiment measures it:

* the **fibre component** -- solutions joined by a *self-motion* path, i.e. a path along which the
  pose does not change at all.  Found by integrating the Jacobian's null-space direction with a
  clearance-controlled step, in both directions, until the path closes or hits a boundary.
* the **chamber** -- solutions joined by *some* path in ``Q\\Sigma``, pose changing freely.  Tested
  constructively, by the witnessed clearance walk of `study/chamber.py`, since for a 6x7 Jacobian
  there is no determinant to sign.

If a chamber ever contains two different fibre components, then "same branch" (same sheet) does not
imply "connected without changing the pose", and the empirical criterion's vocabulary has to be
stated in terms of chambers, not of self-motion.

Run::

    python3 -m study.exp18_franka_redundant --robot franka --seeds 400
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from model import IkType, ModelFactory
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair


def within_limits(model, q: np.ndarray) -> bool:
    """Whether a configuration respects the model's joint limits."""
    return bool(np.all(q >= np.asarray(model.lower_bounds)) and np.all(q <= np.asarray(model.upper_bounds)))


def null_direction(model, q: np.ndarray) -> np.ndarray | None:
    """The unit null-space direction of the pose Jacobian at ``q`` (``None`` if not redundant)."""
    jacobian = np.asarray(model.jacobian(q), dtype=float)
    if jacobian.shape[0] >= jacobian.shape[1]:
        return None
    _u, _s, vh = np.linalg.svd(jacobian)
    direction = vh[-1]
    return direction / np.linalg.norm(direction)


def correct_pose(model, q: np.ndarray, target: np.ndarray, *, iterations: int = 4) -> tuple:
    """Project ``q`` back onto the fibre of ``target``.

    Integrating the null-space direction moves along the self-motion manifold only to first order:
    the pose error accumulates, and an uncorrected trace is not a self-motion path at all (a first
    version of this experiment produced 27 "components" of length 30 rad that never closed -- a
    drift, not a manifold).  Correcting after every step is what makes the trace mean what it says,
    and the residual it reaches is reported as the evidence.

    Returns:
        ``(q, pose_error)``.
    """
    for _ in range(iterations):
        pose = model.fk(q)
        error = _se3_error(pose, target)
        if float(np.linalg.norm(error)) < 1e-12:
            break
        jacobian = np.asarray(model.jacobian(q), dtype=float)
        q = q + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-9 * np.eye(6), error)
    return q, float(np.linalg.norm(_se3_error(model.fk(q), target)))


def _se3_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The library's SE(3) error, as a 6-vector."""
    error = np.zeros(6)
    error[:3] = target[:3, 3] - current[:3, 3]
    spin = target[:3, :3] @ current[:3, :3].T
    cosine = float(np.clip(0.5 * (np.trace(spin) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return error
    axis = np.array(
        [spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0], spin[1, 0] - spin[0, 1]]
    )
    error[3:] = axis / (2.0 * np.sin(angle)) * angle
    return error


@dataclass
class Component:
    """One traced self-motion component."""

    points: list[np.ndarray] = field(default_factory=list)
    closed: bool = False
    length: float = 0.0
    min_sigma: float = float("inf")
    end_reason: str = ""
    worst_pose_error: float = 0.0
    total_length: float = 0.0


def trace_component(
    model, start: np.ndarray, target: np.ndarray, *, delta: float, step: float = 0.05,
    max_steps: int = 600,
) -> Component:
    """Follow the self-motion manifold from ``start`` in both directions, correcting as it goes."""
    component = Component()
    for sense in (1.0, -1.0):
        q = np.asarray(start, dtype=float).copy()
        component.points.append(q.copy())
        travelled = 0.0
        for _ in range(max_steps):
            direction = null_direction(model, q)
            if direction is None:
                component.end_reason = "not redundant"
                break
            moved, pose_error = correct_pose(model, q + sense * step * direction, target)
            probe = step
            while (iks.sigma_min(model, moved) < delta or not within_limits(model, moved)) and probe > 1e-4:
                probe *= 0.5
                moved, pose_error = correct_pose(model, q + sense * probe * direction, target)
            component.worst_pose_error = max(component.worst_pose_error, pose_error)
            if not within_limits(model, moved):
                component.end_reason = "joint limit"
                break
            if iks.sigma_min(model, moved) < delta or pose_error > 1e-9:
                component.end_reason = (
                    f"singular boundary: clearance {iks.sigma_min(model, q):.2e}, "
                    f"pose error {pose_error:.1e}"
                )
                break
            travelled += float(np.linalg.norm(moved - q))
            q = moved
            component.points.append(q.copy())
            component.min_sigma = min(component.min_sigma, iks.sigma_min(model, q))
            if travelled > 2 * step and float(np.linalg.norm(q - start)) < step:
                component.closed = True
                component.end_reason = component.end_reason or "closed"
                break
        component.length = max(component.length, travelled)
        component.total_length += travelled
    return component


def main() -> int:
    """Run the redundant-arm experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--delta", type=float, default=5e-3)
    parser.add_argument("--cluster", type=float, default=2e-2, help="component membership tolerance")
    parser.add_argument("--pairs", type=int, default=6, help="cross-component pairs to walk")
    parser.add_argument("--steps", type=int, default=600,
                        help="integration steps per direction; a component is only *counted* when "
                             "its trace closes or stops at a real boundary, so a small budget "
                             "over-counts (the recorded lower bound 13 came from 600)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create(args.robot, backend="casadi", ik_type=IkType.IK_STANDARD)
    circular = iks.circular_joints(model)
    rng = np.random.default_rng(args.seed)
    target, _ = iks.random_reachable_target(model, rng)
    fiber = census(model, target, solver=make_lm_solver(model), seeds=args.seeds, rng=rng)
    solutions = [q for q in fiber.solutions if iks.sigma_min(model, q) > args.delta]
    print(f"{args.robot}: {model.num_dof} axes, {len(solutions)} solutions "
          f"({fiber.converged}/{fiber.seeds} seeds converged), cluster tolerance {args.cluster:g}")

    # group solutions into self-motion components by tracing from unassigned ones
    assigned = [False] * len(solutions)
    components: list[tuple[Component, list[int]]] = []
    for index, q in enumerate(solutions):
        if assigned[index]:
            continue
        component = trace_component(model, q, target, delta=args.delta,
                                      max_steps=args.steps)
        members = [
            other_index
            for other_index, other in enumerate(solutions)
            if not assigned[other_index]
            and any(
                np.linalg.norm(
                    (other - point + np.pi) % (2 * np.pi) - np.pi
                )
                < args.cluster
                for point in component.points
            )
        ]
        for member in members:
            assigned[member] = True
        components.append((component, members))
    print(f"self-motion components: {len(components)} "
          f"(sizes {[len(members) for _c, members in components]})")
    for number, (component, members) in enumerate(components[:12]):
        print(f"  component {number}: {len(members)} solutions, "
              f"{len(component.points)} traced points, length one way {component.length:.2f} rad, "
              f"both ways {component.total_length:.2f} rad, "
              f"closed {component.closed}, min clearance "
              f"{component.min_sigma if component.min_sigma < np.inf else float('nan'):.2e}, "
              f"worst pose drift {component.worst_pose_error:.1e}, "
              f"end {component.end_reason}")

    reasons: dict[str, int] = {}
    for component, _members in components:
        key = ("closed" if component.closed else component.end_reason or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    complete = sum(1 for component, _members in components
                   if component.closed or "limit" in component.end_reason
                   or "singular" in component.end_reason)
    print(f"component count {len(components)} with a {args.steps}-step budget; "
          f"{complete} of them closed or stopped at a real boundary (a truncated trace would "
          f"over-count); end reasons {reasons}")
    if complete < len(components):
        print("  NOT a count: a trace that ran out of budget can leave part of its component to be "
              "reported as another one -- raise --steps and rerun")

    # are different fibre components ever in one chamber?
    if len(components) > 1:
        pairs_tested = linked = 0
        for first in range(len(components)):
            for second in range(first + 1, len(components)):
                if pairs_tested >= args.pairs:
                    break
                q_a = solutions[components[first][1][0]]
                q_b = solutions[components[second][1][0]]
                pairs_tested += 1
                if resolve_pair(model, q_a, q_b, delta=args.delta, circular=circular).connected:
                    linked += 1
        print(f"cross-component pairs walked: {pairs_tested}, connected in Q\\Sigma: {linked} "
              f"(a connection means one chamber holds two fibre components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
