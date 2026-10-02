"""Homotopy continuation of IK solutions as the DH parameters move.

Finding 2's method: with ``FK(dh, q) = T``, a solution can be *carried* while the parameters move,
by predicting with ``dq = -pinv(J_q) J_dh dh`` and correcting with Newton
(`study/exp11_dh_continuation.py` measured the predictor as second order and the corrector as the
thing that makes it usable).  Here that becomes a tracker with a step controller, because the
measured failure mode is the neighbourhood of the singular set: at ``sigma_min ~ 3e-3`` the
predictor produced a 2.2 rad step and a 0.70 m residual.

The tracker is deliberately simple and observable: every accepted step is corrected to a tolerance,
the step size shrinks when the clearance drops or the corrector struggles and grows again when the
path is easy, and the whole trace (``sigma_min``, step size, residual) is kept so a failure can be
read rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from study.arm_geometry import rot_z


def fk_links(links: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    """Forward kinematics from an explicit list of link transforms."""
    pose = np.eye(4)
    for index, link in enumerate(links):
        pose = pose @ link @ rot_z(float(q[index]))
    return pose


def se3_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The library's SE(3) error from ``current`` to ``target``, as a 6-vector."""
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


def jacobian_q(links: list[np.ndarray], q: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Finite-difference Jacobian of the pose with respect to ``q`` (6 x n)."""
    jacobian = np.zeros((6, len(q)))
    for index in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        jacobian[:, index] = se3_error(fk_links(links, minus), fk_links(links, plus)) / (2 * h)
    return jacobian


def sigma_min_links(links: list[np.ndarray], q: np.ndarray) -> float:
    """Smallest singular value of the pose Jacobian, computed from the link transforms.

    ``study.ik_structure.sigma_min`` needs a model; the tracker moves the link transforms itself,
    so it measures the clearance the same way the model would.
    """
    return float(np.linalg.svd(jacobian_q(links, q), compute_uv=False)[-1])


def correct(
    links: list[np.ndarray], q: np.ndarray, target: np.ndarray, *, iterations: int = 8,
    tolerance: float = 1e-11, damping: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Gauss-Newton correction towards ``target``; returns the configuration and its residual."""
    q = np.asarray(q, dtype=float).copy()
    error = se3_error(fk_links(links, q), target)
    for _ in range(iterations):
        residual = float(np.linalg.norm(error))
        if residual < tolerance:
            break
        jacobian = jacobian_q(links, q)
        normal = jacobian @ jacobian.T + damping * np.eye(6)
        q = q + jacobian.T @ np.linalg.solve(normal, error)
        error = se3_error(fk_links(links, q), target)
    return q, float(np.linalg.norm(error))


@dataclass
class Track:
    """One solution's journey along a parameter path."""

    start: np.ndarray
    q: np.ndarray
    finished: bool
    #: Parameter value reached, in ``[0, 1]``.
    progress: float
    residual: float
    steps: int
    #: Smallest clearance seen and where.
    min_sigma: float
    min_sigma_at: float
    #: Step sizes and clearances, for inspection after a failure.
    history: list[tuple[float, float, float]] = field(default_factory=list)
    reason: str = ""


def track(
    path: Callable[[float], list[np.ndarray]],
    target: np.ndarray,
    start: np.ndarray,
    *,
    ds: float = 0.02,
    min_ds: float = 1e-5,
    max_ds: float = 0.05,
    sigma_floor: float = 2e-3,
    tolerance: float = 1e-10,
    max_steps: int = 4000,
) -> Track:
    """Carry one solution from ``s = 0`` to ``s = 1`` along a parameter path.

    Args:
        path: ``s -> link transforms``, with ``s = 0`` the solvable arm and ``s = 1`` the target.
        target: The pose to keep solving for (fixed along the path).
        start: A solution at ``s = 0``.
        ds: Initial step in ``s``.
        min_ds: Smallest step before the tracker gives up.
        max_ds: Largest step.
        sigma_floor: Clearance below which the step is halved, because the predictor is not
            trustworthy there (measured in :mod:`study.exp11_dh_continuation`).
        tolerance: Pose residual the corrector must reach for a step to count.
        max_steps: Step budget.

    Returns:
        The :class:`Track`.
    """
    s = 0.0
    q = np.asarray(start, dtype=float).copy()
    links = path(s)
    q, residual = correct(links, q, target, tolerance=tolerance)
    result = Track(
        start=np.asarray(start, dtype=float).copy(),
        q=q.copy(),
        finished=False,
        progress=s,
        residual=residual,
        steps=0,
        min_sigma=sigma_min_links(links, q),
        min_sigma_at=s,
    )
    if residual > tolerance:
        result.reason = "the start is not a solution"
        return result

    step = ds
    while s < 1.0 and result.steps < max_steps:
        step = min(step, 1.0 - s)
        probe = min(step, 1.0 - s)
        next_links = path(s + probe)
        # predictor: dq from the parameter derivative, through the same pseudo-inverse
        jacobian = jacobian_q(links, q)
        derivative = se3_error(fk_links(links, q), fk_links(next_links, q)) / probe
        step_q = -np.linalg.pinv(jacobian) @ (derivative * probe)
        candidate_links = next_links
        candidate_q, residual = correct(candidate_links, q + step_q, target, tolerance=tolerance)
        clearance = sigma_min_links(candidate_links, candidate_q)
        if residual > tolerance or clearance < sigma_floor:
            step *= 0.5
            if step < min_ds:
                result.reason = (
                    f"step under {min_ds:g} at s = {s:.3f} "
                    f"(residual {residual:.1e}, clearance {clearance:.1e})"
                )
                return result
            continue
        s += probe
        q = candidate_q
        links = candidate_links
        result.steps += 1
        result.progress = s
        result.q = q.copy()
        result.residual = residual
        if clearance < result.min_sigma:
            result.min_sigma = clearance
            result.min_sigma_at = s
        result.history.append((s, step, clearance))
        if step < max_ds:
            step = min(step * 1.5, max_ds)
    result.finished = s >= 1.0
    result.reason = result.reason or ("arrived" if result.finished else "step budget exhausted")
    return result


__all__ = [
    "Track",
    "correct",
    "fk_links",
    "jacobian_q",
    "se3_error",
    "sigma_min_links",
    "track",
]
