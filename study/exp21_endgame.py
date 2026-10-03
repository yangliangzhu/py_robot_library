#!/usr/bin/env python3
"""exp21: the endgame -- computing the multiple root instead of walking around it

exp16 showed the tracker's stalls are double roots (distances 1.5e-6 .. 9.4e-5 rad) and exp17 got
around them by scanning.  This experiment computes the multiple root itself, so the structure stops
being an observation:

1. **Refine the discriminant point.**  The stall is a configuration where the pose is right and the
   Jacobian is singular.  That is a square system in ``(q, s)``:

   ``FK(dh(s), q) - T = 0``   (6 equations)
   ``det J(q, s) = 0``        (1 equation, exact for a 6-DOF arm)

   Seven equations, seven unknowns, solved by Newton from the stall as the initial guess.  The
   parameter value it lands on is the fold ``s*`` and the configuration is where the two branches
   meet.

2. **Measure how the branches split.**  Near a simple fold the solutions behave like
   ``q(s) = q* + c (s* - s)^(1/2)`` -- a *square-root* splitting, i.e. **two** branches emerging from
   one double root.  The exponent is measurable: fit ``log|q(s) - q*|`` against ``log(s* - s)`` along
   the tracked branch as it approaches the fold.  A slope of 1/2 confirms the fold; a slope of 1/m
   would mean m branches splitting from one multiple root, which is the sharper form of the question
   this study started from.

Run::

    python3 -m study.exp21_endgame --pose-seed 0
"""

from __future__ import annotations

import argparse

import numpy as np

from study import homotopy, sr0
from study import ik_structure as iks
from study.exp11_dh_continuation import link_transforms
from study.exp13_homotopy_sr0_to_sr5 import path_at


def augmented(links: list[np.ndarray], q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``[FK(q) - T (6), det J (1)]`` -- zero exactly at a fold of the pose equations."""
    residual = np.zeros(7)
    residual[:6] = homotopy.se3_error(homotopy.fk_links(links, q), target)
    jacobian = homotopy.jacobian_q(links, q)
    residual[6] = float(np.linalg.det(jacobian[:6, :6]))
    return residual


def refine_fold(
    path, target: np.ndarray, q_start: np.ndarray, s_start: float, *, iterations: int = 40,
    tolerance: float = 1e-12,
) -> tuple[float, np.ndarray, float]:
    """Newton on the augmented system; returns ``(s, q, residual)``."""
    q = np.asarray(q_start, dtype=float).copy()
    s = float(s_start)
    for _ in range(iterations):
        residual = augmented(path(s), q, target)
        size = float(np.linalg.norm(residual))
        if size < tolerance:
            break
        jacobian = np.zeros((7, 7))
        for column in range(6):
            step = np.zeros(6)
            step[column] = 1e-7
            jacobian[:, column] = (
                augmented(path(s), q + step, target) - augmented(path(s), q - step, target)
            ) / 2e-7
        jacobian[:, 6] = (
            augmented(path(s + 1e-7), q, target) - augmented(path(s - 1e-7), q, target)
        ) / 2e-7
        update = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        # Keep the parameter near the path but NOT hard-clamped to [0, 1]: kimi measured that the
        # clamp silently swallows folds at the boundary (0.9911, 0.9936 refined to s = 1.000000 with
        # residuals 7.7e-4 and 1.3) and makes any fold beyond the end (measured: a real fold at
        # s = 1.047164) impossible to refine at all.  The wider window keeps the protection against
        # wild steps while letting the discriminant be found where it actually is.
        s = float(np.clip(s + update[6], -0.3, 1.3))
        q = q + update[:6]
    return s, q, float(np.linalg.norm(augmented(path(s), q, target)))


def approach(path, target: np.ndarray, q_seed: np.ndarray, s_stop: float, *, step: float = 0.02,
             include_rejected: bool = False):
    """Walk from ``s = 0`` towards ``s_stop`` and record ``(s, q)`` at every accepted step.

    Args:
        include_rejected: When the walk gives up (the step has halved below 1e-5 with the residual or
            the clearance still bad), append the rejected probe point as well.  Without it a stalled
            branch contributes only its last *accepted* configuration, which can still be far from the
            singular set -- measured: a fold census built on accepted points alone found zero folds on
            poses where a survival count proves branches die.
    """
    s = 0.0
    q, _ = homotopy.correct(path(0.0), q_seed, target)
    trace = [(0.0, q.copy())]
    while s < s_stop:
        probe = min(step, s_stop - s)
        links = path(s + probe)
        predictor = homotopy.jacobian_q(path(s), q)
        derivative = (
            homotopy.se3_error(homotopy.fk_links(path(s), q), homotopy.fk_links(links, q)) / probe
        )
        dq = -np.linalg.pinv(predictor) @ (derivative * probe)
        candidate, residual = homotopy.correct(links, q + dq, target, tolerance=1e-11)
        if residual > 1e-11 or homotopy.sigma_min_links(links, candidate) < 2e-3:
            step *= 0.5
            if step < 1e-5:
                if include_rejected:
                    trace.append((s + probe, candidate.copy()))
                break
            continue
        s += probe
        q = candidate
        trace.append((s, q.copy()))
    return trace


def geometric_approach(
    path, target: np.ndarray, q_star: np.ndarray, s_star: float, *, decades: int = 5,
) -> tuple[list[float], list[float]]:
    """Distances to the fold at ``s* - 10^-k``, each re-solved from the previous one.

    A geometric sequence of gaps is what makes an exponent measurable: an adaptive tracker's own
    step halving hides it.  Returns ``(gaps, distances)``.
    """
    from study.census import solve_lm

    gaps: list[float] = []
    distances: list[float] = []
    previous = None
    for decade in range(2, 2 + decades):
        gap = 10.0 ** (-decade)
        s = s_star - gap
        if s <= 0.0:
            break
        model = sr0.robot(path(s))
        q, converged = solve_lm(model, q_star if previous is None else previous, target)
        if not converged:
            continue
        gaps.append(float(np.log(gap)))
        distances.append(float(np.log(np.linalg.norm(q - q_star))))
        previous = q
    return gaps, distances


def fit_exponent(gaps: list[float], distances: list[float]) -> tuple[float, float, int]:
    """Slope of ``log|q - q*|`` against ``log(s* - s)``, stopping where the numerics bottom out.

    Close enough to the fold the re-solved configuration stops moving (the solve is too
    ill-conditioned to resolve the branch), and those saturated points drag the fit below the true
    exponent.  The last points are dropped while the distances keep shrinking by at least a factor
    1.5 per decade -- the geometric decay a power law requires.

    Returns:
        ``(slope, intercept, points_used)``.
    """
    used = len(distances)
    while used > 3:
        ratios = [
            distances[index] - distances[index + 1] for index in range(used - 1)
        ]
        if min(ratios) < np.log(1.5):
            used -= 1
            continue
        break
    x = np.array(gaps[:used])
    y = np.array(distances[:used])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), used


def main() -> int:
    """Run the endgame on every stalled branch of one pose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pose-seed", type=int, default=0)
    parser.add_argument("--sigma-floor", type=float, default=2e-3)
    args = parser.parse_args()

    offset = float(link_transforms()[4][1, 3])
    sr0_model = sr0.robot()
    sr5_model = sr0.robot(link_transforms())
    circular = iks.circular_joints(sr5_model)
    rng = np.random.default_rng(args.pose_seed)
    q_true = rng.uniform(-2.0, 2.0, sr5_model.num_dof)
    target = sr5_model.fk(q_true)
    path = path_at(offset)
    print(f"pose seed {args.pose_seed}; folds are computed, not walked around\n")

    for index, seed in enumerate(sr0.solve(sr0_model, target)):
        track = homotopy.track(path, target, seed.q, sigma_floor=args.sigma_floor)
        if track.finished:
            continue
        s_stall = track.progress
        s_star, q_star, residual = refine_fold(path, target, track.q, s_stall)
        # the two branches that meet: distances to the nearest other solution at the fold
        model_at = sr0.robot(path(s_star))
        fibre = [
            q for q in sir_fibre(model_at, target) if iks.configuration_distance(
                q, q_star, circular
            ) < 0.2
        ]
        partner = min(
            (iks.configuration_distance(q, q_star, circular) for q in fibre if
             float(np.linalg.norm(q - q_star)) > 0),
            default=float("nan"),
        )
        gaps, distances = geometric_approach(path, target, q_star, s_star)
        slope, _intercept, used = fit_exponent(gaps, distances)
        print(
            f"  seed {index}: stalled s = {s_stall:.4f} -> fold s* = {s_star:.6f} "
            f"(augmented residual {residual:.1e}) | solutions within 0.2 rad: {len(fibre)}, "
            f"nearest partner {partner:.2e} rad | distances per decade "
            f"{[round(float(np.exp(value)), 5) for value in distances]} | split exponent "
            f"{slope:.3f} over {used} points "
            f"({'square root: two branches' if abs(slope - 0.5) < 0.08 else 'check'})"
        )
    return 0


def sir_fibre(model, target: np.ndarray) -> list[np.ndarray]:
    """A small multi-start fibre for the fold's own configuration (local, for locating partners)."""
    from study.census import census, make_lm_solver

    fiber = census(model, target, solver=make_lm_solver(model), seeds=120,
                   rng=np.random.default_rng(0))
    return fiber.solutions


if __name__ == "__main__":
    raise SystemExit(main())
