#!/usr/bin/env python3
"""exp11: the DH-continuation formula, measured -- order of accuracy and where it dies

Finding 2's premise: with ``FK(dh, q) = T``, differentiating along a path in parameter space
gives ``J_dh dh + J_q dq = 0``, so ``dq = -pinv(J_q) J_dh dh`` moves a solution when the
parameters move.  This experiment puts numbers on it:

* the error of the first-order step as a function of the parameter step -- the order is the
  whole claim, and it is measurable;
* what one Newton correction on top does (the predictor-corrector form), which is what a
  continuation actually uses;
* what happens when ``J_q`` is ill-conditioned (a solution near the singular set), where the
  formula is *defined* but the step is not usable;
* the step along the one parameter that separates the SR5 from its Pieper-solvable neighbour
  (``study/exp10_sr0_geometry.py``: link 5's y offset, +0.136 m), because that is the direction
  the SR0 homotopy will travel.

Run::

    python3 -m study.exp11_dh_continuation --steps 6
"""

from __future__ import annotations

import argparse

import numpy as np

from model import IkType, ModelFactory
from model.configs.loader import load_robot_config
from model.robot_model_numpy import RobotModelNumpy
from study import arm_geometry as ag
from study.census import solve_lm
from study.ik_structure import (
    circular_joints,
    configuration_distance,
    random_regular_configuration,
    sigma_min,
)


def link_transforms(robot: str = "rokae_sr5") -> list[np.ndarray]:
    """The robot's link transforms in the library's ``Ms`` order, without base or tool."""
    config = load_robot_config(robot)
    return [np.asarray(matrix, dtype=float) for matrix in config["param"]]


def fk_with(links: list[np.ndarray], q: np.ndarray, tool: np.ndarray | None = None) -> np.ndarray:
    """Forward kinematics from an explicit list of link transforms.

    Args:
        links: Six 4x4 link transforms, in the library's order.
        q: Joint positions in radians, length 6.
        tool: Optional flange transform; identity when omitted.

    Returns:
        The 4x4 end-effector pose.
    """
    pose = np.eye(4)
    for index, link in enumerate(links):
        pose = pose @ link @ ag.rot_z(float(q[index]))
    return pose if tool is None else pose @ tool


def parameter_vector(links: list[np.ndarray]) -> np.ndarray:
    """Flatten the translation components of every link into one vector."""
    return np.array([link[row, 3] for link in links for row in range(3)])


def with_parameter(links: list[np.ndarray], index: int, value: float) -> list[np.ndarray]:
    """A copy of ``links`` with one translation component replaced."""
    out = [link.copy() for link in links]
    link_index, row = divmod(index, 3)
    out[link_index][row, 3] = value
    return out


def moved_links(
    links: list[np.ndarray], step: float, direction: np.ndarray
) -> list[np.ndarray]:
    """Link transforms after moving every parameter by ``step * direction``.

    Args:
        links: The baseline link transforms.
        step: Scalar step length.
        direction: A vector over the same 18 translation components.

    Returns:
        A new list of link transforms.
    """
    values = parameter_vector(links)
    out = [link.copy() for link in links]
    for index, weight in enumerate(np.asarray(direction, dtype=float)):
        if weight == 0.0:
            continue
        link_index, row = divmod(index, 3)
        out[link_index][row, 3] = values[index] + step * weight
    return out


def parameter_jacobian(links: list[np.ndarray], q: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """``d FK / d parameter`` as a 6x18 matrix, in the library's SE(3) error convention.

    Finite differences, because the library differentiates with respect to ``q`` and not with
    respect to the DH table; the parameter count is small and this runs once per experiment.
    """
    base_pose = fk_with(links, q)
    values = parameter_vector(links)
    jacobian = np.zeros((6, values.size))
    for index in range(values.size):
        plus = with_parameter(links, index, values[index] + h)
        minus = with_parameter(links, index, values[index] - h)
        delta = _se3_error(fk_with(minus, q), fk_with(plus, q))
        jacobian[:, index] = delta / (2.0 * h)
    return jacobian, base_pose


def _se3_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The library's SE(3) error from ``a`` to ``b``, reimplemented to stay dependency-free."""
    error = np.zeros(6)
    error[:3] = b[:3, 3] - a[:3, 3]
    spin = b[:3, :3] @ a[:3, :3].T
    cosine = max(-1.0, min(1.0, 0.5 * (np.trace(spin) - 1.0)))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return error
    axis = np.array(
        [spin[2, 1] - spin[1, 2], spin[0, 2] - spin[2, 0], spin[1, 0] - spin[0, 1]]
    )
    error[3:] = axis / (2.0 * np.sin(angle)) * angle
    return error


def newton_correct(
    links: list[np.ndarray], q: np.ndarray, target: np.ndarray, *, iterations: int = 3,
    damping: float = 1e-8,
) -> np.ndarray:
    """Gauss-Newton steps on ``q`` so that ``FK(links, q)`` reaches ``target``."""
    q = np.asarray(q, dtype=float).copy()
    for _ in range(iterations):
        error = _se3_error(fk_with(links, q), target)
        if max(np.linalg.norm(error[:3]), np.linalg.norm(error[3:])) < 1e-13:
            break
        numbers = _numeric_jacobian_q(links, q)
        normal = numbers @ numbers.T + damping * np.eye(6)
        q = q + numbers.T @ np.linalg.solve(normal, error)
    return q


def _numeric_jacobian_q(links: list[np.ndarray], q: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Finite-difference Jacobian with respect to ``q`` (keeps this module self-contained)."""
    jacobian = np.zeros((6, len(q)))
    for index in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        jacobian[:, index] = _se3_error(fk_with(links, minus), fk_with(links, plus)) / (2.0 * h)
    return jacobian


def model_with(links: list[np.ndarray]) -> RobotModelNumpy:
    """A numpy SR5 model whose link transforms are the given ones."""
    config = load_robot_config("rokae_sr5")
    config["param"] = [np.asarray(link, dtype=float) for link in links]
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config)
    return model


def main() -> int:
    """Run the continuation measurements."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=6, help="parameter step sizes to try")
    parser.add_argument("--largest", type=float, default=0.1, help="largest parameter step, metres")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ModelFactory.create("sr5", backend="numpy", ik_type=IkType.IK_STANDARD)
    circular = circular_joints(model)
    links = link_transforms()
    rng = np.random.default_rng(args.seed)
    # a well-conditioned solution for the accuracy sweep; the near-singular case is separate
    q0 = random_regular_configuration(model, rng, sigma_floor=0.05)
    target = fk_with(links, q0)
    print(f"SR5, sigma_min at the solution: {sigma_min(model, q0):.4f}")

    # the SR0 direction: link 5's y offset (index = (5-1)*3 + 1)
    direction = np.zeros(18)
    direction[(5 - 1) * 3 + 1] = 1.0
    print(f"continuation direction: link 5 y offset (+{links[4][1, 3]:.4f} m at the SR5, "
          f"0 at the SR0)\n")
    print(f"  {'step (m)':>9s} {'first order':>13s} {'+1 Newton':>12s} {'+3 Newton':>12s} "
          f"{'exact IK':>12s} {'|q3 - q_exact|':>15s} {'ratio':>8s}")
    first_error = None
    for step in np.geomspace(args.largest, args.largest / 10 ** (args.steps - 1), args.steps):
        moved = moved_links(links, step, direction)
        jac_q = _numeric_jacobian_q(links, q0)
        jac_p, _ = parameter_jacobian(links, q0)
        dq = -np.linalg.pinv(jac_q) @ (jac_p @ (step * direction))
        q_first = q0 + dq
        error_first = max(pose_error_from(fk_with(moved, q_first), target))
        q_one = newton_correct(moved, q_first, target, iterations=1)
        error_one = max(pose_error_from(fk_with(moved, q_one), target))
        q_three = newton_correct(moved, q_first, target, iterations=3)
        error_three = max(pose_error_from(fk_with(moved, q_three), target))
        # ground truth: a full IK solve on the perturbed arm, from the same seed
        perturbed_model = model_with(moved)
        q_exact, converged = solve_lm(perturbed_model, q0, target)
        error_exact = max(pose_error_from(fk_with(moved, q_exact), target))
        ratio = "-"
        if first_error is not None:
            ratio = f"{first_error / error_first:8.1f}"
        first_error = error_first
        print(f"  {step:9.5f} {error_first:13.3e} {error_one:12.3e} {error_three:12.3e} "
              f"{error_exact:12.3e} {configuration_distance(q_three, q_exact, circular):15.2e} "
              f"{ratio:>8s}" + ("" if converged else "  (IK did not converge)"))

    print("\nthe same step with J_q ill-conditioned (solution driven towards the singular set):")
    singular_q = None
    for _ in range(400):
        candidate = random_regular_configuration(model, rng, sigma_floor=1e-4)
        if sigma_min(model, candidate) < 5e-3:
            singular_q = candidate
            break
    if singular_q is None:
        print("  no near-singular configuration found in the sample")
    else:
        target_s = fk_with(links, singular_q)
        jac_q = _numeric_jacobian_q(links, singular_q)
        jac_p, _ = parameter_jacobian(links, singular_q)
        step = args.largest / 10.0
        moved = moved_links(links, step, direction)
        dq = -np.linalg.pinv(jac_q) @ (jac_p @ (step * direction))
        q_first = singular_q + dq
        error_first = max(pose_error_from(fk_with(moved, q_first), target_s))
        q_three = newton_correct(moved, q_first, target_s, iterations=3)
        error_three = max(pose_error_from(fk_with(moved, q_three), target_s))
        print(f"  sigma_min {sigma_min(model, singular_q):.2e}: |dq| = {np.linalg.norm(dq):.3f} rad, "
              f"first-order error {error_first:.3e} m, after 3 Newton steps {error_three:.3e} m")
        print("  (compare with the well-conditioned case above at the same step)")
    return 0


def pose_error_from(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """``(position, rotation)`` error between two poses, using this module's error function."""
    error = _se3_error(current, target)
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


if __name__ == "__main__":
    raise SystemExit(main())
