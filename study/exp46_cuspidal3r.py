#!/usr/bin/env python3
"""exp46: search a 3R family for an arm that is cuspidal -- the orthogonality witness.

The study's two answers are claimed to be orthogonal: solvability in radicals (complex monodromy) and
branch structure (real cuspidality).  A witness needs one arm that is *both* cuspidal and solvable by
radicals.  SR0 failed the first half (five poses, no clearance-keeping connection within a sign group),
which is expected for a Pieper arm; a 3R arm cannot fail the second half, since its IK is elementary.

So this searches a small 3R family.  The design ingredient is the one Wenger's 3-R cuspidal example
turns on: a *shoulder offset*, i.e. joints 1 and 2 skew rather than intersecting, with joints 2 and 3
orthogonal.  In the library's ``mat`` form the family is

    Ms = [I, Trans(r2, 0, h1) Rx(-90), Trans(L2, 0, 0) Rx(90), Trans(L3, 0, 0)]

with joint 1 about the base z axis.  Each arm is tested the way the study tests branches: census, torus
deduplication, determinant signs, and a witnessed clearance-keeping walk within a sign group.

**Status: the family as written is degenerate and this experiment is unfinished.**  All twelve grid
points returned zero admissible solutions, which is what a *rank-deficient* arm looks like: composing
``Rx(-90)`` with ``Rx(90)`` leaves joint 3's axis parallel to joint 2's, so three Cartesian coordinates
cannot be controlled by three joints and the inverse problem has no solution for a generic target.
Diagnosis to act on: build the family from a proper modified-DH table (or check ``rank J = 3`` on a
sample of configurations before searching), then re-run the cuspidality test.  The negative half of the
witness hunt is already measured elsewhere: SR0 is *not* cuspidal (exp45, five poses, zero
clearance-keeping connections within a sign group), so the witness has to come from a 3R or from a
spherical-wrist 6R that is nonetheless cuspidal.

Run::

    python3 -m study.exp46_cuspidal3r
"""

from __future__ import annotations

import itertools

import numpy as np

from model import IkType
from model.robot_model_numpy import RobotModelNumpy
from study import ik_structure as iks
from study.census import census, make_lm_solver
from study.chamber import resolve_pair


def build(r2: float, h1: float, link2: float, link3: float) -> RobotModelNumpy:
    """A 3R orthogonal arm with a shoulder offset ``r2`` (see the module docstring)."""
    def rot_x(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.eye(4)
        matrix[1:3, 1:3] = np.array([[cosine, -sine], [sine, cosine]])
        return matrix

    def trans(x: float, y: float, z: float) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = (x, y, z)
        return matrix

    config = {
        "name": "cuspidal3r",
        "param_type": "mat",
        "param": [
            trans(r2, 0.0, h1) @ rot_x(-np.pi / 2),
            trans(link2, 0.0, 0.0) @ rot_x(np.pi / 2),
            trans(link3, 0.0, 0.0),
        ],
        "base": np.eye(4).tolist(),
        "ee": np.eye(4).tolist(),
        "lower": [-np.pi] * 3,
        "upper": [np.pi] * 3,
        "joint_unit": "radian",
        "dh_unit": "meter",
    }
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config)
    return model


def cuspidal(model, rng, *, seeds: int = 250, delta: float = 1e-2) -> tuple[int, int, int]:
    """``(solutions, sign groups, connected within-sign pairs)`` at one random pose."""
    lower = np.asarray(model.lower_bounds)
    upper = np.asarray(model.upper_bounds)
    for _ in range(20):
        target = model.fk(rng.uniform(lower, upper))
        fiber = census(model, target, solver=make_lm_solver(model), seeds=seeds, rng=rng)
        solutions: list[np.ndarray] = []
        for candidate in fiber.solutions:
            if iks.sigma_min(model, candidate) <= 1e-3:
                continue
            if any(iks.torus_distance(candidate, other) < 1e-3 for other in solutions):
                continue
            solutions.append(candidate)
        if len(solutions) < 3:
            continue
        signs = [int(np.sign(iks.det_jacobian(model, candidate))) for candidate in solutions]
        tested = connected = 0
        for i, j in itertools.combinations(range(len(solutions)), 2):
            if signs[i] != signs[j]:
                continue
            tested += 1
            if resolve_pair(model, solutions[i], solutions[j], delta=delta,
                            circular=iks.circular_joints(model)).connected:
                connected += 1
        return len(solutions), len(set(signs)), connected
    return 0, 0, 0


def main() -> int:
    """Grid-search the family and report the cuspidal arms."""
    rng = np.random.default_rng(0)
    found: list[tuple[float, float, float, float, int, int]] = []
    tested = 0
    for r2, link2, link3 in itertools.product((0.0, 0.3, 0.7), (1.0, 2.0), (0.6, 1.8)):
        model = build(r2, 0.5, link2, link3)
        tested += 1
        solutions, groups, connected = cuspidal(model, rng)
        flag = "  <-- CUSPIDAL" if connected else ""
        print(f"  r2={r2:.1f} L2={link2:.1f} L3={link3:.1f}: {solutions} solutions, "
              f"{groups} sign groups, within-sign connected {connected}{flag}")
        if connected:
            found.append((r2, 0.5, link2, link3, solutions, connected))
    print(f"\n{len(found)} cuspidal arm(s) among {tested} tested")
    if found:
        r2, h1, link2, link3, solutions, connected = found[0]
        print(f"WITNESS: the 3R arm with shoulder offset r2={r2}, links {link2}/{link3} is cuspidal "
              f"({connected} clearance-keeping connections at one pose) and its IK is elementary, so "
              f"it is both solvable by radicals and cuspidal -- the orthogonality is witnessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
