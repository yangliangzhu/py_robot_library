#!/usr/bin/env python3
"""exp10: which DH parameters make the SR5 Pieper-solvable?  (Finding 2's SR0, measured)

Pieper's condition is about axes: a 6R has a closed form when its last three joint axes meet at a
point.  The SR5 does not -- that is why the engineers said there is no closed form -- and the
question is which parameter to relax to make it so, without guessing.

This experiment measures the pairwise axis geometry of the SR5, then rebuilds the model with one
parameter at a time relaxed and re-measures.  The parameter whose relaxation takes the wrist's
axis-triple distance to zero is the one that separates the arm from its solvable neighbour, and
the size of the gap is exactly the distance a homotopy has to travel.

Run::

    python3 -m study.exp10_sr0_geometry
"""

from __future__ import annotations

import numpy as np

from model import IkType
from model.configs.loader import load_robot_config
from model.robot_model_numpy import RobotModelNumpy
from study import arm_geometry as ag

#: The translation components of the link transforms, named for the report.
COMPONENT_NAMES = ("x", "y", "z")


def build_sr5_with(param: list[np.ndarray]) -> RobotModelNumpy:
    """A numpy SR5 model from an explicit list of link transforms."""
    config = load_robot_config("rokae_sr5")
    config["param"] = [np.asarray(matrix, dtype=float) for matrix in param]
    model = RobotModelNumpy()
    model.build("mat", IkType.IK_STANDARD, config)
    return model


def wrist_gap(model, knee: tuple[int, int, int] = (4, 5, 6)) -> tuple[float, float, float]:
    """The three pairwise axis distances of the wrist, in metres."""
    pairs = {(pair.first, pair.second): pair.distance for pair in ag.axis_fingerprint(model)}
    first, second, third = knee
    return (
        pairs[(first, second)],
        pairs[(second, third)],
        pairs[(first, third)],
    )


def main() -> int:
    """Measure the baseline, then each relaxation."""
    config = load_robot_config("rokae_sr5")
    baseline = [np.asarray(matrix, dtype=float) for matrix in config["param"]]
    model = build_sr5_with(baseline)
    print("SR5 axis fingerprint")
    print(ag.describe(model))
    print(f"\nwrist triples (4-5, 5-6, 4-6) in metres: "
          f"{tuple(round(value, 6) for value in wrist_gap(model))}")
    print(f"the last three axes meet at a point: {wrist_gap(model)[2] < 1e-9}")

    print("\nexhaustive single-parameter relaxation (all 6 links x 3 translation"
          " components), wrist triples in metres:")
    closers = []
    for link in range(1, 7):
        for component in range(3):
            original = float(baseline[link - 1][component, 3])
            if abs(original) < 1e-12:
                continue
            relaxed = [matrix.copy() for matrix in baseline]
            relaxed[link - 1][component, 3] = 0.0
            candidate = build_sr5_with(relaxed)
            gaps = wrist_gap(candidate)
            if max(gaps) < 1e-9:
                closers.append((link, component, original))
                print(f"  link {link} {COMPONENT_NAMES[component]} = {original:+.4f} -> 0.0:  "
                      f"{tuple(round(value, 9) for value in gaps)}  CLOSES THE WRIST")
    if not closers:
        print("  none: no single translation component makes the wrist spherical")

    # the relaxation the geometry points at: the one behind the 4-6 offset
    relaxed = [matrix.copy() for matrix in baseline]
    for link, component, _value in closers:
        relaxed[link - 1][component, 3] = 0.0
    sr0 = build_sr5_with(relaxed)
    print("\nSR0 = SR5 with that component zeroed:")
    print(f"  wrist triples: {tuple(round(value, 9) for value in wrist_gap(sr0))}")
    print(f"  all pairwise intersections: "
          f"{[f'{p.first}-{p.second}' for p in ag.axis_fingerprint(sr0) if p.intersects]}")
    for link, component, value in closers:
        print(f"  travelled distance: link {link} {COMPONENT_NAMES[component]} "
              f"{value:+.4f} -> 0.0 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
