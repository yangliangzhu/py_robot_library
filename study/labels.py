"""Geometric branch labels: the classical shoulder/elbow/wrist signs, measured off the axes.

A 6R's solution branches are usually named by three binary choices -- shoulder front/back, elbow
up/down, wrist flip/no-flip.  Each name is the sign of a quantity that vanishes exactly on one
stratum of the singular set, which is what makes the labels checkable against the measured branch
partition (`study/exp15_branch_labels.py`):

* **shoulder**: the wrist centre's signed distance from the plane spanned by axes 1 and 2.  Zero
  when the wrist centre lies in that plane, where ``q1`` is free.
* **elbow**: the triple product ``(elbow - shoulder) x (wrist - elbow) . axis2``.  Zero when the
  shoulder, elbow and wrist are collinear, i.e. the arm is straight.
* **wrist**: ``(axis4 x axis6) . axis5``.  Zero when axes 4 and 6 line up, the classic wrist
  singularity.

All three are computed from the *axes* rather than from joint indices, so they apply to any 6R
whose structure matches the names -- including the SR5, whose wrist is offset and whose "branches"
therefore need not be eight.
"""

from __future__ import annotations

import numpy as np

from study import arm_geometry as ag

#: Below this magnitude a label is undefined (the configuration sits on the stratum).
UNDEFINED = 1e-6


def _normalise(vector: np.ndarray) -> np.ndarray:
    """Unit vector, or zeros when the input is numerically zero."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def branch_labels(model, q: np.ndarray) -> tuple[int, int, int]:
    """``(shoulder, elbow, wrist)`` signs at ``q``, each ``+1``, ``-1`` or ``0`` when undefined.

    Args:
        model: Robot model.
        q: Joint positions in radians.

    Returns:
        The three signs.
    """
    frames = ag.joint_frames(model, q)
    axes = [frames[index][:3, 2] for index in range(1, 7)]
    origins = [frames[index][:3, 3] for index in range(1, 7)]

    plane_normal = _normalise(np.cross(axes[0], axes[1]))  # axes 1 and 2 span the shoulder plane
    shoulder = float((origins[3] - origins[0]) @ plane_normal)

    elbow = float(np.cross(origins[2] - origins[0], origins[3] - origins[2]) @ axes[1])

    wrist = float(np.cross(axes[3], axes[5]) @ axes[4])

    def sign(value: float) -> int:
        return 0 if abs(value) < UNDEFINED else (1 if value > 0 else -1)

    return sign(shoulder), sign(elbow), sign(wrist)


__all__ = ["UNDEFINED", "branch_labels"]
