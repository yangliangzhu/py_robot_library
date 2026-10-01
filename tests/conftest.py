"""Shared pytest fixtures.

The fixtures keep the test suite honest about the two backends: anything that
should hold for the library as a whole is parametrised over ``numpy`` and
``casadi`` so a divergence between the implementations fails the build.
"""

from __future__ import annotations

import numpy as np
import pytest

from model import IkType, ModelFactory

#: Backends every model-level test should run against.
BACKENDS = ("numpy", "casadi")

#: Robots exercised by the kinematics tests, with their DOF.
ROBOTS = (
    ("sr5", 6),
    ("er3", 7),
)


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrise a test over both model backends."""
    return request.param


@pytest.fixture
def rng() -> np.random.Generator:
    """A deterministic random generator, so failures reproduce."""
    return np.random.default_rng(20240501)


def sample_positions(
    robot,
    rng: np.random.Generator,
    margin: float = 0.15,
) -> np.ndarray:
    """Draw joint positions safely inside the robot's limits.

    Args:
        robot: A built robot model.
        rng: Source of randomness.
        margin: Fraction of each joint range to stay away from both limits.

    Returns:
        Joint positions in radians.
    """
    lower = robot.get_lower_bounds()
    upper = robot.get_upper_bounds()
    return lower + rng.uniform(margin, 1.0 - margin, robot.num_dof) * (upper - lower)


@pytest.fixture
def make_robot():
    """Return a factory that builds and caches a robot model.

    Returns:
        A callable ``(name, backend) -> robot``.
    """
    cache = {}

    def _make(name: str, backend: str = "numpy", ik_type=IkType.IK_STANDARD):
        key = (name, backend, ik_type)
        if key not in cache:
            cache[key] = ModelFactory.create(name, backend=backend, ik_type=ik_type)
        return cache[key]

    return _make
