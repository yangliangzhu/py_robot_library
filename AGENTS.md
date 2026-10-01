# AGENTS.md — Agent Coding Guidelines for py_robot_library

## Project Overview

Python robotics kinematics library: Denavit-Hartenberg parameters, forward and
inverse kinematics, Jacobian/Hessian, manipulability analysis and geometry
utilities, with interchangeable NumPy and CasADi backends.

Dependencies: NumPy, CasADi, PyYAML. Python >= 3.9.

The C++ counterpart is `robot-model-cpp`; the two implement the same algorithms
and are cross-checked against each other. Keep their conventions aligned.

## Quick Start

```bash
# Install (editable installs need pip >= 23)
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Test
pytest                       # whole suite
pytest tests/test_ik.py      # one file
pytest -k manip              # by name
pytest -x                    # stop at the first failure

# Lint
ruff check .
ruff check --fix .
```

There is no `python` executable on some systems in this workspace; use `python3`.

## Code Style

| Element | Convention | Example |
|---------|------------|---------|
| Classes | `PascalCase` | `RobotModelNumpy`, `IkStandard` |
| Functions/methods | `snake_case` | `rot_x`, `derivative_manip` |
| Variables | `snake_case` | `num_dof`, `target_pose` |
| Constants | `UPPER_SNAKE` | `DEFAULT_VIRTUAL_LINK` |
| Private helpers | leading underscore | `_damped_pinv` |
| Enum members | `UPPER_SNAKE` | `IkType.IK_STANDARD` |
| Modules | `snake_case` | `robot_model_numpy.py` |

### Formatting Rules

- **Line length**: 100 (ruff, `pyproject.toml`)
- **Indent**: 4 spaces
- **Ruff rules**: `E`, `F`, `W`, `I`, `UP`, `B`; `E501` is ignored because the
  DH and robot parameter tables are unreadable when wrapped

### Imports

Every module starts with `from __future__ import annotations`, so the modern
`X | None` and `list[int]` syntax is available on Python 3.9.

Order groups with a blank line between them, one import per line, no wildcards:

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from .dh_param import Dh, get_matrix_list
```

The single exception is `model/common.py`, which exists to forward the historical
flat namespace and is marked with `# noqa: F401,F403`.

### Type Hints

Annotate public signatures. The type cannot express units or shapes, so put those
in the docstring:

```python
def manip(self, q: np.ndarray) -> float:
    """Return the Yoshikawa manipulability measure at ``q``.

    Args:
        q: Joint positions in radians, length ``num_dof``.
    """
```

### Documentation

English, Google style, imperative summary line ending with a period. Fill in
`Args:`, `Returns:` and `Raises:` where they apply.

```python
def jacobian(self, q: np.ndarray) -> np.ndarray:
    """Compute the 6xn geometric Jacobian.

    Args:
        q: Joint positions in radians, length ``num_dof``.

    Returns:
        A ``(6, num_dof)`` Jacobian: the first three rows map joint velocities
        to linear velocity, the last three to angular velocity.
    """
```

Document units, array shapes, and whether an argument is mutated — these are the
details that are easiest to get wrong in kinematics code.

### Error Handling

Raise a specific exception with a message saying what was wrong and what was
expected. Never print and return `None` from a library function.

```python
raise ValueError(f"param_type must be 'mat' or one of {dh_types}, got {param_type!r}")
```

| Situation | Exception |
|-----------|-----------|
| Bad value | `ValueError` |
| Wrong type | `TypeError` |
| Unknown lookup key | `KeyError` |
| Unsupported variant | `NotImplementedError` |
| Method called out of order | `RuntimeError` |

Never name a parameter `type`; it shadows the builtin. Use `param_type`,
`dh_type`, `backend` or `solver_type`.

## Architecture

```
Ms = [base, link_1, ..., link_n, tool]
```

Every model reduces to that ordered transform list. `RobotModelBase` builds it
from a robot configuration, tracks joint limits and manages the tool transform.
Subclasses implement only what differs:

- `RobotModelNumpy` evaluates dense arrays per call and reads `Ms` directly.
- `RobotModelCasadi` compiles FK/Jacobian/Hessian once and must override
  `_on_tool_changed()` to recompile when the tool moves.

Adding a behaviour to both backends means adding it to `RobotModelBase`, not
duplicating it.

### Backend parity

Anything that both backends expose must agree. The test suite is parametrised
over `numpy` and `casadi` for exactly this reason. Shapes matter as much as
values: `hessian()` returns `(6, num_dof, num_dof)` on both, and `manip()`
returns `sqrt(det(J @ J.T))` on both.

### Lazy imports

`model/__init__.py` resolves its exports through PEP 562 `__getattr__`, so
importing `model.ik_srs` or `model.dh_param` does not import CasADi. Keep new
exports in `_LAZY_EXPORTS` (and in `__all__`) rather than adding eager imports.

## Testing

- Framework: pytest, configured in `pyproject.toml` (`testpaths = ["tests"]`).
- Fixtures live in `tests/conftest.py`: `backend` parametrises over both
  backends, `rng` is seeded for reproducibility, `make_robot` builds and caches
  models, and `sample_positions` draws joint positions inside the limits.
- Verify the Jacobian and gradients against numerical derivatives rather than
  stored constants, so the tests check the implementation and not today's output.

```python
@pytest.mark.parametrize("name,dof", ROBOTS)
def test_matches_finite_difference(self, backend, make_robot, rng, name, dof):
    robot = make_robot(name, backend)
    q = sample_positions(robot, rng)
    assert np.allclose(robot.jacobian(q), _numeric_jacobian(robot, q), atol=1e-4)
```

- IK is iterative and seeded. Generate the target with `fk(q)` and seed nearby;
  never assert convergence from an arbitrary seed.
- `IK_NORMAL` and `IK_NULL_NORMAL` relax the tool yaw by design. Assert on the
  position and on the residual they actually minimise, not on the full pose.
- Regression fixes get a test named after the bug, with a comment saying what
  used to happen.

## Common Tasks

**Add a robot**: add `model/configs/<name>.yaml`, register it in
`ModelFactory._ROBOTS`, and add it to `KNOWN_ROBOTS` in
`tests/test_model_factory.py` (the suite enforces the last step).

**Add an IK solver**: subclass `_IkSolverBase` in `model/ik_solver.py`,
implement `__call__(angle, target) -> tuple[np.ndarray, bool]`, override
`configure()` if it does not use the standard `IkType` mapping, raise
`NotImplementedError` for variants it does not support, add the `IkType` value,
and register it in `_SOLVER_REGISTRY`.

**Add a geometry helper**: add it to `tools/geometry.py` and export it in
`__all__`.

**Change the public API**: update `model/__init__.py` (`_LAZY_EXPORTS` and
`__all__`), the README, and `CHANGELOG.md`.

## Mathematical Conventions

- Angles are radians unless a function documents otherwise.
- Rotation matrices are 3x3; homogeneous transforms are 4x4.
- Quaternions are `[w, x, y, z]`.
- RPY is `[roll, pitch, yaw]` for the ZYX composition
  `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
- Joint limits are stored in radians regardless of the config file's units.
- Distances are metres; `nervCartToAffine` converts its millimetre input.
- Compare floats with `np.allclose` and an explicit tolerance chosen for the
  method (1e-12 for exact algebra, 1e-4 for finite differences).
