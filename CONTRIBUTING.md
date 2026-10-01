# Contributing

Thanks for taking the time to contribute. This document describes how to set the
project up, the conventions the code follows, and what a change is expected to
include.

## Getting started

```bash
git clone https://gitee.com/yangliangzhu_rob/py_robot_library.git
cd py_robot_library
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

`pip install -e ".[dev]"` installs the runtime dependencies plus pytest and
ruff.

## Before you open a merge request

Run both of these and make sure they are clean:

```bash
ruff check .
pytest
```

A change that adds or alters behaviour is expected to come with tests. The
model-level tests are parametrised over both backends, so if you touch shared
behaviour, both `numpy` and `casadi` will be exercised without extra work.

## Code conventions

### Style

The project is linted with [ruff](https://docs.astral.sh/ruff/) using rules
`E`, `F`, `W`, `I`, `UP` and `B`, with a 100-column limit. Configuration lives in
`pyproject.toml`. Long numeric tables are exempt from the line-length rule.

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Classes | `PascalCase` | `RobotModelNumpy`, `IkStandard` |
| Functions and methods | `snake_case` | `rot_x`, `jacobian`, `derivative_manip` |
| Variables | `snake_case` | `num_dof`, `target_pose` |
| Constants | `UPPER_SNAKE` | `DEFAULT_VIRTUAL_LINK` |
| Private helpers | leading underscore | `_damped_pinv` |
| Enum members | `UPPER_SNAKE` | `IkType.IK_STANDARD` |

### Imports

Group standard library, third-party and local imports, separated by blank lines.
Import names explicitly; do not use wildcard imports. Every module starts with
`from __future__ import annotations` so the modern `X | None` and `list[int]`
type syntax is available on Python 3.9.

### Documentation

Public classes, methods and functions carry a docstring in Google style:

```python
def manip(self, q: np.ndarray) -> float:
    """Return the Yoshikawa manipulability measure at ``q``.

    Args:
        q: Joint positions in radians, length ``num_dof``.

    Returns:
        ``sqrt(det(J @ J.T))``, which vanishes at kinematic singularities.
    """
```

Write the summary line in the imperative mood and end it with a period.
Document units, array shapes and whether an argument is mutated — these are the
details that are easy to get wrong in robotics code.

### Types

Annotate public function signatures. Use `numpy.ndarray` for arrays and say the
expected shape and units in the docstring, since the type itself cannot carry
that information.

### Errors

Raise a specific exception with a message that says what was wrong and what was
expected:

```python
raise ValueError(f"param_type must be 'mat' or one of {dh_types}, got {param_type!r}")
```

Use `ValueError` for bad values, `TypeError` for wrong types, `KeyError` for an
unknown lookup key, `NotImplementedError` for an unsupported variant, and
`RuntimeError` for a method called out of order (for example `ik()` before
`build()`). Do not print a message and return `None` in a library.

## Adding a robot

1. Add a YAML description to `model/configs/`. Use an existing file as a
   template; `param_type` is `mat`, `mdh_param` or `sdh_param`, and units are
   declared with `dh_unit` and `joint_unit` so the loader can normalise them.
2. Add an entry to `ModelFactory._ROBOTS` mapping the public name to the config
   stem and the parameter layout, and a `create_<name>` helper if the robot is
   meant to be part of the documented API.
3. Add the robot to the `KNOWN_ROBOTS` tuple in `tests/test_model_factory.py`.

`tests/test_model_factory.py` checks that every YAML file is reachable from the
factory, so step 3 is enforced by the test suite.

## Adding an IK solver

1. Subclass `_IkSolverBase` in `model/ik_solver.py` and implement
   `__call__(self, angle, target) -> tuple[np.ndarray, bool]`.
2. Override `configure()` if the solver does not use the standard mapping from
   `IkType` to a pose error and Jacobian, and raise `NotImplementedError` for
   variants it does not support.
3. Add the enum value to `model/ik_type.py` and register the class in
   `_SOLVER_REGISTRY`.
4. Add tests to `tests/test_ik.py` covering convergence and joint-limit
   compliance.

## Reporting bugs

Please include the robot name, the backend, and the joint positions or pose that
triggers the problem. A short reproduction against `ModelFactory.create(...)` is
the fastest way to get a fix.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
