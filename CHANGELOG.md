# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-16

First release with a documented public API, a test suite and packaged metadata.

### Added

- `model/robot_model_base.py` with `RobotModelBase`, holding the configuration
  logic the two backends used to duplicate: parsing a robot configuration,
  joint limits, the tool transform, and IK solver selection.
- `tests/` — a pytest suite of 184 tests. Model-level tests are parametrised
  over both backends, so a divergence between NumPy and CasADi fails the build.
  The Jacobian and the manipulability gradient are verified against numerical
  derivatives rather than stored reference values.
- `model.ik_solver.make_ik_solver()`, replacing the two copies of
  `ik_factory_method` that lived on the model classes.
- `model.configs.loader.list_robot_configs()`.
- `LICENSE` (MIT), `CONTRIBUTING.md`, this changelog, and a CI workflow.
- `pyproject.toml` with full project metadata, replacing `setup.py`.
- Type hints, English Google-style docstrings and explicit imports across the
  package, and a ruff configuration to keep them that way.

### Fixed

- `RobotModelNumpy.hessian` and `RobotModelNumpy.derivative_manip` hard-coded 7
  joints, so 6-DOF robots produced the wrong shape and skipped the last joint.
- `RobotModelNumpy.jac_normal` hard-coded the virtual link length instead of
  using `virtual_link`.
- `RobotModelNumpy.manip` returned `det(J @ J.T)`, the square of the Yoshikawa
  measure, while the CasADi backend returned its square root. Both now return
  the square root, matching the C++ implementation and this library's own
  `derivative_manip`.
- `RobotModelCasadi.hessian` returned a `(6 * num_dof, num_dof)` matrix where
  the NumPy backend returned `(6, num_dof, num_dof)`. Both now return the latter.
- `get_matrix_list` raised `ValueError('dh type error')` without saying which
  type it received.
- `model/dh_param.py`: the `Dh` class defined an `order_map` attribute and an
  `order_map()` method; the attribute shadowed the method, making it
  unreachable. The method is now `print_order_map()`.
- `model/ik_solver.py`: `raise NotImplemented(...)` and `raise NotImplemented`
  were not valid exception types.
- `IkNaive` and `IkQp` silently accepted solver types they do not implement.
  They now raise `NotImplementedError`.
- `model/mrobotics.py` defined `inverse_dynamics` twice, so the centre-of-mass
  variant was unreachable. It is now `inverse_dynamics_com`.
- `model/common.py` called `numpy.set_printoptions` as an import side effect,
  changing global state for anything that imported the library.
- `IkQp` printed four debug lines on its first hundred calls.

### Changed

- **Breaking:** `RobotModelBase.build()` renamed its first parameter from `type`
  to `param_type`. It no longer shadows the `type` builtin, and the parameter is
  positional in all shipped call sites.
- **Breaking:** `Dh.__init__` renamed its `type` parameter and `type` attribute
  to `dh_type`.
- **Breaking:** `ModelFactory.create()` renamed its second parameter from `type`
  to `backend`, and now raises `KeyError` instead of printing and returning
  `None` for an unknown robot or backend.
- **Breaking:** `load_robot_config()` raises `FileNotFoundError` for an unknown
  name and `ValueError` for a malformed file, instead of failing later.
- `Dh.__init__` copies the parameter rows it is given rather than holding
  references to the caller's lists.
- `RobotModelBase.set_bounds()` validates that the limits are one-dimensional,
  of equal length, and ordered; passing `None` for both clears them.
- The factory now knows about `franka` and `er3_sdh`, which shipped as
  configuration files but were not reachable from `ModelFactory.create()`.

### Removed

- `setup.py`, superseded by `pyproject.toml`.
- The unused `scipy` import and dependency. `scipy` was imported in
  `model/ik_solver.py` but never used.

## [0.1.0]

Initial version: DH parameter handling, NumPy and CasADi robot models, the IK
solvers, geometry utilities and the robot YAML descriptions.

[Unreleased]: https://gitee.com/yangliangzhu_rob/py_robot_library/compare/v0.2.0...HEAD
[0.2.0]: https://gitee.com/yangliangzhu_rob/py_robot_library/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitee.com/yangliangzhu_rob/py_robot_library/releases/tag/v0.1.0
