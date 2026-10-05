# py_robot_library

Robot kinematics in Python: Denavit-Hartenberg parameters, forward and inverse
kinematics, and geometry utilities, with interchangeable NumPy and CasADi
backends.

The C++ counterpart of this library lives in
[robot-model-cpp](https://gitee.com/yangliangzhu_rob/robot-model-cpp); the two
implement the same algorithms and are cross-checked against each other.

> ### 🔎 The IK branch-structure study lives in [`study/`](study/)
>
> This repository is also the home of a long-running study of **how the inverse-kinematics solutions of a
> 6R arm are connected to each other**, carried out on the ROKAE SR5/SR0 and on a 3R benchmark, and of the
> instruments that make such claims measurable. It answers three questions and records every retraction:
>
> * **Which solutions are reachable from which?** — a two-tier certificate (`sign det J` is a strict
>   one-way test; a clearance-witnessed walk proves connectivity), the classical shoulder/elbow/wrist
>   labels measured to fail in both directions, and the engineering answer: for one prescribed Cartesian
>   loop the ten starting postures can track **100% / 23.2% / 30.8% / 0.9%** of it — a feasible path is a
>   property of a *uniqueness domain*, not of an aspect.
> * **Why is there no closed form, and what does it cost to bypass it?** — the obstruction is the
>   measurable **0.136 m** wrist offset; a solvable neighbour arm (SR0) plus parameter continuation
>   recovers the real solutions, with the fold spectrum, the ±2 law, and the 16-solution bound all
>   measured — including the boundary (**20-30%** of poses have no seed).
> * **Can the inverse kinematics be written with radicals?** — **no**: the measured monodromy group of
>   SR5 contains `S6` (two independent 6-element components, closure 720), so it is not solvable. The
>   same instrument on SR0 measures a solvable (order-2) subgroup — the criterion separates the two arms
>   exactly as the geometry says it should.
>
> **Start here:** [`study/summary/README.md`](study/summary/README.md) — a zero-context write-up in five
> documents with five figures generated from the measurements (`python3 -m study.doc_figures`), plus a
> glossary, a status ledger and the minimum reproduction command set. The academic version is
> [`study/paper.pdf`](study/paper.pdf); the raw record (including the 17-entry instrument-fault archive
> and the list of withdrawn claims) is [`study/NOTES.md`](study/NOTES.md) and
> [`study/collab/QUESTIONS.md`](study/collab/QUESTIONS.md).
>
> The study does **not** modify the library: every experiment is a module under `study/` (outside
> `testpaths`), run as `python3 -m study.<module>`, and every claim carries the command and the measured
> numbers that produced it.

## Features

- **DH parameters** — modified (MDH) and standard (SDH) conventions, with unit
  conversion and per-robot YAML descriptions.
- **Forward kinematics** — end-effector pose from joint positions, on both
  backends.
- **Jacobian and Hessian** — analytic on CasADi, and a NumPy implementation
  checked against a numerical derivative in the test suite.
- **Inverse kinematics** — four solvers: damped least-squares, a redundant-arm
  null-space solver, a single-step servo solver, and an experimental QP solver.
- **Manipulability** — the Yoshikawa measure and its gradient, used by the
  redundancy subtask.
- **Two backends** — `RobotModelNumpy` evaluates dense arrays on every call;
  `RobotModelCasadi` compiles the kinematics, Jacobian and derivatives once.
- **Geometry utilities** — rotation matrices, RPY angles, quaternions and
  homogeneous transforms.

## Requirements

- Python >= 3.9
- [NumPy](https://numpy.org/) >= 1.21
- [CasADi](https://web.casadi.org/) >= 3.6
- [PyYAML](https://pyyaml.org/) >= 6.0

## Installation

```bash
git clone https://gitee.com/yangliangzhu_rob/py_robot_library.git
cd py_robot_library

# Editable installs use PEP 660, which needs pip >= 23.
python -m pip install --upgrade pip
pip install -e .
```

For development, install the test and lint tooling as well:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from model import IkType, ModelFactory

# Build a 7-DOF ROKAE ER3 on the NumPy backend.
robot = ModelFactory.create("er3", backend="numpy", ik_type=IkType.IK_STANDARD)
print(robot.num_dof)  # 7

# Forward kinematics and the geometric Jacobian.
q = np.zeros(robot.num_dof)
pose = robot.fk(q)
jacobian = robot.jacobian(q)

# Inverse kinematics: solve for a pose, starting from a seed configuration.
target = robot.fk(np.radians([10, 20, -15, 30, 5, -25, 40]))
solution, success = robot.ik(q, target)
assert success
print(np.allclose(robot.fk(solution), target, atol=1e-4))
```

## Usage

### Building a model

`ModelFactory.create(name, backend=..., ik_type=...)` looks up a robot
description in `model/configs/` and returns a configured model.

| Name | Robot | DOF | Parameters |
|------|-------|-----|------------|
| `sr5` | ROKAE xMate SR5 | 6 | matrix |
| `sr5_v2` | ROKAE xMate SR5, tightened limits | 6 | matrix |
| `er3` | ROKAE xMate ER3 | 7 | MDH |
| `er3_v2` | ROKAE xMate ER3, expanded DH | 7 | MDH |
| `er3_sdh` | ROKAE xMate ER3 | 7 | SDH |
| `er3_plus` | ROKAE xMate ER3 Plus | 7 | MDH |
| `nerv_er3` | nerv ER3 | 7 | MDH |
| `aubo_c5` | AUBO C5 | 6 | MDH |
| `franka` | Franka Emika Panda | 7 | MDH |

Two backends are available:

```python
from model import ModelFactory

numpy_robot = ModelFactory.create("er3", backend="numpy")
casadi_robot = ModelFactory.create("er3", backend="casadi")
```

Both produce the same kinematics to floating-point precision; the CasADi backend
additionally provides exact derivatives and is faster once the compiled
expressions are built.

### Describing a robot by hand

```python
import numpy as np

from model import Dh, IkType, RobotModelNumpy, get_matrix_list

dh = Dh(
    [
        # [d, alpha, a, theta]
        [0.150, 0.0, 0.0, 0.0],
        [0.0, -np.pi / 2, 0.0, 0.0],
        [0.0, np.pi / 2, 0.0, 0.0],
        [0.610, -np.pi / 2, 0.0, 0.0],
        [0.0, np.pi / 2, 0.0, 0.0],
        [0.110, -np.pi / 2, 0.0, 0.0],
    ],
    dh_type="mdh",
)

config = {
    "param": dh,
    "lower": np.radians([-170, -120, -170, -170, -120, -170]),
    "upper": np.radians([170, 120, 170, 170, 120, 170]),
}

robot = RobotModelNumpy()
robot.build("dh", IkType.IK_STANDARD, config)
```

A robot is described by an ordered transform list
`Ms = [base, link_1, ..., link_n, tool]`. `get_matrix_list` builds it from a DH
table; the `"mat"` layout passes the link transforms directly.

### Inverse kinematics

| `IkType` | Behaviour |
|----------|-----------|
| `IK_STANDARD` | Damped least-squares with adaptive step back-off; constrains the full pose. |
| `IK_NORMAL` | Same, but relaxes the tool yaw, which helps a 7-DOF arm converge. |
| `IK_NULL` | Full pose, plus a redundancy subtask in the Jacobian null space that avoids joint limits and then maximises manipulability. |
| `IK_NULL_NORMAL` | The relaxed-yaw variant of `IK_NULL`. |
| `IK_NAIVE` | A single damped least-squares step; use it inside a servo loop. |
| `IK_QP` | Experimental one-step QP formulation. |

Every solver has the same call signature and returns `(joint_positions, success)`:

```python
from model import IkType, ModelFactory

robot = ModelFactory.create("er3", backend="casadi", ik_type=IkType.IK_NULL)
solution, success = robot.ik(seed, target_pose)
```

IK is iterative and seeded: pass a configuration near the expected solution.
The `IK_NORMAL` and `IK_NULL_NORMAL` variants converge in position but
deliberately do not match the target yaw.

### Tool transforms

`set_tool` mounts an interchangeable tool on the flange. It replaces the previous
tool rather than accumulating on top of it.

```python
tool = np.eye(4)
tool[2, 3] = 0.1  # 10 cm along the tool Z axis
robot.set_tool(tool)
robot.set_tool(np.eye(4))  # remove it again
```

### Geometry utilities

```python
import numpy as np

from tools.geometry import nervCartToAffine, quat2rot, rot2quat, rot_x, rpy2rot

rot_x(np.pi / 2)                      # 3x3 rotation about X
rpy2rot([0.1, 0.2, 0.3])              # ZYX composition, radians
quat2rot([1, 0, 0, 0])                # [w, x, y, z] -> 3x3
rot2quat(np.eye(3))                   # 3x3 -> [w, x, y, z]

# nerv Cartesian poses are [x, y, z, rx, ry, rz] with millimetres and degrees.
nervCartToAffine([100.0, 200.0, 300.0, 0.0, 0.0, 90.0])
```

## Testing

The test suite uses [pytest](https://pytest.org/):

```bash
pip install -e ".[dev]"

pytest                      # the whole suite
pytest tests/test_ik.py     # one file
pytest -k manip             # by name
```

Model-level tests are parametrised over both backends, so a divergence between
the NumPy and CasADi implementations fails the build. The Jacobian and the
manipulability gradient are checked against numerical derivatives rather than
against stored reference values.

Geometry regression data for the robot models lives in `test_data/*.npz`, and
`tools/generate_test_data.py` regenerates it.

## Linting

```bash
ruff check .        # lint
ruff check --fix .  # apply the safe fixes
```

## Project layout

```
py_robot_library/
├── model/                  # package: robot models and kinematics
│   ├── __init__.py         # public API
│   ├── dh_param.py         # DH tables and link transforms
│   ├── ik_type.py          # IkType enum
│   ├── ik_solver.py        # IK solvers + solver factory
│   ├── ik_srs.py           # analytical IK for S-R-S 7-DOF arms
│   ├── mrobotics.py        # screw-theoretic rigid-body helpers
│   ├── robot_model_base.py # shared model configuration logic
│   ├── robot_model_numpy.py
│   ├── robot_model_casadi.py
│   ├── model_factory.py
│   ├── common.py           # backwards-compatible aggregator
│   └── configs/            # per-robot YAML descriptions
├── tools/                  # geometry helpers and standalone scripts
├── study/                  # IK branch-structure study (see the callout above)
│   ├── summary/            #   zero-context write-up: 5 documents + 5 measured figures
│   ├── taskspace.py        #   lift a whole fibre along a task path (births, folds, guards)
│   ├── chamber.py          #   clearance-witnessed walks in joint space
│   ├── sr0.py              #   the solvable neighbour arm's closed-form solver
│   ├── cuspidal3r.py       #   the 3R benchmark (cuspidal, and cusped)
│   ├── exp*.py             #   the experiments, numbered and registered in collab/QUESTIONS.md
│   ├── NOTES.md            #   the detailed log, retractions included
│   ├── REPORT.md           #   the ten-minute Chinese brief
│   ├── paper.typ/.pdf      #   the academic write-up
│   └── collab/             #   the two-agent record: questions, round logs, letters
├── tests/                  # pytest suite
└── test_data/              # geometry regression data
```

## Conventions

- Angles are in radians unless a function documents otherwise.
- Rotation matrices are 3x3; homogeneous transforms are 4x4.
- Quaternions are `[w, x, y, z]`.
- RPY angles are `[roll, pitch, yaw]` for the ZYX composition
  `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
- Joint limits are always stored in radians, whatever the config file declares.
- `study/` is exploration: it may import the library, but never edits it. Experiments are modules run as
  `python3 -m study.<module>`, new experiment numbers are registered in `study/collab/QUESTIONS.md`
  before use, and conclusions are written up in `study/NOTES.md` / `study/REPORT.md`. Failures and
  retractions are recorded, not removed.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a merge request.

## License

[MIT](LICENSE)
