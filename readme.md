# Robot Models Library

A Python library for robot models and tools, providing forward/inverse kinematics, DH parameter handling, and geometry utilities.

## Features

- **DH Parameter Handling**: Define robot kinematics using Denavit-Hartenberg parameters
- **Forward Kinematics (FK)**: Compute end-effector poses from joint angles
- **Inverse Kinematics (IK)**: Multiple solver implementations (standard, null space, SRS)
- **Dual Backends**: 
  - NumPy backend for fast prototyping
  - CasADi backend for symbolic computation and optimization
- **Geometry Utilities**: Rotation matrices, quaternions, homogeneous transforms

## Installation

```bash
pip install -e /home/yang/Personal/py_library
```

Or install from the package:

```bash
pip install -e .
```

## Dependencies

- numpy
- casadi
- pyyaml
- scipy

## Quick Start

```python
import numpy as np
from model import RobotModelNumpy

# Create a robot model
robot = RobotModelNumpy()

# Set DH parameters
dh_params = [
    [0, 0, 0, 0],       # [theta, d, a, alpha]
    [0, 0, 0, -np.pi/2],
    [0, 0.612, 0, 0],
    [0, 0, 0, np.pi/2],
    [0, 0.5723, 0, -np.pi/2],
    [0, 0, 0, np.pi/2],
]
robot.set_dh_param(dh_params)
robot.build()

# Forward kinematics
q = np.array([0, 0, 0, 0, 0, 0])
pose = robot.fk(q)
print("End-effector pose:", pose)

# Jacobian
J = robot.jacobian(q)
print("Jacobian:", J)

# Inverse kinematics
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.3, 0.8]
solution, success = robot.ik(target_pose)
print("IK solution:", solution)
```

## Module Overview

### model/

| File | Description |
|------|-------------|
| `dh_param.py` | DH parameter handling and conversion |
| `robot_model_numpy.py` | Robot model using NumPy backend |
| `robot_model_casadi.py` | Robot model using CasADi backend |
| `ik_solver.py` | Base IK solver classes |
| `ik_type.py` | IK solver type definitions |
| `ik_srs.py` | SRS (Singularity-Robust Inverse) IK solver |
| `model_factory.py` | Factory for creating robot models |
| `common.py` | Common utilities for robot models |

### tools/

| File | Description |
|------|-------------|
| `geometry.py` | Rotation matrices, quaternions, transforms |
| `manip_analysis.py` | Manipulability analysis |
| `generate_pos_traj_for_sim.py` | Trajectory generation for simulation |
| `log_process.py` | Log file processing utilities |
| `time_mask.py` | Time mask utilities |

## Geometry Utilities

```python
from tools.geometry import rot_x, rot_y, rot_z, quat2rot, rot2quat

# Rotation matrices
Rx = rot_x(np.pi / 2)  # 90° rotation around X
Ry = rot_y(np.pi / 4)  # 45° rotation around Y

# Quaternion to rotation matrix
q = [1, 0, 0, 0]  # [w, x, y, z]
R = quat2rot(q)

# Rotation matrix to quaternion
q = rot2quat(R)
```

## IK Solvers

The library provides multiple IK solver implementations:

- **Standard IK**: Basic inverse kinematics
- **Null Space IK**: Exploits null space for redundant robots
- **SRS IK**: Singularity-Robust Inverse solver

```python
from model import RobotModelNumpy
from model.ik_type import IkType

robot = RobotModelNumpy()
robot.set_dh_param(dh_params)
robot.build()

# Use specific IK solver
robot.use_ik_solver(IkType.IK_NULL)
solution, success = robot.ik(target_pose)
```

## Mathematical Conventions

- Angles: radians (default)
- Matrices: 4x4 homogeneous transforms, 3x3 rotation matrices
- Vectors: Column vectors, NumPy arrays
- Tolerance: Use `np.allclose()` for floating-point comparisons

## License

MIT
