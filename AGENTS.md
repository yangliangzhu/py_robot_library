# AGENTS.md - Agentic Coding Guidelines

## Project Overview

This is a Python library for robot models and tools, including:
- DH parameter handling
- Forward/inverse kinematics (FK/IK)
- Robot model implementations (numpy and CasADi)
- Geometry utilities (rotations, quaternions, transforms)

**Dependencies**: numpy, casadi, scipy

---

## Build / Test Commands

### Running the Project
```bash
# Install in development mode
pip install -e .

# Install all dependencies
pip install numpy casadi scipy
```

### Testing
There is **no formal test framework** configured. To run a single test or module:

```bash
# Run a specific Python file directly
python model/robot_model_numpy.py

# Run specific module with pytest (if tests added)
pytest tests/                    # Run all tests
pytest tests/test_file.py        # Run specific test file
pytest tests/test_file.py::test_function  # Run single test function
```

### Linting & Formatting
No linting or formatting tools are configured. If you add them, use:

```bash
# Black (formatting)
black .

# Ruff (linting)
ruff check .
ruff check --fix .

# MyPy (type checking)
mypy .
```

---

## Code Style Guidelines

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `RobotModelNumpy`, `IkStandard` |
| Functions | snake_case | `rot_x()`, `fk()`, `jacobian()` |
| Variables | snake_case | `num_dof`, `target_pose` |
| Constants | UPPER_SNAKE | `MAX_ITER`, `EPS` |
| Enum values | UPPER_SNAKE | `IK_STANDARD`, `IK_NULL` |

### Imports

**Standard format** (one per line):
```python
import numpy as np
import casadi as ca
from scipy.differentiate import jacobian

from .module_name import ClassName
from .module_name import function_name, CONSTANT
```

**Avoid wildcard imports** (`from .module import *`). Use explicit imports.

**Order** (grouped, separated by blank line):
1. Standard library
2. Third-party packages (numpy, casadi, scipy)
3. Local application imports

### Type Hints

**Currently not used** in this codebase. If adding:
```python
# Good
def jacobian(self, q: np.ndarray) -> np.ndarray:
    ...

def set_bounds(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
    ...
```

### Documentation

Use docstrings for all public classes and functions. Two styles observed:

**Google style** (preferred for complex functions):
```python
def quat2rot(q):
    """
    Convert a quaternion to rotation matrix.

    Parameters:
        q (numpy.ndarray): quaternion [w, x, y, z]

    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
```

**Simple style** (acceptable for short functions):
```python
def rot_z(theta):
    """Return rotation matrix around Z axis."""
```

Include Chinese comments where helpful for domain terminology.

### Error Handling

Use specific exception types:
```python
# For unsupported features
raise NotImplementedError("param type not supported")

# For invalid values
raise ValueError("soft upper must be >= soft lower")

# For dimension mismatches
raise ValueError("Input must be a 3x3 rotation matrix")
```

### Code Organization

**Class structure**:
```python
class RobotModelNumpy():
    """Brief description of the class."""

    def __init__(self):
        # Initialize instance variables
        self.virtual_link = 0.15

    # Group methods by functionality with comment headers
    ########################## configure robot ########################################

    def build(self, ...):
        ...

    ########################## kinematics #############################################

    def fk(self, q):
        """Forward kinematics."""
        ...
```

**Main guard**:
```python
if __name__ == '__main__':
    # Test code or demo
    pass
```

### Mathematical Conventions

- **Units**: Angles in radians by default
- **Matrices**: 4x4 homogeneous transforms, 3x3 rotation matrices
- **Vectors**: Column vectors, numpy arrays
- **Tolerance**: Use `np.allclose()` for floating-point comparisons with appropriate `atol`/`rtol`

### Code Patterns

**Factory method pattern** (observed):
```python
def ik_factory_method(self, solver_type):
    if solver_type == IkType.IK_STANDARD:
        self.ik_solver = IkStandard(self, solver_type)
    elif solver_type == IkType.IK_NORMAL:
        ...
```

**Callable classes** (for IK solvers):
```python
class IkStandard:
    def __call__(self, angle, target):
        """Solve IK - can be called as function."""
        ...
```

---

## Common Tasks

### Adding a New Robot Model
1. Create `model/robot_model_<backend>.py`
2. Inherit from base class or implement required interface
3. Implement: `fk()`, `jacobian()`, `ik()`
4. Add to `model_factory.py` if applicable

### Adding a New IK Solver
1. Create new class in `model/ik_solver.py`
2. Implement `__call__(self, angle, target)` returning `(solution, success)`
3. Add enum value to `model/ik_type.py`
4. Register in `ik_factory_method()`

### Adding Geometry Utilities
1. Add to `tools/geometry.py`
2. Follow existing patterns for rotation matrices, quaternions, etc.

---

## Architecture Notes

- **Two backends**: numpy (fast prototyping) and CasADi (symbolic/optimization)
- **IK solvers**: Multiple strategies (standard, null space, QP-based)
- **Separation**: Model definition (DH/matrices) from solvers
- **Config-driven**: Robot parameters passed via config dictionaries
