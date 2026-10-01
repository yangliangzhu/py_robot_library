"""Backwards-compatible aggregator module.

Historically every public name was pulled into one flat namespace here, which is
why the rest of the codebase began with wildcard imports. That is no longer how
the library is organised: import from :mod:`model` (or from the specific module)
instead.

This module is kept so that existing ``from model.common import ...`` code keeps
working.

Note:
    Unlike the original version, importing this module no longer calls
    :func:`numpy.set_printoptions`. A library should not change global printing
    state as a side effect of being imported.
"""

from __future__ import annotations

# Re-export the package's public API. The wildcards are deliberate: this module
# exists purely to forward names for backwards compatibility.
from . import *  # noqa: F401,F403
from . import __all__ as _package_all
from .configs.loader import load_robot_config
from .mrobotics import *  # noqa: F401,F403

__all__ = [*_package_all, "load_robot_config"]
