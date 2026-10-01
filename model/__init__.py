"""Robot models and kinematics utilities.

The public API is re-exported here so that the common entry points can be
imported from the package root::

    >>> from model import RobotModelNumpy, IkType, ModelFactory  # doctest: +SKIP

The names are resolved lazily, on first attribute access. Importing
``model.ik_srs`` or ``model.dh_param`` therefore does not pull in CasADi, which
keeps the import cheap and lets code that only needs the DH or S-R-S helpers run
without it installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0"

#: Public name -> (module holding it, attribute name in that module).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {}


def _export(name: str, module: str, attribute: str | None = None) -> None:
    """Register ``name`` as a lazily resolved export of ``module``."""
    _LAZY_EXPORTS[name] = (module, attribute or name)


# DH parameters and link transforms (numpy only).
for _name in (
    "Dh",
    "Rx",
    "Ry",
    "Rz",
    "Tx",
    "Ty",
    "Tz",
    "mdh_to_matrix_list",
    "sdh_to_matrix_list",
    "get_matrix_list",
):
    _export(_name, ".dh_param")

# Inverse kinematics.
_export("IkType", ".ik_type")
for _name in ("IkStandard", "IkNullSpace", "IkNaive", "IkQp", "make_ik_solver"):
    _export(_name, ".ik_solver")

# Robot models.
_export("RobotModelBase", ".robot_model_base")
_export("DEFAULT_VIRTUAL_LINK", ".robot_model_base")
_export("RobotModelNumpy", ".robot_model_numpy")
_export("RobotModelCasadi", ".robot_model_casadi")
_export("ModelFactory", ".model_factory")

del _name

__all__ = [
    "__version__",
    # parameters
    "Dh",
    "Rx",
    "Ry",
    "Rz",
    "Tx",
    "Ty",
    "Tz",
    "mdh_to_matrix_list",
    "sdh_to_matrix_list",
    "get_matrix_list",
    # models
    "RobotModelBase",
    "RobotModelCasadi",
    "RobotModelNumpy",
    "ModelFactory",
    "DEFAULT_VIRTUAL_LINK",
    # inverse kinematics
    "IkType",
    "IkStandard",
    "IkNullSpace",
    "IkNaive",
    "IkQp",
    "make_ik_solver",
]


def __getattr__(name: str):
    """Resolve a public name on first access (PEP 562).

    Args:
        name: The attribute being looked up.

    Returns:
        The requested object.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value  # cache so later lookups skip this function
    return value


def __dir__() -> list[str]:
    """Return the public names, for tab completion and introspection."""
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from .dh_param import (  # noqa: F401
        Dh,
        Rx,
        Ry,
        Rz,
        Tx,
        Ty,
        Tz,
        get_matrix_list,
        mdh_to_matrix_list,
        sdh_to_matrix_list,
    )
    from .ik_solver import (  # noqa: F401
        IkNaive,
        IkNullSpace,
        IkQp,
        IkStandard,
        make_ik_solver,
    )
    from .ik_type import IkType  # noqa: F401
    from .model_factory import ModelFactory  # noqa: F401
    from .robot_model_base import DEFAULT_VIRTUAL_LINK, RobotModelBase  # noqa: F401
    from .robot_model_casadi import RobotModelCasadi  # noqa: F401
    from .robot_model_numpy import RobotModelNumpy  # noqa: F401
