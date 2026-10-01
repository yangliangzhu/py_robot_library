"""Loading of robot descriptions from the bundled YAML files.

Each file in this package describes one robot: how its kinematics are given
(``mat`` for a list of link transforms, ``mdh``/``sdh`` for a DH table), the DH
rows, the joint limits, and the units used for both.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Union

import numpy as np
import yaml

from ..dh_param import Dh

__all__ = ["load_robot_config", "list_robot_configs"]

#: Parameter layouts a config file may declare. The DH conventions are written
#: with a ``_param`` suffix in the YAML files, e.g. ``mdh_param``.
_PARAM_TYPES = ("mat", "mdh", "sdh")


def _dh_convention(param_type: str) -> str:
    """Map a YAML ``param_type`` onto a :class:`~model.dh_param.Dh` convention.

    Args:
        param_type: The declared layout, e.g. ``"mdh_param"``.

    Returns:
        ``"mdh"`` or ``"sdh"``.
    """
    return param_type[: -len("_param")] if param_type.endswith("_param") else param_type

RobotParam = Union[list[np.ndarray], Dh]


def list_robot_configs() -> list[str]:
    """Return the names of the bundled robot configurations.

    Returns:
        Sorted config stems, without the ``.yaml`` suffix.
    """
    return sorted(
        entry.name[: -len(".yaml")]
        for entry in resources.files(__package__).iterdir()
        if entry.name.endswith(".yaml")
    )


def load_robot_config(name: str) -> dict[str, Any]:
    """Load a bundled robot configuration.

    Args:
        name: Config file stem, e.g. ``"rokae_er3"``. See
            :func:`list_robot_configs`.

    Returns:
        A mapping with:

        ``type``
            The declared parameter layout (``"mat"``, ``"mdh"`` or ``"sdh"``).
        ``param``
            A list of 4x4 link transforms for ``"mat"``, otherwise a
            :class:`~model.dh_param.Dh` table.
        ``upper`` / ``lower``
            Joint limits, always converted to radians.

    Raises:
        FileNotFoundError: If no configuration with that name is bundled.
        ValueError: If the file declares an unsupported ``param_type``, or a
            ``joint_unit`` other than radians or degrees.

    Example:
        >>> config = load_robot_config("rokae_er3")
        >>> config["type"]
        'dh'
    """
    config_file = f"{name}.yaml"
    try:
        resource = resources.files(__package__).joinpath(config_file)
    except (ModuleNotFoundError, AttributeError):  # pragma: no cover - defensive
        raise FileNotFoundError(f"no robot configuration named {name!r}") from None

    try:
        with resource.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no robot configuration named {name!r}; available: {list_robot_configs()}"
        ) from None

    param_type = cfg["param_type"]
    dh_type = _dh_convention(param_type)
    if param_type != "mat" and dh_type not in _PARAM_TYPES:
        raise ValueError(
            f"{config_file}: param_type must be 'mat' or one of "
            f"{['mdh_param', 'sdh_param']}, got {param_type!r}"
        )

    param: RobotParam
    if param_type == "mat":
        param = [np.array(matrix, dtype=float) for matrix in cfg["mat"]]
    else:
        dh = Dh(cfg["dh"], dh_type=dh_type)
        if cfg.get("dh_unit") == "millimeter":
            dh.millimeter_to_meter()
        param = dh

    joint_unit = cfg.get("joint_unit", "radian")
    if joint_unit == "degree":
        upper = np.radians(cfg["upper"])
        lower = np.radians(cfg["lower"])
    elif joint_unit in ("radian", "rad"):
        upper = np.array(cfg["upper"], dtype=float)
        lower = np.array(cfg["lower"], dtype=float)
    else:
        raise ValueError(f"{config_file}: unsupported joint_unit {joint_unit!r}")

    return {
        "type": param_type,
        "param": param,
        "upper": upper,
        "lower": lower,
    }
