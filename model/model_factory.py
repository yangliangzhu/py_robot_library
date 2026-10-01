"""Factory for the robot models shipped with this library.

Robot descriptions live in ``model/configs/*.yaml``; this module maps a short
robot name and a backend name onto a configured model instance::

    >>> from model.model_factory import ModelFactory
    >>> robot = ModelFactory.create("er3", backend="numpy")  # doctest: +SKIP
    >>> robot.num_dof  # doctest: +SKIP
    7
"""

from __future__ import annotations

import numpy as np

from .configs.loader import load_robot_config
from .ik_type import IkType
from .robot_model_base import RobotModelBase
from .robot_model_casadi import RobotModelCasadi
from .robot_model_numpy import RobotModelNumpy

__all__ = ["ModelFactory"]


class ModelFactory:
    """Build configured robot models by name.

    Attributes:
        factory_map: Mapping of robot name to the ``create_*`` method that
            builds it. Kept for backwards compatibility; prefer :meth:`create`.
        type_supported: The accepted backend names.
    """

    #: Backend name -> model class.
    _BACKENDS: dict[str, type[RobotModelBase]] = {
        "casadi": RobotModelCasadi,
        "numpy": RobotModelNumpy,
    }

    #: Robot name -> (config file stem, parameter layout).
    _ROBOTS: dict[str, tuple[str, str]] = {
        "sr5": ("rokae_sr5", "mat"),
        "sr5_v2": ("rokae_sr5", "mat"),
        "er3": ("rokae_er3", "dh"),
        "er3_v2": ("rokae_er3_expand", "dh"),
        "er3_plus": ("rokae_er3_plus", "dh"),
        "er3_sdh": ("rokae_er3_sdh", "dh"),
        "nerv_er3": ("nerv_er3", "dh"),
        "aubo_c5": ("aubo_c5", "dh"),
        "franka": ("franka", "dh"),
    }

    #: Joint limits for the "v2" SR5 variant, which are tighter than the ones in
    #: the YAML description.
    _SR5_V2_BOUNDS_DEG = (
        (-175, -135, -170, -175, -175, -175),
        (175, 135, 140, 175, 175, 175),
    )

    factory_map: dict[str, object] = {}
    type_supported = list(_BACKENDS)

    @classmethod
    def create(
        cls,
        name: str,
        backend: str = "casadi",
        ik_type: IkType = IkType.IK_STANDARD,
    ) -> RobotModelBase:
        """Build a robot model by name.

        Args:
            name: One of the keys of :attr:`_ROBOTS`, e.g. ``"er3"``.
            backend: ``"casadi"`` or ``"numpy"``.
            ik_type: The IK strategy the model will use.

        Returns:
            The configured model.

        Raises:
            KeyError: If ``name`` or ``backend`` is unknown.

        Example:
            >>> robot = ModelFactory.create("sr5", backend="numpy")  # doctest: +SKIP
        """
        try:
            config_name, param_type = cls._ROBOTS[name]
        except KeyError:
            raise KeyError(
                f"unknown robot {name!r}; known robots: {sorted(cls._ROBOTS)}"
            ) from None
        try:
            model_class = cls._BACKENDS[backend]
        except KeyError:
            raise KeyError(
                f"unknown backend {backend!r}; known backends: {sorted(cls._BACKENDS)}"
            ) from None

        config = load_robot_config(config_name)
        if name == "sr5_v2":
            lower_deg, upper_deg = cls._SR5_V2_BOUNDS_DEG
            config["lower"] = np.radians(lower_deg)
            config["upper"] = np.radians(upper_deg)

        robot = model_class()
        robot.build(param_type, ik_type, config)
        return robot

    @staticmethod
    def create_sr5(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE SR5 (matrix parameter description)."""
        return ModelFactory.create("sr5", backend, ik_type)

    @staticmethod
    def create_sr5_v2(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE SR5 with the tightened v2 joint limits."""
        return ModelFactory.create("sr5_v2", backend, ik_type)

    @staticmethod
    def create_er3(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE ER3 (DH parameter description)."""
        return ModelFactory.create("er3", backend, ik_type)

    @staticmethod
    def create_er3_v2(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE ER3 using the expanded DH description."""
        return ModelFactory.create("er3_v2", backend, ik_type)

    @staticmethod
    def create_er3_plus(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE ER3 Plus."""
        return ModelFactory.create("er3_plus", backend, ik_type)

    @staticmethod
    def create_nerv_er3(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build the nerv ER3 variant."""
        return ModelFactory.create("nerv_er3", backend, ik_type)

    @staticmethod
    def create_aubo_c5(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build an AUBO C5."""
        return ModelFactory.create("aubo_c5", backend, ik_type)

    @staticmethod
    def create_franka(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a Franka Emika Panda."""
        return ModelFactory.create("franka", backend, ik_type)

    @staticmethod
    def create_er3_sdh(backend: str, ik_type: IkType) -> RobotModelBase:
        """Build a ROKAE ER3 described with standard (not modified) DH parameters."""
        return ModelFactory.create("er3_sdh", backend, ik_type)


# Populated after the class body so the lambdas can close over the class itself.
# Retained for backwards compatibility with callers that used
# ``ModelFactory.factory_map[name](backend, ik_type)``; prefer ModelFactory.create.
ModelFactory.factory_map = {
    name: (
        lambda backend, ik_type, _name=name: ModelFactory.create(_name, backend, ik_type)
    )
    for name in ModelFactory._ROBOTS
}


if __name__ == "__main__":
    sr5 = ModelFactory.create("sr5")
    er3 = ModelFactory.create("er3", "numpy")
