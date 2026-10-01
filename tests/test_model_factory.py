"""Tests for the model factory and the shared model configuration logic."""

from __future__ import annotations

import numpy as np
import pytest

from model import IkType, ModelFactory, RobotModelBase, RobotModelCasadi, RobotModelNumpy
from model.configs.loader import list_robot_configs

#: Robots the factory is expected to know about.
KNOWN_ROBOTS = ("sr5", "sr5_v2", "er3", "er3_v2", "er3_plus", "nerv_er3", "aubo_c5")


class TestFactoryCreation:
    """Building models by name and backend."""

    @pytest.mark.parametrize("name", KNOWN_ROBOTS)
    @pytest.mark.parametrize("backend", ("numpy", "casadi"))
    def test_creates_every_documented_robot(self, name, backend):
        robot = ModelFactory.create(name, backend=backend)
        assert isinstance(robot, RobotModelBase)
        assert robot.num_dof in (6, 7)
        assert len(robot.Ms) == robot.num_dof + 2

    def test_default_backend_is_casadi(self):
        assert isinstance(ModelFactory.create("sr5"), RobotModelCasadi)

    def test_numpy_backend_is_selected(self):
        assert isinstance(ModelFactory.create("sr5", backend="numpy"), RobotModelNumpy)

    def test_unknown_robot_raises_with_the_known_names(self):
        with pytest.raises(KeyError, match="unknown robot"):
            ModelFactory.create("not_a_robot")

    def test_unknown_backend_raises_with_the_known_names(self):
        with pytest.raises(KeyError, match="unknown backend"):
            ModelFactory.create("sr5", backend="pytorch")

    def test_every_config_file_is_reachable_by_a_factory_name(self):
        # Guard against a YAML file being added without a factory entry.
        mapped = {config for config, _ in ModelFactory._ROBOTS.values()}
        for config_name in list_robot_configs():
            assert config_name in mapped, f"{config_name}.yaml has no factory entry"

    def test_custom_ik_type_is_forwarded(self):
        robot = ModelFactory.create("er3", backend="numpy", ik_type=IkType.IK_NULL)
        assert robot.ik_solver.name is IkType.IK_NULL


class TestFactoryCompatibility:
    """The documented legacy entry points keep working."""

    def test_create_helpers_match_create(self):
        assert ModelFactory.create_sr5("numpy", IkType.IK_STANDARD).num_dof == 6
        assert ModelFactory.create_er3("numpy", IkType.IK_STANDARD).num_dof == 7
        assert ModelFactory.create_er3_v2("numpy", IkType.IK_STANDARD).num_dof == 7
        assert ModelFactory.create_er3_plus("numpy", IkType.IK_STANDARD).num_dof == 7
        assert ModelFactory.create_nerv_er3("numpy", IkType.IK_STANDARD).num_dof == 7
        assert ModelFactory.create_aubo_c5("numpy", IkType.IK_STANDARD).num_dof == 6

    def test_factory_map_is_callable(self):
        # Historical API: factory_map[name](backend, ik_type).
        robot = ModelFactory.factory_map["er3"]("numpy", IkType.IK_STANDARD)
        assert isinstance(robot, RobotModelNumpy)
        assert robot.num_dof == 7

    def test_type_supported_lists_the_backends(self):
        assert set(ModelFactory.type_supported) == {"casadi", "numpy"}


class TestJointLimits:
    """Bound handling inherited from the base class."""

    def test_bounds_are_the_configured_ones(self):
        robot = ModelFactory.create("sr5", backend="numpy")
        assert robot.get_lower_bounds().shape == (6,)
        assert np.all(robot.get_lower_bounds() < robot.get_upper_bounds())

    def test_sr5_v2_limits_are_tighter_than_sr5(self):
        default = ModelFactory.create("sr5", backend="numpy")
        tightened = ModelFactory.create("sr5_v2", backend="numpy")

        assert np.all(tightened.get_lower_bounds() >= default.get_lower_bounds())
        assert np.all(tightened.get_upper_bounds() <= default.get_upper_bounds())
        # The second joint is the one that actually changed.
        assert tightened.get_upper_bounds()[1] < default.get_upper_bounds()[1]


class TestBaseClassBehaviour:
    """Validation performed by the shared model logic."""

    def test_bounds_must_have_matching_lengths(self):
        robot = RobotModelNumpy()
        with pytest.raises(ValueError, match="same length"):
            robot.set_bounds(np.zeros(6), np.ones(7))

    def test_lower_bound_above_upper_bound_is_rejected(self):
        robot = RobotModelNumpy()
        with pytest.raises(ValueError, match="lower bound"):
            robot.set_bounds(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    def test_clearing_bounds_is_allowed(self):
        robot = ModelFactory.create("sr5", backend="numpy")
        robot.set_bounds(None, None)
        assert robot.get_upper_bounds().size == 0

    def test_set_tool_before_build_is_rejected(self):
        with pytest.raises(RuntimeError, match="build"):
            RobotModelNumpy().set_tool(np.eye(4))

    def test_unsupported_param_type_is_rejected(self):
        robot = RobotModelNumpy()
        config = {"param": [], "lower": np.zeros(6), "upper": np.ones(6)}
        with pytest.raises(NotImplementedError, match="param type"):
            robot.build("screw", IkType.IK_STANDARD, config)

    def test_dh_param_type_requires_a_dh_table(self):
        robot = RobotModelNumpy()
        config = {"param": [np.eye(4)], "lower": np.zeros(6), "upper": np.ones(6)}
        with pytest.raises(TypeError, match="Dh"):
            robot.build("dh", IkType.IK_STANDARD, config)


class TestToolTransform:
    """Mounting a tool on the flange."""

    def test_tool_offsets_the_end_effector(self, backend):
        robot = ModelFactory.create("sr5", backend=backend)
        q = np.zeros(robot.num_dof)
        base_pose = robot.fk(q)

        offset = np.eye(4)
        offset[2, 3] = 0.1  # 10 cm along the tool Z axis
        robot.set_tool(offset)

        # The tool frame moved 10 cm along the base pose's Z axis.
        expected = base_pose @ offset
        assert np.allclose(robot.fk(q), expected, atol=1e-12)

    def test_mounting_twice_does_not_accumulate(self, backend):
        robot = ModelFactory.create("sr5", backend=backend)
        q = np.zeros(robot.num_dof)

        offset = np.eye(4)
        offset[2, 3] = 0.1
        robot.set_tool(offset)
        once = robot.fk(q)
        robot.set_tool(offset)
        twice = robot.fk(q)

        assert np.allclose(once, twice)

    def test_tool_is_removable(self, backend):
        robot = ModelFactory.create("sr5", backend=backend)
        q = np.zeros(robot.num_dof)
        original = robot.fk(q)

        offset = np.eye(4)
        offset[2, 3] = 0.1
        robot.set_tool(offset)
        robot.set_tool(np.eye(4))

        assert np.allclose(robot.fk(q), original, atol=1e-12)
