"""Tests for the robot configuration loader."""

from __future__ import annotations

import numpy as np
import pytest

from model.configs.loader import list_robot_configs, load_robot_config
from model.dh_param import Dh


class TestListing:
    """Discovery of the bundled configurations."""

    def test_returns_the_shipped_configs(self):
        names = list_robot_configs()
        assert "rokae_er3" in names
        assert "rokae_sr5" in names
        assert names == sorted(names)

    def test_every_listed_config_loads(self):
        for name in list_robot_configs():
            cfg = load_robot_config(name)
            assert "param" in cfg
            assert "lower" in cfg
            assert "upper" in cfg


class TestLoading:
    """Parsing of an individual configuration."""

    def test_dh_config_yields_a_dh_table(self):
        cfg = load_robot_config("rokae_er3")
        assert isinstance(cfg["param"], Dh)
        assert cfg["param"].dh_type == "mdh"
        assert len(cfg["param"].dh_list) == 7

    def test_matrix_config_yields_transforms(self):
        cfg = load_robot_config("rokae_sr5")
        assert isinstance(cfg["param"], list)
        assert len(cfg["param"]) == 6
        for matrix in cfg["param"]:
            assert matrix.shape == (4, 4)

    def test_lists_the_configs_when_the_name_is_unknown(self):
        with pytest.raises(FileNotFoundError):
            load_robot_config("does_not_exist")

    def test_units_are_normalised_to_radians(self):
        # Every shipped config declares its joint limits in degrees, so the
        # loaded values must be the radian equivalents. (A joint with a +/-360
        # degree range is legal, hence the 2*pi bound.)
        cfg = load_robot_config("rokae_sr5")
        assert np.all(np.abs(cfg["upper"]) <= 2 * np.pi + 1e-9)
        assert np.all(cfg["lower"] <= cfg["upper"])
        # 360 degrees -> 2*pi rad, not 360.
        assert np.isclose(cfg["upper"][0], np.radians(360))
        assert np.isclose(cfg["upper"][1], np.radians(150))

    def test_degrees_are_not_left_as_degrees(self):
        cfg = load_robot_config("franka")
        assert np.max(np.abs(cfg["upper"])) < 2 * np.pi + 1e-9

    def test_millimetre_dh_is_converted_to_metres(self):
        cfg = load_robot_config("rokae_er3")
        dh = cfg["param"]
        # Link lengths of a 7-DOF arm are tens of centimetres, not metres.
        lengths = [abs(row[dh.a_idx]) for row in dh.dh_list]
        assert max(lengths) < 1.0

    def test_sdh_config_is_recognised(self):
        cfg = load_robot_config("rokae_er3_sdh")
        assert isinstance(cfg["param"], Dh)
        assert cfg["param"].dh_type == "sdh"


class TestValidation:
    """Rejection of malformed configuration files."""

    def test_unsupported_param_type_is_rejected(self, tmp_path, monkeypatch):
        import model.configs.loader as loader

        bad = tmp_path / "broken.yaml"
        bad.write_text("param_type: bogus\nupper: [1]\nlower: [0]\n", encoding="utf-8")

        class _FakeTraversable:
            name = "broken.yaml"

            def open(self, *args, **kwargs):
                return open(bad, *args, **kwargs)

        class _FakeFiles:
            def joinpath(self, _name):
                return _FakeTraversable()

            def iterdir(self):
                return [_FakeTraversable()]

        monkeypatch.setattr(loader.resources, "files", lambda _pkg: _FakeFiles())

        with pytest.raises(ValueError, match="param_type"):
            loader.load_robot_config("broken")

    def test_unsupported_joint_unit_is_rejected(self, tmp_path, monkeypatch):
        import model.configs.loader as loader

        bad = tmp_path / "broken.yaml"
        bad.write_text(
            "param_type: mat\nmat: []\njoint_unit: gradians\nupper: [1]\nlower: [0]\n",
            encoding="utf-8",
        )

        class _FakeTraversable:
            def open(self, *args, **kwargs):
                return open(bad, *args, **kwargs)

        class _FakeFiles:
            def joinpath(self, _name):
                return _FakeTraversable()

        monkeypatch.setattr(loader.resources, "files", lambda _pkg: _FakeFiles())

        with pytest.raises(ValueError, match="joint_unit"):
            loader.load_robot_config("broken")
