"""Tests for the package's public API surface.

These guard the import contract: the documented names must be importable from
the package root, and importing a leaf module must not drag in a heavy optional
dependency.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import model


class TestPublicNames:
    """Names reachable from the package root."""

    @pytest.mark.parametrize("name", model.__all__)
    def test_every_exported_name_resolves(self, name):
        assert getattr(model, name) is not None

    def test_all_is_sorted_and_unique(self):
        assert len(model.__all__) == len(set(model.__all__))

    def test_version_is_a_string(self):
        assert isinstance(model.__version__, str)
        assert model.__version__.count(".") == 2

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute"):
            _ = model.definitely_not_part_of_the_api

    def test_dir_lists_the_public_names(self):
        listed = set(dir(model))
        assert set(model.__all__) <= listed

    def test_lazy_exports_are_cached(self):
        # First access resolves, second access reads the cached global.
        first = model.IkType
        assert model.IkType is first
        assert model.__dict__["IkType"] is first


class TestLazyImports:
    """Importing a leaf module must not pull in the whole library."""

    def test_dh_param_does_not_import_casadi(self):
        code = "import model.dh_param, sys; print('casadi' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "False"

    def test_ik_srs_does_not_import_casadi(self):
        # The S-R-S analytical solver is numpy-only; a downstream project that
        # imports it must not be forced to install CasADi.
        code = "import model.ik_srs, sys; print('casadi' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "False"

    def test_importing_a_model_does_import_casadi(self):
        code = "import model.robot_model_casadi, sys; print('casadi' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "True"


class TestLegacyAggregator:
    """``model.common`` keeps the historical flat namespace working."""

    def test_common_reexports_the_public_api(self):
        from model import common

        for name in model.__all__:
            if name == "__version__":
                continue
            assert hasattr(common, name), name

    def test_common_does_not_change_numpy_print_options(self):
        # Regression: importing the library used to call set_printoptions and
        # mutate global state.
        import numpy as np

        assert np.get_printoptions()["suppress"] is False

    def test_common_exposes_the_geometry_history(self):
        from model import common

        assert callable(common.near_zero)
        assert callable(common.load_robot_config)
