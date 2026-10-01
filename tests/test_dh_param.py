"""Tests for DH parameter handling and link-transform construction."""

from __future__ import annotations

import numpy as np
import pytest

from model.dh_param import (
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


def _is_transform(matrix: np.ndarray) -> bool:
    """Whether ``matrix`` is a valid 4x4 homogeneous transform."""
    if matrix.shape != (4, 4):
        return False
    rot = matrix[:3, :3]
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-12):
        return False
    return np.allclose(matrix[3, :], [0, 0, 0, 1])


class TestPrimitives:
    """The individual rotation and translation builders."""

    def test_rotations_are_orthonormal(self):
        for theta in (-np.pi, -0.3, 0.0, 0.7, np.pi / 2):
            for builder in (Rx, Ry, Rz):
                assert _is_transform(builder(theta))

    def test_rotations_match_axis_convention(self):
        # A rotation about X maps +Y onto +Z.
        assert np.allclose(Rx(np.pi / 2) @ [0, 1, 0, 0], [0, 0, 1, 0])
        # A rotation about Z maps +X onto +Y.
        assert np.allclose(Rz(np.pi / 2) @ [1, 0, 0, 0], [0, 1, 0, 0])
        # A rotation about Y maps +Z onto +X.
        assert np.allclose(Ry(np.pi / 2) @ [0, 0, 1, 0], [1, 0, 0, 0])

    def test_translations_place_the_offset(self):
        assert np.allclose(Tx(1.5)[:3, 3], [1.5, 0, 0])
        assert np.allclose(Ty(-2.0)[:3, 3], [0, -2.0, 0])
        assert np.allclose(Tz(0.25)[:3, 3], [0, 0, 0.25])


class TestDhTable:
    """Construction, validation and unit conversion of a DH table."""

    def test_default_order_map_is_d_alpha_a_theta(self):
        dh = Dh([[0.1, 0.2, 0.3, 0.4]])
        assert (dh.d_idx, dh.alpha_idx, dh.a_idx, dh.theta_idx) == (0, 1, 2, 3)

    def test_custom_order_map_is_honoured(self):
        dh = Dh([[0.4, 0.1, 0.2, 0.3]], order_map=["theta", "d", "alpha", "a"])
        assert (dh.theta_idx, dh.d_idx, dh.alpha_idx, dh.a_idx) == (0, 1, 2, 3)

    def test_rejects_unknown_convention(self):
        with pytest.raises(ValueError, match="dh_type"):
            Dh([[0, 0, 0, 0]], dh_type="mdhx")

    def test_rejects_incomplete_order_map(self):
        with pytest.raises(ValueError, match="order_map"):
            Dh([[0, 0, 0, 0]], order_map=["d", "a", "theta"])

    def test_order_map_attribute_is_not_shadowed_by_a_method(self):
        # Regression: the class used to define both an ``order_map`` attribute
        # and an ``order_map()`` method, so the attribute won and the method was
        # unreachable.
        dh = Dh([[0, 0, 0, 0]])
        assert dh.order_map == ["d", "alpha", "a", "theta"]
        assert callable(dh.print_order_map)

    def test_millimeter_to_meter_converts_translations_only(self):
        dh = Dh([[1000.0, 0.5, 2000.0, 0.25]])
        dh.millimeter_to_meter()
        row = dh.dh_list[0]
        assert row[dh.d_idx] == pytest.approx(1.0)
        assert row[dh.a_idx] == pytest.approx(2.0)
        # alpha and theta are angles and must be untouched.
        assert row[dh.alpha_idx] == pytest.approx(0.5)
        assert row[dh.theta_idx] == pytest.approx(0.25)


class TestTransformLists:
    """Conversion from a DH table to the ordered transform list."""

    def test_mdh_composition_order(self):
        dh = Dh([[0.3, 0.1, 0.2, 0.4]])
        expected = Tx(0.2) @ Rx(0.1) @ Tz(0.3) @ Rz(0.4)
        assert np.allclose(mdh_to_matrix_list(dh)[0], expected)

    def test_sdh_composition_order(self):
        dh = Dh([[0.3, 0.1, 0.2, 0.4]], dh_type="sdh")
        expected = Rz(0.4) @ Tz(0.3) @ Rx(0.1) @ Tx(0.2)
        assert np.allclose(sdh_to_matrix_list(dh)[0], expected)

    def test_mdh_list_is_base_links_tool(self):
        dh = Dh([[0.1, 0, 0.2, 0], [0.3, 0, 0.4, 0]])
        base = Tx(1.0)
        ee = Tz(2.0)
        ms = get_matrix_list(dh, base, ee)

        assert len(ms) == dh_len_expected(2)
        assert np.allclose(ms[0], base)
        assert np.allclose(ms[-1], ee)

    def test_sdh_folds_flange_offset_into_the_ee_entry(self):
        dh = Dh([[0.1, 0, 0.2, 0], [0.3, 0, 0.4, 0]], dh_type="sdh")
        ee = Tz(2.0)
        ms = get_matrix_list(dh, np.eye(4), ee)

        links = sdh_to_matrix_list(dh)
        # The last DH link transform must not appear on its own: it is composed
        # with the end-effector transform into the final entry.
        assert np.allclose(ms[-1], links[-1] @ ee)
        assert np.allclose(ms[1], np.eye(4))
        assert np.allclose(ms[2], links[0])

    def test_rejects_unsupported_type(self):
        dh = Dh([[0, 0, 0, 0]])
        dh.dh_type = "unknown"
        with pytest.raises(ValueError):
            get_matrix_list(dh, np.eye(4), np.eye(4))


def dh_len_expected(num_links: int) -> int:
    """Length of the transform list for an MDH robot with ``num_links`` joints."""
    return num_links + 2
