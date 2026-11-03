"""Check exact arithmetic certificates and reject a mismatched FK target."""
from __future__ import annotations

import pytest
import sympy as sp

from study.exp76_sr5_arithmetic_certificate import (
    compute,
    full_modular_reconstruction,
    rabin_irreducible,
)


@pytest.fixture(scope='module')
def certificate():
    return compute()


def test_degree_and_characteristic_zero_full_fk(certificate):
    assert certificate['degree'] == 16
    assert certificate['generic_prefix_geometry_checked']
    assert certificate['generic_degree_bounds_in_r_nr'] == [4, 3, 8]
    assert set(certificate['exceptional_gcd_degrees'].values()) == {0}
    exact = certificate['full_characteristic_zero_reconstruction']
    assert exact['prime'] == 0 and exact['full_FK_equals_target']
    assert exact['all_joint_norms']


def test_modular_witness_types_and_rabin_reducible_control(certificate):
    assert certificate['modular']['41']['degrees'] == [16]
    assert certificate['modular']['43']['degrees'] == [3, 13]
    x = sp.Symbol('x')
    assert not rabin_irreducible(sp.Poly((x*x+1)*(x*x+2), x, modulus=41), 41)
    assert rabin_irreducible(sp.Poly(x*x+2, x, modulus=5), 5)


def test_full_reconstruction_rejects_wrong_target(certificate):
    t = sp.Symbol('t')
    polynomial = sp.Poly.from_list(list(map(sp.Integer, certificate['polynomial_coefficients'])), t)
    kval = sp.sympify(certificate['recovery_k_numerator'])/sp.sympify(certificate['recovery_k_denominator'])
    rotation = sp.Matrix([[sp.Rational(x) for x in row] for row in certificate['target_rotation']])
    position = sp.Matrix(list(map(sp.Rational, certificate['target_position'])))
    position[0] += sp.Rational(1, 1000)
    with pytest.raises(AssertionError):
        full_modular_reconstruction(polynomial, t, -kval, 1, rotation, position, prime=0)
