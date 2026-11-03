"""Check the physical cusp witness and the obstruction to its straight shortcut."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from study import cuspidal3r
from study.exp74_physical_cusp_unfolding import derive


@pytest.fixture(scope='module')
def witness():
    return derive()


def test_cusp_is_triple_not_the_old_quadruple_root(witness):
    assert witness['sturm_root_count'] == 1
    assert witness['IK_cusp_remainders_orders_0_1_2'] == ['0', '0', '0']
    lo, hi = map(sp.Rational, witness['IK_third_derivative_bound'])
    assert lo*hi > 0


def test_straight_shortcut_crosses_singularities_but_detour_does_not(witness):
    # Replacing the lower horizontal leg by the upper one must NOT be certified.
    t = sp.Symbol('t')
    e, z = sp.Rational(1, 10), sp.Rational(39, 25)
    s, c = 2*t/(1+t*t), (1-t*t)/(1+t*t)
    b = e+sp.Rational(3, 2)*s
    numerator = sp.fraction(sp.cancel((4+b*b-z*z)*(e*c+s)**2-(b*c)**2))[0]
    bounds = witness['nonsingular_switching_witness']['endpoint_t_intervals']
    outer = (sp.Rational(bounds[0][0]), sp.Rational(bounds[1][1]))
    assert sp.Poly(numerator, t).count_roots(*outer) == 2
    assert witness['nonsingular_switching_witness']['horizontal_sturm_root_count'] == 0


def test_distinct_joint_endpoints_agree_with_library_position(witness):
    rows = list(cuspidal3r.DEFAULT_ROWS)
    rows[2] = (2.0, np.pi/2, 0.1)
    robot = cuspidal3r.build(rows=rows)
    qs = np.asarray(witness['nonsingular_switching_witness']['approximate_joint_endpoints'])
    target = np.array([np.sqrt(399/1250), 0, 39/25])
    assert np.linalg.norm(qs[0]-qs[1]) > 0.05
    for q in qs:
        assert np.allclose(cuspidal3r.position(robot, q), target, atol=1e-7, rtol=0)
