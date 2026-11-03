"""Check the exact predictor differentiation and malformed-certificate guards."""
from __future__ import annotations

from fractions import Fraction as Q

import pytest

from study import exp70_gate_c1_roots as core
from study.exp73_moving_tube import directional, evaluate, swept_box


def test_directional_laurent_second_derivative_keeps_complex_cross_terms():
    # For 1/z and v=1+i, D_v^2(1/z) at z=2 equals i/2, not zero.
    polynomial = {(-1, 0, 0, 0, 0, 0): (Q(1), Q(0))}
    velocity = ((Q(1), Q(1)),) + ((Q(0), Q(0)),)*5
    point = ((Q(2), Q(0)),)*6
    first = directional(polynomial, velocity)
    second = directional(first, velocity)
    assert core.lp_eval(first, point) == (Q(-1, 4), Q(-1, 4))
    assert core.lp_eval(second, point) == (Q(0), Q(1, 2))


def test_swept_box_contains_both_endpoint_slices():
    centre = tuple(Q(i+1) for i in range(12))
    velocity = tuple(Q((-1)**i*(i+1)) for i in range(12))
    h, radius = Q(1, 8), Q(1, 32)
    box = swept_box(centre, velocity, h, radius)
    for j in range(6):
        for k, interval in enumerate(box[j]):
            for s in (-h, Q(0), h):
                for error in (-radius, radius):
                    value = centre[2*j+k]+s*velocity[2*j+k]+error
                    assert interval.lo <= value <= interval.hi


@pytest.mark.parametrize('length', ['0', '-1/8', '1'])
def test_moving_certificate_rejects_invalid_archived_parameter_length(length):
    record = {'centre': ['1']*12, 'velocity': ['0']*12,
              'B': [['0']*12 for _ in range(12)], 'length': length}
    with pytest.raises(ValueError, match='length outside'):
        evaluate([], [], [], record)


def test_reversed_predictor_is_not_certified_at_the_good_tube_radius():
    # Same center, radius and B; reversing v destroys the small predictor defect.
    # This must fail the actual inequality, not a stored PASS comparison.
    import json

    from study.exp73_moving_tube import OUT, setup

    payload = json.loads(OUT.read_text())
    record = next(row for row in payload['trials'] if row['length'] == '1/16384')
    assert record['bounds']['passed']
    changed = {**record, 'velocity': [str(-Q(x)) for x in record['velocity']]}
    bounds = evaluate(*setup(), changed)
    assert not bounds['passed']
    assert Q(bounds['margin_relative']) < 0
