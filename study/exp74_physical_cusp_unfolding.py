"""Prove a physical 3R perturbation has an ordinary cusp and quartic position IK."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from study import cuspidal3r
from study import exp70_gate_c1_roots as iv
from study.exp61_source_germs import link, rotation

OUT = Path('study/exp74_physical_cusp_unfolding.json')


def rational_interval(expr: sp.Expr, variable: sp.Symbol, lo: sp.Rational,
                      hi: sp.Rational) -> iv.RInterval:
    """Enclose a rational function on a rational interval using exact Horner evaluation."""
    def polynomial(poly):
        result = iv.ri_point(Fraction(0))
        x = iv.RInterval(Fraction(lo), Fraction(hi))
        for coefficient in sp.Poly(poly, variable).all_coeffs():
            result = iv.ri_add(iv.ri_mul(result, x), iv.ri_point(Fraction(coefficient)))
        return result
    numerator, denominator = sp.fraction(sp.cancel(expr))
    return iv.ri_mul(polynomial(numerator), iv.ri_recip(polynomial(denominator)))



def switching_witness() -> dict:
    """Prove two regular IK roots share an image and connect through three regular legs."""
    t = sp.Symbol('t')
    e, top, bottom = sp.Rational(1, 10), sp.Rational(39, 25), sp.Rational(77, 50)
    target_r = sp.Rational(399, 1250)
    sn, cs = 2*t/(1+t*t), (1-t*t)/(1+t*t)
    b = e+sp.Rational(3, 2)*sn
    d = target_r+top*top-sp.Rational(33, 4)-e*e
    recovered_a = (d-3*e*sn+3*cs)/2
    ik = sp.Poly(25807364*t**4+3050250*t**3-2184647*t**2+237750*t-7636, t)
    identity = sp.cancel((1+t*t)**2*(recovered_a**2+top*top-4-b*b))
    assert sp.rem(sp.fraction(identity)[0], ik.as_expr(), t) == 0
    left = (sp.Rational(7336875, 10**8), sp.Rational(7336876, 10**8))
    right = (sp.Rational(11720485, 10**8), sp.Rational(11720487, 10**8))
    for interval in (left, right):
        assert ik.count_roots(*interval) == 1
        assert rational_interval(recovered_a, t, *interval).hi < 0
    outer = (left[0], right[1])
    # Sign of H_v = 3*(e*cos(v)+sin(v)-B*cos(v)/sqrt(4+B²-z²)).
    # All factors used when squaring are positive over the ENTIRE rectangle.
    positive = {'sqrt_argument_min_at_top': 4+b*b-top*top,
                'e_cos_plus_sin': e*cs+sn, 'B_cos': b*cs,
                'local_y': sp.Rational(3, 2)*cs-1}
    sign_bounds = {name: rational_interval(value, t, *outer)
                   for name, value in positive.items()}
    assert all(value.lo > 0 for value in sign_bounds.values())
    def sign_polynomial(z):
        return sp.cancel((4+b*b-z*z)*(e*cs+sn)**2-(b*cs)**2)
    endpoint_bounds = [rational_interval(sign_polynomial(top), t, *interval)
                       for interval in (left, right)]
    assert all(value.lo > 0 for value in endpoint_bounds)
    # For fixed t, decreasing positive z increases this expression strictly.
    horizontal_numerator = sp.Poly(sp.fraction(sign_polynomial(bottom))[0], t)
    assert horizontal_numerator.count_roots(*outer) == 0
    assert sign_polynomial(bottom).subs(t, sum(outer)/2) > 0

    endpoints = []
    for interval in (left, right):
        # Display only; proof uses algebraic roots isolated above.
        tv = float(sum(interval)/2)
        bv, av = float(b.subs(t, tv)), float(recovered_a.subs(t, tv))
        su = (2*av+bv*float(top))/(4+bv*bv)
        cu = (bv*av-2*float(top))/(4+bv*bv)
        q1 = -np.arctan2(float((sp.Rational(3, 2)*cs-1).subs(t, tv)), 1+av)
        endpoints.append([float(q1), float(np.arctan2(su, cu)), float(2*np.arctan(tv))])
    return {'target': ['sqrt(399/1250)', '0', '39/25'],
            'IK_polynomial': str(ik.as_expr()),
            'endpoint_t_intervals': [[str(x) for x in interval] for interval in (left, right)],
            'path_zt': [['39/25', 't_left'], ['77/50', 't_left'],
                        ['77/50', 't_right'], ['39/25', 't_right']],
            'rectangle_positive_bounds': {name: [str(value.lo), str(value.hi)]
                                          for name, value in sign_bounds.items()},
            'endpoint_Hv_sign_bounds': [[str(value.lo), str(value.hi)] for value in endpoint_bounds],
            'horizontal_Hv_sign_numerator': str(horizontal_numerator.as_expr()),
            'horizontal_sturm_root_count': 0,
            'approximate_joint_endpoints': endpoints}


def derive() -> dict:
    """Recompute the physical FK, cusp identities, Sturm count and nondegeneracy bounds."""
    u, v, e, z = sp.symbols('u v e z', real=True)
    b = e + sp.Rational(3, 2)*sp.sin(v)
    a = 2*sp.sin(u)+b*sp.cos(u)
    height = b*sp.sin(u)-2*sp.cos(u)
    position = sp.Matrix([1+a, sp.Rational(3, 2)*sp.cos(v)-1, height])
    product = (link(1, 1)*rotation(u)*link(e, 2)*rotation(v)*link(0, sp.Rational(3, 2)))[:3, 3]
    assert all(sp.trigsimp(x) == 0 for x in product-position)
    assert sp.trigsimp(a*a+height*height-4-b*b) == 0
    assert sp.simplify(sp.diff(height, u)-a) == 0

    # Negative A branch, regular source coordinate change (u,v)->(z,v).
    h = sp.Rational(33, 4)+e*e+3*e*sp.sin(v)-z*z-2*sp.sqrt(4+b*b-z*z)-3*sp.cos(v)
    point = {v: 0, e: 0, z: sp.sqrt(7)/2}
    jets = {
        'H_v_e': sp.simplify(sp.diff(h, v, e).subs(point)),
        'H_vvvv': sp.simplify(sp.diff(h, v, 4).subs(point)),
        'H_zvv': sp.simplify(sp.diff(h, z, v, 2).subs(point)),
    }
    assert jets == {'H_v_e': 1, 'H_vvvv': 18, 'H_zvv': -2*sp.sqrt(7)/3}
    for derivative in (sp.diff(h, v), sp.diff(h, v, 2), sp.diff(h, v, 3)):
        assert sp.simplify(derivative.subs(point)) == 0

    # v derivatives at fixed z, expressed rationally in sin(v), cos(v), A.
    s, c, aa = sp.symbols('s c A')
    bb = e+sp.Rational(3, 2)*s
    def dv(expr):
        return sp.diff(expr, s)*c-sp.diff(expr, c)*s+sp.diff(expr, aa)*bb*sp.Rational(3, 2)*c/aa
    h1 = 3*(e*c+s+bb*c/aa)
    h2, h3 = dv(h1), dv(dv(h1))
    t = sp.Symbol('t')
    sn, cs = 2*t/(1+t*t), (1-t*t)/(1+t*t)
    ee = sp.Rational(1, 10)
    bn = ee+sp.Rational(3, 2)*sn
    an = -bn*cs/(ee*cs+sn)
    substitution = {s: sn, c: cs, e: ee, aa: an}
    f1, f2, f3 = [sp.cancel(expr.subs(substitution, simultaneous=True)) for expr in (h1, h2, h3)]
    assert f1 == 0
    polynomial = sp.Poly(497*t**6+180*t**5-3891*t**4+47640*t**3+5091*t**2+180*t-97, t)
    assert sp.rem(sp.fraction(f2)[0], polynomial.as_expr(), t) == 0
    lo, hi = sp.Rational(92228741610, 10**12), sp.Rational(92228741612, 10**12)
    count = polynomial.count_roots(lo, hi)
    assert count == 1 and polynomial.eval(lo)*polynomial.eval(hi) < 0
    z2 = sp.cancel(4+bn*bn-an*an)
    radius2 = sp.cancel((1+an)**2+(sp.Rational(3, 2)*cs-1)**2)
    mixed_over_z = sp.cancel(3*bn*cs/an**3)
    expressions = {'A': an, 'z_squared': z2, 'R': radius2,
                   'H_vvv': f3, 'H_zv_over_z': mixed_over_z}
    bounds = {name: rational_interval(expr, t, lo, hi) for name, expr in expressions.items()}
    assert bounds['A'].hi < 0
    assert bounds['z_squared'].lo > 0 and bounds['R'].lo > 0
    for name in ('H_vvv', 'H_zv_over_z'):
        assert bounds[name].lo*bounds[name].hi > 0

    # Generic IK numerator in k=tan(v/2). This is quartic for every fixed e.
    k, rr, zz = sp.symbols('k R Z')
    sk, ck = 2*k/(1+k*k), (1-k*k)/(1+k*k)
    dd = rr+zz-sp.Rational(33, 4)-e*e
    numerator = sp.cancel((1+k*k)**2*((dd-3*e*sk+3*ck)**2+4*zz-16-4*(e+sp.Rational(3, 2)*sk)**2))
    assert sp.denom(numerator) == 1 and sp.degree(numerator, k) == 4
    # Exact triple-root check at the algebraic cusp (quotient by polynomial(t)).
    cusp_numerator = numerator.subs({e: ee, rr: radius2, zz: z2})
    triple_checks = []
    for order in range(3):
        value = sp.cancel(sp.diff(cusp_numerator, k, order).subs(k, t))
        remainder = sp.rem(sp.fraction(value)[0], polynomial.as_expr(), t)
        assert remainder == 0
        triple_checks.append(str(remainder))
    third = sp.cancel(sp.diff(cusp_numerator, k, 3).subs(k, t))
    third_bound = rational_interval(third, t, lo, hi)
    assert third_bound.lo*third_bound.hi > 0

    # Auxiliary numeric agreement with the actual library; not part of the proof.
    rows = list(cuspidal3r.DEFAULT_ROWS)
    rows[2] = (2.0, np.pi/2, 0.1)
    model = cuspidal3r.build(rows=rows)
    numeric = sp.lambdify((u, v, e), position, 'numpy')
    rng = np.random.default_rng(74)
    worst = 0.0
    for q in rng.uniform(-np.pi, np.pi, (100, 3)):
        local = np.asarray(numeric(q[1], q[2], .1), dtype=float).reshape(3)
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        turn = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        worst = max(worst, float(np.max(np.abs(turn@local-cuspidal3r.position(model, q)))))
    assert worst < 1e-12
    mid = (lo+hi)/2
    numeric_point = {name: float(expr.subs(t, mid)) for name, expr in expressions.items()}
    result = {
        'geometry': {'rows_d_alpha_a': [['0','0','0'], ['1','pi/2','1'], ['2','pi/2','1/10']],
                     'tool_d_alpha_a': ['3/2','pi/2','0']},
        'cusp_polynomial': str(polynomial.as_expr()), 'isolating_interval': [str(lo), str(hi)],
        'sturm_root_count': int(count), 'jets': {key: str(value) for key, value in jets.items()},
        'strict_bounds': {key: [str(value.lo), str(value.hi)] for key, value in bounds.items()},
        'IK_quartic': str(sp.collect(sp.expand(numerator), k)),
        'IK_cusp_remainders_orders_0_1_2': triple_checks,
        'IK_third_derivative_bound': [str(third_bound.lo), str(third_bound.hi)],
        'approximate_cusp': {'t': float(mid), **numeric_point},
        'library_FK_max_error': worst,
        'nonsingular_switching_witness': switching_witness(),
    }
    return result


def main() -> None:
    """Generate the exact algebraic witness or independently recompute its fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    result = derive()
    if args.verify:
        stored = json.loads(args.verify.read_text())
        for key in result:
            if key not in ('approximate_cusp', 'library_FK_max_error') and stored[key] != result[key]:
                raise ValueError(f'certificate mismatch: {key}')
        print('Exact witness recomputed and matched.')
    else:
        OUT.write_text(json.dumps(result, indent=2)+'\n')
    print('Sturm root count:', result['sturm_root_count'])
    print('Cusp:', result['approximate_cusp'])
    print('Quartic cusp multiplicity: exactly 3; all exact checks passed.')
    print('Library FK cross-check:', result['library_FK_max_error'])


if __name__ == '__main__':
    main()
