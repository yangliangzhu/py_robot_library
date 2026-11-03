"""Derive SR5's degree-16 elimination and certify an arithmetic S16 specialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

OUT = Path('study/exp76_sr5_arithmetic_certificate.json')



def rabin_irreducible(poly: sp.Poly, prime: int) -> bool:
    """Check finite-field irreducibility using Frobenius powers and gcds."""
    x = sp.Poly(poly.gen, poly.gen, modulus=prime)
    def mod_power(base, exponent):
        result = sp.Poly(1, poly.gen, modulus=prime)
        while exponent:
            if exponent & 1:
                result = result*base % poly
            base = base*base % poly
            exponent //= 2
        return result
    degree = poly.degree()
    if mod_power(x, prime**degree) != x % poly:
        return False
    return all(sp.gcd(poly, mod_power(x, prime**(degree//q))-x).degree() == 0
               for q in sp.factorint(degree))


def generic_elimination() -> tuple:
    """Eliminate a wrist displacement from exact planar equations."""
    k, r, z, nr, nz, j, c0 = sp.symbols('k r Z n_r n_z J C_0')
    b = sp.Rational(1, 10)-2*k
    a, b2 = -sp.Rational(4, 5)*r-b*z, -sp.Rational(4, 5)*z+b*r
    c = r*r+z*z+k*k-k/10
    d, e, f = -k*nz, k*nr, j-r*nr
    determinant = a*e-b2*d
    sin_numerator, cos_numerator = b2*f-e*c, d*c-a*f
    norm = sp.expand(sin_numerator**2+cos_numerator**2-determinant**2)
    reduced = sp.Poly(sp.rem(norm, k*k-(r*r+c0), k), k)
    f0, f1 = reduced.nth(0), reduced.nth(1)
    assert sp.Poly(f0, r, nr).total_degree() <= 4
    assert sp.Poly(f1, r, nr).total_degree() <= 3
    assert sp.Poly(f0*f0-(r*r+c0)*f1*f1, r, nr).total_degree() <= 8
    return (k, r, z, nr, nz, j, c0), f0, f1, determinant, sin_numerator, cos_numerator


class Residue:
    """Compute in an exact degree-16 polynomial quotient over Q or a prime field."""
    modulus = None
    prime = 41
    def __init__(self, value):
        self.poly = value % self.modulus if isinstance(value, sp.Poly) else sp.Poly(value, self.modulus.gen, **({'modulus': self.prime} if self.prime else {'domain': sp.QQ})) % self.modulus
    def __add__(self, other):
        other = other if isinstance(other, Residue) else Residue(other)
        return Residue(self.poly+other.poly)
    __radd__ = __add__
    def __neg__(self):
        return Residue(-self.poly)
    def __sub__(self, other):
        return self+-to_residue(other)
    def __rsub__(self, other):
        return to_residue(other)+-self
    def __mul__(self, other):
        return Residue(self.poly*to_residue(other).poly)
    __rmul__ = __mul__
    def __truediv__(self, other):
        return self*Residue(sp.invert(to_residue(other).poly, self.modulus))
    def __eq__(self, other):
        return self.poly == to_residue(other).poly


def to_residue(value):
    if isinstance(value, Residue):
        return value
    if isinstance(value, (sp.Rational, sp.Integer)):
        return Residue(int(sp.numer(value))*pow(int(sp.denom(value)), -1, Residue.prime)) if Residue.prime else Residue(value)
    return Residue(value)


def full_modular_reconstruction(polynomial, t, u0, u1, target_r, target_p, prime=41):
    """Check all sixteen algebraic roots at once against the original full FK."""
    import yaml
    Residue.prime = prime
    domain = {'modulus': prime} if prime else {'domain': sp.QQ}
    Residue.modulus = sp.Poly(polynomial, t, **domain).monic()
    tt = Residue(sp.Poly(t, t, **domain))
    one = to_residue(1)
    def ev(expr):
        num, den = sp.fraction(sp.cancel(expr))
        def horner(poly):
            result = to_residue(0)
            for coefficient in sp.Poly(poly, t, domain=sp.QQ).all_coeffs():
                result = result*tt+to_residue(coefficient)
            return result
        return horner(num)/horner(den)
    def mat(entries):
        return [[to_residue(x) for x in row] for row in entries]
    def mm(a, b):
        return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    def transpose(a):
        return list(map(list, zip(*a, strict=True)))
    def rz(c, s):
        return [[c,-s,to_residue(0)], [s,c,to_residue(0)], [to_residue(0),to_residue(0),one]]
    def homogeneous(rotation, position):
        return [rotation[i]+[position[i][0]] for i in range(3)]+[mat([[0,0,0,1]])[0]]
    cf, sf = (1-tt*tt)/(1+tt*tt), 2*tt/(1+tt*tt)
    rr, pp = to_residue(sp.Rational(3,10))*cf+to_residue(sp.Rational(1,5))*sf, -to_residue(sp.Rational(3,10))*sf+to_residue(sp.Rational(1,5))*cf
    nr = to_residue(sp.Rational(2,3))*cf+to_residue(sp.Rational(1,3))*sf
    nf = -to_residue(sp.Rational(2,3))*sf+to_residue(sp.Rational(1,3))*cf
    z, nz = to_residue(sp.Rational(9,125)), to_residue(sp.Rational(2,3))
    kval = -ev(u0)/ev(u1)
    b = to_residue(sp.Rational(1,10))-2*kval
    a, bb = -to_residue(sp.Rational(4,5))*rr-b*z, -to_residue(sp.Rational(4,5))*z+b*rr
    cc, d, e, f = rr*rr+z*z+kval*kval-kval/10, -kval*nz, kval*nr, pp*nf
    det = a*e-bb*d
    st, ct = (bb*f-e*cc)/det, (d*cc-a*f)/det
    ru = rr-kval*ct-(to_residue(sp.Rational(2,5))*st-to_residue(sp.Rational(1,20))*ct)
    zu = z+kval*st-(to_residue(sp.Rational(1,20))*st+to_residue(sp.Rational(2,5))*ct)
    cu = (to_residue(sp.Rational(1,20))*ru+to_residue(sp.Rational(2,5))*zu)/to_residue(sp.Rational(13,80))
    su = (to_residue(sp.Rational(2,5))*ru-to_residue(sp.Rational(1,20))*zu)/to_residue(sp.Rational(13,80))
    assert ct*ct+st*st == cu*cu+su*su == one
    c3, s3 = cu*ct+su*st, su*ct-cu*st
    rxm, rxp, rxpi = mat([[1,0,0],[0,0,1],[0,-1,0]]), mat([[1,0,0],[0,0,-1],[0,1,0]]), mat([[1,0,0],[0,-1,0],[0,0,-1]])
    q4frame = mm(mm(mm(mm(rz(cf,sf),rxm),rz(ct,st)),rxpi),rxm)
    w = [[kval*ct*cf-pp*sf], [kval*ct*sf+pp*cf], [-kval*st]]
    local = mm(transpose(q4frame), w)
    delta = to_residue(sp.Rational(17,125))
    s4, c4 = -local[0][0]/delta, local[1][0]/delta
    q5frame = mm(mm(q4frame,rz(c4,s4)),rxp)
    localn = mm(transpose(q5frame), mat([[sp.Rational(2,3)],[sp.Rational(1,3)],[sp.Rational(2,3)]]))
    s5, c5 = -localn[0][0], localn[1][0]
    q6frame = mm(mm(q5frame,rz(c5,s5)),rxm)
    residual = mm(transpose(q6frame),mat(target_r.tolist()))
    c6, s6 = residual[0][0], residual[1][0]
    for c, s in [(cf,sf),(cu,su),(c3,s3),(c4,s4),(c5,s5),(c6,s6)]:
        assert c*c+s*s == one
    config = yaml.safe_load(Path('model/configs/rokae_sr5.yaml').read_text())
    fk = mat(sp.eye(4).tolist())
    for link, (c,s) in zip(config['mat'], [(cf,sf),(cu,su),(c3,s3),(c4,s4),(c5,s5),(c6,s6)], strict=True):
        exact_link = [[sp.Rational(str(x)) for x in row] for row in link]
        fk = mm(mm(fk,mat(exact_link)),homogeneous(rz(c,s),mat([[0],[0],[0]])))
    target = homogeneous(mat(target_r.tolist()),mat(target_p.tolist()))
    assert fk == target
    return {'prime': prime, 'extension_degree': 16, 'all_joint_norms': True, 'full_FK_equals_target': True}



def check_prefix_geometry() -> bool:
    """Check the generic shoulder position and wrist axis from exact link matrices."""
    import yaml
    config = yaml.safe_load(Path('model/configs/rokae_sr5.yaml').read_text())
    links = [sp.Matrix([[sp.Rational(str(x)) for x in row] for row in matrix])
             for matrix in config['mat']]
    phi, u, theta = sp.symbols('phi u theta', real=True)
    def rz(angle):
        return sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0, 0],
                          [sp.sin(angle), sp.cos(angle), 0, 0],
                          [0,0,1,0], [0,0,0,1]])
    prefix = links[0]*rz(phi)*links[1]*rz(u)*links[2]*rz(u-theta)*links[3]
    radius = (8*sp.sin(theta)+8*sp.sin(u)-sp.cos(theta)+sp.cos(u))/20
    height = (sp.sin(theta)-sp.sin(u)+8*sp.cos(theta)+8*sp.cos(u))/20+sp.Rational(41,125)
    expected = sp.Matrix([radius*sp.cos(phi), radius*sp.sin(phi), height])
    axis = sp.Matrix([sp.sin(theta)*sp.cos(phi), sp.sin(theta)*sp.sin(phi), sp.cos(theta)])
    assert all(sp.trigsimp(sp.expand_trig(x)) == 0 for x in prefix[:3,3]-expected)
    assert all(sp.trigsimp(sp.expand_trig(x)) == 0 for x in prefix[:3,2]-axis)
    assert links[4][:3,3] == sp.Matrix([0,sp.Rational(17,125),0])
    assert links[5][:3,3] == sp.Rational(207,2000)*links[5][:3,2]
    return True


def compute() -> dict:
    """Build the exact specialized polynomial and its modular certificates."""
    t = sp.Symbol('t')
    symbols, f0, f1, determinant, ns, nc = generic_elimination()
    k, r, z, nr, nz, j, c0 = symbols
    den = 1+t*t
    px, py, zz = sp.Rational(3, 10), sp.Rational(1, 5), sp.Rational(9, 125)
    nx, ny, nnz = sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)
    rr = (px*(1-t*t)+2*py*t)/den
    nnr = (nx*(1-t*t)+2*ny*t)/den
    pp = (-2*px*t+py*(1-t*t))/den
    substitution = {r: rr, z: zz, nr: nnr, nz: nnz,
                    j: px*nx+py*ny, c0: sp.Rational(17, 125)**2-px*px-py*py}
    u0, u1 = [sp.cancel(expr.subs(substitution)) for expr in (f0, f1)]
    kval_squared = sp.cancel(sp.Rational(17, 125)**2-pp*pp)
    numerator = sp.fraction(sp.cancel(u0*u0-kval_squared*u1*u1))[0]
    polynomial = sp.Poly(numerator, t).clear_denoms()[1].primitive()[1]
    if polynomial.LC() < 0:
        polynomial = -polynomial
    assert polynomial.degree() == 16
    assert sp.gcd(polynomial, polynomial.diff()).degree() == 0
    recovery_k = sp.cancel(-u0/u1)
    assert sp.rem(sp.fraction(sp.cancel(recovery_k**2-kval_squared))[0], polynomial.as_expr(), t) == 0
    det_special = sp.cancel(determinant.subs(substitution).subs(k, recovery_k))
    # These are all variable-dependent divisors of the rational reconstruction.
    exceptions = {'half_angle_denominator': den, 'u1_numerator': sp.fraction(u1)[0],
                  'u1_denominator': sp.fraction(u1)[1],
                  'linear_system_determinant_numerator': sp.fraction(det_special)[0],
                  'linear_system_determinant_denominator': sp.fraction(det_special)[1]}
    gcd_degrees = {}
    for name, expr in exceptions.items():
        degree = sp.gcd(polynomial, sp.Poly(expr, t)).degree()
        assert degree == 0
        gcd_degrees[name] = degree

    # A legal rational full orientation with third column n.
    n = sp.Matrix([nx, ny, nnz])
    first = sp.Matrix([sp.Rational(2, 3), -sp.Rational(2, 3), -sp.Rational(1, 3)])
    rotation = sp.Matrix.hstack(first, n.cross(first), n)
    assert rotation.T*rotation == sp.eye(3) and rotation.det() == 1
    position = sp.Matrix([px, py, zz+sp.Rational(41, 125)])+sp.Rational(207, 2000)*n

    modular = {}
    for prime, wanted in [(41, [16]), (43, [3, 13])]:
        reduced = sp.Poly(polynomial.as_expr(), t, modulus=prime).monic()
        assert reduced.degree() == 16 and sp.gcd(reduced, reduced.diff()).degree() == 0
        factors = sp.factor_list(reduced)[1]
        assert sorted(factor.degree() for factor, exponent in factors for _ in range(exponent)) == wanted
        product = sp.Poly(1, t, modulus=prime)
        for factor, exponent in factors:
            assert factor.is_irreducible and rabin_irreducible(factor, prime)
            product *= factor**exponent
        assert product.monic() == reduced
        modular[str(prime)] = {'degrees': wanted,
                               'monic_polynomial': [int(c)%prime for c in reduced.all_coeffs()],
                               'factors': [{'coefficients': [int(c)%prime for c in factor.all_coeffs()],
                                            'exponent': int(exponent)} for factor, exponent in factors]}
    return {'generic_prefix_geometry_checked': check_prefix_geometry(),
            'polynomial_coefficients': list(map(str, polynomial.all_coeffs())),
            'degree': 16, 'generic_reduced_f0': str(f0), 'generic_reduced_f1': str(f1),
            'generic_degree_bounds_in_r_nr': [4, 3, 8],
            'target_rotation': [[str(x) for x in row] for row in rotation.tolist()],
            'target_position': [str(x) for x in position],
            'wrist_position': [str(px), str(py), str(zz+sp.Rational(41, 125))],
            'recovery_k_numerator': str(sp.fraction(recovery_k)[0]),
            'recovery_k_denominator': str(sp.fraction(recovery_k)[1]),
            'exceptional_gcd_degrees': gcd_degrees, 'modular': modular,
            'full_modular_reconstruction': full_modular_reconstruction(polynomial, t, u0, u1, rotation, position),
            'full_characteristic_zero_reconstruction': full_modular_reconstruction(polynomial, t, u0, u1, rotation, position, prime=0),
            'arithmetic_group': 'S16 (Dedekind factor types, primitivity, Jordan)',
            'geometric_claim': 'A16 <= G_geom <= S16; rational reconstruction plus the geometric-normal-subgroup proof in the report.'}


def main() -> None:
    """Generate or recompute the exact elimination certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    result = compute()
    if args.verify:
        if result != json.loads(args.verify.read_text()):
            raise ValueError('certificate mismatch')
        print('Exact arithmetic certificate recomputed.')
    else:
        OUT.write_text(json.dumps(result, indent=2)+'\n')
    print('Degree:', result['degree'], 'modular types:', {p: d['degrees'] for p, d in result['modular'].items()})
    print('Recovery exceptional gcd degrees:', result['exceptional_gcd_degrees'])


if __name__ == '__main__':
    main()
