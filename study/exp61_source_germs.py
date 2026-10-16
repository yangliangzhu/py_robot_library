"""Derive the 3R candidate source germs directly from exact MDH matrices.

Run ``python3 -m study.exp61_source_germs``. SymPy is an optional research
dependency. Exact assertions concern the ideal right-angle MDH model;
the final numerical comparison checks agreement with the library backend.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from study import cuspidal3r


def link(a: sp.Expr, d: sp.Expr) -> sp.Matrix:
    """Construct an exact MDH link with alpha equal to pi/2."""
    return sp.Matrix([[1, 0, 0, a], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]])


def rotation(q: sp.Symbol) -> sp.Matrix:
    """Construct a homogeneous rotation about the Z axis."""
    c, s = sp.cos(q), sp.sin(q)
    return sp.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def main() -> int:
    """Verify exact source formulas and print the four candidate germs."""
    u, v, z, y, w = sp.symbols("u v z y w", real=True)
    half = sp.Rational(1, 2)
    p = (link(1, 1) * rotation(u) * link(0, 2) * rotation(v) * link(0, 3 * half))[
        :3, 3
    ]
    a = 2 * sp.sin(u) + 3 * half * sp.sin(v) * sp.cos(u)
    expected = sp.Matrix([1 + a, 3 * half * sp.cos(v) - 1,
                          3 * half * sp.sin(u) * sp.sin(v) - 2 * sp.cos(u)])
    assert all(sp.trigsimp(entry) == 0 for entry in p - expected)
    radius_squared = sp.trigsimp(p[0] ** 2 + p[1] ** 2)
    jacobian = sp.Matrix([radius_squared, p[2]]).jacobian([u, v])
    determinant = -3 * (a + 3 * half * sp.cos(v)) * sp.sin(v)
    assert sp.trigsimp(sp.expand_trig(jacobian.det() - determinant)) == 0
    assert sp.trigsimp(a**2 + p[2] ** 2 - 4 - 9 * half**2 * sp.sin(v)**2) == 0
    print("Exact FK:", expected.T)
    print("det D(R,z):", determinant)

    for epsilon, c, r0 in [(-1, 1, half), (1, -1, 25 * half)]:
        total = r0 + sp.Rational(7, 4)
        polynomial = 18*w**2 + 6*(total-sp.Rational(33, 4))*w + (
            total-sp.Rational(33, 4))**2 + 7 - 25
        assert sp.expand(polynomial - 18 * (w - c)**2) == 0
        # y=v near v=0; y=v-pi near v=pi. The sine sign disappears after
        # replacing u with the locally regular output coordinate z.
        local_r = sp.Rational(33, 4) - z**2 + 2*epsilon*sp.sqrt(
            4 + sp.Rational(9, 4)*sp.sin(y)**2 - z**2) - 3*c*sp.cos(y)
        b = sp.simplify(sp.diff(local_r, y, 2).subs(y, 0) / 2)
        for sign in (1, -1):
            z0 = sign * sp.sqrt(7) / 2
            assert sp.simplify(b.subs(z, z0)) == 0
            mixed = sp.simplify(sp.diff(b, z).subs(z, z0))
            fourth = sp.simplify(sp.diff(local_r, y, 4).subs({z: z0, y: 0}) / 24)
            assert mixed == epsilon * sign * sp.sqrt(7) / 3
            assert fourth == -epsilon * sp.Rational(3, 4)
            assert sp.simplify(local_r.subs({z: z0, y: 0}) - r0) == 0
            print(f"R0={r0}, z0={z0}, cos(v)={c}, sin(u)={epsilon*sp.Rational(3,4)}, "
                  f"cos(u)={-z0/2}: mixed={mixed}, quartic={fourth}")

    numeric = sp.lambdify((u, v), expected, "numpy")
    model = cuspidal3r.build()
    rng = np.random.default_rng(61)
    worst = 0.0
    for q in rng.uniform(-np.pi, np.pi, (20, 3)):
        local = np.asarray(numeric(q[1], q[2]), dtype=float).reshape(3)
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        turn = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        worst = max(worst, float(np.linalg.norm(turn @ local - cuspidal3r.position(model, q))))
    assert worst < 1e-12
    print(f"All exact checks passed; numerical FK agreement over 20 samples: {worst:.3e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
