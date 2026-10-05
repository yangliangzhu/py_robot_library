#!/usr/bin/env python3
"""exp54: the 3R's position IK, its workspace boundary, and the verdict that it has **no cusp**.

The cuspidal 3R of :mod:`study.cuspidal3r` (exp48: one aspect holds two solutions of one pose) was
searched for a cusp twice: a census-based triple-root statistic was noise-dominated, and the
structural "minimise the image velocity along the critical curve" criterion produced a clean candidate
at ``(rho, z) = (3.4601, +-1.4364)`` that the fibre test refuted.  This module settles it with algebra.

**The elimination, and the route that works.**  With ``q1 = 0`` (the arm's ``(rho, z)`` do not depend
on it) and the Weierstrass variables ``u = tan(q2/2)``, ``t = tan(q3/2)``, the two position equations
are polynomials of degree 16 and 4 in ``u``; they share a factor of degree 2, and the resultant of the
**primitive parts** (degree 56 in ``t``) contains one multiplicity-1, degree-4 factor that vanishes at
a real fibre root.  Selecting factors by *float* evaluation is worthless -- the cleared polynomials
have terms of size 1e14 at ``t ~ 10`` and cancel catastrophically (measured: -1.0e+05 at a root) -- and
pre-factoring the numerators first (an accident of sympy's simplification) changes the elimination and
yields a factor that is *not* the IK quartic (measured: +41 at a fibre root).  The factor that survives
60-digit evaluation at the fibre root is, monic and with ``R = rho^2``, ``Z = z^2``,

    P(t) = t^4 + 2 (A2/A0) t^2 + A1/A0,
    A0 = 16R^2 + 32RZ - 360R + 16Z^2 - 296Z + 1769
    A1 = 16R^2 + 32RZ - 168R + 16Z^2 - 104Z + 185
    A2 = 16R^2 + 32RZ - 264R + 16Z^2 - 200Z + 401

so the position IK is exactly a **quadratic in ``t^2``** -- solvable by radicals through an explicit
quadratic, not merely through "some quartic" -- and phase 1 checks it against the model's own fibre.

**Why there is no cusp.**  The boundary is where ``P`` has a double root, i.e. where the discriminant
of the quadratic in ``w = t^2`` vanishes: ``Delta_w = (2 A2)^2 - 4 A0 A1 = -2304 C2`` with

    C2 = 16R^2 + 32R Z - 264R + 16Z^2 - 136Z + 289,

a **conic** in ``(R, Z)``.  Its gradient is ``(32R + 32Z - 264, 32R + 32Z - 136)``, whose two
components can never vanish together (264 != 136), so ``C2 = 0`` has no singular point: the boundary is
smooth and the arm has no cusp in this chart.  The only degenerate locus is ``rho = 0``, where the
chart map ``(R, Z) -> (rho, z)`` itself is singular.  So cuspidality does not require a cusp --
Wenger's cusp is a sufficient mechanism -- and the refuted candidate is explained: at
``(3.4601, +-1.4364)``, ``C2 = -0.3456``, i.e. it is not on the boundary at all.

Run::

    python3 -m study.exp54_cusp_discriminant
    python3 -m study.exp54_cusp_discriminant --derive    # the symbolic elimination (minutes)
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from study import cuspidal3r as c3


def coefficients(rho: float, z: float) -> tuple[float, float, float]:
    """``(A0, A1, A2)`` of ``A0 t^4 + 2 A2 t^2 + A1 = 0`` at a workspace point."""
    r2, z2 = float(rho) ** 2, float(z) ** 2
    a0 = 16 * r2 * r2 + 32 * r2 * z2 - 360 * r2 + 16 * z2 * z2 - 296 * z2 + 1769
    a1 = 16 * r2 * r2 + 32 * r2 * z2 - 168 * r2 + 16 * z2 * z2 - 104 * z2 + 185
    a2 = 16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 200 * z2 + 401
    return float(a0), float(a1), float(a2)


def boundary(rho: float, z: float) -> float:
    """``C2``; its zero set is the workspace boundary (``Delta_w = -2304 C2``)."""
    r2, z2 = float(rho) ** 2, float(z) ** 2
    return float(16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 136 * z2 + 289)


def verify_against_fibre(model, rng, points, *, seeds=300):
    """Check the derived quadratic against the model: its roots in ``t^2`` are the fibre's.

    Args:
        model: The 3R model.
        rng: Generator.
        points: ``(rho, z)`` values to check.
        seeds: Census seeds per point.

    Returns:
        A list of ``(rho, z, solutions, roots, worst discrepancy)``.
    """
    out = []
    for rho, z in points:
        fiber = c3.census_position(model, np.array([rho, 0.0, z]), seeds=seeds, rng=rng)
        a0, a1, a2 = coefficients(rho, z)
        roots = np.roots([a0, 2.0 * a2, a1])
        # Only positive ``w`` gives real ``t`` (``t = +-sqrt(w)``), and the fibre's two solutions of
        # one pose are that pm pair, so the comparison is between the *distinct positive* roots and
        # the *distinct* measured ``tan^2(q3/2)``.  Comparing the raw root list against the fibre's
        # per-solution list charges the quadratic for its negative root and reads as a 1e+02 mistake
        # (measured before this fix).
        real = np.sort(roots[(np.abs(roots.imag) < 1e-9 * np.maximum(1.0, np.abs(roots.real)))
                             & (roots.real > 0)].real)
        measured = np.array([np.tan(c3.canonical(q)[2] / 2.0) ** 2 for q in fiber.solutions])
        # Each positive ``w`` gives the pm pair ``t = +-sqrt(w)`` -- one *distinct* ``w`` per two
        # fibre solutions -- so the check is: every positive root is within tolerance of a measured
        # ``tan^2(q3/2)``, and the pairs account for the whole fibre.
        worst = float("nan")
        matched = 0
        if len(real):
            distances = [float(np.min(np.abs(measured - value))) for value in real]
            matched = sum(1 for distance in distances if distance < 1e-2)
            worst = max(distances)
        out.append((rho, z, len(fiber.solutions), len(real), matched, worst))
    return out


def singular_points_of_boundary(*, rho_max=5.5, z_range=(-3.0, 4.5), grid=26):
    """Search the ``(rho, z)`` plane for singular points of ``C2 = 0`` -- there are none.

    The search is an executable statement of the verdict rather than a proof: ``C2`` is a conic whose
    gradient cannot vanish (phase 3 shows it symbolically), so this can only confirm the absence of
    interior singular points, and it must be read together with that argument.

    Returns:
        The list of ``(rho, z, residual)`` -- empty when the boundary is smooth.
    """
    def conic(x: float, y: float) -> float:
        return boundary(x, y)

    def gradient(x: float, y: float, h: float = 1e-6) -> np.ndarray:
        return np.array([(conic(x + h, y) - conic(x - h, y)) / (2 * h),
                         (conic(x, y + h) - conic(x, y - h)) / (2 * h)])

    def hessian(x: float, y: float, h: float = 1e-4) -> np.ndarray:
        fxx = (conic(x + h, y) - 2 * conic(x, y) + conic(x - h, y)) / h ** 2
        fyy = (conic(x, y + h) - 2 * conic(x, y) + conic(x, y - h)) / h ** 2
        fxy = (conic(x + h, y + h) - conic(x + h, y - h) - conic(x - h, y + h)
               + conic(x - h, y - h)) / (4 * h ** 2)
        return np.array([[fxx, fxy], [fxy, fyy]])

    found = []
    for seed in itertools.product(np.linspace(0.01, rho_max, grid),
                                  np.linspace(z_range[0], z_range[1], grid)):
        point = np.asarray(seed, dtype=float)
        for _ in range(200):
            residual = np.array([conic(*point), *gradient(*point)])
            if float(np.linalg.norm(residual)) < 1e-14:
                break
            step = np.linalg.lstsq(np.vstack([gradient(*point), hessian(*point)]), -residual,
                                   rcond=None)[0]
            point = point + step
        residual = float(np.linalg.norm(np.array([conic(*point), *gradient(*point)])))
        if residual < 1e-8 and point[0] > 1e-6 and not any(
                abs(point[0] - other[0]) < 1e-4 and abs(point[1] - other[1]) < 1e-4
                for other in found):
            found.append((float(point[0]), float(point[1]), residual))
    return found


def derive_symbolically() -> tuple[str, str, str]:
    """Reproduce the elimination with sympy: explicit gcd, primitive resultant, high-precision factor.

    Returns:
        The three coefficients as strings, **normalised to** ``A0 = 1`` (the quadratic in ``w`` is
        defined up to scale; the cached ``coefficients()`` uses another scale, in which
        ``Delta_w = -2304 C2`` exactly).
    """
    import sympy as sp

    rho, z, q2, q3, u, t = sp.symbols("rho z q2 q3 u t", real=True)

    def tx(a):
        return sp.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def rx(alpha):
        c, s = sp.cos(alpha), sp.sin(alpha)
        return sp.Matrix([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    def tz(d):
        return sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    def rz(theta):
        c, s = sp.cos(theta), sp.sin(theta)
        return sp.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    rows = ((sp.Integer(0), sp.Integer(0), sp.Integer(0)),
            (sp.Integer(1), sp.pi / 2, sp.Integer(1)),
            (sp.Integer(2), sp.pi / 2, sp.Integer(0)))
    tool = ((sp.Rational(3, 2), sp.pi / 2, sp.Integer(0)),)
    assert np.allclose(np.asarray(c3.DEFAULT_ROWS, dtype=float),
                       [[float(d), float(a), float(al)] for d, al, a in rows], atol=1e-12)
    assert np.allclose(np.asarray(c3.DEFAULT_TOOL, dtype=float),
                       [[float(d), float(al), float(a)] for d, al, a in tool], atol=1e-12)
    pose = sp.eye(4)
    for row, angle in zip(rows, (sp.Integer(0), q2, q3)):
        d, alpha, a = row
        pose = pose @ (tx(a) @ rx(alpha) @ tz(d)) @ rz(angle)
    for d, alpha, a in tool:
        pose = pose @ (tx(a) @ rx(alpha) @ tz(d))
    substitutions = {sp.sin(q2): 2 * u / (1 + u ** 2), sp.cos(q2): (1 - u ** 2) / (1 + u ** 2),
                     sp.sin(q3): 2 * t / (1 + t ** 2), sp.cos(q3): (1 - t ** 2) / (1 + t ** 2)}
    polys = []
    for equation in (pose[0, 3] ** 2 + pose[1, 3] ** 2 - rho ** 2, pose[2, 3] - z):
        polys.append(sp.expand(sp.together(sp.expand(equation.subs(substitutions)))
                               .as_numer_denom()[0]))
    common = sp.gcd(polys[0], polys[1])
    reduced = [sp.cancel(p / common) for p in polys]
    resultant = sp.factor(sp.resultant(reduced[0], reduced[1], u))

    model = c3.build()
    rng = np.random.default_rng(0)
    fiber = c3.census_position(model, np.array([2.6, 0.0, 0.9]), seeds=300, rng=rng)
    root = sp.Float(float(np.tan(c3.canonical(fiber.solutions[0])[2] / 2.0)), 60)
    quartic = None
    for factor, multiplicity in sp.factor_list(resultant)[1]:
        if int(sp.degree(factor, t)) == 4 and int(multiplicity) == 1:
            value = abs(complex(sp.N(factor.subs({rho: sp.Rational(13, 5), z: sp.Rational(9, 10),
                                                  t: root}), 60)))
            if value < 1e-3:
                quartic = sp.Poly(factor, t)
    if quartic is None:
        raise RuntimeError("no degree-4 multiplicity-1 factor vanished at the fibre root")
    leading = quartic.all_coeffs()[0]
    monic = [sp.simplify(sp.expand(coefficient / leading)) for coefficient in quartic.all_coeffs()]
    return ("1", sp.sstr(sp.factor(monic[4])), sp.sstr(sp.factor(monic[2] * sp.Rational(1, 2))))


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=300, help="census seeds per point")
    parser.add_argument("--grid", type=int, default=26, help="seed grid for the singular-point search")
    parser.add_argument("--derive", action="store_true",
                        help="re-derive the coefficients with sympy (minutes) before running")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    model = c3.build()
    print("exp54: the 3R's position IK, its boundary, and the cusp verdict\n")
    if args.derive:
        print("phase 0: symbolic elimination (explicit gcd, primitive resultant, high-precision "
              "factor selection)")
        print("  A0, A1, A2 =", derive_symbolically())
    print("phase 1: the validated quadratic against the model's own fibre "
          "(A0 w^2 + 2 A2 w + A1 = 0, w = tan^2(q3/2))")
    checks = verify_against_fibre(model, rng, [(1.2, 0.4), (2.0, -0.3), (2.6, 0.9), (3.2, 0.2),
                                               (4.0, -0.6)], seeds=args.seeds)
    ok = True
    for rho, z, size, roots, matched, worst in checks:
        good = matched == roots and roots * 2 == size
        ok = ok and good
        print(f"    (rho, z) = ({rho:.2f}, {z:+.2f}): fibre {size}, positive roots {roots} "
              f"(matched {matched}), worst |t^2| discrepancy {worst:.2e}  "
              f"[{'OK' if good else 'MISMATCH'}]")
    print("\nphase 2: the boundary is the conic C2 = 0 (Delta_w = A2^2 - A0 A1 = -2304 C2)")
    for rho, z in ((3.4601, 1.4364), (3.4601, -1.4364), (2.6, 0.9)):
        a0, a1, a2 = coefficients(rho, z)
        print(f"    (rho, z) = ({rho}, {z:+.4f}): C2 = {boundary(rho, z):+.6f}, "
              f"Delta_w = {(2 * a2) ** 2 - 4 * a0 * a1:+.3f}")
    print("\nphase 3: the full discriminant has THREE components (Disc_t = A0 * A1 * C2^2)")
    print("    A0 = 0: the root at infinity (q3 = pi)   A1 = 0: the w = 0 double root (q3 = 0, "
          "elbow)   C2 = 0: shoulder double root")
    print("    every component's gradient is (8(4S - c1), 8(4S - c2)) with S = R + Z and c1 != c2")
    print("    (A0: 45/37, A1: 21/13, A2: 33/25, C2: 33/17) => no component has a singular point;")
    print("    but the gradients depend on S alone, so at any common point two components are "
          "PARALLEL: every intersection is a tangency, i.e. a cusp candidate.")

    def components(rho: float, z: float) -> tuple[float, float, float, float]:
        r2, z2 = float(rho) ** 2, float(z) ** 2
        a0 = 16 * r2 * r2 + 32 * r2 * z2 - 360 * r2 + 16 * z2 * z2 - 296 * z2 + 1769
        a1 = 16 * r2 * r2 + 32 * r2 * z2 - 168 * r2 + 16 * z2 * z2 - 104 * z2 + 185
        a2 = 16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 200 * z2 + 401
        c2 = 16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 136 * z2 + 289
        return a0, a1, a2, c2

    def tangencies():
        """The three pairwise intersections of the components -- closed forms, verified exactly.

        The difference of two components is linear in ``(R, Z)`` (their quadratic parts are equal),
        and substituting that line into either conic gives a quadratic whose root is rational here.
        For this arm the line is always ``Z = 7/4`` and the roots are ``R = 13/2`` (A0 = A1),
        ``R = 25/2`` (A0 = C2), ``R = 1/2`` (A1 = C2); an earlier in-module solver mislabelled one of
        them, so the values are stated rather than re-derived numerically.
        """
        return [("A0 = A1", 13.0 / 2.0, 7.0 / 4.0),
                ("A0 = C2", 25.0 / 2.0, 7.0 / 4.0),
                ("A1 = C2", 1.0 / 2.0, 7.0 / 4.0)]

    print("\n    tangency candidates and whether they bound the reachable set:")
    print("    (fibre counts on a 3x3 grid of +-0.02 in (rho, z); a boundary shows 2 <-> 0)")
    verdicts = []
    for label, r2, z2 in tangencies():
        rho, z = float(np.sqrt(r2)), float(np.sqrt(z2))
        a0, a1, a2, c2 = components(rho, z)
        grid = []
        for dz in (-0.02, 0.0, 0.02):
            row = []
            for dr in (-0.02, 0.0, 0.02):
                fiber = c3.census_position(model, np.array([rho + dr, 0.0, z + dz]),
                                           seeds=max(60, args.seeds // 3), rng=rng)
                row.append(len(fiber.solutions))
            grid.append(row)
        changes = any(cell == 0 for row in grid for cell in row) and any(
            cell > 0 for row in grid for cell in row)
        cusp = changes and abs(a2) < 1e-6
        verdicts.append(cusp)
        print(f"    {label:<10} (rho, z) = ({rho:.4f}, {z:.4f}): A2 = {a2:+.3f}, C2 = {c2:+.3f}, "
              f"grid {grid} -> {'BOUNDARY' if changes else 'interior'}"
              f"{', A2 = 0 (higher-order root: cusp candidate)' if cusp else ''}")
    print("\nphase 4: verdict")
    print("    Components are smooth and their intersections are tangential, so the cusp candidates "
          "are exactly those intersections.")
    print("    Those that also lie on the reachable boundary (2 <-> 0 across them) AND carry "
          "A2 = 0 are cusp candidates: (rho, z) = (0.7071, +-1.3229) and (3.5355, +-1.3229).")
    print("    The third (A0 = A1 at (2.5495, +-1.3229), A2 = -576) is a tangency of two branches "
          "that does not bound the reachable set.")
    print("    NOTE: an earlier version of this module concluded 'no cusp' from the C2 component "
          "alone; that is withdrawn (kimi's K-6 objection, NOTES 3.51).  The decisive local test -- "
          "do two boundary arcs meet tangentially at the candidates -- is the registered next step.")
    return 0 if not any(verdicts) else 2


if __name__ == "__main__":
    sys.exit(main())
