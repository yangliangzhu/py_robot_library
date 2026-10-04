#!/usr/bin/env python3
"""exp54: where is the 3R's cusp?  -- the position IK's discriminant, solved and checked.

The 3R of :mod:`study.cuspidal3r` is cuspidal (exp48: an aspect holds two solutions of one pose) but
its cusp was never located.  A census-based triple-root statistic was noise-dominated, and the
structural criterion ("the image velocity along the critical curve is minimised") produced a clean
candidate at ``(rho, z) = (3.4601, +-1.4364)`` that the fibre test *refuted*.  This module does the
algebra instead of searching.

**The elimination.**  With ``q1 = 0`` (the arm's ``(rho, z)`` do not depend on it) and the
Weierstrass variables ``u = tan(q2/2)``, ``t = tan(q3/2)``, the two position equations become
polynomials of degree 4 and 2 in ``u``; their resultant in ``u`` factors as

    (t^2 + 1)^2 * (t^2 + 4)^2 * (4 t^2 + 1)^2 * Q(t),
    Q = A(rho, z) t^4 + B(rho, z) t^2 + C(rho, z),

so the only multiplicity-1 factor is a **quadratic in ``t^2``** (the three repeated factors are the
Weierstrass denominators' asymptotic branches).

**Status: NOT VALIDATED -- do not use these coefficients as if they were the arm's IK.**  Phase 1,
which is the check that matters, currently **fails**: the roots of the derived quadratic do not
reproduce the censused fibre's ``tan^2(q3/2)`` (worst discrepancies 4.8e-02, 1.4e+00, 1.0e+02,
5.2e+00 at five workspace points, with the fibre's size matching the root count), so either the
substitution, the factor selection or the correspondence between ``t`` and the fibre's ``q3`` is
wrong.  Phase 2's "singular points" of ``Delta`` sit at ``rho = 0`` -- the chart boundary of the
cylindrical reduction, where ``Delta`` is a critical point with a *non-degenerate* Hessian
(determinant -7.4e+11, not 0) -- so they are an artefact of the chart and not a cusp either.  Both
phases are kept because they are the instruments the next attempt needs; the honest summary is in
``NOTES`` 3.47.

**The cusp.**  The workspace boundary is the discriminant curve ``Delta = B^2 - 4 A C = 0`` (two
``q3`` solutions merge there), and a cusp is a *singular point* of that curve: ``Delta = 0`` together
with ``grad Delta = 0``.  Three equations in the two unknowns ``(rho, z)``, solved by Gauss-Newton
from a grid of seeds, then classified by the local form of ``Delta``: the Hessian must be degenerate
there and the leading non-vanishing term along its null direction must be cubic -- that is what a cusp
of a plane curve is.  The fibre census at and around each candidate is the independent check.

Run::

    python3 -m study.exp54_cusp_discriminant [--derive]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from study import cuspidal3r as c3

#: ``Q = A t^4 + B t^2 + C`` with ``t = tan(q3/2)``, from the symbolic elimination (see --derive).
CACHED = ("16*r2**2 + 32*r2*z2 - 360*r2 + 16*z2**2 - 296*z2 + 1769",
          "32*r2**2 + 64*r2*z2 - 528*r2 + 32*z2**2 - 400*z2 + 802",
          "16*r2**2 + 32*r2*z2 - 168*r2 + 16*z2**2 - 104*z2 + 185")


def coefficients(rho: float, z: float) -> tuple[float, float, float]:
    """``(A, B, C)`` of the quadratic in ``t^2`` at a workspace point."""
    r2, z2 = float(rho) ** 2, float(z) ** 2
    return tuple(float(eval(expression, {"r2": r2, "z2": z2, "__builtins__": {}}))
                 for expression in CACHED)  # noqa: S307 - the strings are module constants


def delta(rho: float, z: float) -> float:
    """The discriminant ``B^2 - 4AC`` of the quadratic in ``t^2``."""
    a, b, c = coefficients(rho, z)
    return b * b - 4.0 * a * c


def gradient(rho: float, z: float, *, step: float = 1e-6) -> np.ndarray:
    """First derivatives of :func:`delta` (central differences; it is a polynomial)."""
    return np.array([(delta(rho + step, z) - delta(rho - step, z)) / (2 * step),
                     (delta(rho, z + step) - delta(rho, z - step)) / (2 * step)])


def hessian(rho: float, z: float, *, step: float = 1e-4) -> np.ndarray:
    """Second derivatives of :func:`delta`."""
    fxx = (delta(rho + step, z) - 2 * delta(rho, z) + delta(rho - step, z)) / step ** 2
    fzz = (delta(rho, z + step) - 2 * delta(rho, z) + delta(rho, z - step)) / step ** 2
    fxz = (delta(rho + step, z + step) - delta(rho + step, z - step)
           - delta(rho - step, z + step) + delta(rho - step, z - step)) / (4 * step ** 2)
    return np.array([[fxx, fxz], [fxz, fzz]])


def singular_points(seeds, *, iterations=200, tolerance=1e-12):
    """Gauss-Newton on ``(Delta, dDelta/drho, dDelta/dz) = 0`` from each seed."""
    found: list[tuple[float, float, float]] = []
    for seed in seeds:
        point = np.asarray(seed, dtype=float)
        for _ in range(iterations):
            residual = np.array([delta(*point), *gradient(*point)])
            if float(np.linalg.norm(residual)) < tolerance:
                break
            jacobian = np.vstack([gradient(*point), hessian(*point)])
            point = point + np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        residual = float(np.linalg.norm(np.array([delta(*point), *gradient(*point)])))
        if residual < 1e-6 and point[0] > 0 and not any(
                abs(point[0] - other[0]) < 1e-4 and abs(point[1] - other[1]) < 1e-4
                for other in found):
            found.append((float(point[0]), float(point[1]), residual))
    return found


def cusp_form(point, *, step=1e-4):
    """``(Hessian determinant, cubic term along the null direction, null direction)``."""
    rho, z = float(point[0]), float(point[1])
    matrix = hessian(rho, z, step=step)
    values, vectors = np.linalg.eigh(matrix)
    null = vectors[:, 0]
    h = step
    u, v = float(null[0]), float(null[1])
    third = (delta(rho + h * u, z + h * v) - 3 * delta(rho, z)
             + 3 * delta(rho - h * u, z - h * v)
             - delta(rho - 2 * h * u, z - 2 * h * v)) / h ** 3
    return float(np.linalg.det(matrix)), float(-third / 6.0), null


def fibre_at(model, rho, z, rng, *, seeds=200, azimuths=(0.0, 1.1, 2.7)):
    """Census the fibre at a workspace point over several azimuths; returns counts and clearances."""
    counts, clearances = [], []
    for azimuth in azimuths:
        point = np.array([rho * np.cos(azimuth), rho * np.sin(azimuth), z])
        fiber = c3.census_position(model, point, seeds=seeds, rng=rng)
        counts.append(len(fiber.solutions))
        clearances.extend(c3.sigma_min_position(model, q) for q in fiber.solutions)
    return counts, clearances


def verify_coefficients(model, rng, *, points, seeds=300):
    """Check the derived quadratic against the model: its roots must be the fibre's ``tan^2(q3/2)``.

    Args:
        model: The 3R model.
        rng: Generator.
        points: ``(rho, z)`` values to check.
        seeds: Census seeds per point.

    Returns:
        A list of ``(rho, z, worst discrepancy, fibre size, root count)``.
    """
    out = []
    for rho, z in points:
        fiber = c3.census_position(model, np.array([rho, 0.0, z]), seeds=seeds, rng=rng)
        a, b, c = coefficients(rho, z)
        roots = np.roots([a, b, c])
        real = np.sort(np.abs(roots[np.abs(roots.imag) < 1e-9].real))
        measured = np.sort(np.array([np.tan(c3.canonical(q)[2] / 2.0) ** 2
                                     for q in fiber.solutions]))
        worst = float("nan")
        if len(real) == len(measured):
            worst = float(np.max(np.abs(real - measured))) if len(real) else 0.0
        out.append((rho, z, worst, len(fiber.solutions), len(real)))
    return out


def derive_symbolically() -> tuple[str, str, str]:
    """Reproduce the elimination with sympy (slow: minutes) and return ``A``, ``B``, ``C``.

    This is the derivation the module's ``CACHED`` strings came from; it is kept so that the
    coefficients are reproducible rather than transcribed on trust.
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
        numerator = sp.factor(sp.together(sp.expand(equation.subs(substitutions))).as_numer_denom()[0])
        polys.append(sp.Poly(sp.expand(numerator), u))
    resultant = sp.factor(sp.resultant(polys[0].as_expr(), polys[1].as_expr(), u))
    quartic = [factor for factor, multiplicity in sp.factor_list(resultant)[1]
               if multiplicity == 1 and sp.degree(factor, t) == 4
               and sp.expand(factor - factor.subs(t, -t)) == 0]
    a4, a2, a0 = sp.Poly(quartic[0], t).all_coeffs()
    return (sp.sstr(sp.expand(a4)), sp.sstr(sp.expand(a2)), sp.sstr(sp.expand(a0)))


def main() -> int:
    """Run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, default=20, help="seed grid per axis")
    parser.add_argument("--rho-max", type=float, default=5.5)
    parser.add_argument("--z-min", type=float, default=-3.0)
    parser.add_argument("--z-max", type=float, default=4.5)
    parser.add_argument("--seeds", type=int, default=200, help="census seeds per azimuth")
    parser.add_argument("--derive", action="store_true",
                        help="re-derive A, B, C with sympy (minutes) instead of using CACHED")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    model = c3.build()
    print("exp54: the 3R's position IK is a quadratic in tan^2(q3/2); where is its cusp?\n")
    if args.derive:
        print("phase 0: symbolic re-derivation (this takes minutes)")
        print("  A, B, C =", derive_symbolically())
    print("phase 1: the derived quadratic against the model's own fibre")
    checks = verify_coefficients(model, rng, points=[(1.2, 0.4), (2.0, -0.3), (2.6, 0.9),
                                                     (3.2, 0.2), (4.0, -0.6)])
    for rho, z, worst, size, roots in checks:
        print(f"    (rho, z) = ({rho:.2f}, {z:+.2f}): fibre {size} solution(s), quadratic has "
              f"{roots} real root(s) in t^2, worst |t^2| discrepancy {worst:.2e}")

    print(f"\nphase 2: singular points of Delta = B^2 - 4AC (three equations, two unknowns), "
          f"{args.grid}x{args.grid} seeds")
    seeds = list(itertools.product(np.linspace(0.05, args.rho_max, args.grid),
                                   np.linspace(args.z_min, args.z_max, args.grid)))
    found = singular_points(seeds)
    print(f"  {len(found)} singular point(s)")
    for rho, z, residual in found:
        determinant, cubic, null = cusp_form((rho, z))
        print(f"    (rho, z) = ({rho:.6f}, {z:.6f}): residual {residual:.1e}, Hessian determinant "
              f"{determinant:.2e}, cubic term {cubic:.3e} along {np.round(null, 4)}")
        counts, clearances = fibre_at(model, rho, z, rng, seeds=args.seeds)
        print(f"      fibre at the point: solutions per azimuth {counts}, smallest clearance "
              f"{min(clearances, default=float('nan')):.2e}")
        for rho_near, z_near in ((rho * 1.01, z), (rho * 0.99, z), (rho, z + 0.05)):
            near, near_clearances = fibre_at(model, rho_near, z_near, rng, seeds=args.seeds // 4)
            print(f"      nearby ({rho_near:.4f}, {z_near:+.4f}): {near}, smallest clearance "
                  f"{min(near_clearances, default=float('nan')):.2e}")
    print("\n  reading: a cusp needs Delta = 0, grad Delta = 0, a degenerate Hessian *and* a "
          "non-vanishing cubic term along its null direction; the fibre counts nearby are the "
          "independent check that the boundary's two branches meet there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
