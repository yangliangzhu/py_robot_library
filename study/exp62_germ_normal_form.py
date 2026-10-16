#!/usr/bin/env python3
"""exp62: independent verification of GPT's A4 source-germ normal form (task T3).

GPT's claim (``study/collab/GPT_MATH_ADVANCE_20261016.md``, sections A1-A4): at each of the four
3R candidates of ``study/exp54_cusp_discriminant.py`` -- ``(rho, z) = (0.7071, +-1.3229)`` and
``(3.5355, +-1.3229)`` -- the local position map is, after an invertible analytic change of
coordinates, ``(theta, x, y) -> (theta, x, y**4 + x*y**2)``, so the four points are not ordinary
cusps.  A4 reads off the consequences: the critical set is **two** transverse smooth curves, their
images are the line ``V = 0`` and the parabola ``V = -X**2/4``, and the *real* image of the second
one is only the half ``X <= 0`` -- so a faithful boundary plot must not draw the whole parabola.

This module re-derives the picture from ``cuspidal3r.DEFAULT_ROWS/DEFAULT_TOOL`` alone: it does not
import exp61's expressions, and it does not use exp54's elimination to obtain the germ (exp54 is
used only for one cross-check of the local boundary curve against its conic).  Phases:

1. exact MDH chain in sympy: the A1 identities, the A2 necessary condition, the ``(u, v)``
   determinant, and the fact that ``q1`` is a base rotation;
2. the four A2 source configurations and the A3/A4 non-degeneracy data
   (``B(z0) = 0``, ``B'(z0) = +-2 z0/3``, ``C0 = +-3/4``, ``r_c'(z0) = 4 z0/9``);
3. the local algebra: ``R(z0, y) - R0 = C0 y**4 + O(y**6)``, i.e. ``C{y}/(y**4)``, multiplicity 4
   (the cusp normal form has multiplicity 3), plus an argument-principle count of local preimages;
4. A4's construction executed explicitly at 40 digits: ``r_c(z)``, ``k``, ``a``, ``b``, the source
   change ``(z, y) -> (X, Y)``, the target change, and the identity ``V = Y**4 + X Y**2``;
5. the bridge to the model: the library FK confirms that the radius factors through ``(z, y)``,
   that the four A2 configurations are singular, and that ``q1`` is a base rotation;
6. the Jacobian at the candidates: corank 1 and the kernel direction (tangent to branch II);
7. the critical image measured through the library FK: the two branches, their tangency, the
   half-parabola ``V = -X**2/4`` with ``X <= 0``, and the one-sidedness in ``z``;
8. the consequence for boundary plots: a fibre census shows where the reachable boundary actually
   sits on both sides of the tangency (it switches from the parabola to the other branch).

The script prints every number it asserts and exits non-zero if a check fails.  SymPy and MPmath
are research dependencies of ``study/``, not project dependencies.

Run::

    python3 -m study.exp62_germ_normal_form
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mpmath as mp
import numpy as np
import sympy as sp

from study import cuspidal3r as c3
from study.exp54_cusp_discriminant import boundary as elimination_boundary
from study.exp54_cusp_discriminant import coefficients as elimination_coefficients

#: Working precision of the analytic construction; the residuals printed are at this precision.
DPS = 40
#: Truncation order of the exact series ``H(z, r) = sum_k h_k(z) r**k``; the tail is ``O(r**15)``.
ORDER = 14
#: Configurations per candidate in the FK-bridge check.
FK_SAMPLES = 120
#: Restarts per census; every "0 solutions" statement means "none among these restarts".
CENSUS_SEEDS = 250
#: Offset in ``z`` used by the boundary test; the two candidate boundary curves are ~2e-4 apart
#: there, which the census resolves.
BOUNDARY_OFFSET = 3e-2
#: Contour radii (in ``y``) for the argument-principle count of local preimages; the count is
#: required to be 4 at every radius, so the answer does not depend on the contour.
CONTOURS = (0.12, 0.2, 0.4)

mp.mp.dps = DPS

Z0 = sp.sqrt(7) / 2


class Ledger:
    """Collect check outcomes so that the module can exit non-zero on any failure."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        """Record and print one check outcome."""
        self.rows.append((name, bool(ok), detail))
        print(f"    [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return bool(ok)

    @property
    def ok(self) -> bool:
        """Whether every recorded check passed."""
        return all(row[1] for row in self.rows)

    @property
    def failures(self) -> list[str]:
        """The names of the failed checks."""
        return [row[0] for row in self.rows if not row[1]]


@dataclass(frozen=True)
class Candidate:
    """One of the four 3R source candidates, with the exact branch data of A2/A3.

    Attributes:
        label: Which discriminant components meet here (exp54's naming).
        radius_squared: Exact ``R = rho**2`` of the candidate.
        z0: Exact height ``z`` of the candidate.
        eps: Sign of ``A`` on the local branch (``A = eps sqrt(4 + (9/4) sin^2 v - z**2)``).
        c: ``cos(v0)`` at the candidate (``+1`` for ``v0 = 0``, ``-1`` for ``v0 = pi``).
        sin_u: Exact ``sin(u)`` of the source configuration.
        v0: Exact ``v`` of the source configuration, radians.
    """

    label: str
    radius_squared: sp.Expr
    z0: sp.Expr
    eps: int
    c: int
    sin_u: sp.Expr
    v0: sp.Expr

    @property
    def rho(self) -> sp.Expr:
        """Exact ``rho`` of the candidate."""
        return sp.sqrt(self.radius_squared)

    @property
    def cos_u(self) -> sp.Expr:
        """Exact ``cos(u)`` of the source configuration."""
        return -self.z0 / 2

    @property
    def u0(self) -> sp.Expr:
        """The source ``u`` as ``atan2(sin u, cos u)``."""
        return sp.atan2(self.sin_u, self.cos_u)


def candidates() -> list[Candidate]:
    """Return the four candidates of exp54 together with the A2 source data."""
    out: list[Candidate] = []
    for sign in (+1, -1):
        z0 = sign * Z0
        out.append(Candidate("A1 = C2", sp.Rational(1, 2), z0, -1, +1,
                             sp.Rational(-3, 4), sp.Integer(0)))
        out.append(Candidate("A0 = C2", sp.Rational(25, 2), z0, +1, -1,
                             sp.Rational(3, 4), sp.pi))
    return out


def exact_entry(value: float) -> sp.Expr:
    """Convert one float MDH entry to an exact sympy number, asserting the conversion is exact."""
    if float(value) == 0.0:
        return sp.Integer(0)
    if abs(float(value) - float(sp.pi / 2)) < 1e-15:
        return sp.pi / 2
    rational = sp.Rational(str(float(value)))
    if float(rational) != float(value):
        raise ValueError(f"MDH entry {value!r} is not exactly representable")
    return rational


def exact_chain() -> dict[str, sp.Expr]:
    """Build the 3R chain exactly from the library's MDH rows and tool.

    The composition follows ``RobotModelNumpy.fk``: ``Ms[0] Ms[1] Rz(q1) Ms[2] Rz(q2) Ms[3]
    Rz(q3) Ms[4]``, with ``Ms[1:]`` the link matrices ``Tx(a) Rx(alpha) Tz(d)`` of
    ``cuspidal3r.DEFAULT_ROWS`` and ``cuspidal3r.DEFAULT_TOOL``.

    Returns:
        The symbols and exact expressions ``q1, u, v, point, base, px, py, pz, A, R``; ``point``
        is the tool position with ``q1`` kept, ``base`` the same with ``q1 = 0``.
    """
    rows = [tuple(exact_entry(value) for value in row) for row in c3.DEFAULT_ROWS]
    tool = [tuple(exact_entry(value) for value in row) for row in c3.DEFAULT_TOOL]

    def tx(a: sp.Expr) -> sp.Matrix:
        return sp.Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def rx(alpha: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(alpha), sp.sin(alpha)
        return sp.Matrix([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    def tz(d: sp.Expr) -> sp.Matrix:
        return sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    def rz(theta: sp.Expr) -> sp.Matrix:
        c, s = sp.cos(theta), sp.sin(theta)
        return sp.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    link = [tx(a) @ rx(alpha) @ tz(d) for d, alpha, a in rows]
    ee = sp.eye(4)
    for d, alpha, a in tool:
        ee = ee @ (tx(a) @ rx(alpha) @ tz(d))
    q1, u, v = sp.symbols("q1 u v", real=True)
    chain = link[0] @ rz(q1) @ link[1] @ rz(u) @ link[2] @ rz(v) @ ee
    point = sp.simplify(chain[:3, 3])
    base = sp.simplify(point.subs(q1, 0))
    px, py, pz = base
    a = 2 * sp.sin(u) + sp.Rational(3, 2) * sp.sin(v) * sp.cos(u)
    return {
        "q1": q1, "u": u, "v": v, "point": point, "base": base, "rz": rz(q1)[:3, :3],
        "px": px, "py": py, "pz": pz, "A": a, "R": sp.simplify(px**2 + py**2),
    }


def branch_radius(z: sp.Expr, y: sp.Expr, eps: int, c: int) -> sp.Expr:
    """The branch radius ``R`` as a function of ``(z, y)``.

    On the branch where ``A = eps sqrt(4 + (9/4) sin^2 v - z**2)`` the identities of A1 give
    ``R = 33/4 - z**2 - 3 cos(v0 + y) + 2 A``; with ``y`` measured from the candidate this is the
    exact local form of the position map's radius.  It is even in ``y``, which is why A4 writes it
    as ``H(z, r)`` with ``r = y**2``.

    Args:
        z: Height coordinate of the tool point.
        y: Joint-3 offset from the candidate, radians.
        eps: Sign of ``A`` on the branch.
        c: ``cos(v0)`` at the candidate.
    """
    return (sp.Rational(33, 4) - z**2 - 3 * c * sp.cos(y)
            + 2 * eps * sp.sqrt(4 + sp.Rational(9, 4) * sp.sin(y) ** 2 - z**2))


def radial_series(z, eps: int, c: int, *, order: int = ORDER) -> list:
    """Coefficients ``h_k(z)`` of ``H(z, r) = sum_k h_k(z) r**k`` at working precision.

    ``H`` is :func:`branch_radius` in the variables ``(z, r = y**2)``.  It is analytic in ``r`` at
    ``r = 0`` because ``cos(sqrt r)`` and ``sin^2(sqrt r)`` are; the series is built here from the
    exact Taylor coefficients of those two functions plus the square-root recurrence, so it is
    exact up to the ``O(r**(order + 1))`` tail -- below ``1e-40`` on the windows used here.

    Args:
        z: Height coordinate.
        eps: Sign of ``A`` on the branch.
        c: ``cos(v0)`` at the candidate.
        order: Truncation order in ``r``.

    Returns:
        ``[h_0(z), ..., h_order(z)]`` as MPmath numbers.
    """
    z = mp.mpf(z)
    p0 = 4 - z * z
    cosine = [mp.mpf((-1) ** k) / mp.factorial(2 * k) for k in range(order + 1)]
    sine_squared = [mp.mpf(0)] + [mp.mpf((-1) ** (k + 1)) * mp.mpf(2) ** (2 * k - 1)
                                  / mp.factorial(2 * k) for k in range(1, order + 1)]
    inner = [mp.mpf(9) / 4 * value / p0 for value in sine_squared]
    root = [mp.mpf(0)] * (order + 1)
    root[0] = mp.mpf(1)
    for n in range(1, order + 1):
        root[n] = (inner[n] - sum(root[k] * root[n - k] for k in range(1, n))) / 2
    scale = mp.sqrt(p0)
    series = [-3 * c * cosine[k] + 2 * eps * scale * root[k] for k in range(order + 1)]
    series[0] += mp.mpf(33) / 4 - z * z
    return series


def series_value(series: list, r) -> mp.mpf:
    """Evaluate a truncated series in ``r`` (``r`` may be complex)."""
    r = mp.mpmathify(r)
    return sum(coefficient * r**k for k, coefficient in enumerate(series))


def series_derivative(series: list, r, order: int = 1) -> mp.mpf:
    """Evaluate the ``order``-th derivative in ``r`` of a truncated series."""
    r = mp.mpmathify(r)
    total = mp.mpf(0)
    for k in range(order, len(series)):
        factor = mp.mpf(1)
        for j in range(order):
            factor *= k - j
        total += factor * series[k] * r ** (k - order)
    return total


def critical_radius(series: list) -> mp.mpf:
    """Solve ``dH/dr = 0`` for the root near ``r = 0`` (Newton at working precision)."""
    r = -series[1] / (2 * series[2])
    for _ in range(80):
        step = series_derivative(series, r, 1) / series_derivative(series, r, 2)
        r -= step
        if abs(step) < mp.mpf("1e-38"):
            break
    return r


class GermFrame:
    """The A4 normal-form frame at one candidate, built from the exact radius series.

    The frame carries ``eps1 = sign(H_rr)`` and ``sigma = sign(b)``, and the maps

    * ``a(z) = k(z, 0)`` with ``k = (r - r_c(z)) sqrt(|G|)``, ``G = (H - H(z, r_c))/(r-r_c)^2``;
    * ``X = 2 sigma a(z)``, ``Y = y sqrt(|b(z, y**2)|)``, ``b = (k(z, y**2) - a)/y**2``;
    * ``V = (H(z, r) - H(z, r_c(z)))/eps1 - a(z)**2``.

    By construction ``H - H(z, r_c(z)) = eps1 k**2`` and ``k = a + sigma Y**2``, so
    ``V = Y**4 + X Y**2``; ``(z, y) -> (X, Y)`` and ``(z, R) -> (X, V)`` are the source and target
    changes of A4.  The construction is only an equivalence if its hypotheses hold (``G``
    non-vanishing, ``a'(z0) != 0``, ``b(z0, 0) != 0``), which is what the caller checks.
    """

    def __init__(self, candidate: Candidate) -> None:
        self.candidate = candidate
        self.eps = candidate.eps
        self.c = candidate.c
        self.z0 = mp.mpf(str(sp.N(candidate.z0, 30)))
        self.eps1 = 1 if radial_series(self.z0, self.eps, self.c)[2] > 0 else -1
        probe = mp.mpf("1e-6")
        self.sigma = 1 if self.b_of(self.z0, mp.sqrt(probe)) > 0 else -1

    def _k(self, series: list, r_c, r) -> mp.mpf:
        """``k(z, r)`` for one series: ``(r - r_c) sqrt(|G|)`` with ``G`` the divided difference."""
        residual = series_value(series, r) - series_value(series, r_c)
        return (r - r_c) * mp.sqrt(abs(residual / (r - r_c) ** 2))

    def series(self, z) -> list:
        """The radius series at ``z``."""
        return radial_series(z, self.eps, self.c)

    def critical_radius_of(self, z) -> mp.mpf:
        """``r_c(z)``, the (unique, near ``r = 0``) root of ``H_r(z, r) = 0``."""
        return critical_radius(self.series(z))

    def a_of_z(self, z) -> mp.mpf:
        """``a(z) = k(z, 0)``."""
        series = self.series(z)
        return self._k(series, critical_radius(series), mp.mpf(0))

    def b_of(self, z, y) -> mp.mpf:
        """``b(z, y**2) = (k(z, y**2) - a(z))/y**2`` (the caller passes ``y != 0``)."""
        y = mp.mpf(y)
        series = self.series(z)
        r_c = critical_radius(series)
        a = self._k(series, r_c, mp.mpf(0))
        return (self._k(series, r_c, y * y) - a) / (y * y)

    def radius(self, z, r) -> mp.mpf:
        """``H(z, r)``."""
        return series_value(self.series(z), r)

    def coordinates(self, z, y) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        """Return ``(X, Y, V)``, the constructed normal-form coordinates of ``(z, y)``."""
        y = mp.mpf(y)
        r = y * y
        series = self.series(z)
        r_c = critical_radius(series)
        a = self._k(series, r_c, mp.mpf(0))
        if r == 0:
            return mp.mpf(0), mp.mpf(0), mp.mpf(0)
        k = self._k(series, r_c, r)
        y_coordinate = y * mp.sqrt(abs((k - a) / r))
        heights = (series_value(series, r) - series_value(series, r_c)) / self.eps1 - a * a
        return 2 * self.sigma * a, y_coordinate, heights


def local_preimage_count(frame: GermFrame, z_target, radius_target, epsilon) -> mp.mpf:
    """Count the zeros of ``y -> H(z*, y**2) - R*`` inside ``|y| < epsilon``.

    Uses the argument principle, ``N = (1/2 pi) int Re(y f'(y)/f(y)) d theta`` on the circle; the
    count is the multiplicity of the target as a value of the local map.
    """
    series = frame.series(z_target)
    radius_target = mp.mpf(radius_target)

    def integrand(theta):
        y = epsilon * mp.e ** (1j * theta)
        value = series_value(series, y * y) - radius_target
        derivative = 2 * y * series_derivative(series, y * y, 1)
        return mp.re(y * derivative / value)

    return mp.quad(integrand, [0, mp.pi, 2 * mp.pi]) / (2 * mp.pi)


def fk_radius_height(model, u: float, v: float) -> tuple[float, float]:
    """The library FK of ``q = (0, u, v)``, reduced to ``(R, z)``."""
    point = c3.position(model, np.array([0.0, u, v]))
    return float(point[0] ** 2 + point[1] ** 2), float(point[2])


def branch_two_u(v: float, u0: float) -> float:
    """Solve the branch-II condition ``A + (3/2) cos v = 0`` for the ``u`` nearest ``u0``."""
    amplitude = np.hypot(2.0, 1.5 * np.sin(v))
    phase = np.arctan2(1.5 * np.sin(v), 2.0)
    target = float(np.clip(-(1.5 * np.cos(v)) / amplitude, -1.0, 1.0))
    options = [-phase + np.arcsin(target), -phase + np.pi - np.arcsin(target)]
    return min(options, key=lambda value: abs((value - u0 + np.pi) % (2 * np.pi) - np.pi))


def census_count(model, radius_squared: float, z: float, seed: int) -> int:
    """Number of real position-IK solutions found at ``(sqrt(R), 0, z)`` by a multi-start census."""
    if radius_squared <= 0:
        return 0
    fiber = c3.census_position(model, np.array([np.sqrt(radius_squared), 0.0, z]),
                               seeds=CENSUS_SEEDS, rng=np.random.default_rng(seed),
                               tolerance=1e-10, merge=1e-7)
    return len(fiber.solutions)


def phase_derivation(ledger: Ledger) -> dict[str, sp.Expr]:
    """Phase 1: the exact A1 identities, the A2 condition and the determinant."""
    print("phase 1: exact MDH chain (sympy, built here; exp61 is not imported)")
    data = exact_chain()
    u, v = data["u"], data["v"]
    a, radius, pz = data["A"], data["R"], data["pz"]
    ledger.check("q1 acts as a rotation about the world z axis",
                 sp.simplify(data["point"] - data["rz"] * data["base"]) == sp.zeros(3, 1))
    ledger.check("px = 1 + A", sp.simplify(data["px"] - (1 + a)) == 0)
    ledger.check("py = 3/2 cos v - 1",
                 sp.simplify(data["py"] - (sp.Rational(3, 2) * sp.cos(v) - 1)) == 0)
    ledger.check("z = 3/2 sin u sin v - 2 cos u",
                 sp.simplify(pz - (sp.Rational(3, 2) * sp.sin(u) * sp.sin(v)
                                   - 2 * sp.cos(u))) == 0)
    ledger.check("A^2 + z^2 = 4 + (9/4) sin^2 v",
                 sp.simplify(a**2 + pz**2 - 4 - sp.Rational(9, 4) * sp.sin(v) ** 2) == 0)
    ledger.check("R = 2A + 33/4 - z^2 - 3 cos v",
                 sp.trigsimp(radius - (2 * a + sp.Rational(33, 4) - pz**2
                                       - 3 * sp.cos(v))) == 0)
    determinant = sp.trigsimp(sp.Matrix([radius, pz]).jacobian([u, v]).det())
    ledger.check("det D(R, z) = -3 sin v (A + 3/2 cos v)",
                 sp.simplify(determinant + 3 * sp.sin(v)
                             * (a + sp.Rational(3, 2) * sp.cos(v))) == 0)
    total = radius + pz**2
    necessary = (18 * sp.cos(v) ** 2 + 6 * (total - sp.Rational(33, 4)) * sp.cos(v)
                 + (total - sp.Rational(33, 4)) ** 2 + 4 * pz**2 - 25)
    ledger.check("the A2 necessary condition vanishes identically on the FK image",
                 sp.simplify(sp.expand_trig(necessary)) == 0)
    print(f"    det D(R, z) = {sp.factor(determinant)}")
    return data


def phase_candidates(ledger: Ledger, data: dict[str, sp.Expr]) -> list[tuple[Candidate, dict]]:
    """Phase 2: the A2 source configurations and the A3/A4 non-degeneracy data."""
    print("\nphase 2: the four candidates -- A2 source data and A3/A4 non-degeneracy")
    z = sp.Symbol("z", real=True)
    y = sp.Symbol("y", real=True)
    checked: list[tuple[Candidate, dict]] = []
    for candidate in candidates():
        branch = branch_radius(z, y - candidate.v0, candidate.eps, candidate.c)
        tag = f"{candidate.label} z0={sp.nsimplify(candidate.z0)}"
        ledger.check(f"{tag}: R(z0, 0) = R0",
                     sp.simplify(branch.subs({z: candidate.z0, y: candidate.v0})
                                 - candidate.radius_squared) == 0,
                     f"R0 = {sp.nsimplify(candidate.radius_squared)}")
        ledger.check(f"{tag}: sin^2 u + cos^2 u = 1",
                     sp.simplify(candidate.sin_u**2 + candidate.cos_u**2 - 1) == 0,
                     f"sin u = {candidate.sin_u}, cos u = {sp.simplify(candidate.cos_u)}")
        at_source = {sp.sin(data["u"]): candidate.sin_u, sp.cos(data["u"]): candidate.cos_u,
                     sp.sin(data["v"]): sp.sin(candidate.v0),
                     sp.cos(data["v"]): sp.cos(candidate.v0)}
        dz_du = sp.simplify(sp.diff(data["pz"], data["u"]).subs(at_source))
        amplitude = sp.simplify(data["A"].subs(at_source))
        ledger.check(f"{tag}: (u, v) -> (z, v) is a local diffeomorphism (z_u != 0)",
                     sp.simplify(dz_du - 2 * candidate.sin_u) == 0 and dz_du != 0,
                     f"z_u = {dz_du}")
        ledger.check(f"{tag}: A(u0, v0) = 3 eps/2 != 0 (the branch sign is well defined)",
                     sp.simplify(amplitude - sp.Rational(3, 2) * candidate.eps) == 0,
                     f"A = {amplitude}")
        coefficient_b = sp.simplify(sp.diff(branch, y, 2).subs(y, candidate.v0) / 2)
        coefficient_c = sp.simplify(sp.diff(branch, y, 4).subs(y, candidate.v0) / 24)
        prime_b = sp.simplify(sp.diff(coefficient_b, z).subs(z, candidate.z0))
        at_z0 = sp.simplify(coefficient_b.subs(z, candidate.z0))
        c0 = sp.simplify(coefficient_c.subs(z, candidate.z0))
        ledger.check(f"{tag}: B(z0) = 0", at_z0 == 0)
        ledger.check(f"{tag}: B'(z0) = eps * 2 z0 / 3 != 0",
                     sp.simplify(prime_b - candidate.eps * 2 * candidate.z0 / 3) == 0,
                     f"B'(z0) = {sp.simplify(prime_b)} = {float(prime_b):+.6f}")
        ledger.check(f"{tag}: C0 = -eps * 3/4 != 0",
                     sp.simplify(c0 + candidate.eps * sp.Rational(3, 4)) == 0,
                     f"C0 = {sp.simplify(c0)}")
        critical_slope = sp.simplify(-prime_b / (2 * c0))
        ledger.check(f"{tag}: r_c'(z0) = 4 z0 / 9 != 0",
                     sp.simplify(critical_slope - 4 * candidate.z0 / 9) == 0,
                     f"r_c'(z0) = {sp.nsimplify(critical_slope)}")
        phi1_prime = sp.simplify(sp.diff(branch, z).subs({z: candidate.z0, y: candidate.v0}))
        gap = sp.simplify(-prime_b**2 / (4 * c0))
        print(f"    {candidate.label} (rho, z) = ({float(candidate.rho):.4f}, "
              f"{float(candidate.z0):+.4f}): v0 = {float(candidate.v0):.4f}, "
              f"B'(z0) = {float(prime_b):+.6f}, C0 = {float(c0):+.4f}, "
              f"r_c'(z0) = {float(critical_slope):+.6f}, phi1'(z0) = {float(phi1_prime):+.6f}, "
              f"second-order gap = {float(gap):+.6f}")
        checked.append((candidate, {"phi1_prime": phi1_prime, "gap": gap}))
    ledger.check("the second-order gap -B'^2/(4 C0) is non-zero at all four points",
                 all(abs(complex(entry["gap"])) > 1e-6 for _, entry in checked),
                 f"|gap| = {abs(float(checked[0][1]['gap'])):.6f} = 7/27")
    return checked


def phase_local_algebra(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 3: multiplicity of the germ versus the ordinary cusp."""
    print("\nphase 3: the local algebra -- multiplicity 4, not the cusp's 3")
    z, y = sp.symbols("z y", real=True)
    for candidate, _ in checked:
        branch = branch_radius(z, y, candidate.eps, candidate.c)
        taylor = [sp.simplify(sp.diff(branch, y, n).subs({z: candidate.z0, y: 0})
                              / sp.factorial(n)) for n in range(7)]
        printed = ", ".join(f"{float(value):+.6f}" for value in taylor)
        print(f"    {candidate.label} z0={float(candidate.z0):+.4f}: y-Taylor of R(z0, y): "
              f"[{printed}]")
        ledger.check(f"{candidate.label}: only even orders survive and orders 0, 2 vanish",
                     all(taylor[n] == 0 for n in (1, 3, 5))
                     and sp.simplify(taylor[2]) == 0
                     and sp.simplify(taylor[0] - candidate.radius_squared) == 0)
        ledger.check(f"{candidate.label}: the quartic coefficient is C0 = "
                     f"{-candidate.eps * 3 / 4:+.4f} != 0",
                     sp.simplify(taylor[4] + candidate.eps * sp.Rational(3, 4)) == 0)
        frame = GermFrame(candidate)
        counts = [local_preimage_count(frame, frame.z0 + mp.mpf("1e-5"),
                                       frame.radius(frame.z0, 0) + mp.mpf("1e-6"), mp.mpf(radius))
                  for radius in CONTOURS]
        ledger.check(f"{candidate.label}: 4 local preimages over C (multiplicity 4)",
                     all(abs(count - 4) < mp.mpf("1e-6") for count in counts),
                     "argument-principle counts at |y| = "
                     + ", ".join(mp.nstr(radius, 3) for radius in CONTOURS) + ": "
                     + ", ".join(mp.nstr(count, 10) for count in counts))
    x, y_symbol = sp.symbols("x y")
    cusp = sp.groebner([x, y_symbol**3 + x * y_symbol], x, y_symbol, order="lex")
    quartic = sp.groebner([x, y_symbol**4 + x * y_symbol**2], x, y_symbol, order="lex")
    cusp_degree = int(sp.Poly(cusp.exprs[-1], y_symbol).degree())
    quartic_degree = int(sp.Poly(quartic.exprs[-1], y_symbol).degree())
    print("    cusp normal form (x, y^3 + x y):      local algebra C{y}/(y^3), dimension 3")
    print("    A4 normal form (x, y^4 + x y^2):      local algebra C{y}/(y^4), dimension 4")
    ledger.check("the cusp normal form has multiplicity 3 and this germ has 4",
                 cusp_degree == 3 and quartic_degree == 4)


def phase_construction(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 4: A4's construction, executed and checked at working precision."""
    print("\nphase 4: A4's construction executed at 40 digits (coordinate change + normal form)")
    print("    source (z, y) -> (X, Y) with X = 2 sigma a(z), Y = y sqrt(|b(z, y^2)|);")
    print("    target (z, R) -> (X, V) with V = (H - H(z, r_c(z)))/eps1 - a(z)^2.")
    rng = np.random.default_rng(62)
    for candidate, _ in checked:
        frame = GermFrame(candidate)
        worst_identity = mp.mpf(0)
        smallest_jacobian = mp.mpf("1e9")
        smallest_gap = mp.mpf("1e9")
        worst_newton = mp.mpf(0)
        sign_consistent = True
        for _ in range(40):
            step = mp.mpf("1e-3")
            z_value = frame.z0 + mp.mpf(float(rng.uniform(-1, 1))) * step
            y_value = mp.mpf(float(rng.uniform(-2e-2, 2e-2)))
            series = frame.series(z_value)
            r_c = critical_radius(series)
            worst_newton = max(worst_newton, abs(series_derivative(series, r_c, 1)))
            residual = series_value(series, y_value * y_value) - series_value(series, r_c)
            smallest_gap = min(smallest_gap,
                               abs(residual / (y_value * y_value - r_c) ** 2))
            sign_consistent = sign_consistent and (1 if frame.b_of(z_value, y_value) > 0 else -1) \
                == frame.sigma
            x_value, height, v_value = frame.coordinates(z_value, y_value)
            worst_identity = max(worst_identity,
                                 abs(v_value - (height**4 + x_value * height**2)))
            delta = mp.mpf("1e-12")
            dx_dz = (frame.coordinates(z_value + delta, y_value)[0]
                     - frame.coordinates(z_value - delta, y_value)[0]) / (2 * delta)
            dy_dy = (frame.coordinates(z_value, y_value + delta)[1]
                     - frame.coordinates(z_value, y_value - delta)[1]) / (2 * delta)
            smallest_jacobian = min(smallest_jacobian, abs(dx_dz * dy_dy))
        derivative_a = (frame.a_of_z(frame.z0 + step) - frame.a_of_z(frame.z0 - step)) / (2 * step)
        ledger.check(f"{candidate.label}: k(z, r_c(z)) = 0 (Newton residual of H_r)",
                     worst_newton < mp.mpf("1e-30"), f"max |H_r| = {mp.nstr(worst_newton, 4)}")
        ledger.check(f"{candidate.label}: G stays away from 0 (analytic square root of G)",
                     smallest_gap > mp.mpf("1e-3"), f"min |G| = {mp.nstr(smallest_gap, 8)}")
        ledger.check(f"{candidate.label}: b has constant sign, so sigma is well defined",
                     sign_consistent and frame.sigma in (-1, 1),
                     f"sigma = {frame.sigma}, eps1 = {frame.eps1}")
        ledger.check(f"{candidate.label}: a'(z0) = -r_c'(z0) K(z0, 0) != 0",
                     abs(derivative_a) > mp.mpf("1e-3"),
                     f"a'(z0) ~ {mp.nstr(derivative_a, 8)} (exact 4 z0/9 * sqrt|C0| = "
                     f"{mp.nstr(abs(4 * frame.z0 / 9) * mp.sqrt(mp.mpf(3) / 4), 8)})")
        ledger.check(f"{candidate.label}: the normal-form identity V = Y^4 + X Y^2",
                     worst_identity < mp.mpf("1e-35"),
                     f"max residual = {mp.nstr(worst_identity, 4)}")
        ledger.check(f"{candidate.label}: (z, y) -> (X, Y) is invertible",
                     smallest_jacobian > mp.mpf("1e-3"),
                     f"min |det| = {mp.nstr(smallest_jacobian, 8)}")


def phase_bridge(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 5: the germ against the library FK."""
    print("\nphase 5: bridge to the library FK (cuspidal3r model)")
    model = c3.build()
    rng = np.random.default_rng(7)
    worst_rotation = 0.0
    for q in rng.uniform(-np.pi, np.pi, (200, 3)):
        point = c3.position(model, q)
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        turn = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        worst_rotation = max(worst_rotation, float(np.linalg.norm(
            point - turn @ c3.position(model, np.array([0.0, q[1], q[2]])))))
    ledger.check("q1 is a base rotation over 200 random configurations",
                 worst_rotation < 1e-14, f"max |P - Rz(q1) P0| = {worst_rotation:.3e} m")
    for candidate, _ in checked:
        frame = GermFrame(candidate)
        u0 = float(candidate.u0)
        v0 = float(candidate.v0)
        worst_bridge = 0.0
        worst_branch = 0.0
        for _ in range(FK_SAMPLES):
            u = u0 + rng.uniform(-0.02, 0.02)
            v = v0 + rng.uniform(-0.03, 0.03)
            radius, height = fk_radius_height(model, u, v)
            worst_bridge = max(worst_bridge,
                               abs(float(frame.radius(height, (v - v0) ** 2)) - radius))
            amplitude = 2 * np.sin(u) + 1.5 * np.sin(v) * np.cos(u)
            worst_branch = max(worst_branch, abs(amplitude - candidate.eps * 1.5))
        ledger.check(f"{candidate.label}: R_fk = H(z_fk, (v - v0)^2) over {FK_SAMPLES} samples",
                     worst_bridge < 1e-13, f"max residual = {worst_bridge:.3e}")
        ledger.check(f"{candidate.label}: samples stay on the eps = {candidate.eps:+d} branch",
                     worst_branch < 0.1, f"max |A - 3/2 eps| = {worst_branch:.3e}")
        radius, height = fk_radius_height(model, u0, v0)
        determinant = c3.det_position(model, np.array([0.0, u0, v0]))
        ledger.check(f"{candidate.label}: the A2 source maps to the candidate and is singular",
                     abs(radius - float(candidate.radius_squared)) < 1e-12
                     and abs(height - float(candidate.z0)) < 1e-12 and abs(determinant) < 1e-12,
                     f"FK = ({radius:.12f}, {height:+.12f}), det J = {determinant:.2e}")


def jacobian_uv(model, u: float, v: float, *, step: float = 1e-7) -> np.ndarray:
    """Central-difference Jacobian of the library FK's ``(R, z)`` with respect to ``(u, v)``.

    Args:
        model: The 3R model.
        u: Joint-2 value, radians.
        v: Joint-3 value, radians.
        step: Central-difference step.

    Returns:
        The ``(2, 2)`` matrix whose columns are ``d/du`` and ``d/dv``.
    """
    columns = []
    for du, dv in ((step, 0.0), (0.0, step)):
        plus = np.array(fk_radius_height(model, u + du, v + dv))
        minus = np.array(fk_radius_height(model, u - du, v - dv))
        columns.append((plus - minus) / (2 * step))
    return np.column_stack(columns)


def phase_jacobian(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 6: the Jacobian at the candidates -- corank 1 and the kernel direction.

    ``det D(R, z) = -3 sin v (A + 3/2 cos v)`` vanishes at every candidate, so the ``(2, 2)``
    Jacobian there has rank one and a single kernel direction: that direction is the fold
    direction.  In the normal form the kernel of ``(X, Y) -> (X, Y**4 + X Y**2)`` is ``d/dY``,
    which is tangent to the critical branch ``{X = -2 Y**2}`` (branch II here) and transverse to
    ``{Y = 0}`` (branch I) -- the check below asks the library FK for exactly that.
    """
    print("\nphase 6: the (u, v) Jacobian of (R, z) at the candidates (library FK)")
    model = c3.build()
    for candidate, _ in checked:
        u0 = float(candidate.u0)
        v0 = float(candidate.v0)
        matrix = jacobian_uv(model, u0, v0)
        _, singular, right = np.linalg.svd(matrix)
        kernel = right[-1]
        tangent_two = np.array([branch_two_u(v0 + 1e-5, u0) - branch_two_u(v0 - 1e-5, u0), 2e-5])
        tangent_two = tangent_two / np.linalg.norm(tangent_two)
        kernel_residual = float(np.linalg.norm(matrix @ kernel))
        ledger.check(f"{candidate.label}: the Jacobian has corank 1 (one kernel direction)",
                     float(singular[0]) > 1.0 and float(singular[-1]) < 1e-6
                     and kernel_residual < 1e-6,
                     f"singular values = ({singular[0]:.6f}, {singular[-1]:.2e}), "
                     f"|M k| = {kernel_residual:.2e}")
        ledger.check(f"{candidate.label}: the kernel is tangent to branch II, transverse to I",
                     abs(float(kernel @ tangent_two)) > 0.999 and abs(float(kernel[0])) < 0.9,
                     f"|k . t_II| = {abs(float(kernel @ tangent_two)):.6f}, "
                     f"|k . t_I| = {abs(float(kernel[0])):.6f} (t_I = d/du), "
                     f"k = ({kernel[0]:+.4f}, {kernel[1]:+.4f})")


def phase_critical_image(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 7: the critical curves through the library FK, and the half-parabola."""
    print("\nphase 7: critical set and critical image, sampled through the library FK")
    print("    branch I  = {sin v = 0} at v = v0;  branch II = {A + (3/2) cos v = 0}")
    model = c3.build()
    for candidate, data in checked:
        frame = GermFrame(candidate)
        u0 = float(candidate.u0)
        v0 = float(candidate.v0)
        z0 = float(candidate.z0)
        one = [(u, v0) for u in u0 + np.linspace(-2e-3, 2e-3, 201)]
        two = [(branch_two_u(v, u0), v) for v in v0 + np.linspace(-2e-2, 2e-2, 401)]
        images = {"I": [fk_radius_height(model, u, v) for u, v in one],
                  "II": [fk_radius_height(model, u, v) for u, v in two]}
        heights = {key: np.array([height for _, height in value]) for key, value in images.items()}
        radii = {key: np.array([radius for radius, _ in value]) for key, value in images.items()}
        one_sided = bool(heights["II"].min() >= z0 - 1e-13 or heights["II"].max() <= z0 + 1e-13)
        two_sided = bool(heights["I"].min() < z0 < heights["I"].max())
        ledger.check(f"{candidate.label}: branch II is one-sided in z, branch I is not",
                     one_sided and two_sided,
                     f"z-z0 in [{heights['II'].min() - z0:+.2e}, {heights['II'].max() - z0:+.2e}]"
                     f" (II), [{heights['I'].min() - z0:+.2e}, {heights['I'].max() - z0:+.2e}] (I)")
        worst_line = max(abs(radius - float(frame.radius(height, 0)))
                         for radius, height in images["I"])
        ledger.check(f"{candidate.label}: branch I image is the curve R = H(z, 0)",
                     worst_line < 1e-13, f"max |R_fk - H(z, 0)| = {worst_line:.3e}")
        worst_parabola = mp.mpf(0)
        worst_relation = mp.mpf(0)
        largest_x = mp.mpf("-1e9")
        for height, (_, v) in zip(heights["II"], two):
            x_value, y_value, v_value = frame.coordinates(height, v - v0)
            worst_parabola = max(worst_parabola, abs(v_value + x_value**2 / 4))
            worst_relation = max(worst_relation, abs(x_value + 2 * y_value**2))
            largest_x = max(largest_x, x_value)
        ledger.check(f"{candidate.label}: branch II image lies on V = -X^2/4 with X = -2Y^2 <= 0",
                     worst_parabola < mp.mpf("1e-12") and worst_relation < mp.mpf("1e-12")
                     and largest_x <= 0,
                     f"max |V + X^2/4| = {mp.nstr(worst_parabola, 4)}, "
                     f"max |X + 2Y^2| = {mp.nstr(worst_relation, 4)}, "
                     f"largest X = {mp.nstr(largest_x, 4)}")
        slope = {}
        curvature = {}
        for key in ("I", "II"):
            fit = np.polyfit(heights[key] - z0, radii[key], 2)
            curvature[key], slope[key] = float(fit[0]), float(fit[1])
        ledger.check(f"{candidate.label}: the two critical images are tangent (slopes agree)",
                     abs(slope["I"] - slope["II"]) < 1e-4,
                     f"slope I = {slope['I']:+.6f}, slope II = {slope['II']:+.6f}, "
                     f"exact H_z(z0, 0) = {float(data['phi1_prime']):+.6f}")
        ledger.check(f"{candidate.label}: curvature gap = -B'^2/(4 C0)",
                     abs((curvature["II"] - curvature["I"]) - float(data["gap"])) < 1e-3,
                     f"measured {curvature['II'] - curvature['I']:+.5f}, "
                     f"predicted {float(data['gap']):+.5f}")


def elimination_radius(z: float, guess: float) -> float:
    """The local root of exp54's boundary conic ``C2(R, z**2) = 0`` near ``guess``."""
    z2 = z * z
    roots = np.roots([16.0, 32.0 * z2 - 264.0, 16.0 * z2 * z2 - 136.0 * z2 + 289.0])
    real = roots[np.abs(roots.imag) < 1e-9 * np.maximum(1.0, np.abs(roots.real))].real
    if not len(real):
        return float("nan")
    return float(min(real, key=lambda value: abs(value - guess)))


def phase_boundary(ledger: Ledger, checked: list[tuple[Candidate, dict]]) -> None:
    """Phase 8: where the reachable boundary actually is, on both sides of the tangency.

    The normal form predicts both the boundary curve and the *side* the reachable set lies on:
    the image of the germ is ``V >= 0`` for ``X >= 0`` and ``V >= -X**2/4`` for ``X <= 0``, and
    ``V = (R - phi2(z))/eps1 - a(z)**2``.  Hence the reachable set is ``R >= boundary`` when
    ``eps1 = +1`` and ``R <= boundary`` when ``eps1 = -1``, with the boundary equal to ``phi2``
    (the branch-II image) where branch II is real and to ``phi1 = H(z, 0)`` otherwise.  The probes
    below are one census each on the two sides of each candidate curve; the discriminating probe
    is the one between the two curves, which is reachable exactly when the normal form says so.
    """
    print("\nphase 8: the reachable boundary (fibre census) -- the half-parabola restriction")
    print(f"    offsets dz = +-{BOUNDARY_OFFSET}; there the two candidate boundary curves differ "
          f"by ~2e-4, which the census resolves")
    model = c3.build()
    for candidate, _ in checked:
        frame = GermFrame(candidate)
        z0 = float(candidate.z0)
        for offset in (+BOUNDARY_OFFSET, -BOUNDARY_OFFSET):
            z = z0 + offset
            r_c = frame.critical_radius_of(z)
            line = float(frame.radius(z, 0))
            parabola = float(frame.radius(z, r_c))
            branch_two_real = float(r_c) > 0
            low, high = min(line, parabola), max(line, parabola)
            boundary_is_low = (parabola < line) if branch_two_real else (line < parabola)
            grows_with_radius = frame.eps1 > 0
            counts = {name: max(census_count(model, value, z, seed) for seed in (11, 12))
                      for name, value in (("outside_low", low - 1e-4),
                                          ("between", 0.5 * (low + high)),
                                          ("outside_high", high + 1e-4))}
            expected = {
                "outside_low": not grows_with_radius,
                "between": boundary_is_low == grows_with_radius,
                "outside_high": grows_with_radius,
            }
            ok = all((counts[name] > 0) == expected[name] for name in counts)
            ledger.check(f"{candidate.label} dz={offset:+.2f}: boundary switch", ok,
                         f"phi1 = {line:.9f}, phi2 = {parabola:.9f}, branch II real: "
                         f"{branch_two_real} -> boundary = "
                         f"{'phi2' if branch_two_real else 'phi1'}, reachable side: "
                         f"R {'>=' if grows_with_radius else '<='} boundary (eps1 = "
                         f"{frame.eps1:+d}); census outside_low/between/outside_high = "
                         f"{counts['outside_low']}/{counts['between']}/{counts['outside_high']}")
            if branch_two_real:
                # cross-check: the branch-II image is the local branch of exp54's conic C2 = 0
                conic = elimination_radius(z, parabola)
                residual = abs(elimination_boundary(np.sqrt(parabola), z))
                a0, a1, a2 = elimination_coefficients(np.sqrt(line), z)
                elbow = a1 if candidate.label == "A1 = C2" else a0
                print(f"    {candidate.label} dz={offset:+.2f}: exp54 cross-check -- "
                      f"|C2(phi2)| = {residual:.2e}, conic root = {conic:.9f} "
                      f"(phi2 = {parabola:.9f}), elbow component at phi1 = {elbow:+.3e}")
                ledger.check(f"{candidate.label} dz={offset:+.2f}: branch II image is the local "
                             f"C2 = 0 branch", residual < 1e-8 and abs(conic - parabola) < 1e-6)


def main() -> int:
    """Run every phase and return a non-zero exit code if any check failed."""
    print("exp62: independent verification of GPT's A4 germ normal form (task T3)\n")
    ledger = Ledger()
    data = phase_derivation(ledger)
    checked = phase_candidates(ledger, data)
    phase_local_algebra(ledger, checked)
    phase_construction(ledger, checked)
    phase_bridge(ledger, checked)
    phase_jacobian(ledger, checked)
    phase_critical_image(ledger, checked)
    phase_boundary(ledger, checked)
    print("\nsummary")
    failed = ledger.failures
    print(f"    {len(ledger.rows) - len(failed)}/{len(ledger.rows)} checks passed")
    if failed:
        print("    FAILED: " + "; ".join(failed))
        return 1
    print("    verdict: GPT's A4 holds at the four candidate points -- the germ is analytically")
    print("    equivalent to (X, Y) -> (X, Y^4 + X Y^2); its critical set is two transverse smooth")
    print("    curves, its critical image is the line V = 0 plus the *half* parabola")
    print("    V = -X^2/4 (X <= 0), and the reachable boundary switches branches at the tangency.")
    print("    Scope (see study/collab/T3-germ-verification.md): the four points only; the arm's")
    print("    other critical branches and the global elimination relation are not claimed here")
    print("    beyond the local C2 cross-check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
