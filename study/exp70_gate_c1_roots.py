"""exp70: Gate C1 -- exact contraction certificates for the six common-base roots.

C1 of ``study/collab/GPT_GATE_C_SPEC_20261017.md``: implement the exact six-dimensional
square system of section 4 at the base point (lambda = 0) on the frozen exact model of
exp69 together with its analytic Jacobian, and certify a strict contraction box for each
of the six seed roots {1, 4, 6, 8, 10, 13} taken from ``study/exp68_h_loop.npz``.

Backend: exact rational interval arithmetic over ``fractions.Fraction`` endpoints.  No
floating-point rounding enters any certificate constant: interval sums use exact
endpoint arithmetic, reciprocals use exact rational inversion, and every inequality is
an exact rational comparison.  The only floating-point inputs are the box centres
``z = exp(i q_seed)`` (used merely to select the box, then frozen as exact binary
rationals) and the preconditioner ``B`` (a numpy double inverse of the float Jacobian
at the centre, also frozen as exact binary rationals) -- exactly as sections 4 and 5 of
the spec allow.  The contraction constants are Krawczyk-style at ``t = 0``:
``eta = ||B f(c)||_inf`` exactly, and ``kappa_i >= sup_{x in X} ||row_i(I - B D_x
f(x))||_1`` by interval evaluation of the analytic Jacobian over the certified box
``X = c + [-r, r]^12`` with ``r = 2^-40``.  The strict margin ``r - eta - kappa_i r >
0`` together with ``kappa < 1`` makes ``T(x) = x - B f(x)`` a strict contraction of
``X`` into its interior, so ``X`` contains exactly one root, which is regular;
``|trace S - 3| < 4`` over the whole box plus ``z_j != 0`` upgrades it to a genuine
forward-kinematics solution (a skew-free ``S`` with ``S^T S = I``, ``det S = 1`` and
trace 3 is ``S = I``).  The six boxes are checked pairwise disjoint.

``--verify`` recomputes every stored number from the frozen inputs (model JSON, npz
seeds, stored centres and B) and compares exactly, so a reviewer can rerun the
arithmetic and reproduce the same certificate bit for bit.

Boundary: C1 only -- no segment certification, no path certification, no permutation
claims, and no ``path_certified`` flag anywhere.

Run::

    python3 -m study.exp70_gate_c1_roots
    python3 -m study.exp70_gate_c1_roots --root 1 --out /tmp/exp70_root1.json
    python3 -m study.exp70_gate_c1_roots --verify study/exp70_gate_c1_certificate.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sympy as sp
import yaml

CONFIG_PATH = Path("model/configs/rokae_sr5.yaml")
MODEL_JSON = Path("study/exp69_exact_model.json")
SEED_NPZ = Path("study/exp68_h_loop.npz")
CERT_JSON = Path("study/exp70_gate_c1_certificate.json")
TRACKED_INDICES = (1, 4, 6, 8, 10, 13)
DEFAULT_RADIUS_K = 40
MAX_RADIUS_SEARCH_K = (16, 14, 12, 10, 8)
PROVENANCE_TOLERANCE = Fraction(1, 10**12)
B_INVERSE_TOLERANCE = Fraction(1, 10**9)
HALF = Fraction(1, 2)
ZERO = Fraction(0)
ONE = Fraction(1)

Exponents = tuple[int, int, int, int, int, int]
Cplx = tuple[Fraction, Fraction]
Laurent = dict[Exponents, Cplx]
XVector = tuple[Fraction, ...]


# ---------------------------------------------------------------------------
# exact complex rational arithmetic
# ---------------------------------------------------------------------------


def cplx_add(a: Cplx, b: Cplx) -> Cplx:
    """Add two exact Gaussian rationals."""
    return (a[0] + b[0], a[1] + b[1])


def cplx_sub(a: Cplx, b: Cplx) -> Cplx:
    """Subtract two exact Gaussian rationals."""
    return (a[0] - b[0], a[1] - b[1])


def cplx_neg(a: Cplx) -> Cplx:
    """Negate an exact Gaussian rational."""
    return (-a[0], -a[1])


def cplx_mul(a: Cplx, b: Cplx) -> Cplx:
    """Multiply two exact Gaussian rationals."""
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cplx_recip(z: Cplx) -> Cplx:
    """Exact reciprocal of a nonzero Gaussian rational."""
    den = z[0] * z[0] + z[1] * z[1]
    if den == 0:
        raise ValueError("complex reciprocal of zero")
    return (z[0] / den, -z[1] / den)


def cplx_scale(z: Cplx, s: Fraction) -> Cplx:
    """Scale a Gaussian rational by an exact rational."""
    return (z[0] * s, z[1] * s)


# ---------------------------------------------------------------------------
# exact rational interval arithmetic (the certification backend)
# ---------------------------------------------------------------------------


class RInterval(NamedTuple):
    """Real interval with exact rational endpoints; no rounding occurs anywhere."""

    lo: Fraction
    hi: Fraction


class CInterval(NamedTuple):
    """Complex rectangular interval with exact rational corners."""

    re: RInterval
    im: RInterval


def ri_neg(a: RInterval) -> RInterval:
    """Negate an interval."""
    return RInterval(-a.hi, -a.lo)


def ri_add(a: RInterval, b: RInterval) -> RInterval:
    """Exact sum of two intervals."""
    return RInterval(a.lo + b.lo, a.hi + b.hi)


def ri_sub(a: RInterval, b: RInterval) -> RInterval:
    """Exact difference of two intervals."""
    return RInterval(a.lo - b.hi, a.hi - b.lo)


def ri_mul(a: RInterval, b: RInterval) -> RInterval:
    """Exact product of two intervals (endpoint min/max over the four products)."""
    products = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return RInterval(min(products), max(products))


def ri_scale(a: RInterval, s: Fraction) -> RInterval:
    """Scale an interval by an exact rational, preserving the sign case."""
    return RInterval(a.lo * s, a.hi * s) if s >= 0 else RInterval(a.hi * s, a.lo * s)


def ri_recip(a: RInterval) -> RInterval:
    """Exact reciprocal of an interval that does not contain zero."""
    if not (a.lo > 0 or a.hi < 0):
        raise ValueError("interval reciprocal needs a sign-definite interval")
    return RInterval(ONE / a.hi, ONE / a.lo)


def ri_abs_upper(a: RInterval) -> Fraction:
    """Exact upper bound of |x| over the interval."""
    return max(abs(a.lo), abs(a.hi))


def ri_point(value: Fraction) -> RInterval:
    """Zero-width interval at an exact rational."""
    return RInterval(value, value)


def ci_add(a: CInterval, b: CInterval) -> CInterval:
    """Exact sum of two complex intervals."""
    return CInterval(ri_add(a.re, b.re), ri_add(a.im, b.im))


def ci_sub(a: CInterval, b: CInterval) -> CInterval:
    """Exact difference of two complex intervals."""
    return CInterval(ri_sub(a.re, b.re), ri_sub(a.im, b.im))


def ci_mul(a: CInterval, b: CInterval) -> CInterval:
    """Exact product of two complex rectangular intervals."""
    return CInterval(
        ri_sub(ri_mul(a.re, b.re), ri_mul(a.im, b.im)),
        ri_add(ri_mul(a.re, b.im), ri_mul(a.im, b.re)),
    )


def ci_recip(a: CInterval) -> CInterval:
    """Exact reciprocal of a complex interval whose box avoids the origin."""
    squared = ri_add(ri_mul(a.re, a.re), ri_mul(a.im, a.im))
    if not squared.lo > 0:
        raise ValueError("complex interval contains zero")
    inverse = ri_recip(squared)
    return CInterval(ri_mul(a.re, inverse), ri_mul(ri_neg(a.im), inverse))


def ci_abs_upper(a: CInterval) -> Fraction:
    """Exact upper bound of |z| over the complex box: |z| <= |Re z| + |Im z|."""
    return ri_abs_upper(a.re) + ri_abs_upper(a.im)


# ---------------------------------------------------------------------------
# Laurent polynomials in z1..z6 with Gaussian rational coefficients
# ---------------------------------------------------------------------------


def lp_zero() -> Laurent:
    """The zero Laurent polynomial."""
    return {}


def lp_one() -> Laurent:
    """The constant Laurent polynomial 1."""
    return {(0, 0, 0, 0, 0, 0): (ONE, ZERO)}


def lp_constant(value: Cplx) -> Laurent:
    """A constant Laurent polynomial."""
    return {} if value == (ZERO, ZERO) else {(0, 0, 0, 0, 0, 0): value}


def _merge(out: Laurent, exponents: Exponents, value: Cplx) -> None:
    """Add one coefficient into a Laurent dictionary, dropping exact zeros."""
    total = cplx_add(out.get(exponents, (ZERO, ZERO)), value)
    if total == (ZERO, ZERO):
        out.pop(exponents, None)
    else:
        out[exponents] = total


def lp_add(a: Laurent, b: Laurent) -> Laurent:
    """Sum of two Laurent polynomials."""
    out = dict(a)
    for exponents, value in b.items():
        _merge(out, exponents, value)
    return out


def lp_sub(a: Laurent, b: Laurent) -> Laurent:
    """Difference of two Laurent polynomials."""
    out = dict(a)
    for exponents, value in b.items():
        _merge(out, exponents, cplx_neg(value))
    return out


def lp_scale(a: Laurent, value: Cplx) -> Laurent:
    """Scale a Laurent polynomial by a Gaussian rational."""
    if value == (ZERO, ZERO):
        return {}
    return {exponents: cplx_mul(value, coeff) for exponents, coeff in a.items()}


def lp_mul(a: Laurent, b: Laurent) -> Laurent:
    """Product of two Laurent polynomials (exact coefficient convolution)."""
    out: Laurent = {}
    for exponents_a, coeff_a in a.items():
        for exponents_b, coeff_b in b.items():
            exponents = tuple(x + y for x, y in zip(exponents_a, exponents_b))
            _merge(out, exponents, cplx_mul(coeff_a, coeff_b))
    return out


def lp_derivative(poly: Laurent, joint: int) -> Laurent:
    """Holomorphic derivative of a Laurent polynomial with respect to z_joint."""
    out: Laurent = {}
    for exponents, coeff in poly.items():
        power = exponents[joint]
        if power == 0:
            continue
        next_exponents = list(exponents)
        next_exponents[joint] -= 1
        _merge(out, tuple(next_exponents), cplx_scale(coeff, Fraction(power)))
    return out


def lp_eval(poly: Laurent, zs: tuple[Cplx, ...]) -> Cplx:
    """Exact point evaluation of a Laurent polynomial at exact complex rationals."""
    total = (ZERO, ZERO)
    for exponents, coeff in poly.items():
        value = coeff
        for joint in range(6):
            power = exponents[joint]
            if power > 0:
                for _ in range(power):
                    value = cplx_mul(value, zs[joint])
            elif power < 0:
                for _ in range(-power):
                    value = cplx_mul(value, cplx_recip(zs[joint]))
        total = cplx_add(total, value)
    return total


def lp_eval_float(poly: Laurent, zs: np.ndarray) -> complex:
    """Floating evaluation, used only for box selection and informational checks."""
    total = 0.0j
    for exponents, coeff in poly.items():
        value = complex(float(coeff[0]), float(coeff[1]))
        for joint in range(6):
            power = exponents[joint]
            if power:
                value *= zs[joint] ** power
        total += value
    return total


def lp_eval_interval(
    poly: Laurent,
    box: tuple[CInterval, ...],
    cache: dict[Exponents, CInterval] | None = None,
) -> CInterval:
    """Exact interval evaluation of a Laurent polynomial over a complex box.

    Args:
        poly: Laurent polynomial with exact Gaussian rational coefficients.
        box: Six complex rectangles, one per joint; each must avoid the origin.
        cache: Optional monomial-value cache shared across entries.

    Returns:
        A complex interval that provably contains every value of ``poly`` on the box.
    """
    cache = {} if cache is None else cache
    total = CInterval(ri_point(ZERO), ri_point(ZERO))
    for exponents, coeff in poly.items():
        monomial = cache.get(exponents)
        if monomial is None:
            monomial = CInterval(ri_point(ONE), ri_point(ZERO))
            for joint in range(6):
                power = exponents[joint]
                if power > 0:
                    for _ in range(power):
                        monomial = ci_mul(monomial, box[joint])
                elif power < 0:
                    for _ in range(-power):
                        monomial = ci_mul(monomial, ci_recip(box[joint]))
            cache[exponents] = monomial
        real_coeff, imag_coeff = coeff
        term = CInterval(
            ri_sub(ri_scale(monomial.re, real_coeff), ri_scale(monomial.im, imag_coeff)),
            ri_add(ri_scale(monomial.re, imag_coeff), ri_scale(monomial.im, real_coeff)),
        )
        total = ci_add(total, term)
    return total


# ---------------------------------------------------------------------------
# the frozen exact model and the section 4 square system
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> dict:
    """Load the exp69 frozen-model manifest."""
    return json.loads(path.read_text())


def manifest_links(manifest: dict) -> list[list[list[Fraction]]]:
    """Exact link matrices from the manifest."""
    return [[[Fraction(value) for value in row] for row in matrix]
            for matrix in manifest["links"]]


def config_links() -> list[list[list[Fraction]]]:
    """Exact link matrices re-derived from the robot YAML (short decimals as rationals)."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return [[[Fraction(str(value)) for value in row] for row in matrix]
            for matrix in config["mat"]]


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_model_provenance() -> tuple[dict, list[tuple[str, bool, str]]]:
    """Re-derive the frozen model from its sources and check every recorded hash.

    Returns:
        ``(manifest, checks)`` where every check that fails raises nothing here but is
        reported; callers treat a failed check as a fatal error.
    """
    manifest = read_manifest(MODEL_JSON)
    checks: list[tuple[str, bool, str]] = []
    links = manifest_links(manifest)
    derived = config_links()
    same_links = links == derived
    checks.append(("link matrices: YAML config == manifest",
                   same_links, "exact rational equality"))
    for path in (CONFIG_PATH, SEED_NPZ, Path("study/exp66_gate_a_generators.npz")):
        actual = sha256_of(path)
        recorded = manifest["source_sha256"].get(str(path))
        checks.append((f"source sha256 {path}", recorded == actual,
                       f"recorded {recorded}, actual {actual}"))
    rotation = sp.Matrix([[sp.Rational(value) for value in row]
                          for row in manifest["target_rotation"]])
    orthogonal = rotation.T * rotation == sp.eye(3)
    unit_det = rotation.det() == 1
    checks.append(("R* exactly orthogonal with det 1",
                   bool(orthogonal and unit_det), "exact sympy equality"))
    checks.append(("tracked indices {1,4,6,8,10,13}",
                   manifest["tracked_indices"] == list(TRACKED_INDICES),
                   str(manifest["tracked_indices"])))
    return manifest, checks


def rotation_laurent(joint: int) -> list[list[Laurent]]:
    """The 4x4 rotation ``A_j(z_j)`` with c_j=(z_j+1/z_j)/2, s_j=(z_j-1/z_j)/(2i)."""
    plus = [0] * 6
    plus[joint] = 1
    minus = [0] * 6
    minus[joint] = -1
    plus = tuple(plus)
    minus = tuple(minus)
    cosine = {plus: (HALF, ZERO), minus: (HALF, ZERO)}
    sine = {plus: (ZERO, -HALF), minus: (ZERO, HALF)}
    zero = lp_zero()
    one = lp_one()
    return [
        [cosine, lp_scale(sine, (-ONE, ZERO)), zero, zero],
        [sine, cosine, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, one],
    ]


def constant_matrix(entries: list[list[Fraction]]) -> list[list[Laurent]]:
    """A 4x4 constant matrix of exact rationals as Laurent polynomials."""
    return [[lp_constant((value, ZERO)) for value in row] for row in entries]


def mat4_mul(a: list[list[Laurent]], b: list[list[Laurent]]) -> list[list[Laurent]]:
    """Product of two 4x4 Laurent-polynomial matrices."""
    size = len(a)
    out: list[list[Laurent]] = []
    for row in range(size):
        out_row: list[Laurent] = []
        for column in range(size):
            total = lp_zero()
            for k in range(size):
                total = lp_add(total, lp_mul(a[row][k], b[k][column]))
            out_row.append(total)
        out.append(out_row)
    return out


def build_system(
    links: list[list[list[Fraction]]],
    r_star: list[list[Fraction]],
    p0: list[Fraction],
) -> tuple[list[Laurent], list[list[Laurent]], list[list[Laurent]]]:
    """Build the section 4 square system ``F``, its Jacobian ``J``, and ``S``.

    ``FK = L1 A1 L2 A2 ... L6 A6`` exactly as in exp69; ``p = FK[:3,3]``,
    ``R = FK[:3,:3]``, ``S = R*^T R`` and
    ``F(z) = [p(z)-p0 ; vee((S(z)-S(z)^T)/2)]`` with
    ``vee = ((S21-S12)/2, (S02-S20)/2, (S10-S01)/2)``.

    Args:
        links: Six exact 4x4 link matrices from the frozen manifest.
        r_star: The exact orthogonal target rotation R*.
        p0: The exact target position.

    Returns:
        ``(F, J, S)``: six Laurent equations, the 6x6 holomorphic Jacobian, and the
        3x3 matrix ``S`` used for the false-root exclusion.
    """
    pose = [[lp_zero() for _ in range(4)] for _ in range(4)]
    for index in range(4):
        pose[index][index] = lp_one()
    for k in range(6):
        pose = mat4_mul(mat4_mul(pose, constant_matrix(links[k])), rotation_laurent(k))
    position = [pose[row][3] for row in range(3)]
    rotation = [[pose[row][column] for column in range(3)] for row in range(3)]
    s_matrix: list[list[Laurent]] = [[lp_zero() for _ in range(3)] for _ in range(3)]
    for row in range(3):
        for column in range(3):
            total = lp_zero()
            for k in range(3):
                total = lp_add(total, lp_scale(rotation[k][column],
                                               (r_star[k][row], ZERO)))
            s_matrix[row][column] = total
    half_scale = (HALF, ZERO)
    equations = [lp_sub(position[row], lp_constant((p0[row], ZERO)))
                 for row in range(3)]
    equations += [
        lp_scale(lp_sub(s_matrix[2][1], s_matrix[1][2]), half_scale),
        lp_scale(lp_sub(s_matrix[0][2], s_matrix[2][0]), half_scale),
        lp_scale(lp_sub(s_matrix[1][0], s_matrix[0][1]), half_scale),
    ]
    jacobian = [[lp_derivative(equations[k], joint) for joint in range(6)]
                for k in range(6)]
    return equations, jacobian, s_matrix


def system_fingerprint(equations: list[Laurent], jacobian: list[list[Laurent]],
                       s_matrix: list[list[Laurent]]) -> str:
    """Canonical sha256 of all Laurent coefficients, for coefficient-version tracking."""
    payload = []
    groups = (("F", equations), ("J", [entry for row in jacobian for entry in row]),
              ("S", [entry for row in s_matrix for entry in row]))
    for name, polys in groups:
        for poly in polys:
            for exponents in sorted(poly):
                real_coeff, imag_coeff = poly[exponents]
                payload.append(f"{name} {' '.join(map(str, exponents))} "
                               f"{real_coeff} {imag_coeff}")
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# realification to R^12 and the section 5 contraction constants
# ---------------------------------------------------------------------------


def zs_from_x(x: XVector) -> tuple[Cplx, ...]:
    """Complex z_j = x[2j] + i x[2j+1] from the realified 12-vector."""
    return tuple((x[2 * joint], x[2 * joint + 1]) for joint in range(6))


def x_from_zs(zs: tuple[Cplx, ...]) -> XVector:
    """Realified 12-vector from complex z_j."""
    return tuple(part for z in zs for part in z)


def eval_f(equations: list[Laurent], x: XVector) -> list[Fraction]:
    """Exact evaluation of the realified system f: R^12 -> R^12 at an exact point."""
    zs = zs_from_x(x)
    values: list[Fraction] = []
    for poly in equations:
        value = lp_eval(poly, zs)
        values.append(value[0])
        values.append(value[1])
    return values


def matvec(matrix: list[list[Fraction]], vector: list[Fraction]) -> list[Fraction]:
    """Exact matrix-vector product."""
    return [sum(matrix[row][column] * vector[column]
                for column in range(len(vector))) for row in range(len(matrix))]


def inf_norm(vector: list[Fraction]) -> Fraction:
    """Exact infinity norm of a rational vector."""
    return max((abs(value) for value in vector), default=ZERO)


def fraction_matrix_det(matrix: list[list[Fraction]]) -> Fraction:
    """Exact determinant by fraction-free Bareiss elimination."""
    size = len(matrix)
    if size == 0:
        return ONE
    work = [[value for value in row] for row in matrix]
    sign = 1
    previous = ONE
    for k in range(size - 1):
        pivot_row = next((row for row in range(k, size) if work[row][k] != 0), None)
        if pivot_row is None:
            return ZERO
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
            sign = -sign
        pivot = work[k][k]
        for row in range(k + 1, size):
            for column in range(k + 1, size):
                work[row][column] = (pivot * work[row][column]
                                     - work[row][k] * work[k][column]) / previous
        previous = pivot
    return work[size - 1][size - 1] * sign


def _self_test_det() -> None:
    """Exact cross-check of :func:`fraction_matrix_det` against sympy."""
    rng = np.random.default_rng(0)
    for size in (1, 2, 3, 5, 8):
        for _ in range(3):
            matrix = [[Fraction(int(value)) for value in row]
                      for row in rng.integers(-9, 10, (size, size))]
            expected = sp.Matrix([[sp.Integer(matrix[row][column])
                                   for column in range(size)]
                                  for row in range(size)]).det()
            actual = fraction_matrix_det(matrix)
            assert sp.Rational(actual.numerator, actual.denominator) == expected


def box_intervals(x: XVector, radius: Fraction) -> tuple[CInterval, ...]:
    """The six complex joint rectangles of the box ``x + [-r, r]^12``."""
    return tuple(
        CInterval(RInterval(x[2 * joint] - radius, x[2 * joint] + radius),
                  RInterval(x[2 * joint + 1] - radius, x[2 * joint + 1] + radius))
        for joint in range(6))


def contraction_bounds(
    jacobian: list[list[Laurent]],
    b_matrix: list[list[Fraction]],
    x: XVector,
    radius: Fraction,
) -> tuple[list[Fraction], Fraction, list[Fraction]]:
    """Exact upper bounds ``kappa_i >= sup_X ||row_i(I - B Df(x))||_1``.

    Args:
        jacobian: The 6x6 holomorphic Jacobian of the square system.
        b_matrix: The frozen 12x12 rational preconditioner.
        x: The exact box centre (12 rationals).
        radius: The exact box radius.

    Returns:
        ``(kappa_row, kappa, z_lower)``: per-row upper bounds, their maximum, and the
        per-joint lower bound of ``|z_j|^2`` on the box (each must be positive).
    """
    box = box_intervals(x, radius)
    cache: dict[Exponents, CInterval] = {}
    z_lower = []
    for joint in range(6):
        squared = ri_add(ri_mul(box[joint].re, box[joint].re),
                         ri_mul(box[joint].im, box[joint].im))
        z_lower.append(squared.lo)
    # Interval realified Jacobian: row 2k = Re J_k, row 2k+1 = Im J_k with blocks
    # [Re J, -Im J; Im J, Re J] per (k, j).
    df_interval: list[list[RInterval]] = [
        [ri_point(ZERO) for _ in range(12)] for _ in range(12)]
    for k in range(6):
        for joint in range(6):
            value = lp_eval_interval(jacobian[k][joint], box, cache)
            df_interval[2 * k][2 * joint] = value.re
            df_interval[2 * k][2 * joint + 1] = ri_neg(value.im)
            df_interval[2 * k + 1][2 * joint] = value.im
            df_interval[2 * k + 1][2 * joint + 1] = value.re
    kappa_row: list[Fraction] = []
    for row in range(12):
        row_sum = ZERO
        for column in range(12):
            entry = ri_point(ONE) if row == column else ri_point(ZERO)
            for k in range(12):
                entry = ri_sub(entry, ri_scale(df_interval[k][column], b_matrix[row][k]))
            row_sum += ri_abs_upper(entry)
        kappa_row.append(row_sum)
    return kappa_row, max(kappa_row), z_lower


def exclusion_bounds(
    s_matrix: list[list[Laurent]],
    x: XVector,
    radius: Fraction,
) -> tuple[Fraction, Fraction]:
    """Bounds for the false-root exclusion: |trace S - 3| and ||S - I||_inf on the box."""
    box = box_intervals(x, radius)
    cache: dict[Exponents, CInterval] = {}
    trace_poly = lp_sub(
        lp_add(lp_add(s_matrix[0][0], s_matrix[1][1]), s_matrix[2][2]),
        lp_constant((Fraction(3), ZERO)))
    trace_interval = lp_eval_interval(trace_poly, box, cache)
    trace_bound = ri_abs_upper(trace_interval.re) + ri_abs_upper(trace_interval.im)
    row_bounds = []
    for row in range(3):
        row_sum = ZERO
        for column in range(3):
            delta = ONE if row == column else ZERO
            entry = lp_eval_interval(
                lp_sub(s_matrix[row][column], lp_constant((delta, ZERO))), box, cache)
            row_sum += ci_abs_upper(entry)
        row_bounds.append(row_sum)
    return trace_bound, max(row_bounds)


def float_jacobian_check(equations: list[Laurent], jacobian: list[list[Laurent]],
                         seed_q: np.ndarray) -> tuple[bool, float]:
    """Informational float check of the analytic Jacobian against central differences."""
    zs = np.exp(1j * seed_q)
    worst = 0.0
    for k in range(6):
        for joint in range(6):
            step = 1e-7 * max(1.0, abs(zs[joint]))
            z_plus, z_minus = zs.copy(), zs.copy()
            z_plus[joint] += step
            z_minus[joint] -= step
            numeric = (lp_eval_float(equations[k], z_plus)
                       - lp_eval_float(equations[k], z_minus)) / (2 * step)
            analytic = lp_eval_float(jacobian[k][joint], zs)
            error = abs(numeric - analytic) / max(1.0, abs(analytic))
            worst = max(worst, float(error))
    return worst < 1e-6, worst


def float_fk_crosscheck(links: list[list[list[Fraction]]], seed_q: np.ndarray,
                        equations: list[Laurent], p0: list[Fraction]) -> float:
    """Informational float check of the z-parametrised FK against cos/sin(q) directly.

    ``equations[:3]`` equals ``p(z) - p0``, so the check compares it against the direct
    FK position minus the same p0.
    """
    zs = np.exp(1j * seed_q)
    pose = np.eye(4, dtype=complex)
    for k in range(6):
        cosine, sine = np.cos(seed_q[k]), np.sin(seed_q[k])
        rotation = np.array([[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=complex)
        link = np.array([[complex(value) for value in row] for row in links[k]],
                        dtype=complex)
        pose = pose @ link @ rotation
    direct = pose[:3, 3] - np.array([complex(value) for value in p0])
    via_laurent = np.array([lp_eval_float(poly, zs) for poly in equations[:3]])
    return float(np.max(np.abs(direct - via_laurent)))


# ---------------------------------------------------------------------------
# per-root certification
# ---------------------------------------------------------------------------


def frac_str(value: Fraction) -> str:
    """Exact serialization of a rational as ``numerator/denominator``."""
    return f"{value.numerator}/{value.denominator}"


def box_pass(
    kappa_row: list[Fraction],
    kappa: Fraction,
    z_lower: list[Fraction],
    trace_bound: Fraction,
    radius: Fraction,
    eta: Fraction,
) -> tuple[bool, list[Fraction]]:
    """The strict contraction test at one radius: margins, kappa, trace, and z != 0.

    Returns:
        ``(passed, margins)`` with ``margins[i] = r - eta - kappa_i r``.
    """
    margins = [radius - eta - kappa_row[row] * radius for row in range(12)]
    passed = (kappa < HALF
              and min(margins) >= radius / 2
              and trace_bound < Fraction(4)
              and all(value > 0 for value in z_lower))
    return passed, margins


def max_certified_radius(
    jacobian: list[list[Laurent]],
    s_matrix: list[list[Laurent]],
    b_matrix: list[list[Fraction]],
    centre: XVector,
    eta: Fraction,
) -> dict | None:
    """Largest ladder radius ``2^-k`` at which the strict test still passes."""
    best = None
    for k in MAX_RADIUS_SEARCH_K:
        radius = Fraction(1, 2**k)
        kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre, radius)
        trace_bound, _ = exclusion_bounds(s_matrix, centre, radius)
        passed, margins = box_pass(kappa_row, kappa, z_lower, trace_bound, radius, eta)
        if not passed:
            break
        best = {"k": k, "radius": frac_str(radius), "kappa": frac_str(kappa),
                "margin": frac_str(min(margins)), "trace_bound": frac_str(trace_bound)}
    return best


def certify_root(
    manifest: dict,
    seed_q: np.ndarray,
    root_index: int,
    equations: list[Laurent],
    jacobian: list[list[Laurent]],
    s_matrix: list[list[Laurent]],
    radius_k: int,
    with_max_search: bool,
) -> tuple[dict, bool]:
    """Certify one root: exact centre, frozen B, eta/kappa, strict margins.

    Args:
        manifest: The frozen-model manifest.
        seed_q: The joint-angle seed (complex float, six entries) from the npz table.
        root_index: The archive root index (one of {1, 4, 6, 8, 10, 13}).
        equations: The six Laurent equations of the section 4 system.
        jacobian: Their 6x6 holomorphic Jacobian.
        s_matrix: ``S = R*^T R`` for the false-root exclusion.
        radius_k: Preferred certification radius ``2^-k`` (a larger radius is retried
            honestly if the strict test fails; a failure is never loosened silently).
        with_max_search: Whether to also run the informational max-radius ladder.

    Returns:
        ``(record, certified)`` where ``record`` is the per-root certificate dict.
    """
    z_seed = np.exp(1j * seed_q)
    centre0 = tuple(part for z in z_seed
                    for part in (Fraction.from_float(z.real), Fraction.from_float(z.imag)))
    jacobian_float = np.zeros((6, 6), dtype=complex)
    for k in range(6):
        for joint in range(6):
            jacobian_float[k, joint] = lp_eval_float(jacobian[k][joint], z_seed)
    realified = np.zeros((12, 12))
    for k in range(6):
        for joint in range(6):
            real_part, imag_part = jacobian_float[k, joint].real, jacobian_float[k, joint].imag
            realified[2 * k, 2 * joint] = real_part
            realified[2 * k, 2 * joint + 1] = -imag_part
            realified[2 * k + 1, 2 * joint] = imag_part
            realified[2 * k + 1, 2 * joint + 1] = real_part
    b_matrix = [[Fraction.from_float(np.linalg.inv(realified)[row][column])
                 for column in range(12)] for row in range(12)]
    det_b = fraction_matrix_det(b_matrix)
    _, kappa_at_centre0, _ = contraction_bounds(jacobian, b_matrix, centre0, ZERO)

    centre = centre0
    eta = inf_norm(matvec(b_matrix, eval_f(equations, centre)))

    final_record = None
    for k in range(radius_k, 14, -5):
        radius = Fraction(1, 2**k)
        kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre, radius)
        trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
        passed, margins = box_pass(kappa_row, kappa, z_lower, trace_bound, radius, eta)
        if passed:
            margin = min(margins)
            final_record = {
                "radius_k": k,
                "radius": frac_str(radius),
                "kappa_row": [frac_str(value) for value in kappa_row],
                "kappa": frac_str(kappa),
                "margin_row": [frac_str(value) for value in margins],
                "margin": frac_str(margin),
                "margin_relative": frac_str(margin / radius),
                "trace_bound": frac_str(trace_bound),
                "trace_margin": frac_str(Fraction(4) - trace_bound),
                "s_minus_i_inf_bound": frac_str(s_bound),
                "z_modulus_squared_lower": [frac_str(value) for value in z_lower],
            }
            break
    certified = final_record is not None
    record = {
        "seed_index": root_index,
        "seed_q": [frac_str(value) for value in
                   (Fraction.from_float(part)
                    for z in seed_q for part in (z.real, z.imag))],
        "centre0": [frac_str(value) for value in centre0],
        "centre": [frac_str(value) for value in centre],
        "centre_note": ("centre = centre0 = exp(i q_seed) in IEEE double, each component "
                        "frozen as an exact binary rational; the exponential is used only "
                        "to select the box and is not assumed certified (spec section 4)"),
        "B": [[frac_str(value) for value in row] for row in b_matrix],
        "B_det": frac_str(det_b),
        "B_inverse_residual_inf": frac_str(kappa_at_centre0),
        "eta": frac_str(eta),
    }
    if final_record is not None:
        record.update(final_record)
        record["max_certified_radius"] = (max_certified_radius(
            jacobian, s_matrix, b_matrix, centre, eta) if with_max_search else None)
    else:
        record["radius_k"] = None
        record["kappa_row"] = None
        record["kappa"] = None
        record["margin_row"] = None
        record["margin"] = None
        record["trace_bound"] = None
        record["trace_margin"] = None
        record["s_minus_i_inf_bound"] = None
        record["z_modulus_squared_lower"] = None
        record["max_certified_radius"] = None
    record["verdict"] = "certified" if certified else "inconclusive"
    return record, certified



# ---------------------------------------------------------------------------
# batch disjointness, JSON assembly, verification, CLI
# ---------------------------------------------------------------------------


def boxes_disjoint(
    centre_a: XVector,
    radius_a: Fraction,
    centre_b: XVector,
    radius_b: Fraction,
) -> tuple[bool, int, Fraction]:
    """Whether two boxes in R^12 are disjoint, via their most separating coordinate.

    Returns:
        ``(disjoint, coordinate, gap)`` where ``gap`` is the coordinate separation
        minus the two radii, evaluated at the coordinate that maximizes it.
    """
    best_coord, best_gap = -1, None
    for coord in range(12):
        gap = abs(centre_a[coord] - centre_b[coord]) - (radius_a + radius_b)
        if best_gap is None or gap > best_gap:
            best_coord, best_gap = coord, gap
    return best_gap > 0, best_coord, best_gap


def parse_xvector(values: list[str]) -> XVector:
    """Parse a stored 12-rational vector."""
    return tuple(Fraction(value) for value in values)


def print_root_summary(record: dict) -> None:
    """One-line human-readable summary of a root certificate."""
    index = record["seed_index"]
    if record["verdict"] == "certified":
        radius = float(Fraction(record["radius"]))
        eta = float(Fraction(record["eta"]))
        kappa = float(Fraction(record["kappa"]))
        margin = float(Fraction(record["margin"]))
        relative = 100.0 * float(Fraction(record["margin_relative"]))
        trace = float(Fraction(record["trace_bound"]))
        z_low = min(float(Fraction(value)) for value in record["z_modulus_squared_lower"])
        print(f"  root {index:2d}: certified | r = 2^-{record['radius_k']} = {radius:.2e}"
              f" | eta {eta:.2e} | kappa {kappa:.2e} | margin {margin:.2e}"
              f" ({relative:.1f}% of r) | trace bound {trace:.2e} | min|z_j|^2 {z_low:.4f}")
    else:
        print(f"  root {index:2d}: INCONCLUSIVE -- honest numbers in the JSON, "
              f"certificate not loosened")


def pairwise_disjoint_entries(records: dict[str, dict]) -> tuple[bool, list[dict]]:
    """Exact pairwise disjointness of the certified boxes."""
    certified = [(int(index), parse_xvector(record["centre"]),
                  Fraction(record["radius"]))
                 for index, record in records.items()
                 if record["verdict"] == "certified"]
    entries = []
    all_disjoint = True
    for (index_a, centre_a, radius_a), (index_b, centre_b, radius_b) in itertools.combinations(
            certified, 2):
        ok_pair, coord, gap = boxes_disjoint(centre_a, radius_a, centre_b, radius_b)
        all_disjoint = all_disjoint and ok_pair
        entries.append({"pair": [index_a, index_b], "coordinate": coord,
                        "gap": frac_str(gap), "disjoint": ok_pair})
        print(f"  boxes {index_a},{index_b}: {'disjoint' if ok_pair else 'OVERLAP'}"
              f" (coordinate {coord}, gap {frac_str(gap)})")
    return all_disjoint, entries


def build_payload(manifest: dict, records: dict[str, dict],
                  all_disjoint: bool, pair_entries: list[dict]) -> dict:
    """Assemble the certificate JSON payload (exact rational strings only)."""
    certified = [index for index, record in records.items()
                 if record["verdict"] == "certified"]
    inconclusive = [index for index, record in records.items()
                    if record["verdict"] == "inconclusive"]
    return {
        "experiment": "exp70_gate_c1_roots",
        "task": ("Gate C, task C1: certify the six common-base roots (lambda = 0) of the "
                 "exact square system F(z) = [p(z)-p0; vee((S-S^T)/2)] on the exp69 "
                 "frozen model; box, B, eta, kappa, strict margins, false-root "
                 "exclusion, z_j != 0, pairwise disjointness"),
        "spec": "study/collab/GPT_GATE_C_SPEC_20261017.md sections 4, 5, 6 and task C1 of 7",
        "boundary": ("C1 only: six root boxes at the base point; no segment or path "
                     "certification, no permutation claims, and no certification flags "
                     "beyond the per-root boxes"),
        "backend": {
            "name": "exact rational interval arithmetic over Python fractions.Fraction "
                    "endpoints (complex rectangular intervals)",
            "rounding": "none; interval endpoints are exact rationals, products take the "
                        "min/max of the four endpoint products, reciprocals use exact "
                        "rational inversion, every comparison is exact",
            "precision": "exact (unbounded rational arithmetic); no floating-point value "
                         "enters any certificate constant",
            "floating_inputs": [
                "box centres exp(i q_seed) computed in IEEE double only to select the "
                "box, then frozen as exact binary rationals (spec section 4)",
                "B = numpy double inverse of the float Jacobian at centre0, frozen as "
                "exact binary rationals (spec section 5)",
            ],
        },
        "system": {
            "product_order": "FK = L1 A1 L2 A2 ... L6 A6, A_j(z_j) the z-rotation with "
                             "c_j=(z_j+1/z_j)/2, s_j=(z_j-1/z_j)/(2i)",
            "equations": ("F1..F3 = p(z) - p0; F4..F6 = vee((S-S^T)/2) with S = R*^T R(z) "
                          "and vee = ((S21-S12)/2, (S02-S20)/2, (S10-S01)/2); lambda = 0"),
            "realification": "x = (Re z1, Im z1, ..., Re z6, Im z6) in R^12; "
                             "f = (Re F1, Im F1, ..., Re F6, Im F6)",
            "jacobian_blocks": "Df[2k,2j]=Re J_kj, Df[2k,2j+1]=-Im J_kj, "
                               "Df[2k+1,2j]=Im J_kj, Df[2k+1,2j+1]=Re J_kj "
                               "(F holomorphic in z)",
            "false_root_exclusion": ("|trace S - 3| < 4 on the whole box plus z_j != 0; "
                                     "S^T S = I and det S = 1 identically, so a skew-free "
                                     "S with trace 3 is S = I"),
            "contraction": ("T(x) = x - B f(x); eta = ||B f(c)||_inf exactly; "
                            "kappa_i >= sup_{x in X} ||row_i(I - B Df(x))||_1 by interval "
                            "evaluation of the analytic Jacobian; strict margin "
                            "r - eta - kappa_i r > 0 and kappa < 1 give a strict "
                            "contraction of X into its interior and hence exactly one "
                            "regular root in X"),
            "radius_policy": ("uniform certification radius r = 2^-40 with the comfort "
                              "test kappa < 1/2 and margin >= r/2; if it fails, larger "
                              "radii are retried honestly and the failure is recorded, "
                              "never loosened"),
        },
        "coefficient_version": {
            "model_manifest": str(MODEL_JSON),
            "model_manifest_sha256": sha256_of(MODEL_JSON),
            "source_sha256": manifest["source_sha256"],
            "system_fingerprint": system_fingerprint(*build_system(
                manifest_links(manifest),
                [[Fraction(value) for value in row] for row in manifest["target_rotation"]],
                [Fraction(value) for value in manifest["target_position"]])),
        },
        "seed_source": {
            "archive": str(SEED_NPZ),
            "table": "fibre_real / fibre_imag (frozen polished base root table)",
            "indices": list(TRACKED_INDICES),
            "note": ("seed_q stored per root as 12 exact binary rationals, interleaved "
                     "(re q1, im q1, ..., re q6, im q6); centre0 = exp(i q_seed) frozen"),
        },
        "batch": {
            "n_certified": len(certified),
            "n_inconclusive": len(inconclusive),
            "verdicts": {index: record["verdict"] for index, record in records.items()},
            "pairwise_disjoint": {"passed": all_disjoint, "pairs": pair_entries},
            "summary": f"{len(certified)}/{len(records)}",
        },
        "roots": records,
        "verification_command": ("python3 -m study.exp70_gate_c1_roots --verify "
                                 "study/exp70_gate_c1_certificate.json"),
    }


def verify_certificate(path: Path) -> int:
    """Recompute every stored certificate number from the frozen inputs, exactly.

    The verifier trusts nothing inside the JSON: it re-derives the model from the YAML
    config, re-builds the system, re-checks the source hashes, re-runs the interval
    evaluations from the stored centres and B, compares every stored rational exactly,
    and re-checks every strict inequality.
    """
    stored = json.loads(path.read_text())
    all_ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = all_ok and passed
        short = detail if len(detail) <= 60 else detail[:30] + "..." + detail[-30:]
        suffix = f"  ({short})" if detail else ""
        print(f"    {'PASS' if passed else 'FAIL'}  {name}{suffix}")

    print(f"verify: {path}")
    manifest, provenance = check_model_provenance()
    for name, passed, detail in provenance:
        report(name, passed, detail)
    if not all(entry[1] for entry in provenance):
        print("FAIL: model provenance failed -- certificate not re-checkable")
        return 1

    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    fingerprint = system_fingerprint(equations, jacobian, s_matrix)
    report("system coefficient fingerprint matches stored",
           fingerprint == stored["coefficient_version"]["system_fingerprint"],
           f"{fingerprint[:16]}...")
    report("model manifest sha256 matches stored",
           sha256_of(MODEL_JSON) == stored["coefficient_version"]["model_manifest_sha256"])

    with np.load(SEED_NPZ, allow_pickle=False) as data:
        fibre = data["fibre_real"] + 1j * data["fibre_imag"]

    certified_pairs = []
    for key, record in stored["roots"].items():
        root_index = int(key)
        print(f"  root {root_index}:")
        seed_q = fibre[root_index]
        stored_seed = parse_xvector(record["seed_q"])
        derived_seed = tuple(Fraction.from_float(part)
                             for z in seed_q for part in (z.real, z.imag))
        report("  stored seed_q matches the npz table exactly", stored_seed == derived_seed)
        centre0 = parse_xvector(record["centre0"])
        z_seed = np.exp(1j * seed_q)
        centre0_float = np.array([float(value) for value in centre0])
        seed_float = np.array([part for z in z_seed for part in (z.real, z.imag)])
        provenance_close = float(np.max(np.abs(centre0_float - seed_float))) <= 1e-12
        report("  centre0 within 1e-12 of exp(i q_seed) (provenance)", provenance_close)
        b_matrix = [[Fraction(value) for value in row] for row in record["B"]]
        det_b = fraction_matrix_det(b_matrix)
        report("  det B != 0 and matches stored",
               det_b != 0 and frac_str(det_b) == record["B_det"], frac_str(det_b))
        _, kappa0, _ = contraction_bounds(jacobian, b_matrix, centre0, ZERO)
        report("  ||I - B Df(centre0)||_inf < 1e-9 and matches stored",
               kappa0 < B_INVERSE_TOLERANCE
               and frac_str(kappa0) == record["B_inverse_residual_inf"], frac_str(kappa0))
        centre = centre0
        report("  stored centre equals stored centre0",
               centre == parse_xvector(record["centre"]))
        eta = inf_norm(matvec(b_matrix, eval_f(equations, centre)))
        report("  eta matches stored", frac_str(eta) == record["eta"], frac_str(eta))
        if record["verdict"] == "certified":
            radius_k = int(record["radius_k"])
            radius = Fraction(1, 2**radius_k)
            report("  radius = 2^-k matches stored", frac_str(radius) == record["radius"])
            kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre, radius)
            trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
            margins = [radius - eta - kappa_row[row] * radius for row in range(12)]
            report("  kappa_row matches stored",
                   [frac_str(value) for value in kappa_row] == record["kappa_row"])
            report("  kappa matches stored",
                   frac_str(kappa) == record["kappa"], frac_str(kappa))
            report("  margin_row matches stored",
                   [frac_str(value) for value in margins] == record["margin_row"])
            report("  margin matches stored",
                   frac_str(min(margins)) == record["margin"], frac_str(min(margins)))
            report("  trace bound matches stored and |trace S - 3| < 4",
                   frac_str(trace_bound) == record["trace_bound"]
                   and trace_bound < Fraction(4), frac_str(trace_bound))
            report("  ||S - I||_inf bound matches stored",
                   frac_str(s_bound) == record["s_minus_i_inf_bound"], frac_str(s_bound))
            report("  z_modulus_squared_lower matches stored and > 0",
                   [frac_str(value) for value in z_lower] == record["z_modulus_squared_lower"]
                   and all(value > 0 for value in z_lower))
            report("  strict contraction re-checked",
                   kappa < ONE and min(margins) > 0,
                   f"kappa {frac_str(kappa)} < 1, margin {frac_str(min(margins))} > 0")
            certified_pairs.append((root_index, centre, radius))
        else:
            report("  recorded verdict is inconclusive (certificate not loosened)", True)

    for (index_a, centre_a, radius_a), (index_b, centre_b, radius_b) in itertools.combinations(
            certified_pairs, 2):
        ok_pair, coord, gap = boxes_disjoint(centre_a, radius_a, centre_b, radius_b)
        report(f"  boxes {index_a},{index_b} disjoint",
               ok_pair, f"coordinate {coord}, gap {frac_str(gap)}")

    print("PASS: certificate fully re-checked" if all_ok else "FAIL: verification failed")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    """Certify the six common-base roots (or one root), or verify a stored certificate."""
    parser = argparse.ArgumentParser(
        description="exp70: Gate C1 exact root certificates (no path certification)")
    parser.add_argument("--root", type=int, default=None,
                        help="certify one root index from {1,4,6,8,10,13}; default: all six")
    parser.add_argument("--out", type=Path, default=CERT_JSON, help="certificate JSON path")
    parser.add_argument("--radius-k", type=int, default=DEFAULT_RADIUS_K,
                        help="preferred box radius 2^-k")
    parser.add_argument("--verify", type=Path, default=None,
                        help="re-check an existing certificate JSON exactly")
    parser.add_argument("--no-max-search", action="store_true",
                        help="skip the informational max-certified-radius ladder")
    args = parser.parse_args(argv)
    if args.verify is not None:
        return verify_certificate(args.verify)
    if args.root is not None and args.root not in TRACKED_INDICES:
        raise ValueError(f"--root must be one of {list(TRACKED_INDICES)}, got {args.root}")

    _self_test_det()
    print("exp70: Gate C1 -- exact contraction certificates for the six common-base roots\n")
    manifest, provenance = check_model_provenance()
    print("phase 0: model provenance")
    for name, passed, detail in provenance:
        print(f"    {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed -- refusing to certify")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    fingerprint = system_fingerprint(equations, jacobian, s_matrix)
    print(f"    system coefficient fingerprint {fingerprint[:16]}... "
          f"(F {len(equations)} equations, J {len(jacobian)}x{len(jacobian[0])}, "
          f"max monomials per entry "
          f"{max(len(entry) for row in jacobian for entry in row)})")

    indices = [args.root] if args.root is not None else list(TRACKED_INDICES)
    with np.load(SEED_NPZ, allow_pickle=False) as data:
        fibre = data["fibre_real"] + 1j * data["fibre_imag"]
    print(f"\nphase 1: certify roots {indices} (backend: exact rational intervals)\n")
    records: dict[str, dict] = {}
    for root_index in indices:
        record, certified = certify_root(manifest, fibre[root_index], root_index,
                                         equations, jacobian, s_matrix, args.radius_k,
                                         not args.no_max_search)
        records[str(root_index)] = record
        jac_ok, jac_worst = float_jacobian_check(equations, jacobian, fibre[root_index])
        fk_diff = float_fk_crosscheck(links, fibre[root_index], equations, p0)
        record["float_crosschecks"] = {
            "jacobian_vs_central_difference_ok": jac_ok,
            "jacobian_vs_central_difference_max_rel_err": jac_worst,
            "fk_vs_cos_sin_max_abs_diff": fk_diff,
        }
        print_root_summary(record)
        print(f"    float cross-checks (informational only): analytic J vs central "
              f"differences rel err {jac_worst:.2e} ({jac_ok}), z-FK vs cos/sin "
              f"max diff {fk_diff:.2e}")
        print()
    all_disjoint, pair_entries = pairwise_disjoint_entries(records)
    payload = build_payload(manifest, records, all_disjoint, pair_entries)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    certified_count = sum(1 for record in records.values()
                          if record["verdict"] == "certified")
    print(f"batch: {certified_count}/{len(records)} certified, boxes pairwise disjoint: "
          f"{all_disjoint}")
    print(f"certificate written to {args.out}")
    print(f"re-check: python3 -m study.exp70_gate_c1_roots --verify {args.out}")
    return 0 if certified_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
