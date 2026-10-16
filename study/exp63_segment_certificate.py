#!/usr/bin/env python3
"""exp63: turn "no singular configuration on this segment" from a sampled claim into a theorem.

`study/chamber.py` decides chamber membership from a path that is verified *by sampling*: a step
is admissible while ``sigma_min`` stays above ``delta`` at the sampled points, at a density the
caller picks.  A sampled claim is only as good as its samples -- a segment can cross
``det J = 0`` strictly between two samples and still look clear at both.  Section B of
`study/collab/GPT_MATH_ADVANCE_20261016.md` removes that gap with a **global Lipschitz bound for
the Jacobian derivative**, built from the fixed link lengths and the joint displacement alone:

    ||dJbar/dt||_2 <= L,   sigma_min(Jbar(q(t))) >= min_k sigma_min(Jbar(q(t_k))) - L*h,

for ``q(t) = q0 + t*d``, ``t in [0,1]``, sampled with covering radius ``h`` in the *parameter*
``t`` (equispaced ``N`` samples including both ends: ``h = 1/(2(N-1))``).  This module implements
that bound (B1-B3), the floating-point accounting B4 asks for, and an adaptive subdivision that
answers with one of

* ``certified``    -- the bound is above the target (``0``, or ``delta``) on *every* interval;
* ``inconclusive`` -- the sample budget ran out.  Nothing is asserted: B4 is explicit that a
                      failed certificate must **not** be read as "the path crosses Sigma";
* ``invalid``      -- the lift leaves the joint limits, so it is not a path of this arm.

Unit scaling (the caveat in B3)
-------------------------------
The geometric Jacobian is not commensurable: its first three rows are length per radian, its last
three are radian per radian, so ``sigma_min(J)`` depends on the length unit.  The certificate is
therefore computed for the *scaled* matrix ``Jbar = diag(alpha*I3, beta*I3) @ J`` with

* ``alpha = 1`` per length unit of the model (metres here, the library's ``dh_unit``),
* ``beta = 1`` (dimensionless),

which makes both blocks dimensionless.  Invariance: ``(alpha, beta) -> (t*alpha, t*beta)`` leaves
the reported bound unchanged, and changing the length unit by ``lambda`` together with
``alpha -> alpha/lambda`` leaves the ``Jbar``-level certificate *identical* (``alpha*b_j`` and
``beta*a_j`` are invariant).  ``certified_lower`` is reported for the unscaled ``J``:
``sigma_min(J) >= sigma_min(Jbar)/max(alpha, beta)``, which is exact when ``alpha == beta``.

What is, and is not, machine-verified
-------------------------------------
The analytic part -- the bound ``L``, the covering argument, the subdivision -- is a proof.  The
numerical part is not.  B4 warns that a plain float SVD cannot serve as a rigorous sample lower
bound, so every sample gets

* a rigorous lower bound from an approximate inverse with an inflated residual:
  ``e = ||I - B J||_2 < 1``, ``M = ||B||_2`` gives ``sigma_min(J) >= (1-e)/M`` (B4), and
* an explicit floating-point allowance for the library's own FK/Jacobian evaluation.

Both are reported, and every certificate carries ``machine_verified = False``: the honest label is
**analytic bound + floating-point verification**.  Phase two of T5 (a guaranteed interval backend)
would replace the sample bounds; it is not implemented here.

Run::

    python3 -m study.exp63_segment_certificate            # full run, exit != 0 on a failed check
    python3 -m study.exp63_segment_certificate --quick    # smoke test, smaller budgets
    python3 -m study.exp63_segment_certificate --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

from model import ModelFactory
from study import cuspidal3r as c3
from study import ik_structure as iks
from study.chamber import clearance_path

#: Scale of the translation rows, in 1/length units: ``1`` means "one metre" for this library.
DEFAULT_ALPHA = 1.0
#: Scale of the rotation rows; dimensionless.
DEFAULT_BETA = 1.0
#: Unit round-off of float64, the currency of the numerical allowance and of B4's residual bound.
EPS = float(np.finfo(float).eps)

#: Row classification of a 6-row geometric Jacobian: three linear, three angular rows.
FULL_ROWS: tuple[str, ...] = ("v", "v", "v", "w", "w", "w")
#: Row classification of a position-only task Jacobian (the 3R of `study/cuspidal3r.py`).
POSITION_ROWS: tuple[str, ...] = ("v", "v", "v")


# --------------------------------------------------------------------------- #
# floating-point tools (B4)
# --------------------------------------------------------------------------- #


def _gamma(count: int) -> float:
    """Higham's ``gamma_n = n*u/(1 - n*u)``: the relative cost of ``n`` rounded operations."""
    scaled = count * EPS
    return scaled / (1.0 - scaled) if scaled < 1.0 else 1.0


def _frobenius_upper(matrix: np.ndarray) -> float:
    """An upper bound for the Frobenius (hence the spectral) norm, inflated for its own rounding.

    Args:
        matrix: Any real matrix.

    Returns:
        ``||matrix||_F`` plus a relative and an absolute inflation, so the returned value is an
        upper bound of the exact Frobenius norm of the *given* float entries.
    """
    values = np.abs(np.asarray(matrix, dtype=float))
    if values.size == 0:
        return 0.0
    plain = float(np.sqrt(np.sum(values * values)))
    inflation = 8.0 * EPS * values.size
    return float(plain * (1.0 + inflation) + inflation * float(np.max(values)))


def sigma_min_sample_bounds(jacobian: np.ndarray) -> tuple[float, float]:
    """Return ``(s_float, s_lower)`` for the matrix whose float entries are ``jacobian``.

    ``s_float`` is what :func:`numpy.linalg.svd` reports.  ``s_lower`` is a lower bound for
    ``sigma_min`` of the **exact** matrix with those float entries, computed by *two* independent
    rigorous routes whose maximum is taken:

    * B4's route: with an approximate inverse ``B``, ``e = ||I - B J||_2 < 1`` and
      ``M = ||B||_2`` imply ``sigma_min(J) >= (1 - e)/M``.  The square case inverts the matrix, the
      rectangular case the Gram matrix of the smaller side (plus a term for the rounding of that
      Gram product).  This route is conservative because ``M`` is bounded by norms that overestimate
      the spectral norm (measured: up to 25% for the SR5, where the inverse has several large
      singular values).
    * the SVD factors themselves: with ``J ~ U diag(s) V^T`` and *verified* residuals
      ``delta_R >= ||U diag(s) V^T - J||_2``, ``delta_U >= ||U^T U - I||_2``,
      ``delta_V >= ||V^T V - I||_2`` (all three computed in float with inflated bounds), one has
      ``sigma_min(J) >= s_min sqrt((1 - delta_U)(1 - delta_V)) - delta_R``.  The tall form of ``J``
      is used, because ``sigma_min`` of a wide matrix equals that of its transpose and only the
      tall form has an orthonormal right factor.  This route is tight (deficit ~1e-15 relative) and
      does not rely on the backward stability of the SVD algorithm, only on the verified residuals.

    Both are lower bounds of the same quantity, so their maximum is one too.  A singular or
    hopelessly ill-conditioned matrix returns ``s_lower = 0.0``.

    Args:
        jacobian: The matrix, shape ``(rows, num_dof)``.

    Returns:
        ``(s_float, s_lower)``, both non-negative, with ``s_lower <= s_float`` up to round-off.
    """
    matrix = np.asarray(jacobian, dtype=float)
    rows, columns = matrix.shape
    tall = matrix if rows >= columns else matrix.T
    left, singular, right = np.linalg.svd(tall, full_matrices=False)
    s_float = float(singular[-1])
    best = 0.0
    try:
        best = max(best, _inverse_sigma_min_lower(matrix))
    except np.linalg.LinAlgError:
        pass
    svd_bound = _svd_sigma_min_lower(tall, left, singular, right)
    if svd_bound is not None:
        best = max(best, svd_bound)
    return s_float, float(best)


def _inverse_sigma_min_lower(matrix: np.ndarray) -> float:
    """B4's ``(1 - e)/M`` lower bound for ``sigma_min`` of ``matrix`` (0.0 when unusable)."""
    rows, columns = matrix.shape
    if rows == columns:
        inverse = np.linalg.inv(matrix)
        residual = _inverse_residual_bound(inverse, matrix)
        if residual is None:
            return 0.0
        e, magnitude = residual
        return (1.0 - e) / magnitude if e < 1.0 and magnitude > 0.0 else 0.0
    if rows < columns:
        gram = matrix @ matrix.T
        absolute = np.abs(matrix) @ np.abs(matrix).T
    else:
        gram = matrix.T @ matrix
        absolute = np.abs(matrix).T @ np.abs(matrix)
    inverse = np.linalg.inv(gram)
    residual = _inverse_residual_bound(inverse, gram)
    if residual is None:
        return 0.0
    e, magnitude = residual
    # the Gram matrix is itself a float product: its exact value differs by at most this much.
    # Each entry is an inner product of the *longer* side of ``matrix`` (``AA^T`` for a wide ``A``
    # sums ``columns`` terms, ``A^T A`` for a tall ``A`` sums ``rows`` terms), so the rounding
    # bound uses ``max(rows, columns)``, not ``min(rows, columns)``.
    e = e + magnitude * _gamma(max(rows, columns)) * _spectral_upper(absolute)
    if e >= 1.0 or magnitude <= 0.0:
        return 0.0
    return float(np.sqrt(max(0.0, (1.0 - e) / magnitude)))


def _svd_sigma_min_lower(
    tall: np.ndarray, left: np.ndarray, singular: np.ndarray, right: np.ndarray
) -> float | None:
    """Rigorous ``sigma_min`` lower bound from the SVD factors and their verified residuals.

    Args:
        tall: The matrix in tall form (rows >= columns), whose ``sigma_min`` is the one wanted.
        left: Its left singular vectors, shape ``(rows, columns)``.
        singular: Its singular values, descending.
        right: Its right singular vectors, ``V^T``, shape ``(columns, columns)``.

    Returns:
        ``s_min sqrt((1 - delta_U)(1 - delta_V)) - delta_R``, or ``None`` when a residual cannot
        be bounded below one.
    """
    inner = singular[:, None] * right
    approximation = left @ inner
    rounding = 4.0 * _gamma(singular.size) * (np.abs(left) @ np.abs(inner))
    delta_r = _spectral_upper(np.abs(approximation - tall) + rounding)
    identity = np.eye(singular.size)
    # ``U^T U`` is a Gram product of the *tall* side: each entry is an inner product of two length
    # ``left.shape[0]`` columns of ``U``, so its rounding is bounded by gamma of that length (not
    # ``singular.size``, which equals ``left.shape[1]`` and is smaller for a tall ``U``).
    delta_u = _spectral_upper(
        np.abs(left.T @ left - identity)
        + _gamma(left.shape[0]) * (np.abs(left).T @ np.abs(left))
    )
    delta_v = _spectral_upper(
        np.abs(right @ right.T - identity) + _gamma(singular.size) * (np.abs(right) @ np.abs(right).T)
    )
    if delta_u >= 1.0 or delta_v >= 1.0:
        return None
    # ``sqrt((1 - delta_U)(1 - delta_V))`` is the genuine lower bound of
    # ``sigma_min(U) * sigma_min(V)``; ``1 - (delta_U + delta_V)/2`` is an *upper* bound of that
    # square root (arithmetic-geometric mean) and would overstate the certificate.
    return float(
        max(0.0, singular[-1] * np.sqrt((1.0 - delta_u) * (1.0 - delta_v)) - delta_r)
    )


def _inverse_residual_bound(inverse: np.ndarray, matrix: np.ndarray) -> tuple[float, float] | None:
    """Return inflated ``(e, M)`` with ``e >= ||I - inverse @ matrix||_2``, ``M >= ||inverse||_2``."""
    size = matrix.shape[0]
    if inverse.shape != (size, size) or not np.all(np.isfinite(inverse)):
        return None
    product = inverse @ matrix
    elementwise = np.abs(inverse) @ np.abs(matrix)
    residual = np.abs(np.eye(size) - product) + _gamma(size) * elementwise
    return _spectral_upper(residual), _spectral_upper(inverse)


def _spectral_upper(matrix: np.ndarray) -> float:
    """An upper bound for the spectral norm: the smaller of the Frobenius and 1-infinity bounds.

    Both ``||A||_F`` and ``sqrt(||A||_1 * ||A||_inf)`` bound ``||A||_2`` from above; the second is
    much tighter for the inverse of a well-scaled Jacobian, where the Frobenius norm grows with the
    *number* of singular values, and the first is tighter for small matrices.  The minimum of two
    upper bounds is still an upper bound (B4 prescribes Frobenius "or an even more conservative
    element-wise sum"; this is the other direction, a strictly better upper bound).

    Args:
        matrix: Any real matrix.

    Returns:
        An upper bound of ``||matrix||_2``, inflated for the rounding of the sums.
    """
    values = np.abs(np.asarray(matrix, dtype=float))
    if values.size == 0:
        return 0.0
    rows, columns = values.shape
    inflation = 1.0 + _gamma(max(rows, columns)) + 4.0 * EPS
    one = float(np.max(np.sum(values, axis=0))) * inflation + 4.0 * EPS
    infinity = float(np.max(np.sum(values, axis=1))) * inflation + 4.0 * EPS
    return min(_frobenius_upper(values), float(np.sqrt(one * infinity)))


# --------------------------------------------------------------------------- #
# the analytic bound, B1-B3
# --------------------------------------------------------------------------- #


def row_kinds(jacobian_rows: int, rows: str | tuple[str, ...] | list[str] = "auto") -> tuple[str, ...]:
    """Classify each row of a task Jacobian as translation (``"v"``) or rotation (``"w"``).

    Args:
        jacobian_rows: Number of rows of the Jacobian that will be certified.
        rows: ``"auto"`` for the two shapes this study uses (``6`` -> three linear plus three
            angular rows, ``3`` -> three linear rows), or an explicit sequence of ``"v"``/``"w"``.

    Returns:
        The row kinds, length ``jacobian_rows``.

    Raises:
        ValueError: If the kinds cannot be inferred, or do not match ``jacobian_rows``.
    """
    if isinstance(rows, str):
        if rows != "auto":
            raise ValueError(f"rows must be 'auto' or a sequence of 'v'/'w', got {rows!r}")
        if jacobian_rows == 6:
            return FULL_ROWS
        if jacobian_rows == 3:
            return POSITION_ROWS
        raise ValueError(
            f"cannot infer the row kinds of a {jacobian_rows}-row Jacobian; pass rows=('v', ...)"
        )
    kinds = tuple(rows)
    if len(kinds) != jacobian_rows:
        raise ValueError(f"rows has {len(kinds)} entries but the Jacobian has {jacobian_rows}")
    if any(kind not in ("v", "w") for kind in kinds):
        raise ValueError(f"rows entries must be 'v' or 'w', got {kinds!r}")
    return kinds


def chain_lengths(model) -> np.ndarray:
    """The reach bounds ``ell_j >= ||p - o_j||`` of B1, from the fixed link geometry alone.

    ``Ms = [base, link_1, ..., link_n, tool]``; joint ``j`` (1-based) turns about the ``z`` axis
    of ``Ms[0] @ Ms[1] @ Rz(q_1) @ ... @ Ms[j] @ Rz(q_j)``, so the tool point relative to that
    joint's origin is reached through the fixed transforms ``Ms[j+1], ..., Ms[n+1]``.  Fixed
    rotations do not change lengths, so the bound is the sum of the translation norms -- the
    translation of ``Ms[j]`` itself is *not* counted, it is already part of the joint's origin.

    Args:
        model: Robot model exposing ``Ms`` and ``num_dof``.

    Returns:
        ``ell``, shape ``(num_dof,)``, in the model's length unit.

    Raises:
        ValueError: If ``Ms`` does not have the ``num_dof + 2`` entries the layout promises.
    """
    matrices = [np.asarray(matrix, dtype=float) for matrix in model.Ms]
    joint_count = int(model.num_dof)
    if len(matrices) != joint_count + 2:
        raise ValueError(
            f"expected Ms = [base, link_1..link_{joint_count}, tool] "
            f"({joint_count + 2} matrices), got {len(matrices)}"
        )
    lengths = np.zeros(joint_count)
    tail = 0.0
    for index in range(joint_count - 1, -1, -1):
        tail += float(np.linalg.norm(matrices[index + 2][:3, 3]))
        lengths[index] = tail
    return lengths


@dataclass
class LipschitzBound:
    """The constants of B1-B3 for one joint-space displacement ``d``."""

    displacement: np.ndarray
    #: ``ell_j``: an upper bound of the distance from joint ``j``'s axis to the tool point.
    lengths: np.ndarray
    #: ``a_j = sum_{k<j} |d_k|``: the upstream rotation budget of joint ``j``.
    upstream: np.ndarray
    #: ``b_j = a_j*ell_j + sum_{k>=j} |d_k|*ell_k``: the bound on ``||dJv_j/dt||_2``.
    derivative: np.ndarray
    row_kinds: tuple[str, ...]
    alpha: float
    beta: float
    #: ``L = sqrt(sum_j alpha^2 b_j^2 + beta^2 a_j^2)``: bounds ``||dJbar/dt||_2``.
    l_frobenius: float
    #: ``L1 = sum_j (alpha b_j + beta a_j)``: the conservative no-square-root alternative of B3.
    l_one: float

    @property
    def lipschitz(self) -> float:
        """The bound in use: the Frobenius bound, valid for the spectral norm as well."""
        return self.l_frobenius


def lipschitz_bound(
    model,
    d: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    row_kinds: tuple[str, ...] = FULL_ROWS,
) -> LipschitzBound:
    """The global bound ``L`` of B1-B3 for ``||dJbar/dt||_2`` on the segment ``q0 + t*d``.

    With ``a_j = sum_{k<j} |d_k|``, ``b_j = a_j*ell_j + sum_{k>=j} |d_k|*ell_k`` and
    ``Jbar = diag(alpha*I3, beta*I3) @ J``,

        ``||dJv_j/dt||_2 <= b_j``, ``||dJw_j/dt||_2 <= a_j``,
        ``L = sqrt(sum_j alpha^2 b_j^2 + beta^2 a_j^2) >= ||dJbar/dt||_F >= ||dJbar/dt||_2``.

    The angular term is dropped when the task has no angular rows (B3: "for a position-only 3R,
    delete the angular velocity term"); the translation term is dropped if there are no linear
    rows.  Both ``a_j`` and ``b_j`` are linear in ``|d|``, so a sub-segment of parameter length
    ``s`` has bound ``s*L`` -- which is what makes adaptive subdivision pay.

    Args:
        model: Robot model (``Ms``, ``num_dof``).
        d: Joint-space displacement, radians, length ``num_dof``.
        alpha: Scale of the translation rows, in 1/length units.
        beta: Scale of the rotation rows, dimensionless.
        row_kinds: Row classification, see :func:`row_kinds`.

    Returns:
        The :class:`LipschitzBound`.

    Raises:
        ValueError: If ``d`` has the wrong length, or is not finite.
    """
    displacement = np.asarray(d, dtype=float)
    lengths = chain_lengths(model)
    if displacement.shape != lengths.shape:
        raise ValueError(
            f"d must have num_dof = {lengths.size} entries, got shape {displacement.shape}"
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("d must be finite")
    steps = np.abs(displacement)
    upstream = np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    suffix = np.zeros_like(lengths)
    total = 0.0
    for index in range(lengths.size - 1, -1, -1):
        total += steps[index] * lengths[index]
        suffix[index] = total
    derivative = upstream * lengths + suffix
    linear = any(kind == "v" for kind in row_kinds)
    angular = any(kind == "w" for kind in row_kinds)
    squared = 0.0
    if linear:
        squared += float(np.sum((alpha * derivative) ** 2))
    if angular:
        squared += float(np.sum((beta * upstream) ** 2))
    l_one = 0.0
    if linear:
        l_one += float(np.sum(alpha * derivative))
    if angular:
        l_one += float(np.sum(beta * upstream))
    return LipschitzBound(
        displacement=displacement,
        lengths=lengths,
        upstream=upstream,
        derivative=derivative,
        row_kinds=tuple(row_kinds),
        alpha=float(alpha),
        beta=float(beta),
        l_frobenius=float(np.sqrt(squared)),
        l_one=l_one,
    )


def numerical_allowance(
    model,
    *,
    row_kinds: tuple[str, ...] = FULL_ROWS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """A generous floating-point allowance for the *scaled* Jacobian of one configuration.

    B4 replaces the float SVD at a sample with a rigorous lower bound, but the *entries* of the
    Jacobian are themselves rounded: the library evaluates a chain of ``num_dof + 2`` transforms
    and a cross-product formula.  With Higham's ``gamma_k``, every 4x4 product adds an error term
    of order ``gamma_4`` times the magnitude of the accumulated translation, bounded by
    ``1 + sum_k ||translation(Ms[k])||_2`` -- a *length*, not the product of the transform norms
    (the product form is a relative bound and is hopelessly pessimistic for a chain).  The
    rotation rows carry no length, so their error is that of a unit vector.  Each block is scaled
    by its own ``alpha``/``beta``, which is what makes the allowance invariant under a change of
    length unit.  The rotations contribute a libm term of their own and the factor ``8`` covers
    the cross products and the index bookkeeping: a *generous estimate* with explicitly named
    constants, not a machine-verified interval bound.

    Args:
        model: Robot model (``Ms``, ``num_dof``).
        row_kinds: Row classification of the task being certified.
        alpha: Scale of the translation rows, in 1/length units.
        beta: Scale of the rotation rows, dimensionless.

    Returns:
        An upper bound on ``||Jbar_exact - Jbar_float||_2``, i.e. on the change of ``sigma_min``
        caused by evaluating the model in floating point.
    """
    matrices = [np.asarray(matrix, dtype=float) for matrix in model.Ms]
    span = 1.0 + float(sum(np.linalg.norm(matrix[:3, 3]) for matrix in matrices))
    count = 4 * (len(matrices) - 1 + int(model.num_dof))
    chain = 8.0 * (_gamma(count) + 8.0 * EPS * int(model.num_dof))
    total = 0.0
    if any(kind == "v" for kind in row_kinds):
        total += alpha * span * chain
    if any(kind == "w" for kind in row_kinds):
        total += beta * 1.0 * chain
    return float(total)


# --------------------------------------------------------------------------- #
# the certificate interface
# --------------------------------------------------------------------------- #


@dataclass
class SegmentCertificate:
    """What is known about ``sigma_min`` on one straight segment after certification.

    ``certified_lower`` is always a valid lower bound on ``sigma_min(J)`` over the *whole*
    segment (``0.0`` when nothing better than the trivial bound was proven); whether it is worth
    anything is what ``status`` says.  ``bracket`` is ``[lower, sampled]`` in the units of
    ``sigma_min(J)``: the true minimum lies between them, up to the reported allowances.
    """

    #: ``"certified"``, ``"inconclusive"`` or ``"invalid"``.
    status: str
    #: Certified lower bound on ``sigma_min(J)`` along the segment, ``>= 0`` (``0`` if unproven).
    certified_lower: float
    #: ``certified_lower > 0`` (and the status is ``certified``).
    positive: bool
    #: ``certified_lower > delta`` when a ``delta`` was requested, else ``None``.
    delta_ok: bool | None
    #: ``(lower, sampled)`` bracket on the true minimum, in ``sigma_min(J)`` units.
    bracket: tuple[float, float]
    #: Smallest ``sigma_min`` measured at any sample (the "sampled" number of the old instrument).
    sampled_min: float
    #: The bound ``L`` of B3 for the whole segment, in ``Jbar`` units.
    lipschitz: float
    #: Covering radius (in ``t``) of the interval that attains the bound.
    h: float
    #: Joint-space length of the whole segment, radians.
    span: float
    #: Number of Jacobian evaluations used.
    samples: int
    #: ``L*h``: the analytic part of the margin, in ``Jbar`` units.
    margin_lipschitz: float
    #: Floating-point allowance for the model evaluation, in ``Jbar`` units.
    margin_numeric: float
    #: Largest loss of the rigorous sample bound against the float SVD, over the samples.
    sample_deficit: float
    #: Whether the sampled ``det J`` changes sign (a sampled crossing witness; ``None`` if not square).
    det_sign_change: bool | None
    #: Whether the lift left the joint limits (``invalid`` when true).
    out_of_bounds: bool
    #: The bound before the unit conversion, i.e. on ``sigma_min(Jbar)``.
    lower_jbar: float
    #: Seconds spent in this call.
    seconds: float
    #: Always ``False``: the analytic part is a theorem, the numeric part is float-verified only.
    machine_verified: bool = False
    #: Why the subdivision stopped: ``converged``, ``collapsed``, ``budget``, ``uniform`` or
    #: ``zero``.
    stop: str = ""
    #: Human-readable remark (why it is inconclusive, what was assumed).
    note: str = ""


def _limits_of(source, model) -> tuple[np.ndarray, np.ndarray]:
    """Joint bounds of ``source`` when it has them, else of ``model``."""
    lower = getattr(source, "lower_bounds", None)
    upper = getattr(source, "upper_bounds", None)
    if lower is None or upper is None:
        lower, upper = model.lower_bounds, model.upper_bounds
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def certify_segment(
    model,
    q0: np.ndarray,
    d: np.ndarray,
    *,
    task=None,
    delta: float | None = None,
    target: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    rows: str | tuple[str, ...] = "auto",
    rel_tol: float = 0.05,
    abs_tol: float = 1e-9,
    max_samples: int = 4096,
    uniform: int | None = None,
    check_bounds: bool = True,
    margin_extra: float = 0.0,
) -> SegmentCertificate:
    """Certify a lower bound on ``sigma_min`` along the whole segment ``q(t) = q0 + t*d``.

    The segment is subdivided adaptively: an interval ``[a, b]`` with samples at both ends is
    certified by ``min(s_a, s_b) - L*(b-a)/2`` (the covering radius of two samples is half the
    interval), and the interval that currently attains the worst bound is bisected until either
    the claim ``sigma_min > target`` holds on every interval, or the bracket on the minimum is
    tighter than ``rel_tol``, or the sample budget runs out.

    Args:
        model: Robot model; its ``Ms`` supplies the link lengths of the bound and its limits the
            path check when ``task`` has none.
        q0: Start configuration, radians, length ``num_dof``.
        d: Displacement, radians; ``t = 1`` is ``q0 + d``.
        task: Optional task view exposing ``jacobian(q)`` for the *same* joints (for example
            :class:`study.cuspidal3r.PositionTask`).  ``model.jacobian`` is used when omitted.
        delta: Optional clearance to certify as well; reported through ``delta_ok``.
        target: Value the certified bound must exceed for the status to be ``certified``.
        alpha: Scale of the translation rows, in 1/length units (see the module docstring).
        beta: Scale of the rotation rows, dimensionless.
        rows: ``"auto"`` or an explicit ``"v"``/``"w"`` sequence per Jacobian row.
        rel_tol: Refine while the bracket on the minimum is wider than this fraction of the
            sampled minimum.  This controls the accuracy of the reported *number*, not the status.
        abs_tol: Floor of the relative bracket tolerance.
        max_samples: Sample budget; exhausting it gives ``inconclusive`` (never "crossing").
        uniform: When given, use exactly this many equispaced samples and no adaptive refinement;
            the answer is then a deterministic function of the samples, which is what the
            unit-scaling invariance in this module's self-test compares.
        check_bounds: Check the joint limits along the lift; leaving them gives ``invalid``.
        margin_extra: Extra subtraction from the bound, in ``Jbar`` units.

    Returns:
        The :class:`SegmentCertificate`.

    Raises:
        ValueError: If ``q0``/``d`` are not ``num_dof`` long, or the Jacobian's shape does not
            match ``rows``.
    """
    started = time.perf_counter()
    source = model if task is None else task
    start = np.asarray(q0, dtype=float)
    displacement = np.asarray(d, dtype=float)
    joint_count = int(model.num_dof)
    if start.shape != (joint_count,) or displacement.shape != (joint_count,):
        raise ValueError(f"q0 and d must have num_dof = {joint_count} entries")
    # the row classification has to be known before the bound is built (a position-only task has
    # no angular rows, and B3 drops the corresponding term), so probe the Jacobian once.
    probe = np.asarray(source.jacobian(start), dtype=float)
    if probe.shape[1] != joint_count:
        raise ValueError(f"jacobian has {probe.shape[1]} columns, expected {joint_count}")
    kinds = row_kinds(probe.shape[0], rows)
    bound = lipschitz_bound(model, displacement, alpha=alpha, beta=beta, row_kinds=kinds)
    allowance = numerical_allowance(model, row_kinds=kinds, alpha=alpha, beta=beta) + float(
        margin_extra
    )
    scale = np.array([alpha if kind == "v" else beta for kind in kinds])
    lower_limit, upper_limit = _limits_of(source, model)

    knots: dict[float, tuple[float, float]] = {}
    deficits: list[float] = []
    state = {"out_of_bounds": False, "first_sign": 0, "sign_change": False}

    def value(t: float) -> tuple[float, float]:
        """``(s_float, s_lower)`` at parameter ``t``; evaluates and checks once."""
        if t not in knots:
            q = start + t * displacement
            jacobian = np.asarray(source.jacobian(q), dtype=float)
            if jacobian.shape[1] != joint_count:
                raise ValueError(f"jacobian has {jacobian.shape[1]} columns, expected {joint_count}")
            sample_kinds = row_kinds(jacobian.shape[0], rows)
            if sample_kinds != bound.row_kinds:
                raise ValueError(f"rows {sample_kinds} disagree with the bound's {bound.row_kinds}")
            if check_bounds and not bool(
                np.all(q >= lower_limit - 1e-9) and np.all(q <= upper_limit + 1e-9)
            ):
                state["out_of_bounds"] = True
            scaled = jacobian * scale[:, None]
            s_float, s_lower = sigma_min_sample_bounds(scaled)
            knots[t] = (s_float, s_lower)
            deficits.append(max(0.0, s_float - s_lower))
            if scaled.shape[0] == scaled.shape[1]:
                determinant = float(np.linalg.det(scaled))
                if determinant != 0.0:
                    sign = int(np.sign(determinant))
                    if state["first_sign"] == 0:
                        state["first_sign"] = sign
                    elif sign != state["first_sign"]:
                        state["sign_change"] = True
        return knots[t]

    if bound.lipschitz <= 0.0:
        s_float, s_lower = value(0.0)
        lower = s_lower - allowance
        convert = 1.0 / max(alpha, beta)
        return SegmentCertificate(
            status="certified" if lower > target else "inconclusive",
            certified_lower=max(0.0, lower * convert),
            positive=bool(lower > target and lower > 0.0),
            delta_ok=None if delta is None else bool(lower > delta),
            bracket=(lower * convert, s_float * convert),
            sampled_min=s_float * convert,
            lipschitz=0.0, h=0.0, span=0.0, samples=1,
            margin_lipschitz=0.0, margin_numeric=allowance * convert,
            sample_deficit=max(deficits), det_sign_change=state["sign_change"],
            out_of_bounds=state["out_of_bounds"], lower_jbar=lower, seconds=0.0,
            stop="zero", note="zero displacement: the segment is a point",
        )

    def interval_bound(left: float, right: float) -> float:
        """``min(s_left, s_right) - L*(right-left)/2 - allowance``, valid on the whole interval."""
        s_left = value(left)[1]
        s_right = value(right)[1]
        return min(s_left, s_right) - bound.lipschitz * (right - left) * 0.5 - allowance

    value(0.0)
    value(1.0)
    # an interval is [left, right, bound, frozen]; ``frozen`` marks one that can no longer be
    # split because the bisection reached float resolution (which happens exactly at a crossing,
    # where the bound keeps the minimum pinned down at ~0 and the subdivision would otherwise
    # spin forever adding zero-width intervals).
    intervals: list[list] = [[0.0, 1.0, interval_bound(0.0, 1.0), False]]
    upper = min(sample[0] for sample in knots.values())
    budget = max(2, int(max_samples))
    split_count = 0
    stop_reason = "budget"
    if uniform is not None:
        # deterministic mode: a fixed equispaced grid, no adaptive refinement.  The answer is then
        # a pure function of the sample parameters and the float Jacobian entries, which is what
        # the unit-scaling invariance test in this module compares.
        count = max(2, int(uniform))
        grid = [index / (count - 1) for index in range(count)]
        for t in grid:
            value(t)
        intervals = [
            [left, right, interval_bound(left, right), False]
            for left, right in zip(grid, grid[1:])
        ]
        upper = min(sample[0] for sample in knots.values())
    while uniform is None and len(knots) < budget:
        if state["out_of_bounds"]:
            break
        live = [interval for interval in intervals if not interval[3]]
        if not live:
            stop_reason = "collapsed"
            break
        lower = min(interval[2] for interval in intervals)
        # the bracket cannot be tightened below the loss of the rigorous sample bound itself
        # (``s_float - s_lower``), so the tolerance has to carry that deficit; otherwise the
        # subdivision chases a gap it can never close.
        tolerance = max(abs_tol, rel_tol * upper) + max(deficits)
        if lower > max(target, 0.0) and (upper - lower) <= tolerance:
            stop_reason = "converged"
            break
        worst = min(live, key=lambda interval: interval[2])
        frozen = [interval[2] for interval in intervals if interval[3]]
        if frozen and worst[2] >= min(frozen):
            # only the unrefinable intervals still hold the global bound down
            stop_reason = "collapsed"
            break
        left, right, _, _ = worst
        middle = 0.5 * (left + right)
        if middle == left or middle == right or middle in knots:
            worst[3] = True
            stop_reason = "collapsed"
            continue
        value(middle)
        worst[0], worst[1], worst[2] = left, middle, interval_bound(left, middle)
        intervals.append([middle, right, interval_bound(middle, right), False])
        upper = min(upper, knots[middle][0])
        split_count += 1
    if uniform is not None:
        stop_reason = "uniform"
    lower_jbar = min(interval[2] for interval in intervals)
    convert = 1.0 / max(alpha, beta)
    worst_interval = min(intervals, key=lambda interval: interval[2])
    h = 0.5 * (worst_interval[1] - worst_interval[0])
    out_of_bounds = bool(state["out_of_bounds"])
    if out_of_bounds:
        status = "invalid"
    elif lower_jbar > max(target, 0.0):
        status = "certified"
    else:
        status = "inconclusive"
    note = ""
    if out_of_bounds:
        note = "the lift leaves the joint limits, so it is not a path of this arm"
    elif uniform is not None:
        note = f"uniform {int(uniform)}-sample grid, no adaptive refinement"
    elif status == "inconclusive" and stop_reason == "collapsed":
        note = (
            "the binding interval collapsed to float resolution (a sampled crossing or a "
            "tangency) and its bound stays at or below the target; nothing is asserted, because "
            "a failed certificate is not a crossing proof (B4)"
        )
    elif status == "inconclusive":
        note = (
            f"budget of {budget} samples spent after {split_count} bisections; nothing asserted "
            "(a failed certificate is not a crossing proof, B4)"
        )
    return SegmentCertificate(
        status=status,
        certified_lower=max(0.0, lower_jbar * convert),
        positive=bool(status == "certified" and lower_jbar > 0.0),
        delta_ok=None if delta is None else bool(status == "certified" and lower_jbar > delta),
        bracket=(lower_jbar * convert, upper * convert),
        sampled_min=upper * convert,
        lipschitz=bound.lipschitz,
        h=h,
        span=float(np.linalg.norm(displacement)),
        samples=len(knots),
        margin_lipschitz=bound.lipschitz * h * convert,
        margin_numeric=allowance * convert,
        sample_deficit=max(deficits) * convert if deficits else 0.0,
        det_sign_change=state["sign_change"],
        out_of_bounds=out_of_bounds,
        lower_jbar=lower_jbar,
        seconds=time.perf_counter() - started,
        stop=stop_reason,
        note=note,
    )


@dataclass
class PathCertificate:
    """The certificate of a whole polyline: every straight sub-segment certified in turn."""

    #: Every sub-segment certified with ``sigma_min > 0``.
    certified: bool
    #: Every sub-segment certified with ``sigma_min > delta`` (``None`` without a ``delta``).
    delta_ok: bool | None
    #: ``min`` over the sub-segments of the certified lower bound (``0`` when not all certified).
    certified_lower: float
    #: ``min`` over the samples of the float ``sigma_min`` (the old sampled number).
    sampled_min: float
    segments: int
    skipped: int
    certified_segments: int
    #: Total Jacobian evaluations.
    samples: int
    max_l: float
    h_worst: float
    margin_numeric: float
    #: Largest joint-space step between consecutive path points (a wrap artifact shows up here).
    max_step: float
    #: Indices of the sub-segments that are not certified.
    failures: list[int] = field(default_factory=list)
    #: Sub-segment with the smallest certified bound.
    worst_segment: int = -1
    det_sign_changes: int = 0
    seconds: float = 0.0
    statuses: dict[str, int] = field(default_factory=dict)


def certify_path(
    model,
    path,
    *,
    task=None,
    delta: float | None = None,
    rows: str | tuple[str, ...] = "auto",
    **kwargs,
) -> PathCertificate:
    """Certify every straight sub-segment of a recorded path.

    The path is the polyline the instrument actually walked, so the certificate transfers: if
    every sub-segment keeps ``sigma_min > 0``, the polyline lies in ``Q \\ Sigma`` and its
    endpoints are in the same chamber/spect.  Consecutive points are joined by their *literal*
    difference ``q[k+1] - q[k]`` (the lift the instrument took); a jump of several radians is
    reported through ``max_step`` rather than silently re-wrapped.

    Args:
        model: Robot model.
        path: Sequence of configurations, shape ``(points, num_dof)``.
        task: Optional task view for the Jacobians (see :func:`certify_segment`).
        delta: Optional clearance to certify as well.
        rows: Row classification, forwarded to :func:`certify_segment`.
        **kwargs: Forwarded to :func:`certify_segment`.

    Returns:
        The :class:`PathCertificate`.

    Raises:
        ValueError: If fewer than two points are given.
    """
    points = [np.asarray(point, dtype=float) for point in path]
    if len(points) < 2:
        raise ValueError(f"a path needs at least two points, got {len(points)}")
    started = time.perf_counter()
    certificates: list[SegmentCertificate] = []
    indices: list[int] = []
    skipped = 0
    for index, (left, right) in enumerate(zip(points, points[1:])):
        step = right - left
        if float(np.linalg.norm(step)) == 0.0:
            skipped += 1
            continue
        certificates.append(
            certify_segment(model, left, step, task=task, delta=delta, rows=rows, **kwargs)
        )
        indices.append(index)
    if not certificates:
        raise ValueError("every sub-segment of the path has zero length")
    certified_segments = [c for c in certificates if c.status == "certified"]
    failures = [index for index, c in zip(indices, certificates) if c.status != "certified"]
    complete = len(certified_segments) == len(certificates)
    worst = min(range(len(certificates)), key=lambda k: certificates[k].certified_lower)
    statuses: dict[str, int] = {}
    for certificate in certificates:
        statuses[certificate.status] = statuses.get(certificate.status, 0) + 1
    return PathCertificate(
        certified=complete,
        delta_ok=None if delta is None else bool(complete and all(c.delta_ok for c in certificates)),
        certified_lower=(
            min(c.certified_lower for c in certificates) if complete else 0.0
        ),
        sampled_min=min(c.sampled_min for c in certificates),
        segments=len(certificates),
        skipped=skipped,
        certified_segments=len(certified_segments),
        samples=sum(c.samples for c in certificates),
        max_l=max(c.lipschitz for c in certificates),
        h_worst=certificates[worst].h,
        margin_numeric=max(c.margin_numeric for c in certificates),
        max_step=max(float(np.linalg.norm(right - left)) for left, right in zip(points, points[1:])),
        failures=failures,
        worst_segment=indices[worst],
        det_sign_changes=sum(1 for c in certificates if c.det_sign_change == 1),
        seconds=time.perf_counter() - started,
        statuses=statuses,
    )


# --------------------------------------------------------------------------- #
# witness-path replays
# --------------------------------------------------------------------------- #


def exp49_witness_path(
    *, pose_seed: int = 0, seeds: int = 400, least: int = 8, tries: int = 14,
    delta: float = 5e-3, steps: int = 1500,
):
    """Rebuild the witness path of `study/exp49_sr5_uniqueness.py`, phases 1-2, verbatim.

    The call sequence is the recorded one (``rich_pose`` then the first same-sign pair the
    clearance walk connects, in the same ``itertools.combinations`` order), so the returned path
    is the path the recorded conclusions rest on.

    Returns:
        ``(model, solutions, signs, pair, walk)`` or ``None`` when the pose search fails.
    """
    from study.exp49_sr5_uniqueness import rich_pose

    model = ModelFactory.create("sr5")
    rng = np.random.default_rng(pose_seed)
    target, solutions = rich_pose(model, rng, seeds=seeds, least=least, tries=tries)
    if not solutions:
        return None
    signs = [int(np.sign(iks.det_jacobian(model, q))) for q in solutions]
    groups = {
        1: [i for i, sign in enumerate(signs) if sign > 0],
        -1: [i for i, sign in enumerate(signs) if sign < 0],
    }
    circular = iks.circular_joints(model)
    for i, j in itertools.combinations(groups[1] + groups[-1], 2):
        if signs[i] != signs[j]:
            continue
        walk = clearance_path(
            model, solutions[i], solutions[j], delta=delta, circular=circular, steps=steps
        )
        if walk.connected:
            return model, solutions, signs, (i, j), walk
    return None


def exp48_witness_route(*, grid: int = 300, fine_grid: int = 25, samples: int = 200,
                        delta: float = 5e-3):
    """Rebuild the same-aspect route of `study/exp48_cuspidal3r.py`, phase 4, verbatim.

    The route depends on the random state left by the two census sweeps of phases 2 and 3, so
    both are replayed with their recorded budgets before ``same_aspect_pair`` is called.

    Returns:
        ``(model, labels, pair, route, refined)`` or ``None`` when the replay finds no route;
        ``refined`` says whether the clearance legs succeeded (the raw grid route is returned when
        they did not, as in exp48).
    """
    from study.chamber import clearance_path as walk_segment
    from study.exp48_cuspidal3r import same_aspect_pair

    model = c3.build()
    rng = np.random.default_rng(0)
    c3.rank_report(model, samples=200)
    labels, clearance, _ = c3.aspect_labels(model, grid=grid)
    for _ in range(40):  # phase 2's aspect histogram: 40 censuses of 200 seeds
        target = c3.position(model, rng.uniform(-2.0, 2.0, model.num_dof))
        c3.census_position(model, target, seeds=samples, rng=rng)
    for x in np.linspace(-3.2, 3.2, fine_grid):  # phase 3's workspace grid: 25 seeds a cell
        for z in np.linspace(-2.4, 2.4, fine_grid):
            c3.census_position(model, np.array([abs(x), 0.0, z]), seeds=25, rng=rng)
    found = same_aspect_pair(model, labels, rng)
    if not found:
        return None
    _, _, _, pair = found
    route = None
    for node_delta in (0.03, 0.02, 0.01, 5e-3, 2e-3):
        route = c3.aspect_path(
            model, labels, pair[0], pair[1], clearance=clearance, delta=node_delta
        )
        if route is not None:
            break
    if route is None:
        return None
    adapter = c3.PositionTask(model)
    waypoints = route[:: max(1, len(route) // 40)] + [route[-1]]
    refined: list[np.ndarray] = [route[0]]
    refined_ok = True
    for left, right in zip(waypoints, waypoints[1:]):
        leg = walk_segment(adapter, left, right, delta=delta, steps=800)
        if not leg.connected:
            refined = list(route)
            refined_ok = False
            break
        refined.extend(leg.path[1:])
    return model, labels, pair, (refined if len(refined) > 1 else route), refined_ok


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

CHECKS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one pass/fail line."""
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _unit(vector: np.ndarray) -> np.ndarray:
    """The unit vector in the direction of ``vector``."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _inside(q: np.ndarray, model) -> bool:
    """Whether ``q`` is inside the model's joint limits."""
    return bool(
        np.all(q >= np.asarray(model.lower_bounds) - 1e-9)
        and np.all(q <= np.asarray(model.upper_bounds) + 1e-9)
    )


def find_det_crossing(det_of, q_start: np.ndarray, direction: np.ndarray, model,
                      *, span: float = 1.2, samples: int = 160,
                      bisections: int = 80) -> tuple[np.ndarray, float] | None:
    """Bisect for a sign change of ``det_of`` along a ray: a sampled witness of ``det = 0``.

    Args:
        det_of: Callable ``q -> float``, the (signed) determinant.
        q_start: Start configuration, radians.
        direction: Search direction, joint space; normalised internally.
        model: Robot model, used for the joint limits.
        span: Length of the ray to scan, radians.
        samples: Points scanned before bisection.
        bisections: Bisection steps.

    Returns:
        ``(q_cross, |det J(q_cross)|)`` or ``None`` if no sign change is found inside the limits.
    """
    step = _unit(np.asarray(direction, dtype=float))
    values = []
    for index in range(samples + 1):
        q = q_start + (span * index / samples) * step
        if not _inside(q, model):
            break
        values.append((q, det_of(q)))
    if len(values) < 2:
        return None
    left = None
    for (q_a, d_a), (q_b, d_b) in zip(values, values[1:]):
        if d_a == 0.0:
            return q_a, 0.0
        if d_a * d_b < 0.0:
            left = (q_a, d_a, q_b, d_b)
            break
    if left is None:
        return None
    q_a, d_a, q_b, _ = left
    for _ in range(bisections):
        middle = 0.5 * (q_a + q_b)
        d_m = det_of(middle)
        if d_m == 0.0:
            return middle, 0.0
        if d_a * d_m < 0.0:
            q_b = middle
        else:
            q_a, d_a = middle, d_m
    q_cross = 0.5 * (q_a + q_b)
    return q_cross, abs(det_of(q_cross))


def crossing_segment(model, q_cross: np.ndarray, direction: np.ndarray, *, ratio: float = 0.55,
                     span: float = 0.2, coarse: int = 9, det_of=None, task=None,
                     floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float] | None:
    """A segment through a crossing, with the crossing strictly between the coarse samples.

    The crossing sits at ``t = ratio`` (``0.55``), which is *not* one of the ``coarse`` equispaced
    sample parameters, so the sampled minimum alone reports a healthy clearance while the segment
    really does cross ``Sigma``.  ``span`` is grown until the coarse minimum clears ``floor``.

    Args:
        model: Robot model.
        q_cross: A configuration (numerically) on ``Sigma``.
        direction: Direction of the segment, joint space.
        ratio: Parameter of the crossing inside the segment, in ``(0, 1)``.
        span: Starting segment length, radians.
        coarse: Number of equispaced samples used for the "sampled minimum" reading.
        det_of: Callable ``q -> det J``, for the sign check (optional).
        task: Optional task view for the Jacobian.
        floor: Smallest acceptable coarse-sample minimum.

    Returns:
        ``(q0, d, coarse_min)`` or ``None`` when no adequate span is found inside the limits.
    """
    step = _unit(np.asarray(direction, dtype=float))
    source = model if task is None else task
    for factor in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        length = span * factor
        q0 = q_cross - ratio * length * step
        d = length * step
        if not (_inside(q0, model) and _inside(q0 + d, model)):
            continue
        if det_of is not None and det_of(q0) * det_of(q0 + d) >= 0.0:
            continue
        sigmas = [
            float(np.linalg.svd(np.asarray(source.jacobian(q0 + t * d), dtype=float),
                                compute_uv=False)[-1])
            for t in np.linspace(0.0, 1.0, coarse)
        ]
        if min(sigmas) >= floor:
            return q0, d, min(sigmas)
    return None


def _dense_min(source, q0: np.ndarray, d: np.ndarray, samples: int) -> float:
    """Densely sampled ``sigma_min`` along a segment: the "ground truth" of the demonstrations."""
    return min(
        float(np.linalg.svd(np.asarray(source.jacobian(q0 + t * d), dtype=float),
                            compute_uv=False)[-1])
        for t in np.linspace(0.0, 1.0, samples)
    )


def _scaled_3r(scale: float):
    """A copy of the 3R with every length multiplied by ``scale`` (a change of length unit)."""
    rows = tuple((d * scale, alpha, a * scale) for d, alpha, a in c3.DEFAULT_ROWS)
    tool = tuple((d * scale, alpha, a * scale) for d, alpha, a in c3.DEFAULT_TOOL)
    return c3.build(rows=rows, tool=tool)


# --------------------------------------------------------------------------- #
# the experiment
# --------------------------------------------------------------------------- #


def phase_svd_regressions() -> dict:
    """Phase 0: regression checks for the SVD lower bound (T5b, GPT_NEXT_GATE_20261017.md §2).

    ``_svd_sigma_min_lower`` used ``1 - (delta_U + delta_V)/2``, which is an *upper* bound of
    ``sqrt((1 - delta_U)(1 - delta_V))`` (arithmetic-geometric mean), so a nearly orthogonal factor
    could be certified above its true ``sigma_min``.  These checks pin the corrected formula and
    the corrected inner-product lengths of the tall/wide (non-square) paths.
    """
    print("\nphase 0: SVD lower-bound regressions (T5b)")
    record: dict = {}

    # issue 1: the formula.  GPT's exact 1x1 counterexample.
    a = np.array([[np.sqrt(0.9)]])
    bound = _svd_sigma_min_lower(a, a, np.array([1.0]), np.array([[1.0]]))
    true = float(a[0, 0])
    _check(
        "1x1 counterexample is a valid lower bound (old code reported 0.95 > 0.94868)",
        bound is not None and bound <= true + 1e-15 and abs(bound - true) < 1e-12,
        f"bound {bound:.15e} vs true {true:.15e}",
    )

    # issue 1 in a tall factorisation: a (5x2) U with sigma_min(U) = sqrt(0.9).
    tall_u = np.zeros((5, 2))
    tall_u[0, 0] = np.sqrt(0.9)
    tall_u[1, 1] = 1.0
    singular2 = np.ones(2)
    identity2 = np.eye(2)
    tall_bound = _svd_sigma_min_lower(tall_u, tall_u, singular2, identity2)
    _check(
        "tall (5x2) SVD-factor counterexample is a valid lower bound (old code: 0.95)",
        tall_bound is not None and tall_bound <= true + 1e-15 and abs(tall_bound - true) < 1e-12,
        f"bound {tall_bound:.15e} vs true {true:.15e}",
    )

    # the full interface stays a valid lower bound on both non-square shapes.
    rng = np.random.default_rng(3)
    for name, matrix in (("tall 7x3", rng.normal(size=(7, 3))),
                         ("wide 3x7", rng.normal(size=(3, 7)))):
        s_float, s_lower = sigma_min_sample_bounds(matrix)
        true_sigma = float(np.linalg.svd(matrix, compute_uv=False)[-1])
        _check(
            f"{name}: sigma_min_sample_bounds is a valid lower bound",
            s_lower <= true_sigma + 1e-12,
            f"lower {s_lower:.6e} <= true {true_sigma:.6e}",
        )
        record[name] = {"lower": s_lower, "true": true_sigma}

    # issue 2: the U^T U rounding bound sums ``U.shape[0]`` terms, not ``singular.size``.  An
    # exactly orthonormal (N x 2) U has ``|U^T U - I| = 0``, so delta_u is exactly the rounding
    # allowance ``gamma(rows)``; the old ``gamma(singular.size) = gamma(2)`` overstates the bound.
    rows = 100_000
    orth = np.zeros((rows, 2))
    orth[0, 0] = 1.0
    orth[1, 1] = 1.0
    singular_desc = np.array([2.0, 1.0])
    length_bound = _svd_sigma_min_lower(orth @ np.diag(singular_desc), orth, singular_desc, np.eye(2))
    rigorous_u = float(np.sqrt((1.0 - _gamma(rows)) * (1.0 - _gamma(2))))
    _check(
        "tall U^T U rounding uses U.shape[0], not singular.size",
        length_bound is not None and length_bound <= rigorous_u + 1e-12,
        f"bound {length_bound:.15e} <= rigorous {rigorous_u:.15e}",
    )

    # issue 3: the Gram rounding bound of the rectangular inverse route sums the *longer* side
    # (``AA^T`` for a wide A sums ``columns`` terms, ``A^T A`` for a tall A sums ``rows`` terms),
    # so it uses ``max(rows, columns)``.  A 1xN row of ones has ``sigma_min = sqrt(N)`` and an
    # exactly-computable Gram ``AA^T = N``, making the allowance exactly ``gamma(N)``.
    ones_wide = np.ones((1, rows))
    ones_tall = np.ones((rows, 1))
    rigorous_gram = float(np.sqrt(rows * (1.0 - _gamma(rows))))
    inv_wide = _inverse_sigma_min_lower(ones_wide)
    inv_tall = _inverse_sigma_min_lower(ones_tall)
    _check(
        "wide (1xN) Gram rounding uses max(rows, columns)",
        inv_wide <= rigorous_gram + 1e-12,
        f"bound {inv_wide:.15e} <= rigorous {rigorous_gram:.15e}",
    )
    _check(
        "tall (Nx1) Gram rounding uses max(rows, columns)",
        inv_tall <= rigorous_gram + 1e-12,
        f"bound {inv_tall:.15e} <= rigorous {rigorous_gram:.15e}",
    )
    record["u_length"] = {"bound": length_bound, "rigorous": rigorous_u}
    record["gram_length"] = {"wide": inv_wide, "tall": inv_tall, "rigorous": rigorous_gram}
    return record


def phase_bound(model, rng, *, probes: int = 48) -> dict:
    """Phase 1: the analytic bound of B1-B3, and a finite-difference indexing check.

    The finite-difference comparison is explicitly *not* a proof of the theorem (B4): it only
    shows that the implemented indices are the ones the formula names, by sampling
    ``||dJ/dt||`` along the segment and comparing it with ``L``.
    """
    print("phase 1: the global bound L of B1-B3 (analytic; finite differences check indexing only)")
    lengths = chain_lengths(model)
    d = rng.uniform(-0.3, 0.3, model.num_dof)
    bound = lipschitz_bound(model, d)
    print(f"  ell_j (reach bounds from the link lengths, m): {np.round(lengths, 4)}")
    print(f"  |d| = {np.linalg.norm(d):.4f} rad; a_j (upstream budget) = {np.round(bound.upstream, 4)}")
    print(f"  b_j (||dJv_j/dt|| bound) = {np.round(bound.derivative, 4)}")
    print(f"  L = {bound.lipschitz:.4f} (Frobenius bound on ||dJbar/dt||_2), "
          f"L1 = {bound.l_one:.4f} (B3's square-root-free form)")
    worst = 0.0
    for _ in range(probes):
        q = rng.uniform(-1.2, 1.2, model.num_dof)
        t = rng.uniform(0.05, 0.95)
        h = 1e-6
        plus = np.asarray(model.jacobian(q + (t + h) * d), dtype=float)
        minus = np.asarray(model.jacobian(q + (t - h) * d), dtype=float)
        ratio = float(np.linalg.norm((plus - minus) / (2.0 * h), 2)) / bound.lipschitz
        worst = max(worst, ratio)
    _check(
        "finite-difference ||dJ/dt|| stays below L",
        worst <= 1.0,
        f"worst ratio {worst:.3f} of L (indexing check, not the proof)",
    )
    return {
        "lengths": lengths.tolist(),
        "l": bound.lipschitz,
        "l_one": bound.l_one,
        "fd_ratio": worst,
        "allowance": numerical_allowance(model),
    }


def phase_manufactured(model, rng, *, max_samples: int, rel_tol: float, dense: int) -> dict:
    """Phase 2: a certified success, a refused false positive, and negative controls."""
    print("\nphase 2: manufactured segments")
    source = model
    record: dict = {}

    print("\n  2a. a segment with healthy clearance is certified, and the bound is consistent")
    good = None
    for _ in range(40):
        q0 = rng.uniform([-2.0, -1.6, -1.6, -2.0, -2.0, -2.0], [2.0, 1.6, 1.6, 2.0, 2.0, 2.0])
        d = rng.uniform(-0.05, 0.05, model.num_dof)
        candidate = certify_segment(model, q0, d, rel_tol=rel_tol, max_samples=max_samples)
        if candidate.status == "certified" and candidate.certified_lower > 1e-3:
            good = (q0, d, candidate)
            break
    if good is None:
        _check("a healthy segment certifies positive", False, "no candidate certified")
        return record
    q0, d, candidate = good
    dense_min = _dense_min(source, q0, d, dense)
    print(f"  q0 = {np.round(q0, 4)}, d = {np.round(d, 4)} (|d| = {np.linalg.norm(d):.4f} rad)")
    print(f"  L = {candidate.lipschitz:.4f}, h = {candidate.h:.3e}, samples {candidate.samples}, "
          f"margin L*h = {candidate.margin_lipschitz:.3e}, numeric {candidate.margin_numeric:.2e}")
    print(f"  certified sigma_min >= {candidate.certified_lower:.6e} "
          f"(bracket [{candidate.bracket[0]:.6e}, {candidate.bracket[1]:.6e}], "
          f"dense {dense}-sample minimum {dense_min:.6e})")
    _check("healthy segment is certified positive", candidate.positive,
           f"certified lower {candidate.certified_lower:.3e} > 0")
    _check(
        "the certificate never exceeds a dense scan of the same segment",
        candidate.certified_lower <= dense_min + 1e-9,
        f"certified {candidate.certified_lower:.3e} <= dense min {dense_min:.3e}",
    )
    record["healthy"] = {
        "q0": q0.tolist(), "d": d.tolist(), "certified_lower": candidate.certified_lower,
        "sampled_min": candidate.sampled_min, "dense_min": dense_min, "l": candidate.lipschitz,
        "h": candidate.h, "samples": candidate.samples,
        "margin_lipschitz": candidate.margin_lipschitz,
        "margin_numeric": candidate.margin_numeric, "seconds": candidate.seconds,
    }

    print("\n  2b. a segment that crosses Sigma between two samples: sampling says fine, so does "
          "the project's own step checker; the certificate refuses")
    crossing = None
    for _ in range(8):
        q_start = rng.uniform([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        direction = rng.normal(size=model.num_dof)
        found = find_det_crossing(
            lambda q: iks.det_jacobian(model, q), q_start, direction, model
        )
        if found is None:
            continue
        q_cross, residual = found
        built = crossing_segment(
            model, q_cross, direction, ratio=0.55, span=0.2, coarse=9,
            det_of=lambda q: iks.det_jacobian(model, q), floor=1e-3,
        )
        if built is not None:
            crossing = (q_cross, residual, built, direction)
            break
    if crossing is None:
        _check("a crossing segment could be manufactured", False, "no crossing found")
        return record
    q_cross, residual, (q_seg, d_seg, coarse_min), _ = crossing
    false_certificate = certificate_step_check(model, q_seg, d_seg, 1e-3, 9)
    dense_cross = _dense_min(source, q_seg, d_seg, max(dense, 20001))
    adaptive = certify_segment(model, q_seg, d_seg, delta=1e-3, rel_tol=rel_tol,
                               max_samples=max_samples)
    print(f"  crossing bisected to |det J| = {residual:.2e} at t = 0.55 of the segment")
    print(f"  coarse 9-sample minimum sigma_min = {coarse_min:.3e} > 0: a sampled certificate "
          f"would claim clearance")
    print(f"  chamber's own step checker (_segment_clears, density 9, delta 1e-3) says "
          f"clear: {false_certificate}")
    print(f"  dense 20001-sample minimum = {dense_cross:.3e} (the crossing is real)")
    print(f"  certificate: status {adaptive.status!r}, certified lower "
          f"{adaptive.certified_lower:.3e}, bracket [{adaptive.bracket[0]:.3e}, "
          f"{adaptive.bracket[1]:.3e}], samples {adaptive.samples}")
    _check("the sampled minimum alone reports a healthy clearance", coarse_min > 1e-3,
           f"coarse min {coarse_min:.3e} > 1e-3")
    _check("chamber's sampled step check is fooled by the same segment", bool(false_certificate))
    _check("the segment really crosses Sigma", dense_cross < 1e-8,
           f"dense min {dense_cross:.3e}")
    _check("the certificate refuses the crossing", not adaptive.positive,
           f"status {adaptive.status!r}, certified lower {adaptive.certified_lower:.3e}")
    _check("Sampled determinant changes sign on the segment (crossing witness)",
           adaptive.det_sign_change is True)
    record["crossing"] = {
        "q_seg": q_seg.tolist(), "d_seg": d_seg.tolist(), "det_residual": residual,
        "coarse_min": coarse_min, "dense_min": dense_cross, "status": adaptive.status,
        "certified_lower": adaptive.certified_lower, "bracket": list(adaptive.bracket),
        "samples": adaptive.samples, "sampled_min": adaptive.sampled_min,
    }

    print("\n  2c. negative controls: opposite endpoint signs of det J never certify")
    controls = []
    tries = 0
    while len(controls) < 4 and tries < 200:
        tries += 1
        q_a = rng.uniform([-1.5, -1.0, -1.0, -1.5, -1.5, -1.5], [1.5, 1.0, 1.0, 1.5, 1.5, 1.5])
        q_b = rng.uniform([-1.5, -1.0, -1.0, -1.5, -1.5, -1.5], [1.5, 1.0, 1.0, 1.5, 1.5, 1.5])
        if iks.det_jacobian(model, q_a) * iks.det_jacobian(model, q_b) >= 0.0:
            continue
        controls.append((q_a, q_b - q_a))
    refused = []
    for q_a, d_a in controls:
        certificate = certify_segment(model, q_a, d_a, rel_tol=rel_tol,
                                      max_samples=max_samples)
        refused.append(certificate)
    print(f"  {len(controls)} random segments with det J of opposite signs at the endpoints; "
          f"highest certified lower {max(c.certified_lower for c in refused):.3e}, "
          f"statuses {[c.status for c in refused]}")
    _check(
        "no opposite-sign segment is ever certified",
        all(not c.positive for c in refused),
        f"{len(refused)} segments, statuses {sorted({c.status for c in refused})}",
    )
    _check(
        "every opposite-sign segment does show a sampled sign change",
        all(c.det_sign_change for c in refused),
        "",
    )
    _check(
        "a failed certificate is reported as inconclusive, not as a crossing",
        all(c.status in ("inconclusive", "invalid") for c in refused),
        "",
    )
    record["controls"] = {
        "count": len(controls),
        "statuses": [c.status for c in refused],
        "max_certified_lower": max(c.certified_lower for c in refused),
        "samples": sum(c.samples for c in refused),
    }

    print("\n  2d. a lift that leaves the joint limits is invalid, not certified")
    lower = np.asarray(model.lower_bounds, dtype=float)
    q_edge = lower + 1e-3
    d_mid = np.zeros(model.num_dof)
    d_mid[0] = -0.5
    invalid = certify_segment(model, q_edge, d_mid, max_samples=max_samples)
    _check("a lift past the joint limit is invalid", invalid.status == "invalid",
           f"status {invalid.status!r}, out_of_bounds {invalid.out_of_bounds}")
    record["out_of_bounds"] = {"status": invalid.status, "out_of_bounds": invalid.out_of_bounds}
    return record


def certificate_step_check(model, q_a: np.ndarray, q_b: np.ndarray, delta: float,
                           density: int) -> bool:
    """The project's sampled step check (`study.chamber._segment_clears`) on one segment.

    Imported here rather than at module level so that the *default* instrument of
    `study/chamber.py` stays exactly the module that produced the recorded numbers.
    """
    from study.chamber import _segment_clears

    circular = iks.circular_joints(model)
    return bool(_segment_clears(model, q_a, q_b, delta, circular, density=density))


def phase_three_r(model, rng, *, max_samples: int, rel_tol: float, dense: int) -> dict:
    """Phase 3: the 3R position task (three linear rows) and the unit-scaling invariance."""
    print("\nphase 3: the position-only 3R task, and the unit-scaling invariance of B3")
    adapter = c3.PositionTask(model)
    q0 = np.array([0.3, 0.4, -0.7])
    d = np.array([0.05, -0.03, 0.02])
    certificate = certify_segment(model, q0, d, task=adapter, rel_tol=rel_tol,
                                  max_samples=max_samples)
    dense_min = _dense_min(adapter, q0, d, dense)
    print(f"  3R (position task, rows {POSITION_ROWS}): L = {certificate.lipschitz:.4f}, "
          f"certified sigma_min >= {certificate.certified_lower:.4e}, "
          f"bracket [{certificate.bracket[0]:.4e}, {certificate.bracket[1]:.4e}], "
          f"dense {dense_min:.4e}, samples {certificate.samples}")
    _check("3R position task certifies positive", certificate.positive,
           f"certified lower {certificate.certified_lower:.3e}")
    _check("3R certificate is consistent with a dense scan",
           certificate.certified_lower <= dense_min + 1e-9)

    print("\n  invariance (i): (alpha, beta) -> (2*alpha, 2*beta) must not change the answer")
    scaled_metric = certify_segment(model, q0, d, task=adapter, alpha=2.0, beta=2.0,
                                    rel_tol=rel_tol, max_samples=max_samples)
    print(f"  lower on Jbar: {certificate.lower_jbar:.10e} vs {scaled_metric.lower_jbar:.10e}; "
          f"certified lower on J: {certificate.certified_lower:.10e} vs "
          f"{scaled_metric.certified_lower:.10e}")
    _check(
        "the certificate is invariant under a common rescaling of (alpha, beta)",
        abs(certificate.certified_lower - scaled_metric.certified_lower)
        <= 1e-12 * max(1.0, certificate.certified_lower),
    )

    print("\n  invariance (ii): change the length unit (x1000) and alpha -> alpha/1000")
    big = _scaled_3r(1000.0)
    big_adapter = c3.PositionTask(big)
    fixed = 17
    reference = certify_segment(model, q0, d, task=adapter, uniform=fixed)
    rescaled = certify_segment(big, q0, d, task=big_adapter, alpha=DEFAULT_ALPHA / 1000.0,
                               uniform=fixed)
    relative = abs(rescaled.lower_jbar - reference.lower_jbar) / max(
        1e-300, abs(reference.lower_jbar)
    )
    print(f"  with a fixed {fixed}-sample grid the two Jacobians are the same matrix up to "
          f"round-off:")
    print(f"    Jbar-level lower: {reference.lower_jbar:.10e} (m) vs "
          f"{rescaled.lower_jbar:.10e} (mm with alpha = 1e-3); relative gap {relative:.2e}")
    print(f"    L: {reference.lipschitz:.10e} vs {rescaled.lipschitz:.10e}")
    print(f"    the reported sigma_min(J) lower stays in the model's own length unit: "
          f"{reference.certified_lower:.6e} m vs {rescaled.certified_lower:.6e} mm")
    _check("the Jbar-level certificate is invariant under a change of length unit",
           relative <= 1e-9, f"relative gap {relative:.2e}")
    _check("L itself is invariant under a change of length unit",
           abs(rescaled.lipschitz - reference.lipschitz)
           <= 1e-12 * max(1.0, reference.lipschitz))
    adaptive_big = certify_segment(big, q0, d, task=big_adapter, alpha=DEFAULT_ALPHA / 1000.0,
                                   rel_tol=rel_tol, max_samples=max_samples)
    adaptive_gap = abs(adaptive_big.certified_lower - certificate.certified_lower)
    print(f"    adaptive mode: {certificate.certified_lower:.6e} m vs "
          f"{adaptive_big.certified_lower:.6e} mm (gap {adaptive_gap:.2e}; bracket width "
          f"{certificate.bracket[1] - certificate.bracket[0]:.2e} -- the stopping rule can tip on "
          f"round-off, both numbers remain valid bounds)")
    _check("adaptive and fixed subdivisions agree within the bracket tolerance",
           adaptive_gap <= (certificate.bracket[1] - certificate.bracket[0]) + 1e-12)

    print("\n  the same 3R segment with a wrong row classification must not silently pass")
    raised = ""
    try:
        certify_segment(model, q0, d, task=adapter, rows=FULL_ROWS, max_samples=max_samples)
    except ValueError as error:
        raised = str(error)
    _check("row-shape mismatches raise instead of certifying", bool(raised), raised)
    return {
        "three_r": {
            "certified_lower": certificate.certified_lower,
            "bracket": list(certificate.bracket),
            "dense_min": dense_min,
            "l": certificate.lipschitz,
            "samples": certificate.samples,
        },
        "invariance_metric": {
            "lower_jbar_1": certificate.lower_jbar,
            "lower_jbar_2": scaled_metric.lower_jbar,
        },
        "invariance_units": {"lower_jbar_m": certificate.lower_jbar,
                             "lower_jbar_mm": rescaled.lower_jbar,
                             "relative": relative},
    }


def phase_models(*, max_samples: int) -> dict:
    """Phase 3b: the certificate runs on every shape of Jacobian the library has.

    The 6x6 geometric Jacobian is the main case, the 3x3 position task of the 3R is the
    position-only case (B3 drops the angular term), and a redundant arm gives a 6x7 Jacobian,
    where ``sigma_min`` is the smallest of the six singular values: the sample bound uses the
    tall form of the matrix (``J^T``), because only the tall form has an orthonormal right factor.
    """
    print("\nphase 3b: the same interface on other models of the library")
    rows = []
    for name in ("aubo_c5", "er3", "franka"):
        model = ModelFactory.create(name)
        count = int(model.num_dof)
        q0 = np.zeros(count) + 0.2
        d = np.full(count, 0.02)
        certificate = certify_segment(model, q0, d, max_samples=max_samples)
        shape = np.asarray(model.jacobian(q0)).shape
        print(f"  {name}: {count} joints, Jacobian {shape}, L {certificate.lipschitz:.4f}, "
              f"status {certificate.status}, certified lower "
              f"{certificate.certified_lower:.4e}, samples {certificate.samples}")
        rows.append({"model": name, "jacobian": list(shape), "l": certificate.lipschitz,
                     "status": certificate.status,
                     "certified_lower": certificate.certified_lower,
                     "samples": certificate.samples})
    _check(
        "the certificate certifies a segment on 6x6, 6x7 and other library models",
        all(row["status"] == "certified" for row in rows),
        ", ".join(f"{row['model']} {row['jacobian']}" for row in rows),
    )
    return {"models": rows}


def _failure_diagnostics(model, path, index: int, task=None, *, points: int = 5) -> str:
    """Sample a sub-segment the certificate refused: ``sigma_min`` and ``det J`` across it.

    This is the independent evidence that separates "the certificate failed for budget reasons"
    from "the sub-segment really does cross ``Sigma``": a sign change of the sampled determinant
    with a sampled ``sigma_min`` at float noise is a crossing witness the certificate cannot
    produce on its own (B4 forbids reading a failed certificate as a crossing).
    """
    source = model if task is None else task
    left = np.asarray(path[index], dtype=float)
    step = np.asarray(path[index + 1], dtype=float) - left
    parts = []
    for t in np.linspace(0.0, 1.0, points):
        jacobian = np.asarray(source.jacobian(left + t * step), dtype=float)
        sigma = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        determinant = float(np.linalg.det(jacobian)) if jacobian.shape[0] == jacobian.shape[1] else 0.0
        parts.append(f"t={t:.2f} sigma={sigma:.2e} det={determinant:+.2e}")
    return f"    refused sub-segment {index}: " + "; ".join(parts)


def _row(name: str, certificate: PathCertificate, old_sampled: float, segments: int) -> dict:
    """One printed row of the witness-path table."""
    verdict = "CERTIFIED" if certificate.certified else "sampled only"
    print(f"  {name}")
    print(f"    points {segments + 1}, sub-segments {certificate.segments} "
          f"(skipped {certificate.skipped}), largest step {certificate.max_step:.4f} rad")
    print(f"    old sampled minimum (the recorded instrument)      {old_sampled:.6e}")
    print(f"    certificate's own sampled minimum                  {certificate.sampled_min:.6e}")
    print(f"    certified lower bound on sigma_min                 {certificate.certified_lower:.6e}")
    print(f"    status {verdict}: {certificate.certified_segments}/{certificate.segments} "
          f"segments certified, statuses {certificate.statuses}")
    print(f"    L_max {certificate.max_l:.4f}, h_worst {certificate.h_worst:.3e}, samples "
          f"{certificate.samples}, {certificate.seconds:.2f} s, "
          f"numeric allowance {certificate.margin_numeric:.2e}")
    print(f"    certified > delta: {certificate.delta_ok}; sampled det J sign changes: "
          f"{certificate.det_sign_changes}")
    if certificate.failures:
        print(f"    not certified at sub-segments {certificate.failures[:12]}"
              f"{' ...' if len(certificate.failures) > 12 else ''}")
    return {
        "name": name,
        "segments": certificate.segments,
        "old_sampled_min": old_sampled,
        "sampled_min": certificate.sampled_min,
        "certified_lower": certificate.certified_lower,
        "certified": certificate.certified,
        "delta_ok": certificate.delta_ok,
        "statuses": certificate.statuses,
        "failures": certificate.failures[:32],
        "samples": certificate.samples,
        "seconds": certificate.seconds,
        "max_l": certificate.max_l,
        "h_worst": certificate.h_worst,
        "margin_numeric": certificate.margin_numeric,
        "max_step": certificate.max_step,
    }


def phase_paths(model, args) -> dict:
    """Phase 4: the witness paths the project already trusts, sampled versus certified."""
    print("\nphase 4: recorded witness paths -- which conclusions get upgraded")
    record: dict = {}

    print("\n  4a. exp49 SR5 witness path (rich_pose + chamber.clearance_path, phase 1-2 verbatim)")
    started = time.perf_counter()
    replay = exp49_witness_path(
        pose_seed=args.pose_seed, seeds=args.seeds, least=args.least, tries=args.tries
    )
    replay_seconds = time.perf_counter() - started
    if replay is None:
        print("  exp49: no witness path reproduced (the pose search failed at this budget)")
        record["exp49"] = None
        _check("exp49 witness path reproduced", False, "pose search failed")
    else:
        sr5, solutions, signs, pair, walk = replay
        points = np.array(walk.path)
        certificate = certify_path(sr5, points, delta=5e-3, rel_tol=args.rel_tol,
                                   max_samples=args.max_samples)
        print(f"    replay: pose-seed {args.pose_seed}, {len(solutions)} solutions, "
              f"pair {pair} both det J sign {signs[pair[0]]:+d}, walk reason {walk.reason!r}, "
              f"{replay_seconds:.1f} s to rebuild")
        row = _row("exp49 SR5 clearance walk (same-aspect pair)", certificate, walk.min_sigma,
                   len(points) - 1)
        record["exp49"] = row
        _check("exp49 witness walk reproduced", True, f"{len(points)} configurations")
        _check(
            "exp49 same-aspect conclusion is certified (sigma_min > 0 on the whole polyline)",
            certificate.certified,
            f"{certificate.certified_segments}/{certificate.segments} segments certified, "
            f"certified lower {certificate.certified_lower:.3e}",
        )

    print("\n  4b. exp48 3R route (aspect labels + clearance legs, phase 1-4 verbatim)")
    started = time.perf_counter()
    built = exp48_witness_route(grid=args.grid, fine_grid=args.fine_grid)
    replay_seconds = time.perf_counter() - started
    if built is None:
        print("  exp48: no same-aspect route reproduced at this budget")
        record["exp48"] = None
        _check("exp48 route reproduced", False, "no route")
    else:
        three_r, labels, pair, route, refined = built
        adapter = c3.PositionTask(three_r)
        nodes = min(c3.sigma_min_position(three_r, q) for q in route)
        certificate = certify_path(three_r, route, task=adapter, delta=5e-3,
                                   rel_tol=args.rel_tol, max_samples=args.max_samples)
        print(f"    replay: {len(labels)}x{len(labels)} label grid, pair distance "
              f"{c3.torus_distance(pair[0], pair[1]):.4f} rad, "
              f"{'clearance-refined' if refined else 'raw grid route (legs failed)'}, "
              f"{replay_seconds:.1f} s to rebuild")
        row = _row("exp48 3R same-aspect route", certificate, nodes, len(route) - 1)
        row["refined"] = bool(refined)
        record["exp48"] = row
        for index in certificate.failures[:3]:
            print(_failure_diagnostics(three_r, route, index, task=adapter))
        if certificate.certified:
            print("    VERDICT: every grid edge is certified non-singular -- the route's topology "
                  "(aspect labels) and its clearance are both certified now")
        else:
            print("    VERDICT: NOT upgraded. The aspect labels still certify the *topology* "
                  "(exact grid components), but the route's clearance stays sampled -- and here it "
                  "is worse than sampled: the refused sub-segment has a sampled det J sign change, "
                  "i.e. the recorded grid route crosses Sigma between its nodes.")
        record["exp48"]["diagnostics"] = [
            _failure_diagnostics(three_r, route, index, task=adapter)
            for index in certificate.failures[:3]
        ]
        record["exp48"]["verdict"] = (
            "certified" if certificate.certified else "not certified (crosses Sigma between nodes)"
        )
        _check("exp48 witness route reproduced", True, f"{len(route)} configurations")
        _check(
            "exp48 route: the verdict is consistent with an independent crossing witness",
            certificate.certified or certificate.det_sign_changes > 0,
            f"{certificate.certified_segments}/{certificate.segments} certified, "
            f"{certificate.det_sign_changes} sampled det J sign change(s)",
        )

    print("\n  4c. the opt-in chamber wrapper leaves the default numbers alone")
    endpoints = None
    if replay is not None:
        walk0 = replay[4]
        endpoints = (walk0.path[0], walk0.path[-1], circular_from(sr5), 5e-3, 1500)
        print("    endpoints: the exp49 witness pair, where the greedy walk is known to arrive")
    else:
        print("    endpoints: a random short displacement (the greedy walk may spin below its "
              "step size; that is the planner, not the certificate)")
    opt_in = opt_in_check(model, endpoints)
    record["opt_in"] = opt_in
    _check("chamber.clearance_path's default behaviour is untouched", opt_in["unchanged"],
           opt_in["detail"])
    _check("chamber.clearance_path accepts the opt-in certificate",
           opt_in["opt_in_works"], opt_in["opt_in_detail"])
    _check("the opt-in flag is true exactly when the walk arrived and every segment certified",
           opt_in["flag_consistent"], opt_in["opt_in_detail"])
    return record


def circular_from(model) -> np.ndarray:
    """The circular-joint mask of ``model`` (a named helper so the opt-in replay reads clearly)."""
    return iks.circular_joints(model)


def opt_in_check(model, endpoints=None) -> dict:
    """Check that the certificate hook in `study/chamber.py` is opt-in and functional.

    The invariance of the default instrument is checked by running the plain walk and the wrapped
    one on the same endpoints and comparing every field.  When ``endpoints`` are supplied (the
    exp49 witness pair), the greedy planner is known to arrive, so the interesting case -- a
    connected walk whose entire polyline is certified -- is exercised; without them a random short
    displacement is used, where the greedy walk may spin (a pre-existing property of the planner
    for spans below its ``step_size``, not something this opt-in changes).
    """
    from study.chamber import ClearanceWalk, certified_clearance_path

    if endpoints is not None:
        q_from, q_to, circular, delta, steps = endpoints
    else:
        rng = np.random.default_rng(7)
        q_from = rng.uniform([-1.0, -0.8, -0.8, -1.0, -1.0, -1.0],
                             [1.0, 0.8, 0.8, 1.0, 1.0, 1.0])
        q_to = q_from + 0.25 * np.array([0.15, -0.1, 0.1, 0.0, 0.0, 0.0])
        circular, delta, steps = None, 1e-3, 2000
    options = {} if circular is None else {"circular": circular}
    plain = clearance_path(model, q_from, q_to, delta=delta, steps=steps, **options)
    walk, certificate = certified_clearance_path(
        model, q_from, q_to, delta=delta, steps=steps, **options
    )
    same = (
        plain.connected == walk.connected
        and plain.reason == walk.reason
        and plain.min_sigma == walk.min_sigma
        and len(plain.path) == len(walk.path)
    )
    default_flag = ClearanceWalk(connected=True, min_sigma=1.0, steps=0).certified is False
    flag_consistent = walk.certified == bool(
        walk.connected and certificate is not None and certificate.certified
    )
    detail = (
        f"plain walk connected {plain.connected} ({plain.reason!r}), min sigma "
        f"{plain.min_sigma:.3e}, {len(plain.path)} points; wrapped walk identical: {same}; "
        f"new field defaults to False: {default_flag}"
    )
    return {
        "unchanged": bool(same and default_flag),
        "detail": detail,
        "flag_consistent": bool(flag_consistent),
        "walk_connected": bool(walk.connected),
        "walk_certified": bool(walk.certified),
        "opt_in_works": bool(certificate is not None and certificate.certified and flag_consistent),
        "opt_in_detail": (
            f"walk.certified {walk.certified}, certificate {certificate.certified_segments}/"
            f"{certificate.segments} segments certified, lower "
            f"{certificate.certified_lower:.3e}, flag consistent: {flag_consistent}"
            if certificate is not None
            else "no certificate returned (walk has fewer than two points)"
        ),
        "min_sigma": plain.min_sigma,
    }


def main() -> int:
    """Run every phase, print the measurements, and return a non-zero exit on a failed check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small budgets, for a smoke test")
    parser.add_argument("--max-samples", type=int, default=4096, help="sample budget per segment")
    parser.add_argument("--rel-tol", type=float, default=0.05,
                        help="bracket width of the reported bound, relative to the sampled min")
    parser.add_argument("--dense", type=int, default=2001, help="dense-scan samples for controls")
    parser.add_argument("--pose-seed", type=int, default=0, help="exp49 replay seed")
    parser.add_argument("--seeds", type=int, default=400, help="exp49 census seeds")
    parser.add_argument("--least", type=int, default=8, help="exp49 minimum fibre size")
    parser.add_argument("--tries", type=int, default=14, help="exp49 pose attempts")
    parser.add_argument("--grid", type=int, default=300, help="exp48 aspect-label grid")
    parser.add_argument("--fine-grid", type=int, default=25, help="exp48 workspace grid")
    parser.add_argument("--skip-paths", action="store_true", help="skip phase 4")
    parser.add_argument("--json", type=str, default=None, help="write the measurements as JSON")
    args = parser.parse_args()
    if args.quick:
        args.max_samples = min(args.max_samples, 256)
        args.seeds = min(args.seeds, 120)
        args.tries = min(args.tries, 4)
        args.grid = min(args.grid, 120)
        args.fine_grid = min(args.fine_grid, 8)
        args.dense = min(args.dense, 401)

    print("exp63: certifying 'no singularity on this segment' instead of sampling it")
    print("bound of study/collab/GPT_MATH_ADVANCE_20261016.md B1-B3; "
          "floating-point accounting of B4\n")
    rng = np.random.default_rng(0)
    model = ModelFactory.create("sr5")
    record: dict = {"command": "python3 -m study.exp63_segment_certificate"}
    record["svd_regressions"] = phase_svd_regressions()
    record["bound"] = phase_bound(model, rng)
    record["manufactured"] = phase_manufactured(
        model, rng, max_samples=args.max_samples, rel_tol=args.rel_tol, dense=args.dense
    )
    three_r = c3.build()
    record["three_r"] = phase_three_r(
        three_r, np.random.default_rng(1), max_samples=args.max_samples, rel_tol=args.rel_tol,
        dense=args.dense,
    )
    record["models"] = phase_models(max_samples=args.max_samples)
    if args.skip_paths:
        print("\nphase 4 skipped (--skip-paths)")
    else:
        record["paths"] = phase_paths(model, args)

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED checks:")
        for name in failed:
            print(f"  - {name}")
    record["checks"] = [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in CHECKS]
    record["failed"] = failed
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"measurements written to {args.json}")
    print("VERDICT: " + ("all checks passed" if not failed else f"{len(failed)} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKS",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "FULL_ROWS",
    "LipschitzBound",
    "POSITION_ROWS",
    "PathCertificate",
    "SegmentCertificate",
    "certify_path",
    "certify_segment",
    "chain_lengths",
    "crossing_segment",
    "exp48_witness_route",
    "exp49_witness_path",
    "find_det_crossing",
    "lipschitz_bound",
    "numerical_allowance",
    "row_kinds",
    "sigma_min_sample_bounds",
]
