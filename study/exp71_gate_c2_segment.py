"""exp71: Gate C2 -- exact endpoint-formula tubes for two consecutive parameter segments.

C2 of the Gate C plan (``GPT_C1_REVIEW_C2_ADDENDUM_20261017.md``): certify a whole
parameter segment with a fixed-centre tube using the exact endpoint formula, and splice
two consecutive segments with the endpoint-enclosure inclusion, for the six common-base
roots {1, 4, 6, 8, 10, 13} of the frozen exact model (exp69/exp70).

The segments lie on the first leg of the exp68 main loop: the archived path starts at
``u = 0`` and runs straight up the imaginary axis (``u = i*t``, pose ``p0 + u*d`` with
``d = -e_x``), so the certified segments take ``lambda = i*s`` with dyadic ``s`` inside
the first archived node interval ``[u[0], u[1]]`` (same straight line, checkable).  The
orientation target is frozen and the position target is affine in ``lambda``, so
``f(x, lambda) = f_base(x) - g(lambda)`` is affine in the parameter and the Jacobian
``Df(x, lambda) = Df_base(x)`` is parameter-independent.  Consequently
``v(lambda) = B f(c, lambda)`` satisfies ``v(lambda(t)) = (1-t) v_A + t v_B`` EXACTLY
(the certificate stores an exact equality check of this identity at the midpoint), and
``eta_i = max(|v_A[i]|, |v_B[i]|)`` bounds row ``i`` of ``B f(c, .)`` on the whole
segment with no sampling assumption.  With the interval row bounds
``kappa_i >= sup_X ||row_i(I - B Df)||_1`` (reusing the exact rational interval backend
of exp70), the strict per-row margins ``r - eta_i - kappa_i r > 0`` make
``T_t(x) = x - B f(x, lambda(t))`` a strict contraction of the tube into its interior
for every ``t``, so the tube contains a unique regular root for every parameter value,
continuous in ``t``.

Splice: ``E = c - B f(c, lambda_B) + (I - B Df(X))(X - c)`` provably contains the
right-end root; the strict inclusion ``E`` inside the next tube plus uniqueness in both
tubes establishes the root identity chain, and the C1 root box inside the first tube
ties the chain to the exp70 roots.  Negative control: the second tube of a DIFFERENT
root (individually valid) must be rejected by the gluing check.  ``--verify`` first
re-runs the repaired complete C1 check and then recomputes every stored C2 number
exactly.

Boundary: C2 only -- no three-loop certification, no permutation claims, no
``path_certified`` flag.

Run::

    python3 -m study.exp71_gate_c2_segment
    python3 -m study.exp71_gate_c2_segment --verify study/exp71_gate_c2_certificate.json
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from study.exp70_gate_c1_roots import (
    B_INVERSE_TOLERANCE,
    HALF,
    MODEL_JSON,
    ONE,
    SEED_NPZ,
    TRACKED_INDICES,
    ZERO,
    RInterval,
    XVector,
    build_system,
    check_model_provenance,
    contraction_bounds,
    eval_f,
    exclusion_bounds,
    frac_str,
    fraction_matrix_det,
    interval_df_matrix,
    lp_eval_float,
    manifest_links,
    matvec,
    parse_xvector,
    ri_add,
    ri_mul,
    ri_point,
    ri_scale,
    ri_sub,
    system_fingerprint,
)
from study.exp70_gate_c1_roots import (
    CERT_JSON as C1_CERT_JSON,
)
from study.exp70_gate_c1_roots import (
    verify_payload as verify_c1_payload,
)

C2_CERT_JSON = Path("study/exp71_gate_c2_certificate.json")
SEG1_HI_K = 25
SEG2_HI_K = 24
RADIUS_SEARCH_MAX_K = 12
COMFORT_FACTOR = Fraction(1, 4)

Lam = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------
# parameter lambda, the affine shift, and float Newton seeds
# ---------------------------------------------------------------------------


def lam_str(lam: Lam) -> list[str]:
    """Serialize a complex dyadic parameter as exact re/im rational strings."""
    return [frac_str(lam[0]), frac_str(lam[1])]


def lam_mid(lam_lo: Lam, lam_hi: Lam) -> Lam:
    """Exact midpoint of a parameter segment."""
    return ((lam_lo[0] + lam_hi[0]) / 2, (lam_lo[1] + lam_hi[1]) / 2)


def load_direction() -> tuple[Fraction, Fraction, Fraction]:
    """The archived translation direction, frozen as exact binary rationals."""
    with np.load(SEED_NPZ, allow_pickle=False) as data:
        direction = np.asarray(data["direction"], dtype=float)
    exact = tuple(Fraction.from_float(float(value)) for value in direction)
    if exact != (Fraction(-1), ZERO, ZERO):
        raise ValueError(f"archived direction {exact} is not (-1, 0, 0)")
    return exact  # type: ignore[return-value]


def shift_vector(direction: tuple[Fraction, Fraction, Fraction], lam: Lam) -> list[Fraction]:
    """``g(lambda) = realify((lambda*d, 0, 0, 0))``: the affine parameter shift."""
    re_part, im_part = lam
    dx, dy, dz = direction
    return [re_part * dx, im_part * dx, re_part * dy, im_part * dy,
            re_part * dz, im_part * dz, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO]


def eval_f_lam(
    equations: list,
    x: XVector,
    lam: Lam,
    direction: tuple[Fraction, Fraction, Fraction],
) -> list[Fraction]:
    """Exact evaluation of the realified system ``f(x, lambda)`` at an exact point."""
    base = eval_f(equations, x)
    shift = shift_vector(direction, lam)
    return [value - delta for value, delta in zip(base, shift)]


def f_complex_float(equations: list, z: np.ndarray, lam: Lam,
                    direction: tuple[Fraction, Fraction, Fraction]) -> np.ndarray:
    """Float evaluation of the complex system at ``(z, lambda)`` (Newton seeds only)."""
    out = np.zeros(6, dtype=complex)
    for k in range(6):
        out[k] = lp_eval_float(equations[k], z)
    lam_complex = complex(float(lam[0]), float(lam[1]))
    for k in range(3):
        out[k] -= lam_complex * complex(float(direction[k]), 0)
    return out


def newton_z(equations: list, jacobian: list, z0: np.ndarray, lam: Lam,
             direction: tuple[Fraction, Fraction, Fraction]) -> np.ndarray:
    """Float Newton on the six-dimensional complex system (seed polishing only)."""
    z = z0.copy()
    for _ in range(20):
        residual = f_complex_float(equations, z, lam, direction)
        if np.max(np.abs(residual)) < 1e-15:
            break
        jf = np.zeros((6, 6), dtype=complex)
        for k in range(6):
            for joint in range(6):
                jf[k, joint] = lp_eval_float(jacobian[k][joint], z)
        z = z - np.linalg.solve(jf, residual)
    return z


def freeze_z(z: np.ndarray) -> XVector:
    """Freeze float complex joints as exact binary rationals (box selection only)."""
    return tuple(Fraction.from_float(part)
                 for value in z for part in (value.real, value.imag))


# ---------------------------------------------------------------------------
# tube certification (exact endpoint formula)
# ---------------------------------------------------------------------------


def preconditioner_at(equations: list, jacobian: list, centre: XVector,
                      direction: tuple[Fraction, Fraction, Fraction],
                      lam: Lam) -> list[list[Fraction]]:
    """Frozen 12x12 preconditioner: float inverse of the realified Jacobian at the centre.

    ``Df`` is parameter-independent, so the evaluation point lambda is irrelevant here;
    it is kept as an argument only for the call-site documentation.
    """
    del direction, lam  # Df does not depend on the parameter (addendum formula)
    z_float = np.array([complex(float(centre[2 * j]), float(centre[2 * j + 1]))
                        for j in range(6)])
    jacobian_float = np.zeros((6, 6), dtype=complex)
    for k in range(6):
        for joint in range(6):
            jacobian_float[k, joint] = lp_eval_float(jacobian[k][joint], z_float)
    realified = np.zeros((12, 12))
    for k in range(6):
        for joint in range(6):
            real_part, imag_part = (jacobian_float[k, joint].real,
                                    jacobian_float[k, joint].imag)
            realified[2 * k, 2 * joint] = real_part
            realified[2 * k, 2 * joint + 1] = -imag_part
            realified[2 * k + 1, 2 * joint] = imag_part
            realified[2 * k + 1, 2 * joint + 1] = real_part
    return [[Fraction.from_float(np.linalg.inv(realified)[row][column])
             for column in range(12)] for row in range(12)]


def certify_tube(
    equations: list,
    jacobian: list,
    s_matrix: list,
    direction: tuple[Fraction, Fraction, Fraction],
    lam_lo: Lam,
    lam_hi: Lam,
    centre: XVector,
    centre_source: str,
) -> tuple[dict, bool]:
    """Certify one fixed-centre tube over the whole segment ``[lam_lo, lam_hi]``.

    Uses the addendum endpoint formula: ``eta_i = max(|v_A[i]|, |v_B[i]|)`` with
    ``v_A = B f(c, lam_lo)`` and ``v_B = B f(c, lam_hi)`` (exact; the affine identity
    is checked exactly at the midpoint), and the parameter-free interval Jacobian
    bounds over the tube.  The radius ladder picks the smallest ``r = 2^-k`` whose
    strict margins ``r - eta_i - kappa_i r >= r/4`` all hold with ``kappa < 1/2``,
    ``|trace S - 3| < 4`` and ``z_j != 0``; a failure is recorded honestly and never
    loosened.
    """
    b_matrix = preconditioner_at(equations, jacobian, centre, direction, lam_lo)
    det_b = fraction_matrix_det(b_matrix)
    if det_b == 0:
        raise RuntimeError("frozen preconditioner is singular -- refusing to certify")
    _, kappa0, _ = contraction_bounds(jacobian, b_matrix, centre, ZERO)
    v_lo = matvec(b_matrix, eval_f_lam(equations, centre, lam_lo, direction))
    v_hi = matvec(b_matrix, eval_f_lam(equations, centre, lam_hi, direction))
    eta_row = [max(abs(a), abs(b)) for a, b in zip(v_lo, v_hi)]
    midpoint = lam_mid(lam_lo, lam_hi)
    v_mid = matvec(b_matrix, eval_f_lam(equations, centre, midpoint, direction))
    affine_exact = all(2 * middle == lo + hi
                       for middle, lo, hi in zip(v_mid, v_lo, v_hi))
    eta_max = max(eta_row)
    start_k = 26 if eta_max == 0 else min(26, max(RADIUS_SEARCH_MAX_K,
                                                 int(np.ceil(-np.log2(float(eta_max))))))
    search = []
    chosen = None
    for k in range(start_k, RADIUS_SEARCH_MAX_K - 1, -1):
        radius = Fraction(1, 2**k)
        kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre, radius)
        trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
        margins = [radius - eta_row[row] - kappa_row[row] * radius for row in range(12)]
        passed = (kappa < HALF and min(margins) >= COMFORT_FACTOR * radius
                  and trace_bound < Fraction(4) and all(value > 0 for value in z_lower))
        search.append({"k": k, "radius": frac_str(radius), "kappa": frac_str(kappa),
                       "min_margin": frac_str(min(margins)),
                       "trace_bound": frac_str(trace_bound),
                       "z_min": frac_str(min(z_lower)), "pass": passed})
        if passed:
            chosen = (k, radius, kappa_row, kappa, z_lower, trace_bound, s_bound, margins)
            break
    certified = chosen is not None
    record = {
        "lam_lo": lam_str(lam_lo),
        "lam_hi": lam_str(lam_hi),
        "centre": [frac_str(value) for value in centre],
        "centre_source": centre_source,
        "B": [[frac_str(value) for value in row] for row in b_matrix],
        "B_det": frac_str(det_b),
        "B_inverse_residual_inf": frac_str(kappa0),
        "v_lo": [frac_str(value) for value in v_lo],
        "v_hi": [frac_str(value) for value in v_hi],
        "eta_row": [frac_str(value) for value in eta_row],
        "affine_endpoint_identity_exact": affine_exact,
    }
    if certified:
        k, radius, kappa_row, kappa, z_lower, trace_bound, s_bound, margins = chosen
        record.update({
            "radius_k": k,
            "radius": frac_str(radius),
            "kappa_row": [frac_str(value) for value in kappa_row],
            "kappa": frac_str(kappa),
            "margin_row": [frac_str(value) for value in margins],
            "margin": frac_str(min(margins)),
            "margin_relative": frac_str(min(margins) / radius),
            "trace_bound": frac_str(trace_bound),
            "trace_margin": frac_str(Fraction(4) - trace_bound),
            "s_minus_i_inf_bound": frac_str(s_bound),
            "z_modulus_squared_lower": [frac_str(value) for value in z_lower],
        })
    else:
        record.update({"radius_k": None, "radius": None, "kappa_row": None, "kappa": None,
                       "margin_row": None, "margin": None, "margin_relative": None,
                       "trace_bound": None, "trace_margin": None,
                       "s_minus_i_inf_bound": None, "z_modulus_squared_lower": None})
    record["radius_search"] = search
    record["verdict"] = "certified" if certified else "inconclusive"
    return record, certified


# ---------------------------------------------------------------------------
# endpoint enclosure and splice inclusion
# ---------------------------------------------------------------------------


def endpoint_enclosure(
    equations: list,
    jacobian: list,
    b_matrix: list[list[Fraction]],
    centre: XVector,
    radius: Fraction,
    lam: Lam,
    direction: tuple[Fraction, Fraction, Fraction],
) -> tuple[list[Fraction], list[RInterval]]:
    """``E = c - B f(c, lam) + (I - B Df(X))(X - c)`` as exact rational intervals.

    Returns:
        ``(g, E)`` with the exact Newton point ``g = c - B f(c, lam)`` and the interval
        enclosure ``E`` of the segment's root at parameter ``lam``.
    """
    df_interval, _ = interval_df_matrix(jacobian, centre, radius)
    g = [centre[row] - value
         for row, value in enumerate(matvec(
             b_matrix, eval_f_lam(equations, centre, lam, direction)))]
    half = RInterval(-radius, radius)
    enclosure = []
    for row in range(12):
        acc = ri_point(g[row])
        for column in range(12):
            entry = ri_point(ONE) if row == column else ri_point(ZERO)
            for k in range(12):
                entry = ri_sub(entry, ri_scale(df_interval[k][column], b_matrix[row][k]))
            acc = ri_add(acc, ri_mul(entry, half))
        enclosure.append(acc)
    return g, enclosure


def inclusion_into(enclosure: list[RInterval], centre2: XVector,
                   radius2: Fraction) -> tuple[bool, list[Fraction], list[Fraction]]:
    """Strict per-coordinate margins of ``E`` inside the interior of the next tube."""
    lo_margins = [enclosure[row].lo - (centre2[row] - radius2) for row in range(12)]
    hi_margins = [(centre2[row] + radius2) - enclosure[row].hi for row in range(12)]
    return all(value > 0 for value in lo_margins + hi_margins), lo_margins, hi_margins


def c1_box_inclusion(c1_centre: XVector, c1_radius: Fraction, tube_centre: XVector,
                     tube_radius: Fraction) -> tuple[bool, list[Fraction]]:
    """Strict per-coordinate margins of the C1 root box inside the first tube."""
    margins = [tube_radius - abs(tube_centre[row] - c1_centre[row]) - c1_radius
               for row in range(12)]
    return all(value > 0 for value in margins), margins


# ---------------------------------------------------------------------------
# certificate assembly
# ---------------------------------------------------------------------------


def archived_leg_reference() -> dict:
    """Freeze the first archived node interval and check the segments lie inside it."""
    with np.load(SEED_NPZ, allow_pickle=False) as data:
        u_points = (np.asarray(data["main_u_real"], dtype=float)
                    + 1j * np.asarray(data["main_u_imag"], dtype=float))
    u0, u1 = u_points[0], u_points[1]
    if u0 != 0 or u1.real != 0 or u1.imag <= 0:
        raise RuntimeError("archived first leg is not the imaginary-axis line from 0")
    frozen_u1 = (Fraction.from_float(u1.real), Fraction.from_float(u1.imag))
    bound = Fraction(1, 2**SEG2_HI_K)
    if not frozen_u1[1] >= bound:
        raise RuntimeError("segment endpoints exceed the first archived node interval")
    return {"archive": str(SEED_NPZ),
            "node_interval_u0": ["0/1", "0/1"],
            "node_interval_u1": [frac_str(frozen_u1[0]), frac_str(frozen_u1[1])],
            "line_note": ("the archived first leg runs from u=0 up the imaginary axis "
                          "(pose p0 + u*d, d=-e_x); the certified segments use "
                          "lambda = i*s with dyadic s on the same line")}


def load_c1_boxes() -> dict[str, tuple[XVector, Fraction]]:
    """The six certified C1 root boxes from the exp70 certificate."""
    stored = json.loads(C1_CERT_JSON.read_text())
    boxes = {}
    for index in TRACKED_INDICES:
        record = stored["roots"][str(index)]
        if record["verdict"] != "certified":
            raise RuntimeError(f"C1 root {index} is not certified -- cannot glue")
        boxes[str(index)] = (parse_xvector(record["centre"]), Fraction(record["radius"]))
    return boxes


def build_certificate() -> dict:
    """Certify the six roots on segment 1 and splice segments 1 and 2 per root."""
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed -- refusing to certify")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_direction()
    lam0 = (ZERO, ZERO)
    lam_a = (ZERO, Fraction(1, 2**SEG1_HI_K))
    lam_b = (ZERO, Fraction(1, 2**SEG2_HI_K))
    c1_boxes = load_c1_boxes()

    with np.load(SEED_NPZ, allow_pickle=False) as data:
        fibre = data["fibre_real"] + 1j * data["fibre_imag"]

    roots: dict[str, dict] = {}
    for _position, root_index in enumerate(TRACKED_INDICES):
        print(f"  root {root_index}:")
        z0 = np.exp(1j * fibre[root_index])
        z_a = newton_z(equations, jacobian, z0.copy(), lam_a, direction)
        z_b = newton_z(equations, jacobian, z_a.copy(), lam_b, direction)
        print(f"    float seeds: |zA-z0|max {np.max(np.abs(z_a - z0)):.2e}, "
              f"|zB-zA|max {np.max(np.abs(z_b - z_a)):.2e}")

        c1 = freeze_z((z0 + z_a) / 2)
        tube1, ok1 = certify_tube(
            equations, jacobian, s_matrix, direction, lam0, lam_a, c1,
            "frozen midpoint of exp(i q_base) and the float Newton root at lambda_A")
        print(f"    segment 1 [{lam_str(lam0)} -> {lam_str(lam_a)}]: "
              f"{tube1['verdict']} "
              + (f"r=2^-{tube1['radius_k']}={float(Fraction(tube1['radius'])):.2e} "
                 f"eta {float(max(Fraction(v) for v in tube1['eta_row'])):.2e} "
                 f"kappa {float(Fraction(tube1['kappa'])):.2e} "
                 f"margin {float(Fraction(tube1['margin'])):.2e} "
                 f"({100 * float(Fraction(tube1['margin_relative'])):.1f}% of r)"
                 if ok1 else "(numbers in record)"))

        b1 = [[Fraction(value) for value in row] for row in tube1["B"]]
        g, enclosure = endpoint_enclosure(equations, jacobian, b1, c1,
                                          Fraction(tube1["radius"]), lam_a, direction)
        c2 = freeze_z(np.array([complex(float(g[2 * j]), float(g[2 * j + 1]))
                                for j in range(6)]))
        tube2, ok2 = certify_tube(
            equations, jacobian, s_matrix, direction, lam_a, lam_b, c2,
            "float-frozen exact Newton step c1 - B1 f(c1, lambda_A)")
        print(f"    segment 2 [{lam_str(lam_a)} -> {lam_str(lam_b)}]: "
              f"{tube2['verdict']} "
              + (f"r=2^-{tube2['radius_k']}={float(Fraction(tube2['radius'])):.2e} "
                 f"eta {float(max(Fraction(v) for v in tube2['eta_row'])):.2e} "
                 f"kappa {float(Fraction(tube2['kappa'])):.2e} "
                 f"margin {float(Fraction(tube2['margin'])):.2e} "
                 f"({100 * float(Fraction(tube2['margin_relative'])):.1f}% of r)"
                 if ok2 else "(numbers in record)"))

        splice_pass, lo_margins, hi_margins = (False, [], [])
        c1_pass, c1_margins = False, []
        if ok1 and ok2:
            c1_centre, c1_radius = c1_boxes[str(root_index)]
            c1_pass, c1_margins = c1_box_inclusion(
                c1_centre, c1_radius, c1, Fraction(tube1["radius"]))
            splice_pass, lo_margins, hi_margins = inclusion_into(
                enclosure, c2, Fraction(tube2["radius"]))
        roots[str(root_index)] = {
            "seed_q": [frac_str(value) for value in
                       (Fraction.from_float(part)
                        for z in fibre[root_index] for part in (z.real, z.imag))],
            "tubes": {"1": tube1, "2": tube2},
            "splice": {
                "lambda_common": lam_str(lam_a),
                "g": [frac_str(value) for value in g],
                "enclosure_lo": [frac_str(value.lo) for value in enclosure],
                "enclosure_hi": [frac_str(value.hi) for value in enclosure],
                "inclusion_lo_margins": [frac_str(value) for value in lo_margins],
                "inclusion_hi_margins": [frac_str(value) for value in hi_margins],
                "passed": splice_pass,
            },
            "c1_box_inclusion": {
                "c1_centre": [frac_str(value) for value in c1_boxes[str(root_index)][0]],
                "c1_radius": frac_str(c1_boxes[str(root_index)][1]),
                "margins": [frac_str(value) for value in c1_margins],
                "passed": c1_pass,
            },
            "verdict": "certified" if (ok1 and ok2 and splice_pass and c1_pass)
                       else "inconclusive",
        }
        print(f"    splice E into tube 2: {splice_pass}, C1 box into tube 1: {c1_pass}")

    for position, root_index in enumerate(TRACKED_INDICES):
        other = TRACKED_INDICES[(position + 1) % len(TRACKED_INDICES)]
        tube1 = roots[str(root_index)]["tubes"]["1"]
        b1 = [[Fraction(value) for value in row] for row in tube1["B"]]
        c1_centre = parse_xvector(tube1["centre"])
        g, enclosure = endpoint_enclosure(
            equations, jacobian, b1, c1_centre, Fraction(tube1["radius"]),
            lam_a, direction)
        other_tube2 = roots[str(other)]["tubes"]["2"]
        other_centre = parse_xvector(other_tube2["centre"])
        other_radius = Fraction(other_tube2["radius"])
        holds, lo_margins, hi_margins = inclusion_into(enclosure, other_centre,
                                                       other_radius)
        if holds:
            raise RuntimeError(f"swapped splice control for root {root_index} -> "
                               f"{other} unexpectedly held -- proof checker bug")
        roots[str(root_index)]["negative_swapped"] = {
            "second_tube_root": other,
            "inclusion_lo_margins": [frac_str(value) for value in lo_margins],
            "inclusion_hi_margins": [frac_str(value) for value in hi_margins],
            "inclusion_holds": False,
            "rejected": True,
        }
        print(f"    negative control {root_index} -> second tube of root {other}: "
              f"gluing rejected (min margin {float(min(lo_margins + hi_margins)):.2e})")

    certified = [index for index, record in roots.items() if record["verdict"] == "certified"]
    payload = {
        "experiment": "exp71_gate_c2_segment",
        "task": ("Gate C, task C2: exact endpoint-formula tubes over one whole segment for "
                 "the six common-base roots, and the splice of two consecutive segments "
                 "(E inside the next tube + C1 box inside the first tube + uniqueness)"),
        "spec": ("study/collab/GPT_C1_REVIEW_C2_ADDENDUM_20261017.md (C2: exact endpoint "
                 "formula; C2 deliverable and acceptance), together with "
                 "GPT_GATE_C_SPEC_20261017.md sections 4-6"),
        "boundary": ("C2 only: two consecutive certified segments; no three-loop "
                     "certification, no permutation claims, no certification flags "
                     "beyond the per-root tubes and splices"),
        "backend": {
            "name": "exact rational interval arithmetic over Python fractions.Fraction "
                    "endpoints (complex rectangular intervals), as in exp70",
            "endpoint_formula": ("fixed orientation + linear translation: Df is "
                                 "parameter-independent and B f(c, lambda(t)) = "
                                 "(1-t) v_A + t v_B exactly; eta_i = max(|v_A[i]|, "
                                 "|v_B[i]|) is an exact whole-segment bound (exact "
                                 "midpoint identity stored per tube)"),
            "floating_inputs": [
                "tube centres frozen as exact binary rationals from float Newton seeds "
                "(box selection only)",
                "B = numpy double inverse of the float Jacobian at the centre, frozen "
                "as exact binary rationals",
            ],
        },
        "coefficient_version": {
            "model_manifest": str(MODEL_JSON),
            "model_manifest_sha256": _sha256(MODEL_JSON),
            "source_sha256": manifest["source_sha256"],
            "system_fingerprint": system_fingerprint(equations, jacobian, s_matrix),
            "c1_certificate": str(C1_CERT_JSON),
            "c1_certificate_sha256": _sha256(C1_CERT_JSON),
        },
        "direction": [frac_str(value) for value in direction],
        "segments": {
            "archived_reference": archived_leg_reference(),
            "1": {"lam_lo": lam_str(lam0), "lam_hi": lam_str(lam_a),
                  "hi_k": SEG1_HI_K},
            "2": {"lam_lo": lam_str(lam_a), "lam_hi": lam_str(lam_b),
                  "hi_k": SEG2_HI_K},
        },
        "roots": roots,
        "batch": {
            "n_certified": len(certified),
            "n_inconclusive": len(TRACKED_INDICES) - len(certified),
            "verdicts": {index: roots[index]["verdict"] for index in roots},
            "splices_passed": sum(1 for index in roots
                                  if roots[index]["splice"]["passed"]),
            "negative_swapped_rejected": sum(1 for index in roots
                                             if roots[index].get("negative_swapped",
                                                                 {}).get("rejected")),
            "summary": f"{len(certified)}/{len(roots)}",
        },
        "verification_command": ("python3 -m study.exp71_gate_c2_segment --verify "
                                 "study/exp71_gate_c2_certificate.json"),
    }
    return payload


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _valid_tube_shape(tube: dict, label: str,
                      report) -> bool:
    """Shape-validate one stored tube record; print explicit rejects."""
    ok = True
    if not isinstance(tube, dict):
        report(f"  {label}: record is not an object", False, type(tube).__name__)
        return False
    for field in ("centre", "v_lo", "v_hi", "eta_row"):
        values = tube.get(field)
        valid = (isinstance(values, list) and len(values) == 12
                 and all(isinstance(value, str) for value in values))
        ok &= valid
        if not valid:
            report(f"  {label}: malformed {field} (12 rational strings required)", False,
                   f"length {len(values) if isinstance(values, list) else '?'}")
    b_rows = tube.get("B")
    valid_b = (isinstance(b_rows, list) and len(b_rows) == 12
               and all(isinstance(row, list) and len(row) == 12 for row in b_rows))
    ok &= valid_b
    if not valid_b:
        report(f"  {label}: malformed B (12x12 rational strings required)", False, "")
    for field in ("lam_lo", "lam_hi"):
        lam = tube.get(field)
        valid = isinstance(lam, list) and len(lam) == 2
        ok &= valid
        if not valid:
            report(f"  {label}: malformed {field} (2 rational strings required)", False, "")
    return ok


def verify_payload_c2(stored: dict) -> int:
    """Recompute every stored C2 number from the frozen inputs, exactly.

    The check trusts nothing inside the JSON: provenance, coefficient fingerprint,
    direction, segment geometry, per-tube exact recomputation, splice inclusions, the
    C1-box inclusion, and the swapped-tube negative controls are all re-derived; the
    recorded verdict must be ``certified`` for all six roots.
    """
    all_ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = all_ok and passed
        short = detail if len(detail) <= 60 else detail[:30] + "..." + detail[-30:]
        suffix = f"  ({short})" if detail else ""
        print(f"    {'PASS' if passed else 'FAIL'}  {name}{suffix}")

    print("verify: full C2 payload")
    required = [str(index) for index in TRACKED_INDICES]
    roots = stored.get("roots")
    if not isinstance(roots, dict) or sorted(roots.keys()) != sorted(required):
        report(f"roots keys exactly {required}", False,
               f"got {sorted(roots.keys()) if isinstance(roots, dict) else type(roots)}")
        print("FAIL: verification failed (root set incomplete or extraneous)")
        return 1
    report(f"roots keys exactly {required}", True, f"{len(roots)} roots")

    manifest, provenance = check_model_provenance()
    for name, passed, detail in provenance:
        report(name, passed, detail)
    if not all(entry[1] for entry in provenance):
        print("FAIL: model provenance failed")
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
           _sha256(MODEL_JSON)
           == stored["coefficient_version"]["model_manifest_sha256"])
    report("C1 certificate sha256 matches stored",
           _sha256(C1_CERT_JSON)
           == stored["coefficient_version"]["c1_certificate_sha256"])
    direction = load_direction()
    report("direction matches the frozen archive",
           stored["direction"] == [frac_str(value) for value in direction],
           str(direction))

    segments = stored["segments"]
    lam0 = (ZERO, ZERO)
    lam_a = (ZERO, Fraction(1, 2**SEG1_HI_K))
    lam_b = (ZERO, Fraction(1, 2**SEG2_HI_K))
    seg1_lo = (Fraction(segments["1"]["lam_lo"][0]), Fraction(segments["1"]["lam_lo"][1]))
    seg1_hi = (Fraction(segments["1"]["lam_hi"][0]), Fraction(segments["1"]["lam_hi"][1]))
    seg2_lo = (Fraction(segments["2"]["lam_lo"][0]), Fraction(segments["2"]["lam_lo"][1]))
    seg2_hi = (Fraction(segments["2"]["lam_hi"][0]), Fraction(segments["2"]["lam_hi"][1]))
    report("segment 1 is [0, i 2^-25]",
           seg1_lo == lam0 and seg1_hi == lam_a and segments["1"]["hi_k"] == SEG1_HI_K,
           lam_str(seg1_hi))
    report("segment 2 is [i 2^-25, i 2^-24]",
           seg2_lo == lam_a and seg2_hi == lam_b and segments["2"]["hi_k"] == SEG2_HI_K,
           lam_str(seg2_hi))
    u0 = (Fraction(segments["archived_reference"]["node_interval_u0"][0]),
          Fraction(segments["archived_reference"]["node_interval_u0"][1]))
    u1 = (Fraction(segments["archived_reference"]["node_interval_u1"][0]),
          Fraction(segments["archived_reference"]["node_interval_u1"][1]))
    report("segments inside the archived first node interval (u0 = 0, Im u1 >= 2^-24)",
           u0 == (ZERO, ZERO) and u1[0] == ZERO and u1[1] >= Fraction(1, 2**SEG2_HI_K),
           f"u1 = {lam_str(u1)}")

    c1_stored = json.loads(C1_CERT_JSON.read_text())
    for index in required:
        record = roots[index]
        print(f"  root {index}:")
        if record.get("verdict") != "certified":
            report("  verdict is certified", False, repr(record.get("verdict")))
            continue
        tube1, tube2 = record["tubes"]["1"], record["tubes"]["2"]
        tube_data = {}
        for tube_name, tube in (("1", tube1), ("2", tube2)):
            if not _valid_tube_shape(tube, f"  tube {tube_name}", report):
                continue
            centre = parse_xvector(tube["centre"])
            b_matrix = [[Fraction(value) for value in row] for row in tube["B"]]
            lam_lo = (Fraction(tube["lam_lo"][0]), Fraction(tube["lam_lo"][1]))
            lam_hi = (Fraction(tube["lam_hi"][0]), Fraction(tube["lam_hi"][1]))
            det_b = fraction_matrix_det(b_matrix)
            report(f"  tube {tube_name}: det B != 0 and matches stored",
                   det_b != 0 and frac_str(det_b) == tube["B_det"], frac_str(det_b))
            try:
                _, kappa0, _ = contraction_bounds(jacobian, b_matrix, centre, ZERO)
            except ValueError:
                report(f"  tube {tube_name}: box avoids z_j = 0", False,
                       "interval division rejected the box")
                continue
            report(f"  tube {tube_name}: ||I - B Df(c)||_inf < 1e-9 and matches stored",
                   kappa0 < B_INVERSE_TOLERANCE
                   and frac_str(kappa0) == tube["B_inverse_residual_inf"], frac_str(kappa0))
            v_lo = matvec(b_matrix, eval_f_lam(equations, centre, lam_lo, direction))
            v_hi = matvec(b_matrix, eval_f_lam(equations, centre, lam_hi, direction))
            eta_row = [max(abs(a), abs(b)) for a, b in zip(v_lo, v_hi)]
            report(f"  tube {tube_name}: v_lo/v_hi/eta_row match stored",
                   [frac_str(value) for value in v_lo] == tube["v_lo"]
                   and [frac_str(value) for value in v_hi] == tube["v_hi"]
                   and [frac_str(value) for value in eta_row] == tube["eta_row"], "")
            midpoint = lam_mid(lam_lo, lam_hi)
            v_mid = matvec(b_matrix, eval_f_lam(equations, centre, midpoint, direction))
            affine_exact = all(2 * middle == lo + hi
                               for middle, lo, hi in zip(v_mid, v_lo, v_hi))
            report(f"  tube {tube_name}: affine endpoint identity re-checked exactly",
                   affine_exact and tube["affine_endpoint_identity_exact"], "")
            if tube.get("verdict") != "certified":
                report(f"  tube {tube_name}: verdict is certified", False,
                       repr(tube.get("verdict")))
                continue
            radius = Fraction(tube["radius"])
            report(f"  tube {tube_name}: radius = 2^-k matches stored",
                   radius == Fraction(1, 2**int(tube["radius_k"]))
                   and frac_str(radius) == tube["radius"], frac_str(radius))
            try:
                kappa_row, kappa, z_lower = contraction_bounds(
                    jacobian, b_matrix, centre, radius)
                trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
            except ValueError:
                report(f"  tube {tube_name}: box avoids z_j = 0", False,
                       "interval division rejected the box")
                continue
            margins = [radius - eta_row[row] - kappa_row[row] * radius
                       for row in range(12)]
            report(f"  tube {tube_name}: kappa_row/kappa/margin_row match stored",
                   [frac_str(value) for value in kappa_row] == tube["kappa_row"]
                   and frac_str(kappa) == tube["kappa"]
                   and [frac_str(value) for value in margins] == tube["margin_row"], "")
            report(f"  tube {tube_name}: trace bound matches stored and < 4",
                   frac_str(trace_bound) == tube["trace_bound"]
                   and trace_bound < Fraction(4), frac_str(trace_bound))
            report(f"  tube {tube_name}: ||S - I||_inf bound matches stored",
                   frac_str(s_bound) == tube["s_minus_i_inf_bound"], frac_str(s_bound))
            report(f"  tube {tube_name}: z_modulus_squared_lower matches stored and > 0",
                   [frac_str(value) for value in z_lower] == tube["z_modulus_squared_lower"]
                   and all(value > 0 for value in z_lower), "")
            report(f"  tube {tube_name}: strict segment contraction re-checked",
                   kappa < ONE and min(margins) > 0,
                   f"kappa {frac_str(kappa)} < 1, margin {frac_str(min(margins))} > 0")
            report(f"  tube {tube_name}: comfort policy re-checked "
                   f"(kappa < 1/2, margin >= r/4)",
                   kappa < HALF and min(margins) >= COMFORT_FACTOR * radius,
                   f"kappa {frac_str(kappa)}, margin {frac_str(min(margins))}")
            tube_data[tube_name] = (centre, b_matrix, radius, lam_lo, lam_hi)

        if "1" not in tube_data or "2" not in tube_data:
            continue
        centre1, b1, radius1, _, lam_hi1 = tube_data["1"]
        centre2, _, radius2, lam_lo2, _ = tube_data["2"]
        splice = record["splice"]
        report("  splice parameter common endpoint matches both tubes",
               lam_hi1 == lam_lo2
               and splice["lambda_common"] == tube1["lam_hi"] == tube2["lam_lo"], "")
        g, enclosure = endpoint_enclosure(equations, jacobian, b1, centre1, radius1,
                                          lam_hi1, direction)
        report("  splice g and enclosure intervals match stored",
               [frac_str(value) for value in g] == splice["g"]
               and [frac_str(value.lo) for value in enclosure] == splice["enclosure_lo"]
               and [frac_str(value.hi) for value in enclosure] == splice["enclosure_hi"], "")
        holds, lo_margins, hi_margins = inclusion_into(enclosure, centre2, radius2)
        report("  splice E strictly inside the next tube (root identity chain)",
               holds and splice["passed"]
               and [frac_str(value) for value in lo_margins] == splice["inclusion_lo_margins"]
               and [frac_str(value) for value in hi_margins] == splice["inclusion_hi_margins"],
               f"min margin {frac_str(min(lo_margins + hi_margins))}")
        c1_record = c1_stored["roots"][index]
        c1_centre = parse_xvector(c1_record["centre"])
        c1_radius = Fraction(c1_record["radius"])
        stored_c1 = record["c1_box_inclusion"]
        report("  stored C1 box equals the C1 certificate box",
               [frac_str(value) for value in c1_centre] == stored_c1["c1_centre"]
               and frac_str(c1_radius) == stored_c1["c1_radius"], "")
        c1_holds, c1_margins = c1_box_inclusion(c1_centre, c1_radius, centre1, radius1)
        report("  C1 root box strictly inside the first tube",
               c1_holds and stored_c1["passed"]
               and [frac_str(value) for value in c1_margins] == stored_c1["margins"],
               f"min margin {frac_str(min(c1_margins))}")
        negative = record.get("negative_swapped", {})
        other = negative.get("second_tube_root")
        if other is None:
            report("  negative swapped control present", False, "missing")
            continue
        other_tube2 = roots[str(other)]["tubes"]["2"]
        other_centre = parse_xvector(other_tube2["centre"])
        other_radius = Fraction(other_tube2["radius"])
        swapped_holds, swapped_lo, swapped_hi = inclusion_into(
            enclosure, other_centre, other_radius)
        report(f"  negative control: gluing to root {other}'s tube must be rejected",
               not swapped_holds and negative["rejected"] is True
               and [frac_str(value) for value in swapped_lo] == negative["inclusion_lo_margins"]
               and [frac_str(value) for value in swapped_hi] == negative["inclusion_hi_margins"],
               f"min margin {frac_str(min(swapped_lo + swapped_hi))} (< 0 required)")

    batch = stored["batch"]
    report("batch summary consistent",
           batch["n_certified"] == len(TRACKED_INDICES)
           and batch["n_inconclusive"] == 0
           and batch["summary"] == f"{len(TRACKED_INDICES)}/{len(TRACKED_INDICES)}"
           and batch["splices_passed"] == len(TRACKED_INDICES)
           and batch["negative_swapped_rejected"] == len(TRACKED_INDICES), batch["summary"])

    print("PASS: C2 certificate fully re-checked (6/6 roots, 12/12 tubes, "
          "6/6 splices, 6/6 negative controls)" if all_ok
          else "FAIL: verification failed")
    return 0 if all_ok else 1


def _self_test_c2(stored: dict) -> None:
    """Negative regressions: tampered payloads must be rejected by the C2 verifier."""
    import contextlib
    import copy as _copy
    import io

    def rejected(name: str, payload: dict) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = verify_payload_c2(payload)
        assert code != 0, f"negative control {name} was accepted (exit 0)"
        print(f"    PASS  negative control rejected as required: {name}")

    canonical = isinstance(stored.get("roots"), dict) and set(stored["roots"]) == set(
        map(str, TRACKED_INDICES))
    if not canonical:
        print("    NOTE  negative self-tests skipped: the payload is already malformed "
              "(the real check below must reject it)")
        return
    empty = {**_copy.deepcopy(stored), "roots": {}}
    rejected("empty roots", empty)
    missing = _copy.deepcopy(stored)
    del missing["roots"]["13"]
    rejected("missing root 13", missing)
    mismatch = _copy.deepcopy(stored)
    mismatch["roots"]["1"]["splice"]["lambda_common"] = lam_str(
        (ZERO, Fraction(1, 2**SEG2_HI_K)))
    rejected("splice endpoint mismatch", mismatch)
    z_zero = _copy.deepcopy(stored)
    z_zero["roots"]["13"]["tubes"]["2"]["centre"][0] = "0/1"
    z_zero["roots"]["13"]["tubes"]["2"]["centre"][1] = "0/1"
    rejected("tube containing z_j = 0", z_zero)
    tampered_direction = _copy.deepcopy(stored)
    tampered_direction["direction"] = ["-1/1", "0/1", "1/2"]
    rejected("tampered direction", tampered_direction)
    tampered_model = _copy.deepcopy(stored)
    tampered_model["coefficient_version"]["model_manifest_sha256"] = "0/1"
    rejected("tampered model sha256", tampered_model)


def verify_certificate(path: Path) -> int:
    """Full C2 acceptance: repaired C1 check first, then the exact C2 recomputation."""
    stored = json.loads(path.read_text())
    print("C1 prerequisite: re-running the repaired complete C1 check")
    c1_code = verify_c1_payload(json.loads(C1_CERT_JSON.read_text()),
                                list(TRACKED_INDICES), f"full C1 ({C1_CERT_JSON})")
    if c1_code != 0:
        print("FAIL: C1 prerequisite failed -- C2 acceptance refused")
        return 1
    print()
    print("verifier negative self-tests:")
    _self_test_c2(stored)
    print()
    return verify_payload_c2(stored)


def main(argv: list[str] | None = None) -> int:
    """Certify the two consecutive C2 segments, or verify a stored certificate."""
    parser = argparse.ArgumentParser(
        description="exp71: Gate C2 exact segment certificates (no path certification)")
    parser.add_argument("--out", type=Path, default=C2_CERT_JSON,
                        help="certificate JSON path")
    parser.add_argument("--verify", type=Path, default=None,
                        help="re-check an existing certificate JSON exactly")
    args = parser.parse_args(argv)
    if args.verify is not None:
        return verify_certificate(args.verify)

    print("exp71: Gate C2 -- exact endpoint-formula segment certificates\n")
    print("phase 0: model provenance")
    manifest, provenance = check_model_provenance()
    for name, passed, detail in provenance:
        print(f"    {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed -- refusing to certify")
    print(f"\nphase 1: certify segment 1 [{lam_str((ZERO, ZERO))} -> "
          f"{lam_str((ZERO, Fraction(1, 2**SEG1_HI_K)))}] for all six roots and splice "
          f"with segment 2 up to {lam_str((ZERO, Fraction(1, 2**SEG2_HI_K)))}\n")
    payload = build_certificate()
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    batch = payload["batch"]
    print(f"\nbatch: {batch['summary']} certified, splices {batch['splices_passed']}/6, "
          f"negative controls rejected {batch['negative_swapped_rejected']}/6")
    print(f"certificate written to {args.out}")
    print(f"re-check: python3 -m study.exp71_gate_c2_segment --verify {args.out}")
    return 0 if batch["n_inconclusive"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
