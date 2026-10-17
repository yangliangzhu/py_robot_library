"""exp72: Gate C3a -- certify the first complete archived edge of the a loop, six roots.

C3 first stage (``GPT_C3_EXECUTION_ADDENDUM_20261017.md``, section C3a): certify the
WHOLE first archived edge of the a loop (exp66 ``fold3_main``, from node 0 at ``u = 0``
to node 1 at ``u = i * 5.263157894736842e-4``, direction ``d = (0, -1, 0)`` frozen as
exact binary rationals) for all six common-base roots {1, 4, 6, 8, 10, 13}.

Method (all from the frozen exact model of exp69/exp70/exp71):

* adaptive rational subdivisions: each root gets its own uniform rational grid over
  the edge (endpoints ``k * lambda_edge / n``, exact), with local dyadic bisection
  when a tube fails; accepted and rejected attempts are both recorded;
* per-tube fixed-centre certification with the exact endpoint formula (the residual
  ``B f(c, lambda)`` is affine in ``lambda``; ``eta_i = max(|v_A[i]|, |v_B[i]|)`` is an
  exact whole-tube bound, with the exact midpoint identity stored);
* the addendum's optional exact improvement is used for the Jacobian bounds: the
  combined Laurent polynomials ``W_ij(z) = sum_k (B[i,2k] - i B[i,2k+1]) J[k,j](z)``
  are interval-evaluated directly (``(B Df)[i,2j] = Re W_ij``,
  ``(B Df)[i,2j+1] = -Im W_ij``), which is pointwise EXACTLY identical to the old
  formulation (stored cross-check) and tightens kappa by about 2.2x; a benchmark of
  both methods per root is stored;
* junction gluing uses the addendum's node boxes: an independently certified small
  endpoint box ``N`` at each junction, proved strictly inside BOTH adjacent tubes, so
  uniqueness in each tube identifies the lifted roots (no propagating enclosure);
  the C1 root box is proved strictly inside the first tube;
* policies are exactly C1/C2: tubes ``kappa < 1/2`` and margin ``>= r/4``, node boxes
  ``kappa < 1/2`` and margin ``>= r/2``, ``|trace S - 3| < 4`` and ``z_j != 0``
  everywhere; failures are recorded honestly, never loosened.

The six roots are certified in six parallel workers; the certificate stores every
exact number (dyadic values in compact hexadecimal ``n/2^e`` form), the adaptive-step
statistics, the interval-evaluation wall time, the certificate size, the minimum
step, and the limiting root.  ``--verify`` re-runs the repaired complete C1 check and
then recomputes every stored C3a number exactly.

Boundary: C3a only -- the d and h loops are untouched, no permutation is read or
claimed, and no ``path_certified`` flag exists.

Run::

    python3 -m study.exp72_gate_c3a_edge
    python3 -m study.exp72_gate_c3a_edge --verify study/exp72_gate_c3a_certificate.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from study.exp70_gate_c1_roots import (
    CERT_JSON as C1_CERT_JSON,
)
from study.exp70_gate_c1_roots import (
    HALF,
    MODEL_JSON,
    ONE,
    SEED_NPZ,
    TRACKED_INDICES,
    ZERO,
    XVector,
    build_system,
    check_model_provenance,
    contraction_bounds,
    exclusion_bounds,
    frac_str,
    fraction_matrix_det,
    inf_norm,
    lp_add,
    lp_eval_interval,
    lp_scale,
    lp_zero,
    manifest_links,
    matvec,
    parse_xvector,
    ri_abs_upper,
    ri_add,
    ri_point,
    ri_sub,
    system_fingerprint,
)
from study.exp70_gate_c1_roots import (
    verify_payload as verify_c1_payload,
)
from study.exp71_gate_c2_segment import (
    eval_f_lam,
    lam_mid,
    lam_str,
    newton_z,
    preconditioner_at,
)

C3A_CERT_JSON = Path("study/exp72_gate_c3a_certificate.json")


def load_a_direction() -> tuple[Fraction, Fraction, Fraction]:
    """The a-loop translation direction (exp66 fold3), frozen as exact binary rationals."""
    with np.load(A66_ARCHIVE, allow_pickle=False) as data:
        direction = np.asarray(data["fold3_direction"], dtype=float)
    exact = tuple(Fraction.from_float(float(value)) for value in direction)
    if exact != (ZERO, Fraction(-1), ZERO):
        raise ValueError(f"archived fold3 direction {exact} is not (0, -1, 0)")
    return exact  # type: ignore[return-value]
A66_ARCHIVE = Path("study/exp66_gate_a_generators.npz")
LOOP_KEY = "fold3_main"
TUBE_COMFORT = Fraction(1, 4)
NODE_COMFORT = HALF
RADIUS_MIN_K = 12
RADIUS_MAX_K = 26
NODE_K_EST_RANGE = 6
MAX_BISECT_DEPTH = 12

Lam = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------------
# compact exact serialization of dyadic rationals (B matrices and centres)
# ---------------------------------------------------------------------------


def frac_compact(value: Fraction) -> str:
    """Serialize a rational compactly as hexadecimal ``n/2^e*m`` (m odd, 1 omitted)."""
    if value == 0:
        return "0/1"
    numerator, denominator = value.numerator, value.denominator
    exponent = (denominator & -denominator).bit_length() - 1
    odd_part = denominator >> exponent
    suffix = f"2^{exponent}"
    if odd_part != 1:
        suffix += f"*{odd_part:x}"
    return f"{numerator:x}/{suffix}"


def parse_frac(value: str) -> Fraction:
    """Parse a compact or plain rational string."""
    if "/2^" in value:
        mantissa, rest = value.split("/2^")
        if "*" in rest:
            exponent, odd_hex = rest.split("*")
            denominator = 2 ** int(exponent) * int(odd_hex, 16)
        else:
            denominator = 2 ** int(rest)
        return Fraction(int(mantissa, 16), denominator)
    return Fraction(value)


def parse_x_compact(values: list[str]) -> XVector:
    """Parse a stored 12-vector written in compact form."""
    return tuple(parse_frac(value) for value in values)


def xvec_str(values: XVector) -> list[str]:
    """Compact serialization of a 12-rational vector."""
    return [frac_compact(value) for value in values]


def matrix_str(matrix: list[list[Fraction]]) -> list[list[str]]:
    """Compact serialization of a rational matrix."""
    return [[frac_compact(value) for value in row] for row in matrix]


# ---------------------------------------------------------------------------
# the combined-preconditioner Jacobian bounds (addendum optional improvement)
# ---------------------------------------------------------------------------


def build_w(jacobian: list, b_matrix: list[list[Fraction]]) -> list[list]:
    """``W[i][j] = sum_k (B[i,2k] - i B[i,2k+1]) J[k][j]`` as combined Laurent polys."""
    w_matrix = [[None for _ in range(6)] for _ in range(12)]
    for row in range(12):
        for joint in range(6):
            acc = lp_zero()
            for k in range(6):
                coefficient = (b_matrix[row][2 * k], -b_matrix[row][2 * k + 1])
                acc = lp_add(acc, lp_scale(jacobian[k][joint], coefficient))
            w_matrix[row][joint] = acc
    return w_matrix


def kappa_row_combined(
    w_matrix: list[list],
    centre: XVector,
    radius: Fraction,
    box: tuple | None = None,
) -> list[Fraction]:
    """Row bounds ``kappa_i >= sup_X ||row_i(I - B Df)||_1`` via the combined W entries.

    ``(B Df)[i,2j] = Re W_ij`` and ``(B Df)[i,2j+1] = -Im W_ij`` (an exact identity),
    so row ``i`` of ``I - B Df`` is bounded by interval evaluation of the combined
    polynomials, preserving the algebraic cancellations that the old two-step
    evaluation lost.
    """
    from study.exp70_gate_c1_roots import box_intervals

    box = box_intervals(centre, radius) if box is None else box
    cache: dict = {}
    rows = []
    for row in range(12):
        row_sum = ZERO
        for joint in range(6):
            value = lp_eval_interval(w_matrix[row][joint], box, cache)
            even = ri_sub(ri_point(ONE if row == 2 * joint else ZERO), value.re)
            odd = ri_add(ri_point(ONE if row == 2 * joint + 1 else ZERO), value.im)
            row_sum += ri_abs_upper(even) + ri_abs_upper(odd)
        rows.append(row_sum)
    return rows


def w_exact_point_crosscheck(
    jacobian: list,
    w_matrix: list[list],
    b_matrix: list[list[Fraction]],
    zs: tuple,
) -> bool:
    """Exact pointwise identity of the combined and the two-step formulations."""
    from study.exp70_gate_c1_roots import lp_eval

    for row in range(12):
        for joint in range(6):
            combined = lp_eval(w_matrix[row][joint], zs)
            re_sum = sum(b_matrix[row][2 * k] * lp_eval(jacobian[k][joint], zs)[0]
                         + b_matrix[row][2 * k + 1] * lp_eval(jacobian[k][joint], zs)[1]
                         for k in range(6))
            im_sum = sum(b_matrix[row][2 * k] * lp_eval(jacobian[k][joint], zs)[1]
                         - b_matrix[row][2 * k + 1] * lp_eval(jacobian[k][joint], zs)[0]
                         for k in range(6))
            if combined != (re_sum, im_sum):
                return False
    return True


# ---------------------------------------------------------------------------
# per-tube and node-box certification
# ---------------------------------------------------------------------------


def tube_record(
    lam_lo: Lam,
    lam_hi: Lam,
    centre: XVector,
    centre_source: str,
    b_matrix: list[list[Fraction]],
    w_matrix: list[list],
    equations: list,
    jacobian: list,
    s_matrix: list,
    direction: tuple[Fraction, Fraction, Fraction],
    radius: Fraction,
    interval_timer: list,
) -> tuple[dict, bool]:
    """One exact whole-tube certificate at a fixed radius (W-method kappa)."""
    start = time.perf_counter()
    v_lo = matvec(b_matrix, eval_f_lam(equations, centre, lam_lo, direction))
    v_hi = matvec(b_matrix, eval_f_lam(equations, centre, lam_hi, direction))
    eta_row = [max(abs(a), abs(b)) for a, b in zip(v_lo, v_hi)]
    midpoint = lam_mid(lam_lo, lam_hi)
    v_mid = matvec(b_matrix, eval_f_lam(equations, centre, midpoint, direction))
    affine_exact = all(2 * middle == lo + hi
                       for middle, lo, hi in zip(v_mid, v_lo, v_hi))
    kappa_row = kappa_row_combined(w_matrix, centre, radius)
    kappa = max(kappa_row)
    trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
    from study.exp70_gate_c1_roots import interval_df_matrix

    _, z_lower = interval_df_matrix(jacobian, centre, radius)
    margins = [radius - eta_row[row] - kappa_row[row] * radius for row in range(12)]
    passed = (kappa < HALF and min(margins) >= TUBE_COMFORT * radius
              and trace_bound < Fraction(4) and all(value > 0 for value in z_lower))
    interval_timer[0] += time.perf_counter() - start
    record = {
        "lam_lo": lam_str(lam_lo),
        "lam_hi": lam_str(lam_hi),
        "centre": xvec_str(centre),
        "centre_source": centre_source,
        "B": matrix_str(b_matrix),
        "B_det": frac_compact(fraction_matrix_det(b_matrix)),
        "eta": frac_compact(max(eta_row)),
        "affine_endpoint_identity_exact": affine_exact,
        "method": "W_combined",
        "radius_k": None,
        "radius": frac_compact(radius),
        "kappa": frac_compact(kappa),
        "margin": frac_compact(min(margins)),
        "margin_relative": frac_compact(min(margins) / radius),
        "row_margins_strict": all(value > 0 for value in margins),
        "trace_bound": frac_compact(trace_bound),
        "s_minus_i_inf_bound": frac_compact(s_bound),
        "z_modulus_squared_lower": [frac_compact(value) for value in z_lower],
        "verdict": "certified" if passed else "inconclusive",
        "accepted": passed,
    }
    record["radius_k"] = (radius.denominator.bit_length() - 1
                          if radius.numerator == 1 and radius.denominator & (
                              radius.denominator - 1) == 0 else None)
    return record, passed


def certify_node_box(
    equations: list,
    jacobian: list,
    s_matrix: list,
    direction: tuple[Fraction, Fraction, Fraction],
    lam: Lam,
    centre: XVector,
    b_matrix: list[list[Fraction]],
    interval_timer: list,
) -> tuple[dict, bool]:
    """Independently certified small endpoint box at one exact parameter value (C1 policy)."""
    eta = inf_norm(matvec(b_matrix, eval_f_lam(equations, centre, lam, direction)))
    det_b = fraction_matrix_det(b_matrix)
    start_k = (26 if eta == 0
               else min(46, max(NODE_K_EST_RANGE,
                                int(np.ceil(-np.log2(float(eta)))) - 1)))
    chosen = None
    attempts = []
    for k in range(start_k, max(start_k - NODE_K_EST_RANGE, 4), -1):
        start = time.perf_counter()
        radius = Fraction(1, 2**k)
        kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre, radius)
        trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
        interval_timer[0] += time.perf_counter() - start
        margins = [radius - eta - kappa_row[row] * radius for row in range(12)]
        passed = (kappa < HALF and min(margins) >= NODE_COMFORT * radius
                  and trace_bound < Fraction(4) and all(value > 0 for value in z_lower))
        attempts.append({"k": k, "radius": frac_str(radius), "kappa": frac_str(kappa),
                         "min_margin": frac_str(min(margins)), "pass": passed})
        if passed:
            chosen = (k, radius, kappa_row, kappa, z_lower, trace_bound, s_bound, margins)
            break
    certified = chosen is not None
    record = {
        "lambda": lam_str(lam),
        "centre": xvec_str(centre),
        "B": matrix_str(b_matrix),
        "B_det": frac_compact(det_b),
        "eta": frac_compact(eta),
    }
    if certified:
        k, radius, kappa_row, kappa, z_lower, trace_bound, s_bound, margins = chosen
        record.update({
            "radius_k": k,
            "radius": frac_compact(radius),
            "kappa": frac_compact(kappa),
            "margin": frac_compact(min(margins)),
            "row_margins_strict": all(value > 0 for value in margins),
            "trace_bound": frac_compact(trace_bound),
            "s_minus_i_inf_bound": frac_compact(s_bound),
            "z_modulus_squared_lower": [frac_compact(value) for value in z_lower],
        })
    if not certified:
        record["radius_search"] = attempts
    record["verdict"] = "certified" if certified else "inconclusive"
    return record, certified


def containment_margins(node_centre: XVector, node_radius: Fraction,
                        tube_centre: XVector, tube_radius: Fraction) -> list[Fraction]:
    """Per-coordinate strict margins of the node box inside the tube's interior."""
    return [tube_radius - abs(tube_centre[row] - node_centre[row]) - node_radius
            for row in range(12)]


# ---------------------------------------------------------------------------
# per-root chain worker
# ---------------------------------------------------------------------------


def plan_root(
    equations: list,
    jacobian: list,
    s_matrix: list,
    direction: tuple[Fraction, Fraction, Fraction],
    z0: np.ndarray,
    edge: Fraction,
) -> dict:
    """Float-based adaptive plan: per-row rates, W-kappa slopes, grid level, radius."""
    probe = min(edge, Fraction(1, 10**6))
    z_probe = newton_z(equations, jacobian, z0.copy(), (ZERO, probe), direction)
    rates = [abs(z_probe[j].real - z0[j].real) / float(probe)
             for j in range(6)] + [abs(z_probe[j].imag - z0[j].imag) / float(probe)
                                   for j in range(6)]
    centre0 = tuple(Fraction.from_float(part)
                    for z in z0 for part in (z.real, z.imag))
    b0 = preconditioner_at(equations, jacobian, centre0, direction, (ZERO, ZERO))
    w0 = build_w(jacobian, b0)
    best = None
    ladder = []
    for k in range(RADIUS_MIN_K, RADIUS_MAX_K + 1):
        radius = Fraction(1, 2**k)
        kappa_row = kappa_row_combined(w0, centre0, radius)
        kappa = max(kappa_row)
        lengths = [2 * radius * (Fraction(3, 4) - kappa_row[row]) / Fraction(rates[row])
                   for row in range(12) if rates[row] > 0]
        length = min(lengths) if lengths else None
        feasible = kappa < HALF and length is not None
        ladder.append({"k": k, "kappa": float(kappa), "step_est": float(length)
                       if length is not None else None, "feasible": feasible})
        if feasible and (best is None or length > best["step"]):
            best = {"k": k, "step": length}
    if best is None:
        return {"plan": None, "ladder": ladder, "rates": rates}
    step_est = best["step"]
    count = int(np.ceil(edge / step_est))
    step = edge / count
    refinements = 0
    for _ in range(4):
        z_first = newton_z(equations, jacobian, z0.copy(), (ZERO, step), direction)
        centre_first = tuple(Fraction.from_float(part)
                             for z in (z0 + z_first) / 2
                             for part in (z.real, z.imag))
        b_first = preconditioner_at(equations, jacobian, centre_first, direction,
                                    (ZERO, ZERO))
        w_first = build_w(jacobian, b_first)
        _, first_ok = tube_record((ZERO, ZERO), (ZERO, step), centre_first,
                                  "planner probe (discarded)", b_first, w_first,
                                  equations, jacobian, s_matrix, direction,
                                  Fraction(1, 2**best["k"]), [0.0])
        if first_ok:
            break
        count *= 2
        step = edge / count
        refinements += 1
    # drift probe: W-kappa at the far end centre at the chosen radius
    z_end = z0
    chunks = 8
    for chunk in range(1, chunks + 1):
        z_end = newton_z(equations, jacobian, z_end.copy(),
                         (ZERO, edge * chunk / chunks), direction)
    centre_end = tuple(Fraction.from_float(part)
                       for z in z_end for part in (z.real, z.imag))
    b_end = preconditioner_at(equations, jacobian, centre_end, direction, (ZERO, edge))
    w_end = build_w(jacobian, b_end)
    radius = Fraction(1, 2**best["k"])
    kappa_end = max(kappa_row_combined(w_end, centre_end, radius))
    return {"plan": {"k": best["k"], "radius": frac_str(radius), "count": count,
                     "step": lam_str((ZERO, step))},
            "ladder": ladder, "rates": rates,
            "drift_probe": {"kappa_base": frac_str(max(kappa_row_combined(
                w0, centre0, radius))), "kappa_end": frac_str(kappa_end)},
            "plan_refinements": refinements}



def root_planner(payload: dict) -> dict:
    """Plan one root's grid and compute the float Newton continuation over it."""
    root_index = payload["root_index"]
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed in planner")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_a_direction()
    with np.load(SEED_NPZ, allow_pickle=False) as data:
        fibre = data["fibre_real"] + 1j * data["fibre_imag"]
    with np.load(A66_ARCHIVE, allow_pickle=False) as data:
        archived_roots = (np.asarray(data[f"{LOOP_KEY}_roots_real"], dtype=float)
                          + 1j * np.asarray(data[f"{LOOP_KEY}_roots_imag"], dtype=float))
    edge = payload["edge"]
    limit = payload.get("limit")
    q0 = fibre[root_index]
    z0 = np.exp(1j * q0)
    print(f"    [root {root_index}] planning ...", flush=True)
    plan = plan_root(equations, jacobian, s_matrix, direction, z0, edge)
    if plan["plan"] is None:
        return {"root_index": root_index, "plan": plan, "grid": None,
                "archived_node1_gap_float": None}
    count = plan["plan"]["count"]
    piece_count = count if limit is None else min(count, limit)
    step = edge / count
    grid = [z0]
    z_walk = z0
    for sub in range(piece_count):
        z_walk = newton_z(equations, jacobian, z_walk.copy(),
                          (ZERO, step * (sub + 1)), direction)
        grid.append(z_walk)
    # archived node-1 sanity (informational float, z-space)
    if limit is None:
        archived_end_z = np.exp(1j * archived_roots[1][root_index])
        gap = float(np.max(np.abs(z_walk - archived_end_z)))
    else:
        gap = None
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        slice_path = Path(checkpoint) / f"planner_{root_index}.json"
        slice_path.write_text(json.dumps({
            "root_index": root_index, "plan": plan, "piece_count": piece_count,
            "archived_node1_gap_float": gap,
            "grid": [[[float(part.real), float(part.imag)] for part in point]
                     for point in grid]}) + "\n")
        print(f"    [root {root_index}] planner slice -> {slice_path}", flush=True)
    return {"root_index": root_index, "plan": plan, "grid": grid,
            "piece_count": piece_count, "archived_node1_gap_float": gap}


def certify_chunk(payload: dict) -> dict:
    """Certify a slice of one root's uniform grid: tubes, node boxes, stats."""
    start_index = payload["start_index"]
    grid = payload["grid"]
    step = payload["step"]
    k_root = payload["k_root"]
    radius = Fraction(1, 2**k_root)
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed in chunk worker")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_a_direction()
    interval_timer = [0.0]
    tubes = []
    node_boxes = []
    accepted = 0
    rejected = 0

    def certify_recursive(lam_lo: Fraction, lam_hi: Fraction, z_lo: np.ndarray,
                          z_hi: np.ndarray, depth: int) -> list:
        nonlocal accepted, rejected
        centre = tuple(Fraction.from_float(part)
                       for z in (z_lo + z_hi) / 2 for part in (z.real, z.imag))
        b_matrix = preconditioner_at(equations, jacobian, centre, direction,
                                     (ZERO, lam_lo))
        w_matrix = build_w(jacobian, b_matrix)
        lam_lo_c, lam_hi_c = (ZERO, lam_lo), (ZERO, lam_hi)
        record, ok = tube_record(lam_lo_c, lam_hi_c, centre,
                                 "frozen midpoint of the two endpoint Newton roots",
                                 b_matrix, w_matrix, equations, jacobian, s_matrix,
                                 direction, radius, interval_timer)
        if ok:
            accepted += 1
            return [record]
        for fallback_k in (k_root + 1, k_root + 2):
            smaller = Fraction(1, 2**fallback_k)
            record2, ok2 = tube_record(lam_lo_c, lam_hi_c, centre,
                                       "frozen midpoint of the two endpoint Newton roots",
                                       b_matrix, w_matrix, equations, jacobian, s_matrix,
                                       direction, smaller, interval_timer)
            if ok2:
                accepted += 1
                return [record2]
        rejected += 1
        failure = {"lam_lo": lam_str(lam_lo_c), "lam_hi": lam_str(lam_hi_c),
                   "accepted": False, "centre": xvec_str(centre),
                   "B": matrix_str(b_matrix), "eta": record["eta"],
                   "failure": {"reason": (f"kappa {record['kappa']} >= 1/2"
                                          if parse_frac(record["kappa"]) >= HALF
                                          else f"margin {record['margin']} < r/4"),
                               "best_radius_k": record["radius_k"],
                               "best_kappa": record["kappa"],
                               "best_margin": record["margin"]}}
        if depth >= MAX_BISECT_DEPTH:
            failure["failure"]["reason"] = ("bisection depth limit reached; recorded "
                                            "inconclusive, not loosened")
            return [failure]
        midpoint = (lam_lo + lam_hi) / 2
        z_mid = newton_z(equations, jacobian, z_lo.copy(), (ZERO, midpoint), direction)
        left = certify_recursive(lam_lo, midpoint, z_lo, z_mid, depth + 1)
        right = certify_recursive(midpoint, lam_hi, z_mid, z_hi, depth + 1)
        return [failure] + left + right

    for position in range(len(grid) - 1):
        z_lo = grid[position]
        z_hi = grid[position + 1]
        lam_lo = step * (start_index + position)
        lam_hi = step * (start_index + position + 1)
        for piece in certify_recursive(lam_lo, lam_hi, z_lo, z_hi, 0):
            if piece.get("accepted") is False:
                tubes.append(piece)
                continue
            tubes.append(piece)
            centre_n = tuple(Fraction.from_float(part)
                             for z in newton_z(equations, jacobian, z_lo.copy(),
                                               (ZERO, Fraction(piece["lam_hi"][1])),
                                               direction)
                             for part in (z.real, z.imag))
            b_n = preconditioner_at(equations, jacobian, centre_n, direction,
                                    (ZERO, Fraction(piece["lam_hi"][1])))
            node_record, node_ok = certify_node_box(
                equations, jacobian, s_matrix, direction,
                (ZERO, Fraction(piece["lam_hi"][1])), centre_n, b_n, interval_timer)
            node_record["in_prev_margins"] = None
            node_record["in_next_margins"] = None
            if node_ok:
                node_record["in_prev_margins"] = [frac_compact(value) for value in
                                                  containment_margins(
                                                      parse_x_compact(node_record["centre"]),
                                                      parse_frac(node_record["radius"]),
                                                      parse_x_compact(piece["centre"]),
                                                      parse_frac(piece["radius"]))]
            node_boxes.append(node_record)
    result = {"root_index": payload["root_index"], "tubes": tubes,
              "node_boxes": node_boxes,
              "stats": {"n_accepted": accepted, "n_rejected": rejected,
                        "interval_eval_seconds": interval_timer[0]}}
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        path = Path(checkpoint) / f"chunk_{payload['root_index']}_{start_index}.json"
        path.write_text(json.dumps(result) + "\n")
        print(f"    [root {payload['root_index']}] chunk @{start_index} done: "
              f"{accepted} accepted, {rejected} rejected -> {path}", flush=True)
    return result


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_edge() -> dict:
    """Freeze the a-loop first archived node interval from the exp66 archive."""
    with np.load(A66_ARCHIVE, allow_pickle=False) as data:
        u_points = (np.asarray(data[f"{LOOP_KEY}_u_real"], dtype=float)
                    + 1j * np.asarray(data[f"{LOOP_KEY}_u_imag"], dtype=float))
    u0, u1 = u_points[0], u_points[1]
    if u0 != 0 or u1.real != 0 or u1.imag <= 0:
        raise RuntimeError("archived fold3_main first edge is not the imaginary-axis line")
    return {"archive": str(A66_ARCHIVE), "loop": LOOP_KEY,
            "node0": ["0/1", "0/1"],
            "node1": [frac_compact(Fraction.from_float(u1.real)),
                      frac_compact(Fraction.from_float(u1.imag))],
            "edge_im": frac_compact(Fraction.from_float(u1.imag)),
            "line_note": ("the archived first leg runs from u=0 up the imaginary axis "
                          "(pose p0 + u*d, d=(0,-1,0)); the certified edge uses "
                          "lambda = i*s on the same line")}


def assemble_payload(planner_results: list[dict], chunk_results: list[dict],
                     started: float) -> dict:
    """Merge the per-root plans and chunk certificates into the payload with stats."""
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_a_direction()
    edge_info = archived_edge()
    planners = {item["root_index"]: item for item in planner_results}
    roots: dict[str, dict] = {}
    for root_index in TRACKED_INDICES:
        planner = planners[root_index]
        chunks = [item for item in chunk_results if item["root_index"] == root_index]
        tubes = [tube for chunk in chunks for tube in chunk["tubes"]]
        node_boxes = [node for chunk in chunks for node in chunk["node_boxes"]]
        accepted_chain = [item for item in tubes if item.get("accepted") is not False]
        edge = parse_frac(edge_info["edge_im"])
        # complete the in_next containment margins along the accepted chain
        for number, node in enumerate(node_boxes):
            if node["verdict"] != "certified" or number + 1 >= len(accepted_chain):
                continue
            next_centre = parse_x_compact(accepted_chain[number + 1]["centre"])
            next_radius = parse_frac(accepted_chain[number + 1]["radius"])
            node["in_next_margins"] = [frac_compact(value) for value in containment_margins(
                parse_x_compact(node["centre"]), parse_frac(node["radius"]),
                next_centre, next_radius)]
        covers = (bool(accepted_chain)
                  and accepted_chain[0]["lam_lo"] == ["0/1", "0/1"]
                  and accepted_chain[-1]["lam_hi"] == lam_str((ZERO, edge))
                  and all(accepted_chain[i]["lam_hi"] == accepted_chain[i + 1]["lam_lo"]
                          for i in range(len(accepted_chain) - 1)))
        c1_stored = json.loads(C1_CERT_JSON.read_text())
        c1_centre = parse_xvector(c1_stored["roots"][str(root_index)]["centre"])
        c1_radius = Fraction(c1_stored["roots"][str(root_index)]["radius"])
        c1_margins = []
        if accepted_chain:
            first_centre = parse_x_compact(accepted_chain[0]["centre"])
            first_radius = parse_frac(accepted_chain[0]["radius"])
            c1_margins = containment_margins(c1_centre, c1_radius, first_centre,
                                             first_radius)
        n_accepted = sum(item["stats"]["n_accepted"] for item in chunks)
        n_rejected = sum(item["stats"]["n_rejected"] for item in chunks)
        interval_seconds = sum(item["stats"]["interval_eval_seconds"] for item in chunks)
        nodes_ok = all(item["verdict"] == "certified" for item in node_boxes)
        ok = (covers and nodes_ok and planner["piece_count"] == len(planner["grid"]) - 1
              and all(item.get("accepted") is not False for item in tubes)
              and all(value > 0 for value in c1_margins)
              and planner["plan"]["plan"] is not None)
        with np.load(SEED_NPZ, allow_pickle=False) as data:
            fibre = data["fibre_real"] + 1j * data["fibre_imag"]
        roots[str(root_index)] = {
            "seed_q": [frac_compact(value) for value in
                       (Fraction.from_float(part)
                        for z in fibre[root_index] for part in (z.real, z.imag))],
            "plan": planner["plan"],
            "subdivisions": tubes,
            "node_boxes": node_boxes,
            "c1_box_inclusion": {
                "c1_centre": xvec_str(c1_centre),
                "c1_radius": frac_str(c1_radius),
                "margins": [frac_compact(value) for value in c1_margins],
                "passed": all(value > 0 for value in c1_margins),
            },
            "archived_node1_gap_float": planner["archived_node1_gap_float"],
            "stats": {"n_accepted": n_accepted, "n_rejected": n_rejected,
                      "min_step": frac_str(min(
                          (Fraction(item["lam_hi"][1]) - Fraction(item["lam_lo"][1])
                           for item in accepted_chain), default=ZERO)),
                      "interval_eval_seconds": interval_seconds,
                      "wall_seconds": None},
            "verdict": "certified" if ok else "inconclusive",
        }
    certified = [index for index, record in roots.items()
                 if record["verdict"] == "certified"]
    n_accepted = sum(record["stats"]["n_accepted"] for record in roots.values())
    n_rejected = sum(record["stats"]["n_rejected"] for record in roots.values())
    min_steps = [Fraction(record["stats"]["min_step"]) for record in roots.values()
                 if record["stats"]["min_step"] != "0/1"]
    total_interval = sum(record["stats"]["interval_eval_seconds"]
                         for record in roots.values())
    total_wall = time.perf_counter() - started
    payload = {
        "experiment": "exp72_gate_c3a_edge",
        "task": ("Gate C, stage C3a: certify the first COMPLETE archived edge of the a loop "
                 "(exp66 fold3_main, node 0 -> node 1) for the six common-base roots, with "
                 "adaptive rational subdivisions, the combined-preconditioner W interval "
                 "bounds, and node-box junction gluing"),
        "spec": ("study/collab/GPT_C3_EXECUTION_ADDENDUM_20261017.md section C3a, with "
                 "GPT_GATE_C_SPEC_20261017.md sections 4-6 and the C2 machinery of exp71"),
        "boundary": ("C3a only: one archived edge; the d and h loops are untouched, no "
                     "loop permutation is read or claimed, no certification flags beyond "
                     "the per-root chains"),
        "backend": {
            "name": "exact rational interval arithmetic over Python fractions.Fraction "
                    "endpoints (complex rectangular intervals), as in exp70/exp71",
            "combined_preconditioner": ("W_ij(z) = sum_k (B[i,2k] - i B[i,2k+1]) J[k,j](z) "
                                        "evaluated as one Laurent polynomial; "
                                        "(B Df)[i,2j] = Re W_ij, (B Df)[i,2j+1] = -Im W_ij "
                                        "-- an exact identity (pointwise cross-check stored "
                                        "per root), used because the two-step evaluation "
                                        "loses the algebraic cancellations"),
            "junction_gluing": ("independently certified small node boxes N at each exact "
                                "endpoint, proved strictly inside both adjacent tubes; "
                                "uniqueness in each tube identifies the lifted roots "
                                "(addendum rule; no propagating endpoint enclosure)"),
            "policies": ("tubes: kappa < 1/2 and per-row margin >= r/4 (C2 policy); node "
                         "boxes: kappa < 1/2 and margin >= r/2 (C1 policy); |trace S - 3| "
                         "< 4 and z_j != 0 everywhere; failures recorded, never loosened"),
            "floating_inputs": [
                "tube/node centres frozen as exact binary rationals from float Newton "
                "seeds (box selection only)",
                "B = numpy double inverse of the float Jacobian at the centre, frozen as "
                "exact binary rationals",
            ],
        },
        "coefficient_version": {
            "model_manifest": str(MODEL_JSON),
            "model_manifest_sha256": _sha256(MODEL_JSON),
            "source_sha256": manifest["source_sha256"],
            "a_loop_archive_sha256": _sha256(A66_ARCHIVE),
            "system_fingerprint": system_fingerprint(equations, jacobian, s_matrix),
            "c1_certificate": str(C1_CERT_JSON),
            "c1_certificate_sha256": _sha256(C1_CERT_JSON),
        },
        "direction": [frac_str(value) for value in direction],
        "edge": edge_info,
        "roots": roots,
        "batch": {
            "n_certified": len(certified),
            "n_inconclusive": len(TRACKED_INDICES) - len(certified),
            "verdicts": {index: roots[index]["verdict"] for index in roots},
            "summary": f"{len(certified)}/{len(roots)}",
        },
        "stats": {
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "min_step": frac_str(min(min_steps)) if min_steps else "0/1",
            "limiting_root": sorted(roots, key=lambda index: roots[index]["stats"][
                "n_accepted"], reverse=True)[0],
            "interval_eval_seconds": total_interval,
            "total_wall_seconds": total_wall,
            "certificate_bytes": None,
        },
        "verification_command": ("python3 -m study.exp72_gate_c3a_edge --verify "
                                 "study/exp72_gate_c3a_certificate.json"),
    }
    payload["stats"]["certificate_bytes"] = len(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_chunk(payload: dict) -> list[tuple[str, bool, str]]:
    """Recompute one slice of a root's stored tubes and node boxes exactly."""
    root_index = payload["root_index"]
    tubes = payload["tubes"]
    node_boxes = payload["node_boxes"]
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed in verify chunk")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_a_direction()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((f"root {root_index}: {name}", passed, detail))

    for tube in tubes:
        centre = parse_x_compact(tube["centre"])
        b_matrix = [[parse_frac(value) for value in row] for row in tube["B"]]
        lam_lo = (Fraction(tube["lam_lo"][0]), Fraction(tube["lam_lo"][1]))
        lam_hi = (Fraction(tube["lam_hi"][0]), Fraction(tube["lam_hi"][1]))
        w_matrix = build_w(jacobian, b_matrix)
        radius = parse_frac(tube["radius"])
        tag = f"tube [{tube['lam_lo'][1][:12]}..{tube['lam_hi'][1][:12]}]"
        det_b = fraction_matrix_det(b_matrix)
        check(f"{tag}: det B != 0 and matches stored",
              det_b != 0 and frac_compact(det_b) == tube["B_det"], "")
        v_lo = matvec(b_matrix, eval_f_lam(equations, centre, lam_lo, direction))
        v_hi = matvec(b_matrix, eval_f_lam(equations, centre, lam_hi, direction))
        eta_row = [max(abs(a), abs(b)) for a, b in zip(v_lo, v_hi)]
        v_mid = matvec(b_matrix, eval_f_lam(equations, centre,
                                            lam_mid(lam_lo, lam_hi), direction))
        affine = all(2 * middle == lo + hi
                     for middle, lo, hi in zip(v_mid, v_lo, v_hi))
        check(f"{tag}: eta/affine-identity match stored",
              frac_compact(max(eta_row)) == tube["eta"]
              and affine and tube["affine_endpoint_identity_exact"], "")
        try:
            kappa_row = kappa_row_combined(w_matrix, centre, radius)
            trace_bound, s_bound = exclusion_bounds(s_matrix, centre, radius)
            from study.exp70_gate_c1_roots import interval_df_matrix
            _, z_lower = interval_df_matrix(jacobian, centre, radius)
        except ValueError:
            check(f"{tag}: box avoids z_j = 0", False, "division rejected")
            continue
        margins = [radius - eta_row[row] - kappa_row[row] * radius for row in range(12)]
        check(f"{tag}: kappa/margin scalars and row-strictness match stored",
              frac_compact(max(kappa_row)) == tube["kappa"]
              and frac_compact(min(margins)) == tube["margin"]
              and frac_compact(min(margins) / radius) == tube["margin_relative"]
              and (all(value > 0 for value in margins)) == tube["row_margins_strict"], "")
        check(f"{tag}: trace/z/S-I bounds match stored and strict",
              frac_compact(trace_bound) == tube["trace_bound"]
              and trace_bound < Fraction(4)
              and frac_compact(s_bound) == tube["s_minus_i_inf_bound"]
              and [frac_compact(value) for value in z_lower] == tube["z_modulus_squared_lower"]
              and all(value > 0 for value in z_lower), "")
        check(f"{tag}: strict whole-tube contraction re-checked (kappa < 1/2, "
              f"margin >= r/4)",
              max(kappa_row) < HALF and min(margins) >= TUBE_COMFORT * radius,
              f"kappa {frac_compact(max(kappa_row))}, margin {frac_compact(min(margins))}")
        check(f"{tag}: W formulation pointwise identical (exact)",
              w_exact_point_crosscheck(
                  jacobian, w_matrix, b_matrix,
                  tuple((centre[2 * j], centre[2 * j + 1]) for j in range(6))), "")

    for number, node in enumerate(node_boxes):
        centre = parse_x_compact(node["centre"])
        b_matrix = [[parse_frac(value) for value in row] for row in node["B"]]
        lam = (Fraction(node["lambda"][0]), Fraction(node["lambda"][1]))
        node_radius = parse_frac(node["radius"])
        tag = f"node [{node['lambda'][1][:12]}]"
        eta = inf_norm(matvec(b_matrix, eval_f_lam(equations, centre, lam, direction)))
        check(f"{tag}: eta matches stored",
              frac_compact(eta) == node["eta"], frac_compact(eta))
        try:
            kappa_row, kappa, z_lower = contraction_bounds(jacobian, b_matrix, centre,
                                                           node_radius)
            trace_bound, s_bound = exclusion_bounds(s_matrix, centre, node_radius)
        except ValueError:
            check(f"{tag}: box avoids z_j = 0", False, "division rejected")
            continue
        margins = [node_radius - eta - kappa_row[row] * node_radius for row in range(12)]
        check(f"{tag}: kappa/margin scalars and row-strictness match stored "
              f"(C1 policy: kappa < 1/2, margin >= r/2)",
              frac_compact(kappa) == node["kappa"]
              and frac_compact(min(margins)) == node["margin"]
              and (all(value > 0 for value in margins)) == node["row_margins_strict"]
              and kappa < HALF and min(margins) >= NODE_COMFORT * node_radius, "")
        check(f"{tag}: trace/z/S-I bounds match stored",
              frac_compact(trace_bound) == node["trace_bound"]
              and frac_compact(s_bound) == node["s_minus_i_inf_bound"]
              and [frac_compact(value) for value in z_lower] == node["z_modulus_squared_lower"]
              and all(value > 0 for value in z_lower), "")
        if node["in_prev_margins"] is not None and number < len(tubes):
            tube_centre = parse_x_compact(tubes[number]["centre"])
            tube_radius = parse_frac(tubes[number]["radius"])
            prev_margins = containment_margins(centre, node_radius, tube_centre,
                                               tube_radius)
            check(f"{tag}: strictly inside the preceding tube",
                  [frac_compact(value) for value in prev_margins] == node["in_prev_margins"]
                  and all(value > 0 for value in prev_margins), "")
    return checks



def verify_rejected_chunk(payload: dict) -> list[tuple[str, bool, str]]:
    """Re-verify the recorded rejected attempts (they must still fail, never loosened)."""
    root_index = payload["root_index"]
    rejected = payload["rejected"]
    manifest, provenance = check_model_provenance()
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed in rejected chunk")
    links = manifest_links(manifest)
    r_star = [[Fraction(value) for value in row] for row in manifest["target_rotation"]]
    p0 = [Fraction(value) for value in manifest["target_position"]]
    equations, jacobian, s_matrix = build_system(links, r_star, p0)
    direction = load_a_direction()
    checks: list[tuple[str, bool, str]] = []
    for number, item in enumerate(rejected):
        centre = parse_x_compact(item["centre"])
        b_matrix = [[parse_frac(value) for value in row] for row in item["B"]]
        w_matrix = build_w(jacobian, b_matrix)
        lam_lo = (Fraction(item["lam_lo"][0]), Fraction(item["lam_lo"][1]))
        lam_hi = (Fraction(item["lam_hi"][0]), Fraction(item["lam_hi"][1]))
        v_lo = matvec(b_matrix, eval_f_lam(equations, centre, lam_lo, direction))
        v_hi = matvec(b_matrix, eval_f_lam(equations, centre, lam_hi, direction))
        eta_row = [max(abs(a), abs(b)) for a, b in zip(v_lo, v_hi)]
        best_k = int(item["failure"]["best_radius_k"])
        best_radius = Fraction(1, 2**best_k)
        kappa_row = kappa_row_combined(w_matrix, centre, best_radius)
        margins = [best_radius - eta_row[row] - kappa_row[row] * best_radius
                   for row in range(12)]
        still_fails = (max(kappa_row) >= HALF
                       or min(margins) < TUBE_COMFORT * best_radius)
        checks.append((f"root {root_index}: rejected {number}: recorded failure re-verified",
                       still_fails
                       and frac_compact(max(eta_row)) == item["eta"]
                       and frac_compact(max(kappa_row)) == item["failure"]["best_kappa"]
                       and frac_compact(min(margins)) == item["failure"]["best_margin"],
                       item["failure"]["reason"]))
    return checks


def fast_structural_checks(stored: dict, report) -> bool:
    """Cheap completeness checks that run before the expensive recomputation."""
    ok = True
    roots = stored.get("roots")
    if not isinstance(roots, dict) or sorted(roots.keys()) != sorted(map(str, TRACKED_INDICES)):
        report("roots keys exactly {1,4,6,8,10,13}", False,
               f"got {sorted(roots.keys()) if isinstance(roots, dict) else type(roots)}")
        return False
    report("roots keys exactly {1,4,6,8,10,13}", True, "")
    direction = load_a_direction()
    report("direction matches the frozen a-loop archive",
           stored["direction"] == [frac_str(value) for value in direction], "")
    edge_info = archived_edge()
    report("edge geometry matches the archived node interval",
           stored["edge"] == edge_info, "")
    edge = parse_frac(stored["edge"]["edge_im"])
    for index in TRACKED_INDICES:
        record = stored["roots"][str(index)]
        accepted = [item for item in record["subdivisions"]
                    if item.get("accepted") is not False]
        covers = (bool(accepted)
                  and accepted[0]["lam_lo"] == ["0/1", "0/1"]
                  and accepted[-1]["lam_hi"] == lam_str((ZERO, edge))
                  and all(accepted[i]["lam_hi"] == accepted[i + 1]["lam_lo"]
                          for i in range(len(accepted) - 1)))
        report(f"root {index}: accepted chain covers [0, lambda_edge] exactly", covers,
               f"{len(accepted)} pieces")
        ok = ok and covers
        if record.get("verdict") != "certified":
            report(f"root {index}: verdict is certified", False, record.get("verdict"))
            ok = False
        for tube in accepted:
            centre = [parse_frac(value) for value in tube["centre"]]
            radius = parse_frac(tube["radius"])
            safe = all(centre[2 * j] * centre[2 * j] + centre[2 * j + 1] * centre[2 * j + 1]
                       > radius * radius for j in range(6))
            if not safe:
                report(f"root {index}: tube box avoids z_j = 0 (fast check)", False, "")
                ok = False
            if radius != Fraction(1, 2**int(tube["radius_k"])):
                report(f"root {index}: radius = 2^-k consistent", False, "")
                ok = False
    return ok


def verify_payload_c3a(stored: dict) -> int:
    """Full C3a re-check: fast structural checks, then exact parallel recomputation."""
    all_ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = all_ok and passed
        short = detail if len(detail) <= 60 else detail[:30] + "..." + detail[-30:]
        suffix = f"  ({short})" if detail else ""
        print(f"    {'PASS' if passed else 'FAIL'}  {name}{suffix}")

    print("verify: full C3a payload")
    if not fast_structural_checks(stored, report):
        print("FAIL: verification failed (structural checks)")
        return 1
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
           _sha256(MODEL_JSON) == stored["coefficient_version"]["model_manifest_sha256"])
    report("C1 certificate sha256 matches stored",
           _sha256(C1_CERT_JSON) == stored["coefficient_version"]["c1_certificate_sha256"])
    report("a-loop archive sha256 matches stored",
           _sha256(A66_ARCHIVE) == stored["coefficient_version"]["a_loop_archive_sha256"])
    edge = parse_frac(stored["edge"]["edge_im"])
    c1_stored = json.loads(C1_CERT_JSON.read_text())

    chunk_payloads = []
    rejected_payloads = []
    for index in TRACKED_INDICES:
        record = stored["roots"][str(index)]
        plan = record["plan"]["plan"]
        if plan is None:
            report(f"root {index}: plan exists", False, "None")
            continue
        report(f"root {index}: plan radius/step/count consistent",
               Fraction(plan["radius"]) == Fraction(1, 2**int(plan["k"]))
               and (ZERO, edge / int(plan["count"]))
               == (Fraction(plan["step"][0]), Fraction(plan["step"][1])),
               f"k={plan['k']}, count={plan['count']}")
        accepted = [item for item in record["subdivisions"]
                    if item.get("accepted") is not False]
        rejected = [item for item in record["subdivisions"]
                    if item.get("accepted") is False]
        node_boxes = record["node_boxes"]
        size = 400
        for start in range(0, len(accepted), size):
            chunk_payloads.append({
                "root_index": index,
                "tubes": accepted[start:start + size],
                "node_boxes": node_boxes[start:start + size],
            })
        for start in range(0, len(rejected), size):
            rejected_payloads.append({
                "root_index": index,
                "rejected": rejected[start:start + size],
            })
        # cheap parent-side checks: in_next containments, C1 inclusion
        for number, node in enumerate(node_boxes):
            if node["verdict"] != "certified":
                continue
            if number + 1 < len(accepted):
                next_centre = parse_x_compact(accepted[number + 1]["centre"])
                next_radius = parse_frac(accepted[number + 1]["radius"])
                next_margins = containment_margins(
                    parse_x_compact(node["centre"]), parse_frac(node["radius"]),
                    next_centre, next_radius)
                report(f"root {index}: node {number} strictly inside the following tube",
                       [frac_compact(value) for value in next_margins]
                       == node["in_next_margins"]
                       and all(value > 0 for value in next_margins), "")
            else:
                report(f"root {index}: final node defers in_next to C3b",
                       node["in_next_margins"] is None, "")
        c1_centre = parse_xvector(c1_stored["roots"][str(index)]["centre"])
        c1_radius = Fraction(c1_stored["roots"][str(index)]["radius"])
        if accepted:
            first_centre = parse_x_compact(accepted[0]["centre"])
            first_radius = parse_frac(accepted[0]["radius"])
            c1_margins = containment_margins(c1_centre, c1_radius, first_centre,
                                             first_radius)
            report(f"root {index}: C1 root box strictly inside the first tube",
                   record["c1_box_inclusion"]["passed"]
                   and [frac_compact(value) for value in c1_margins]
                   == record["c1_box_inclusion"]["margins"]
                   and all(value > 0 for value in c1_margins), "")

    with mp.Pool(12) as pool:
        results = pool.map(verify_chunk, chunk_payloads)
    for name, passed, detail in sorted((entry for group in results for entry in group),
                                       key=lambda entry: entry[0]):
        report(name, passed, detail)
    if rejected_payloads:
        with mp.Pool(min(6, len(rejected_payloads))) as pool:
            rejected_results = pool.map(verify_rejected_chunk, rejected_payloads)
        for name, passed, detail in sorted((entry for group in rejected_results
                                            for entry in group), key=lambda entry: entry[0]):
            report(name, passed, detail)
    batch = stored["batch"]
    report("batch summary consistent",
           batch["n_certified"] == len(TRACKED_INDICES)
           and batch["n_inconclusive"] == 0
           and batch["summary"] == f"{len(TRACKED_INDICES)}/{len(TRACKED_INDICES)}",
           batch["summary"])
    print("PASS: C3a certificate fully re-checked" if all_ok else "FAIL: verification failed")
    return 0 if all_ok else 1


def _self_test_c3a(stored: dict) -> None:
    """Negative regressions: tampered payloads must be rejected fast."""
    import contextlib
    import copy as _copy
    import io

    def rejected(name: str, payload: dict) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = verify_payload_c3a(payload)
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
    gap = _copy.deepcopy(stored)
    tubes = [item for item in gap["roots"]["1"]["subdivisions"]
             if item.get("accepted") is not False]
    if len(tubes) > 2:
        del gap["roots"]["1"]["subdivisions"][1]
        rejected("coverage gap (dropped tube)", gap)
    z_zero = _copy.deepcopy(stored)
    first_tube = next(item for item in z_zero["roots"]["13"]["subdivisions"]
                      if item.get("accepted") is not False)
    first_tube["centre"][0] = "0/1"
    first_tube["centre"][1] = "0/1"
    rejected("tube containing z_j = 0", z_zero)
    tampered_direction = _copy.deepcopy(stored)
    tampered_direction["direction"] = ["0/1", "-1/1", "1/2"]
    rejected("tampered direction", tampered_direction)
    tampered_model = _copy.deepcopy(stored)
    tampered_model["coefficient_version"]["model_manifest_sha256"] = "0/1"
    rejected("tampered model sha256", tampered_model)
    bad_radius = _copy.deepcopy(stored)
    bad_radius["roots"]["1"]["subdivisions"][0]["radius_k"] = (
        int(bad_radius["roots"]["1"]["subdivisions"][0]["radius_k"]) + 1)
    rejected("tampered radius_k", bad_radius)


def verify_certificate(path: Path) -> int:
    """Full C3a acceptance: repaired C1 check first, then the exact C3a recomputation."""
    stored = json.loads(path.read_text())
    print("C1 prerequisite: re-running the repaired complete C1 check")
    c1_code = verify_c1_payload(json.loads(C1_CERT_JSON.read_text()),
                                list(TRACKED_INDICES), f"full C1 ({C1_CERT_JSON})")
    if c1_code != 0:
        print("FAIL: C1 prerequisite failed -- C3a acceptance refused")
        return 1
    print()
    print("verifier negative self-tests:")
    _self_test_c3a(stored)
    print()
    return verify_payload_c3a(stored)


def merge_certificate(checkpoint_dir: Path, out: Path) -> int:
    """Merge the six planner slices and chunk files into the final certificate."""
    started = time.perf_counter()
    planner_results = []
    chunk_results = []
    missing = []
    for index in TRACKED_INDICES:
        planner_path = checkpoint_dir / f"planner_{index}.json"
        if not planner_path.exists():
            missing.append(f"planner_{index}")
            continue
        stored = json.loads(planner_path.read_text())
        stored.pop("grid", None)
        planner_results.append(stored)
    for path in sorted(checkpoint_dir.glob("chunk_*.json")):
        chunk_results.append(json.loads(path.read_text()))
    if missing:
        print(f"FAIL: merge refused -- missing {missing}")
        return 1
    payload = assemble_payload(planner_results, chunk_results, started)
    text = json.dumps(payload, indent=2) + "\n"
    payload["stats"]["certificate_bytes"] = len(text)
    out.write_text(text)
    batch = payload["batch"]
    stats = payload["stats"]
    print(f"merge: {batch['summary']} certified | accepted pieces {stats['n_accepted']} "
          f"rejected {stats['n_rejected']} | min step {stats['min_step']} | "
          f"limiting root {stats['limiting_root']}")
    print(f"interval evaluation {stats['interval_eval_seconds']:.0f}s | certificate "
          f"{stats['certificate_bytes']} bytes -> {out}")
    return 0 if batch["n_inconclusive"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Certify the first a-loop edge for the six roots, or verify a stored certificate."""
    parser = argparse.ArgumentParser(
        description="exp72: Gate C3a -- first archived edge of the a loop (no path claims)")
    parser.add_argument("--out", type=Path, default=C3A_CERT_JSON,
                        help="certificate JSON path")
    parser.add_argument("--verify", type=Path, default=None,
                        help="re-check an existing certificate JSON exactly")
    parser.add_argument("--roots", type=str, default=None,
                        help="comma-separated root subset (dev/smoke only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="certify only the first N uniform pieces per root "
                             "(dev/smoke only; produces a partial certificate)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint directory: per-root planner slices and per-chunk "
                             "results are written there and reused on resume")
    parser.add_argument("--merge", type=Path, default=None,
                        help="merge the planner/chunk files of this checkpoint directory "
                             "into the final certificate (requires all six roots)")
    args = parser.parse_args(argv)
    if args.verify is not None:
        return verify_certificate(args.verify)
    if args.merge is not None:
        return merge_certificate(args.merge, args.out)

    started = time.perf_counter()
    print("exp72: Gate C3a -- the first complete archived edge of the a loop\n")
    manifest, provenance = check_model_provenance()
    print("phase 0: model provenance")
    for name, passed, detail in provenance:
        print(f"    {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
    if not all(entry[1] for entry in provenance):
        raise RuntimeError("model provenance failed -- refusing to certify")
    edge_info = archived_edge()
    edge = parse_frac(edge_info["edge_im"])
    print(f"    edge: lambda in [0, i * {edge_info['edge_im'][:24]}...] "
          f"(~{float(edge):.3e}), direction (0,-1,0)\n")

    root_list = ([int(value) for value in args.roots.split(",")] if args.roots
                 else list(TRACKED_INDICES))
    checkpoint = args.checkpoint
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    planners: dict[int, dict] = {}
    planner_payloads = []
    for index in root_list:
        planner_path = checkpoint / f"planner_{index}.json" if checkpoint else None
        if planner_path is not None and planner_path.exists():
            stored = json.loads(planner_path.read_text())
            stored["grid"] = [np.array([[part[0], part[1]] for part in point], dtype=complex)
                              for point in stored["grid"]]
            planners[index] = stored
            print(f"    [root {index}] planner slice loaded -> resume", flush=True)
            continue
        planner_payloads.append({"root_index": index, "edge": edge,
                                 "limit": args.limit, "checkpoint": checkpoint})
    if planner_payloads:
        with mp.Pool(min(len(planner_payloads), 6)) as pool:
            planner_results = pool.map(root_planner, planner_payloads)
        for planner in planner_results:
            planners[planner["root_index"]] = planner
    chunk_payloads = []
    for index in root_list:
        planner = planners.get(index)
        if planner is None or planner["plan"]["plan"] is None:
            continue
        count = planner["plan"]["plan"]["count"]
        step = edge / count
        size = 400
        for start in range(0, planner["piece_count"], size):
            chunk_path = (checkpoint / f"chunk_{index}_{start}.json"
                          if checkpoint else None)
            if chunk_path is not None and chunk_path.exists():
                continue
            chunk_payloads.append({
                "root_index": index,
                "start_index": start,
                "grid": planner["grid"][start:start + size + 1],
                "step": step,
                "k_root": int(planner["plan"]["plan"]["k"]),
                "checkpoint": checkpoint,
            })
    print(f"    {len(chunk_payloads)} certification chunks to run, "
          f"{min(12, len(chunk_payloads) or 1)} workers\n", flush=True)
    if chunk_payloads:
        with mp.Pool(min(12, len(chunk_payloads))) as pool:
            pool.map(certify_chunk, chunk_payloads)
    if checkpoint is not None:
        print(f"    checkpoint files written under {checkpoint}; run --merge to assemble",
              flush=True)
        return 0
    payload = assemble_payload(list(planners.values()), [], started)
    payload["batch"]["summary"] = (f"{payload['batch']['n_certified']}/"
                                   f"{len(root_list)}")
    payload["stats"]["certificate_bytes"] = None
    text = json.dumps(payload, indent=2) + "\n"
    payload["stats"]["certificate_bytes"] = len(text)
    args.out.write_text(text)
    batch = payload["batch"]
    stats = payload["stats"]
    print(f"\nbatch: {batch['summary']} certified | accepted pieces {stats['n_accepted']} "
          f"rejected {stats['n_rejected']} | min step {stats['min_step']} | "
          f"limiting root {stats['limiting_root']}")
    print(f"interval evaluation {stats['interval_eval_seconds']:.0f}s | total wall "
          f"{stats['total_wall_seconds']:.0f}s | certificate {stats['certificate_bytes']} "
          f"bytes")
    print(f"certificate written to {args.out}")
    print(f"re-check: python3 -m study.exp72_gate_c3a_edge --verify {args.out}")
    return 0 if batch["n_inconclusive"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
