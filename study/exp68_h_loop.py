#!/usr/bin/env python3
"""exp68: the h loop -- full Gate-A control on the exp66 -x radius-0.015 scan.

GPT's group computation (``GPT_GATE_B_WITNESS_20261017.md``) shows that the three complete
permutations ``a = (1 6)`` (exp64), ``d = (0 14)(4 6)`` (exp66 fold2) and
``h = (1 8)(4 10 13)(7 12)(9 15)`` yield, by conjugation alone, the four transpositions
(1 6), (6 8), (1 4), (8 10), which generate S5 of order 120.  ``a`` and ``d`` passed the full
Gate-A control protocol in exp64/exp66; ``h`` is the exp66 archive's *radius-0.015 scan*
result on the -x slice (``fold1_scan_r0.015``) and has never been run under the full control.
This module closes exactly that gap and nothing else:

* the fold and path are re-derived from the archive, not reinvented: fold ``u* =
  +0.22628458830430512`` on the -x slice (``folds_json`` entry 1, seeding pair (2,6), augmented
  residual 5.5e-11), loop = ``build_loop(0, u*, 0.015, samples=96, orientation=+1, arc=0.05)``;
  the re-derived 669-node path is asserted bit-identical to the archived one;
* the frozen polished base root table is loaded straight from the exp66 archive (same 16-root
  numbering as exp64/exp66, never renumbered) and re-polished in place;
* four loops are re-run with the exp66 discipline: main, reverse (h has a 3-cycle and is no
  involution, so the reverse loop must read ``h^-1 = (1 8)(4 13 10)(7 12)(9 15)``, never h),
  empty, and the 96->192 step-refinement re-run -- all 16 roots tracked, tangential linear
  prediction, jump rejection, per-step closest-pair gap with a near-collision guard, and
  injective endpoint matching where any unmatched root is a failure state, never silently
  fixed;
* the empty loop uses the probe-floor radius 1e-4: exp66's per-slice empty-loop probes measured
  an off-axis (14,15) branch point 2e-3..3.2e-3 from the base point on the -x slice
  (``emptyprobe_-x_r0.005`` reads (14 15) and fails; r <= 2e-3 reads identity), so 1e-4 is the
  largest radius the control may use;
* per-step raw data of every loop is written to ``study/exp68_h_loop.npz``: the 16-root complex
  trajectory (real+imag) at every parameter node, per-root residual/correction/actual step,
  the closest-pair gap and pairing, survival flags, the failure records, and per-root endpoint
  residual and gap-to-image.

Run::

    python3 -m study.exp68_h_loop

Exit code 0 iff the main loop reproduces h, the reverse loop reads h^-1 (not h), the empty loop
reads the identity, the 96->192 re-run agrees with the main loop, and every loop's permutation
is complete (all 16 matched, injective, no failures).  Any disagreement exits non-zero.  This
certifies the *measurement protocol* for h only: the non-solvability conclusion stays with Gate
B's group theory and path certification stays with Gate C.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from typing import Callable

import numpy as np

from study import sr0
from study.exp37_complex_loop import residual12
from study.exp38_monodromy_group import torus_distance_complex
from study.exp64_pose_loop_gate_a import (
    base_pose,
    build_empty_loop,
    build_loop,
    compose,
    cycle_type,
    fingerprint,
    fixed_sr5_links,
    inverse,
    target_at,
)
from study.exp66_gate_a_generators import polish_fibre, run_loop_quiet

#: h in image notation, exactly as the exp66 archive reads it (``fold1_scan_r0.015_permutation``).
EXPECTED_H = [0, 8, 2, 3, 10, 5, 6, 12, 1, 15, 13, 11, 7, 4, 14, 9]

#: The reverse loop must read h^-1 -- h contains the 3-cycle (4 10 13), so h is not an
#: involution and h^-1 = (1 8)(4 13 10)(7 12)(9 15) differs from h.
EXPECTED_H_INV = inverse(EXPECTED_H)

#: Cycle type of h: one 3-cycle, three 2-cycles, seven fixed points.
H_CYCLE_TYPE = "3+2+2+2+1+1+1+1+1+1+1"


def _closest_pair(qs: list[np.ndarray], alive: list[int]) -> tuple[float, int, int]:
    """Smallest complex torus distance among the alive roots, and the pair attaining it.

    Args:
        qs: All tracked configurations (complex).
        alive: Indices still being tracked.

    Returns:
        ``(gap, i, j)``; ``(inf, -1, -1)`` when fewer than two roots are alive.
    """
    best = (float("inf"), -1, -1)
    for a, i in enumerate(alive):
        for j in alive[a + 1:]:
            distance = torus_distance_complex(qs[i], qs[j])
            if distance < best[0]:
                best = (distance, i, j)
    return best


def _endpoint_numbers(links: list[np.ndarray], q_end: list[np.ndarray],
                      permutation: list[int], fibre: list[np.ndarray],
                      final_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-root endpoint residual (against the closing pose) and gap to the matched image.

    Args:
        links: The fixed SR5 link transforms (complex).
        q_end: Final configuration per root.
        permutation: The loop's permutation (``-1`` for failure states).
        fibre: The frozen polished base fibre.
        final_pose: The loop's closing pose (the base pose ``target0`` for a closed loop).

    Returns:
        ``(endpoint_residual, gap_to_image)``: length-16 arrays; ``gap_to_image`` is NaN for
        unmatched roots.
    """
    endpoint_residual = np.array(
        [float(np.linalg.norm(residual12(links, q, final_pose))) for q in q_end])
    gap_to_image = np.array([
        torus_distance_complex(q_end[index], fibre[image]) if 0 <= image < len(fibre)
        else float("nan")
        for index, image in enumerate(permutation)
    ])
    return endpoint_residual, gap_to_image


def _print_loop(label: str, links: list[np.ndarray], q_end: list[np.ndarray],
                permutation: list[int], ok: bool, status: list[str | None],
                final_pose: np.ndarray, fibre: list[np.ndarray],
                record: dict[str, np.ndarray], failures: list[dict],
                endpoint_residual: np.ndarray, gap_to_image: np.ndarray) -> None:
    """Print one loop's permutation, cycle type, per-root numbers, and the step summary.

    Args:
        label: Human-readable loop name.
        links: The fixed SR5 link transforms (complex), for the endpoint residual.
        q_end: Final configuration per root.
        permutation: The permutation (``-1`` for failure states).
        ok: Whether the permutation is complete and injective.
        status: Per-root failure string.
        final_pose: The loop's closing pose.
        fibre: The base fibre, for the return gap.
        record: The per-node record from exp66's :func:`walk_fibre_recorded`.
        failures: The failure event list.
        endpoint_residual: Per-root endpoint residual.
        gap_to_image: Per-root gap to the matched base image.
    """
    moved = [(i, image) for i, image in enumerate(permutation) if image >= 0 and image != i]
    finite_correction = record["correction"][np.isfinite(record["correction"])]
    finite_step = record["actual_step"][np.isfinite(record["actual_step"])]
    print(f"  {label}: permutation {permutation}")
    print(f"    cycle type {cycle_type(permutation)} | complete {ok} | moved {moved} "
          f"| failures {len(failures)}")
    print(f"    nodes {record['residual'].shape[0]} | max residual "
          f"{float(np.nanmax(record['residual'])):.1e} | max correction "
          f"{float(np.nanmax(finite_correction)):.1e} | max actual step "
          f"{float(np.nanmax(finite_step)):.1e} | min pair gap "
          f"{float(np.nanmin(record['min_gap'])):.2e}"
          f"{' (merge risk flagged)' if record['merge_risk'].any() else ''}")
    print(f"    endpoint: max residual {float(np.nanmax(endpoint_residual)):.1e}, "
          f"max gap to image {float(np.nanmax(gap_to_image)):.1e}")
    for index in range(len(q_end)):
        state = "FAILED" if status[index] is not None else f"-> {permutation[index]}"
        print(f"    root {index:2d}: endpoint residual {endpoint_residual[index]:.1e}, "
              f"gap to image {gap_to_image[index]:.1e}, {state}")
    for failure in failures:
        print(f"    FAILURE node {failure['node']} root {failure['root']}: "
              f"{failure['reason']}")


def _agreement_checks(permutations: dict[str, list[int]], oks: dict[str, bool],
                      expected_h: list[int]) -> list[tuple[str, bool, str]]:
    """Build the four-loop Gate-A agreement checks for the h protocol.

    Args:
        permutations: Per-tag permutations (main, reverse, empty, refined).
        oks: Per-tag completeness flags.
        expected_h: The archived h image list that the main loop must reproduce.

    Returns:
        The named check list ``(name, passed, detail)``.
    """
    identity = list(range(16))
    checks: list[tuple[str, bool, str]] = []
    for tag in ("main", "reverse", "empty", "refined"):
        checks.append((f"{tag} loop is a complete permutation", oks[tag],
                       "all 16 matched, injective"))
    if oks["main"]:
        checks.append(("main reproduces the archived h (fold1_scan_r0.015)",
                       permutations["main"] == expected_h,
                       f"{permutations['main']} vs {expected_h}"))
    if oks["main"] and oks["reverse"]:
        checks.append(("reverse == main^-1",
                       permutations["reverse"] == inverse(permutations["main"]),
                       f"{permutations['reverse']} vs {inverse(permutations['main'])}"))
        checks.append(("reverse is h^-1, not h (h has a 3-cycle)",
                       permutations["reverse"] == EXPECTED_H_INV
                       and permutations["reverse"] != expected_h,
                       f"{permutations['reverse']} vs {EXPECTED_H_INV}"))
    if oks["main"] and oks["empty"]:
        checks.append(("empty loop is identity", permutations["empty"] == identity,
                       f"{permutations['empty']}"))
    if oks["main"] and oks["refined"]:
        checks.append(("step-refinement agrees with main",
                       permutations["refined"] == permutations["main"],
                       f"{permutations['refined']} vs {permutations['main']}"))
    if oks["main"] and oks["reverse"]:
        composed = compose(permutations["main"], permutations["reverse"])
        checks.append(("main . reverse == identity", composed == identity, f"{composed}"))
    return checks


def _print_checks(checks: list[tuple[str, bool, str]]) -> bool:
    """Print one protocol's agreement checks and return whether all passed.

    Args:
        checks: The named check list from :func:`_agreement_checks`.

    Returns:
        True iff every check passed.
    """
    all_ok = True
    for name, passed, detail in checks:
        print(f"    {'PASS' if passed else 'FAIL'}  {name}  ({detail})")
        all_ok = all_ok and passed
    return all_ok


def _store_loop(payload: dict[str, np.ndarray], prefix: str, u_points: list[complex],
                record: dict[str, np.ndarray], permutation: list[int], ok: bool,
                failures: list[dict], q_end: list[np.ndarray],
                endpoint_residual: np.ndarray, gap_to_image: np.ndarray) -> None:
    """Archive one loop's per-node arrays and endpoint numbers under ``prefix``.

    The array-name convention is exp66's: ``u_real/u_imag`` for the parameter path,
    ``roots_real/roots_imag (N,16,6)``, ``residual/correction/actual_step (N,16)``,
    ``min_gap (N,)``, ``closest_pair (N,2)``, ``alive (N,16)``, ``merge_risk (N,)``,
    ``permutation (16,)``, ``complete``, ``failures_json``; exp68 adds ``q_end_real/imag``,
    ``endpoint_residual`` and ``gap_to_image`` per loop.

    Args:
        payload: The npz payload being built.
        prefix: Array-name prefix (``main``, ``reverse``, ``empty``, ``refined``).
        u_points: The ``u``-values of the loop.
        record: The per-node record from exp66's :func:`walk_fibre_recorded`.
        permutation: The loop's permutation.
        ok: Whether the permutation is complete and injective.
        failures: The failure event list.
        q_end: Final configuration per root.
        endpoint_residual: Per-root endpoint residual.
        gap_to_image: Per-root gap to the matched base image.
    """
    payload[f"{prefix}_u_real"] = np.asarray([u.real for u in u_points], dtype=float)
    payload[f"{prefix}_u_imag"] = np.asarray([u.imag for u in u_points], dtype=float)
    for array_key in ("roots_real", "roots_imag", "residual", "correction",
                      "actual_step", "min_gap", "closest_pair", "alive", "merge_risk"):
        payload[f"{prefix}_{array_key}"] = record[array_key]
    payload[f"{prefix}_permutation"] = np.asarray(permutation, dtype=int)
    payload[f"{prefix}_complete"] = np.asarray(ok)
    payload[f"{prefix}_failures_json"] = np.asarray(json.dumps(failures))
    q_end_arr = np.asarray(q_end, dtype=complex)
    payload[f"{prefix}_q_end_real"] = q_end_arr.real
    payload[f"{prefix}_q_end_imag"] = q_end_arr.imag
    payload[f"{prefix}_endpoint_residual"] = endpoint_residual
    payload[f"{prefix}_gap_to_image"] = gap_to_image


def main() -> int:
    """Replay the archived h loop under the full Gate-A control protocol and archive it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="study/exp66_gate_a_generators.npz",
                        help="exp66 archive the h loop is replayed from")
    parser.add_argument("--out", default="study/exp68_h_loop.npz",
                        help="npz path for the per-step raw data")
    parser.add_argument("--radius", type=float, default=0.015,
                        help="loop radius around the fold (the h scan radius)")
    parser.add_argument("--arc", type=float, default=0.05,
                        help="height of the approach arc above the real axis, metres")
    parser.add_argument("--samples", type=int, default=96, help="circle samples")
    parser.add_argument("--refined-samples", type=int, default=192,
                        help="circle samples for the step-refinement re-run")
    parser.add_argument("--empty-radius", type=float, default=1e-4,
                        help="empty-loop radius (exp66's probe floor: near-base branch points)")
    parser.add_argument("--jump-tolerance", type=float, default=0.5)
    parser.add_argument("--merge-floor", type=float, default=1e-2,
                        help="closest-pair gap below which a loop node is refused")
    args = parser.parse_args()

    assert cycle_type(EXPECTED_H) == H_CYCLE_TYPE, "EXPECTED_H is not h's cycle type"
    assert cycle_type(EXPECTED_H_INV) == H_CYCLE_TYPE, "EXPECTED_H_INV is not h^-1's cycle type"

    archive = np.load(args.input, allow_pickle=False)
    print("exp68: the h loop -- full Gate-A control on the exp66 -x radius-0.015 scan\n")

    print("phase 0: provenance and mechanical-parameter fingerprint")
    folds = json.loads(str(archive["folds_json"]))
    fold = folds[1]
    if fold["direction"] != "-x":
        raise RuntimeError(f"archive fold index 1 is {fold['direction']}, expected -x")
    u_star = complex(archive["fold1_u_star"])
    direction = np.asarray(archive["fold1_direction"], dtype=float)
    print(f"  archive fold: -x slice, seeding pair {fold['pair']}, event u "
          f"{fold['event_u']:+.5f} ({fold['event_kind']}), refined u* = "
          f"{u_star.real:+.9f}{u_star.imag:+.1e}i (augmented residual "
          f"{fold['refine_residual']:.1e})")
    archived_h = [int(image) for image in archive["fold1_scan_r0.015_permutation"]]
    archived_complete = bool(archive["fold1_scan_r0.015_complete"])
    archived_failures = json.loads(str(archive["fold1_scan_r0.015_failures_json"]))
    print(f"  archived scan: complete {archived_complete}, failures {archived_failures}, "
          f"permutation {archived_h} (cycle type {cycle_type(archived_h)})")
    if archived_h != EXPECTED_H:
        raise RuntimeError(f"archived fold1_scan_r0.015 permutation {archived_h} is not the "
                           f"spec's h {EXPECTED_H}")
    if not archived_complete or archived_failures:
        raise RuntimeError("the archived h scan was not complete -- refusing to replay it")

    links_real = fixed_sr5_links()
    print(f"  18 translation components: {np.array2string(fingerprint(links_real), precision=6)}")
    print(f"  link 5 y offset (the SR0<->SR5 parameter): {links_real[4][1, 3]:.6f} m")
    links = [link.astype(complex) for link in links_real]
    target0 = np.asarray(archive["target0"], dtype=complex)
    model = sr0.robot(links_real)
    target0_check, _q_true = base_pose(0, model)
    if not np.array_equal(np.asarray(target0_check, dtype=complex), target0):
        raise RuntimeError("the archived base pose is not base_pose(seed=0) on the fixed SR5")
    print(f"  base position p0 = {np.array2string(target0[:3, 3].real, precision=6)} m "
          f"(reproduced from seed 0 exactly)\n")

    print("phase 1: frozen polished base root table (loaded from the archive, re-polished)")
    fibre = [archive["fibre_real"][i] + 1j * archive["fibre_imag"][i] for i in range(16)]
    polished = polish_fibre(links, fibre, target0)
    polish_move = np.array([float(np.linalg.norm(polished[i] - fibre[i])) for i in range(16)])
    fibre_residual = np.array(
        [float(np.linalg.norm(residual12(links, q, target0))) for q in polished])
    base_gap, base_first, base_second = _closest_pair(polished, list(range(16)))
    archived_gap = float(archive["base_min_gap"])
    archived_pair = tuple(int(x) for x in archive["base_closest_pair"])
    print(f"  re-polish: max movement {float(np.max(polish_move)):.1e} rad, max residual "
          f"{float(np.max(fibre_residual)):.1e}")
    print(f"  frozen table min pair gap {base_gap:.9f} rad (pair {base_first},{base_second}) "
          f"| archived {archived_gap:.9f} (pair {archived_pair})")
    if not np.isclose(base_gap, archived_gap, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"base pair gap {base_gap:.12f} disagrees with the archived "
                           f"{archived_gap:.12f}")
    if (base_first, base_second) != archived_pair:
        raise RuntimeError(f"base closest pair {(base_first, base_second)} disagrees with the "
                           f"archived {archived_pair}")

    print("\nphase 2: path re-derivation (must reproduce the archived 669-node loop)")
    loop = build_loop(0.0, u_star, args.radius, samples=args.samples, orientation=1.0,
                      arc=args.arc)
    archived_u = (archive["fold1_scan_r0.015_u_real"]
                  + 1j * archive["fold1_scan_r0.015_u_imag"])
    loop_arr = np.asarray(loop, dtype=complex)
    if len(loop) != len(archived_u) or not np.array_equal(loop_arr, archived_u):
        raise RuntimeError("the re-derived loop path is not bit-identical to the archived "
                           "fold1_scan_r0.015 path -- the path construction changed")
    reverse_loop = list(reversed(loop))
    empty_loop = build_empty_loop(0.0, args.empty_radius, samples=args.samples)
    refined_loop = build_loop(0.0, u_star, args.radius, samples=args.refined_samples,
                              orientation=1.0, arc=args.arc)
    print(f"  main {len(loop)} nodes (matches archive), reverse {len(reverse_loop)}, "
          f"empty (r={args.empty_radius:g}) {len(empty_loop)}, refined {len(refined_loop)}")
    print("  parameter trajectory (u, real+imag) -- first/last/circle samples:")
    for index in (0, 1, len(loop) // 2, len(loop) - 2, len(loop) - 1):
        u = loop[index]
        print(f"    u[{index}] = ({u.real:+.6f}, {u.imag:+.6f})")

    target_at_fn: Callable[[complex], np.ndarray] = partial(target_at, target0, direction)
    final_pose = target_at_fn(complex(loop[-1]))

    print("\nphase 3: four control loops (exp66 discipline, all 16 roots tracked)")
    loops: dict[str, dict] = {}
    permutations: dict[str, list[int]] = {}
    oks: dict[str, bool] = {}
    for tag, u_points in (("main", loop), ("reverse", reverse_loop), ("empty", empty_loop),
                          ("refined", refined_loop)):
        perm, ok, status, q_end, record, failures = run_loop_quiet(
            links, polished, u_points, target_at_fn, jump_tolerance=args.jump_tolerance,
            merge_floor=args.merge_floor)
        endpoint_residual, gap_to_image = _endpoint_numbers(
            links, q_end, perm, polished, final_pose)
        loops[tag] = {"u_points": u_points, "record": record, "failures": failures,
                      "status": status, "permutation": perm, "ok": ok, "q_end": q_end,
                      "endpoint_residual": endpoint_residual, "gap_to_image": gap_to_image}
        permutations[tag] = perm
        oks[tag] = ok
        _print_loop(f"{tag} loop", links, q_end, perm, ok, status, final_pose, polished,
                    record, failures, endpoint_residual, gap_to_image)
        print()

    print("phase 4: Gate-A agreement checks")
    checks = _agreement_checks(permutations, oks, EXPECTED_H)
    all_ok = _print_checks(checks)
    print(f"\n  the four permutations: main = {permutations['main']} "
          f"(cycle type {cycle_type(permutations['main'])})")
    print(f"                          reverse = {permutations['reverse']} "
          f"(cycle type {cycle_type(permutations['reverse'])}, must be h^-1)")
    print(f"                          empty = {permutations['empty']}")
    print(f"                          refined = {permutations['refined']}")
    print("  h = (1 8)(4 10 13)(7 12)(9 15) | h^-1 = (1 8)(4 13 10)(7 12)(9 15)")

    print("\nphase 5: archive the per-step raw data")
    _dump_raw_data(args.out, args, links, target0, direction, u_star, fold, polished,
                   polish_move, fibre_residual, base_gap, (base_first, base_second),
                   loops, checks, all_ok)
    if not all_ok:
        print("FAIL: a Gate-A control disagreed (see the checks above); exit non-zero")
        return 1
    print("PASS: main = h, reverse = h^-1 (not h), empty = identity, refined = main; "
          "all four loops complete with no failures")
    return 0


def _dump_raw_data(path: str, args, links: list[np.ndarray], target0: np.ndarray,
                   direction: np.ndarray, u_star: complex, fold: dict,
                   polished: list[np.ndarray], polish_move: np.ndarray,
                   fibre_residual: np.ndarray, base_gap: float, base_pair: tuple[int, int],
                   loops: dict[str, dict], checks: list[tuple[str, bool, str]],
                   all_ok: bool) -> None:
    """Write the full h-protocol record (base table, four loops, checks) to an npz.

    Complex values are stored as separate real and imaginary arrays, never truncated.  For
    every node of every loop (main, reverse, empty, refined) the archive holds the 16-root
    trajectory, per-root correction/residual/actual step, the closest-pair gap and pair, the
    alive flags, the merge-risk flags and the failure records; each loop additionally holds its
    permutation, completeness flag, endpoint residual, gap-to-image and final configuration.

    Args:
        path: Output ``.npz`` path.
        args: The parsed arguments (for the summary).
        links: The fixed SR5 link transforms (complex).
        target0: Base 4x4 pose.
        direction: Unit translation direction of the -x slice.
        u_star: The branch point's ``u`` coordinate.
        fold: The archive's fold metadata (provenance).
        polished: The re-polished frozen base fibre.
        polish_move: Per-root re-polish movement.
        fibre_residual: Per-root base residual after re-polishing.
        base_gap: Smallest pair gap of the frozen table.
        base_pair: The pair attaining ``base_gap``.
        loops: The four loop dicts.
        checks: The named agreement checks.
        all_ok: Whether every check passed.
    """
    fibre_arr = np.asarray(polished, dtype=complex)
    payload: dict[str, np.ndarray] = {
        "target0": target0,
        "direction": direction,
        "u_star": np.asarray(u_star),
        "radius": np.asarray(args.radius),
        "arc": np.asarray(args.arc),
        "samples": np.asarray(args.samples),
        "refined_samples": np.asarray(args.refined_samples),
        "empty_radius": np.asarray(args.empty_radius),
        "jump_tolerance": np.asarray(args.jump_tolerance),
        "merge_floor": np.asarray(args.merge_floor),
        "fold_pair": np.asarray(fold["pair"], dtype=int),
        "fold_event_u": np.asarray(fold["event_u"]),
        "fold_event_gap": np.asarray(fold["event_gap"]),
        "fold_event_kind_json": np.asarray(json.dumps(fold["event_kind"])),
        "fold_refine_residual": np.asarray(fold["refine_residual"]),
        "fibre_real": fibre_arr.real,
        "fibre_imag": fibre_arr.imag,
        "fibre_residual": fibre_residual,
        "fibre_polish_move": polish_move,
        "base_min_gap": np.asarray(base_gap),
        "base_closest_pair": np.asarray(base_pair, dtype=int),
        "expected_h": np.asarray(EXPECTED_H, dtype=int),
        "expected_h_inv": np.asarray(EXPECTED_H_INV, dtype=int),
    }
    for tag, loop in loops.items():
        _store_loop(payload, tag, loop["u_points"], loop["record"], loop["permutation"],
                    loop["ok"], loop["failures"], loop["q_end"],
                    loop["endpoint_residual"], loop["gap_to_image"])
    payload["checks_json"] = np.asarray(json.dumps(
        [[name, passed, detail] for name, passed, detail in checks]))
    payload["all_ok"] = np.asarray(all_ok)
    payload["summary_json"] = np.asarray(json.dumps({
        "source_archive": args.input,
        "source_loop": "fold1_scan_r0.015",
        "u_star": [u_star.real, u_star.imag],
        "direction": direction.tolist(),
        "radius": args.radius, "arc": args.arc, "samples": args.samples,
        "refined_samples": args.refined_samples, "empty_radius": args.empty_radius,
        "jump_tolerance": args.jump_tolerance, "merge_floor": args.merge_floor,
        "base_min_gap": base_gap, "base_pair": list(base_pair),
        "loops": {
            tag: {
                "permutation": loop["permutation"], "complete": loop["ok"],
                "nodes": len(loop["u_points"]), "failures": len(loop["failures"]),
                "max_residual": float(np.nanmax(loop["record"]["residual"])),
                "max_correction": float(np.nanmax(
                    loop["record"]["correction"][np.isfinite(loop["record"]["correction"])])),
                "max_actual_step": float(np.nanmax(
                    loop["record"]["actual_step"][np.isfinite(loop["record"]["actual_step"])])),
                "min_pair_gap": float(np.nanmin(loop["record"]["min_gap"])),
                "endpoint_residual_max": float(np.nanmax(loop["endpoint_residual"])),
                "gap_to_image_max": float(np.nanmax(loop["gap_to_image"])),
            }
            for tag, loop in loops.items()
        },
        "all_ok": all_ok,
        "machine_verified": False,
    }))
    np.savez_compressed(path, **payload)
    print(f"  raw data written to {path} "
          f"({sum(value.nbytes for value in payload.values()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    raise SystemExit(main())
