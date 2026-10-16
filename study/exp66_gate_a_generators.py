#!/usr/bin/env python3
"""exp66: Gate A generators -- pose-space transpositions that involve root 1 or 6.

Q27 / T4, second batch.  The audited exp64 loop delivered the trustworthy transposition
``(1 6)``; a single transposition generates C2 (solvable), so it is no non-solvability
evidence by itself.  The audit's strategy (``GPT_GATE_A_AUDIT_20261017.md`` section 5) is to
grow a tree of transpositions: the next generators must pair root 1 or 6 with a *new* root.
This module follows it exactly:

* the same base pose (seed 0), the same 16-root fibre construction and the same root
  numbering as exp64 -- the base roots are Newton-polished and **frozen** (the audit's
  requirement: one polished root table for every loop and for all matching, never
  renumbered);
* the same legal C1 pose coordinate ``p(u) = p0 + u * d`` (orientation fixed, ``u`` complex),
  but swept along the four remaining axis directions ``d = [+-1,0,0], [0,+-1,0]`` -- the z
  slice already gave ``(1 6)``.  The sweep walks **all 16 roots** along the real ``u``-axis
  and records every real coalescence event; an event whose pair ``(i, j)`` has exactly one
  of ``{1, 6}`` is a candidate generator;
* each candidate is refined with the augmented branch-point system ``[err6; det J] = 0``
  over ``(q, u)`` exactly as exp64 (2 pi-aligned midpoint seed);
* each refined fold is looped with the exp64 protocol -- approach through the upper
  half-plane (``Im = 0.05``) so the complex roots' real-axis birth folds are bypassed, all
  16 roots carried with *tangential* linear prediction (the audit's recommendation), jump
  rejection, a per-step closest-pair gap diagnostic with a near-collision guard, and
  injective endpoint matching where any unmatched root is a failure state, never a silently
  fixed point.  A radius scan (``5e-3``, ``1.5e-2``, ``3e-2``, ``1e-3``, ``1e-4``) maps how
  many branch points the default circle encloses, and when the net permutation's support
  exceeds the fold pair a second full protocol is run at the smallest isolating radius;
* four loops per fold -- main, reverse, empty, and a 96->192 step-refinement re-run -- must
  agree; each closed loop yields one complete permutation read from the **whole** fibre,
  never from a single pair (shared critical values are tracked as one product permutation).
  The empty loop uses the probe floor radius ``1e-4`` because the per-slice empty-loop
  radius probes (``5e-3``..``1e-4``, all archived) show off-axis branch points as close as
  about ``1e-3`` to the base point (the -x ``(14,15)`` point);
* per-step raw data of **every** loop (main, reverse, empty, refined) and of every sweep
  node is written to ``study/exp66_gate_a_generators.npz``: the 16-root complex trajectory
  (real+imag), per-root correction and residual, the closest-pair gap and pair, the actual
  step taken, and the failure records.

Boundary (stated, not negotiated): these are numerical permutations; a tree of
transpositions on five roots is the *target evidence* for Gate B, and path certification is
Gate C -- neither is done here.

Run::

    python3 -m study.exp66_gate_a_generators

Exit code 0 iff every loop that was run passed all its Gate-A agreement self-checks and at
least one delivered permutation's support involves root 1 or 6 without being ``(1 6)``
itself.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from typing import Callable

import numpy as np

from study import ik_structure as iks
from study import sr0
from study.exp37_complex_loop import fk_complex, newton_complex, residual12
from study.exp38_monodromy_group import torus_distance_complex
from study.exp57_sr0_monodromy import refine_branch_pose
from study.exp64_pose_loop_gate_a import (
    base_pose,
    build_empty_loop,
    build_full_fibre,
    build_loop,
    compose,
    cycle_type,
    fingerprint,
    fixed_sr5_links,
    inverse,
    read_permutation,
    target_at,
)

#: Axis directions swept by this module (z was exp64's slice, already delivering (1 6)).
SWEEP_DIRECTIONS: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
}


def polish_fibre(links: list[np.ndarray], fibre: list[np.ndarray],
                 target0: np.ndarray) -> list[np.ndarray]:
    """Newton-polish every base root and freeze the table (audit requirement).

    The numbering is the exp64 fibre order and is never changed; each root is only moved
    within its own Newton basin (any root moving further than ``0.1`` rad is an error).

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The 16 base roots (complex).
        target0: The base 4x4 pose.

    Returns:
        The polished root table, same order.
    """
    polished: list[np.ndarray] = []
    for index, root in enumerate(fibre):
        q_new, residual = newton_complex(links, root, target0, tolerance=1e-13)
        move = float(np.linalg.norm(q_new - root))
        if residual > 1e-10 or not np.all(np.isfinite(q_new)) or move > 0.1:
            raise RuntimeError(f"base root {index} did not polish: residual {residual:.1e}, "
                               f"moved {move:.1e}")
        polished.append(q_new)
    return polished


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


def _empty_record(node_count: int, n: int) -> dict[str, np.ndarray]:
    """NaN-filled per-node record arrays with the exp66 shape convention.

    Args:
        node_count: Number of parameter nodes.
        n: Number of tracked roots.

    Returns:
        The record dict (``roots_real``, ``roots_imag``, ``residual``, ``correction``,
        ``actual_step``, ``min_gap``, ``closest_pair``, ``alive``, ``merge_risk``).
    """
    return {
        "roots_real": np.full((node_count, n, 6), np.nan),
        "roots_imag": np.full((node_count, n, 6), np.nan),
        "residual": np.full((node_count, n), np.nan),
        "correction": np.full((node_count, n), np.nan),
        "actual_step": np.full((node_count, n), np.nan),
        "min_gap": np.full(node_count, np.nan),
        "closest_pair": np.full((node_count, 2), -1, dtype=int),
        "alive": np.zeros((node_count, n), dtype=np.int8),
        "merge_risk": np.zeros(node_count, dtype=np.int8),
    }


def walk_fibre_recorded(
    links: list[np.ndarray],
    roots: list[np.ndarray],
    poses: list[np.ndarray],
    parameters: list[complex],
    *,
    jump_tolerance: float = 0.5,
    tolerance: float = 1e-9,
    iterations: int = 40,
    merge_floor: float = 1e-2,
) -> tuple[list[np.ndarray], list[str | None], dict[str, np.ndarray], list[dict]]:
    """Track every root through a pose list, recording every node's raw numbers.

    Tangential prediction (the audit's recommendation over exp64's fixed ``2 q_cur - q_prev``):
    the prediction extends the previous node's *parameter-rate* direction,
    ``q_cur + (q_cur - q_prev) * |du_k| / |du_{k-1}|`` clamped to ``[0, 2]``; corners,
    repeated nodes and the first step fall back to a carry-forward prediction.  A correction
    further from its prediction than ``jump_tolerance``, or above ``tolerance``, is a failure
    state for that root -- recorded, never silently carried.  After every node the closest
    alive pair is computed; a gap below ``merge_floor`` marks both roots failed at that node
    (a near-collision on a supposedly regular path is refused, per the audit).  All record
    arrays hold NaN for dead roots.

    Args:
        links: The fixed SR5 link transforms (complex).
        roots: The root configurations at ``poses[0]`` (complex).
        poses: The pose sequence (complex 4x4 arrays), in order.
        parameters: The complex ``u``-value of each pose (for the tangential prediction).
        jump_tolerance: Largest accepted distance from the prediction, radians.
        tolerance: Residual the corrector must reach for a step to count.
        iterations: Newton iterations per correction.
        merge_floor: Closest-pair gap below which both roots are failed at that node.

    Returns:
        ``(q_end, status, record, failures)``: final configuration per root, per-root failure
        string (``None`` = survived), the per-node record dict, and the failure event list
        (dicts with ``node``, ``root`` and ``reason``).
    """
    node_count = len(poses)
    n = len(roots)
    q_prev = [np.asarray(root, dtype=complex).copy() for root in roots]
    q_cur = [q.copy() for q in q_prev]
    status: list[str | None] = [None] * n
    failures: list[dict] = []
    record = _empty_record(node_count, n)

    def _record_node(node_index: int) -> None:
        alive_now = [i for i in range(n) if status[i] is None]
        for index in alive_now:
            record["roots_real"][node_index, index] = q_cur[index].real
            record["roots_imag"][node_index, index] = q_cur[index].imag
            record["alive"][node_index, index] = 1
        if len(alive_now) >= 2:
            gap, first, second = _closest_pair(q_cur, alive_now)
            record["min_gap"][node_index] = gap
            record["closest_pair"][node_index] = (first, second)

    # Node 0: initial polish at the loop's base pose; the input roots already sit on the
    # base fibre, this confirms it and gives the first record.
    for index in range(n):
        q_new, res = newton_complex(links, q_cur[index], poses[0], iterations=iterations,
                                    tolerance=tolerance)
        if res > tolerance or not np.all(np.isfinite(q_new)):
            status[index] = f"start residual {res:.1e}"
            failures.append({"node": 0, "root": index, "reason": status[index]})
            continue
        record["correction"][0, index] = float(np.linalg.norm(q_new - q_cur[index]))
        record["residual"][0, index] = res
        q_prev[index] = q_cur[index]
        q_cur[index] = q_new
        record["actual_step"][0, index] = float(np.linalg.norm(q_cur[index] - q_prev[index]))
    _record_node(0)

    for pose_index, pose in enumerate(poses[1:], start=1):
        du_prev = (parameters[pose_index - 1] - parameters[pose_index - 2]
                   if pose_index >= 2 else None)
        du_k = parameters[pose_index] - parameters[pose_index - 1]
        alive_now = [i for i in range(n) if status[i] is None]
        q_new_list = [q_cur[index].copy() for index in range(n)]
        for index in alive_now:
            if du_prev is not None and abs(du_prev) > 1e-15:
                ratio = float(np.clip(abs(du_k) / abs(du_prev), 0.0, 2.0))
                prediction = q_cur[index] + (q_cur[index] - q_prev[index]) * ratio
            else:
                prediction = q_cur[index]
            candidate, res = newton_complex(links, prediction, pose, iterations=iterations,
                                            tolerance=tolerance)
            if res > tolerance or not np.all(np.isfinite(candidate)):
                status[index] = f"died at pose {pose_index} (residual {res:.1e})"
                failures.append({"node": pose_index, "root": index, "reason": status[index]})
                continue
            if float(np.linalg.norm(candidate - prediction)) > jump_tolerance:
                status[index] = f"jumped at pose {pose_index}"
                failures.append({"node": pose_index, "root": index, "reason": status[index]})
                continue
            q_new_list[index] = candidate
            record["residual"][pose_index, index] = res
            record["correction"][pose_index, index] = float(
                np.linalg.norm(candidate - prediction))
        for index in range(n):
            if status[index] is None:
                q_prev[index] = q_cur[index]
                q_cur[index] = q_new_list[index]
                record["actual_step"][pose_index, index] = float(
                    np.linalg.norm(q_cur[index] - q_prev[index]))
        _record_node(pose_index)
        # Near-collision guard: the path should be regular, so a closest-pair gap below the
        # floor at an accepted node is refused -- both roots fail at this node, honestly.
        alive_now = [i for i in range(n) if status[i] is None]
        if len(alive_now) >= 2:
            gap, first, second = _closest_pair(q_cur, alive_now)
            if gap < merge_floor:
                record["merge_risk"][pose_index] = 1
                for index in (first, second):
                    status[index] = f"merge risk at pose {pose_index} (gap {gap:.1e})"
                    failures.append({"node": pose_index, "root": index,
                                     "reason": status[index]})
    return q_cur, status, record, failures


def run_loop_quiet(
    links: list[np.ndarray],
    fibre: list[np.ndarray],
    u_points: list[complex],
    target_at_fn: Callable[[complex], np.ndarray],
    *,
    jump_tolerance: float,
    merge_floor: float,
) -> tuple[list[int], bool, list[str | None], list[np.ndarray], dict[str, np.ndarray],
           list[dict]]:
    """Carry the full fibre around one closed loop, recording every step (no printing).

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The frozen polished base fibre.
        u_points: The ``u``-values of the loop.
        target_at_fn: ``u -> complex pose``.
        jump_tolerance: Largest accepted prediction distance.
        merge_floor: Closest-pair gap floor (near-collision guard).

    Returns:
        ``(permutation, ok, status, q_end, record, failures)``.
    """
    poses = [target_at_fn(u) for u in u_points]
    q_end, status, record, failures = walk_fibre_recorded(
        links, fibre, poses, u_points, jump_tolerance=jump_tolerance,
        merge_floor=merge_floor)
    permutation, ok = read_permutation(q_end, status, fibre)
    return permutation, ok, status, q_end, record, failures


def run_loop_recorded(
    links: list[np.ndarray],
    fibre: list[np.ndarray],
    u_points: list[complex],
    target_at_fn: Callable[[complex], np.ndarray],
    *,
    label: str,
    jump_tolerance: float,
    merge_floor: float,
) -> tuple[list[int], bool, list[str | None], list[np.ndarray], dict[str, np.ndarray],
           list[dict]]:
    """Carry the full fibre around one closed loop, recording and printing every step.

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The frozen polished base fibre.
        u_points: The ``u``-values of the loop.
        target_at_fn: ``u -> complex pose``.
        label: Human-readable loop name.
        jump_tolerance: Largest accepted prediction distance.
        merge_floor: Closest-pair gap floor (near-collision guard).

    Returns:
        ``(permutation, ok, status, q_end, record, failures)``.
    """
    permutation, ok, status, q_end, record, failures = run_loop_quiet(
        links, fibre, u_points, target_at_fn, jump_tolerance=jump_tolerance,
        merge_floor=merge_floor)
    poses = [target_at_fn(u) for u in u_points]
    _print_loop_result(label, links, q_end, permutation, ok, status, poses[-1], fibre,
                       record, failures)
    return permutation, ok, status, q_end, record, failures


def _print_loop_result(label: str, links: list[np.ndarray], q_end: list[np.ndarray],
                       permutation: list[int], ok: bool, status: list[str | None],
                       final_pose: np.ndarray, fibre: list[np.ndarray],
                       record: dict[str, np.ndarray], failures: list[dict]) -> None:
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
        record: The per-node record from :func:`walk_fibre_recorded`.
        failures: The failure event list.
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
    for index in range(len(q_end)):
        residual = float(np.linalg.norm(residual12(links, q_end[index], final_pose)))
        matched = permutation[index]
        gap = (torus_distance_complex(q_end[index], fibre[matched])
               if 0 <= matched < len(fibre) else float("nan"))
        state = "FAILED" if status[index] is not None else f"-> {matched}"
        print(f"    root {index:2d}: endpoint residual {residual:.1e}, gap to image {gap:.1e}, "
              f"{state}")
    for failure in failures:
        print(f"    FAILURE node {failure['node']} root {failure['root']}: "
              f"{failure['reason']}")


def directional_sweep(
    links: list[np.ndarray],
    fibre: list[np.ndarray],
    target0: np.ndarray,
    direction: np.ndarray,
    *,
    span: float,
    dip: float,
    jump_tolerance: float = 0.5,
    tolerance: float = 1e-10,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray],
           dict[tuple[int, int], tuple[float, float]]]:
    """Walk all 16 roots along the real ``u``-axis and record every coalescence event.

    The exp64 sweep discipline (adaptive step, linear prediction, jump rejection) applied to
    the *whole* fibre.  Two kinds of event are recorded: a **dip** event, when the closest
    alive pair's gap drops below ``dip`` (the pair is then marked dead -- a coalescing pair
    cannot be followed through the fold on the real axis); and a **jump** event, when several
    roots converge to solutions far from their predictions at the same node (the birth-fold
    signature: a complex pair merging into a real double root, or a steep death fold) -- the
    jumped roots are marked dead and paired by closest distance into events.  A single jumped
    root with no partner is recorded as a death, never silently carried.  Every accepted
    node is recorded, as are the running closest approaches of every pair that involves root
    1 or 6.

    Args:
        links: The fixed SR5 link transforms (complex).
        fibre: The frozen polished base fibre (complex).
        target0: The base pose.
        direction: Unit translation direction.
        span: Sweep length in metres.
        dip: Pair gap below which a dip event is recorded.
        jump_tolerance: Largest accepted distance from the prediction, radians.
        tolerance: Residual the corrector must reach for a step to count.

    Returns:
        ``(events, deaths, record, approaches)``: the event list (dicts with ``u``, ``i``,
        ``j``, ``gap``, ``kind`` and the two configurations), the single-root death list, the
        per-node record dict (same keys as :func:`walk_fibre_recorded`), and the best
        approach per unordered pair that involves root 1 or 6 as ``(gap, u)``.
    """
    n = len(fibre)
    qs = [np.asarray(root, dtype=complex).copy() for root in fibre]
    previous = [q.copy() for q in qs]
    alive = list(range(n))
    status: list[str | None] = [None] * n
    events: list[dict] = []
    deaths: list[dict] = []
    approaches: dict[tuple[int, int], tuple[float, float]] = {}

    u_values: list[float] = [0.0]
    record = _empty_record(1, n)

    for index in range(n):
        q_new, res = newton_complex(links, qs[index], target0)
        if res > tolerance or not np.all(np.isfinite(q_new)):
            status[index] = f"start residual {res:.1e}"
            continue
        record["correction"][0, index] = float(np.linalg.norm(q_new - qs[index]))
        record["residual"][0, index] = res
        previous[index] = qs[index]
        qs[index] = q_new
        record["actual_step"][0, index] = float(np.linalg.norm(qs[index] - previous[index]))
        record["roots_real"][0, index] = q_new.real
        record["roots_imag"][0, index] = q_new.imag
        record["alive"][0, index] = 1
    if len(alive) >= 2:
        gap, first, second = _closest_pair(qs, alive)
        record["min_gap"][0] = gap
        record["closest_pair"][0] = (first, second)

    step = 2e-3
    u = 0.0
    while u < span and len(alive) >= 2:
        trial = u + step
        pose = target_at(target0, direction, complex(trial))
        accepted: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
        died: list[tuple[int, float]] = []
        faults = 0
        for index in alive:
            prediction = 2.0 * qs[index] - previous[index]
            q_new, res = newton_complex(links, prediction, pose)
            if res > tolerance or not np.all(np.isfinite(q_new)):
                faults += 1
                continue
            jump = float(np.linalg.norm(q_new - prediction))
            if jump > jump_tolerance:
                died.append((index, jump))
                continue
            accepted[index] = (q_new, res, prediction)
        if faults:
            step *= 0.5
            if step < 1e-7:
                print(f"    sweep step floor reached at u={u:+.5f} (faults {faults})")
                break
            continue
        for index, jump in died:
            status[index] = f"sweep jump at u={trial:+.5f} (|dq|={jump:.2f})"
            deaths.append({"u": trial, "i": index, "jump": jump, "reason": status[index]})
            alive.remove(index)
        for index, (q_new, _res, _prediction) in accepted.items():
            previous[index] = qs[index]
            qs[index] = q_new
        u = trial
        u_values.append(u)
        record = _append_sweep_node(record, n)
        for index, (q_new, res, prediction) in accepted.items():
            record["residual"][-1, index] = res
            record["correction"][-1, index] = float(np.linalg.norm(q_new - prediction))
            record["actual_step"][-1, index] = float(np.linalg.norm(q_new - previous[index]))
            record["roots_real"][-1, index] = q_new.real
            record["roots_imag"][-1, index] = q_new.imag
            record["alive"][-1, index] = 1
        alive_now = list(alive)
        gap, first, second = _closest_pair(qs, alive_now)
        if len(alive_now) >= 2:
            record["min_gap"][-1] = gap
            record["closest_pair"][-1] = (first, second)
            for a, i in enumerate(alive_now):
                for j in alive_now[a + 1:]:
                    if i in (1, 6) or j in (1, 6):
                        key = (min(i, j), max(i, j))
                        distance = torus_distance_complex(qs[i], qs[j])
                        old = approaches.get(key)
                        if old is None or distance < old[0]:
                            approaches[key] = (distance, u)
        dip_pair: tuple[int, int] | None = None
        if gap < dip and len(alive_now) >= 2:
            dip_pair = (first, second)
            events.append({"u": u, "i": first, "j": second, "gap": gap, "kind": "dip",
                           "q_a": qs[first].copy(), "q_b": qs[second].copy()})
            alive.remove(first)
            alive.remove(second)
        if len(died) >= 2:
            for (a, b) in _pair_up(died, qs):
                if dip_pair is not None and {a, b} == set(dip_pair):
                    continue
                events.append({"u": u, "i": a, "j": b,
                               "gap": torus_distance_complex(qs[a], qs[b]), "kind": "jump",
                               "q_a": qs[a].copy(), "q_b": qs[b].copy()})
        step = min(step * 1.5, 4e-3)

    record["u"] = np.asarray(u_values, dtype=float)
    return events, deaths, record, approaches


def _events_json(events: list[dict]) -> str:
    """JSON-serialise sweep events with complex configurations split into real/imag lists.

    Args:
        events: The sweep event list (``q_a``/``q_b`` are complex arrays).

    Returns:
        The JSON string (complex arrays become ``[real_list, imag_list]`` pairs).
    """
    cleaned: list[dict] = []
    for event in events:
        item = dict(event)
        for key in ("q_a", "q_b"):
            if key in item:
                value = np.asarray(item[key])
                item[key] = [value.real.tolist(), value.imag.tolist()]
        cleaned.append(item)
    return json.dumps(cleaned)


def _pair_up(died: list[tuple[int, float]], qs: list[np.ndarray]) -> list[tuple[int, int]]:
    """Greedily pair roots that died at the same sweep node by closest torus distance.

    Args:
        died: The roots that jumped at the node, with their jump sizes (unused in pairing).
        qs: The last accepted configurations (the pre-jump states).

    Returns:
        The unordered pairs, closest first.
    """
    remaining = [index for index, _jump in died]
    pairs: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        best = min(((torus_distance_complex(qs[i], qs[j]), i, j)
                    for a, i in enumerate(remaining) for j in remaining[a + 1:]),
                   default=None)
        if best is None:
            break
        _gap, i, j = best
        pairs.append((min(i, j), max(i, j)))
        remaining.remove(i)
        remaining.remove(j)
    return pairs


def _append_sweep_node(record: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Append one NaN row to every sweep record array (kept allocation-light).

    Args:
        record: The sweep record dict being grown.
        n: Number of tracked roots.

    Returns:
        The same dict, with one extra node row per array.
    """
    node_count = record["roots_real"].shape[0]
    grown = _empty_record(node_count + 1, n)
    for key in grown:
        grown[key][:node_count] = record[key]
    if "u" in record:
        grown["u"] = record["u"]
    return grown


def refine_candidate(links: list[np.ndarray], target_at_fn: Callable[[complex], np.ndarray],
                     event: dict) -> tuple[complex, float]:
    """Refine one sweep event with the augmented system ``[err6; det J] = 0`` over ``(q, u)``.

    The two event configurations are aligned to a common 2 pi sheet before the midpoint seed
    (exp64's protocol), then :func:`refine_branch_pose` runs the trust-region Newton.

    Args:
        links: The fixed SR5 link transforms (complex).
        target_at_fn: ``u -> complex pose``.
        event: One sweep event dict (``u``, ``q_a``, ``q_b``).

    Returns:
        ``(u_star, residual)``.
    """
    q_a, q_b = event["q_a"], event["q_b"]
    shift = 2.0 * np.pi * np.round((q_a.real - q_b.real) / (2.0 * np.pi))
    q_b_aligned = q_b + shift
    seed_q = 0.5 * (q_a + q_b_aligned)
    y, residual = refine_branch_pose(links, target_at_fn, seed_q, event["u"])
    return complex(y[6]), float(residual)


def _sweep_label(direction: np.ndarray) -> str:
    """Short name of a sweep direction for array keys (``+x``, ``-x``, ``+y``, ``-y``)."""
    index = int(np.argmax(np.abs(direction)))
    sign = "+" if direction[index] > 0 else "-"
    return sign + "xyz"[index]


def _permutation_is_new(permutation: list[int]) -> bool:
    """True iff the permutation's support involves root 1 or 6 without being ``(1 6)``.

    A complete permutation that moves a new root while touching root 1 or 6 is exactly the
    deliverable of this module; composite permutations count (they are kept whole, per the
    audit).  ``(1 6)`` itself and the identity do not.

    Args:
        permutation: The image list (``-1`` entries make it False).

    Returns:
        Whether the permutation is complete, non-identity, not ``(1 6)``, and its support
        meets ``{1, 6}``.
    """
    if any(image < 0 for image in permutation):
        return False
    moved = {i for i, image in enumerate(permutation) if image != i}
    return bool(moved & {1, 6}) and moved != {1, 6}


def _agreement_checks(permutations: dict[str, list[int]], oks: dict[str, bool],
                      identity: list[int]) -> list[tuple[str, bool, str]]:
    """Build the four-loop Gate-A agreement checks for one protocol.

    Args:
        permutations: Per-tag permutations (main, reverse, empty, refined).
        oks: Per-tag completeness flags.
        identity: The identity permutation.

    Returns:
        The named check list ``(name, passed, detail)``.
    """
    checks: list[tuple[str, bool, str]] = []
    checks.append(("main is a complete permutation", oks["main"],
                   "all 16 matched, injective"))
    checks.append(("reverse is a complete permutation", oks["reverse"],
                   "all 16 matched, injective"))
    checks.append(("empty is a complete permutation", oks["empty"],
                   "all 16 matched, injective"))
    checks.append(("refined is a complete permutation", oks["refined"],
                   "all 16 matched, injective"))
    if oks["main"] and oks["reverse"]:
        checks.append(("reverse == main^-1",
                       permutations["reverse"] == inverse(permutations["main"]),
                       f"{permutations['reverse']} vs {inverse(permutations['main'])}"))
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


def main() -> int:
    """Run Gate A's generator hunt: sweep, refine, loop, and archive every step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="base-pose seed")
    parser.add_argument("--census-seeds", type=int, default=400, help="census seeds for the fibre")
    parser.add_argument("--span", type=float, default=1.5, help="sweep length, metres")
    parser.add_argument("--dip", type=float, default=5e-2,
                        help="pair gap below which the sweep records an event")
    parser.add_argument("--radius", type=float, default=5e-3, help="loop radius around the fold")
    parser.add_argument("--arc", type=float, default=0.05,
                        help="height of the approach arc above the real axis, metres")
    parser.add_argument("--samples", type=int, default=96, help="circle samples")
    parser.add_argument("--refined-samples", type=int, default=192,
                        help="circle samples for the step-refinement re-run")
    parser.add_argument("--jump-tolerance", type=float, default=0.5)
    parser.add_argument("--merge-floor", type=float, default=1e-2,
                        help="closest-pair gap below which a loop node is refused")
    parser.add_argument("--out", default="study/exp66_gate_a_generators.npz",
                        help="npz path for the raw data")
    args = parser.parse_args()

    links = fixed_sr5_links()
    model = sr0.robot(links)
    print("exp66: Gate A generators -- transpositions involving root 1 or 6\n")

    print("phase 0: mechanical-parameter fingerprint (the arm is fixed, never swept)")
    print(f"  18 translation components: {np.array2string(fingerprint(links), precision=6)}")
    print(f"  link 5 y offset (the SR0<->SR5 parameter): {links[4][1, 3]:.6f} m")
    target0, q_true = base_pose(args.seed, model)
    position = target0[:3, 3]
    print(f"  base position p0 = {np.array2string(position, precision=6)} m, "
          f"|p0| = {float(np.linalg.norm(position)):.4f} m")
    print(f"  sigma_min at the producing configuration: {iks.sigma_min(model, q_true):.3e}\n")

    print("phase 1: full 16-root fibre at the base pose, Newton-polished and frozen")
    fibre = build_full_fibre(links, target0, seeds=args.census_seeds)
    links_c = [link.astype(complex) for link in links]
    fibre = polish_fibre(links_c, fibre, target0)
    real_count = sum(1 for q in fibre if float(np.abs(q.imag).max()) < 1e-9)
    print(f"  {len(fibre)} roots: {real_count} real + {len(fibre) - real_count} complex")
    for index, q in enumerate(fibre):
        residual = float(np.linalg.norm(
            np.concatenate([fk_complex(links_c, q)[:3, 3] - target0[:3, 3],
                            (fk_complex(links_c, q)[:3, :3] - target0[:3, :3]).ravel()])))
        print(f"    root {index:2d}: residual {residual:.1e} | imag "
              f"{float(np.abs(q.imag).max()):.1e}")
    base_gap, base_first, base_second = _closest_pair(fibre, list(range(len(fibre))))
    print(f"  frozen table min pair gap {base_gap:.6f} rad (pair "
          f"{base_first},{base_second})")

    print("\nphase 2: sweep +-x/+-y for real coalescence events (all 16 roots tracked)")
    sweep_records: dict[str, dict[str, np.ndarray]] = {}
    all_events: dict[str, list[dict]] = {}
    all_deaths: dict[str, list[dict]] = {}
    all_approaches: dict[str, dict] = {}
    for name, direction in SWEEP_DIRECTIONS.items():
        print(f"  direction {name}:")
        events, deaths, record, approaches = directional_sweep(
            links_c, fibre, target0, direction, span=args.span, dip=args.dip,
            jump_tolerance=args.jump_tolerance)
        sweep_records[_sweep_label(direction)] = record
        all_events[name] = events
        all_deaths[name] = deaths
        all_approaches[name] = approaches
        for event in events:
            marker = ("CANDIDATE (1 or 6 with a new root)"
                      if (event["i"] in (1, 6)) != (event["j"] in (1, 6))
                      else "known (1 6)" if {event["i"], event["j"]} == {1, 6} else "")
            print(f"    event u={event['u']:+.5f} ({event['kind']}) pair "
                  f"({event['i']},{event['j']}) gap {event['gap']:.2e}  {marker}")
        for death in deaths:
            print(f"    death u={death['u']:+.5f} root {death['i']} ({death['reason']})")
        for (i, j), (gap, u) in sorted(approaches.items()):
            print(f"    best approach of pair ({i},{j}): gap {gap:.3e} at u={u:+.5f}")
        print(f"    {len(record['u'])} accepted nodes recorded")

    print("\nphase 2b: empty-loop radius probes per slice (near-base branch points)")
    empty_probes: dict[str, dict[str, dict]] = {}
    for name, direction in SWEEP_DIRECTIONS.items():
        target_at_fn = partial(target_at, target0, direction)
        probes: dict[str, dict] = {}
        for radius in (5e-3, 2e-3, 1e-3, 5e-4, 1e-4):
            loop = build_empty_loop(0.0, radius, samples=args.samples)
            perm, ok, _status, _q_end, record, failures = run_loop_quiet(
                links_c, fibre, loop, target_at_fn, jump_tolerance=args.jump_tolerance,
                merge_floor=args.merge_floor)
            moved = {i for i, image in enumerate(perm) if image != i}
            print(f"    {name} empty r={radius:g}: permutation {perm} complete {ok} "
                  f"moved {sorted(moved)} failures {len(failures)}")
            probes[f"r{radius:g}"] = {"radius": radius, "u_points": loop,
                                      "permutation": perm, "ok": ok, "record": record,
                                      "failures": failures}
        empty_probes[name] = probes

    print("\nphase 3: refine every 1/6-involving event with [err6; det J] = 0 over (q, u)")
    folds: list[dict] = []
    for name, direction in SWEEP_DIRECTIONS.items():
        for event in all_events[name]:
            involves_one = (event["i"] in (1, 6)) != (event["j"] in (1, 6))
            is_16 = {event["i"], event["j"]} == {1, 6}
            if not (involves_one or is_16):
                continue

            target_at_fn = partial(target_at, target0, direction)

            u_star, residual = refine_candidate(links_c, target_at_fn, event)
            print(f"  {name} pair ({event['i']},{event['j']}) at u={event['u']:+.5f}: "
                  f"refined u* = {u_star.real:+.6f}{u_star.imag:+.1e}i "
                  f"(augmented residual {residual:.1e})")
            refined = residual < 1e-8 and abs(u_star.imag) < 1e-3
            folds.append({
                "name": name, "direction": direction, "pair": (event["i"], event["j"]),
                "event_u": event["u"], "event_gap": event["gap"], "event_kind": event["kind"],
                "u_star": u_star, "residual": residual, "refined": refined,
                "shared_with": None, "radius": args.radius,
            })
            if not refined:
                print("    refinement failed or not real -- recorded, not looped")

    # Shared critical values: folds refining to the same u* are one product permutation,
    # tracked together, never split by renumbering.
    for a, fold in enumerate(folds):
        for b, other in enumerate(folds):
            if b >= a or not (fold["refined"] and other["refined"]):
                continue
            if abs(fold["u_star"] - other["u_star"]) < 1e-3:
                fold["shared_with"] = other["pair"]
                other["shared_with"] = fold["pair"]
    looped = [fold for fold in folds if fold["refined"] and fold["shared_with"] is None]
    shared = [fold for fold in folds if fold["refined"] and fold["shared_with"] is not None]
    print(f"  refined folds: {len(looped)} looped + {len(shared)} merged into shared "
          f"critical values (tracked as one product permutation)")

    print("\nphase 4: closed loops (main, reverse, empty, 96->192 refined) per fold")
    results: list[dict] = []
    scan_radii = sorted({args.radius, 1.5e-2, 3e-2, 1e-3, 1e-4})
    for fold in looped:
        direction = np.asarray(fold["direction"], dtype=float)
        u_star = fold["u_star"]
        pair = fold["pair"]

        target_at_fn = partial(target_at, target0, direction)

        print(f"  fold from {fold['name']} pair {pair}: u* = "
              f"{u_star.real:+.6f}{u_star.imag:+.1e}i")
        # Radius scan: the default circle, the merge-guard escalation ladder, and the
        # isolation ladder that separates a shared cluster's members (all archived).
        scan: dict[str, dict] = {}
        for radius in scan_radii:
            if u_star.real <= radius:
                continue
            loop = build_loop(0.0, u_star, radius, samples=args.samples,
                              orientation=1.0, arc=args.arc)
            perm, ok, status, q_end, record, failures = run_loop_recorded(
                links_c, fibre, loop, target_at_fn,
                label=f"scan main loop (radius {radius:g})",
                jump_tolerance=args.jump_tolerance, merge_floor=args.merge_floor)
            scan[f"r{radius:g}"] = {"radius": radius, "u_points": loop,
                                    "permutation": perm, "ok": ok, "status": status,
                                    "q_end": q_end, "record": record, "failures": failures}
        net_radius = next((radius for radius in (args.radius, 1.5e-2, 3e-2)
                           if f"r{radius:g}" in scan and scan[f"r{radius:g}"]["ok"]), None)
        if net_radius is None:
            print("    no complete loop at any radius -- fold recorded, not delivered")
            results.append({"fold": fold, "fold_index": folds.index(fold), "scan": scan,
                            "net_radius": None, "loops": {}, "checks": [], "all_ok": False,
                            "isolation_radius": None, "iso": {}, "iso_checks": [],
                            "iso_all_ok": False, "reason": "no complete loop"})
            continue
        fold["radius"] = net_radius
        net_main = scan[f"r{net_radius:g}"]
        isolation_radius = next(
            (radius for radius in scan_radii
             if f"r{radius:g}" in scan and scan[f"r{radius:g}"]["ok"]
             and radius < net_radius
             and scan[f"r{radius:g}"]["permutation"] != net_main["permutation"]),
            None)

        net_loop = net_main["u_points"]
        reverse_loop = list(reversed(net_loop))
        # The empty loop must enclose no branch point; the per-slice probes showed off-axis
        # branch points as close as ~1e-3 to the base (the -x (14,15) point), so the protocol
        # radius is the probe floor 1e-4 and any violation fails the check honestly.
        empty_radius = 1e-4
        empty_loop = build_empty_loop(0.0, empty_radius, samples=args.samples)
        refined_loop = build_loop(0.0, u_star, net_radius, samples=args.refined_samples,
                                  orientation=1.0, arc=args.arc)
        print(f"    net protocol at radius {net_radius:g}: loop length {len(net_loop)}, "
              f"reverse {len(reverse_loop)}, empty (r={empty_radius:g}) {len(empty_loop)}, "
              f"refined {len(refined_loop)}")
        print("    parameter trajectory (u, real+imag) -- first/last/circle samples:")
        for index in (0, 1, len(net_loop) // 2, len(net_loop) - 2, len(net_loop) - 1):
            u = net_loop[index]
            print(f"      u[{index}] = ({u.real:+.6f}, {u.imag:+.6f})")

        loops: dict[str, dict] = {"main": net_main}
        permutations = {"main": net_main["permutation"]}
        oks = {"main": net_main["ok"]}
        for tag, u_points in (("reverse", reverse_loop), ("empty", empty_loop),
                              ("refined", refined_loop)):
            perm, ok, status, q_end, record, failures = run_loop_recorded(
                links_c, fibre, u_points, target_at_fn, label=f"{tag} loop",
                jump_tolerance=args.jump_tolerance, merge_floor=args.merge_floor)
            loops[tag] = {"u_points": u_points, "record": record, "failures": failures,
                          "status": status, "permutation": perm, "ok": ok, "q_end": q_end}
            permutations[tag] = perm
            oks[tag] = ok
        identity = list(range(len(fibre)))
        print("  net consistency checks:")
        checks = _agreement_checks(permutations, oks, identity)
        all_ok = _print_checks(checks)
        print(f"  complete net permutation: {permutations['main']} "
              f"(cycle type {cycle_type(permutations['main'])})")

        iso: dict[str, dict] = {}
        iso_checks: list[tuple[str, bool, str]] = []
        iso_all_ok = True
        if isolation_radius is not None:
            iso_main = scan[f"r{isolation_radius:g}"]
            iso_loop = iso_main["u_points"]
            iso_reverse = list(reversed(iso_loop))
            iso_empty = build_empty_loop(0.0, 1e-4, samples=args.samples)
            iso_refined = build_loop(0.0, u_star, isolation_radius,
                                     samples=args.refined_samples, orientation=1.0,
                                     arc=args.arc)
            print(f"    isolation protocol at radius {isolation_radius:g} "
                  f"(the net permutation's support exceeds the fold pair)")
            iso = {"main": iso_main}
            iso_permutations = {"main": iso_main["permutation"]}
            iso_oks = {"main": iso_main["ok"]}
            for tag, u_points in (("reverse", iso_reverse), ("empty", iso_empty),
                                  ("refined", iso_refined)):
                perm, ok, status, q_end, record, failures = run_loop_recorded(
                    links_c, fibre, u_points, target_at_fn, label=f"iso {tag} loop",
                    jump_tolerance=args.jump_tolerance, merge_floor=args.merge_floor)
                iso[tag] = {"u_points": u_points, "record": record, "failures": failures,
                            "status": status, "permutation": perm, "ok": ok, "q_end": q_end}
                iso_permutations[tag] = perm
                iso_oks[tag] = ok
            print("  isolation consistency checks:")
            iso_checks = _agreement_checks(iso_permutations, iso_oks, identity)
            iso_all_ok = _print_checks(iso_checks)
            print(f"  complete isolation permutation: {iso_permutations['main']} "
                  f"(cycle type {cycle_type(iso_permutations['main'])})")

        results.append({"fold": fold, "fold_index": folds.index(fold), "scan": scan,
                        "net_radius": net_radius, "loops": loops, "checks": checks,
                        "all_ok": all_ok, "isolation_radius": isolation_radius,
                        "iso": iso, "iso_checks": iso_checks, "iso_all_ok": iso_all_ok})

    print("\nphase 5: archive the per-step raw data")
    _dump_raw_data(args.out, links_c, target0, fibre, base_gap, (base_first, base_second),
                   sweep_records, all_events, all_deaths, all_approaches, empty_probes,
                   folds, results, args)
    delivered = [r for r in results if r["all_ok"]]
    iso_delivered = [r for r in results
                     if r["iso_all_ok"] and r["isolation_radius"] is not None]
    new_delivered = [r for r in delivered
                     if _permutation_is_new(r["loops"]["main"]["permutation"])]
    new_iso = [r for r in iso_delivered
               if _permutation_is_new(r["iso"]["main"]["permutation"])]
    print(f"  delivered {len(delivered)}/{len(results)} net protocols with all checks "
          f"passing; {len(iso_delivered)} isolation protocols")
    for result in delivered:
        perm = result["loops"]["main"]["permutation"]
        moved = {i for i, image in enumerate(perm) if image != i}
        print(f"  net permutation {perm} (cycle type {cycle_type(perm)}, moved {sorted(moved)}, "
              f"involves 1/6 with a new root: {_permutation_is_new(perm)})")
    for result in iso_delivered:
        perm = result["iso"]["main"]["permutation"]
        moved = {i for i, image in enumerate(perm) if image != i}
        print(f"  iso permutation {perm} (cycle type {cycle_type(perm)}, moved {sorted(moved)}, "
              f"involves 1/6 with a new root: {_permutation_is_new(perm)})")
    if not (new_delivered or new_iso):
        print("FAIL: no new permutation (support involving 1 or 6, not (1 6)) passed all "
              "self-checks -- see the records above")
        return 1
    print("PASS: at least one new Gate A permutation delivered with all self-checks passing")
    return 0


def _dump_raw_data(path: str, links: list[np.ndarray], target0: np.ndarray,
                   fibre: list[np.ndarray], base_gap: float, base_pair: tuple[int, int],
                   sweep_records: dict[str, dict[str, np.ndarray]],
                   all_events: dict[str, list[dict]],
                   all_deaths: dict[str, list[dict]],
                   all_approaches: dict[str, dict],
                   empty_probes: dict[str, dict[str, dict]],
                   folds: list[dict], results: list[dict], args) -> None:
    """Write the full Gate-A-generator record (sweeps, loops, failures) to an npz.

    Complex values are stored as separate real and imaginary arrays, never truncated.  For
    every sweep node and every loop node (main, reverse, empty, refined) the archive holds
    the 16-root trajectory, per-root correction/residual, the closest-pair gap and pair, the
    actual step taken, the alive flags and the failure records; each loop additionally holds
    its permutation and the four-loop agreement outcome.

    Args:
        path: Output ``.npz`` path.
        links: The fixed SR5 link transforms (complex).
        target0: Base 4x4 pose.
        fibre: The frozen polished base fibre (complex).
        base_gap: Smallest pair gap of the frozen table.
        base_pair: The pair attaining ``base_gap``.
        sweep_records: Per-direction sweep record dicts.
        all_events: Per-direction event lists (dip and jump kinds).
        all_deaths: Per-direction single-root death lists.
        all_approaches: Per-direction best-approach tables.
        empty_probes: Per-direction empty-loop radius probes (permutation per radius).
        folds: The fold dicts (refined and not).
        results: The looped-fold result dicts.
        args: The parsed arguments (for the summary).
    """
    fibre_arr = np.asarray(fibre, dtype=complex)
    payload: dict[str, np.ndarray] = {
        "target0": np.asarray(target0, dtype=complex),
        "directions": np.asarray(list(SWEEP_DIRECTIONS.values()), dtype=float),
        "fibre_real": fibre_arr.real,
        "fibre_imag": fibre_arr.imag,
        "fibre_residual": np.array(
            [float(np.linalg.norm(residual12(links, q, target0))) for q in fibre]),
        "base_min_gap": np.asarray(base_gap),
        "base_closest_pair": np.asarray(base_pair, dtype=int),
    }
    for key, record in sweep_records.items():
        payload[f"sweep_{key}_u"] = record["u"]
        for array_key in ("roots_real", "roots_imag", "residual", "correction",
                          "actual_step", "min_gap", "closest_pair", "alive"):
            payload[f"sweep_{key}_{array_key}"] = record[array_key]
        payload[f"sweep_{key}_events_json"] = np.asarray(_events_json(all_events[key]))
        payload[f"sweep_{key}_deaths_json"] = np.asarray(json.dumps(all_deaths[key]))
        payload[f"sweep_{key}_approaches_json"] = np.asarray(json.dumps(
            {f"{i},{j}": value for (i, j), value in all_approaches[key].items()}))

    for direction_name, probes in empty_probes.items():
        for probe_tag, probe in probes.items():
            _store_loop(payload, f"emptyprobe_{direction_name}_{probe_tag}", probe)

    fold_table: list[dict] = []
    for fold in folds:
        fold_table.append({
            "direction": fold["name"], "pair": list(fold["pair"]),
            "event_u": fold["event_u"], "event_gap": fold["event_gap"],
            "event_kind": fold["event_kind"],
            "u_star": [fold["u_star"].real, fold["u_star"].imag],
            "refine_residual": fold["residual"], "refined": fold["refined"],
            "shared_with": fold["shared_with"], "radius": fold["radius"],
        })
    payload["folds_json"] = np.asarray(json.dumps(fold_table))

    for index, result in enumerate(results):
        fold = result["fold"]
        prefix = f"fold{index}"
        payload[f"{prefix}_u_star"] = np.asarray(complex(fold["u_star"]))
        payload[f"{prefix}_direction"] = np.asarray(fold["direction"], dtype=float)
        payload[f"{prefix}_net_radius"] = np.asarray(
            result["net_radius"] if result["net_radius"] is not None else float("nan"))
        payload[f"{prefix}_isolation_radius"] = np.asarray(
            result["isolation_radius"] if result["isolation_radius"] is not None
            else float("nan"))
        for scan_tag, loop in result["scan"].items():
            _store_loop(payload, f"{prefix}_scan_{scan_tag}", loop)
        for tag, loop in result["loops"].items():
            _store_loop(payload, f"{prefix}_{tag}", loop)
        for tag, loop in result["iso"].items():
            _store_loop(payload, f"{prefix}_iso_{tag}", loop)
        payload[f"{prefix}_checks_json"] = np.asarray(json.dumps(
            [[name, passed, detail] for name, passed, detail in result["checks"]]))
        payload[f"{prefix}_all_ok"] = np.asarray(result["all_ok"])
        payload[f"{prefix}_iso_checks_json"] = np.asarray(json.dumps(
            [[name, passed, detail] for name, passed, detail in result["iso_checks"]]))
        payload[f"{prefix}_iso_all_ok"] = np.asarray(result["iso_all_ok"])

    payload["summary_json"] = np.asarray(json.dumps({
        "seed": args.seed, "census_seeds": args.census_seeds, "span": args.span,
        "dip": args.dip, "radius": args.radius, "arc": args.arc, "samples": args.samples,
        "refined_samples": args.refined_samples,
        "jump_tolerance": args.jump_tolerance, "merge_floor": args.merge_floor,
        "base_min_gap": base_gap, "base_pair": list(base_pair),
        "machine_verified": False,
    }))
    np.savez_compressed(path, **payload)
    print(f"  raw data written to {path} "
          f"({sum(value.nbytes for value in payload.values()) / 1e6:.1f} MB)")


def _store_loop(payload: dict[str, np.ndarray], prefix: str, loop: dict) -> None:
    """Archive one loop's per-node arrays under ``prefix``.

    Args:
        payload: The npz payload being built.
        prefix: Array-name prefix (e.g. ``fold0_main``).
        loop: The loop dict (``u_points``, ``record``, ``permutation``, ``ok``,
            ``failures``).
    """
    u_points = loop["u_points"]
    record = loop["record"]
    payload[f"{prefix}_u_real"] = np.asarray([u.real for u in u_points], dtype=float)
    payload[f"{prefix}_u_imag"] = np.asarray([u.imag for u in u_points], dtype=float)
    for array_key in ("roots_real", "roots_imag", "residual", "correction",
                      "actual_step", "min_gap", "closest_pair", "alive", "merge_risk"):
        payload[f"{prefix}_{array_key}"] = record[array_key]
    payload[f"{prefix}_permutation"] = np.asarray(loop["permutation"], dtype=int)
    payload[f"{prefix}_complete"] = np.asarray(loop["ok"])
    payload[f"{prefix}_failures_json"] = np.asarray(json.dumps(loop["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
