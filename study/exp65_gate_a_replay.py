"""Audit and replay the archived fixed-SR5 loop without rebuilding its fibre.

Run ``python3 -m study.exp65_gate_a_replay``. This checks numerical evidence,
not certified continuation. The Newton corrector is shared with exp64.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from study.exp37_complex_loop import newton_complex, residual12
from study.exp64_pose_loop_gate_a import fixed_sr5_links, read_permutation, target_at


def pair_gap(roots: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return the smallest complex angle distance modulo real 2*pi shifts."""
    differences = roots[:, None, :] - roots[None, :, :]
    differences -= 2 * np.pi * np.round(differences.real / (2 * np.pi))
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return float(distances[i, j]), (int(i), int(j))


def main() -> int:
    """Polish base roots, replay the loop, and save all accepted root samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='study/exp64_gate_a.npz')
    parser.add_argument('--out', default='study/exp65_gate_a_audit.npz')
    args = parser.parse_args()
    data = np.load(args.input)
    links = [matrix.astype(complex) for matrix in fixed_sr5_links()]
    fibre = data['fibre_real'] + 1j * data['fibre_imag']
    endpoint = data['q_end_real'] + 1j * data['q_end_imag']
    target, direction = data['target0'], data['direction']
    parameters = data['loop_u_real'] + 1j * data['loop_u_imag']
    polished = np.array([
        newton_complex(links, q, target, tolerance=1e-13)[0] for q in fibre
    ])
    base_residual = np.array([np.linalg.norm(residual12(links, q, target)) for q in fibre])
    polished_residual = np.array([
        np.linalg.norm(residual12(links, q, target)) for q in polished
    ])
    shifts = endpoint - polished[data['permutation']]
    shifts -= 2 * np.pi * np.round(shifts.real / (2 * np.pi))
    previous, current = polished.copy(), polished.copy()
    trace = [current.copy()]
    residuals = [polished_residual]
    corrections = [np.zeros(len(fibre))]
    steps = [np.zeros(len(fibre))]
    gaps = [pair_gap(current)[0]]
    pairs = [pair_gap(current)[1]]
    for index, parameter in enumerate(parameters[1:], start=1):
        prediction = 2 * current - previous
        candidates, errors = [], []
        for root_index, seed in enumerate(prediction):
            q, error = newton_complex(
                links, seed, target_at(target, direction, parameter),
                iterations=40, tolerance=1e-9,
            )
            if error > 1e-9 or not np.all(np.isfinite(q)):
                raise RuntimeError(f'correction failed at sample {index}, root {root_index}')
            candidates.append(q)
            errors.append(error)
        candidates = np.array(candidates)
        correction = np.linalg.norm(candidates - prediction, axis=1)
        if np.max(correction) > .5:
            raise RuntimeError(f'correction exceeded jump limit at sample {index}')
        steps.append(np.linalg.norm(candidates - current, axis=1))
        previous, current = current, candidates
        trace.append(current.copy())
        residuals.append(np.array(errors))
        corrections.append(correction)
        gap, pair = pair_gap(current)
        gaps.append(gap)
        pairs.append(pair)
    permutation, ok = read_permutation(list(current), [None] * len(fibre), list(polished))
    summary = {
        'base_residual_max': float(np.max(base_residual)),
        'polished_residual_max': float(np.max(polished_residual)),
        'base_min_gap': pair_gap(polished)[0],
        'archived_endpoint_to_polished_max': float(np.max(np.linalg.norm(shifts, axis=1))),
        'max_absolute_base_real_angle': float(np.max(np.abs(fibre.real))),
        'min_abs_exp_iq': float(np.min(np.abs(np.exp(1j * fibre)))),
        'max_abs_exp_iq': float(np.max(np.abs(np.exp(1j * fibre)))),
        'sample_min_gap': float(min(gaps)),
        'sample_min_gap_index': int(np.argmin(gaps)),
        'max_residual': float(np.max(residuals)),
        'max_correction': float(np.max(corrections)),
        'max_actual_step': float(np.max(steps)),
        'permutation': permutation,
        'matches_archive': bool(ok and np.array_equal(permutation, data['permutation'])),
        'machine_verified': False,
    }
    np.savez_compressed(
        args.out, u=parameters, roots=np.array(trace), residual=np.array(residuals),
        correction=np.array(corrections), actual_step=np.array(steps),
        min_gap=np.array(gaps), closest_pair=np.array(pairs), links=np.array(links),
        target0=target, direction=direction, permutation=np.array(permutation),
        summary_json=np.array(json.dumps(summary)),
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary['matches_archive'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
