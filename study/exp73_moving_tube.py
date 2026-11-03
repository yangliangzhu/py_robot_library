"""Benchmark exact moving-center certificates on the root-13 initial a-loop edge."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from study import exp70_gate_c1_roots as core
from study import exp71_gate_c2_segment as seg
from study import exp72_gate_c3a_edge as edge

OUT = Path('study/exp73_moving_tube.json')


def directional(poly: dict, velocity: tuple) -> dict:
    """Differentiate a Laurent polynomial along a frozen complex velocity."""
    result = core.lp_zero()
    for j in range(6):
        result = core.lp_add(result, core.lp_scale(core.lp_derivative(poly, j), velocity[j]))
    return result


def swept_box(c: tuple, v: tuple, h: Q, r: Q) -> tuple:
    """Enclose all c+s*v+e with |s|<=h and real-component |e|<=r."""
    return tuple(core.CInterval(*(
        core.RInterval(c[2*j+k]-h*abs(v[2*j+k])-r,
                       c[2*j+k]+h*abs(v[2*j+k])+r) for k in range(2))) for j in range(6))


def pow2_upper(x: Q) -> Q:
    """Return a power of two at least x, without floating arithmetic."""
    r = Q(1)
    while r < x:
        r *= 2
    while r/2 >= x and r > Q(1, 2**48):
        r /= 2
    return r


def evaluate(equations: list, jac: list, smat: list, record: dict) -> dict:
    """Recompute exact sufficient bounds from frozen inputs, not stored verdicts."""
    start = time.perf_counter()
    c = tuple(map(Q, record['centre']))
    v = tuple(map(Q, record['velocity']))
    b = [list(map(Q, row)) for row in record['B']]
    h = Q(record['length'])/2
    with np.load(edge.A66_ARCHIVE, allow_pickle=False) as archive:
        endpoint = Q.from_float(float(archive['fold3_main_u_imag'][1]))
        if archive['fold3_main_u_real'][1] != 0:
            raise ValueError('initial archived edge is not imaginary')
    if not 0 < 2*h <= endpoint:
        raise ValueError('length outside initial archived edge')
    if len(c) != 12 or len(v) != 12 or len(b) != 12 or any(len(row) != 12 for row in b):
        raise ValueError('expected 12-dimensional realified inputs')
    zs = core.zs_from_x(c)
    velocity = core.zs_from_x(v)
    direction = edge.load_a_direction()
    w = edge.build_w(jac, b)
    base = core.matvec(b, seg.eval_f_lam(equations, c, (Q(0), h), direction))
    forcing = core.matvec(b, seg.shift_vector(direction, (Q(0), Q(1))))
    center_box = swept_box(c, v, h, Q(0))
    cache = {}
    eta = []
    for i in range(12):
        first = core.lp_zero()
        for j in range(6):
            first = core.lp_add(first, core.lp_scale(w[i][j], velocity[j]))
        defect = core.lp_eval(first, zs)[0] - forcing[i]
        second = directional(first, velocity)
        curvature = core.ri_abs_upper(core.lp_eval_interval(second, center_box, cache).re)
        eta.append(abs(base[i])+h*abs(defect)+h*h*curvature/2)
    r = Q(record['radius']) if 'radius' in record else pow2_upper(4*max(eta))
    if r <= 0:
        raise ValueError('radius must be positive')
    box = swept_box(c, v, h, r)
    cache_center, cache_swept = {}, {}
    rows = []
    s_interval = core.RInterval(-h, h)
    error_interval = core.CInterval(core.RInterval(-r, r), core.RInterval(-r, r))
    for i in range(12):
        row = Q(0)
        for j in range(6):
            poly = w[i][j]
            p = core.lp_eval(poly, zs)
            tangent = core.lp_eval_interval(directional(poly, velocity), center_box, cache_center)
            value = core.CInterval(
                core.ri_add(core.ri_point(p[0]), core.ri_mul(s_interval, tangent.re)),
                core.ri_add(core.ri_point(p[1]), core.ri_mul(s_interval, tangent.im)))
            for k in range(6):
                derivative = core.lp_eval_interval(core.lp_derivative(poly, k), box, cache_swept)
                value = core.ci_add(value, core.ci_mul(derivative, error_interval))
            row += core.ri_abs_upper(core.ri_sub(core.ri_point(Q(i == 2*j)), value.re))
            row += core.ri_abs_upper(core.ri_add(core.ri_point(Q(i == 2*j+1)), value.im))
        rows.append(row)
    # Compare the old natural interval W evaluation on exactly the same swept box.
    natural = max(edge.kappa_row_combined(w, c, r, box=box))
    # Conservative isotropic envelope only for pole/rotation exclusion.
    envelope = r+h*max(map(abs, v))
    trace, _ = core.exclusion_bounds(smat, c, envelope)
    z_lower = []
    for z in core.box_intervals(c, envelope):
        z_lower.append(core.ri_add(core.ri_mul(z.re, z.re), core.ri_mul(z.im, z.im)).lo)
    margins = [1-eta[i]/r-rows[i] for i in range(12)]
    c1 = json.loads(core.CERT_JSON.read_text())['roots']['13']
    start_centre = tuple(c[i]-h*v[i] for i in range(12))
    c1_centre = tuple(map(Q, c1['centre']))
    entry_margin = min(r-abs(start_centre[i]-c1_centre[i])-Q(c1['radius']) for i in range(12))
    passed = (max(rows) < Q(1, 2) and min(margins) >= Q(1, 4)
              and trace < 4 and min(z_lower) > 0 and entry_margin > 0)
    return {'radius': str(r), 'eta': list(map(str, eta)), 'kappa_rows': list(map(str, rows)),
            'kappa': str(max(rows)), 'natural_kappa': str(natural),
            'margin_relative': str(min(margins)), 'trace': str(trace),
            'z_squared_lower': list(map(str, z_lower)), 'entry_margin': str(entry_margin),
            'passed': passed, 'seconds': time.perf_counter()-start}


def setup() -> tuple:
    """Load the exact model and check its provenance."""
    manifest, checks = core.check_model_provenance()
    if not all(item[1] for item in checks):
        raise ValueError('model provenance failed')
    return core.build_system(core.manifest_links(manifest),
                             [list(map(Q, row)) for row in manifest['target_rotation']],
                             list(map(Q, manifest['target_position'])))


def main() -> None:
    """Generate or exactly recheck a bounded ladder of moving tubes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--powers', nargs='+', type=int, default=[23, 20, 17])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    equations, jac, smat = setup()
    source_paths = [core.MODEL_JSON, core.CERT_JSON, edge.A66_ARCHIVE,
                    Path('study/exp73_moving_tube.py'),
                    Path('study/exp70_gate_c1_roots.py'),
                    Path('study/exp71_gate_c2_segment.py'),
                    Path('study/exp72_gate_c3a_edge.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.verify:
        payload = json.loads(args.verify.read_text())
        if payload['hashes'] != hashes or not payload['trials'] or payload['root'] != 13:
            raise ValueError('provenance mismatch or empty trials')
        for record in payload['trials']:
            result = evaluate(equations, jac, smat, record)
            for key, value in result.items():
                if key != 'seconds' and value != record['bounds'][key]:
                    raise ValueError(f'recomputed {key} mismatch')
            print('verified', record['length'], result['passed'], flush=True)
        return
    with np.load(core.SEED_NPZ, allow_pickle=False) as data:
        z0 = np.exp(1j*(data['fibre_real'][13]+1j*data['fibre_imag'][13]))
    trials = []
    for power in args.powers:
        length = Q(1, 2**power)
        if not 0 < length <= Q.from_float(0.0005263157894736842):
            raise ValueError('length outside initial archived edge')
        h = length/2
        direction = edge.load_a_direction()
        z = seg.newton_z(equations, jac, z0, (Q(0), h), direction)
        jf = np.array([[core.lp_eval_float(p, z) for p in row] for row in jac])
        velocity = np.linalg.solve(jf, np.array([0, -1j, 0, 0, 0, 0]))
        c, v = seg.freeze_z(z), seg.freeze_z(velocity)
        b = seg.preconditioner_at(equations, jac, c, direction, (Q(0), h))
        record = {'length': str(length), 'centre': list(map(str, c)),
                  'velocity': list(map(str, v)), 'B': [list(map(str, row)) for row in b]}
        bounds = evaluate(equations, jac, smat, record)
        record.update(radius=bounds['radius'], bounds=bounds)
        trials.append(record)
        args.out.write_text(json.dumps({'hashes': hashes, 'root': 13, 'trials': trials}, indent=2)+'\n')
        print('length', float(length), 'pass', bounds['passed'],
              'kappa', float(Q(bounds['kappa'])), 'natural', float(Q(bounds['natural_kappa'])),
              'margin', float(Q(bounds['margin_relative'])), 'seconds', bounds['seconds'], flush=True)


if __name__ == '__main__':
    main()
