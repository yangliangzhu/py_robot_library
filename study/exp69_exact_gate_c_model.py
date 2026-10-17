"""Freeze exact rational input for Gate C and prove generic FK rank six.

Run ``python3 -m study.exp69_exact_gate_c_model``. SymPy performs exact
algebra here; numerical trajectories are not certified by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
import yaml


def rotation_and_derivative(z: sp.Expr) -> tuple[sp.Matrix, sp.Matrix]:
    """Return Laurent rotation Rz and its derivative with respect to z."""
    cosine, sine = (z + 1/z)/2, (z - 1/z)/(2*sp.I)
    dc, ds = (1 - 1/z**2)/2, (1 + 1/z**2)/(2*sp.I)
    rotation = sp.Matrix([[cosine, -sine, 0, 0], [sine, cosine, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]])
    derivative = sp.Matrix([[dc, -ds, 0, 0], [ds, dc, 0, 0],
                            [0, 0, 0, 0], [0, 0, 0, 0]])
    return rotation, derivative


def rational_matrix(matrix: sp.Matrix) -> list[list[str]]:
    """Serialize exact matrix entries without decimal rounding."""
    return [[str(sp.cancel(matrix[i, j])) for j in range(matrix.cols)]
            for i in range(matrix.rows)]


def main() -> int:
    """Write an exact model manifest and check algebraic rank and root labels."""
    config_path = Path('model/configs/rokae_sr5.yaml')
    config = yaml.safe_load(config_path.read_text())
    # This file uses short terminating decimals and signed permutation rotations.
    links = [sp.Matrix([[sp.Rational(str(x)) for x in row] for row in matrix])
             for matrix in config['mat']]
    for matrix in links:
        assert matrix[:3, :3].T * matrix[:3, :3] == sp.eye(3)
        assert matrix[:3, :3].det() == 1

    archive_path = Path('study/exp68_h_loop.npz')
    with np.load(archive_path, allow_pickle=False) as data:
        target = data['target0'].real
    r_float = target[:3, :3]
    r_stored = sp.Matrix([[sp.Rational(float(x)) for x in row] for row in r_float])
    stored_orthogonal = r_stored.T*r_stored == sp.eye(3)
    cayley_float = np.array([
        r_float[2, 1]-r_float[1, 2], r_float[0, 2]-r_float[2, 0],
        r_float[1, 0]-r_float[0, 1],
    ]) / (1 + np.trace(r_float))
    w = [sp.Rational(float(x)) for x in cayley_float]
    skew = sp.Matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    r_exact = (sp.eye(3)+skew)*(sp.eye(3)-skew).inv()
    assert r_exact.T*r_exact == sp.eye(3)
    assert r_exact.det() == 1
    position = [sp.Rational(float(x)) for x in target[:3, 3]]

    # A rational complex point, unrelated to a floating approximate IK root.
    witness = [sp.Integer(j) + sp.I for j in range(1, 7)]
    rotations = [rotation_and_derivative(z) for z in witness]
    columns = []
    for j in range(6):
        derivative_fk = sp.eye(4)
        for k in range(6):
            derivative_fk = derivative_fk * links[k] * rotations[k][int(k == j)]
            derivative_fk = derivative_fk.applyfunc(sp.cancel)
        # Differentiate six regular functions on SE(3,C): p and skew(R).
        column = list(derivative_fk[:3, 3]) + [
            (derivative_fk[2, 1]-derivative_fk[1, 2])/2,
            (derivative_fk[0, 2]-derivative_fk[2, 0])/2,
            (derivative_fk[1, 0]-derivative_fk[0, 1])/2,
        ]
        columns.append(sp.Matrix(column))
    jacobian = sp.Matrix.hstack(*columns)
    determinant = sp.cancel(jacobian.det(method='domain-ge'))
    assert determinant != 0

    indices = [1, 4, 6, 8, 10, 13]
    archive66 = np.load('study/exp66_gate_a_generators.npz', allow_pickle=False)
    archive68 = np.load(archive_path, allow_pickle=False)
    generators = [archive66['fold3_main_permutation'], archive66['fold2_main_permutation'],
                  archive68['main_permutation']]
    for permutation in generators:
        assert sorted(int(permutation[j]) for j in indices) == indices

    manifest = {
        'geometry_policy': 'short decimal lengths in YAML interpreted as exact rationals',
        'links': [rational_matrix(matrix) for matrix in links],
        'target_rotation': rational_matrix(r_exact),
        'target_position': [str(x) for x in position],
        'target_cayley': [str(x) for x in w],
        'tracked_indices': indices,
        'stored_rotation_exactly_orthogonal': stored_orthogonal,
        'rank_witness_z': [str(x) for x in witness],
        'rank_witness_jacobian': rational_matrix(jacobian),
        'rank_witness_determinant': str(determinant),
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [config_path, archive_path, Path('study/exp66_gate_a_generators.npz')]
        },
        'path_certified': False,
    }
    Path('study/exp69_exact_model.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Stored rotation exactly orthogonal:', stored_orthogonal)
    print('Exact replacement R^T R=I and det R=1: passed')
    print('Numerical rotation change:', np.linalg.norm(np.array(r_exact.tolist(), float)-r_float))
    print('Exact rank-six witness determinant nonzero:', determinant != 0)
    print('Six-root subset invariant under a,d,h:', indices)
    print('Manifest saved; path certification NOT performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
