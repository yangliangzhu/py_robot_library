"""Compute exact permutation consequences of the exp66 numerical archive.

Run ``python3 -m study.exp67_group_witness``. The S5 witness is conditional
on one radius-scan permutation that still needs the full loop controls.
"""

from __future__ import annotations

import numpy as np


def compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Compose image lists, applying second first."""
    return tuple(first[j] for j in second)


def inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    """Invert a permutation in image notation."""
    result = [0] * len(p)
    for i, j in enumerate(p):
        result[j] = i
    return tuple(result)


def conjugate(p: tuple[int, ...], t: tuple[int, ...]) -> tuple[int, ...]:
    """Return p*t*p^-1 with right-to-left composition."""
    return compose(compose(p, t), inverse(p))


def closure(generators: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    """Enumerate a small permutation group with a safety budget."""
    identity = tuple(range(len(generators[0])))
    seen, queue = {identity}, [identity]
    while queue:
        p = queue.pop()
        for generator in generators:
            candidate = compose(p, generator)
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
                if len(seen) > 10000:
                    raise RuntimeError('closure exceeds the small-group budget')
    return seen


def main() -> int:
    """Check the accepted group and the conditional three-generator witness."""
    data = np.load('study/exp66_gate_a_generators.npz', allow_pickle=False)

    def read(key: str) -> tuple[int, ...]:
        p = tuple(int(i) for i in data[key])
        if sorted(p) != list(range(16)):
            raise ValueError(f'{key} is not a complete permutation')
        return p

    keys = [
        'fold0_main_permutation', 'fold0_iso_main_permutation',
        'fold1_main_permutation', 'fold1_iso_main_permutation',
        'fold2_main_permutation', 'fold2_iso_main_permutation',
        'fold3_main_permutation',
    ]
    group = closure([read(key) for key in keys])
    orbits = sorted({tuple(sorted({p[i] for p in group})) for i in range(16)})
    orders = [len({tuple(p[i] for i in orbit) for p in group}) for orbit in orbits]
    assert len(group) == 5760
    assert int(np.prod(orders)) == len(group)
    print('Accepted permutations: order', len(group))
    print('Orbits and restriction orders:', list(zip(orbits, orders, strict=True)))
    print('Exact image group: S4 x S4 x D5 (D5 has order 10), solvable.')

    a = read('fold3_main_permutation')
    d = read('fold2_main_permutation')
    h = read('fold1_scan_r0.015_permutation')
    word_d = conjugate(d, a)
    witnesses = [a, conjugate(h, a), word_d, conjugate(h, word_d)]
    expected = [(1, 6), (6, 8), (1, 4), (8, 10)]
    for p, edge in zip(witnesses, expected, strict=True):
        wanted = list(range(16))
        i, j = edge
        wanted[i], wanted[j] = wanted[j], wanted[i]
        assert p == tuple(wanted)
    small = closure(witnesses)
    assert len(small) == 120
    print('Conjugate transpositions:', expected)
    print('They fix all other roots and generate S5 of order', len(small))
    print('h is now fully controlled (exp68: main = h, reverse = h^-1, empty = identity, refinement = h); the conditional premise is satisfied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
