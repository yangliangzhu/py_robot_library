"""Extract a five-label, two-loop A5 witness and audit incomplete-cycle reasoning."""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from study.exp67_group_witness import closure, compose, inverse

OUT = Path('study/exp75_minimal_group_witness.json')


def power(p: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    """Return a nonnegative permutation power."""
    result = tuple(range(len(p)))
    for _ in range(exponent):
        result = compose(p, result)
    return result


def derived(group: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
    """Compute the derived subgroup from every pair of commutators."""
    commutators = set()
    for a in group:
        for b in group:
            commutators.add(compose(compose(compose(a, b), inverse(a)), inverse(b)))
    return closure(sorted(commutators))


def cycles(p: tuple[int, ...]) -> list[list[int]]:
    """List all nontrivial cycles."""
    seen, result = set(), []
    for i in range(len(p)):
        if i in seen or p[i] == i:
            continue
        orbit, j = [], i
        while j not in seen:
            orbit.append(j)
            seen.add(j)
            j = p[j]
        result.append(orbit)
    return result


def edge_action(vertex_permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Induce the action of four vertices on their six unordered edges."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return tuple(edges.index(tuple(sorted((vertex_permutation[i], vertex_permutation[j]))))
                 for i, j in edges)


def transport_ledger(words: list[list[str]], labels: tuple,
                     permutations: dict[str, tuple]) -> dict:
    """Collect distinct generator/input-label pairs in execution order."""
    needed = {key: set() for key in permutations}
    for word in words:
        for label in labels:
            current = label
            for letter in word:
                needed[letter].add(current)
                current = permutations[letter][current]
    return {key: sorted(value) for key, value in needed.items()}



def fifteen_lift_search(permutations: dict[str, tuple], labels: tuple) -> dict:
    """Exhaust all five-sheet chart graphs using only 15 distinct base-loop lifts.

    An omitted label is a vertex. Five certified lifts for g give one reversible
    edge from omitted o to g(o). Holonomies in each component describe every
    composite loop realizable with those certificates and constant width five.
    """
    counts = Counter()
    largest = 0
    cases = []
    identity = tuple(range(6))
    generators = {key: tuple(labels.index(p[i]) for i in labels)
                  for key, p in permutations.items()}
    for omitted in itertools.product(range(6), repeat=3):
        adjacency = [[] for _ in labels]
        for key, source in zip(('a', 'd', 'h'), omitted, strict=True):
            permutation = generators[key]
            target = permutation[source]
            adjacency[source].append((target, permutation))
            adjacency[target].append((source, inverse(permutation)))
        unseen = set(range(6))
        component_orders = []
        while unseen:
            root = min(unseen)
            transports, queue = {root: identity}, [root]
            for source in queue:
                for target, permutation in adjacency[source]:
                    if target not in transports:
                        transports[target] = compose(permutation, transports[source])
                        queue.append(target)
            unseen.difference_update(transports)
            kept = tuple(i for i in range(6) if i != root)
            loops = []
            for source in transports:
                for target, permutation in adjacency[source]:
                    loop = compose(inverse(transports[target]),
                                   compose(permutation, transports[source]))
                    assert loop[root] == root
                    loops.append(tuple(kept.index(loop[i]) for i in kept))
            group = closure(loops or [tuple(range(5))])
            series = [len(group)]
            while len(group) > 1:
                next_group = derived(group)
                if next_group == group:
                    break
                group = next_group
                series.append(len(group))
            component_orders.append({'series': series, 'solvable': len(group) == 1,
                                     'root_omitted': labels[root]})
            counts[series[0]] += 1
            largest = max(largest, series[0])
        cases.append({'omitted': [labels[i] for i in omitted],
                      'component_derived_series_orders': component_orders})
    return {'case_count': len(cases), 'nonsolvable_cases': [case for case in cases
               if any(not entry['solvable'] for entry in case['component_derived_series_orders'])],
            'largest_holonomy_order': largest,
            'component_order_counts': dict(sorted(counts.items())), 'cases': cases}


def compute() -> dict:
    """Compute exact finite-group consequences of the numerical archive."""
    with np.load('study/exp66_gate_a_generators.npz', allow_pickle=False) as data:
        a = tuple(map(int, data['fold3_main_permutation']))
        d = tuple(map(int, data['fold2_main_permutation']))
        h = tuple(map(int, data['fold1_scan_r0.015_permutation']))
    for p in (a, d, h):
        assert sorted(p) == list(range(16))
    alpha, beta = power(compose(a, d), 2), power(h, 2)
    labels = (1, 4, 6, 10, 13)
    # This equality concerns the archived permutations, not unverified true monodromy.
    assert cycles(alpha) == [[1, 4, 6]]
    assert cycles(beta) == [[4, 13, 10]]
    restricted = [tuple(labels.index(p[i]) for i in labels) for p in (alpha, beta)]
    group = closure(restricted)
    assert len(group) == 60 and derived(group) == group
    assert len(closure([restricted[0]])) == len(closure([restricted[1]])) == 3
    ledger = transport_ledger([['d', 'a', 'd', 'a'], ['h', 'h']], labels,
                              {'a': a, 'd': d, 'h': h})
    assert ledger == {'a': list(labels), 'd': list(labels), 'h': [1, 4, 6, 8, 10, 13]}

    # An open five-root transport along a, reused in reverse, is cheaper than
    # treating a as a permutation of the SAME five-root set.
    chart_t, chart_u = (4, 6, 8, 10, 13), (1, 4, 8, 10, 13)
    assert {a[i] for i in chart_t} == set(chart_u)
    g = compose(inverse(a), compose(h, a))
    optimal = [tuple(chart_t.index(p[i]) for i in chart_t) for p in (d, g)]
    optimal_group = closure(optimal)
    assert len(optimal_group) == 120 and len(derived(optimal_group)) == 60
    fifteen = fifteen_lift_search({'a': a, 'd': d, 'h': h}, (1, 4, 6, 8, 10, 13))
    assert len(fifteen['nonsolvable_cases']) == 4
    assert all(13 not in case['omitted'] for case in fifteen['nonsolvable_cases'])
    six = (1, 4, 6, 8, 10, 13)
    pair_orders = []
    for pair in itertools.combinations((a, d, h), 2):
        subgroup = closure([tuple(six.index(p[i]) for i in six) for p in pair])
        pair_orders.append(len(subgroup))
        while len(subgroup) > 1:
            smaller = derived(subgroup)
            assert smaller != subgroup
            subgroup = smaller

    # Solvable completion: A4 acting on the six edges of a tetrahedron.
    # Two observed length-three orbits share exactly one edge. Their other orbits
    # have not been observed, and the generated group is nevertheless solvable.
    p, q = edge_action((1, 2, 0, 3)), edge_action((1, 3, 2, 0))
    cp, cq = cycles(p), cycles(q)
    triangle_p, triangle_q = set(cp[0]), set(cq[0])
    assert len(triangle_p & triangle_q) == 1
    fake = closure([p, q])
    fake_derived = derived(fake)
    fake_second = derived(fake_derived)
    assert (len(fake), len(fake_derived), len(fake_second)) == (12, 4, 1)
    return {
        'labels': list(labels),
        'archived_composite_cycles': {'(a*d)^2': cycles(alpha), 'h^2': cycles(beta)},
        'restricted_generators': [list(p) for p in restricted],
        'group_order': len(group), 'derived_order': len(derived(group)),
        'composite_execution_words': [['d', 'a', 'd', 'a'], ['h', 'h']],
        'distinct_base_loop_lifts': ledger,
        'distinct_base_loop_lift_count': sum(map(len, ledger.values())),
        'optimal_transport_witness': {
            'chart_T': list(chart_t), 'chart_U': list(chart_u),
            'a_input_labels': list(chart_t), 'd_input_labels': list(chart_t),
            'h_input_labels': list(chart_u),
            'a_maps_T_onto_U': [a[i] for i in chart_t],
            'd_on_T': list(optimal[0]), 'a_inverse_h_a_on_T': list(optimal[1]),
            'group_order': len(optimal_group), 'derived_order': len(derived(optimal_group)),
            'base_pair_group_orders_ad_ah_dh': pair_orders,
        },
        'fifteen_lift_search': fifteen,
        'solvable_incomplete_cycle_counterexample': {
            'permutations': [list(p), list(q)], 'cycles': [cp, cq],
            'observed_cycles': [cp[0], cq[0]],
            'derived_series_orders': [len(fake), len(fake_derived), len(fake_second)],
        },
        'status': 'Exact group inference from archived permutations; path certification incomplete.',
    }


def main() -> None:
    """Write a reproducible exact group calculation."""
    result = compute()
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print('A5 order / derived order:', result['group_order'], result['derived_order'])
    print('Lift ledger:', result['distinct_base_loop_lifts'])
    print('15-lift cases:', result['fifteen_lift_search']['case_count'])
    print('Nonsolvable 15-lift cases:', len(result['fifteen_lift_search']['nonsolvable_cases']))
    print('Largest 15-lift holonomy group:', result['fifteen_lift_search']['largest_holonomy_order'])


if __name__ == '__main__':
    main()
