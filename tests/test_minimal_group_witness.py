"""Check exact compression and guard against incomplete-cycle inference."""
from __future__ import annotations

import pytest

from study.exp75_minimal_group_witness import compute


@pytest.fixture(scope='module')
def witness():
    return compute()


def test_five_label_transport_generates_nonsolvable_image(witness):
    result = witness['optimal_transport_witness']
    assert result['group_order'] == 120
    assert result['derived_order'] == 60
    assert set(result['a_maps_T_onto_U']) == set(result['chart_U'])
    assert sum(len(result[key]) for key in ('a_input_labels', 'd_input_labels', 'h_input_labels')) == 15


def test_all_minimum_transport_schemes_keep_expensive_roots(witness):
    search = witness['fifteen_lift_search']
    assert search['case_count'] == len(search['cases']) == 216
    cases = search['nonsolvable_cases']
    assert {tuple(case['omitted']) for case in cases} == {(1, 1, 6), (4, 4, 6), (4, 6, 6), (6, 1, 6)}
    assert all(not set(case['omitted']) & {8, 10, 13} for case in cases)


def test_two_observed_overlapping_three_cycles_can_have_solvable_completion(witness):
    example = witness['solvable_incomplete_cycle_counterexample']
    first, second = example['observed_cycles']
    assert len(first) == len(second) == 3
    assert len(set(first) & set(second)) == 1
    assert example['derived_series_orders'] == [12, 4, 1]
