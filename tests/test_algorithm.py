"""Kern von Hand an Mini-Instanzen: Gradfolgen, Graphikalität, Havel–Hakimi, Stub-Matching, Kantentausch, Chung–Lu, Kennzahlen."""

import math

import numpy as np
import pytest

import kfm_algorithm as K


def test_kappa_by_hand(mini_degrees):
    assert K.kappa(mini_degrees) == pytest.approx(2.5)
    assert K.kappa(np.array([2, 2, 2])) == pytest.approx(2.0)  # 12 / 6: genau auf der Schwelle, noch keine Riesenkomponente
    assert K.kappa(np.array([0, 0])) == 0.0
    assert K.giant_by_molloy_reed(mini_degrees) and not K.giant_by_molloy_reed(np.array([2, 2, 2])) and not K.giant_by_molloy_reed(np.array([1, 1]))


def test_mean_degree_and_parity_fix():
    assert K.mean_degree(np.array([1, 2, 3])) == pytest.approx(2.0) and K.mean_degree(np.array([])) == 0.0
    assert K.fix_parity(np.array([1, 2, 3])).tolist() == [1, 2, 3]  # gerade Summe 6: unverändert
    assert K.fix_parity(np.array([1, 2, 2])).tolist() == [2, 2, 2]  # ungerade Summe 5: der erste Knoten bekommt eins dazu
    assert K.fix_parity(np.array([], dtype=int)).tolist() == []


def test_two_point_degrees_have_even_sum_and_only_the_two_degrees_apart_from_the_parity_fix():
    d = K.two_point_degrees(1001, 0.7, np.random.default_rng(1))
    assert d.sum() % 2 == 0 and set(d[1:].tolist()) <= {1, 3} and d[0] in (1, 2, 3, 4)
    assert K.two_point_degrees(50, 1.0, np.random.default_rng(1))[1:].tolist() == [1] * 49
    assert set(K.two_point_degrees(50, 0.0, np.random.default_rng(1))[1:].tolist()) == {3}


def test_power_law_degrees_respect_the_bounds_and_the_parity():
    d = K.power_law_degrees(5000, 2.5, 2, 40, np.random.default_rng(2))
    assert d.sum() % 2 == 0 and d[1:].min() >= 2 and d[1:].max() <= 40
    same = K.power_law_degrees(300, 2.5, 3, 3, np.random.default_rng(2))
    assert set(same[1:].tolist()) == {3}  # kmin = kmax: nur ein Grad
    low = K.power_law_degrees(20000, 3.5, 1, 100, np.random.default_rng(3))
    high = K.power_law_degrees(20000, 2.1, 1, 100, np.random.default_rng(3))
    assert high.mean() > low.mean()  # flacheres Potenzgesetz: mehr große Grade


def test_structural_cutoff_by_hand():
    assert K.structural_cutoff(10000, 3.0) == 173  # int(√30000)
    assert K.structural_cutoff(1, 0.0) == 2


@pytest.mark.parametrize("deg, expected", [
    ([3, 3, 3, 3], True), ([2, 2, 2], True), ([3, 1, 1, 1], True), ([4, 1, 1, 1, 1], True), ([0, 0, 0], True), ([], True),
    ([3, 3, 1, 1], False), ([4, 4, 4, 1, 1], False), ([2, 1], False), ([3, 3], False), ([1, 1, 3, 3], False),
])
def test_is_graphical_by_hand(deg, expected):
    assert K.is_graphical(deg) is expected


def test_is_graphical_rejects_negative_and_odd_sums():
    assert not K.is_graphical([-1, 1]) and not K.is_graphical([1, 1, 1])


def test_havel_hakimi_builds_a_simple_graph_with_exactly_these_degrees():
    deg = [3, 2, 2, 1]
    edges = K.havel_hakimi(deg)
    assert sorted(K.degrees_of(4, np.array(edges)).tolist()) == sorted(deg) and K.degrees_of(4, np.array(edges)).tolist() == deg
    assert len(set(edges)) == len(edges) and all(u < v for u, v in edges)
    assert K.havel_hakimi([3, 3, 1, 1]) is None and K.havel_hakimi([1, 1, 1]) is None and K.havel_hakimi([-1, 1]) is None
    assert K.havel_hakimi([0, 0]) == [] and K.havel_hakimi([]) == []


def test_stub_matching_uses_every_stub_once_and_keeps_the_degrees():
    deg = np.array([2, 2, 3, 1])
    e = K.stub_matching(deg, np.random.default_rng(4))
    assert e.shape == (4, 2) and K.degrees_of(4, e).tolist() == deg.tolist()


def test_stub_matching_with_two_stubs_each_can_only_make_loops_or_a_double_edge():
    seen = set()
    for seed in range(40):
        e = K.stub_matching(np.array([2, 2]), np.random.default_rng(seed))
        seen.add(tuple(sorted(map(tuple, np.sort(e, axis=1).tolist()))))
    assert seen <= {((0, 0), (1, 1)), ((0, 1), (0, 1))} and len(seen) == 2


def test_defect_counts_by_hand():
    assert K.defect_counts(np.array([[0, 0], [1, 1]])) == (2, 0)
    assert K.defect_counts(np.array([[0, 1], [1, 0]])) == (0, 1)
    assert K.defect_counts(np.array([[0, 1], [0, 1], [1, 0], [2, 3]])) == (0, 2)  # dreimal dieselbe Kante: zwei überzählige
    assert K.defect_counts(np.zeros((0, 2), dtype=int)) == (0, 0)
    assert K.defect_counts(np.array([[0, 1], [1, 2]])) == (0, 0)


def test_simplify_by_hand():
    assert K.simplify(np.array([[0, 1], [1, 0], [2, 2], [1, 2]])).tolist() == [[0, 1], [1, 2]]
    assert K.simplify(np.zeros((0, 2), dtype=int)).shape == (0, 2)
    assert K.simplify(np.array([[3, 3]])).shape == (0, 2)


def test_degrees_of_counts_a_loop_twice():
    assert K.degrees_of(3, np.array([[0, 0], [1, 2]])).tolist() == [2, 1, 1]
    assert K.degrees_of(2, np.zeros((0, 2), dtype=int)).tolist() == [0, 0]


def test_edge_switches_keep_degrees_and_stay_simple():
    start = K.havel_hakimi([3, 3, 3, 2, 2, 2, 1, 1, 1])
    edges, done, tries = K.edge_switches(np.array(start), 30, np.random.default_rng(5))
    assert done == 30 and tries >= done
    assert K.degrees_of(9, edges).tolist() == [3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert K.defect_counts(edges) == (0, 0)
    assert sorted(map(tuple, edges.tolist())) != sorted(start)  # der Graph hat sich geändert


def test_edge_switches_by_hand_on_two_disjoint_edges():
    edges, done, _ = K.edge_switches(np.array([[0, 1], [2, 3]]), 1, np.random.default_rng(1))
    assert done == 1
    assert sorted(map(tuple, edges.tolist())) in ([(0, 2), (1, 3)], [(0, 3), (1, 2)])


def test_edge_switches_find_no_switch_in_a_path_of_two_edges():
    edges, done, tries = K.edge_switches(np.array([[0, 1], [1, 2]]), 3, np.random.default_rng(1), max_tries_factor=50)
    assert done == 0 and tries == 150 and sorted(map(tuple, edges.tolist())) == [(0, 1), (1, 2)]
    e2, d2, t2 = K.edge_switches(np.array([[0, 1]]), 5, np.random.default_rng(1))
    assert d2 == 0 and t2 == 0 and e2.tolist() == [[0, 1]]
    e3, d3, _ = K.edge_switches(np.array([[0, 1], [2, 3]]), 0, np.random.default_rng(1))
    assert d3 == 0 and sorted(map(tuple, e3.tolist())) == [(0, 1), (2, 3)]


def test_chung_lu_probability_by_hand():
    assert K.chung_lu_probability(3, 2, 10) == pytest.approx(0.6)
    assert K.chung_lu_probability(4, 5, 10) == 1.0  # 2,0 wird auf 1 gekappt
    assert K.chung_lu_probability(1, 1, 0) == 0.0


def test_capped_pairs_by_hand():
    w = [4.0, 4.0, 1.0]  # Σw = 9: nur das Paar der beiden Hubs überschreitet 1 (16/9)
    assert K.capped_pairs(w) == 1
    assert K.capped_pairs(w, subset=[0, 2]) == 0 and K.capped_pairs([1.0, 1.0, 1.0]) == 0


def test_chung_lu_edges_are_simple_and_have_about_the_expected_count():
    w = [1.0] * 40
    counts = [len(K.chung_lu_edges(w, np.random.default_rng(s))) for s in range(30)]
    expected = 40 * 39 / 2 * (1 / 40)
    assert abs(np.mean(counts) - expected) < 4 * math.sqrt(expected / 30)
    e = K.chung_lu_edges([5.0, 5.0, 5.0], np.random.default_rng(1))  # Σw = 15: jedes Paar mit Wahrscheinlichkeit 25/15 → gekappt auf 1
    assert sorted(map(tuple, e.tolist())) == [(0, 1), (0, 2), (1, 2)]
    assert K.chung_lu_edges([2.0], np.random.default_rng(1)).shape == (0, 2)


def test_hub_ratio_by_hand():
    assert K.hub_ratio(np.array([5, 1, 3]), np.array([4, 1, 3]), 2) == pytest.approx(0.9)
    assert K.hub_ratio(np.array([5, 1, 3]), np.array([5, 1, 3]), 3) == 1.0
    assert K.hub_ratio(np.array([0, 0]), np.array([0, 0]), 1) == 0.0


def test_edges_between_by_hand():
    e = np.array([[0, 1], [1, 2], [2, 2], [0, 1], [3, 4]])
    assert K.edges_between(e, [0, 1]) == 2 and K.edges_between(e, [2]) == 0 and K.edges_between(e, [0, 1, 2]) == 3


def test_transitivity_and_assortativity_by_hand(triangle_with_tail):
    n, edges = triangle_with_tail
    assert K.transitivity(n, edges) == pytest.approx(0.6)
    assert K.transitivity(3, [(0, 1)]) == 0.0
    star = [(0, 1), (0, 2), (0, 3)]
    assert K.assortativity(4, star) == pytest.approx(-1.0)
    assert K.assortativity(3, [(0, 1), (1, 2), (0, 2)]) == 0.0  # alle Grade gleich
    assert K.assortativity(3, []) == 0.0


def test_components_by_hand():
    assert K.component_sizes(7, [(0, 1), (1, 2), (3, 4), (4, 5), (3, 5)]) == [3, 3, 1]
    assert K.giant_fraction(4, [(0, 1), (1, 2)]) == 0.75 and K.giant_fraction(0, []) == 0.0


def test_ring_lattice_without_rewiring_has_the_textbook_values():
    e = K.ring_lattice_rewired(12, 4, 0.0, np.random.default_rng(1))
    assert len(e) == 24 and set(K.degrees_of(12, e).tolist()) == {4}
    assert K.transitivity(12, e) == pytest.approx(0.5)  # 3 (k − 2) / (4 (k − 1)) bei k = 4


def test_ring_lattice_rewiring_keeps_the_graph_simple_and_lowers_the_transitivity():
    e = K.ring_lattice_rewired(200, 6, 0.5, np.random.default_rng(2))
    assert K.defect_counts(e) == (0, 0) and len(e) == 600
    assert K.transitivity(200, e) < 0.5


def test_edge_switch_steps_count_every_proposal_and_keep_the_degrees():
    start = np.array(K.havel_hakimi([3, 3, 3, 2, 2, 2, 1, 1, 1]))
    edges, accepted = K.edge_switch_steps(start, 200, np.random.default_rng(5))
    assert 0 < accepted <= 200 and K.degrees_of(9, edges).tolist() == [3, 3, 3, 2, 2, 2, 1, 1, 1] and K.defect_counts(edges) == (0, 0)
    e0, a0 = K.edge_switch_steps(start, 0, np.random.default_rng(5))
    assert a0 == 0 and sorted(map(tuple, e0.tolist())) == sorted(map(tuple, start.tolist()))
    assert K.edge_switch_steps(np.array([[0, 1]]), 10, np.random.default_rng(1))[1] == 0


def test_try_switch_by_hand():
    E, S = [(0, 1), (2, 3)], {(0, 1), (2, 3)}
    assert K.try_switch(E, S, 0, 1, False) and E == [(0, 3), (1, 2)] and S == {(0, 3), (1, 2)}
    E, S = [(0, 1), (2, 3)], {(0, 1), (2, 3)}
    assert K.try_switch(E, S, 0, 1, True) and sorted(E) == [(0, 2), (1, 3)]
    E, S = [(0, 1), (1, 2)], {(0, 1), (1, 2)}
    assert not K.try_switch(E, S, 0, 1, False) and E == [(0, 1), (1, 2)]  # (0,2) und (1,1): Schleife
    E, S = [(0, 1), (2, 3), (0, 3)], {(0, 1), (2, 3), (0, 3)}
    assert not K.try_switch(E, S, 0, 1, False) and S == {(0, 1), (2, 3), (0, 3)}  # (0,3) gäbe es schon
    assert not K.try_switch(E, S, 1, 1, False)


def test_capped_pairs_does_not_count_a_product_exactly_equal_to_the_total():
    assert K.capped_pairs([2.0, 2.0]) == 0  # 2·2 / 4 = 1: nicht über 1, die Wahrscheinlichkeit ist genau 1
    assert K.capped_pairs([2.0, 2.0, 0.0]) == 0


def test_chung_lu_works_for_small_positive_totals():
    assert K.chung_lu_probability(0.5, 0.4, 1.0) == pytest.approx(0.2)
    total_edges = sum(len(K.chung_lu_edges([0.6, 0.4], np.random.default_rng(s))) for s in range(300))
    assert 40 < total_edges < 110  # Wahrscheinlichkeit 0,24 je Ziehung: etwa 72 von 300


def test_edge_switch_steps_with_one_step_or_two_edges_can_still_switch():
    two = np.array([[0, 1], [2, 3]])
    assert sum(K.edge_switch_steps(two, 1, np.random.default_rng(s))[1] for s in range(40)) > 5  # ein Schritt, zwei Kanten: Vorschlag (i != j) wird angenommen
    assert sum(K.edge_switch_steps(two, 30, np.random.default_rng(s))[1] for s in range(10)) > 5


def test_giant_fraction_of_a_single_node_is_one():
    assert K.giant_fraction(1, []) == 1.0 and K.giant_fraction(2, []) == 0.5


def test_negative_degrees_are_never_graphical_even_when_the_sum_is_even():
    assert K.havel_hakimi([-1, 1, 1, -1]) is None and not K.is_graphical([-1, 1, 1, -1])


def test_simplify_keeps_a_single_edge():
    assert K.simplify(np.array([[1, 0]])).tolist() == [[0, 1]] and K.simplify(np.array([[2, 2], [0, 1]])).tolist() == [[0, 1]]
