"""Orakel-Tests: networkx für Graphikalität, Havel–Hakimi, Transitivität, Assortativität; exakte Aufzählung aller einfachen Graphen mit gegebenen Graden
gegen die Gleichverteilung der Sampler; Chung–Lu gegen die exakte Erwartung; Komponenten gegen scipy."""

import itertools
import math
from collections import Counter

import numpy as np
import pytest

import kfm_algorithm as K

nx = pytest.importorskip("networkx")


def all_simple_graphs_with_degrees(deg):
    """Alle einfachen Graphen (als Mengen von Kanten) auf den Knoten 0..n-1 mit genau diesen Graden, durch Aufzählung aller Kantenmengen."""
    n = len(deg)
    pairs = list(itertools.combinations(range(n), 2))
    out = []
    for mask in range(1 << len(pairs)):
        d = [0] * n
        chosen = []
        for i, (u, v) in enumerate(pairs):
            if mask >> i & 1:
                d[u] += 1
                d[v] += 1
                chosen.append((u, v))
        if d == list(deg):
            out.append(frozenset(chosen))
    return out


def test_is_graphical_agrees_with_networkx_on_random_sequences():
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        deg = rng.integers(0, n + 1, size=n).tolist()
        assert K.is_graphical(deg) == nx.is_graphical(deg, method="eg"), deg


def test_havel_hakimi_matches_graphicality_and_the_degrees():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        deg = rng.integers(0, min(n, 6) + 1, size=n)
        edges = K.havel_hakimi(deg.tolist())
        if K.is_graphical(deg.tolist()):
            assert edges is not None and K.degrees_of(n, np.array(edges).reshape(-1, 2)).tolist() == deg.tolist()
            assert len(set(edges)) == len(edges) and all(u < v for u, v in edges)
        else:
            assert edges is None, deg.tolist()


def test_transitivity_and_assortativity_agree_with_networkx():
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(8, 60))
        g = nx.gnp_random_graph(n, float(rng.uniform(0.05, 0.4)), seed=int(rng.integers(1 << 30)))
        edges = list(g.edges())
        assert K.transitivity(n, edges) == pytest.approx(nx.transitivity(g), abs=1e-12)
        if len(edges) > 2 and len({d for _, d in g.degree()}) > 1:
            assert K.assortativity(n, np.array(edges)) == pytest.approx(nx.degree_assortativity_coefficient(g), abs=1e-9)


def test_components_agree_with_scipy():
    scipy_graph = pytest.importorskip("scipy.sparse.csgraph")
    from scipy.sparse import coo_matrix

    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.integers(5, 200))
        m = int(rng.integers(0, 2 * n))
        e = rng.integers(0, n, size=(m, 2))
        a = coo_matrix((np.ones(m), (e[:, 0], e[:, 1])), shape=(n, n))
        k, lab = scipy_graph.connected_components(a, directed=False)
        assert K.component_sizes(n, e) == sorted(np.bincount(lab).tolist(), reverse=True)


def test_enumeration_counts_the_simple_graphs_with_two_regular_degrees_on_six_nodes():
    graphs = all_simple_graphs_with_degrees([2] * 6)
    assert len(graphs) == 70  # 60 Sechsecke und 10 Paare von Dreiecken


def test_stub_matching_conditioned_on_simple_is_uniform_and_the_acceptance_probability_is_exact():
    """Jeder einfache Graph entsteht aus genau 2!^6 = 64 von 11!! = 10 395 Paarungen: die Annahmewahrscheinlichkeit ist 70 · 64 / 10 395, und die angenommenen Graphen sind gleichverteilt."""
    deg = np.array([2] * 6)
    states = all_simple_graphs_with_degrees(deg.tolist())
    p_exact = 70 * 64 / 10395
    rng = np.random.default_rng(5)
    reps = 40000
    counts: Counter = Counter()
    accepted = 0
    for _ in range(reps):
        e = K.stub_matching(deg, rng)
        if K.defect_counts(e) == (0, 0):
            accepted += 1
            counts[frozenset(map(tuple, np.sort(e, axis=1).tolist()))] += 1
    se = math.sqrt(p_exact * (1 - p_exact) / reps)
    assert abs(accepted / reps - p_exact) < 4 * se
    assert set(counts) == set(states)
    expected = accepted / 70
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    assert chi2 < 110  # 69 Freiheitsgrade, 99,9-%-Grenze etwa 109


def chi2_of_chain(deg, samples, steps_per_sample, seed, lazy=True):
    states = all_simple_graphs_with_degrees(deg)
    rng = np.random.default_rng(seed)
    edges = np.array(K.havel_hakimi(deg))
    counts: Counter = Counter()
    for _ in range(samples):
        edges = K.edge_switch_steps(edges, steps_per_sample, rng)[0] if lazy else K.edge_switches(edges, steps_per_sample, rng)[0]
        counts[frozenset(map(tuple, edges.tolist()))] += 1
    expected = samples / len(states)
    chi2 = sum((counts.get(s, 0) - expected) ** 2 / expected for s in states)
    return chi2, len(states), set(counts) == set(states)


def test_lazy_edge_switch_chain_is_uniform_over_the_simple_graphs_of_a_degree_sequence():
    chi2, k, reached = chi2_of_chain([2] * 6, 14000, 60, 6)
    assert k == 70 and reached
    assert chi2 < 110  # 69 Freiheitsgrade, 99,9-%-Grenze etwa 109


def test_lazy_edge_switch_chain_is_uniform_for_an_unbalanced_sequence():
    chi2, k, reached = chi2_of_chain([3, 2, 2, 2, 1, 1, 1], 5000, 100, 7)
    df = k - 1
    assert k > 20 and reached
    assert chi2 < df + 5 * math.sqrt(2 * df) + 10  # etwa fünf Standardabweichungen über dem Mittel


def test_counting_only_accepted_switches_gives_a_measurably_non_uniform_chain():
    """Wer die gelungenen Tausche zählt statt der Versuche (so zählt `networkx.double_edge_swap`), bevorzugt Zustände mit vielen möglichen Tauschen."""
    lazy, k, _ = chi2_of_chain([3, 2, 2, 2, 1, 1, 1], 12000, 100, 7)
    accepted, k2, _ = chi2_of_chain([3, 2, 2, 2, 1, 1, 1], 12000, 25, 7, lazy=False)
    assert k == k2 and accepted > 1.5 * lazy and accepted > (k - 1) + 5 * math.sqrt(2 * (k - 1)) + 10 and lazy < (k - 1) + 5 * math.sqrt(2 * (k - 1)) + 10


def test_chung_lu_expected_degree_matches_the_formula():
    w = np.array([6.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    total = w.sum()
    exact = np.array([sum(K.chung_lu_probability(w[i], w[j], total) for j in range(len(w)) if j != i) for i in range(len(w))])
    rng = np.random.default_rng(8)
    reps = 6000
    acc = np.zeros(len(w))
    for _ in range(reps):
        acc += K.degrees_of(len(w), K.chung_lu_edges(w, rng))
    mean = acc / reps
    assert np.all(np.abs(mean - exact) < 5 * np.sqrt(np.maximum(exact, 0.1) / reps))
    assert exact[0] < w[0]  # der größte Hub bekommt weniger als sein erwarteter Grad: die Wahrscheinlichkeiten werden gekappt


def test_chung_lu_without_capping_gives_exactly_the_target_degrees_in_expectation():
    w = np.full(50, 3.0)  # w_i w_j / Σw = 9 / 150 < 1: nie gekappt, E[Grad] = 3 · 49 / 50
    total = w.sum()
    expected = sum(K.chung_lu_probability(3.0, 3.0, total) for _ in range(49))
    assert expected == pytest.approx(3.0 * 49 / 50)
    assert K.capped_pairs(w) == 0


def test_molloy_reed_threshold_matches_a_large_simulation():
    rng = np.random.default_rng(9)
    n = 20000
    sub, sup = [], []
    for seed in range(3):
        for a, bucket in ((0.8, sub), (0.65, sup)):
            deg = K.two_point_degrees(n, a, np.random.default_rng(100 + seed))
            bucket.append(K.giant_fraction(n, K.simplify(K.stub_matching(deg, rng))))
    assert max(sub) < 0.05 < 0.3 < min(sup)
    assert K.kappa(K.two_point_degrees(n, 0.8, np.random.default_rng(100))) < 2 < K.kappa(K.two_point_degrees(n, 0.65, np.random.default_rng(100)))
