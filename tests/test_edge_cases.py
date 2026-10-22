"""Randfälle vor dem ersten Mutationstest: sehr kleine Netze, kmin größer als n, gerade Summen, leere Kantenlisten, alle Sampler an jeder Grenze."""

import numpy as np
import pytest

import kfm_algorithm as K
import kfm_evaluation as ev
import kfm_visualization as V
from kfm_scenario import Instance, make_instance


@pytest.mark.parametrize("sampler", ["stub", "swap", "chunglu"])
@pytest.mark.parametrize("n, gamma, kmin, cutoff", [(2, 2.5, 1, "free"), (3, 2.0, 5, "free"), (5, 4.5, 1, "structural"), (50, 2.0, 1, "free"), (4, 2.5, 3, "structural")])
def test_every_sampler_survives_tiny_and_degenerate_instances(sampler, n, gamma, kmin, cutoff):
    a = ev.analyse(ev.Settings(n=n, gamma=gamma, kmin=kmin, cutoff=cutoff, sampler=sampler, seed=1))
    assert 0.0 <= a.giant <= 1.0 and len(a.realised) == n and a.net.inst.deg.sum() % 2 == 0
    assert ev.verdict_network(a)[0] in {"info", "success", "warning"}


def test_empty_edge_list_has_all_components_as_singletons_and_no_defects():
    e = np.zeros((0, 2), dtype=np.int64)
    assert K.component_sizes(3, e) == [1, 1, 1] and K.defect_counts(e) == (0, 0) and K.simplify(e).shape == (0, 2)
    assert K.degrees_of(3, e).tolist() == [0, 0, 0] and K.transitivity(3, e) == 0.0 and K.assortativity(3, e) == 0.0


def test_stub_matching_with_an_odd_number_of_stubs_drops_the_last_one():
    e = K.stub_matching(np.array([1, 1, 1]), np.random.default_rng(1))
    assert e.shape == (1, 2)


def test_zero_degrees_are_allowed_everywhere():
    deg = np.array([0, 2, 1, 1])
    assert K.is_graphical(deg) and K.havel_hakimi(deg.tolist()) == [(1, 2), (1, 3)]
    assert K.kappa(np.array([0, 0, 0])) == 0.0 and K.hub_ratio(np.array([0, 1]), np.array([0, 1]), 2) == 0.0


def test_chung_lu_with_all_zero_weights_has_no_edges():
    assert K.chung_lu_edges(np.zeros(5), np.random.default_rng(1)).shape == (0, 2)


def test_graphical_check_agrees_with_havel_hakimi_at_the_boundary():
    for deg in ([4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 2, 2, 2, 2]):
        assert K.is_graphical(deg) == (K.havel_hakimi(deg) is not None), deg


def test_figures_survive_a_network_without_edges_and_a_single_degree():
    deg = np.array([1, 1])
    assert len(V.build_degrees(deg, np.array([0, 0])).data) == 2
    fig = V.build_hubs(np.array([2, 1]), np.array([0, 0]), top=5)
    assert len(fig.data[1].x) == 2 and min(fig.data[1].y) > 0  # Ist-Grad 0 wird für die Log-Achse auf 0,8 angehoben


def test_instance_with_only_one_possible_degree():
    inst = make_instance(10, 2.5, 3, "structural", 1)
    assert inst.kmax >= 3 and inst.deg.sum() % 2 == 0 and isinstance(inst, Instance)
