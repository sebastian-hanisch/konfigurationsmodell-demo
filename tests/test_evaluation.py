"""Auswertung: Sampler-Aufruf, Kennzahlen von Hand, Struktur der Messreihen."""

import numpy as np
import pytest

import kfm_algorithm as K
import kfm_constants as C
import kfm_evaluation as ev
from kfm_scenario import Instance, make_instance


def test_num_uses_a_decimal_comma():
    assert ev.num(3.14159, 2) == "3,14" and ev.num(10, 0) == "10" and ev.num(2.5, 2) == "2,50"


def test_build_network_stub_matching_removes_loops_and_double_edges():
    inst = make_instance(300, 2.2, 1, "free", 3)
    net = ev.build_network(inst, "stub", 3)
    assert net.sampler_used == "stub" and K.defect_counts(net.edges) == (0, 0)
    assert (net.loops, net.multi) == K.defect_counts(net.raw_edges)
    assert K.degrees_of(inst.n, net.raw_edges).tolist() == inst.deg.tolist()  # vor dem Vereinfachen stimmen alle Grade


def test_build_network_swap_hits_every_degree_when_graphical():
    inst = make_instance(400, 2.5, 2, "structural", 4)
    assert K.is_graphical(inst.deg)
    net = ev.build_network(inst, "swap", 4)
    assert net.sampler_used == "swap" and net.graphical and net.accepted > 0
    assert K.degrees_of(inst.n, net.edges).tolist() == inst.deg.tolist() and K.defect_counts(net.edges) == (0, 0)


def test_build_network_swap_falls_back_when_the_sequence_is_not_graphical():
    inst = Instance(4, np.array([3, 3, 1, 1]), 3)
    net = ev.build_network(inst, "swap", 1)
    assert not net.graphical and net.sampler_used == "stub (Rückfall)" and net.accepted == 0


def test_build_network_chung_lu_has_no_defects_and_depends_on_the_seed():
    inst = make_instance(300, 2.5, 2, "structural", 5)
    a, b, c = ev.build_network(inst, "chunglu", 5), ev.build_network(inst, "chunglu", 5), ev.build_network(inst, "chunglu", 6)
    assert a.sampler_used == "chunglu" and (a.loops, a.multi) == (0, 0) and (a.edges == b.edges).all()
    assert not (len(a.edges) == len(c.edges) and (a.edges == c.edges).all())


def test_analyse_network_by_hand():
    inst = Instance(3, np.array([2, 1, 1]), 2)
    net = ev.Network(inst, np.array([[0, 1], [0, 2]]), np.array([[0, 1], [0, 2]]), 0, 0, "swap", True, 0)
    a = ev.analyse_network(net, hubs=1)
    assert a.kappa == pytest.approx(6 / 4) and not a.prediction and a.giant == 1.0 and a.realised.tolist() == [2, 1, 1]
    assert a.hub_ratio == 1.0 and a.hubhub == 0 and a.lost_share == 0.0
    lost = ev.Network(inst, np.array([[0, 1]]), np.array([[0, 1]]), 0, 0, "stub", True, 0)
    b = ev.analyse_network(lost, hubs=2)
    assert b.lost_share == pytest.approx(0.5) and b.giant == pytest.approx(2 / 3) and b.hub_ratio == pytest.approx((1 / 2 + 1 / 1) / 2)
    assert b.hubhub == 1  # die zwei größten Hubs sind Knoten 0 und 1 (stabile Reihenfolge bei Gleichstand), die Kante liegt zwischen ihnen


def test_analyse_is_deterministic():
    s = ev.Settings(n=500, gamma=2.5, kmin=2, cutoff="structural", sampler="stub", seed=9)
    a, b = ev.analyse(s), ev.analyse(s)
    assert (a.realised == b.realised).all() and a.giant == b.giant


def test_sweep_molloy_reed_structure():
    rows = ev.sweep_molloy_reed(n=2000, shares=(0.9, 0.6), seeds=(1, 2, 3))
    assert [r["share"] for r in rows] == [0.9, 0.6]
    for r in rows:
        assert r["low"] <= r["giant"] <= r["high"] and r["predicted"] == (r["kappa"] > 2.0)
    assert rows[1]["kappa"] > rows[0]["kappa"] and rows[1]["giant"] > rows[0]["giant"]


def test_sweep_defects_structure():
    rows = ev.sweep_defects(gammas=(2.1, 3.5), kmin=1, n=3000, seeds=(1, 2))
    assert [r["gamma"] for r in rows] == [2.1, 3.5]
    assert rows[0]["free"] > rows[0]["structural"] >= 0 and rows[0]["free_kmax"] > rows[0]["structural_kmax"]
    assert rows[0]["free"] > rows[1]["free"]


def test_series_chung_lu_structure():
    rows = ev.series_chung_lu(n=600, gamma=2.2, kmin=1, hubs=5, seeds=(1, 2, 3))
    assert [r["cutoff"] for r in rows] == ["free", "structural"] and rows[0]["kmax"] > rows[1]["kmax"]
    assert rows[1]["capped"] <= 2 and rows[0]["capped"] > rows[1]["capped"]  # die Obergrenze nutzt den erwarteten mittleren Grad: bei kleiner Summe der Ziel-Grade bleibt ein gekapptes Paar möglich
    assert rows[0]["chunglu_ratio"] < rows[1]["chunglu_ratio"] + 1e-9
    for r in rows:
        assert 0 <= r["graphical"] <= r["plans"] == 3


def test_series_mixing_structure_and_direction():
    res = ev.series_mixing(n=300, k=6, p=0.01, steps=(0.25, 1, 4), reference=20, seeds=(1, 2, 3))
    assert [r["steps"] for r in res["rows"]] == [0.25, 1, 4] and res["plans"] == 3 and res["reference"] == 20
    assert res["start"] > res["rows"][0]["transitivity"] > res["rows"][-1]["transitivity"] >= res["plateau"] * 0.5
    assert res["rows"][0]["distance"] > res["rows"][-1]["distance"]


def test_the_fixed_plans_are_the_documented_ones():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and C.MR_N == 20000 and C.DEFECT_N == 20000 and (C.CHUNGLU_N, C.CHUNGLU_GAMMA, C.CHUNGLU_KMIN, C.CHUNGLU_HUBS) == (3000, 2.2, 1, 10)
    assert (C.MIX_N, C.MIX_K, C.MIX_P, C.MIX_REFERENCE) == (1000, 10, 0.01, 40) and C.MR_SHARES[0] == 0.9 and C.MR_SHARES[-1] == 0.4
