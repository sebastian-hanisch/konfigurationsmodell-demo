"""Jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4): über dieselben Auswertungsfunktionen wie in der App, mit den festen Plänen.

Die Zufallszahlen eines numpy-Generators sind über Versionen nicht stabil: die Messwerte gelten deshalb als Bänder um die in der README genannten Zahlen;
exakt sind die Zählungen am Orakel (siehe tests/test_oracle_konfigurationsmodell.py) und die Theoriewerte."""

import numpy as np
import pytest

import kfm_algorithm as K
import kfm_constants as C
import kfm_evaluation as ev


@pytest.fixture(scope="module")
def molloy():
    return {r["share"]: r for r in ev.sweep_molloy_reed()}


@pytest.fixture(scope="module")
def defects():
    return {r["gamma"]: r for r in ev.sweep_defects()}


@pytest.fixture(scope="module")
def chung():
    return {r["cutoff"]: r for r in ev.series_chung_lu()}


@pytest.fixture(scope="module")
def mixing():
    return ev.series_mixing()


def test_the_series_use_the_documented_fixed_plans():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005)) and C.MR_N == 20000 and C.DEFECT_N == 20000 and C.MR_SHARES == (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4)
    assert (C.CHUNGLU_N, C.CHUNGLU_GAMMA, C.CHUNGLU_KMIN, C.CHUNGLU_HUBS) == (3000, 2.2, 1, 10) and (C.MIX_N, C.MIX_K, C.MIX_P, C.MIX_REFERENCE) == (1000, 10, 0.01, 40)
    assert C.DEFECT_GAMMAS == (2.1, 2.3, 2.5, 2.75, 3.0, 3.5) and C.SWITCH_STEPS_PER_EDGE == 20 and len(C.MIX_SEEDS) == 4


def test_molloy_reed_threshold(molloy):
    assert molloy[0.75]["kappa"] == pytest.approx(2.0, abs=0.03) and molloy[0.75]["giant"] == pytest.approx(0.043, abs=0.04)
    assert molloy[0.7]["kappa"] == pytest.approx(2.12, abs=0.03) and molloy[0.7]["giant"] == pytest.approx(0.309, abs=0.1)
    assert molloy[0.65]["kappa"] == pytest.approx(2.23, abs=0.03) and molloy[0.65]["giant"] == pytest.approx(0.521, abs=0.08)
    assert molloy[0.5]["kappa"] == pytest.approx(2.50, abs=0.03) and molloy[0.5]["giant"] == pytest.approx(0.812, abs=0.06)
    assert molloy[0.9]["kappa"] == pytest.approx(1.50, abs=0.03) and molloy[0.9]["giant"] < 0.01
    assert molloy[0.75]["giant"] < 0.15 < molloy[0.7]["giant"] and K.kappa(np.array([1, 3])) == pytest.approx(2.5)  # zwei Grade 1 und 3 zu gleichen Teilen: κ = 10 / 4


def test_stub_matching_defects(defects):
    assert defects[2.1]["free"] == pytest.approx(0.116, abs=0.04) and defects[2.1]["structural"] == pytest.approx(0.010, abs=0.006)
    assert defects[2.1]["structural_kmax"] == pytest.approx(296, abs=60) and defects[2.1]["free_kmax"] > 19000
    assert defects[2.3]["free"] == pytest.approx(0.025, abs=0.015) and defects[2.3]["structural"] == pytest.approx(0.004, abs=0.003)
    assert defects[3.5]["free"] < 0.0001 and defects[3.5]["structural"] < 0.0001
    assert defects[2.1]["free"] > defects[2.3]["free"] > defects[2.5]["free"] > defects[3.0]["free"]


def test_chung_lu_caps_the_hubs(chung):
    free, structural = chung["free"], chung["structural"]
    assert free["chunglu_ratio"] == pytest.approx(0.84, abs=0.08) and free["capped"] == pytest.approx(36, abs=6) and free["capped"] <= 45
    assert free["chunglu_hubhub"] == pytest.approx(44, abs=8) and free["swap_hubhub"] == pytest.approx(36, abs=8) and free["chunglu_hubhub"] > free["swap_hubhub"]
    assert free["swap_ratio"] == 1.0 and free["graphical"] == free["plans"] == 5 and structural["graphical"] == 5
    assert structural["kmax"] == pytest.approx(96, abs=20) and structural["capped"] <= 1 and structural["chunglu_ratio"] == pytest.approx(1.0, abs=0.04) and structural["swap_ratio"] == 1.0


def test_the_edge_switch_forgets_a_clustered_start_within_a_few_proposals_per_edge(mixing):
    assert mixing["start"] == pytest.approx(0.65, abs=0.03) and mixing["m"] == 5000 and mixing["plans"] == 4
    assert mixing["plateau"] == pytest.approx(0.008, abs=0.003) and mixing["plateau"] > 0
    rows = {r["steps"]: r for r in mixing["rows"]}
    assert rows[0.5]["transitivity"] == pytest.approx(0.040, abs=0.02) and rows[1]["transitivity"] == pytest.approx(0.0098, abs=0.004) and rows[2]["transitivity"] == pytest.approx(0.0084, abs=0.003)
    assert rows[0.5]["distance"] > 10 and rows[2]["distance"] <= 3.0
    first = min(r["steps"] for r in mixing["rows"] if r["distance"] <= 2.0)
    assert first in (2, 4)  # README: ab 2 Vorschlägen je Kante


def test_presets_hold_what_their_help_texts_say():
    by = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, free, swap, chunglu, steep = (by[name] for name in C.PRESETS)
    assert std.kappa == pytest.approx(13.0, abs=4.0) and std.prediction and std.giant > 0.95 and 0.002 < std.lost_share < 0.02 and std.net.sampler_used == "stub"
    assert free.kappa == pytest.approx(56.0, abs=25.0) and 0.03 < free.lost_share < 0.12 and free.hub_ratio == pytest.approx(0.80, abs=0.12) and free.net.loops + free.net.multi > 100
    assert swap.net.sampler_used == "swap" and swap.net.graphical and swap.hub_ratio == 1.0 and swap.lost_share == 0.0
    assert K.degrees_of(swap.net.inst.n, swap.net.edges).tolist() == swap.net.inst.deg.tolist()
    assert chunglu.net.sampler_used == "chunglu" and chunglu.hub_ratio == pytest.approx(0.88, abs=0.1) and chunglu.hub_ratio < 0.97
    assert steep.kappa == pytest.approx(1.3, abs=0.3) and steep.kappa < 2 and not steep.prediction and steep.giant < 0.05


def test_the_series_run_in_a_few_seconds():
    import time

    t0 = time.perf_counter()
    ev.sweep_molloy_reed(), ev.sweep_defects(), ev.series_chung_lu(), ev.series_mixing()
    assert time.perf_counter() - t0 < 180  # weite Obergrenze (gemessen einige Sekunden)
