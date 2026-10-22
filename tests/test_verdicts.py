"""Verdikte (reine Funktionen): Art und jede Zahl im Text an handgesetzten Werten, einschließlich der Schwellen und Zweige."""

import numpy as np

import kfm_evaluation as ev
from kfm_scenario import Instance


def make(sampler_used="stub", kappa=13.0, prediction=True, giant=0.594, loops=6, multi=48, lost=0.0086, hub_ratio=0.953, n=3000):
    inst = Instance(n, np.array([1, 1]), 10)
    net = ev.Network(inst, np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int), loops, multi, sampler_used, sampler_used != "stub (Rückfall)", 0)
    return ev.Analysis(net=net, kappa=kappa, prediction=prediction, giant=giant, realised=np.array([1, 1]), hub_ratio=hub_ratio, hubhub=0, lost_share=lost)


def test_network_verdict_for_stub_matching_names_defects_and_losses():
    kind, text = ev.verdict_network(make())
    assert kind == "info" and "κ = ⟨k²⟩/⟨k⟩ = 13,0" in text and "eine Riesenkomponente" in text and "59,4 % der 3000 Knoten" in text
    assert "6 Schleifen und 48 überzählige Mehrfachkanten" in text and "0,9 % der Halbkanten" in text and "95 % ihres Ziel-Grades" in text


def test_network_verdict_for_the_edge_switch_hits_every_degree():
    kind, text = ev.verdict_network(make("swap", giant=0.77))
    assert kind == "success" and "genau" in text and "1,0" in text and "77,0 %" in text


def test_network_verdict_warns_when_prediction_and_measurement_disagree():
    kind, text = ev.verdict_network(make("swap", kappa=1.3, prediction=False, giant=0.6))
    assert kind == "warning" and "keine Riesenkomponente" in text and "60,0 %" in text and "1,3" in text


def test_network_verdict_without_a_giant_component_and_without_prediction_is_consistent():
    kind, text = ev.verdict_network(make("swap", kappa=1.3, prediction=False, giant=0.006))
    assert kind == "success" and "keine Riesenkomponente" in text and "0,6 %" in text


def test_network_verdict_for_chung_lu_and_the_fallback():
    kind, text = ev.verdict_network(make("chunglu", hub_ratio=0.884))
    assert kind == "info" and "Erwartungswert" in text and "88 % ihres Ziel-Grades" in text
    kind, text = ev.verdict_network(make("stub (Rückfall)"))
    assert kind == "warning" and "nicht graphisch" in text and "Stub-Matching" in text


def test_molloy_verdict_uses_the_last_row_below_and_the_first_above_the_threshold():
    rows = [
        {"share": 0.85, "kappa": 1.69, "giant": 0.003, "low": 0.002, "high": 0.004, "predicted": False},
        {"share": 0.75, "kappa": 2.0, "giant": 0.043, "low": 0.01, "high": 0.1, "predicted": False},
        {"share": 0.7, "kappa": 2.125, "giant": 0.309, "low": 0.28, "high": 0.33, "predicted": True},
        {"share": 0.5, "kappa": 2.5, "giant": 0.812, "low": 0.8, "high": 0.82, "predicted": True},
    ]
    kind, text = ev.verdict_molloy(rows)
    assert kind == "info" and "κ = 2,00" in text and "75 %" in text and "4,3 %" in text and "κ = 2,12" in text and "70 %" in text and "30,9 %" in text and "verschmiert" in text


def test_defects_verdict_names_the_flattest_and_the_steepest_exponent():
    rows = [
        {"gamma": 2.1, "free": 0.1164, "structural": 0.0100, "free_kmax": 19999.0, "structural_kmax": 296.0},
        {"gamma": 3.5, "free": 0.0000841, "structural": 0.0000842, "free_kmax": 19999.0, "structural_kmax": 154.0},
    ]
    kind, text = ev.verdict_defects(rows)
    assert kind == "info" and "γ = 2,1" in text and "11,6 %" in text and "1,0 %" in text and "296 statt 19999" in text and "γ = 3,5" in text and "0,0 % und 0,0 %" in text


def test_chung_lu_verdict_with_and_without_the_edge_switch_reference():
    free = {"cutoff": "free", "kmax": 2999, "capped": 36, "chunglu_ratio": 0.8445, "chunglu_hubhub": 44, "swap_ratio": 1.0, "swap_hubhub": 36, "graphical": 5, "plans": 5}
    structural = {"cutoff": "structural", "kmax": 96, "capped": 0, "chunglu_ratio": 1.0014, "chunglu_hubhub": 21, "swap_ratio": 1.0, "swap_hubhub": 18, "graphical": 5, "plans": 5}
    kind, text = ev.verdict_chung_lu([free, structural])
    assert kind == "info" and "84 % ihres erwarteten Grades (Kantentausch: 100 %)" in text and "36 von 45 Hub-Paaren" in text
    assert "44 Kanten, beim Kantentausch 36" in text and "größter Grad 96" in text and "100 % bei 0 gekappten Paaren" in text
    none = dict(free, swap_ratio=None, swap_hubhub=None)
    kind, text = ev.verdict_chung_lu([none, structural])
    assert "Kantentausch:" not in text and "beim Kantentausch" not in text


def test_mixing_verdict_names_the_first_step_inside_two_deviations():
    res = {"rows": [{"steps": 0.25, "distance": 276.0, "transitivity": 0.15}, {"steps": 1, "distance": 3.0, "transitivity": 0.0098}, {"steps": 2, "distance": 0.44, "transitivity": 0.0084},
                    {"steps": 4, "distance": 0.6, "transitivity": 0.0085}], "start": 0.648, "plateau": 0.0082, "plateau_sd": 0.0005, "plans": 4, "reference": 40, "m": 5000}
    kind, text = ev.verdict_mixing(res)
    assert kind == "info" and "Transitivität 0,65" in text and "0,008" in text and "ab 2 Vorschlägen je Kante" in text and "4 Pläne" in text
    res["rows"][2]["distance"], res["rows"][3]["distance"] = 5.0, 5.0
    assert "innerhalb der gemessenen Schritte nicht" in ev.verdict_mixing(res)[1]
    res["rows"][0]["distance"] = 1.0
    assert "ab 0,25 Vorschlägen je Kante" in ev.verdict_mixing(res)[1]
