"""Abbildungen: Zählung der Grade, Hub-Punkte, Spuren je Figur, Log-Achsen; alle Achsen gesperrt."""

import numpy as np

import kfm_evaluation as ev
import kfm_visualization as V


def test_degree_counts_by_hand():
    assert V.degree_counts([1, 1, 2, 0, 5, 2, 1]) == [(1, 3), (2, 2), (5, 1)]  # Grad 0 zählt nicht
    assert V.degree_counts([]) == []


def test_hub_points_order_and_clipping():
    tx, ry = V.hub_points(np.array([1, 5, 3, 5, 2]), np.array([1, 4, 3, 5, 2]), top=3)
    assert tx.tolist() == [5, 5, 3] and ry.tolist() == [4, 5, 3]  # bei Gleichstand die kleinere Knotennummer zuerst
    tx2, _ = V.hub_points(np.array([1, 2]), np.array([1, 2]), top=10)
    assert len(tx2) == 2


def test_degree_figure_has_target_and_network_traces_on_log_axes():
    fig = V.build_degrees(np.array([1, 1, 2, 3]), np.array([1, 2, 2, 3]))
    assert [t.name for t in fig.data] == ["Ziel", "gebautes Netz"] and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert list(fig.data[0].x) == [1, 2, 3] and list(fig.data[0].y) == [2, 1, 1] and list(fig.data[1].y) == [1, 2, 1]


def test_hub_figure_has_the_diagonal_and_the_hubs():
    fig = V.build_hubs(np.array([10, 8, 1, 1]), np.array([9, 8, 1, 1]), top=2)
    assert fig.data[0].name == "Ziel = Ist" and list(fig.data[0].x) == [1, 10] and list(fig.data[1].x) == [10, 8] and list(fig.data[1].y) == [9, 8]
    assert fig.data[1].name == "die 2 größten Hubs"


def test_series_figures_use_the_rows():
    mr = ev.sweep_molloy_reed(n=1000, shares=(0.9, 0.6), seeds=(1, 2))
    f = V.build_molloy(mr)
    assert [t.name for t in f.data] == ["Minimum bis Maximum", "gemessen (Median)"] and len(f.layout.shapes) == 1
    assert list(f.data[1].x) == sorted(r["kappa"] for r in mr)
    d = V.build_defects(ev.sweep_defects(gammas=(2.1, 3.5), n=2000, seeds=(1, 2)))
    assert [t.name for t in d.data] == ["Grade bis n − 1", "strukturelle Obergrenze"] and d.layout.yaxis.type == "log" and min(d.data[0].y) > 0
    c = V.build_chung_lu(ev.series_chung_lu(n=500, gamma=2.2, kmin=1, hubs=5, seeds=(1, 2)))
    assert [t.name for t in c.data] == ["Chung–Lu", "Kantentausch"] and list(c.data[0].x) == ["frei", "strukturell"]
    m = V.build_mixing(ev.series_mixing(n=200, k=6, p=0.01, steps=(0.5, 2), reference=10, seeds=(1, 2)))
    assert [t.name for t in m.data] == ["Plateau ± 2 Standardabweichungen", "Transitivität (Mittel der Pläne)"] and m.layout.xaxis.type == "log"


def test_all_axes_are_locked():
    mr = ev.sweep_molloy_reed(n=500, shares=(0.9, 0.6), seeds=(1,))
    figs = [
        V.build_degrees(np.array([1, 2]), np.array([1, 2])), V.build_hubs(np.array([3, 1]), np.array([3, 1])), V.build_molloy(mr),
        V.build_defects(ev.sweep_defects(gammas=(2.5,), n=500, seeds=(1,))),
        V.build_chung_lu(ev.series_chung_lu(n=300, gamma=2.2, kmin=1, hubs=3, seeds=(1,))),
        V.build_mixing(ev.series_mixing(n=100, k=4, p=0.01, steps=(1,), reference=4, seeds=(1,))),
    ]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange
