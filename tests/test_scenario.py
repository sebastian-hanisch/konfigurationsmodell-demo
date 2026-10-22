"""Szenario: deterministisch aus dem Zufalls-Seed, Obergrenzen der Grade, Grade in den Schranken, gerade Summe."""

import numpy as np
import pytest

import kfm_algorithm as K
import kfm_scenario as S


def test_expected_mean_degree_by_hand():
    # γ = 2, Grade 1 und 2: Gewichte 1 und 1/4, Mittelwert (1·1 + 2·0,25) / 1,25 = 1,2
    assert S.expected_mean_degree(2.0, 1, 2) == pytest.approx(1.2)
    assert S.expected_mean_degree(2.5, 3, 3) == pytest.approx(3.0)  # nur ein Grad
    assert S.expected_mean_degree(2.5, 4, 2) == pytest.approx(4.0)  # kmax < kmin: nur kmin


def test_cutoff_value_by_hand():
    assert S.cutoff_value(1000, 2.5, 2, "free") == 999
    assert S.cutoff_value(3, 2.5, 5, "free") == 5  # kmin dominiert
    c = S.cutoff_value(10000, 2.5, 2, "structural")
    assert 2 <= c < 9999 and c == K.structural_cutoff(10000, S.expected_mean_degree(2.5, 2, 9999))


def test_structural_cutoff_is_never_above_the_free_one_and_never_below_kmin():
    for n in (50, 500, 3000):
        for gamma in (2.0, 2.5, 4.0):
            for kmin in (1, 2, 5):
                s = S.cutoff_value(n, gamma, kmin, "structural")
                f = S.cutoff_value(n, gamma, kmin, "free")
                assert kmin <= s <= f


def test_make_instance_is_deterministic_and_in_bounds():
    a, b, c = S.make_instance(500, 2.5, 2, "structural", 7), S.make_instance(500, 2.5, 2, "structural", 7), S.make_instance(500, 2.5, 2, "structural", 8)
    assert (a.deg == b.deg).all() and not (a.deg == c.deg).all()
    assert a.deg.sum() % 2 == 0 and a.deg[1:].min() >= 2 and a.deg[1:].max() <= a.kmax and a.n == 500


def test_a_flatter_exponent_gives_larger_degrees_and_free_cutoff_allows_the_largest():
    flat = S.make_instance(3000, 2.1, 1, "free", 1)
    steep = S.make_instance(3000, 3.5, 1, "free", 1)
    assert flat.deg.max() > steep.deg.max() and flat.deg.mean() > steep.deg.mean()
    capped = S.make_instance(3000, 2.1, 1, "structural", 1)
    assert capped.deg.max() <= capped.kmax < flat.kmax


def test_two_point_instance():
    d = S.make_two_point(400, 0.75, 3)
    assert d.sum() % 2 == 0 and set(d[1:].tolist()) <= {1, 3}
    assert (S.make_two_point(400, 0.75, 3) == d).all() and isinstance(d, np.ndarray)
