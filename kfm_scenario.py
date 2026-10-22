"""Szenario: Gradfolgen mit Potenzgesetz, zwei Graden und das Watts–Strogatz-Startnetz der Mischzeit-Reihe; deterministisch aus dem Zufalls-Seed."""

from dataclasses import dataclass

import numpy as np

import kfm_algorithm as K
import kfm_constants as C


@dataclass(eq=False)
class Instance:
    n: int
    deg: np.ndarray          # Ziel-Grade (Summe gerade)
    kmax: int                # größter erlaubter Grad bei der Ziehung


def expected_mean_degree(gamma: float, kmin: int, kmax: int) -> float:
    """Erwarteter mittlerer Grad des abgeschnittenen Potenzgesetzes auf kmin..kmax."""
    ks = np.arange(kmin, max(kmin, kmax) + 1, dtype=float)
    p = ks ** (-gamma)
    return float((ks * p).sum() / p.sum())


def cutoff_value(n: int, gamma: float, kmin: int, cutoff: str) -> int:
    """Größter erlaubter Grad: „free“ gleich n − 1, „structural“ gleich √(n · ⟨k⟩) mit dem mittleren Grad des Potenzgesetzes bis n − 1 (mindestens kmin)."""
    free = max(kmin, n - 1)
    if cutoff == "free":
        return free
    return max(kmin, min(free, K.structural_cutoff(n, expected_mean_degree(gamma, kmin, free))))


def make_instance(n: int, gamma: float, kmin: int, cutoff: str, seed: int) -> Instance:
    """Potenzgesetz-Grade für n Knoten, deterministisch aus (n, gamma, kmin, cutoff, seed)."""
    kmax = cutoff_value(n, gamma, kmin, cutoff)
    rng = np.random.default_rng(seed)
    return Instance(n, K.power_law_degrees(n, gamma, kmin, kmax, rng), kmax)


def make_two_point(n: int, share_low: float, seed: int) -> np.ndarray:
    """Grade 1 und 3 (Anteil `share_low` mit Grad 1), deterministisch aus dem Seed."""
    return K.two_point_degrees(n, share_low, np.random.default_rng(seed))
