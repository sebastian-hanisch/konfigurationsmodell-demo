"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_degrees():
    """Vier Knoten mit den Graden 1, 1, 3, 3 (Summe 8, vier Kanten möglich?): Handrechnung ⟨k²⟩/⟨k⟩ = (1 + 1 + 9 + 9) / 8 = 2,5; der einfache Graph ist nicht möglich (zwei Knoten mit Grad 3 brauchen je drei Nachbarn)."""
    return np.array([1, 1, 3, 3])


@pytest.fixture
def triangle_with_tail():
    """Dreieck 0-1-2 mit einem Endknoten 3 an Knoten 0: Grade 3, 2, 2, 1; drei Dreier an Knoten 0, je einer an 1 und 2, also fünf; ein Dreieck: Transitivität 3 / 5 = 0,6."""
    return 4, [(0, 1), (1, 2), (0, 2), (0, 3)]
