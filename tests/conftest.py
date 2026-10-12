"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NAMES = "ABCDEF"


@pytest.fixture
def mini():
    """6 Alternativen A bis F, 3 Kriterien (alle: höher besser). Handrechnung:
    C wird von A, B, E dominiert; F von A, B, C, D, E; B und E sind identisch und dominieren sich nicht.
    Front = {A, B, D, E}; Schichten (1, 1, 2, 1, 1, 3); vergleichbare Paare 8 von 15."""
    return np.array([
        [9, 5, 3],   # A
        [7, 6, 4],   # B
        [6, 4, 2],   # C
        [3, 8, 7],   # D
        [7, 6, 4],   # E (identisch mit B)
        [2, 3, 1],   # F
    ], dtype=float)


@pytest.fixture
def mini_weights():
    """Verborgene Gewichte des Entscheiders der Mini-Instanz; Nutzen 6,6 / 6,1 / 4,6 / 5,3 / 6,1 / 2,1, der Beste ist A."""
    return np.array([0.5, 0.3, 0.2])
