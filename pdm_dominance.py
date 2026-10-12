"""Kern: Pareto-Dominanz, Front, Schichten und die Kennzahlen darum. Nur numpy, jede Regel eine eigene Funktion.

Alle Funktionen arbeiten auf einer Matrix `G` (n x m), in der HÖHER BESSER gilt (Kriterien mit Richtung ↓ vorher mit -1 multipliziert,
siehe `Matrix.signed`). Dominanz ist eine paarweise Beziehung: sie hängt nur von den beiden Alternativen ab, nie von den übrigen.
"""

from __future__ import annotations

import numpy as np


def dominance_matrix(G: np.ndarray, strict: bool = False) -> np.ndarray:
    """D[i, j] = True, wenn Alternative i die Alternative j dominiert.

    Schwach (Standard): i ist in jedem Kriterium mindestens so gut und in mindestens einem besser. Identische Alternativen dominieren
    sich nicht. `strict=True`: i ist in JEDEM Kriterium besser (die strenge Variante; Zweig-Test).
    """
    G = np.asarray(G, dtype=float)
    ge = G[:, None, :] >= G[None, :, :]
    gt = G[:, None, :] > G[None, :, :]
    if strict:
        return gt.all(axis=2)
    return ge.all(axis=2) & gt.any(axis=2)


def front_mask(D: np.ndarray) -> np.ndarray:
    """Alternativen, die von keiner anderen dominiert werden (Schicht 1, nichtdominierte Menge)."""
    return ~D.any(axis=0)


def layers(D: np.ndarray) -> np.ndarray:
    """Nichtdominierte Sortierung: Schicht 1 = Front; Schicht k = Front der Menge ohne die Schichten 1 bis k-1."""
    n = D.shape[0]
    layer = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    k = 0
    while remaining.any():
        k += 1
        sub = D[np.ix_(remaining, remaining)]
        current = np.flatnonzero(remaining)[~sub.any(axis=0)]
        layer[current] = k
        remaining[current] = False
    return layer


def comparable_share(D: np.ndarray) -> float:
    """Anteil der Paare i < j, für die eine der beiden Alternativen die andere dominiert (Vergleichbarkeit unter Dominanz)."""
    n = D.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, 1)
    return float((D | D.T)[iu].mean())


def dominated_by(D: np.ndarray, j: int) -> np.ndarray:
    """Indizes der Alternativen, die j dominieren."""
    return np.flatnonzero(D[:, j])


def identical_pairs_share(G: np.ndarray) -> float:
    """Anteil der Paare i < j mit in jedem Kriterium gleichen Werten (diese dominieren sich nicht und bleiben beide in der Front)."""
    n = len(G)
    if n < 2:
        return 0.0
    eq = (G[:, None, :] == G[None, :, :]).all(axis=2)
    return float(eq[np.triu_indices(n, 1)].mean())


def expected_front_size(n: int, m: int) -> float:
    """Erwartete Zahl nichtdominierter Alternativen bei n unabhängigen Alternativen mit unabhängigen stetigen Kriterien.

    A(n, 1) = 1 und A(n, m) = Summe_{k=1..n} A(k, m-1) / k. Die Rekursion ist hier gegen Aufzählung (n = 4) und Simulation geprüft,
    nicht aus der Primärquelle übernommen.
    """
    a = np.ones(n)
    for _ in range(m - 1):
        a = np.cumsum(a / np.arange(1, n + 1))
    return float(a[-1])


def comparable_share_independent(m: int) -> float:
    """Anteil vergleichbarer Paare bei unabhängigen stetigen Kriterien: 2 * 2^-m (a >= b in allen m Kriterien hat Wahrscheinlichkeit 2^-m)."""
    return 2.0 ** (1 - m)


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendalls tau-b (Gleichstände in a oder b berücksichtigt); 0, wenn eine Seite keine Unterschiede hat."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = np.sign(a[:, None] - a[None, :])
    sb = np.sign(b[:, None] - b[None, :])
    denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
    return 0.0 if denom == 0 else float((sa * sb).sum() / denom)


def equal_weight_score(goodness: np.ndarray) -> np.ndarray:
    """Gleichgewichtete Summe der Güten: der einfachste Skalarwert ohne Wissen über die Gewichte."""
    return np.asarray(goodness, dtype=float).mean(axis=1)


def best_index(utility: np.ndarray) -> int:
    """Index der Alternative mit dem höchsten Nutzen; bei Gleichstand die kleinste Nummer."""
    return int(np.argmax(utility))


def top_k(utility: np.ndarray, k: int) -> np.ndarray:
    """Indizes der k besten Alternativen nach Nutzen, absteigend (Gleichstand: kleinere Nummer zuerst)."""
    order = np.lexsort((np.arange(len(utility)), -np.asarray(utility)))
    return order[:k]
