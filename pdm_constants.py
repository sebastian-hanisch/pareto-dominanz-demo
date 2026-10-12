"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

N_MIN, N_MAX, DEFAULT_N = 5, 60, 8          # Dominanzmatrix n x n x m: bei n = 60, m = 8 rund 29 000 Vergleiche, O(n^2 m), unter 5 ms
M_MIN, M_MAX, DEFAULT_M = 2, 8, 3
RHO_MAX, RHO_STEP, DEFAULT_RHO = 0.95, 0.05, 0.0
SEED_MAX, DEFAULT_SEED = 999999, 35
LEVEL_CHOICES = (0, 10, 5, 3)               # 0 = stetige Rohwerte, sonst Bewertungsstufen 1 bis L
LEVEL_LABELS = {0: "Stetig (Punkte, Euro, Minuten)", 10: "10 Stufen", 5: "5 Stufen", 3: "3 Stufen"}
FORM_DEFAULT = "linear"

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                  # Experimente 3 bis 5 und Annahmen-Tabelle: feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_N_SWEEP = (10, 20, 50)                  # Experiment 1
EXP_M_SWEEP = (2, 3, 4, 5, 6, 8)
EXP_RHO_M = (3, 5)                          # Experiment 2
EXP_RHO_SWEEP = (-0.4, -0.2, 0.0, 0.3, 0.6, 0.9)
EXP_CHANGE_N, EXP_CHANGE_M = 15, 3
EXP_CHANGE_K = (1, 2, 3, 5)                 # hinzugefügte Alternativen
EXP_NOISE_M = (3, 5)
EXP_NOISE_SIGMA = (0.02, 0.05, 0.10)        # Standardabweichung auf der Güte-Skala [0, 1]

COLOR_FRONT = "#2F6B65"
COLOR_OTHER = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"
COLOR_WARN = "#c0392b"
COLOR_GOLD = "#D68A2E"


def rho_min(m: int) -> float:
    """Kleinster zulässiger Wert des Korrelationsreglers: knapp über -1/(m-1), auf Schrittweite gerundet."""
    return round(math.ceil((-1.0 / (m - 1) + 0.001) / RHO_STEP) * RHO_STEP, 2)


PRESETS = {
    "Standardfall (Voreinstellung)": {"n": DEFAULT_N, "m": DEFAULT_M, "rho": 0.0, "levels": 0, "form": FORM_DEFAULT, "seed": DEFAULT_SEED},
    "Zwei gegenläufige Kriterien": {"n": 12, "m": 2, "rho": -0.6, "levels": 0, "form": FORM_DEFAULT, "seed": DEFAULT_SEED},
    "Fünf unabhängige Kriterien": {"n": 20, "m": 5, "rho": 0.0, "levels": 0, "form": FORM_DEFAULT, "seed": DEFAULT_SEED},
    "Gleichläufige Kriterien": {"n": 20, "m": 4, "rho": 0.6, "levels": 0, "form": FORM_DEFAULT, "seed": DEFAULT_SEED},
    "Grobe Skala": {"n": 20, "m": 3, "rho": 0.0, "levels": 3, "form": FORM_DEFAULT, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "8 Alternativen, 3 unabhängige Kriterien: wenig Dominanz, aber noch gut zu überblicken.",
    "Zwei gegenläufige Kriterien": "12 Alternativen, 2 Kriterien, wer in einem gut ist, ist im anderen eher schlecht: die Front ist hier groß, der Kompromiss ist das Thema.",
    "Fünf unabhängige Kriterien": "20 Alternativen, 5 Kriterien: nur wenige Paare sind überhaupt vergleichbar, die Mehrheit bleibt unbesiegt.",
    "Gleichläufige Kriterien": "20 Alternativen, 4 Kriterien, wer in einem gut ist, ist meist auch in den anderen gut: Dominanz ordnet hier viel.",
    "Grobe Skala": "20 Alternativen, 3 Kriterien auf 3 Bewertungsstufen: viele Gleichstände, die Dominanz wirkt schärfer, als sie ist.",
}
