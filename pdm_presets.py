"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet; es gibt deshalb kein KEPT-Muster. Einzige berechnete Grenze:
die Untergrenze der Korrelation hängt von der Kriterienzahl m ab (`C.rho_min`); `clip_rho_state` rückt den Wert vor dem Zeichnen zurecht.
"""

import random
from dataclasses import dataclass

import streamlit as st

import pdm_constants as C
from entscheidungsmatrix_generator import FORMS


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "rho_slider": SettingSpec("rho", float, C.DEFAULT_RHO, -1.0, C.RHO_MAX),
    "levels_select": SettingSpec("lv", int, 0, choices=C.LEVEL_CHOICES),
    "form_select": SettingSpec("form", str, C.FORM_DEFAULT, choices=FORMS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "m": "m_slider", "rho": "rho_slider", "levels": "levels_select", "form": "form_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clip_rho_state():
    """Vor dem Zeichnen des Korrelationsreglers: den Wert in [rho_min(m), RHO_MAX] halten (die Untergrenze hängt von m ab)."""
    m = int(st.session_state.get("m_slider", C.DEFAULT_M))
    value = float(st.session_state.get("rho_slider", C.DEFAULT_RHO))
    st.session_state["rho_slider"] = min(C.RHO_MAX, max(C.rho_min(m), round(value / C.RHO_STEP) * C.RHO_STEP))


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
