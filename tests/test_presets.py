"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import entscheidungsmatrix_generator as gen
import pdm_constants as C
import pdm_evaluation as ev
import pdm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert C.rho_min(p["m"]) <= p["rho"] <= C.RHO_MAX, name
        assert p["levels"] in C.LEVEL_CHOICES and p["form"] in gen.FORMS, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


@pytest.mark.parametrize("m", range(C.M_MIN, C.M_MAX + 1))
def test_rho_lower_bound_is_inside_the_open_interval_of_the_generator(m):
    assert gen.rho_lower(m) < C.rho_min(m) < C.RHO_MAX
    gen.copula_uniforms(4, m, C.rho_min(m), gen.SplitMix64(1))          # darf nicht werfen
    gen.copula_uniforms(4, m, C.RHO_MAX, gen.SplitMix64(1))


def test_slider_limits_fit_the_generator():
    assert C.M_MAX <= gen.M_MAX
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200
