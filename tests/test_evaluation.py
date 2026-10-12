"""Auswertung: Kennzahlen der Instanz und Form der Experimente (kleine Seed-Mengen, damit die Suite schnell bleibt)."""

import numpy as np
import pytest

import pdm_constants as C
import pdm_dominance as dom
import pdm_evaluation as ev

FEW = C.SWEEP_SEEDS[:12]


def test_analysis_is_consistent_with_the_core():
    a = ev.analyse(ev.Settings(n=15, m=4, rho=0.2, levels=0, form="linear", seed=7))
    assert a.front_size == int(a.front.sum()) == int((a.layer == 1).sum())
    assert a.front_share == pytest.approx(a.front_size / 15)
    assert a.n_layers == int(a.layer.max())
    assert a.comparable == pytest.approx(dom.comparable_share(a.D))
    assert a.comparable_expected is None and a.front_expected is None        # ρ ≠ 0: die Formel gilt nicht
    assert sorted(a.utility_rank.tolist()) == list(range(1, 16)) and a.utility_rank[a.best] == 1
    assert a.best_in_front == bool(a.front[a.best])
    assert a.top3_in_front == int(a.front[a.top3].sum())


def test_expected_values_are_only_offered_for_independent_continuous_criteria():
    a = ev.analyse(ev.Settings(n=20, m=3, rho=0.0, levels=0))
    assert a.comparable_expected == pytest.approx(0.25) and a.front_expected == pytest.approx(dom.expected_front_size(20, 3))
    assert ev.analyse(ev.Settings(n=20, m=3, rho=0.0, levels=3)).comparable_expected is None


def test_same_settings_give_the_same_analysis():
    a, b = ev.analyse(ev.Settings(seed=5)), ev.analyse(ev.Settings(seed=5))
    assert (a.layer == b.layer).all() and a.inst.raw.tolist() == b.inst.raw.tolist()


def test_front_sweep_has_one_row_per_cell_and_a_reference():
    rows = ev.sweep_front_vs_m(ns=(10, 20), ms=(2, 5), seeds=FEW)
    assert [(r["n"], r["m"]) for r in rows] == [(10, 2), (10, 5), (20, 2), (20, 5)]
    for r in rows:
        assert 0 < r["share"] <= 1 and 0 < r["expected"] <= 1
    assert rows[1]["share"] > rows[0]["share"]          # mehr Kriterien: größere Front


def test_rho_sweep_skips_values_below_the_equal_correlation_bound():
    rows = ev.sweep_front_vs_rho(ms=(5,), rhos=(-0.4, -0.2, 0.0), seeds=FEW)
    assert [r["rho"] for r in rows] == [-0.2, 0.0]       # -0,4 < -1/4


def test_order_experiment_fields_and_expected_comparable_share():
    rows = ev.order_experiment(ms=(2, 5), seeds=FEW)
    assert [r["m"] for r in rows] == [2, 5]
    assert rows[0]["comparable_expected"] == 0.5 and rows[1]["comparable_expected"] == pytest.approx(1 / 16)
    for r in rows:
        assert -1 <= r["tau_layer"] <= 1 and -1 <= r["tau_equal"] <= 1 and 1 <= r["layers_to_top3"]


def test_change_experiment_has_the_removal_row_and_no_flip_for_one_change():
    rows = ev.change_experiment(ks=(1, 2), seeds=C.SWEEP_SEEDS[:60])
    assert [r["k"] for r in rows] == [1, 2, -1]
    assert rows[0]["flip_share"] == 0.0 and rows[2]["flip_share"] == 0.0
    assert rows[2]["front_lost_share"] == 0.0           # Entfernen kann kein Mitglied aus der Front werfen


def test_scale_experiment_levels_and_identical_pairs():
    rows = ev.scale_experiment(ms=(3,), levels=(0, 3), seeds=FEW)
    assert rows[0]["identical"] == 0.0 and rows[1]["identical"] > 0.0
    assert rows[1]["front_share"] < rows[0]["front_share"]


def test_noise_experiment_zero_noise_would_change_nothing_and_noise_changes_something():
    rows = ev.noise_experiment(ms=(3,), sigmas=(0.1,), seeds=FEW)
    assert 0 < rows[0]["flip"] < 1


def test_monotony_experiment_rows_and_values():
    rows = ev.monotony_experiment(ms=(3,), seeds=C.SWEEP_SEEDS[:80])
    by_form = {r["form"]: r["best_in_front"] for r in rows}
    assert by_form["linear"] == by_form["konkav"] == by_form["cobb"] == 1.0
    assert by_form["zielwert"] < 1.0


def test_verdicts_return_a_kind_and_a_text():
    rows1 = ev.sweep_front_vs_m(ns=(20,), ms=(2, 5), seeds=FEW)
    assert ev.verdict_front(rows1)[0] in ("warning", "info")
    rows3 = ev.order_experiment(ms=(5,), seeds=FEW)
    assert "τ" in ev.verdict_order(rows3)[1]
    rows4 = ev.scale_experiment(ms=(3,), levels=(0, 3), seeds=FEW)
    assert ev.verdict_scale(rows4)[0] == "warning"
    rows5 = ev.monotony_experiment(ms=(3,), forms=("linear", "zielwert"), seeds=FEW)
    assert "Zielwert" in ev.verdict_monotony(rows5)[1]


def test_worst_case_at_the_slider_maximum_terminates():
    a = ev.analyse(ev.Settings(n=C.N_MAX, m=C.M_MAX, rho=0.0, levels=0, seed=1))
    assert a.D.shape == (C.N_MAX, C.N_MAX) and a.front_size >= 1
    assert np.isfinite(a.tau_layer) and np.isfinite(a.tau_equal)
