"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SWEEP_SEEDS (200 feste Matrizen), nie über einen einzelnen Seed. Prozentwerte mit Toleranz 0,5 Punkte, τ mit 0,01,
damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen.
"""

import pytest

import pdm_constants as C
import pdm_dominance as dom
import pdm_evaluation as ev


@pytest.fixture(scope="module")
def front_vs_m():
    return ev.sweep_front_vs_m()


@pytest.fixture(scope="module")
def front_vs_rho():
    return ev.sweep_front_vs_rho()


@pytest.fixture(scope="module")
def order():
    return ev.order_experiment()


@pytest.fixture(scope="module")
def change():
    return ev.change_experiment()


@pytest.fixture(scope="module")
def scale():
    return ev.scale_experiment()


@pytest.fixture(scope="module")
def noise():
    return ev.noise_experiment()


@pytest.fixture(scope="module")
def monotony():
    return ev.monotony_experiment()


def pct(x):
    return 100 * x


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def test_seed_set_is_the_documented_one():
    assert len(C.SWEEP_SEEDS) == 200 and C.EXP_N == 20


@pytest.mark.parametrize("n,m,expected", [
    (20, 2, 18.4), (20, 3, 36.6), (20, 4, 54.6), (20, 5, 70.8), (20, 6, 82.2), (20, 8, 94.5),
    (50, 2, 8.8), (50, 3, 21.8), (50, 5, 55.3), (50, 8, 88.7),
])
def test_front_share_against_criteria_count(front_vs_m, n, m, expected):
    assert pct(cell(front_vs_m, n=n, m=m)["share"]) == pytest.approx(expected, abs=0.5)


def test_measured_front_share_follows_the_recursion(front_vs_m):
    for r in front_vs_m:
        assert pct(r["share"]) == pytest.approx(pct(r["expected"]), abs=2.5), r


def test_more_criteria_never_shrink_the_front(front_vs_m):
    for n in C.EXP_N_SWEEP:
        shares = [cell(front_vs_m, n=n, m=m)["share"] for m in C.EXP_M_SWEEP]
        assert shares == sorted(shares)


@pytest.mark.parametrize("m,rho,expected", [
    (3, -0.4, 70.4), (3, 0.0, 36.6), (3, 0.6, 17.9), (3, 0.9, 9.7),
    (5, -0.2, 96.5), (5, 0.0, 70.8), (5, 0.6, 25.7), (5, 0.9, 12.1),
])
def test_front_share_against_correlation(front_vs_rho, m, rho, expected):
    assert pct(cell(front_vs_rho, m=m, rho=rho)["share"]) == pytest.approx(expected, abs=0.5)


def test_front_share_falls_with_correlation(front_vs_rho):
    for m in C.EXP_RHO_M:
        shares = [r["share"] for r in front_vs_rho if r["m"] == m]
        assert shares == sorted(shares, reverse=True)


@pytest.mark.parametrize("m,comparable,tau_layer,tau_equal", [
    (2, 49.2, 0.67, 0.72), (3, 24.5, 0.55, 0.64), (5, 6.1, 0.40, 0.58), (8, 0.8, 0.16, 0.58),
])
def test_order_against_hidden_utility(order, m, comparable, tau_layer, tau_equal):
    r = cell(order, m=m)
    assert pct(r["comparable"]) == pytest.approx(comparable, abs=0.5)
    assert r["tau_layer"] == pytest.approx(tau_layer, abs=0.02)
    assert r["tau_equal"] == pytest.approx(tau_equal, abs=0.02)


def test_the_equal_weight_sum_orders_better_than_the_layers_for_every_m(order):
    for r in order:
        assert r["tau_equal"] > r["tau_layer"], r


def test_comparable_share_follows_two_to_the_one_minus_m(order):
    for r in order:
        assert pct(r["comparable"]) == pytest.approx(pct(dom.comparable_share_independent(r["m"])), abs=1.0)


def test_top_alternatives_in_the_front(order):
    r2, r5 = cell(order, m=2), cell(order, m=5)
    assert pct(r2["top2_in_front"]) == pytest.approx(64.0, abs=0.5) and pct(r2["top3_in_front"]) == pytest.approx(34.0, abs=0.5)
    assert pct(r5["top2_in_front"]) == pytest.approx(95.5, abs=0.5) and pct(r5["top3_in_front"]) == pytest.approx(90.5, abs=0.5)


def test_one_added_or_removed_alternative_never_flips_the_layer_order(change):
    assert cell(change, k=1)["flip_share"] == 0.0 and cell(change, k=-1)["flip_share"] == 0.0
    assert cell(change, k=-1)["front_lost_share"] == 0.0


def test_added_alternatives_cost_front_members_and_flip_the_order_from_two_on(change):
    assert pct(cell(change, k=1)["front_lost_share"]) == pytest.approx(13.5, abs=0.5)
    assert pct(cell(change, k=5)["front_lost_share"]) == pytest.approx(42.0, abs=0.5)
    assert pct(cell(change, k=5)["flip_share"]) == pytest.approx(4.5, abs=0.5)
    assert cell(change, k=5)["flip_share"] > 0.0


@pytest.mark.parametrize("m,levels,front,comparable", [
    (3, 0, 36.6, 24.5), (3, 10, 25.2, 32.6), (3, 5, 17.3, 41.3), (3, 3, 13.1, 51.5),
])
def test_coarser_scales_give_more_dominance(scale, m, levels, front, comparable):
    r = cell(scale, m=m, levels=levels)
    assert pct(r["front_share"]) == pytest.approx(front, abs=0.5)
    assert pct(r["comparable"]) == pytest.approx(comparable, abs=0.5)


def test_coarser_scales_shrink_the_front_for_every_m(scale):
    for m in (3, 5, 8):
        shares = [cell(scale, m=m, levels=lv)["front_share"] for lv in (0, 10, 5, 3)]
        assert shares == sorted(shares, reverse=True)


@pytest.mark.parametrize("m,sigma,flip", [(3, 0.05, 10.5), (5, 0.05, 10.0), (3, 0.10, 17.2), (5, 0.10, 16.3)])
def test_noise_flips_front_membership(noise, m, sigma, flip):
    assert pct(cell(noise, m=m, sigma=sigma)["flip"]) == pytest.approx(flip, abs=0.7)


def test_noise_moves_members_out_and_outsiders_in(noise):
    r = cell(noise, m=5, sigma=0.05)
    assert pct(r["outsider_enters"]) == pytest.approx(19.5, abs=1.0)
    assert pct(r["member_leaves"]) == pytest.approx(6.9, abs=1.0)


def test_best_is_always_in_the_front_for_monotone_forms_and_not_with_a_target_value(monotony):
    for m in (3, 5):
        for form in ("linear", "konkav", "cobb"):
            assert cell(monotony, m=m, form=form)["best_in_front"] == 1.0
    assert pct(cell(monotony, m=3, form="zielwert")["best_in_front"]) == pytest.approx(83.0, abs=2.0)
    assert pct(cell(monotony, m=5, form="zielwert")["best_in_front"]) == pytest.approx(98.5, abs=1.5)
