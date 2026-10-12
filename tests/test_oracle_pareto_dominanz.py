"""Orakel-Tests pareto-dominanz-demo: Front, Schichten und Kennzahlen gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12).

Jeder Test rechnet auf einem anderen Weg als der Kern: Definitionsschleifen in reinem Python, networkx, Aufzählung mit Brüchen,
geschlossene Formeln, Eigenschaften, die aus der Definition folgen (Invarianz, der Beste liegt in der Front).
"""

from fractions import Fraction
from itertools import permutations, product

import numpy as np
import pytest

import entscheidungsmatrix_generator as gen
import pdm_dominance as dom


def loop_dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def loop_front(G):
    n = len(G)
    return [not any(loop_dominates(G[i], G[j]) for i in range(n) if i != j) for j in range(n)]


def loop_layers(G):
    n = len(G)
    layer, remaining, k = [0] * n, set(range(n)), 0
    while remaining:
        k += 1
        cur = [j for j in remaining if not any(loop_dominates(G[i], G[j]) for i in remaining if i != j)]
        for j in cur:
            layer[j] = k
        remaining -= set(cur)
    return layer


def instances(count=200):
    for s in range(count):
        n = 3 + s % 9
        m = 2 + s % 4
        levels = (0, 3, 5)[s % 3]
        yield gen.make_matrix(n, m, 0.0, gen.SplitMix64(s), levels).signed


def test_front_and_layers_match_the_definition_loops_on_200_instances():
    for G in instances():
        D = dom.dominance_matrix(G)
        assert dom.front_mask(D).tolist() == loop_front(G.tolist())
        assert dom.layers(D).tolist() == loop_layers(G.tolist())


def test_layers_equal_the_depth_in_the_dominance_dag_networkx():
    nx = pytest.importorskip("networkx")
    for G in list(instances(60)):
        D = dom.dominance_matrix(G)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(G)))
        graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.nonzero(D)))
        depth = {}
        for gi, generation in enumerate(nx.topological_generations(graph), start=1):
            for node in generation:
                depth[node] = gi
        assert dom.layers(D).tolist() == [depth[i] for i in range(len(G))]


def exact_expected_front(n, m):
    """Aufzählung: alle (n!)^m Rangkonfigurationen sind gleich wahrscheinlich (stetige, unabhängige Kriterien)."""
    total, count = Fraction(0), 0
    for combo in product(list(permutations(range(n))), repeat=m):
        rows = [tuple(combo[k][i] for k in range(m)) for i in range(n)]
        total += sum(1 for j in range(n) if not any(loop_dominates(rows[i], rows[j]) for i in range(n) if i != j))
        count += 1
    return total / count


@pytest.mark.parametrize("m,expected", [(2, Fraction(25, 12)), (3, Fraction(415, 144))])
def test_expected_front_size_recursion_matches_exact_enumeration(m, expected):
    assert exact_expected_front(4, m) == expected
    assert dom.expected_front_size(4, m) == pytest.approx(float(expected), rel=1e-12)


@pytest.mark.parametrize("m", [3, 5])
def test_simulated_front_size_is_inside_the_band_of_the_recursion(m):
    sizes = [dom.front_mask(dom.dominance_matrix(gen.make_matrix(20, m, 0.0, gen.SplitMix64(1000 + s)).signed)).sum() for s in range(1500)]
    se = np.std(sizes, ddof=1) / np.sqrt(len(sizes))
    assert abs(np.mean(sizes) - dom.expected_front_size(20, m)) < 4.5 * se


@pytest.mark.parametrize("m", [2, 3, 5])
def test_comparable_pair_share_matches_two_to_the_one_minus_m(m):
    shares = [dom.comparable_share(dom.dominance_matrix(gen.make_matrix(40, m, 0.0, gen.SplitMix64(5000 + s)).signed)) for s in range(200)]
    se = np.std(shares, ddof=1) / np.sqrt(len(shares))
    assert abs(np.mean(shares) - dom.comparable_share_independent(m)) < 4.5 * se


@pytest.mark.parametrize("form", ["linear", "konkav", "cobb"])
def test_the_best_of_a_monotone_decision_maker_is_always_in_the_front(form):
    for s in range(300):
        m = 2 + s % 5
        x = gen.make_matrix(15, m, [0.0, 0.4][s % 2], gen.SplitMix64(7000 + s), 0, form)
        assert dom.front_mask(dom.dominance_matrix(x.signed))[dom.best_index(x.utility)]


def test_the_target_value_form_can_lose_its_best_alternative():
    misses = 0
    for s in range(400):
        x = gen.make_matrix(15, 3, 0.0, gen.SplitMix64(9000 + s), 0, "zielwert")
        misses += not dom.front_mask(dom.dominance_matrix(x.signed))[dom.best_index(x.utility)]
    assert misses > 0      # Gegenprobe: die Eigenschaft hängt an der Monotonie


def test_front_and_layers_are_invariant_under_strictly_monotone_rescaling():
    transforms = [np.exp, lambda v: v ** 3, lambda v: -1.0 / (v + 0.1), np.log]
    for s in range(100):
        g = gen.copula_uniforms(15, 4, 0.2, gen.SplitMix64(11000 + s))
        h = np.column_stack([t(g[:, k]) for k, t in enumerate(transforms)])
        assert (dom.dominance_matrix(g) == dom.dominance_matrix(h)).all()


def test_reversing_a_direction_with_the_sign_gives_the_same_front_as_negating_the_values():
    x = gen.make_matrix(25, 4, 0.0, gen.SplitMix64(13))
    D_signed = dom.dominance_matrix(x.signed)
    D_loop = np.array([[loop_dominates(x.raw[i] * x.directions, x.raw[j] * x.directions) for j in range(25)] for i in range(25)])
    assert (D_signed == D_loop).all()
    assert not (D_signed == dom.dominance_matrix(x.raw)).all()      # Richtung ignorieren ändert die Front (Zweig-Test)


def layer_order_flips(L_old, L_new):
    ordered = L_old[:, None] < L_old[None, :]
    return bool((ordered & (L_new[:, None] > L_new[None, :])).any())


def test_adding_or_removing_one_alternative_never_flips_the_layer_order():
    for s in range(400):
        m = 2 + s % 2
        n = 4 + s % 5
        g = gen.copula_uniforms(n + 1, m, 0.0, gen.SplitMix64(15000 + s))
        L_old = dom.layers(dom.dominance_matrix(g[:n]))
        L_new = dom.layers(dom.dominance_matrix(g))[:n]
        assert not layer_order_flips(L_old, L_new), ("hinzufügen", s)
        drop = s % (n + 1)
        keep = np.arange(n + 1) != drop
        L_all = dom.layers(dom.dominance_matrix(g))
        assert not layer_order_flips(L_all[keep], dom.layers(dom.dominance_matrix(g[keep]))), ("entfernen", s)


def test_adding_two_alternatives_can_flip_the_layer_order():
    found = 0
    for s in range(3000):
        g = gen.copula_uniforms(12, 2, 0.0, gen.SplitMix64(20000 + s))
        L_old = dom.layers(dom.dominance_matrix(g[:10]))
        L_new = dom.layers(dom.dominance_matrix(g))[:10]
        found += layer_order_flips(L_old, L_new)
    assert found > 0      # Gegenprobe: der Nullbefund für eine Alternative gilt nicht für zwei


def test_kendall_tau_b_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.integers(0, 5, 12).astype(float)
        b = rng.integers(0, 5, 12).astype(float)
        ref = stats.kendalltau(a, b).statistic
        got = dom.kendall_tau_b(a, b)
        assert got == pytest.approx(0.0 if np.isnan(ref) else ref, abs=1e-12)


def test_mini_instance_winners_under_linear_weights_by_linear_programming(mini):
    optimize = pytest.importorskip("scipy.optimize")

    def strictly_best_possible(i):
        diffs = np.delete(mini[i] - mini, i, axis=0)
        if (np.abs(diffs).sum(axis=1) == 0).any():          # identische Alternative: nie strikt besser
            return False
        m = mini.shape[1]
        res = optimize.linprog(np.r_[np.zeros(m), -1.0], A_ub=np.hstack([-diffs, np.ones((len(diffs), 1))]), b_ub=np.zeros(len(diffs)),
                               A_eq=[[1.0] * m + [0.0]], b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs")
        return res.status == 0 and -res.fun > 1e-9

    assert [strictly_best_possible(i) for i in range(6)] == [True, False, False, True, False, False]


def test_mini_instance_utilities_and_best(mini, mini_weights):
    u = mini @ mini_weights
    assert u.tolist() == pytest.approx([6.6, 6.1, 4.6, 5.3, 6.1, 2.1])
    assert dom.best_index(u) == 0
