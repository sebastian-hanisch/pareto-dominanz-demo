"""Kern gegen Handrechnung: jede Einheit an der Mini-Instanz (conftest) und an kleinen Gegenbeispielen."""

import numpy as np
import pytest

import pdm_dominance as dom


def test_dominance_matrix_of_the_mini_instance(mini):
    D = dom.dominance_matrix(mini)
    pairs = {(i, j) for i in range(6) for j in range(6) if D[i, j]}
    # A->C, A->F, B->C, B->F, C->F, D->F, E->C, E->F
    assert pairs == {(0, 2), (0, 5), (1, 2), (1, 5), (2, 5), (3, 5), (4, 2), (4, 5)}


def test_identical_alternatives_do_not_dominate_each_other(mini):
    D = dom.dominance_matrix(mini)
    assert not D[1, 4] and not D[4, 1]


def test_front_and_layers_of_the_mini_instance(mini):
    D = dom.dominance_matrix(mini)
    assert dom.front_mask(D).tolist() == [True, True, False, True, True, False]
    assert dom.layers(D).tolist() == [1, 1, 2, 1, 1, 3]


def test_comparable_share_of_the_mini_instance(mini):
    assert dom.comparable_share(dom.dominance_matrix(mini)) == pytest.approx(8 / 15)


def test_dominated_by_lists(mini):
    D = dom.dominance_matrix(mini)
    assert dom.dominated_by(D, 2).tolist() == [0, 1, 4]
    assert dom.dominated_by(D, 5).tolist() == [0, 1, 2, 3, 4]
    assert dom.dominated_by(D, 0).tolist() == []


def test_identical_pairs_share_counts_exact_duplicates(mini):
    assert dom.identical_pairs_share(mini) == pytest.approx(1 / 15)


def test_strict_dominance_differs_from_weak_on_a_tie():
    G = np.array([[1.0, 2.0], [1.0, 3.0]])
    assert dom.dominance_matrix(G)[1, 0] and not dom.dominance_matrix(G, strict=True)[1, 0]


def test_weak_and_strict_agree_without_ties(mini):
    assert (dom.dominance_matrix(mini) == dom.dominance_matrix(mini, strict=True)).all()


def test_chain_has_one_layer_per_alternative():
    G = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert dom.layers(dom.dominance_matrix(G)).tolist() == [3, 2, 1]


def test_single_criterion_has_exactly_one_front_member_without_ties():
    G = np.array([[0.3], [0.9], [0.5]])
    assert dom.front_mask(dom.dominance_matrix(G)).tolist() == [False, True, False]


def test_opposed_criteria_have_everyone_in_the_front():
    x = np.linspace(0, 1, 6)
    G = np.column_stack([x, 1 - x])
    D = dom.dominance_matrix(G)
    assert dom.front_mask(D).all() and dom.comparable_share(D) == 0.0


def test_kendall_tau_b_by_hand():
    assert dom.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert dom.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert dom.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(6))   # ein Gleichstand in a
    assert dom.kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0


def test_expected_front_size_known_values():
    assert dom.expected_front_size(1, 5) == pytest.approx(1.0)
    assert dom.expected_front_size(7, 1) == pytest.approx(1.0)
    assert dom.expected_front_size(4, 2) == pytest.approx(25 / 12)       # harmonische Zahl H_4
    assert dom.expected_front_size(4, 3) == pytest.approx(415 / 144)     # Aufzählung aller Rangkonfigurationen (Plan, Phase 0)


def test_comparable_share_for_independent_criteria():
    assert dom.comparable_share_independent(2) == 0.5
    assert dom.comparable_share_independent(5) == pytest.approx(1 / 16)


def test_top_k_breaks_ties_by_lower_index():
    u = np.array([6.1, 6.6, 6.1, 5.0])
    assert dom.top_k(u, 3).tolist() == [1, 0, 2]
    assert dom.best_index(u) == 1


def test_equal_weight_score_is_the_row_mean(mini):
    assert dom.equal_weight_score(mini).tolist() == pytest.approx([17 / 3, 17 / 3, 4.0, 6.0, 17 / 3, 2.0])


def test_boundary_sizes_of_the_pair_shares():
    one = np.array([[1.0, 2.0]])
    assert dom.comparable_share(dom.dominance_matrix(one)) == 0.0 and dom.identical_pairs_share(one) == 0.0
    pair = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert dom.comparable_share(dom.dominance_matrix(pair)) == 1.0 and dom.identical_pairs_share(pair) == 0.0
    free = np.array([[1.0, 3.0], [2.0, 2.0]])
    assert dom.comparable_share(dom.dominance_matrix(free)) == 0.0
    same = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert dom.identical_pairs_share(same) == 1.0 and dom.comparable_share(dom.dominance_matrix(same)) == 0.0


def test_layers_and_front_of_a_single_alternative():
    D = dom.dominance_matrix(np.array([[5.0, 5.0]]))
    assert dom.front_mask(D).tolist() == [True] and dom.layers(D).tolist() == [1]
