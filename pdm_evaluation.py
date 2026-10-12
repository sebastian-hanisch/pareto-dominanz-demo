"""Auswertung: Kennzahlen der Instanz, die Experimente über feste Instanzen (SWEEP_SEEDS) und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pdm_constants as C
import pdm_dominance as dom
from entscheidungsmatrix_generator import Matrix, SplitMix64, copula_uniforms, make_matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    rho: float = C.DEFAULT_RHO
    levels: int = 0
    form: str = C.FORM_DEFAULT
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    inst: Matrix
    D: np.ndarray
    front: np.ndarray
    layer: np.ndarray
    front_size: int
    front_share: float
    n_layers: int
    comparable: float
    comparable_expected: float | None     # nur bei unabhängigen stetigen Kriterien
    front_expected: float | None          # dito: Rekursion A(n, m)
    best: int                             # Index des besten nach verborgenem Nutzen
    best_in_front: bool
    top3: np.ndarray
    top3_in_front: int
    utility_rank: np.ndarray              # Rang 1..n je Alternative nach Nutzen
    tau_layer: float
    tau_equal: float
    identical: float


def analyse(settings: Settings) -> Analysis:
    inst = make_matrix(settings.n, settings.m, settings.rho, SplitMix64(settings.seed), settings.levels, settings.form)
    D = dom.dominance_matrix(inst.signed)
    front = dom.front_mask(D)
    layer = dom.layers(D)
    top3 = dom.top_k(inst.utility, min(3, inst.n))
    rank = np.empty(inst.n, dtype=int)
    rank[dom.top_k(inst.utility, inst.n)] = np.arange(1, inst.n + 1)
    independent = settings.rho == 0.0 and settings.levels == 0
    return Analysis(
        settings=settings, inst=inst, D=D, front=front, layer=layer,
        front_size=int(front.sum()), front_share=float(front.mean()), n_layers=int(layer.max()),
        comparable=dom.comparable_share(D),
        comparable_expected=dom.comparable_share_independent(inst.m) if independent else None,
        front_expected=dom.expected_front_size(inst.n, inst.m) if independent else None,
        best=dom.best_index(inst.utility), best_in_front=bool(front[dom.best_index(inst.utility)]),
        top3=top3, top3_in_front=int(front[top3].sum()), utility_rank=rank,
        tau_layer=dom.kendall_tau_b(-layer, inst.utility),
        tau_equal=dom.kendall_tau_b(dom.equal_weight_score(inst.goodness), inst.utility),
        identical=dom.identical_pairs_share(inst.signed),
    )


def _matrices(n, m, rho=0.0, levels=0, form="linear", seeds=C.SWEEP_SEEDS):
    for s in seeds:
        yield make_matrix(n, m, rho, SplitMix64(s), levels, form)


# ----------------------------------------------------------------------------- Experiment 1: wie viel bleibt übrig?

def sweep_front_vs_m(ns=C.EXP_N_SWEEP, ms=C.EXP_M_SWEEP, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for n in ns:
        for m in ms:
            shares = [dom.front_mask(dom.dominance_matrix(x.signed)).mean() for x in _matrices(n, m, seeds=seeds)]
            rows.append({"n": n, "m": m, "share": float(np.mean(shares)), "expected": dom.expected_front_size(n, m) / n})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: Korrelation

def sweep_front_vs_rho(ms=C.EXP_RHO_M, rhos=C.EXP_RHO_SWEEP, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for m in ms:
        lo = -1.0 / (m - 1)
        for rho in rhos:
            if rho <= lo:
                continue
            shares = [dom.front_mask(dom.dominance_matrix(x.signed)).mean() for x in _matrices(n, m, rho, seeds=seeds)]
            rows.append({"m": m, "rho": rho, "share": float(np.mean(shares))})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: wie viel Ordnung steckt drin?

def order_experiment(ms=(2, 3, 5, 8), n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for m in ms:
        comp, tl, te, p2, p3, kk = [], [], [], [], [], []
        for x in _matrices(n, m, seeds=seeds):
            D = dom.dominance_matrix(x.signed)
            L = dom.layers(D)
            top = dom.top_k(x.utility, 3)
            comp.append(dom.comparable_share(D))
            tl.append(dom.kendall_tau_b(-L, x.utility))
            te.append(dom.kendall_tau_b(dom.equal_weight_score(x.goodness), x.utility))
            p2.append(bool((L[top[:2]] == 1).all()))
            p3.append(bool((L[top] == 1).all()))
            kk.append(int(L[top].max()))
        rows.append({
            "m": m, "comparable": float(np.mean(comp)), "comparable_expected": dom.comparable_share_independent(m),
            "tau_layer": float(np.mean(tl)), "tau_equal": float(np.mean(te)),
            "top2_in_front": float(np.mean(p2)), "top3_in_front": float(np.mean(p3)), "layers_to_top3": float(np.mean(kk)),
        })
    return rows


def change_experiment(n=C.EXP_CHANGE_N, m=C.EXP_CHANGE_M, ks=C.EXP_CHANGE_K, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Eine Alternative kommt hinzu (k = 1, 2, ...) oder fällt weg (k = -1): kippt die Schichtenordnung alter Paare, verliert die Front ein Mitglied?"""
    rows = []
    for k in tuple(ks) + (-1,):
        flips = lost = 0
        for s in seeds:
            rng = SplitMix64(s)
            g = copula_uniforms(n + abs(k), m, 0.0, rng)
            if k > 0:
                old, new = g[:n], g
                L_old, L_new = dom.layers(dom.dominance_matrix(old)), dom.layers(dom.dominance_matrix(new))[:n]
            else:
                drop = int(rng.u64(1)[0] % np.uint64(n + 1))
                keep = np.arange(n + 1) != drop
                L_all = dom.layers(dom.dominance_matrix(g))
                L_old, L_new = L_all[keep], dom.layers(dom.dominance_matrix(g[keep]))
            ordered = L_old[:, None] < L_old[None, :]
            flips += bool((ordered & (L_new[:, None] > L_new[None, :])).any())
            lost += bool(((L_old == 1) & (L_new != 1)).any())
        rows.append({"k": k, "flip_share": flips / len(seeds), "front_lost_share": lost / len(seeds)})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Skala und Rauschen

def scale_experiment(ms=(3, 5, 8), levels=(0, 10, 5, 3), n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for m in ms:
        for lv in levels:
            fr, cp, idn = [], [], []
            for x in _matrices(n, m, 0.0, lv, seeds=seeds):
                D = dom.dominance_matrix(x.signed)
                fr.append(dom.front_mask(D).mean())
                cp.append(dom.comparable_share(D))
                idn.append(dom.identical_pairs_share(x.signed))
            rows.append({"m": m, "levels": lv, "front_share": float(np.mean(fr)), "comparable": float(np.mean(cp)), "identical": float(np.mean(idn))})
    return rows


def noise_experiment(ms=C.EXP_NOISE_M, sigmas=C.EXP_NOISE_SIGMA, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Messrauschen auf der Güte: Anteil der Alternativen, deren Zugehörigkeit zur Front kippt."""
    rows = []
    for m in ms:
        for sg in sigmas:
            flip = out = into = 0.0
            for s in seeds:
                rng = SplitMix64(s)
                g = copula_uniforms(n, m, 0.0, rng)
                noisy = g + sg * rng.normal(n * m).reshape(n, m)
                f0 = dom.front_mask(dom.dominance_matrix(g))
                f1 = dom.front_mask(dom.dominance_matrix(noisy))
                flip += float((f0 != f1).mean())
                out += float((f0 & ~f1).sum() / max(f0.sum(), 1))
                into += float((~f0 & f1).sum() / max((~f0).sum(), 1))
            k = len(seeds)
            rows.append({"m": m, "sigma": sg, "flip": flip / k, "member_leaves": out / k, "outsider_enters": into / k})
    return rows


# ----------------------------------------------------------------------------- Annahmen: Monotonie

def monotony_experiment(ms=(3, 5), forms=("linear", "konkav", "cobb", "zielwert"), n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Liegt der Beste des Entscheiders in der Front? Streng monotone Nutzenformen immer, mit Zielwert-Kriterium nicht."""
    rows = []
    for m in ms:
        for form in forms:
            hit = [bool(dom.front_mask(dom.dominance_matrix(x.signed))[dom.best_index(x.utility)]) for x in _matrices(n, m, 0.0, 0, form, seeds)]
            rows.append({"m": m, "form": form, "best_in_front": float(np.mean(hit)), "runs": len(hit)})
    return rows


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def verdict_front(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1: bleibt bei 5 Kriterien mehr als die Hälfte unbesiegt?"""
    row = next(r for r in rows if r["n"] == C.EXP_N and r["m"] == 5)
    low = next(r for r in rows if r["n"] == C.EXP_N and r["m"] == 2)
    kind = "warning" if row["share"] > 0.5 else "info"
    return kind, (f"Bei {C.EXP_N} Alternativen sind mit 2 Kriterien im Mittel {de(100 * low['share'], 0)} % unbesiegt, mit 5 Kriterien {de(100 * row['share'], 0)} %. "
                  "Je mehr Kriterien, desto weniger scheidet die Dominanz aus.")


def verdict_order(rows: list[dict]) -> tuple[str, str]:
    r = next(x for x in rows if x["m"] == 5)
    better = r["tau_equal"] > r["tau_layer"]
    text = (f"Sicher, aber stumpf: Die Dominanz verwirft den Besten nie, ordnet aber bei 5 Kriterien nur τ = {de(r['tau_layer'], 2)} gegen den verborgenen Nutzen, "
            f"eine gleichgewichtete Summe {de(r['tau_equal'], 2)}. Vergleichbar sind {de(100 * r['comparable'], 1)} % der Paare.")
    return ("warning" if better else "info"), text


def verdict_scale(rows: list[dict]) -> tuple[str, str]:
    base = next(r for r in rows if r["m"] == 3 and r["levels"] == 0)
    coarse = next(r for r in rows if r["m"] == 3 and r["levels"] == 3)
    return "warning", (f"Bei 3 Kriterien bleiben auf stetiger Skala {de(100 * base['front_share'], 0)} % unbesiegt, auf 3 Stufen {de(100 * coarse['front_share'], 0)} %: "
                       "Gleichstände erzeugen zusätzliche Dominanz. Das ist ein Skalenartefakt, kein Informationsgewinn.")


def verdict_monotony(rows: list[dict]) -> tuple[str, str]:
    z = next(r for r in rows if r["m"] == 3 and r["form"] == "zielwert")
    return "warning", (f"Mit einem Zielwert-Kriterium liegt der Beste des Entscheiders bei 3 Kriterien nur in {de(100 * z['best_in_front'], 1)} % der Fälle in der Front, "
                       "mit streng monotonen Nutzenformen immer: Die Sicherheit der Dominanz setzt monotone Kriterien voraus.")
