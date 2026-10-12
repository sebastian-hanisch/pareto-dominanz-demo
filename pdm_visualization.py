"""Plotly-Figuren und die Tabelle der Matrix. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pdm_constants as C
from entscheidungsmatrix_generator import CRITERIA
from pdm_evaluation import Analysis, de

ARROW = {1: "↑", -1: "↓"}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def criterion_label(a: Analysis, k: int) -> str:
    """Kopfzeile eines Kriteriums: Nummer, Richtung, Einheit."""
    return f"K{k + 1} {ARROW[int(a.inst.directions[k])]} {a.inst.units[k]}"


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, reveal: bool = False) -> pd.DataFrame:
    """Matrix mit Schicht, Front-Zugehörigkeit und den Alternativen, die dominieren (optional: Rang nach verborgenem Nutzen)."""
    names = alt_names(a.inst.n)
    cols = {"Alternative": names}
    for k in range(a.inst.m):
        cols[criterion_label(a, k)] = a.inst.raw[:, k]
    cols["Schicht"] = a.layer
    cols["In der Front"] = np.where(a.front, "ja", "nein")
    dominators = []
    for j in range(a.inst.n):
        idx = np.flatnonzero(a.D[:, j])
        text = ", ".join(names[i] for i in idx[:6]) + (f" und {len(idx) - 6} weitere" if len(idx) > 6 else "")
        dominators.append(text if len(idx) else "–")
    cols["Dominiert von"] = dominators
    if reveal:
        cols["Rang nach Nutzen"] = a.utility_rank
    return pd.DataFrame(cols)


def style_table(a: Analysis, df: pd.DataFrame):
    """Front-Zeilen hell hervorgehoben; Rohwerte mit der Anzeigegenauigkeit des Generators und Dezimalkomma."""
    def row_style(row):
        color = "background-color: rgba(47,107,101,0.16)" if row["In der Front"] == "ja" else ""
        return [color] * len(row)

    decimals = {criterion_label(a, k): (0 if a.inst.levels else CRITERIA[k][5]) for k in range(a.inst.m)}
    formats = {col: (lambda v, d=d: de(v, d)) for col, d in decimals.items()}
    return df.style.apply(row_style, axis=1).format(formats)


def build_scatter(a: Analysis, ix: int, iy: int) -> go.Figure:
    """Zwei Kriterien als Streudiagramm; nach oben und rechts ist immer besser (↓-Kriterien kehren die Achse um).
    Die Farbe zeigt die Front im VOLLEN Raum aller Kriterien, bei mehr als zwei Kriterien also nicht die Front dieser Projektion."""
    inst = a.inst
    names = alt_names(inst.n)
    fig = go.Figure()
    for flag, label, color in ((False, "dominiert", C.COLOR_OTHER), (True, "in der Front", C.COLOR_FRONT)):
        sel = a.front == flag
        fig.add_trace(go.Scatter(
            x=inst.raw[sel, ix], y=inst.raw[sel, iy], mode="markers+text", text=[names[i] for i in np.flatnonzero(sel)], textposition="top center",
            marker=dict(size=11, color=color, line=dict(width=1, color="white")), name=label,
            hovertemplate="%{text}<br>" + criterion_label(a, ix) + ": %{x}<br>" + criterion_label(a, iy) + ": %{y}<extra></extra>"))
    if inst.m == 2 and a.front.sum() > 1:
        fr = np.flatnonzero(a.front)
        order = fr[np.argsort(inst.signed[fr, ix])]
        xs, ys = inst.raw[order, ix], inst.raw[order, iy]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_FRONT, width=1.2, dash="dot", shape="vh"), name="Grenze der dominierten Fläche", hoverinfo="skip"))
    fig.update_xaxes(title=criterion_label(a, ix) + " (rechts besser)", autorange="reversed" if inst.directions[ix] < 0 else True)
    fig.update_yaxes(title=criterion_label(a, iy) + " (oben besser)", autorange="reversed" if inst.directions[iy] < 0 else True)
    return _fixed(fig, 420)


def build_dominance_heatmap(a: Analysis) -> go.Figure:
    """Dominanzmatrix, nach Schicht und Nummer geordnet: Zeile i dominiert Spalte j."""
    order = np.lexsort((np.arange(a.inst.n), a.layer))
    names = alt_names(a.inst.n)
    labels = [f"{names[i]} (S{a.layer[i]})" for i in order]
    z = a.D[np.ix_(order, order)].astype(int)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale=[[0, "rgba(240,240,240,1)"], [1, C.COLOR_FRONT]], showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y} dominiert %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(title="wird dominiert", side="top", tickangle=-60)
    fig.update_yaxes(title="dominiert", autorange="reversed")
    return _fixed(fig, 460)


def build_front_vs_m(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    palette = {10: "#8A96A6", 20: C.COLOR_FRONT, 50: C.COLOR_GOLD}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["m"] for r in rs], y=[100 * r["share"] for r in rs], mode="lines+markers", name=f"n = {n} (gemessen)",
                                 line=dict(color=palette.get(n, C.COLOR_LINE))))
        fig.add_trace(go.Scatter(x=[r["m"] for r in rs], y=[100 * r["expected"] for r in rs], mode="lines", name=f"n = {n} (Rekursion)",
                                 line=dict(color=palette.get(n, C.COLOR_LINE), dash="dot"), hoverinfo="skip"))
    fig.update_xaxes(title="Kriterien m", dtick=1)
    fig.update_yaxes(title="Anteil unbesiegter Alternativen (%)", range=[0, 102])
    return _fixed(fig, 380)


def build_front_vs_rho(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for m in sorted({r["m"] for r in rows}):
        rs = [r for r in rows if r["m"] == m]
        fig.add_trace(go.Scatter(x=[r["rho"] for r in rs], y=[100 * r["share"] for r in rs], mode="lines+markers", name=f"m = {m}"))
    fig.update_xaxes(title="Gleichkorrelation ρ der Güten")
    fig.update_yaxes(title="Anteil unbesiegter Alternativen (%)", range=[0, 102])
    return _fixed(fig, 360)


def build_order(rows: list[dict]) -> go.Figure:
    ms = [r["m"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=[r["tau_layer"] for r in rows], mode="lines+markers", name="τ der Schichten", line=dict(color=C.COLOR_FRONT)))
    fig.add_trace(go.Scatter(x=ms, y=[r["tau_equal"] for r in rows], mode="lines+markers", name="τ der gleichgewichteten Summe", line=dict(color=C.COLOR_GOLD)))
    fig.add_trace(go.Scatter(x=ms, y=[r["comparable"] for r in rows], mode="lines+markers", name="vergleichbare Paare", line=dict(color=C.COLOR_LINE)))
    fig.add_trace(go.Scatter(x=ms, y=[r["comparable_expected"] for r in rows], mode="lines", name="2^(1−m)", line=dict(color=C.COLOR_LINE, dash="dot"), hoverinfo="skip"))
    fig.update_xaxes(title="Kriterien m", dtick=1)
    fig.update_yaxes(title="τ gegen den verborgenen Nutzen bzw. Anteil der Paare", range=[0, 1.02])
    return _fixed(fig, 380)


def build_scale(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    labels = {0: "stetig", 10: "10 Stufen", 5: "5 Stufen", 3: "3 Stufen"}
    for m in sorted({r["m"] for r in rows}):
        rs = [r for r in rows if r["m"] == m]
        fig.add_trace(go.Bar(x=[labels[r["levels"]] for r in rs], y=[100 * r["front_share"] for r in rs], name=f"m = {m}"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Anteil unbesiegter Alternativen (%)", range=[0, 102])
    return _fixed(fig, 340)


def build_noise(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for m in sorted({r["m"] for r in rows}):
        rs = [r for r in rows if r["m"] == m]
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rs], y=[100 * r["flip"] for r in rs], mode="lines+markers", name=f"m = {m}"))
    fig.update_xaxes(title="Rauschen σ auf der Güte-Skala [0, 1]")
    fig.update_yaxes(title="Anteil der Alternativen, deren Front-Zugehörigkeit kippt (%)", rangemode="tozero")
    return _fixed(fig, 340)


def build_weights(a: Analysis) -> go.Figure:
    """Verborgene Gewichte des Entscheiders."""
    fig = go.Figure(go.Bar(x=[f"K{k + 1}" for k in range(a.inst.m)], y=a.inst.weights, marker_color=C.COLOR_GOLD))
    fig.update_yaxes(title="Gewicht", range=[0, 1])
    return _fixed(fig, 240)
