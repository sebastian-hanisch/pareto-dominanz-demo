"""Pareto-Dominanz: was ohne Gewichte übrig bleibt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Linie Mehrkriterien-Ranking. Eine neutrale, simulierte Entscheidungsmatrix; ein simulierter Entscheider mit verborgenen
Gewichten ist das Orakel. Kern: pdm_dominance.py, Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import pdm_constants as C
from entscheidungsmatrix_generator import FORM_LABELS, FORMS
from pdm_evaluation import (
    Settings,
    analyse,
    change_experiment,
    de,
    monotony_experiment,
    noise_experiment,
    order_experiment,
    scale_experiment,
    sweep_front_vs_m,
    sweep_front_vs_rho,
    verdict_front,
    verdict_monotony,
    verdict_order,
    verdict_scale,
)
from pdm_pdf_export import build_pdf
from pdm_presets import (
    apply_preset,
    bounds,
    clip_rho_state,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pdm_visualization import (
    alt_names,
    build_dominance_heatmap,
    build_front_vs_m,
    build_front_vs_rho,
    build_noise,
    build_order,
    build_scale,
    build_scatter,
    build_table,
    build_weights,
    criterion_label,
    style_table,
)

st.set_page_config(page_title="Pareto-Dominanz – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über 200 feste Matrizen …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _front_vs_m():
    return sweep_front_vs_m()


@st.cache_data(show_spinner=SPINNER)
def _front_vs_rho():
    return sweep_front_vs_rho()


@st.cache_data(show_spinner=SPINNER)
def _order():
    return order_experiment()


@st.cache_data(show_spinner=SPINNER)
def _change():
    return change_experiment()


@st.cache_data(show_spinner=SPINNER)
def _scale():
    return scale_experiment()


@st.cache_data(show_spinner=SPINNER)
def _noise():
    return noise_experiment()


@st.cache_data(show_spinner=SPINNER)
def _monotony():
    return monotony_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧭 Pareto-Dominanz: was ohne Gewichte übrig bleibt")
st.markdown(
    """
**Sie verwirft nie den Besten des Entscheiders, aber sie ordnet kaum.** Dies ist das erste Stück der Linie **Mehrkriterien-Ranking**: Alternativen, die nach mehreren
Kriterien gemessen sind, sollen in eine Rangfolge. Anders als in einem Turnier gibt es hier keine wahre Stärke; Orakel ist ein **simulierter Entscheider**, dessen
verborgene Gewichte die Simulation kennt. Eine Alternative **dominiert** eine andere, wenn sie in keinem Kriterium schlechter und in mindestens einem besser ist, und
dafür braucht man keine Gewichte. Die Demo zeigt, wie viel dabei übrig bleibt.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Dominanz", expanded=True):
    st.markdown(
        """
1. **Rohwerte mit Richtung.** Jede Alternative hat in jedem Kriterium einen Wert; bei **↑** ist mehr besser, bei **↓** weniger. Die Demo gleicht die Richtungen
   intern an, damit überall „höher ist besser“ gilt.
2. **Dominanz ist ein Paarvergleich.** A dominiert B, wenn A in jedem Kriterium mindestens so gut und in mindestens einem besser ist. Identische Alternativen
   dominieren sich nicht. Wer in einem Kriterium vorn und im anderen hinten liegt, ist mit dem anderen **nicht vergleichbar**.
3. **Front und Schichten.** Die **Front** (Schicht 1) sind die Alternativen, die von keiner anderen dominiert werden. Entfernt man sie, bildet die nächste
   Front die Schicht 2, und so weiter.
4. **Warum das sicher ist.** Wer in keinem Kriterium verliert, kann für einen Entscheider, dem mehr in jedem Kriterium immer nützt, nie schlechter sein. Der
   Beste liegt deshalb **immer** in der Front, ohne dass man seine Gewichte kennt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    clip_rho_state()
    rho = st.slider(
        "Korrelation ρ der Güten", C.rho_min(int(m)), C.RHO_MAX, step=C.RHO_STEP, key="rho_slider",
        help="Gleichkorrelation der Güten zwischen den Kriterien. Negativ: wer in einem gut ist, ist in den anderen eher schlecht (Kompromiss). "
             "Positiv: die Kriterien laufen gleich. Die Untergrenze hängt von der Kriterienzahl ab (−1/(m−1)).")
    levels = st.selectbox("Skala", C.LEVEL_CHOICES, key="levels_select", format_func=lambda v: C.LEVEL_LABELS[v],
                          help="Stetig: Rohwerte in Punkten, Euro und Minuten. Sonst Bewertungsstufen 1 bis L für alle Kriterien (alle ↑).")
    st.markdown("**Entscheider (simuliert)**")
    form = st.selectbox("Nutzenform", FORMS, key="form_select", format_func=lambda f: FORM_LABELS[f],
                        help="Wie der verborgene Entscheider aus den Güten einen Nutzen bildet. Die ersten drei sind in jedem Kriterium streng monoton; "
                             "Zielwert ist es nicht (siehe 🚧 Wo die Annahmen enden).")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "rho_slider": round(float(rho), 2), "levels_select": int(levels),
                   "form_select": form, "seed_input": int(seed)})

settings = Settings(int(n), int(m), round(float(rho), 2), int(levels), form, int(seed))
a = _analysis(settings)
inst = a.inst
names = alt_names(inst.n)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unbesiegt (Front)", f"{a.front_size} von {inst.n}", delta=f"{de(100 * a.front_share, 0)} % aller", delta_color="off")
c2.metric("Vergleichbare Paare", f"{de(100 * a.comparable, 1)} %",
          delta=(f"2^(1−m) = {de(100 * a.comparable_expected, 1)} %" if a.comparable_expected is not None else "keine Formel (ρ ≠ 0 oder Stufen)"),
          delta_color="off",
          help="Anteil der Paare, in denen eine Alternative die andere dominiert. Bei unabhängigen stetigen Kriterien ist er 2^(1−m); mit Korrelation oder "
               "Bewertungsstufen gilt die Formel nicht.")
c3.metric("Schichten", f"{a.n_layers}", delta="Schicht 1 = Front", delta_color="off",
          help="Schicht 1 ist die Front; entfernt man sie, bildet die nächste Front Schicht 2 usw.")
c4.metric("Identische Paare", f"{de(100 * a.identical, 1)} %", delta="dominieren sich nicht", delta_color="off",
          help="Paare mit in jedem Kriterium gleichen Werten. Sie dominieren sich nicht und bleiben beide in der Front.")

tab_table, tab_scatter, tab_heat = st.tabs(["Tabelle", "Streudiagramm", "Dominanzmatrix"])
reveal = st.session_state.get("reveal_dm", False)
with tab_table:
    st.dataframe(style_table(a, build_table(a, reveal)), hide_index=True, width="stretch", height=min(40 + 35 * inst.n, 560))
    st.caption("Hell hinterlegt: die Front. „Dominiert von“ nennt die Alternativen, die diese Zeile dominieren; „–“ heißt unbesiegt.")
with tab_scatter:
    sc1, sc2 = st.columns(2)
    for key, default in (("ax_x", 0), ("ax_y", 1)):
        if int(st.session_state.get(key, default)) >= inst.m:
            st.session_state[key] = min(default, inst.m - 1)
    ix = sc1.selectbox("Horizontale Achse", list(range(inst.m)), key="ax_x", format_func=lambda k: criterion_label(a, k))
    iy = sc2.selectbox("Vertikale Achse", list(range(inst.m)), index=min(1, inst.m - 1), key="ax_y", format_func=lambda k: criterion_label(a, k))
    st.plotly_chart(build_scatter(a, int(ix), int(iy)), width="stretch", key="scatter_chart")
    st.caption("Rechts und oben ist immer besser. Die Farbe zeigt die Front im Raum **aller** Kriterien: bei mehr als zwei Kriterien kann ein Punkt in dieser "
               "Projektion unbesiegt aussehen und trotzdem von einem Punkt dominiert sein, der in einem anderen Kriterium vorn liegt.")
with tab_heat:
    st.plotly_chart(build_dominance_heatmap(a), width="stretch", key="heat_chart")
    st.caption("Zeile dominiert Spalte, geordnet nach Schicht (S1 = Front). Wenige gefüllte Felder heißen: wenige Paare sind vergleichbar.")

if st.button("📄 PDF der aktuellen Matrix erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="pareto-dominanz.pdf", mime="application/pdf", key="pdf_download")

# --- Entscheider enthüllen -----------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Verwirft die Dominanz den Besten?")
st.toggle("🙈 Entscheider enthüllen (verborgene Gewichte und Nutzen)", key="reveal_dm")
if st.session_state.get("reveal_dm", False):
    r1, r2 = st.columns([1, 1])
    with r1:
        st.plotly_chart(build_weights(a), width="stretch", key="weights_chart")
        st.caption("Die verborgenen Gewichte des simulierten Entscheiders (gleichverteilt über dem Simplex gezogen). Die Dominanz kennt sie nicht.")
    with r2:
        top = [f"{names[i]}: {'in der Front' if a.front[i] else f'dominiert von {len(a.D[:, i].nonzero()[0])}'}" for i in a.top3]
        st.markdown("**Die drei besten nach Nutzen**\n\n" + "\n".join(f"{r + 1}. {t}" for r, t in enumerate(top)))
if a.best_in_front:
    st.success(f"Der beste nach verborgenem Nutzen ({names[a.best]}) liegt in der Front. Von den drei besten sind {a.top3_in_front} von {len(a.top3)} unbesiegt.")
else:
    st.warning(f"Der beste nach verborgenem Nutzen ({names[a.best]}) wird dominiert: Mit dieser Nutzenform (nicht streng monoton in jedem Kriterium) "
               "ist die Dominanz kein sicherer Filter mehr.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie viel bleibt übrig?")
st.caption("Anteil unbesiegter Alternativen bei unabhängigen stetigen Kriterien, Mittel über 200 feste Matrizen (SWEEP_SEEDS); gepunktet die erwartete Größe aus der "
           "Rekursion der mathematischen Formulierung. Das Experiment rechnet mit n = 10, 20 und 50, unabhängig vom Regler.")
rows1 = _front_vs_m()
st.plotly_chart(build_front_vs_m(rows1), width="stretch", key="front_vs_m_chart")
verdict(verdict_front(rows1))

st.subheader("🔬 Experiment 2: Wenn Kriterien zusammenlaufen")
st.caption(f"{C.EXP_N} Alternativen, Gleichkorrelation ρ der Güten, Mittel über 200 feste Matrizen. Die Untergrenze der Gleichkorrelation ist −1/(m−1).")
rows2 = _front_vs_rho()
st.plotly_chart(build_front_vs_rho(rows2), width="stretch", key="front_vs_rho_chart")
r_neg = next(r for r in rows2 if r["m"] == 3 and r["rho"] == -0.4)
r_pos = next(r for r in rows2 if r["m"] == 3 and r["rho"] == 0.9)
st.info(f"Bei 3 Kriterien bleiben mit ρ = −0,4 im Mittel {de(100 * r_neg['share'], 0)} % unbesiegt, mit ρ = 0,9 nur {de(100 * r_pos['share'], 0)} %: "
        "Dominanz ordnet viel, wenn die Kriterien gleichläufig sind, und wenig, wenn sie gegeneinander arbeiten.")

st.subheader("🔬 Experiment 3: Wie viel Ordnung steckt drin?")
st.caption(f"{C.EXP_N} Alternativen, unabhängige Kriterien, linearer Entscheider mit verborgenen Gewichten, Mittel über 200 feste Matrizen. τ ist Kendalls τ-b "
           "gegen den verborgenen Nutzen; die Schichten sind die Rangfolge, die die Dominanz allein liefert.")
rows3 = _order()
st.plotly_chart(build_order(rows3), width="stretch", key="order_chart")
verdict(verdict_order(rows3))
r5 = next(r for r in rows3 if r["m"] == 5)
st.markdown(
    f"Der Zweitbeste des Entscheiders liegt bei 5 Kriterien in {de(100 * r5['top2_in_front'], 0)} % der Fälle in der Front, die drei besten in "
    f"{de(100 * r5['top3_in_front'], 0)} %; bei 2 Kriterien sind es {de(100 * rows3[0]['top2_in_front'], 0)} % bzw. {de(100 * rows3[0]['top3_in_front'], 0)} %."
)
rows_change = _change()
st.markdown("**Was passiert, wenn eine Alternative hinzukommt oder wegfällt?** (15 Alternativen, 3 Kriterien)")
st.dataframe(
    [{"Änderung": ("1 Alternative fällt weg" if r["k"] < 0 else f"{r['k']} Alternative(n) kommen hinzu"),
      "Front verliert ein Mitglied": ("– (kann nicht passieren)" if r["k"] < 0 else f"{de(100 * r['front_lost_share'], 1)} %"),
      "Schichtenordnung eines alten Paares kippt": f"{de(100 * r['flip_share'], 1)} %"}
     for r in rows_change],
    hide_index=True, width="stretch")
st.caption("Die Dominanz zweier Alternativen hängt nur von diesen beiden ab, nie von den übrigen. Die Schichten ändern sich dennoch; ihre Reihenfolge kippt aber nach "
           "dieser Messung erst, wenn mehr als eine Alternative hinzukommt.")

st.subheader("🔬 Experiment 4: Skala und Rauschen")
st.caption(f"{C.EXP_N} Alternativen, unabhängige Kriterien, Mittel über 200 feste Matrizen. Oben: dieselben Alternativen auf gröberen Skalen. Unten: Messrauschen "
           "auf der Güte, danach die Front neu bestimmt.")
rows4 = _scale()
st.plotly_chart(build_scale(rows4), width="stretch", key="scale_chart")
verdict(verdict_scale(rows4))
rows4b = _noise()
st.plotly_chart(build_noise(rows4b), width="stretch", key="noise_chart")
rn = next(r for r in rows4b if r["m"] == 3 and r["sigma"] == 0.05)
st.warning(f"Bei Rauschen σ = 0,05 (rund ein Sechstel der Spannweite einer Güte) wechseln bei 3 Kriterien im Mittel {de(100 * rn['flip'], 0)} % der Alternativen die "
           f"Front-Zugehörigkeit: {de(100 * rn['member_leaves'], 0)} % der Mitglieder verlassen sie, {de(100 * rn['outsider_enters'], 0)} % der übrigen rücken ein. "
           "Der Rand der Front ist unscharf.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
rows_mono = _monotony()
z3 = next(r for r in rows_mono if r["m"] == 3 and r["form"] == "zielwert")
z5 = next(r for r in rows_mono if r["m"] == 5 and r["form"] == "zielwert")
rho_pos = next(r for r in rows2 if r["m"] == 5 and r["rho"] == 0.6)
rho_zero = next(r for r in rows2 if r["m"] == 5 and r["rho"] == 0.0)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jedes Kriterium ist monoton** (mehr bzw. weniger ist immer besser) | Mit einem Zielwert-Kriterium liegt der Beste des Entscheiders bei 3 Kriterien nur in {de(100 * z3['best_in_front'], 1)} % der Fälle in der Front (bei 5 Kriterien {de(100 * z5['best_in_front'], 1)} %, weil dort fast alles unbesiegt ist). | Aus Beispielen lernen |
| **Die Werte sind exakt** | Rauschen σ = 0,05 lässt bei 3 Kriterien {de(100 * rn['flip'], 0)} % der Alternativen die Front wechseln (Experiment 4). | – |
| **Die Kriterien sind unabhängig** | Im Generator sind sie es bis auf die gewählte Gleichkorrelation. Bei 5 Kriterien und ρ = 0,6 bleiben {de(100 * rho_pos['share'], 0)} % unbesiegt statt {de(100 * rho_zero['share'], 0)} % bei ρ = 0 (Experiment 2); echte Kriterien sind selten unabhängig. | Kriterien sind nicht unabhängig |
| **Die Skala ist fein** | Auf 3 Stufen erzeugen Gleichstände zusätzliche Dominanz, ohne dass sich die Güte der Alternativen unterscheidet (Experiment 4). | – |
| **Es gibt einen Entscheider** | Die Gewichte sind hier gleichverteilt über dem Simplex gezogen; ein realer Entscheider mit extremen oder inkonsistenten Gewichten fehlt. | Gewichte erheben |
"""
)
st.markdown("**Liegt der Beste in der Front?** (20 Alternativen, unabhängige Kriterien, 200 feste Matrizen)")
st.dataframe(
    [{"Kriterien": r["m"], "Nutzenform": FORM_LABELS[r["form"]], "Beste in der Front": f"{de(100 * r['best_in_front'], 1)} %"} for r in rows_mono],
    hide_index=True, width="stretch")
verdict(verdict_monotony(rows_mono))

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Matrix.** $n$ Alternativen, $m$ Kriterien; $g_{ik}$ ist die Güte der Alternative $i$ im Kriterium $k$ (höher ist besser; bei Kriterien mit Richtung ↓
steht das Negative des Rohwerts).

**Dominanz.** $i \succ j$ genau dann, wenn $g_{ik} \ge g_{jk}$ für alle $k$ und $g_{ik} > g_{jk}$ für mindestens ein $k$. Die Relation ist transitiv und
hängt nur von den beiden Alternativen ab. Die **Front** ist $\{j : \nexists\, i \succ j\}$; die **Schicht** von $j$ ist die Länge der längsten Kette von
Dominanzen, die in $j$ endet, plus eins.

**Sicherheit.** Ist der Nutzen $u$ des Entscheiders in jedem Kriterium streng wachsend und $i \succ j$, so gilt $u(i) > u(j)$. Ein Maximierer von $u$ liegt
deshalb in der Front. Mit einem Zielwert-Kriterium (nicht monoton) gilt das nicht.

**Invarianz.** Eine streng wachsende Abbildung je Kriterium ändert $\succ$ nicht: Die Dominanz kennt weder Einheiten noch Normierung.

**Vergleichbare Paare.** Bei unabhängigen stetigen Kriterien hat $g_{ik} \ge g_{jk}$ für alle $k$ die Wahrscheinlichkeit $2^{-m}$; ein Paar ist also mit
Wahrscheinlichkeit $2 \cdot 2^{-m} = 2^{1-m}$ vergleichbar.

**Erwartete Front-Größe.** Für $n$ unabhängige Alternativen mit unabhängigen stetigen Kriterien ist $A(n,1) = 1$ und
$A(n,m) = \sum_{k=1}^{n} A(k,m-1)/k$; für $m = 2$ ist $A(n,2) = H_n$ die harmonische Zahl. Die Rekursion ist hier gegen Aufzählung aller Rangkonfigurationen
für $n = 4$ ($25/12$ und $415/144$) und gegen Simulation geprüft, nicht aus der Quelle übernommen. Die Größenordnung $O((\ln n)^{m-1})$ steht bei
Bentley, Kung, Schkolnick und Thompson.

**Generator.** Güten über eine Gauß-Copula mit Gleichkorrelation $\rho$ (zulässig für $\rho > -1/(m-1)$), Randverteilungen gleichverteilt auf $[0,1]$.
Rohwerte aus einer streng monotonen Abbildung je Kriterium; der Nutzen des Entscheiders $u_i = \sum_k w_k\, v(g_{ik})$ mit $w$ gleichverteilt über dem Simplex
und $v$ = Identität, Wurzel oder Logarithmus.

**Literatur** (bibliografisch geprüft, Inhalt nur anhand der Zusammenfassung): J. L. Bentley, H. T. Kung, M. Schkolnick, C. D. Thompson, *On the Average Number
of Maxima in a Set of Vectors and Applications*, Journal of the ACM 25(4), 1978, S. 536–543. O. Barndorff-Nielsen, M. Sobel, *On the Distribution of the Number
of Admissible Points in a Vector Random Sample*, Theory of Probability and its Applications 11(2), 1966, S. 249–269.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
