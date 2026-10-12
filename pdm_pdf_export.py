"""PDF-Export der aktuellen Matrix (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from pdm_evaluation import Analysis, de
from pdm_visualization import alt_names

_REPLACE = {"↑": "+", "↓": "-", "–": "-", "—": "-", "τ": "tau", "ρ": "rho", "≥": ">=", "²": "2", "−": "-", "€": "EUR"}
MAX_ROWS = 45


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    inst = a.inst
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Pareto-Dominanz: was ohne Gewichte übrig bleibt"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    s = a.settings
    stufen = "stetig" if s.levels == 0 else f"{s.levels} Stufen"
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {s.n}, Kriterien m = {s.m}, Korrelation rho = {de(s.rho, 2)}, Skala: {stufen}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Unbesiegt (Front): {a.front_size} von {inst.n} ({de(100 * a.front_share, 0)} %), Schichten: {a.n_layers}.",
        f"Vergleichbare Paare: {de(100 * a.comparable, 1)} %"
        + (f" (bei unabhaengigen stetigen Kriterien erwartet: {de(100 * a.comparable_expected, 1)} %)." if a.comparable_expected is not None else "."),
        f"Der verborgene Beste des simulierten Entscheiders liegt in der Front: {'ja' if a.best_in_front else 'nein'}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    names = alt_names(inst.n)
    head = ["Alt."] + [f"K{k + 1} ({'+' if inst.directions[k] > 0 else '-'}) {inst.units[k]}" for k in range(inst.m)] + ["Schicht", "Front"]
    widths = [14] + [min(36, 150 / inst.m)] * inst.m + [18, 16]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for i in range(min(inst.n, MAX_ROWS)):
        row = [names[i]] + [f"{inst.raw[i, k]:g}" for k in range(inst.m)] + [str(a.layer[i]), "ja" if a.front[i] else "nein"]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if inst.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {inst.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Dominanz: A dominiert B, wenn A in keinem Kriterium schlechter und in mindestens einem besser ist. Identische Alternativen dominieren "
                            "sich nicht. Quelle: pareto-dominanz-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
