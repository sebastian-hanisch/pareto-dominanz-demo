# Pareto-Dominanz: was ohne Gewichte übrig bleibt – Streamlit-Demo

<!-- Nach dem Deploy hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-pareto-dominanz-demo.streamlit.app/)** -->

Wurzel der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie fragt,
wie man Alternativen, die nach mehreren Kriterien gemessen sind, in eine Rangfolge bringt. Dieses erste Stück zeigt die einzige Ausscheidung, die ganz ohne Gewichte
auskommt: die **Pareto-Dominanz**. Die Folgestücke beginnen dort, wo sie aufhört. Die Linie ist die zweite Ranking-Linie des Portfolios; die Turnier-Linie schätzt eine
Rangfolge aus Paarvergleichen mit bekannter wahrer Stärke, hier gibt es keine wahre Stärke.

Vehikel ist eine **neutrale, simulierte Entscheidungsmatrix**: n Alternativen, m Kriterien mit Richtung und Einheit, über eine Gauß-Copula mit wählbarer Korrelation
verbunden. Orakel ist ein **simulierter Entscheider** mit verborgenen Gewichten und einer Nutzenform, die nur die Simulation kennt.

## Warum dieses Problem

**Sie verwirft nie den Besten des Entscheiders, aber sie ordnet kaum.** Eine Alternative dominiert eine andere, wenn sie in keinem Kriterium schlechter und in
mindestens einem besser ist. Das braucht keine Gewichte und keine Normierung, und es ist sicher: Wer in keinem Kriterium verliert, kann für einen Entscheider, dem mehr
immer nützt, nie schlechter sein. Die Kernfrage der Demo ist, wie viel dabei übrig bleibt, wenn man nur das tut.

## Befunde und Korrekturen gegenüber dem Plan

- Die Vorab-Messreihe (Phase 0) nutzte `numpy.default_rng` mit 2 000 bis 20 000 Wiederholungen; die Demo nutzt den SplitMix64-Strom und 200 feste Matrizen. Die
  Werte weichen um bis zu einem Prozentpunkt ab (etwa Front 36,6 % statt 36,2 % bei n = 20, m = 3). Alle Zahlen unten stammen aus der Demo.
- Der Plan sah vor, dass die Konvexhüllen-Frage („welche unbesiegten Alternativen können bei irgendeiner linearen Gewichtung Platz 1 belegen?“) in der Wurzel
  nicht erscheint. Dabei ist es geblieben; die Messung (18 bis 26 % der Front nie) gehört zur gewichteten Summe.
- Die erwartete Front-Größe folgt der Rekursion A(n,m) aus der mathematischen Formulierung. Sie ist hier gegen Aufzählung (n = 4) und Simulation geprüft, nicht
  aus der Quelle übernommen; die Quelle liefert nur die Größenordnung.
- Die Dominanz selbst hat keine Rangumkehr, und auch die Schichtenordnung kippt bei **einer** hinzugefügten oder entfernten Alternative nie. Das ist eine eigene Herleitung
  (nicht aus der Literatur), als Property-Test auf 400 Instanzen belegt (`test_adding_or_removing_one_alternative_never_flips_the_layer_order`), mit Gegenprobe: bei zwei
  hinzugefügten Alternativen kippt die Ordnung. Es ist ein Befund über die Dominanz, kein Freispruch für Verfahren mit Skalarisierung.

## Modell

- **Matrix:** n Alternativen (5 bis 60), m Kriterien (2 bis 8). Die Güten sind gleichverteilt auf [0, 1] und über eine Gauß-Copula mit Gleichkorrelation ρ verbunden
  (ρ > −1/(m−1)). Rohwerte entstehen über eine streng monotone Abbildung je Kriterium (Punkte, Euro, Minuten; Richtung ↑ oder ↓), auf Anzeigegenauigkeit gerundet. Auf
  Wunsch Bewertungsstufen 1 bis L (alle ↑).
- **Dominanz:** schwach (in jedem Kriterium mindestens so gut, in einem besser). Identische Alternativen dominieren sich nicht und bleiben beide in der Front.
- **Entscheider:** verborgene Gewichte, gleichverteilt über dem Simplex; Nutzenform linear, konkav, Cobb-Douglas oder mit Zielwert (nicht monoton).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20 (Experiment 1: 10, 20, 50), unabhängig vom Regler n.

Formeln und Literatur stehen im 📐-Expander der App.

## Methodik

Die Demo vergleicht nicht Verfahren, sondern misst eine Relation an einer simulierten Matrix:
1. **Front und Schichten** (nichtdominierte Sortierung) aus der paarweisen Dominanzmatrix.
2. **Vergleichbare Paare** gegen die Formel 2^(1−m) bei unabhängigen stetigen Kriterien.
3. **Ordnung:** Kendalls τ-b der Schichten gegen den verborgenen Nutzen, mit einer gleichgewichteten Summe als Maßstab.
4. **Stabilität:** Hinzufügen und Entfernen von Alternativen, gröbere Skalen, Messrauschen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mittel über 200 feste Matrizen, `tests/test_claims.py`, wenn nicht anders vermerkt.

| Frage | Befund | Test |
|---|---|---|
| Liegt der Beste des Entscheiders in der Front? | **Ja, immer**, bei linearer, konkaver und Cobb-Douglas-Nutzenform (je 300 Instanzen, 3 Formen; im Experiment 200 von 200 bei m = 3 und m = 5). | `test_oracle_pareto_dominanz.py::test_the_best_of_a_monotone…`, `test_best_is_always_in_the_front…` |
| Wie viel der Alternativen bleibt unbesiegt? (n = 20, unabhängig) | m = 2: 18 %, m = 3: 37 %, m = 4: 55 %, m = 5: 71 %, m = 6: 82 %, m = 8: 94,5 %. Bei n = 50: 9 %, 22 %, 55 %, 89 % (m = 2, 3, 5, 8). | `test_front_share_against_criteria_count` |
| Wie viele Paare sind überhaupt vergleichbar? | 2^(1−m): 49 % (m = 2), 25 % (3), 6 % (5), 0,8 % (8). | `test_order_against_hidden_utility`, `test_comparable_pair_share_matches…` |
| Welche Rolle spielt die Korrelation? (n = 20) | m = 3: ρ = −0,4: 70 %, ρ = 0: 37 %, ρ = 0,6: 18 %, ρ = 0,9: 10 %. m = 5: ρ = −0,2: 97 %, ρ = 0: 71 %, ρ = 0,6: 26 %, ρ = 0,9: 12 %. | `test_front_share_against_correlation` |
| Wie viel Ordnung steckt in den Schichten? | τ gegen den verborgenen Nutzen: 0,67 / 0,55 / 0,40 / 0,16 für m = 2 / 3 / 5 / 8; eine gleichgewichtete Summe 0,72 / 0,64 / 0,58 / 0,58. **Die Summe ordnet in jeder Zeile besser.** | `test_order_against_hidden_utility`, `test_the_equal_weight_sum_orders_better…` |
| Wie oft liegen die besten zwei bzw. drei in der Front? | m = 2: 64 % bzw. 34 %; m = 5: 95,5 % bzw. 90,5 %. | `test_top_alternatives_in_the_front` |
| Gröbere Skala (m = 3) | Front 37 % (stetig), 25 % (10 Stufen), 17 % (5), 13 % (3); vergleichbare Paare 25 % → 33 % → 41 % → 52 %. Gleichstände erzeugen zusätzliche Dominanz. | `test_coarser_scales_give_more_dominance` |
| Messrauschen σ = 0,05 auf der Güte | 10,5 % (m = 3) bzw. 10,0 % (m = 5) der Alternativen wechseln die Front-Zugehörigkeit; bei m = 5 rücken 19,5 % der übrigen ein. | `test_noise_flips_front_membership` |
| Eine Alternative kommt hinzu | Schichtenordnung kippt nie (0,0 %), die Front verliert aber in 13,5 % der Matrizen ein Mitglied; bei 5 hinzugefügten kippt die Ordnung in 4,5 %, die Front verliert in 42 %. Entfernen: kippt nie. | `test_one_added_or_removed…`, `test_added_alternatives_cost…` |
| Gilt die Sicherheit ohne Monotonie? | **Nein:** mit einem Zielwert-Kriterium liegt der Beste bei m = 3 nur in 83 % der Fälle in der Front (m = 5: 98,5 %, weil dort fast alles unbesiegt ist). | `test_best_is_always_in_the_front…` |

## Ehrliche Grenzen

- **Alles simuliert.** Der Generator ist eine Gauß-Copula mit Gleichkorrelation. Echte Kriterien haben Blockstruktur, Ausreißer und Ordinalskalen; die Aussagen gelten
  für diesen Generator.
- **Der Entscheider ist eine Formel.** Die Gewichte sind gleichverteilt über dem Simplex gezogen; ein Mensch mit extremen oder inkonsistenten Gewichten fehlt. Die
  Ordnungsaussage (Experiment 3) hängt an dieser Verteilung.
- **„Unbesiegt“ ist keine Empfehlung.** Eine große Front heißt nicht, dass die Entscheidung schwer wäre, nur dass Dominanz nichts ausscheidet. Gröbere Skalen verkleinern die
  Front, ohne dass die Alternativen unterscheidbarer würden.
- **Die Front ist unscharf.** Schon geringes Rauschen verschiebt Mitglieder (Experiment 4).
- **Kleine Instanzen.** n = 20 in den Experimenten 2 bis 4, Experiment 1 bis n = 50; bei einigen hundert Alternativen ändern sich die Absolutwerte.
- **Monotonie ist Voraussetzung.** Mit einem Zielwert-Kriterium ist die Dominanz kein sicherer Filter mehr.

## Verifikation

- **Kern:** Front, Schichten und Vergleichbarkeit gegen Handrechnung auf der Mini-Instanz (`tests/conftest.py`: 6 Alternativen, Front {A, B, D, E}, Schichten 1-1-2-1-1-3, 8 von 15
  Paaren vergleichbar) und gegen kleine Gegenbeispiele (Gleichstand, einziges Kriterium, gegenläufig).
- **Orakel** (`tests/test_oracle_pareto_dominanz.py`): Front und Schichten gegen Definitionsschleifen in reinem Python (200 Instanzen mit Gleichständen), Schichten gegen die
  Tiefe im Dominanz-DAG (`networkx`), erwartete Front-Größe gegen exakte Aufzählung aller Rangkonfigurationen (n = 4: 25/12 und 415/144) und gegen Simulation, Anteil
  vergleichbarer Paare gegen 2^(1−m), der Beste in der Front für drei Nutzenformen **und** die Gegenprobe mit Zielwert, Invarianz gegen streng monotone
  Umskalierung, Richtungsumkehr, Nullbefund bei einer hinzugefügten oder entfernten Alternative (Vollprüfung auf 400 Instanzen) mit Gegenprobe bei zwei, Kendalls τ-b
  gegen `scipy.stats`, LP-Aussage zur Mini-Instanz gegen `scipy.optimize.linprog`.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula gegen (6/π)·arcsin(ρ/2),
  Richtung und Bereich der Rohwerte, Bewertungsstufen, Entscheider.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Presets, Korrelationsgrenze bei wachsendem m, Permalink, PDF nach Klick, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `pdm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `pdm_dominance.py` | Kern: Dominanz, Front, Schichten, Vergleichbarkeit, Kendalls τ-b |
| `pdm_evaluation.py` | Kennzahlen der Instanz, Experimente über feste Matrizen, Verdikte |
| `pdm_pdf_export.py` | PDF-Export der aktuellen Matrix (fpdf2, erst nach Klick) |
| `pdm_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button, Korrelationsgrenze |
| `pdm_visualization.py` | Tabelle und Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy, networkx (nur Gegenproben) |
| `tests/` | Mini-Instanz, Kern, Generator, Orakel, Auswertung, Aussagen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Eine Rangfolge aus Gewichten** (gewichtete Summe, Normierung): Die Dominanz braucht beides nicht; die Wirkung der Normierung zeigt erst die gewichtete Summe. Hier steht
  nur die gleichgewichtete Summe als Maßstab in Experiment 3.
- **Die Konvexhülle** (welche unbesiegten Alternativen bei irgendeiner linearen Gewichtung Sieger sein können): gehört zur gewichteten Summe; das Mehrkriterien-Routing zeigt
  die entsprechende Verfehlungsquote an Pfaden.
- **Evolutionäre Suche und Sättigung der Population** (NSGA-II, NSGA-III): stehen in der Populations-Metaheuristik-Linie; hier ist die Menge gegeben.
- **Hypervolumen und Referenzpunkte:** nicht Teil der Dominanz an einer gegebenen Menge.
- **Rangumkehr der Verfahren mit Skalarisierung:** wird bei den Verfahren gemessen, nicht hier.
- **Unsichere Gewichte, Lernen aus Beispielurteilen, mehrere Entscheider:** der Entscheider ist hier ein einzelnes, verborgenes Orakel.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
