# Konfigurationsmodell – Netze zu gegebenen Graden – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-konfigurationsmodell-demo.streamlit.app/)** -->

Zweites Stück der **Netzmodelle-Linie** (Graphentheorie) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es baut auf dem [Zufallsgraphen](https://github.com/sebastian-hanisch/zufallsgraph-demo) auf:
dort sind alle Paare gleich wahrscheinlich, die Grade sind Poisson-verteilt, hier ist die **Gradfolge vorgegeben** und nur noch die Verbindung zufällig. Verwandt sind
[strukturkennzahlen-demo](https://github.com/sebastian-hanisch/strukturkennzahlen-demo) (das Konfigurationsmodell als Nullmodell für Clustering und Assortativität) und [robustheit-demo](https://github.com/sebastian-hanisch/robustheit-demo) (Hubs und Perkolation):
dort dient das Modell als Vergleich, hier ist es das Thema, und die Frage ist, wie man es **richtig zieht**.

Die Demo zieht Gradfolgen aus einem Potenzgesetz (oder zwei Graden), baut Netze mit drei **Samplern** (Stub-Matching, Kantentausch, Chung–Lu), prüft das **Kriterium von Molloy und Reed** (Riesenkomponente genau dann, wenn ⟨k²⟩/⟨k⟩ größer als 2)
und misst, wie viele Grade jeder Sampler trifft. Gradfolgen, Graphikalität (Erdős–Gallai), Havel–Hakimi, die Sampler und die Kennzahlen sind eigene Python-Umsetzungen; `networkx` und `scipy` kommen nur als Gegenproben in die Tests.

## Warum dieses Problem

Reale Netze haben Hubs. Wer prüfen will, ob ein Muster mehr ist als „es gibt Hubs“, braucht Zufallsnetze mit **denselben Graden**. Dafür gibt es mehrere Verfahren, und sie sind nicht gleichwertig:
Das Stub-Matching ist schnell, erzeugt aber Schleifen und Mehrfachkanten; der Kantentausch trifft jeden Grad, braucht aber einen Startgraphen und eine Mischzeit; Chung–Lu trifft die Grade nur im Erwartungswert.
Die Kernfrage der Demo: **Wann hat ein Netz zu gegebenen Graden eine Riesenkomponente, und wie genau trifft welcher Sampler die Grade?**

## Modell

Gradfolge d₁, …, dₙ mit gerader Summe 2m. **Stub-Matching:** Jeder Knoten bekommt dᵢ Halbkanten, alle 2m werden gemischt und paarweise verbunden; Schleifen und Mehrfachkanten werden entfernt. **Kantentausch:** Start mit einem einfachen Graphen mit genau diesen Graden
(Havel–Hakimi), dann aus zwei Kanten (a, b) und (c, d) die Kanten (a, d) und (c, b), wenn keine Schleife und keine Mehrfachkante entsteht. **Chung–Lu:** jede Kante {i, j} unabhängig mit Wahrscheinlichkeit min(1, wᵢ·wⱼ / Σw).
**Molloy–Reed:** Riesenkomponente für n → ∞ genau dann, wenn κ = ⟨k²⟩/⟨k⟩ > 2 (unter schwachen Bedingungen an die Folge). Formeln und Lesehilfe stehen im 📐-Expander der App; die Literaturangaben dort stammen aus dem Gedächtnis und sind nicht im Original geprüft.

## Methodik

- **Kern** (`kfm_algorithm.py`): abgeschnittenes Potenzgesetz und zwei Grade als Gradfolgen, Erdős–Gallai-Test und Havel–Hakimi, Stub-Matching mit Zählung von Schleifen und Mehrfachkanten, Kantentausch als **träge Kette** (jeder Vorschlag zählt, auch abgelehnte) und als Zählung gelungener Tausche, Chung–Lu zeilenweise vektorisiert, Union-Find, Transitivität, Assortativität.
- **Auswertung** (`kfm_evaluation.py`): Sampler-Aufruf mit Rückfall auf das Stub-Matching bei nicht graphischer Folge, Kennzahlen (κ, Riesenkomponente, Hub-Verhältnis, Verlust an Halbkanten), vier Messreihen mit festen Plänen, alle Verdikte als reine Funktionen.
- **Oberfläche** (`app.py`): Ergebnis mit Gradverteilung und Hub-Streuung, Kernfrage, Messreihen auf Klick, Annahmen und Formeln.

## Befunde (gemessen, keine Behauptungen)

Feste Pläne (`SWEEP_SEEDS` 100000 bis 100004), Mediane. **Die Zufallszahlen eines `numpy`-Generators sind nicht über Versionen stabil:** die Tests prüfen deshalb Bänder um die Messzahlen; exakt sind nur die Zählungen am Orakel.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der Kern? | Graphikalität = `networkx` auf 300 Folgen, Havel–Hakimi trifft die Grade auf 200, Transitivität und Assortativität = `networkx` auf 25 Graphen, Komponenten = `scipy` auf 30. | `tests/test_oracle_konfigurationsmodell.py` |
| Sind die Sampler gleichverteilt? | Es gibt genau **70** einfache Graphen mit sechs Knoten und Grad 2 (Aufzählung aller Kantenmengen). Das Stub-Matching erzeugt einen einfachen Graphen mit Wahrscheinlichkeit **70 · 64 / 10 395 = 0,431** (jeder Graph aus 2!⁶ = 64 der 1·3·5·7·9·11 = 10 395 Paarungen), und die angenommenen sind gleichverteilt. Der träge Kantentausch ist es auch: bei der Folge (3, 2, 2, 2, 1, 1, 1) mit **88** einfachen Graphen ergibt er χ² = **87,5** bei 87 Freiheitsgraden. | `test_stub_matching_conditioned_on_simple…`, `test_lazy_edge_switch_chain…` |
| Verzerrt die Zählung gelungener Tausche? | **Ja, leicht.** Zählt man nur gelungene Tausche (so zählt `networkx.double_edge_swap`), steigt χ² bei gleicher Zahl der Stichproben auf **174,5** gegen 87,5: Zustände mit vielen möglichen Tauschen werden bevorzugt. | `test_counting_only_accepted_switches_gives_a_measurably_non_uniform_chain` |
| Wo liegt die Schwelle? | Zwei Grade (1 und 3), n = 20 000: bei κ = **2,00** hat die größte Komponente **4,3 %** der Knoten, bei κ = **2,12** **30,9 %**, bei κ = 2,23 **52,1 %**, bei κ = 2,50 **81,2 %**; unter der Schwelle (κ = 1,50) nur **0,2 %**. | `test_claims.py::test_molloy_reed_threshold` |
| Wie viele Kanten gehen beim Stub-Matching verloren? | Potenzgesetz ab Grad 1, n = 20 000: Schleifen und Mehrfachkanten sind bei γ = 2,1 und Graden bis n − 1 **11,6 %** der Kanten, mit struktureller Obergrenze (größter Grad 296) **1,0 %**; bei γ = 2,3 **2,5 %** und **0,4 %**; bei γ = 3,5 unter **0,01 %**. | `test_claims.py::test_stub_matching_defects` |
| Trifft Chung–Lu die Grade? | n = 3 000, γ = 2,2: ohne Obergrenze erreichen die zehn größten Hubs **84 %** ihres erwarteten Grades, **36 von 45** Hub-Paaren werden gekappt; zwischen den Hubs liegen **44** Kanten, beim Kantentausch **36**. Mit struktureller Obergrenze (größter Grad 96) sind es **100 %** bei **0** gekappten Paaren. | `test_claims.py::test_chung_lu_caps_the_hubs` |
| Wie lange braucht der Kantentausch? | Start: Watts–Strogatz-Netz mit Transitivität **0,65** (n = 1 000, 5 000 Kanten); nach 0,5 Vorschlägen je Kante **0,040**, nach 1 **0,0098**, nach 2 **0,0084**, Plateau **0,008**: ab **2 Vorschlägen je Kante** innerhalb von zwei Standardabweichungen (4 Pläne). | `test_claims.py::test_the_edge_switch_forgets_a_clustered_start_within_a_few_proposals_per_edge` |
| Presets | Standardfall: κ = **13,0**, Riesenkomponente **100 %**, **0,9 %** der Halbkanten gehen verloren; Hubs ohne Obergrenze: **6,5 %** verloren, die zehn größten Hubs erreichen **80 %**; Kantentausch: Hubs genau **100 %**; Chung–Lu ohne Obergrenze: Hubs **88 %**; γ = 4: κ = **1,3**, größte Komponente **0,6 %**. | `test_claims.py::test_presets_hold_what_their_help_texts_say` |

## Ehrliche Grenzen

- **Nur ein Potenzgesetz und zwei Grade.** Andere Gradverteilungen (Poisson, Exponentialverteilung, gemessene Folgen) fehlen; Poisson ist der Zufallsgraph im ersten Stück der Linie.
- **κ > 2 ist ein Grenzwert.** Bei endlicher Größe ist die Schwelle verschmiert (4,3 % bei κ = 2,00 statt 0), und bei Potenzgesetz-Graden mit γ < 3 wächst κ mit der Obergrenze ohne Schranke: dann sagt das Kriterium nur „ja“.
- **Der Vergleich Chung–Lu gegen Kantentausch gilt für diese Folgen.** Ohne Obergrenze ist die Folge in allen fünf Plänen graphisch; bei anderen γ und n kann der Kantentausch ausfallen (die App fällt dann auf das Stub-Matching zurück und sagt es).
- **Mischzeit nur für das gewählte Startnetz.** Das Watts–Strogatz-Netz mit Transitivität 0,65 ist ein ungünstiger, aber einfacher Start; das Plateau (0,008) ist der Mittelwert über vier Pläne bei 40 Vorschlägen je Kante, nicht bewiesen. Für andere Netze kann die Mischzeit länger sein; sie ist allgemein nicht bekannt.
- **Der Kantentausch hier ist die träge Kette.** Die üblichen Bibliotheken zählen gelungene Tausche; das ist für Kennzahlen meist unkritisch, verzerrt aber die Gleichverteilung messbar (χ² 174,5 gegen 87,5 bei sieben Knoten).
- **Literatur nicht im Original geprüft** (Molloy und Reed 1995, Chung und Lu 2002; aus dem Gedächtnis).

## Befunde und Korrekturen gegenüber dem Plan

- **Chung–Lu und die großen Grade:** der Plan nannte die Aussage eines Preprints (Chung–Lu ziehe zu viele Kanten zwischen großen Graden) „nicht reproduziert“. Die Vormessung hatte Chung–Lu mit dem **Stub-Matching-Mehrfachgraphen** verglichen (dort in einem Lauf 45 gegen 187 Hub-Kanten), und das ist kein fairer Vergleich. Mit dem Kantentausch als Referenz (ein einfacher Graph mit genau den Graden) liegen bei Chung–Lu **44** Kanten zwischen den zehn größten Hubs gegen **36**: Chung–Lu zieht **mehr** Hub–Hub-Kanten, obwohl die Hubs ihren Grad im Mittel nicht erreichen (84 %). Beides stimmt zugleich; die frühere Gegenaussage des Plans war falsch.
- **Mischzeit in Vorschlägen statt in gelungenen Tauschen:** die Vormessung zählte gelungene Tausche (etwa 2 m genügen). Die Demo zählt Vorschläge (träge Kette, abgelehnte zählen mit): ab 2 Vorschlägen je Kante.
- **Port:** 9157 statt 9156 (9156 vergab eine parallele Sitzung inzwischen an `parametrisiert-demo`).

## Verifikation

- **Kern:** Gradfolgen, Graphikalität an bekannten Folgen, Havel–Hakimi, Stub-Matching (zwei Halbkanten je Knoten erzeugen nur Doppelkante oder Schleifen), Zähler für Schleifen und Mehrfachkanten, Kantentausch an zwei Kanten und einem Weg, Chung–Lu-Wahrscheinlichkeit und Kappung, Kennzahlen am Dreieck mit Endknoten und am Stern.
- **Orakel:** `networkx` (Graphikalität, Havel–Hakimi, Transitivität, Assortativität), `scipy` (Komponenten), exakte Aufzählung aller einfachen Graphen mit gegebenen Graden gegen die Gleichverteilung beider Sampler, exakte Erwartung der Chung–Lu-Grade, große Simulation gegen Molloy–Reed.
- **Verdikte:** reine Funktionen, jede Zahl im Text an handgesetzten Werten (`tests/test_verdicts.py`).
- **Presets, Layout, Oberfläche:** wie in allen Demos; Auswahlfelder nur mit erlaubten Werten im Permalink.
- **Mutationsprüfung** (`tools/mutate_demo.py`): 197 Stellen im Kern `kfm_algorithm.py`, **166 gefangen** (19 davon durch wortgleiche Zeilen in Nachbar-Demos), 27 als äquivalent festgehalten, 4 enden in Zeitüberschreitungen (vermutlich Endlosschleifen; das Werkzeug zählt sie nicht). Die äquivalenten Stellen sind Vergleiche mit Zufallszahlen (`<` gegen `<=`, Wahrscheinlichkeit null), Wachen, die nichts ändern (leere Schleifen, `n_steps = 0`, doppelt abgesicherte Negativgrade und Summen), Schleifengrenzen am leeren Rand, die Reihenfolge, in der das Startnetz der Mischzeit-Reihe aufgebaut wird, und `e1`/`e2` im Kantentausch (die Gleichheit der Endpunkte ist vorher ausgeschlossen). Echte Lücken, die der erste Lauf zeigte, schließen `test_capped_pairs_does_not_count…`, `test_chung_lu_works_for_small_positive_totals`, `test_edge_switch_steps_with_one_step_or_two_edges_can_still_switch`, `test_giant_fraction_of_a_single_node_is_one`, `test_negative_degrees_are_never_graphical…` und `test_simplify_keeps_a_single_edge`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `kfm_algorithm.py` | Kern: Gradfolgen, Graphikalität, Sampler, Kennzahlen |
| `kfm_constants.py` | Regler-Grenzen, feste Pläne der Messreihen, Farben, Presets |
| `kfm_evaluation.py` | Sampler-Aufruf, Kennzahlen, Messreihen, Verdikte |
| `kfm_layout.py` | Fester Kopf (Stylesheet) |
| `kfm_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kfm_scenario.py` | Gradfolgen und Instanzen |
| `kfm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Verdikte, Randfälle, Presets, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

- **Wachstumsmodelle, Netze im Raum, Motive:** eigene Folgestücke der Linie.
- **Exponential Random Graphs (ERGM) mit nichtlokalen Nebenbedingungen:** nur Einordnung, Schätzung und Sampling sind aufwändig und entarten leicht.
- **Gerichtete und gewichtete Konfigurationsmodelle:** andere Gradfolgen (ein- und ausgehend), nicht Teil dieser Demo.
- **Exakte Gleichverteilung für beliebige Folgen:** bewiesen ist nur die Grenzverteilung der trägen Kette; die Mischzeit allgemein ist offen.

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

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
