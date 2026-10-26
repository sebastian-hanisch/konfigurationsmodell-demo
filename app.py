"""Konfigurationsmodell – Netze zu gegebenen Graden - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Linie „Netzmodelle“: Wie erzeugt man Netze zu einer vorgegebenen Gradfolge, wann haben sie eine Riesenkomponente (Molloy–Reed), und welcher Sampler trifft die Grade?
Der Kern liegt in kfm_algorithm.py (Gradfolgen, Graphikalität, Stub-Matching, Kantentausch, Chung–Lu).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import kfm_constants as C
import kfm_evaluation as ev
from kfm_evaluation import Settings, analyse, num, verdict_chung_lu, verdict_defects, verdict_mixing, verdict_molloy, verdict_network
from kfm_layout import dunkles_design, kopf_css
from kfm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kfm_visualization import build_chung_lu, build_defects, build_degrees, build_hubs, build_mixing, build_molloy

st.set_page_config(page_title="Konfigurationsmodell – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _series():
    """Die vier Messreihen (festes Programm, keine Eingabe)."""
    return ev.sweep_molloy_reed(), ev.sweep_defects(), ev.series_chung_lu(), ev.series_mixing()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kfm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🕸️ Konfigurationsmodell – Netze zu gegebenen Graden")
st.markdown(
    """
Ein Zufallsnetz mit lauter gleichen Wahrscheinlichkeiten kennt keine Hubs. Das **Konfigurationsmodell** legt statt dessen die **Gradfolge** fest, also wie viele Kanten jeder Knoten hat, und würfelt nur noch, wer mit wem verbunden wird.
Daraus folgt eine genaue Aussage: Ab dem Verhältnis **κ = ⟨k²⟩ / ⟨k⟩ größer als 2** (Molloy und Reed) hat das Netz eine **Riesenkomponente**; bei einem flachen Potenzgesetz mit Hubs ist κ groß, bei einem steilen kann es darunter liegen. Die Demo prüft das und vergleicht
drei **Sampler**: das **Stub-Matching** (schnell, aber mit Schleifen und Mehrfachkanten), den **Kantentausch** (ein einfacher Graph mit genau den Graden) und **Chung–Lu** (Grade nur im Erwartungswert).
Zweites Stück der Linie „Netzmodelle“; es baut auf dem Zufallsgraphen auf, dessen Grade Poisson-verteilt sind.

Weiter unten: 🔬 **Messreihen** (Molloy–Reed, Schleifen und Mehrfachkanten, Chung–Lu gegen Kantentausch, Mischzeit), 🚧 **Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Konfigurationsmodell", expanded=True):
    st.markdown(
        """
1. **Grade festlegen:** Jeder Knoten bekommt einen Grad aus einem Potenzgesetz (viele kleine, wenige große Grade); die Summe der Grade muss gerade sein.
2. **Halbkanten verbinden:** Jeder Knoten hat so viele **Halbkanten** wie sein Grad; alle werden gemischt und paarweise verbunden (Stub-Matching). Dabei können Schleifen und doppelte Kanten entstehen, die man entfernt.
3. **Oder genau treffen:** Der **Kantentausch** startet mit einem einfachen Graphen mit genau diesen Graden (Havel–Hakimi) und tauscht immer wieder die Enden zweier Kanten: die Grade bleiben, der Graph wird zufällig.
4. **Prüfen:** Die Riesenkomponente wird gezählt und mit dem Kriterium κ > 2 verglichen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Gradfolge**")
    n = st.slider("Knoten n", *bounds("n_slider"), key="n_slider", step=100, help="Wie viele Knoten das Netz hat. Chung–Lu rechnet quadratisch in n.")
    gamma = st.slider("Exponent γ", *bounds("gamma_slider"), key="gamma_slider", step=0.1, help="Potenzgesetz P(k) ~ k^(−γ): kleiner heißt flacher, mehr und größere Hubs.")
    kmin = st.slider("Kleinster Grad", *bounds("kmin_slider"), key="kmin_slider", help="Kleinster erlaubter Grad.")
    cutoff = st.selectbox("Obergrenze der Grade", list(C.CUTOFFS), format_func=C.CUTOFFS.get, key="cutoff_select",
                          help="Strukturell: höchstens √(n·⟨k⟩), dort sind Kanten zwischen Hubs nicht mehr überbucht. Frei: bis n − 1.")
    st.markdown("**Sampler**")
    sampler = st.selectbox("Verfahren", list(C.SAMPLERS), format_func=C.SAMPLERS.get, key="sampler_select",
                           help="Stub-Matching und Chung–Lu sind schnell; der Kantentausch liefert einen einfachen Graphen mit genau den Graden (wenn die Folge graphisch ist).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "gamma_slider": round(float(gamma), 1), "kmin_slider": int(kmin), "cutoff_select": cutoff, "sampler_select": sampler, "seed_input": int(seed)})

settings = Settings(n=int(n), gamma=round(float(gamma), 1), kmin=int(kmin), cutoff=cutoff, sampler=sampler, seed=int(seed))
with st.spinner("Baue das Netz …"):
    a = _analysis(settings)
net = a.net
inst = net.inst

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Knoten / Kanten", f"{inst.n} / {len(net.edges)}", delta=f"größter Ziel-Grad {int(inst.deg.max())}", delta_color="off")
m2.metric("κ = ⟨k²⟩ / ⟨k⟩", num(a.kappa, 1), delta="Riesenkomponente erwartet" if a.prediction else "keine Riesenkomponente erwartet", delta_color="off")
m3.metric("Größte Komponente", f"{num(a.giant * 100, 1)} %", delta=f"Sampler: {C.SAMPLERS.get(net.sampler_used.split(' ')[0], net.sampler_used)}", delta_color="off")
if net.sampler_used.startswith("stub"):
    m4.metric("Schleifen / Mehrfachkanten", f"{net.loops} / {net.multi}", delta=f"{num(a.lost_share * 100, 1)} % der Halbkanten verloren", delta_color="off")
else:
    m4.metric("Hubs erreichen", f"{num(a.hub_ratio * 100, 0)} %", delta="ihres Ziel-Grades (Mittel der größten zehn)", delta_color="off")
c1, c2 = st.columns(2)
c1.plotly_chart(build_degrees(inst.deg, a.realised), width="stretch", key="main_degrees")
c2.plotly_chart(build_hubs(inst.deg, a.realised), width="stretch", key="main_hubs")
st.caption(
    "Links die Gradverteilung (Ziel grau, gebautes Netz grün, beide Achsen logarithmisch); rechts der Ist-Grad der größten Hubs gegen ihren Ziel-Grad: "
    "Punkte unter der gestrichelten Linie haben ihren Grad nicht erreicht."
)

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Sagt κ die Riesenkomponente voraus, und trifft der Sampler die Grade?")
kind, text = verdict_network(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Messreihen: Molloy–Reed, Schleifen, Chung–Lu und Mischzeit")
st.caption(
    f"Vier feste Programme, unabhängig von den Reglern: zwei Grade mit {C.MR_N} Knoten, Potenzgesetze mit {C.DEFECT_N} Knoten, Chung–Lu mit {C.CHUNGLU_N} Knoten und ein "
    f"Watts–Strogatz-Netz mit {C.MIX_N} Knoten für die Mischzeit; Mediane über fünf feste Zufalls-Seeds. Das Rechnen dauert einige Sekunden."
)
if st.button("Messreihen rechnen", key="series_button"):
    t0 = time.perf_counter()
    with st.spinner("Rechne die Messreihen (einige Sekunden) …"):
        _series()
    st.session_state["series_done"] = time.perf_counter() - t0
if "series_done" in st.session_state:
    mr, defects, chung, mixing = _series()
    tab1, tab2, tab3, tab4 = st.tabs(["Molloy–Reed", "Schleifen und Mehrfachkanten", "Chung–Lu gegen Kantentausch", "Mischzeit des Kantentauschs"])
    with tab1:
        st.markdown("**Zwei Grade, 1 und 3:** Je größer der Anteil der Knoten mit Grad 3, desto größer κ. Bei κ = 2 beginnt die Riesenkomponente.")
        st.plotly_chart(build_molloy(mr), width="stretch", key="molloy_chart")
        kind, text = verdict_molloy(mr)
        getattr(st, kind)(text)
    with tab2:
        st.markdown("**Stub-Matching bei Potenzgesetz-Graden:** Je flacher das Potenzgesetz und je größer der größte Grad, desto mehr Halbkanten landen bei Schleifen und Mehrfachkanten.")
        st.plotly_chart(build_defects(defects), width="stretch", key="defects_chart")
        kind, text = verdict_defects(defects)
        getattr(st, kind)(text)
    with tab3:
        st.markdown("**Chung–Lu** zieht jede Kante mit Wahrscheinlichkeit min(1, w_i·w_j / Σw): bei zwei großen Hubs liegt sie über 1 und wird gekappt.")
        st.plotly_chart(build_chung_lu(chung), width="stretch", key="chung_chart")
        kind, text = verdict_chung_lu(chung)
        getattr(st, kind)(text)
        st.caption(f"Potenzgesetz γ = {num(C.CHUNGLU_GAMMA, 1)}, kleinster Grad {C.CHUNGLU_KMIN}, {C.CHUNGLU_HUBS} größte Hubs; der Kantentausch läuft nur bei graphischer Gradfolge "
                   f"({chung[0]['graphical']} von {chung[0]['plans']} Plänen ohne Obergrenze).")
    with tab4:
        st.markdown("**Mischzeit:** Der Kantentausch startet hier bei einem Netz mit sehr hoher Transitivität und vergisst dessen Transitivität nach einigen Vorschlägen je Kante (gemessen an diesem einen Startnetz).")
        st.plotly_chart(build_mixing(mixing), width="stretch", key="mixing_chart")
        kind, text = verdict_mixing(mixing)
        getattr(st, kind)(text)
        st.caption(f"Träge Kette: jeder Vorschlag zählt, auch abgelehnte. Wer nur gelungene Tausche zählt, bekommt eine verzerrte Verteilung (bei sieben Knoten messbar, Test `test_counting_only_accepted_switches_gives_a_measurably_non_uniform_chain`).")
    st.caption(f"Rechenzeit dieser Sitzung: {num(st.session_state['series_done'], 1)} s (beim ersten Mal; danach liegt das Ergebnis im Zwischenspeicher).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die Grade zählen** | Das Modell kennt keine Gemeinschaften, keine Dreiecke und keinen Raum: alles Weitere ist Zufall unter den Graden. | Netze im Raum, Motive (Folgestücke) |
| **Die Gradfolge ist gegeben** | Woher die Grade kommen, erklärt das Modell nicht. | Wachstumsmodelle (Folgestück) |
| **Das Kriterium κ > 2 gilt für n → ∞** | Bei endlicher Größe ist die Schwelle verschmiert; die Messreihe zeigt, wie stark. | diese Demo |
| **Die Folge ist graphisch** | Gibt es keinen einfachen Graphen mit genau diesen Graden (zum Beispiel zu viele Hubs mit Grad nahe n), fällt der Kantentausch auf das Stub-Matching zurück. | diese Demo |
| **Gleichverteilung** | Der Kantentausch ist nur als träge Kette gleichverteilt; wer nur gelungene Tausche zählt, verzerrt leicht. | diese Demo |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Gradfolge $d_1, \dots, d_n$ mit gerader Summe $2m$; gesucht ist ein zufälliges Netz, in dem Knoten $i$ den Grad $d_i$ hat.

| Symbol | Bedeutung |
|---|---|
| $d_i$ | Grad (Zahl der Halbkanten) von Knoten $i$ |
| $m$ | Zahl der Kanten, $2m = \sum_i d_i$ |
| $\kappa$ | $\langle k^2 \rangle / \langle k \rangle$, das Verhältnis aus mittlerem Quadrat und Mittelwert der Grade |
| $w_i$ | erwarteter Grad bei Chung–Lu |

**Stub-Matching.** Alle $2m$ Halbkanten werden gleichverteilt paarweise verbunden; es gibt $1 \cdot 3 \cdot 5 \cdots (2m-1)$ Paarungen. Jeder einfache Graph mit den gegebenen Graden entsteht aus genau $\prod_i d_i!$ Paarungen, also ist das Ergebnis **bedingt darauf, einfach zu sein, gleichverteilt** über alle einfachen Graphen mit diesen Graden.

**Molloy–Reed.** Eine Riesenkomponente entsteht (für $n \to \infty$ und unter schwachen Bedingungen an die Folge) genau dann, wenn

$$\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle} > 2.$$

**Chung–Lu.** Jede Kante $\{i, j\}$ kommt unabhängig mit Wahrscheinlichkeit

$$p_{ij} = \min\!\left(1, \frac{w_i w_j}{\sum_k w_k}\right)$$

vor; wo $w_i w_j > \sum_k w_k$, wird gekappt, und die erwarteten Grade werden unterschritten.

**Strukturelle Obergrenze.** Bei $k_{\max} \approx \sqrt{n \langle k \rangle}$ ist $w_i w_j / \sum_k w_k$ im Erwartungswert höchstens 1, das Kappen entfällt weitgehend. Die tatsächliche Gradsumme einer Ziehung schwankt, deshalb kappt Chung–Lu auch dann gelegentlich ein Hub-Paar.

**Kantentausch.** Aus zwei Kanten $(a, b)$ und $(c, d)$ werden $(a, d)$ und $(c, b)$, sofern weder eine Schleife noch eine Mehrfachkante entsteht; abgelehnte Vorschläge zählen als Schritt der Kette. Die so erhaltene träge Kette hat die Gleichverteilung als Grenzverteilung.

**Literatur** (aus dem Gedächtnis, nicht im Original geprüft): Molloy und Reed, *A critical point for random graphs with a given degree sequence*, Random Structures & Algorithms 6 (1995); Chung und Lu, *Connected components in random graphs with given expected degree sequences*, Annals of Combinatorics 6 (2002).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
