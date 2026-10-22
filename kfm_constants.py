"""Konstanten: Regler-Grenzen, Rechengrenzen, feste Pläne der Messreihen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 500, 10000, 3000          # Knoten; Chung–Lu ist O(n²): 10 000 Knoten rechnen einige Sekunden, Stub-Matching und Kantentausch fast linear in den Kanten
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 2.0, 4.5, 2.5  # Exponent des Potenzgesetzes der Grade
KMIN_MIN, KMIN_MAX, DEFAULT_KMIN = 1, 5, 2          # kleinster Grad
SEED_MAX, DEFAULT_SEED = 999999, 35

CUTOFFS = {"structural": "strukturell (√(n·⟨k⟩))", "free": "frei (n − 1)"}
DEFAULT_CUTOFF = "structural"
SAMPLERS = {"stub": "Stub-Matching (vereinfacht)", "swap": "Kantentausch (einfacher Graph)", "chunglu": "Chung–Lu (weiche Grade)"}
DEFAULT_SAMPLER = "stub"

SWITCH_STEPS_PER_EDGE = 20                          # Vorschläge des trägen Kantentauschs je Kante in der Hauptansicht (abgelehnte zählen mit)
SWEEP_SEEDS = tuple(range(100000, 100005))          # feste Pläne für alle Messreihen: nie ein einzelner Seed als Befund

MR_N = 20000                                        # Molloy–Reed-Reihe: Knoten
MR_SHARES = (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4)   # Anteil der Knoten mit Grad 1 (sonst Grad 3): κ = (9 − 8a) / (3 − 2a)
DEFECT_N = 20000                                    # Reihe „Schleifen und Mehrfachkanten“: Knoten
DEFECT_GAMMAS = (2.1, 2.3, 2.5, 2.75, 3.0, 3.5)
CHUNGLU_N, CHUNGLU_GAMMA, CHUNGLU_KMIN, CHUNGLU_HUBS = 3000, 2.2, 1, 10   # Reihe „Chung–Lu“: Knoten, Exponent, kleinster Grad, Zahl der betrachteten Hubs
MIX_N, MIX_K, MIX_P = 1000, 10, 0.01                # Reihe „Mischzeit“: Watts–Strogatz-Start (Knoten, Nachbarn, Umverdrahtung)
MIX_STEPS = (0.25, 0.5, 1, 2, 4, 8, 16)             # Vorschläge je Kante, an denen gemessen wird
MIX_REFERENCE = 40                                  # Vorschläge je Kante für den Plateauwert (Mittel über die Pläne)
MIX_SEEDS = SWEEP_SEEDS[:4]

COLOR_TARGET = "rgba(120,120,120,0.55)"
COLOR_REALISED = "#2F6B65"
COLOR_THEORY = "#4c78a8"
COLOR_THIRD = "#e8a33d"
COLOR_ALERT = "#d62728"

PRESETS = {
    "Standardfall (γ = 2,5, strukturelle Obergrenze)": {"n": DEFAULT_N, "gamma": DEFAULT_GAMMA, "kmin": DEFAULT_KMIN, "cutoff": "structural", "sampler": "stub", "seed": DEFAULT_SEED},
    "Hubs ohne Obergrenze (γ = 2,2)": {"n": DEFAULT_N, "gamma": 2.2, "kmin": 1, "cutoff": "free", "sampler": "stub", "seed": DEFAULT_SEED},
    "Kantentausch statt Stub-Matching": {"n": DEFAULT_N, "gamma": 2.2, "kmin": 1, "cutoff": "structural", "sampler": "swap", "seed": DEFAULT_SEED},
    "Chung–Lu ohne Obergrenze": {"n": DEFAULT_N, "gamma": 2.2, "kmin": 1, "cutoff": "free", "sampler": "chunglu", "seed": DEFAULT_SEED},
    "Keine Riesenkomponente (γ = 4)": {"n": DEFAULT_N, "gamma": 4.0, "kmin": 1, "cutoff": "structural", "sampler": "stub", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (γ = 2,5, strukturelle Obergrenze)": "3 000 Knoten mit Potenzgesetz-Graden ab 2, Stub-Matching ohne Schleifen und Mehrfachkanten: κ liegt weit über 2, eine Riesenkomponente ist sicher, und nur wenige Kanten gehen beim Vereinfachen verloren.",
    "Hubs ohne Obergrenze (γ = 2,2)": "Grade bis n − 1 erlaubt: die Hubs brauchen mehr Kanten zwischen sich, als es verschiedene Paare gibt, und das Stub-Matching erzeugt viele Schleifen und Mehrfachkanten, die beim Vereinfachen die Grade der Hubs senken.",
    "Kantentausch statt Stub-Matching": "Dieselben Grade mit strukturellem Abschneiden, aber ein einfacher Graph mit genau den Graden (Havel–Hakimi, dann Kantentausch): jeder Grad wird getroffen.",
    "Chung–Lu ohne Obergrenze": "Grade nur im Erwartungswert: wo w_i · w_j / Σw über 1 liegt, kappt Chung–Lu die Wahrscheinlichkeit, und die Hubs bleiben unter ihrem erwarteten Grad.",
    "Keine Riesenkomponente (γ = 4)": "Steiles Potenzgesetz ab Grad 1: κ liegt unter 2, es bleiben nur kleine Komponenten.",
}
