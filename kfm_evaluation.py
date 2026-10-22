"""Sampler-Aufruf, Kennzahlen, Messreihen und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text)."""

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import kfm_algorithm as K
import kfm_constants as C
from kfm_scenario import Instance, make_instance, make_two_point


def num(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    gamma: float = C.DEFAULT_GAMMA
    kmin: int = C.DEFAULT_KMIN
    cutoff: str = C.DEFAULT_CUTOFF
    sampler: str = C.DEFAULT_SAMPLER
    seed: int = C.DEFAULT_SEED


@dataclass(eq=False)
class Network:
    inst: Instance
    edges: np.ndarray        # Kanten nach dem Sampler (beim Stub-Matching nach dem Vereinfachen)
    raw_edges: np.ndarray    # beim Stub-Matching die Kanten vor dem Vereinfachen, sonst gleich `edges`
    loops: int
    multi: int
    sampler_used: str        # „stub“, „swap“, „chunglu“ oder „stub (Rückfall)“ bei nicht graphischer Folge für den Kantentausch
    graphical: bool
    accepted: int            # beim Kantentausch: angenommene Tausche


@dataclass(eq=False)
class Analysis:
    net: Network
    kappa: float
    prediction: bool         # Molloy–Reed: κ > 2
    giant: float
    realised: np.ndarray     # Grade des gebauten Netzes
    hub_ratio: float         # mittleres Verhältnis realisiert / Ziel der größten Hubs
    hubhub: int              # Kanten zwischen den größten Hubs
    lost_share: float        # Anteil der Ziel-Halbkanten, der verloren ging (1 − Σ realisiert / Σ Ziel)


def build_network(inst: Instance, sampler: str, seed: int) -> Network:
    """Baut ein Netz zu den Zielgraden mit dem gewählten Sampler (Zufalls-Seed getrennt von dem der Gradfolge)."""
    rng = np.random.default_rng(seed + 7919)
    deg = inst.deg
    graphical = K.is_graphical(deg)
    if sampler == "chunglu":
        e = K.chung_lu_edges(deg.astype(float), rng)
        return Network(inst, e, e, 0, 0, "chunglu", graphical, 0)
    if sampler == "swap" and graphical:
        start = np.array(K.havel_hakimi(deg.tolist()), dtype=np.int64).reshape(-1, 2)
        e, accepted = K.edge_switch_steps(start, C.SWITCH_STEPS_PER_EDGE * len(start), rng)
        return Network(inst, e, e, 0, 0, "swap", True, accepted)
    raw = K.stub_matching(deg, rng)
    loops, multi = K.defect_counts(raw)
    used = "stub" if sampler != "swap" else "stub (Rückfall)"
    return Network(inst, K.simplify(raw), raw, loops, multi, used, graphical, 0)


def analyse_network(net: Network, hubs: int = C.CHUNGLU_HUBS) -> Analysis:
    inst = net.inst
    realised = K.degrees_of(inst.n, net.edges)
    top = np.argsort(-inst.deg, kind="stable")[:hubs]
    total = float(inst.deg.sum())
    return Analysis(
        net=net, kappa=K.kappa(inst.deg), prediction=K.giant_by_molloy_reed(inst.deg), giant=K.giant_fraction(inst.n, net.edges),
        realised=realised, hub_ratio=K.hub_ratio(inst.deg, realised, hubs), hubhub=K.edges_between(net.edges, top),
        lost_share=0.0 if total == 0 else 1.0 - float(realised.sum()) / total,
    )


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.n, settings.gamma, settings.kmin, settings.cutoff, settings.seed)
    return analyse_network(build_network(inst, settings.sampler, settings.seed))


# --- Messreihen ----------------------------------------------------------------------------------------------------------------------------


def sweep_molloy_reed(n: int = C.MR_N, shares=C.MR_SHARES, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Zwei Grade (1 und 3): Riesenkomponente über dem Anteil der Grad-1-Knoten, neben κ = ⟨k²⟩/⟨k⟩ und der Vorhersage κ > 2 (Median über die festen Pläne)."""
    rows = []
    for a in shares:
        giants, kappas = [], []
        for s in seeds:
            deg = make_two_point(n, a, s)
            kappas.append(K.kappa(deg))
            giants.append(K.giant_fraction(n, K.simplify(K.stub_matching(deg, np.random.default_rng(s + 7919)))))
        k = median(kappas)
        rows.append({"share": a, "kappa": k, "giant": median(giants), "low": min(giants), "high": max(giants), "predicted": k > 2.0})
    return rows


def sweep_defects(gammas=C.DEFECT_GAMMAS, kmin: int = 1, n: int = C.DEFECT_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Anteil der Kanten, die beim Stub-Matching Schleifen oder überzählige Mehrfachkanten sind, über dem Exponenten, ohne und mit struktureller Obergrenze (Median über die Pläne)."""
    rows = []
    for g in gammas:
        row = {"gamma": g}
        for cutoff in ("free", "structural"):
            shares, kmaxs = [], []
            for s in seeds:
                inst = make_instance(n, g, kmin, cutoff, s)
                loops, multi = K.defect_counts(K.stub_matching(inst.deg, np.random.default_rng(s + 7919)))
                shares.append((loops + multi) / (inst.deg.sum() / 2))
                kmaxs.append(inst.kmax)
            row[cutoff] = float(median(shares))
            row[f"{cutoff}_kmax"] = float(median(kmaxs))
        rows.append(row)
    return rows


def series_chung_lu(n: int = C.CHUNGLU_N, gamma: float = C.CHUNGLU_GAMMA, kmin: int = C.CHUNGLU_KMIN, hubs: int = C.CHUNGLU_HUBS, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Chung–Lu gegen den Kantentausch (einfacher Graph mit genau den Graden), ohne und mit struktureller Obergrenze: Verhältnis realisiert/Ziel der größten Hubs, Kanten zwischen ihnen
    und Zahl der gekappten Hub-Paare (Median über die Pläne; ohne graphische Folge entfällt der Kantentausch für diesen Plan)."""
    rows = []
    for cutoff in ("free", "structural"):
        cl_ratio, cl_hh, sw_ratio, sw_hh, capped, graphical = [], [], [], [], [], 0
        for s in seeds:
            inst = make_instance(n, gamma, kmin, cutoff, s)
            top = np.argsort(-inst.deg, kind="stable")[:hubs]
            capped.append(K.capped_pairs(inst.deg.astype(float), subset=top))
            a_cl = analyse_network(build_network(inst, "chunglu", s), hubs)
            cl_ratio.append(a_cl.hub_ratio)
            cl_hh.append(a_cl.hubhub)
            if K.is_graphical(inst.deg):
                graphical += 1
                a_sw = analyse_network(build_network(inst, "swap", s), hubs)
                sw_ratio.append(a_sw.hub_ratio)
                sw_hh.append(a_sw.hubhub)
        rows.append({
            "cutoff": cutoff, "kmax": make_instance(n, gamma, kmin, cutoff, seeds[0]).kmax, "capped": median(capped),
            "chunglu_ratio": median(cl_ratio), "chunglu_hubhub": median(cl_hh),
            "swap_ratio": median(sw_ratio) if sw_ratio else None, "swap_hubhub": median(sw_hh) if sw_hh else None, "graphical": graphical, "plans": len(seeds),
        })
    return rows


def series_mixing(n: int = C.MIX_N, k: int = C.MIX_K, p: float = C.MIX_P, steps=C.MIX_STEPS, reference: int = C.MIX_REFERENCE, seeds=C.MIX_SEEDS) -> dict:
    """Mischzeit des trägen Kantentauschs: Transitivität und Assortativität nach `steps` Vorschlägen je Kante, gemessen an einer Kette je Plan, neben dem Plateau nach `reference` Vorschlägen je Kante.
    Startnetz: Watts–Strogatz mit hoher Transitivität. Abstand zum Plateau in Standardabweichungen der Pläne am Plateau."""
    traj = {s: [] for s in seeds}
    start_t = None
    plateau_t, plateau_a = [], []
    grid = sorted(set(steps) | {reference})
    for s in seeds:
        rng = np.random.default_rng(s)
        edges = K.ring_lattice_rewired(n, k, p, rng)
        m = len(edges)
        if start_t is None:
            start_t = K.transitivity(n, edges)
        done = 0
        for st in grid:
            target = int(st * m)
            edges, acc = K.edge_switch_steps(edges, target - done, rng)
            done = target
            tr, asr = K.transitivity(n, edges), K.assortativity(n, edges)
            traj[s].append((st, tr, asr, acc))
        plateau_t.append(traj[s][-1][1])
        plateau_a.append(traj[s][-1][2])
    plat, sd = float(np.mean(plateau_t)), float(np.std(plateau_t))
    rows = []
    for idx, st in enumerate(grid):
        if st == reference:
            continue
        ts = [traj[s][idx][1] for s in seeds]
        rows.append({"steps": st, "transitivity": float(np.mean(ts)), "distance": abs(float(np.mean(ts)) - plat) / sd if sd > 0 else float("inf")})
    return {"rows": rows, "start": start_t, "plateau": plat, "plateau_sd": sd, "reference": reference, "m": m, "plans": len(seeds)}


# --- Verdikte --------------------------------------------------------------------------------------------------------------------------------


def verdict_network(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: sagt Molloy–Reed die Riesenkomponente richtig voraus, und wie genau trifft der Sampler die Grade?"""
    n = a.net.inst.n
    ok = a.prediction == (a.giant > 0.1)
    pred = "eine Riesenkomponente" if a.prediction else "keine Riesenkomponente"
    base = (f"κ = ⟨k²⟩/⟨k⟩ = {num(a.kappa, 1)}: das Kriterium von Molloy und Reed sagt {pred} voraus; gemessen hat die größte Komponente {num(a.giant * 100, 1)} % der {n} Knoten")
    if a.net.sampler_used == "stub (Rückfall)":
        return "warning", base + ". Die Gradfolge ist nicht graphisch (kein einfacher Graph hat genau diese Grade): statt des Kantentauschs läuft das vereinfachte Stub-Matching."
    if a.net.sampler_used == "swap":
        return "success" if ok else "warning", base + ". Der Kantentausch trifft jeden Grad genau (Verhältnis der Hubs 1,0)."
    if a.net.sampler_used == "chunglu":
        return "info", base + f". Chung–Lu trifft die Grade nur im Erwartungswert: die größten Hubs erreichen im Mittel {num(a.hub_ratio * 100, 0)} % ihres Ziel-Grades."
    return "info", base + (f". Das Stub-Matching erzeugt {a.net.loops} Schleifen und {a.net.multi} überzählige Mehrfachkanten; nach dem Vereinfachen fehlen {num(a.lost_share * 100, 1)} % der Halbkanten, "
                           f"die größten Hubs erreichen im Mittel {num(a.hub_ratio * 100, 0)} % ihres Ziel-Grades.")


def verdict_molloy(rows: list[dict]) -> tuple[str, str]:
    """Experiment Molloy–Reed: knapp unter und knapp über der Schwelle κ = 2 (n = 20 000); die erste Zeile mit κ über 2 und die letzte darunter."""
    above = [r for r in rows if r["kappa"] > 2.0]
    below = [r for r in rows if r["kappa"] <= 2.0]
    lo, hi = max(below, key=lambda r: r["kappa"]), min(above, key=lambda r: r["kappa"])
    return "info", (f"Bei κ = {num(lo['kappa'], 2)} (Anteil Grad 1: {num(lo['share'] * 100, 0)} %) hat die größte Komponente im Median {num(lo['giant'] * 100, 1)} % der Knoten, "
                    f"bei κ = {num(hi['kappa'], 2)} ({num(hi['share'] * 100, 0)} %) sind es {num(hi['giant'] * 100, 1)} %: die Schwelle liegt bei κ = 2, der Übergang ist bei endlicher Größe verschmiert.")


def verdict_defects(rows: list[dict]) -> tuple[str, str]:
    """Experiment Schleifen und Mehrfachkanten: steilstes und flachstes Potenzgesetz, ohne und mit Obergrenze."""
    flat, steep = rows[0], rows[-1]
    return "info", (f"Beim flachsten Potenzgesetz (γ = {num(flat['gamma'], 1)}) sind {num(flat['free'] * 100, 1)} % der Kanten Schleifen oder Mehrfachkanten, mit struktureller Obergrenze "
                    f"(größter Grad {num(flat['structural_kmax'], 0)} statt {num(flat['free_kmax'], 0)}) nur {num(flat['structural'] * 100, 1)} %; beim steilsten (γ = {num(steep['gamma'], 1)}) sind es "
                    f"{num(steep['free'] * 100, 1)} % und {num(steep['structural'] * 100, 1)} %.")


def verdict_chung_lu(rows: list[dict]) -> tuple[str, str]:
    """Experiment Chung–Lu: Hub-Verhältnis ohne und mit Obergrenze, neben dem Kantentausch (falls graphisch)."""
    free, structural = rows[0], rows[1]
    sw = "" if free["swap_ratio"] is None else f" (Kantentausch: {num(free['swap_ratio'] * 100, 0)} %)"
    hh = "" if free["swap_hubhub"] is None else f" Zwischen den größten Hubs liegen bei Chung–Lu im Median {num(free['chunglu_hubhub'], 0)} Kanten, beim Kantentausch {num(free['swap_hubhub'], 0)}."
    return "info", (f"Ohne Obergrenze erreichen die größten Hubs bei Chung–Lu im Median {num(free['chunglu_ratio'] * 100, 0)} % ihres erwarteten Grades{sw}; {num(free['capped'], 0)} von "
                    f"{C.CHUNGLU_HUBS * (C.CHUNGLU_HUBS - 1) // 2} Hub-Paaren werden gekappt.{hh} Mit struktureller Obergrenze (größter Grad {num(structural['kmax'], 0)}) sind es "
                    f"{num(structural['chunglu_ratio'] * 100, 0)} % bei {num(structural['capped'], 0)} gekappten Paaren.")


def verdict_mixing(res: dict) -> tuple[str, str]:
    """Experiment Mischzeit: ab wie vielen Vorschlägen je Kante liegt die Transitivität innerhalb von zwei Standardabweichungen am Plateau."""
    ok = [r["steps"] for r in res["rows"] if r["distance"] <= 2.0]
    first = min(ok) if ok else None
    start, plat = res["start"], res["plateau"]
    when = f"ab {num(first, 2).rstrip('0').rstrip(',')} Vorschlägen je Kante" if first is not None else "innerhalb der gemessenen Schritte nicht"
    return "info", (f"Das Startnetz hat Transitivität {num(start, 2)}, das Plateau liegt bei {num(plat, 3)}; die Kette ist {when} innerhalb von zwei Standardabweichungen "
                    f"({res['plans']} Pläne) am Plateau.")
