"""Kern: Netze zu gegebenen Gradfolgen (Konfigurationsmodell). Gradfolgen, Graphikalität, drei Sampler (Stub-Matching, Kantentausch, Chung–Lu),
Molloy–Reed-Kriterium und Kennzahlen. Jede Regel ist eine eigene kleine Funktion (DEMO-PLAYBOOK §1b); Zufall nur über einen übergebenen Generator."""

import heapq
import math

import numpy as np


# --- Gradfolgen ----------------------------------------------------------------------------------------------------------------------------


def fix_parity(deg: np.ndarray) -> np.ndarray:
    """Die Summe der Grade muss gerade sein: bei ungerader Summe wird der Grad des ersten Knotens um eins erhöht."""
    deg = np.array(deg, dtype=np.int64)
    if len(deg) and deg.sum() % 2:
        deg[0] += 1
    return deg


def two_point_degrees(n: int, share_low: float, rng: np.random.Generator) -> np.ndarray:
    """Grad 1 mit Wahrscheinlichkeit `share_low`, sonst Grad 3 (Summe gerade gemacht)."""
    return fix_parity(np.where(rng.random(n) < share_low, 1, 3))


def power_law_degrees(n: int, gamma: float, kmin: int, kmax: int, rng: np.random.Generator) -> np.ndarray:
    """Grade aus dem abgeschnittenen Potenzgesetz P(k) ~ k^(−gamma) auf kmin..kmax (Summe gerade gemacht)."""
    kmax = max(kmin, int(kmax))
    ks = np.arange(kmin, kmax + 1)
    p = ks.astype(float) ** (-gamma)
    p /= p.sum()
    return fix_parity(rng.choice(ks, size=n, p=p))


def structural_cutoff(n: int, mean_degree: float) -> int:
    """Strukturelle Obergrenze für Grade, bis zu der Kanten zwischen zwei Hubs nicht gekappt werden müssen: √(n · ⟨k⟩), mindestens 2."""
    return max(2, int(math.sqrt(max(0.0, n * mean_degree))))


def mean_degree(deg: np.ndarray) -> float:
    return float(np.mean(deg)) if len(deg) else 0.0


def kappa(deg: np.ndarray) -> float:
    """Molloy–Reed-Größe ⟨k²⟩ / ⟨k⟩ (0 ohne Kanten): eine Riesenkomponente entsteht, wenn sie über 2 liegt."""
    s = float(np.sum(deg))
    return 0.0 if s == 0 else float(np.sum(np.asarray(deg, dtype=float) ** 2)) / s


def giant_by_molloy_reed(deg: np.ndarray) -> bool:
    """Vorhersage des Kriteriums: Riesenkomponente genau dann, wenn κ > 2."""
    return kappa(deg) > 2.0


def is_graphical(deg) -> bool:
    """Erdős–Gallai: gibt es einen einfachen Graphen (keine Schleifen, keine Mehrfachkanten) mit genau diesen Graden?"""
    d = np.sort(np.asarray(deg, dtype=np.int64))[::-1]
    n = len(d)
    if n == 0:
        return True
    if d.min() < 0 or d.sum() % 2:
        return False
    prefix = np.concatenate([[0], np.cumsum(d)])
    asc = d[::-1]
    for k in range(1, n + 1):
        t = n - int(np.searchsorted(asc, k, side="left"))  # Zahl der Grade >= k (die ersten t der absteigenden Folge)
        tail = max(0, t - k) * k + (prefix[n] - prefix[max(t, k)])
        if prefix[k] > k * (k - 1) + tail:
            return False
    return True


def havel_hakimi(deg) -> list[tuple[int, int]] | None:
    """Ein einfacher Graph (Kantenliste, u < v) mit genau diesen Graden nach Havel–Hakimi, None wenn es keinen gibt."""
    deg = [int(x) for x in deg]
    if sum(deg) % 2 or any(x < 0 for x in deg):
        return None
    heap = [(-d, v) for v, d in enumerate(deg) if d > 0]
    heapq.heapify(heap)
    edges = []
    while heap:
        d, v = heapq.heappop(heap)
        d = -d
        if d > len(heap):
            return None
        taken = [heapq.heappop(heap) for _ in range(d)]
        for dd, u in taken:
            edges.append((min(u, v), max(u, v)))
            if -(dd + 1) > 0:
                heapq.heappush(heap, (dd + 1, u))
    return edges


# --- Sampler -------------------------------------------------------------------------------------------------------------------------------


def stub_matching(deg, rng: np.random.Generator) -> np.ndarray:
    """Konfigurationsmodell: jeder Knoten bekommt so viele Halbkanten wie sein Grad, alle werden gemischt und paarweise verbunden. Ergebnis (m, 2) mit möglichen Schleifen und Mehrfachkanten."""
    stubs = np.repeat(np.arange(len(deg)), np.asarray(deg, dtype=np.int64))
    rng.shuffle(stubs)
    return stubs[: (len(stubs) // 2) * 2].reshape(-1, 2)


def defect_counts(edges: np.ndarray) -> tuple[int, int]:
    """(Schleifen, überzählige Mehrfachkanten) einer Kantenliste: eine Kante, die k-mal vorkommt, zählt k − 1 Mehrfachkanten; Schleifen zählen getrennt."""
    if len(edges) == 0:
        return 0, 0
    e = np.sort(np.asarray(edges), axis=1)
    loops = int(np.sum(e[:, 0] == e[:, 1]))
    plain = e[e[:, 0] != e[:, 1]]
    multi = int(len(plain) - len(np.unique(plain, axis=0))) if len(plain) else 0
    return loops, multi


def simplify(edges: np.ndarray) -> np.ndarray:
    """Schleifen und Mehrfachkanten entfernen: (k, 2)-Feld verschiedener Kanten mit u < v (sortiert)."""
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    e = np.sort(np.asarray(edges), axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0) if len(e) else np.zeros((0, 2), dtype=np.int64)


def degrees_of(n: int, edges: np.ndarray) -> np.ndarray:
    """Grade aus einer Kantenliste (eine Schleife zählt zweimal)."""
    if len(edges) == 0:
        return np.zeros(n, dtype=np.int64)
    return np.bincount(np.asarray(edges).ravel(), minlength=n).astype(np.int64)


def try_switch(E: list, S: set, i: int, j: int, flipped: bool) -> bool:
    """Ein Vorschlag des Kantentauschs: Kanten i = (a, b) und j = (c, d) werden zu (a, d) und (c, b) (bei `flipped` zu (a, c) und (d, b)); angenommen, wenn weder Schleife noch Mehrfachkante entsteht.
    Ändert E und S nur bei Annahme. Rückgabe: ob der Vorschlag angenommen wurde."""
    if i == j:
        return False
    a, b = E[i]
    c, d = E[j]
    if flipped:
        c, d = d, c
    if a == d or c == b:
        return False
    e1 = (a, d) if a < d else (d, a)
    e2 = (c, b) if c < b else (b, c)
    if e1 == e2 or e1 in S or e2 in S:
        return False
    S.discard(E[i])
    S.discard(E[j])
    S.add(e1)
    S.add(e2)
    E[i], E[j] = e1, e2
    return True


def edge_switch_steps(edges, n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Träger Kantentausch: genau `n_steps` Vorschläge, abgelehnte zählen mit. Diese Kette hat die Gleichverteilung über alle einfachen Graphen mit den gegebenen Graden als Grenzverteilung.
    Rückgabe: (Kanten, Zahl angenommener Tausche)."""
    E = [(int(u), int(v)) for u, v in edges]
    S = set(E)
    m = len(E)
    if m < 2 or n_steps <= 0:
        return np.array(E, dtype=np.int64).reshape(-1, 2), 0
    ri = rng.integers(0, m, size=(n_steps, 2))
    flip = rng.random(n_steps) < 0.5
    accepted = 0
    for t in range(n_steps):
        accepted += try_switch(E, S, int(ri[t, 0]), int(ri[t, 1]), bool(flip[t]))
    return np.array(E, dtype=np.int64).reshape(-1, 2), accepted


def edge_switches(edges, n_switches: int, rng: np.random.Generator, max_tries_factor: int = 200) -> tuple[np.ndarray, int, int]:
    """Kantentausch bis `n_switches` Tausche gelungen sind (oder `max_tries_factor` · n_switches Versuche verbraucht sind): so zählt die gängige Praxis (auch `networkx`).
    Rückgabe: (Kanten, gelungene Tausche, Versuche). Achtung: Die Kette der gelungenen Tausche allein ist nicht gleichverteilt (Zustände mit vielen möglichen Tauschen werden bevorzugt); für exakte Gleichverteilung `edge_switch_steps`."""
    E = [(int(u), int(v)) for u, v in edges]
    S = set(E)
    m = len(E)
    done = tries = 0
    if m < 2 or n_switches <= 0:
        return np.array(E, dtype=np.int64).reshape(-1, 2), 0, 0
    budget = max_tries_factor * n_switches
    block = 4096
    while done < n_switches and tries < budget:
        ri = rng.integers(0, m, size=(block, 2))
        flip = rng.random(block) < 0.5
        for t in range(block):
            if done >= n_switches or tries >= budget:
                break
            tries += 1
            done += try_switch(E, S, int(ri[t, 0]), int(ri[t, 1]), bool(flip[t]))
    return np.array(E, dtype=np.int64).reshape(-1, 2), done, tries


def chung_lu_probability(wi: float, wj: float, total: float) -> float:
    """Wahrscheinlichkeit einer Kante zwischen zwei Knoten mit erwarteten Graden wi und wj: min(1, wi·wj / Σw)."""
    return 0.0 if total <= 0 else min(1.0, wi * wj / total)


def chung_lu_edges(w, rng: np.random.Generator) -> np.ndarray:
    """Chung–Lu: jede Kante {i, j} unabhängig mit Wahrscheinlichkeit min(1, w_i w_j / Σw). Zeilenweise vektorisiert, O(n²)."""
    w = np.asarray(w, dtype=float)
    n, total = len(w), float(w.sum())
    if total <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    out = []
    for i in range(n - 1):
        p = np.minimum(1.0, w[i] * w[i + 1:] / total)
        hit = np.flatnonzero(rng.random(n - 1 - i) < p)
        if len(hit):
            out.append(np.stack([np.full(len(hit), i), hit + i + 1], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def capped_pairs(w, subset=None) -> int:
    """Zahl der Knotenpaare (unter den Knoten `subset`, sonst allen), bei denen w_i w_j / Σw über 1 liegt: dort kappt Chung–Lu die Wahrscheinlichkeit."""
    w = np.asarray(w, dtype=float)
    idx = np.arange(len(w)) if subset is None else np.asarray(subset)
    sub = w[idx]
    big = np.outer(sub, sub) / w.sum() > 1.0
    return int(np.sum(np.triu(big, k=1)))


# --- Komponenten und Kennzahlen -----------------------------------------------------------------------------------------------------------


def find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def component_sizes(n: int, edges) -> list[int]:
    """Größen aller Komponenten, absteigend (Union-Find mit Pfadhalbierung und Vereinigung nach Größe)."""
    parent = list(range(n))
    size = [1] * n
    for u, v in edges:
        ra, rb = find(parent, int(u)), find(parent, int(v))
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    counts: dict[int, int] = {}
    for x in range(n):
        r = find(parent, x)
        counts[r] = counts.get(r, 0) + 1
    return sorted(counts.values(), reverse=True)


def giant_fraction(n: int, edges) -> float:
    """Anteil der Knoten in der größten Komponente."""
    return component_sizes(n, edges)[0] / n if n > 0 else 0.0


def adjacency_sets(n: int, edges) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        u, v = int(u), int(v)
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def transitivity(n: int, edges) -> float:
    """Anteil geschlossener Dreier unter allen Dreiern: 3 · Dreiecke / Zahl der Knotenpaare an einem gemeinsamen Nachbarn (0 ohne solche Paare)."""
    adj = adjacency_sets(n, edges)
    triads = sum(len(a) * (len(a) - 1) // 2 for a in adj)
    if triads == 0:
        return 0.0
    triangles = sum(len(adj[u] & adj[v]) for u in range(n) for v in adj[u] if v > u) / 3.0
    return 3.0 * triangles / triads


def assortativity(n: int, edges) -> float:
    """Pearson-Korrelation der Grade an den beiden Enden einer Kante (0 bei konstantem Grad)."""
    if len(edges) == 0:
        return 0.0
    e = np.asarray(edges)
    deg = degrees_of(n, e)
    x = np.concatenate([deg[e[:, 0]], deg[e[:, 1]]]).astype(float)
    y = np.concatenate([deg[e[:, 1]], deg[e[:, 0]]]).astype(float)
    if x.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def hub_ratio(target: np.ndarray, realised: np.ndarray, top: int) -> float:
    """Mittleres Verhältnis realisierter zu Ziel-Grad der `top` Knoten mit den größten Ziel-Graden (1 = alle Grade getroffen)."""
    hubs = np.argsort(-np.asarray(target), kind="stable")[:top]
    t = np.asarray(target, dtype=float)[hubs]
    return float(np.mean(np.asarray(realised, dtype=float)[hubs] / t)) if len(hubs) and np.all(t > 0) else 0.0


def edges_between(edges: np.ndarray, subset) -> int:
    """Zahl der Kanten (ohne Schleifen), bei denen beide Enden in `subset` liegen (Mehrfachkanten zählen mehrfach)."""
    s = set(int(x) for x in subset)
    return sum(1 for u, v in edges if u != v and int(u) in s and int(v) in s)


def ring_lattice_rewired(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Watts–Strogatz-Netz als Start mit hoher Transitivität: Ringgitter mit k Nachbarn je Knoten, jede Kante mit Wahrscheinlichkeit p auf ein zufälliges Ende umverdrahtet (ohne Schleifen und Doppelkanten)."""
    edges = set()
    for u in range(n):
        for j in range(1, k // 2 + 1):
            edges.add((min(u, (u + j) % n), max(u, (u + j) % n)))
    out = set(edges)
    for (u, v) in sorted(edges):
        if rng.random() < p:
            for _ in range(20):
                w = int(rng.integers(0, n))
                new = (min(u, w), max(u, w))
                if w != u and new not in out:
                    out.discard((u, v))
                    out.add(new)
                    break
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)
