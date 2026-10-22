"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from collections import Counter

import numpy as np
import plotly.graph_objects as go

import kfm_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def degree_counts(deg) -> list[tuple[int, int]]:
    """(Grad, Zahl der Knoten mit diesem Grad) für Grade über null, aufsteigend."""
    return sorted((k, c) for k, c in Counter(int(x) for x in deg).items() if k > 0)


def build_degrees(target, realised):
    """Gradverteilung (beide Achsen logarithmisch): Ziel als graue Punkte, gebautes Netz als grüne Rauten."""
    t, r = degree_counts(target), degree_counts(realised)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[k for k, _ in t], y=[c for _, c in t], mode="markers", marker=dict(size=10, color=C.COLOR_TARGET), name="Ziel",
                             hovertemplate="Grad %{x}: %{y} Knoten<extra>Ziel</extra>"))
    fig.add_trace(go.Scatter(x=[k for k, _ in r], y=[c for _, c in r], mode="markers", marker=dict(size=7, color=C.COLOR_REALISED, symbol="diamond"), name="gebautes Netz",
                             hovertemplate="Grad %{x}: %{y} Knoten<extra>Netz</extra>"))
    fig.update_xaxes(title_text="Grad", type="log")
    fig.update_yaxes(title_text="Zahl der Knoten", type="log")
    return _base(fig)


def hub_points(target, realised, top: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Ziel- und Ist-Grad der `top` Knoten mit den größten Ziel-Graden (stabile Reihenfolge)."""
    order = np.argsort(-np.asarray(target), kind="stable")[:top]
    return np.asarray(target)[order], np.asarray(realised)[order]


def build_hubs(target, realised, top: int = 200):
    """Ist-Grad gegen Ziel-Grad der größten Hubs (log-log) mit der Winkelhalbierenden: Punkte unter der Linie haben ihren Grad nicht erreicht."""
    tx, ry = hub_points(target, realised, top)
    hi = max(2, int(max(tx.max(initial=1), ry.max(initial=1))))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, hi], y=[1, hi], mode="lines", line=dict(color=C.COLOR_THEORY, dash="dash", width=2), name="Ziel = Ist"))
    fig.add_trace(go.Scatter(x=tx, y=np.maximum(ry, 0.8), mode="markers", marker=dict(size=7, color=C.COLOR_REALISED), name=f"die {len(tx)} größten Hubs",
                             hovertemplate="Ziel %{x}: Ist %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Ziel-Grad", type="log")
    fig.update_yaxes(title_text="Grad im gebauten Netz", type="log")
    return _base(fig)


def build_molloy(rows):
    """Riesenkomponente über κ = ⟨k²⟩/⟨k⟩ (zwei Grade, 1 und 3) mit Band Minimum bis Maximum; die Schwelle κ = 2 ist markiert."""
    xs = [r["kappa"] for r in rows]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    rr = [rows[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["high"] for r in rr] + [r["low"] for r in rr][::-1], fill="toself", fillcolor="rgba(47,107,101,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=xs, y=[r["giant"] for r in rr], mode="lines+markers", line=dict(color=C.COLOR_REALISED, width=3), name="gemessen (Median)",
                             hovertemplate="κ = %{x:.2f}: %{y:.3f}<extra></extra>"))
    fig.add_vline(x=2.0, line=dict(color=C.COLOR_ALERT, width=1.5, dash="dash"))
    fig.update_xaxes(title_text="κ = ⟨k²⟩ / ⟨k⟩")
    fig.update_yaxes(title_text="Anteil der Riesenkomponente", range=[-0.02, 1.02])
    return _base(fig)


def build_defects(rows):
    """Anteil der Kanten, die Schleifen oder überzählige Mehrfachkanten sind, über dem Exponenten (Log-Achse), ohne und mit struktureller Obergrenze."""
    gs = [r["gamma"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gs, y=[max(r["free"], 1e-6) * 100 for r in rows], mode="lines+markers", line=dict(color=C.COLOR_ALERT, width=3), name="Grade bis n − 1",
                             hovertemplate="γ = %{x}: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=gs, y=[max(r["structural"], 1e-6) * 100 for r in rows], mode="lines+markers", line=dict(color=C.COLOR_REALISED, width=3), name="strukturelle Obergrenze",
                             hovertemplate="γ = %{x}: %{y:.2f} %<extra></extra>"))
    fig.update_xaxes(title_text="Exponent γ")
    fig.update_yaxes(title_text="Schleifen und Mehrfachkanten (% der Kanten)", type="log")
    return _base(fig)


def build_chung_lu(rows):
    """Verhältnis realisierter zu Ziel-Grad der größten Hubs: Chung–Lu und Kantentausch, ohne und mit struktureller Obergrenze."""
    labels = [C.CUTOFFS[r["cutoff"]].split(" (")[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["chunglu_ratio"] for r in rows], name="Chung–Lu", marker_color=C.COLOR_THIRD, hovertemplate="%{y:.2f}<extra>Chung–Lu</extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["swap_ratio"] if r["swap_ratio"] is not None else 0 for r in rows], name="Kantentausch", marker_color=C.COLOR_REALISED,
                         hovertemplate="%{y:.2f}<extra>Kantentausch</extra>"))
    fig.update_xaxes(title_text="Obergrenze der Grade")
    fig.update_yaxes(title_text="Ist-Grad / Ziel-Grad der größten Hubs", range=[0, 1.15])
    fig.update_layout(barmode="group")
    return _base(fig)


def build_mixing(res):
    """Transitivität über der Zahl der Vorschläge je Kante (Log-Achse), das Plateau als Linie mit Band von zwei Standardabweichungen."""
    xs = [r["steps"] for r in res["rows"]]
    ys = [r["transitivity"] for r in res["rows"]]
    plat, sd = res["plateau"], res["plateau_sd"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min(xs), max(xs), max(xs), min(xs)], y=[plat + 2 * sd, plat + 2 * sd, max(0.0, plat - 2 * sd), max(0.0, plat - 2 * sd)], fill="toself",
                             fillcolor="rgba(76,120,168,0.18)", line=dict(width=0), hoverinfo="skip", name="Plateau ± 2 Standardabweichungen"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C.COLOR_REALISED, width=3), name="Transitivität (Mittel der Pläne)",
                             hovertemplate="%{x} Vorschläge je Kante: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Vorschläge je Kante", type="log")
    fig.update_yaxes(title_text="Transitivität", type="log")
    return _base(fig)
