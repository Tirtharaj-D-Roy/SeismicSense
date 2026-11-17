"""viz.py - all charts and the interactive Moon globe (Plotly)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import catalog as C

MOON_R = 1737.4
FONT = dict(family="Inter, Segoe UI, sans-serif", color="#E8E6F0")
BG = "rgba(0,0,0,0)"

# Approximate centres/radii of the big dark 'seas' (maria) - decoration so the globe looks like the Moon
MARIA = [
    ("Oceanus Procellarum", 18.4, -57.4, 1050, 1.6), ("Mare Imbrium", 32.8, -15.6, 560, 1.0),
    ("Mare Serenitatis", 28.0, 17.5, 340, 1.0), ("Mare Tranquillitatis", 8.5, 31.4, 430, 1.1),
    ("Mare Crisium", 17.0, 59.1, 280, 1.2), ("Mare Fecunditatis", -7.8, 51.3, 400, 0.8),
    ("Mare Nectaris", -15.2, 35.5, 170, 1.0), ("Mare Nubium", -21.3, -16.6, 360, 1.0),
    ("Mare Humorum", -24.4, -38.6, 210, 1.0), ("Mare Frigoris", 56.0, 1.4, 900, 0.18),
    ("Mare Vaporum", 13.3, 3.6, 120, 1.0), ("Mare Cognitum", -10.0, -23.0, 175, 1.0),
    ("Mare Insularum", 7.5, -30.9, 255, 1.0), ("Mare Smythii", 1.3, 87.0, 190, 1.0),
]


def _xyz(lat, lon, r=1.0):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    r = np.asarray(r, float)
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def _moon_surface(opacity=1.0):
    """Sphere with darker 'seas' (maria). Pure WebGL - works fully offline."""
    lat = np.linspace(-90, 90, 91)
    lon = np.linspace(-180, 180, 181)
    LO, LA = np.meshgrid(lon, lat)
    x, y, z = _xyz(LA, LO)
    rng = np.random.default_rng(7)
    shade = 0.72 + 0.06 * rng.standard_normal(LA.shape)
    la1, lo1 = np.radians(LA), np.radians(LO)
    for _, mla, mlo, r, flat in MARIA:
        la2, lo2 = np.radians(mla), np.radians(mlo)
        dlat, dlon = la1 - la2, (lo1 - lo2 + np.pi) % (2 * np.pi) - np.pi
        if flat < 1:   # elongated sea (e.g. Mare Frigoris): ellipse narrow in latitude
            d_km = MOON_R * np.sqrt((dlat / flat) ** 2 + (dlon * np.cos(la2)) ** 2)
        else:          # great-circle distance from the sea's centre
            d_km = MOON_R * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1))
        shade = np.where(d_km < r, shade - 0.32 * np.clip((r - d_km) / (0.35 * r), 0, 1), shade)
    return go.Surface(x=x, y=y, z=z, surfacecolor=shade, colorscale=[[0, "#2b2b31"], [0.5, "#6f6f78"], [1, "#c9c9cf"]],
                      cmin=0.25, cmax=0.95, showscale=False, opacity=opacity, hoverinfo="skip",
                      lighting=dict(ambient=0.65, diffuse=0.7, specular=0.05, roughness=0.9), name="Moon")


def _pts(fig, lat, lon, name, color, size=8, symbol="circle", text=None, r=1.012, opacity=0.9, line=None, show=True):
    x, y, z = _xyz(lat, lon, r)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", name=name, showlegend=show,
                               marker=dict(size=size, color=color, symbol=symbol, opacity=opacity,
                                           line=line or dict(color="white", width=0.5)),
                               text=text, hovertemplate="%{text}<extra></extra>" if text is not None else None))


def moon_map(locs: dict, nest_counts: pd.Series | None = None, layers=("deep", "meteoroid", "shallow", "artificial"),
             highlight: dict | None = None, felt: dict | None = None, height=620, rotation=(0, 0),
             xray: bool = False) -> go.Figure:
    """Interactive 3D Moon (drag to spin, scroll to zoom). rotation[0] = longitude facing the viewer.
    xray=True makes the surface see-through and draws deep moonquake nests at their real depth."""
    fig = go.Figure(_moon_surface(0.28 if xray else 1.0))
    if "deep" in layers:
        dm = locs["deep_nests"].copy()
        dm["n"] = dm["nest"].map(nest_counts).fillna(1) if nest_counts is not None else 5
        r = (1 - dm["depth_km"] / MOON_R) if xray else 1.012
        _pts(fig, dm.lat, dm.lon, "🌑 Deep moonquake nests" + (" (at true depth)" if xray else ""),
             C.CLASS_INFO["deep"]["color"], size=np.clip(np.sqrt(dm["n"]) * 1.3, 3, 16), r=r,
             text=[f"Deep moonquake nest A{a}<br>~{d} km underground<br>{int(n)} quakes recorded"
                   for a, d, n in zip(dm.nest, dm.depth_km, dm["n"])])
    if "meteoroid" in layers:
        m = locs["meteoroid"]
        _pts(fig, m.lat, m.lon, "☄️ Located meteoroid impacts", C.CLASS_INFO["meteoroid"]["color"], size=7,
             symbol="diamond", text=("Meteoroid impact<br>" + m.time.dt.strftime("%d %b %Y")).tolist())
    if "shallow" in layers:
        s = locs["shallow"]
        _pts(fig, s.lat, s.lon, "⚡ Shallow moonquakes", C.CLASS_INFO["shallow"]["color"], size=5 + 2 * s.Magnitude,
             symbol="square", text=("Shallow moonquake<br>" + s.time.dt.strftime("%d %b %Y") + "<br>magnitude "
                                    + s.Magnitude.astype(str)).tolist())
    if "artificial" in layers:
        a = locs["artificial"]
        _pts(fig, a.lat, a.lon, "🚀 Apollo rocket/lander crashes", C.CLASS_INFO["artificial"]["color"], size=7,
             symbol="x", text=(a.name + "<br>" + a.time.dt.strftime("%d %b %Y")).tolist(), line=dict(width=0))
    if felt:
        sites = C.APOLLO_SITES.set_index("site")
        rows = [(f"Apollo {k}", v) for k, v in felt.items() if v and v > 0]
        if rows:
            names, amps = zip(*rows)
            _pts(fig, [sites.loc[n, "lat"] for n in names], [sites.loc[n, "lon"] for n in names],
                 "How strongly each station felt it", "#FFB000",
                 size=[16 + 8 * np.log10(a + 1) for a in amps], r=1.018, opacity=0.45,
                 text=[f"{n} felt strength {a:g}" for n, a in rows], line=dict(width=0))
    st_ = C.APOLLO_SITES
    x, y, z = _xyz(st_.lat, st_.lon, 1.03)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers+text", text=st_.site, textposition="top center",
                               textfont=dict(color="#FFE066", size=11), name="◆ Apollo landing sites",
                               marker=dict(size=4.5, color="#FFE066", symbol="diamond"), hovertemplate="%{text}<extra></extra>"))
    if highlight:
        x, y, z = _xyz([highlight["lat"]], [highlight["lon"]], 1.04)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers+text", text=[highlight.get("label", "Source")],
                                   textposition="bottom center", textfont=dict(color="#FF6B8B", size=13),
                                   marker=dict(size=14, color="rgba(255,46,99,0.25)", line=dict(color="#FF2E63", width=5)),
                                   name="📍 Known source of this event", hovertemplate="%{text}<extra></extra>"))
    lo = np.radians(rotation[0])
    axis = dict(visible=False, showbackground=False)
    fig.update_layout(height=height, paper_bgcolor=BG, font=FONT, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data", bgcolor=BG,
                                 camera=dict(eye=dict(x=1.45 * np.cos(lo), y=1.45 * np.sin(lo), z=0.2), up=dict(x=0, y=0, z=1))),
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.5, xanchor="center",
                                  bgcolor="rgba(20,18,35,0.65)", font=dict(size=12)))
    return fig


def timeline(ev: pd.DataFrame, label_col="label", height=430) -> go.Figure:
    fig = go.Figure()
    for c in ["deep", "meteoroid", "shallow", "artificial", "unknown"]:
        d = ev[(ev[label_col] == c) & (ev.strength > 0.1)]
        if d.empty:
            continue
        info = C.CLASS_INFO[c]
        fig.add_trace(go.Scattergl(x=d.time, y=d.strength, mode="markers", name=f"{info['emoji']} {info['name']}",
                                   marker=dict(size=5 if c in ("deep", "meteoroid", "unknown") else 11,
                                               color=info["color"], opacity=0.55 if c in ("deep", "unknown") else 0.85),
                                   text=d.event_id, hovertemplate="%{text}<br>%{x|%d %b %Y %H:%M}<br>strength %{y:.1f}<extra></extra>"))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor="rgba(255,255,255,0.03)", font=FONT,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
                      yaxis=dict(title="Shaking strength (catalog amplitude, log scale)", type="log", gridcolor="rgba(255,255,255,0.08)"),
                      xaxis=dict(title="", gridcolor="rgba(255,255,255,0.08)"))
    return fig


def monthly_counts(ev: pd.DataFrame, label_col="label", height=330) -> go.Figure:
    g = ev.groupby([pd.Grouper(key="time", freq="MS"), label_col]).size().unstack(fill_value=0)
    fig = go.Figure()
    for c in ["deep", "meteoroid", "shallow", "artificial", "unknown"]:
        if c in g:
            info = C.CLASS_INFO[c]
            fig.add_trace(go.Bar(x=g.index, y=g[c], name=f"{info['emoji']} {info['name']}", marker_color=info["color"]))
    fig.update_layout(barmode="stack", height=height, paper_bgcolor=BG, plot_bgcolor="rgba(255,255,255,0.03)",
                      font=FONT, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      yaxis=dict(title="Events per month", gridcolor="rgba(255,255,255,0.08)"), bargap=0.05)
    return fig


def tidal_heartbeat(ev: pd.DataFrame, label_col="label", height=380) -> go.Figure:
    """Share of events at each point of the Moon's 27.55-day orbit (0 = closest to Earth)."""
    fig = go.Figure()
    bins = np.linspace(0, 27.5546, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    for c in ["deep", "meteoroid"]:
        d = ev.loc[ev[label_col] == c, "days_since_perigee"]
        h, _ = np.histogram(d, bins=bins)
        info = C.CLASS_INFO[c]
        fig.add_trace(go.Scatter(x=centers, y=100 * h / max(h.sum(), 1), mode="lines+markers", line_shape="spline",
                                 name=f"{info['emoji']} {info['name']}", line=dict(color=info["color"], width=4),
                                 marker=dict(size=8)))
    fig.add_hline(y=100 / 9, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                  annotation_text="if timing were random", annotation_font_color="#ccc")
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor="rgba(255,255,255,0.03)", font=FONT,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      xaxis=dict(title="Days after the Moon was closest to Earth (perigee)", gridcolor="rgba(255,255,255,0.08)"),
                      yaxis=dict(title="% of events", gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"))
    return fig


def probability_bars(probs: dict, height=210) -> go.Figure:
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v * 100 for _, v in items], y=[f"{C.CLASS_INFO[k]['emoji']} {C.CLASS_INFO[k]['name']}" for k, _ in items],
        orientation="h", marker_color=[C.CLASS_INFO[k]["color"] for k, _ in items],
        text=[f"{v * 100:.0f}%" for _, v in items], textposition="outside", cliponaxis=False))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=FONT, margin=dict(l=10, r=50, t=10, b=10),
                      xaxis=dict(range=[0, 125], visible=False), yaxis=dict(tickfont=dict(size=14)))
    return fig


def confusion(cm: list, classes: list, height=380) -> go.Figure:
    cm = np.array(cm)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    names = [f"{C.CLASS_INFO[c]['emoji']} {C.CLASS_INFO[c]['name']}" for c in classes]
    fig = go.Figure(go.Heatmap(z=pct, x=names, y=names, colorscale="Purples", showscale=False,
                               text=[[f"{pct[i, j]:.0f}%<br>({cm[i, j]})" for j in range(len(classes))] for i in range(len(classes))],
                               texttemplate="%{text}", textfont=dict(size=15)))
    fig.update_layout(height=height, paper_bgcolor=BG, font=FONT, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="What the AI said", side="top"), yaxis=dict(title="What NASA scientists said", autorange="reversed"))
    return fig


def importance_bars(imp: dict, height=330) -> go.Figure:
    items = sorted(((k, v) for k, v in imp.items() if v > 0.001), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color="#8B6CFF", text=[f"{v * 100:.0f}%" for _, v in items], textposition="outside"))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=FONT, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(visible=False, range=[0, max(v for _, v in items) * 125]))
    return fig
