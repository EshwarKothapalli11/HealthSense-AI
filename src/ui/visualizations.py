"""
visualizations.py — Plotly chart helpers for HealthSense AI.

Light glassmorphism theme with clinical blue/teal color palette.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

LIGHT_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,251,255,0.60)",
    font=dict(
        color="#2d4060",
        family="Inter, sans-serif"
    ),
    margin=dict(l=20, r=20, t=44, b=20),
)

GAUGE_COLORS = {
    "low": "#48bb78",
    "moderate": "#ed8936",
    "high": "#f56565",
    "critical": "#e53e3e",
}


def plot_gauge(value: float, title: str) -> go.Figure:
    """Render a single risk gauge (0–100 scale)."""
    color = "#48bb78" if value < 33 else "#ed8936" if value < 66 else "#f56565"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 16, "color": "#1a3a7a"}},
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(99,150,210,0.50)"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(99,150,210,0.06)",
            "bordercolor": "rgba(99,150,210,0.20)",
            "steps": [
                {"range": [0, 33], "color": "rgba(72,187,120,0.15)"},
                {"range": [33, 66], "color": "rgba(237,137,54,0.15)"},
                {"range": [66, 100], "color": "rgba(245,101,101,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value
            }
        }
    ))
    fig.update_layout(**LIGHT_LAYOUT, height=260)
    return fig


def plot_three_gauges(diabetes: float, heart: float, mental: float) -> go.Figure:
    """Render three risk gauges side-by-side."""
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    for col, (val, title) in enumerate([
        (diabetes, "Diabetes Risk"),
        (heart, "Heart Risk"),
        (mental, "Mental Health")
    ], start=1):
        color = "#48bb78" if val < 33 else "#ed8936" if val < 66 else "#f56565"
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": title, "font": {"size": 14, "color": "#1a3a7a"}},
            number={"suffix": "%", "font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(99,150,210,0.40)"},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(99,150,210,0.04)",
                "steps": [
                    {"range": [0, 33], "color": "rgba(72,187,120,0.10)"},
                    {"range": [33, 66], "color": "rgba(237,137,54,0.10)"},
                    {"range": [66, 100], "color": "rgba(245,101,101,0.10)"},
                ],
            }
        ), row=1, col=col)
    fig.update_layout(**LIGHT_LAYOUT, height=280)
    return fig


def plot_radar_chart(diabetes: float, heart: float, mental: float) -> go.Figure:
    """Render a health risk radar chart. Values in 0–100 range."""
    categories = ["Diabetes Risk", "Heart Risk", "Mental Health", "Overall Wellness"]
    overall = 100 - (diabetes * 0.35 + heart * 0.35 + mental * 0.30)
    values = [diabetes, heart, mental, overall]
    values += [values[0]]
    categories += [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(74, 144, 217, 0.12)",
        line=dict(color="#4a90d9", width=2.5),
        marker=dict(size=7, color="#4a90d9",
                    line=dict(color="#ffffff", width=2)),
    ))
    fig.update_layout(
        **LIGHT_LAYOUT,
        polar=dict(
            bgcolor="rgba(255,255,255,0.55)",
            angularaxis=dict(
                tickcolor="rgba(99,150,210,0.50)",
                linecolor="rgba(99,150,210,0.30)",
                gridcolor="rgba(99,150,210,0.18)",
                tickfont=dict(size=12, color="#2d4060"),
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickcolor="rgba(99,150,210,0.40)",
                gridcolor="rgba(99,150,210,0.18)",
                tickfont=dict(size=10, color="rgba(60,90,140,0.55)"),
            ),
        ),
        height=380,
    )
    return fig


def plot_feature_bar(feature_names: list[str], scores: list[float], title: str) -> go.Figure:
    """Horizontal bar chart for top features, light glass theme."""
    sorted_pairs = sorted(zip(scores, feature_names), reverse=True)[:8]
    if not sorted_pairs:
        s, f = [0], ["None"]
    else:
        s, f = zip(*sorted_pairs)
    max_s = max(s) if max(s) > 0 else 1
    colors = [
        f"rgba(74,144,217,{0.4 + 0.6 * (v / max_s)})" for v in s
    ]
    fig = go.Figure(go.Bar(
        x=list(s),
        y=list(f),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.update_layout(
        **LIGHT_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#1a3a7a")),
        xaxis=dict(gridcolor="rgba(99,150,210,0.10)", zerolinecolor="rgba(99,150,210,0.20)"),
        yaxis=dict(gridcolor="rgba(99,150,210,0.10)"),
        height=320,
        showlegend=False,
    )
    return fig
