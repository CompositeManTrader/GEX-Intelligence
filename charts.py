"""
charts.py — visualizaciones estilo gexbot.com
─────────────────────────────────────────────
Paleta terminal oscura (Bloomberg-like).
Cada chart es una función pura: recibe dataframes, devuelve go.Figure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

# Paleta (gexbot / bloomberg terminal)
CLR_BG      = "#0D1117"
CLR_PANEL   = "#161B22"
CLR_GRID    = "#21262D"
CLR_TXT     = "#C9D1D9"
CLR_TXT_DIM = "#8B949E"
CLR_GREEN   = "#3FB950"
CLR_RED     = "#F85149"
CLR_AMBER   = "#D29922"
CLR_BLUE    = "#58A6FF"
CLR_PURPLE  = "#BC8CFF"

BASE_LAYOUT = dict(
    paper_bgcolor=CLR_BG,
    plot_bgcolor=CLR_BG,
    font=dict(family="JetBrains Mono, SF Mono, Consolas, monospace",
              color=CLR_TXT, size=11),
    margin=dict(l=60, r=30, t=60, b=50),
    xaxis=dict(gridcolor=CLR_GRID, zerolinecolor=CLR_GRID, linecolor=CLR_GRID),
    yaxis=dict(gridcolor=CLR_GRID, zerolinecolor=CLR_GRID, linecolor=CLR_GRID),
    hovermode="x unified",
    showlegend=True,
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor=CLR_GRID, borderwidth=1),
)

def _ax(which: str, **overrides) -> dict:
    """Merge BASE_LAYOUT[axis] con overrides. Los overrides ganan."""
    base = dict(BASE_LAYOUT[which])
    base.update(overrides)
    return base

def _base(**overrides) -> dict:
    """BASE_LAYOUT sin xaxis/yaxis, más overrides que sobrescriben cualquier key duplicada."""
    out = {k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    out.update(overrides)
    return out

def _layout_no_axes() -> dict:
    """Retrocompat: BASE_LAYOUT sin xaxis/yaxis."""
    return {k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")}

# ═══════════════════════════════════════════════════════════════
# GEX PROFILE — bar chart por strike (el icónico de gexbot)
# ═══════════════════════════════════════════════════════════════
def gex_profile_chart(gex_df: pd.DataFrame, spot: float, summary: Dict,
                      strike_range_pct: float = 0.08, ticker: str = "") -> go.Figure:
    """
    Barras horizontales verdes/rojas = calls/puts GEX por strike.
    Líneas horizontales: spot, zero-gamma, call wall, put wall.
    """
    fig = go.Figure()
    if gex_df.empty or spot <= 0:
        fig.update_layout(**BASE_LAYOUT, title="Sin datos")
        return fig
    lo, hi = spot * (1 - strike_range_pct), spot * (1 + strike_range_pct)
    df = gex_df[gex_df["strike"].between(lo, hi)].copy().sort_values("strike")
    if df.empty:
        df = gex_df.copy()
    # Barras CALLS (negativo en convención dealer short-calls, lo mostramos rojo)
    fig.add_trace(go.Bar(
        y=df["strike"], x=df["calls_gex"], orientation="h",
        name="Calls GEX",
        marker=dict(color=CLR_RED, line=dict(width=0)),
        hovertemplate="Strike: $%{y:.0f}<br>Calls GEX: $%{x:.2f}M<extra></extra>",
    ))
    # Barras PUTS (positivo, verde)
    fig.add_trace(go.Bar(
        y=df["strike"], x=df["puts_gex"], orientation="h",
        name="Puts GEX",
        marker=dict(color=CLR_GREEN, line=dict(width=0)),
        hovertemplate="Strike: $%{y:.0f}<br>Puts GEX: $%{x:.2f}M<extra></extra>",
    ))
    # Líneas clave
    lines = [
        (spot, CLR_BLUE,  "dash",   f"Spot ${spot:.2f}"),
    ]
    if summary.get("zero_gamma"):
        lines.append((summary["zero_gamma"], CLR_AMBER, "solid",
                      f"Zero-Γ ${summary['zero_gamma']:.2f}"))
    if summary.get("call_wall"):
        lines.append((summary["call_wall"], CLR_RED, "dot",
                      f"Call Wall ${summary['call_wall']:.0f}"))
    if summary.get("put_wall"):
        lines.append((summary["put_wall"], CLR_GREEN, "dot",
                      f"Put Wall ${summary['put_wall']:.0f}"))
    if summary.get("max_pain"):
        lines.append((summary["max_pain"], CLR_PURPLE, "dashdot",
                      f"Max Pain ${summary['max_pain']:.0f}"))
    shapes, annots = [], []
    x_abs_max = float(df[["calls_gex", "puts_gex"]].abs().max().max()) or 1.0
    for yval, clr, dash, lbl in lines:
        shapes.append(dict(type="line", x0=-x_abs_max*1.2, x1=x_abs_max*1.2,
                           y0=yval, y1=yval,
                           line=dict(color=clr, width=1.3, dash=dash)))
        annots.append(dict(x=x_abs_max*1.15, y=yval, text=lbl,
                           xanchor="right", yanchor="bottom",
                           font=dict(color=clr, size=10),
                           showarrow=False, bgcolor="rgba(13,17,23,0.7)",
                           bordercolor=clr, borderwidth=0.5, borderpad=2))

    regime = summary.get("regime", "?")
    total = summary.get("total_net_gex_m", 0)
    regime_clr = CLR_GREEN if regime == "POSITIVE" else CLR_RED
    title = (f"<b>GEX PROFILE {ticker}</b>"
             f"<sup>  Net ${total:+.1f}M · "
             f"<span style='color:{regime_clr}'>{regime} GAMMA</span></sup>")

    fig.update_layout(
        **_layout_no_axes(),
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=13)),
        barmode="relative",
        height=620,
        shapes=shapes, annotations=annots,
        xaxis=_ax("xaxis",
                  title=dict(text="GEX ($M per 1% move)",
                             font=dict(size=10, color=CLR_TXT_DIM)),
                  zeroline=True, zerolinewidth=1.5, zerolinecolor=CLR_TXT_DIM),
        yaxis=_ax("yaxis",
                  title=dict(text="Strike",
                             font=dict(size=10, color=CLR_TXT_DIM)),
                  range=[lo, hi]),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# CUMULATIVE GEX — muestra el cruce por cero (Zero-Gamma visible)
# ═══════════════════════════════════════════════════════════════
def cumulative_gex_chart(gex_df: pd.DataFrame, spot: float,
                         summary: Dict, ticker: str = "") -> go.Figure:
    fig = go.Figure()
    if gex_df.empty: return fig
    df = gex_df.sort_values("strike").reset_index(drop=True)
    df["cum"] = df["net_gex"].cumsum()
    colors = [CLR_GREEN if v >= 0 else CLR_RED for v in df["cum"]]
    fig.add_trace(go.Scatter(
        x=df["strike"], y=df["cum"], mode="lines",
        line=dict(color=CLR_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
        name="Cumulative GEX",
        hovertemplate="Strike: $%{x:.0f}<br>Cum GEX: $%{y:.2f}M<extra></extra>",
    ))
    shapes = [
        dict(type="line", x0=spot, x1=spot,
             y0=df["cum"].min()*1.1, y1=df["cum"].max()*1.1,
             line=dict(color=CLR_BLUE, width=1.3, dash="dash")),
    ]
    if summary.get("zero_gamma"):
        shapes.append(dict(type="line",
                           x0=summary["zero_gamma"], x1=summary["zero_gamma"],
                           y0=df["cum"].min()*1.1, y1=df["cum"].max()*1.1,
                           line=dict(color=CLR_AMBER, width=1.5)))
    fig.update_layout(
        **_layout_no_axes(),
        title=dict(text=f"<b>CUMULATIVE GEX {ticker}</b>"
                        "<sup>  Cruce por cero = Gamma Flip</sup>",
                   x=0.01, y=0.97, font=dict(size=13)),
        height=360, shapes=shapes,
        xaxis=_ax("xaxis", title="Strike"),
        yaxis=_ax("yaxis", title="Cum. GEX ($M)",
                  zeroline=True, zerolinewidth=1.5, zerolinecolor=CLR_TXT_DIM),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# GEX BY EXPIRY — stacked por vencimiento
# ═══════════════════════════════════════════════════════════════
def gex_by_expiry_chart(gex_by_exp: Dict[str, float], ticker: str = "") -> go.Figure:
    """gex_by_exp = {exp_str: net_gex_m}"""
    fig = go.Figure()
    if not gex_by_exp:
        return fig
    exps = list(gex_by_exp.keys())
    vals = list(gex_by_exp.values())
    colors = [CLR_GREEN if v >= 0 else CLR_RED for v in vals]
    fig.add_trace(go.Bar(
        x=exps, y=vals,
        marker=dict(color=colors, line=dict(color=CLR_GRID, width=0.5)),
        text=[f"{v:+.1f}M" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color=CLR_TXT),
        hovertemplate="%{x}<br>GEX: $%{y:.2f}M<extra></extra>",
    ))
    fig.update_layout(
        **_base(showlegend=False),
        title=dict(text=f"<b>GEX BY EXPIRY {ticker}</b>"
                        "<sup>  Vencimiento que concentra gamma</sup>",
                   x=0.01, y=0.97, font=dict(size=13)),
        height=360,
        xaxis=_ax("xaxis", title="Expiry"),
        yaxis=_ax("yaxis", title="Net GEX ($M)",
                  zeroline=True, zerolinewidth=1.5, zerolinecolor=CLR_TXT_DIM),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# DEX PROFILE
# ═══════════════════════════════════════════════════════════════
def dex_profile_chart(dex_df: pd.DataFrame, spot: float, ticker: str = "") -> go.Figure:
    fig = go.Figure()
    if dex_df.empty: return fig
    lo, hi = spot * 0.92, spot * 1.08
    df = dex_df[dex_df["strike"].between(lo, hi)].copy()
    fig.add_trace(go.Bar(
        y=df["strike"], x=df["calls_dex"], orientation="h",
        name="Calls DEX",
        marker_color=CLR_RED,
        hovertemplate="Strike: $%{y:.0f}<br>DEX: $%{x:.2f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["strike"], x=df["puts_dex"], orientation="h",
        name="Puts DEX",
        marker_color=CLR_GREEN,
        hovertemplate="Strike: $%{y:.0f}<br>DEX: $%{x:.2f}M<extra></extra>",
    ))
    fig.update_layout(
        **_layout_no_axes(),
        title=dict(text=f"<b>DELTA EXPOSURE {ticker}</b>"
                        "<sup>  Dealer hedge flow direccional</sup>",
                   x=0.01, y=0.97, font=dict(size=13)),
        height=520, barmode="relative",
        shapes=[dict(type="line",
                     x0=df[["calls_dex","puts_dex"]].min().min()*1.1,
                     x1=df[["calls_dex","puts_dex"]].max().max()*1.1,
                     y0=spot, y1=spot,
                     line=dict(color=CLR_BLUE, width=1.3, dash="dash"))],
        xaxis=_ax("xaxis", title="DEX ($M)",
                  zeroline=True, zerolinecolor=CLR_TXT_DIM),
        yaxis=_ax("yaxis", title="Strike", range=[lo, hi]),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
# VANNA & CHARM combo
# ═══════════════════════════════════════════════════════════════
def vanna_charm_chart(vex_df: pd.DataFrame, charm_df: pd.DataFrame,
                      spot: float, ticker: str = "") -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Vanna Exposure", "Charm Exposure"),
                        horizontal_spacing=0.08)
    if not vex_df.empty:
        v = vex_df[vex_df["strike"].between(spot*0.9, spot*1.1)]
        fig.add_trace(go.Bar(y=v["strike"], x=v["net_vex"], orientation="h",
                             marker_color=[CLR_GREEN if x>=0 else CLR_RED for x in v["net_vex"]],
                             name="VEX", showlegend=False,
                             hovertemplate="Strike %{y:.0f}<br>VEX %{x:.2f}M<extra></extra>"),
                      row=1, col=1)
    if not charm_df.empty:
        c = charm_df[charm_df["strike"].between(spot*0.9, spot*1.1)]
        fig.add_trace(go.Bar(y=c["strike"], x=c["net_charm"], orientation="h",
                             marker_color=[CLR_AMBER if x>=0 else CLR_PURPLE for x in c["net_charm"]],
                             name="Charm", showlegend=False,
                             hovertemplate="Strike %{y:.0f}<br>Charm %{x:.2f}M<extra></extra>"),
                      row=1, col=2)
    for col in (1, 2):
        fig.add_shape(type="line", x0=-100, x1=100, y0=spot, y1=spot,
                      line=dict(color=CLR_BLUE, width=1.2, dash="dash"),
                      row=1, col=col, xref=f"x{col} domain" if col>1 else "x")
    fig.update_layout(
        **_base(showlegend=False),
        title=dict(text=f"<b>VANNA & CHARM {ticker}</b>"
                        "<sup>  Flujos por IV y por tiempo</sup>",
                   x=0.01, y=0.97, font=dict(size=13)),
        height=500,
    )
    for i in (1, 2):
        fig.update_xaxes(gridcolor=CLR_GRID, zerolinecolor=CLR_TXT_DIM,
                         zeroline=True, row=1, col=i)
        fig.update_yaxes(gridcolor=CLR_GRID, range=[spot*0.9, spot*1.1], row=1, col=i)
    return fig

# ═══════════════════════════════════════════════════════════════
# OI BY STRIKE (calls vs puts) — reference chart
# ═══════════════════════════════════════════════════════════════
def oi_chart(chains: Dict, spot: float, ticker: str = "") -> go.Figure:
    fig = go.Figure()
    call_oi = {}; put_oi = {}
    for exp, ch in chains.items():
        for _, r in ch["calls"].iterrows():
            call_oi[r["strike"]] = call_oi.get(r["strike"], 0) + r["openInterest"]
        for _, r in ch["puts"].iterrows():
            put_oi[r["strike"]] = put_oi.get(r["strike"], 0) + r["openInterest"]
    strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))
    strikes = [s for s in strikes if spot*0.9 <= s <= spot*1.1]
    fig.add_trace(go.Bar(x=strikes, y=[call_oi.get(s, 0) for s in strikes],
                         name="Calls OI", marker_color=CLR_RED,
                         hovertemplate="$%{x:.0f}<br>Calls OI: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=strikes, y=[-put_oi.get(s, 0) for s in strikes],
                         name="Puts OI", marker_color=CLR_GREEN,
                         hovertemplate="$%{x:.0f}<br>Puts OI: %{y:,.0f}<extra></extra>"))
    fig.update_layout(
        **_layout_no_axes(),
        title=dict(text=f"<b>OPEN INTEREST {ticker}</b>",
                   x=0.01, y=0.97, font=dict(size=13)),
        height=360, barmode="relative",
        shapes=[dict(type="line", x0=spot, x1=spot,
                     y0=-max(put_oi.values(), default=0)*1.1,
                     y1=max(call_oi.values(), default=0)*1.1,
                     line=dict(color=CLR_BLUE, width=1.3, dash="dash"))],
        xaxis=_ax("xaxis", title="Strike"),
        yaxis=_ax("yaxis", title="Open Interest (Puts ← → Calls)",
                  zeroline=True, zerolinecolor=CLR_TXT_DIM),
    )
    return fig

def compute_gex_by_expiry(chains, spot, r=0.045, q=0.013):
    """Helper: devuelve dict {exp: net_gex_m} para chart de expiries."""
    from gex_engine import compute_gex_profile
    out = {}
    for exp in chains:
        sub = compute_gex_profile(chains, spot, r, q, expiries=[exp])
        out[exp] = float(sub["net_gex"].sum()) if not sub.empty else 0.0
    return out
