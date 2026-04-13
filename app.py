"""
app.py — GEX DASHBOARD (clone funcional de gexbot.com)
══════════════════════════════════════════════════════
Standalone Streamlit app para análisis de Gamma Exposure.

Ejecutar:
    streamlit run app.py

Arquitectura:
    data_layer.py → fetcher con anti-rate-limit
    gex_engine.py → Black-Scholes + IV + GEX/DEX/VEX/Charm
    charts.py     → visualizaciones Bloomberg-dark
    app.py        → orquestación + UI
"""
from __future__ import annotations
import logging
import time
from datetime import datetime
import pytz
import pandas as pd
import streamlit as st

from data_layer import fetch_options_yahoo
from gex_engine import (
    enrich_chain_with_greeks, compute_gex_profile, compute_dex_profile,
    compute_vex_profile, compute_charm_profile, compute_summary,
)
from charts import (
    gex_profile_chart, cumulative_gex_chart, gex_by_expiry_chart,
    dex_profile_chart, vanna_charm_chart, oi_chart, compute_gex_by_expiry,
    CLR_GREEN, CLR_RED, CLR_AMBER, CLR_BLUE, CLR_TXT, CLR_TXT_DIM,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s · %(name)s · %(message)s")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS (gexbot look)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Reset & dark theme */
.stApp {
    background-color: #0D1117;
    color: #C9D1D9;
}
#MainMenu, footer, header { visibility: hidden; }

/* Fuente monoespaciada profesional */
html, body, [class*="css"], .stMarkdown, .stButton>button, .stTextInput>div>input,
.stSelectbox, .stMetric, .stDataFrame, .stTabs {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, 'Courier New', monospace !important;
}

/* Métricas estilo terminal */
[data-testid="stMetricValue"] {
    color: #F0F6FC !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Panel metric cards */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 4px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.metric-card h4 {
    color: #8B949E;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 6px 0;
    font-weight: 500;
}
.metric-card .val {
    color: #F0F6FC;
    font-size: 1.5rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
    color: #8B949E;
    font-size: 0.75rem;
    margin-top: 4px;
}
.pos { color: #3FB950 !important; }
.neg { color: #F85149 !important; }
.amb { color: #D29922 !important; }

/* Header ticker strip */
.ticker-strip {
    background: #161B22;
    border-bottom: 1px solid #21262D;
    padding: 8px 16px;
    margin: -16px -16px 16px -16px;
    display: flex;
    gap: 32px;
    align-items: center;
    flex-wrap: wrap;
}
.ticker-strip .item {
    display: flex;
    flex-direction: column;
    min-width: 80px;
}
.ticker-strip .item .lbl {
    color: #8B949E;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.ticker-strip .item .val {
    color: #F0F6FC;
    font-size: 0.95rem;
    font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0D1117;
    border-bottom: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    background: #0D1117;
    color: #8B949E;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
    background: #0D1117 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #010409;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #8B949E !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Buttons */
.stButton>button {
    background: #161B22;
    color: #58A6FF;
    border: 1px solid #21262D;
    border-radius: 3px;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.stButton>button:hover {
    background: #1F242C;
    border-color: #58A6FF;
}

/* Dataframe */
.stDataFrame {
    background: #161B22;
    border: 1px solid #21262D;
}

/* Section headers */
.section-h {
    color: #58A6FF;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #21262D;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
}

/* Regime badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-pos { background: rgba(63,185,80,0.15); color: #3FB950; border: 1px solid #3FB950; }
.badge-neg { background: rgba(248,81,73,0.15); color: #F85149; border: 1px solid #F85149; }
.badge-neu { background: rgba(210,153,34,0.15); color: #D29922; border: 1px solid #D29922; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLES
# ═══════════════════════════════════════════════════════════════
TICKERS_PRESET = ["SPX", "SPY", "QQQ", "IWM", "DIA", "NDX", "AAPL", "MSFT",
                  "NVDA", "TSLA", "META", "GOOG", "AMZN"]

with st.sidebar:
    st.markdown("### GEX DASHBOARD")
    st.caption("v1.0 · Terminal edition")
    st.markdown("---")

    ticker = st.selectbox("Ticker", TICKERS_PRESET, index=1)
    custom = st.text_input("Custom ticker", value="").strip().upper()
    if custom:
        ticker = custom

    n_exp = st.slider("Vencimientos a incluir", 1, 10, 5,
                      help="Más vencimientos = GEX más 'completo' pero más lento")

    risk_free = st.number_input("Risk-free rate (r)", 0.0, 0.15, 0.045, 0.005,
                                 format="%.3f")
    div_yield = st.number_input("Dividend yield (q)", 0.0, 0.10, 0.013, 0.002,
                                 format="%.3f")

    strike_rng = st.slider("Rango de strikes (± %)", 3, 20, 8) / 100.0

    st.markdown("---")
    refresh_clicked = st.button("🔄 REFRESH DATA", use_container_width=True)
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    st.caption("⚠️ Yahoo tiene rate-limit — usar con moderación")

    st.markdown("---")
    st.markdown("#### NOTAS")
    st.caption("""
    **Convención dealer:**
    Dealers = short calls, long puts (retail compra protección).
    
    **GEX > 0:** pinning, vol baja
    **GEX < 0:** amplificador, vol alta
    **Zero-Gamma:** frontera entre regímenes
    """)

# ═══════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════
CDMX = pytz.timezone("America/Mexico_City")

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, n_exp: int, r: float, q: float):
    """TTL = 5 min para evitar hammering a Yahoo."""
    chains_raw, spot = fetch_options_yahoo(ticker, n_exp=n_exp)
    if not chains_raw or not spot:
        return None, None, None
    chains = enrich_chain_with_greeks(chains_raw, spot, r, q)
    return chains, spot, datetime.now(CDMX).strftime("%Y-%m-%d %H:%M:%S CDMX")

if refresh_clicked:
    load_data.clear()

with st.spinner(f"Fetching {ticker} chains (Yahoo anti-rate-limit)..."):
    chains, spot, ts_fetch = load_data(ticker, n_exp, risk_free, div_yield)

if not chains or not spot:
    st.error(f"❌ No se pudieron obtener datos para **{ticker}**. "
             "Posibles causas: rate-limit de Yahoo, ticker sin opciones, "
             "o red bloqueada. Intenta en 1-2 min con REFRESH.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# COMPUTE ALL PROFILES (cached en memoria de sesión)
# ═══════════════════════════════════════════════════════════════
gex_df   = compute_gex_profile(chains, spot, risk_free, div_yield)
dex_df   = compute_dex_profile(chains, spot)
vex_df   = compute_vex_profile(chains, spot)
charm_df = compute_charm_profile(chains, spot)
summary  = compute_summary(gex_df, chains, spot)
gex_by_exp = compute_gex_by_expiry(chains, spot, risk_free, div_yield)

# ═══════════════════════════════════════════════════════════════
# HEADER — TICKER STRIP
# ═══════════════════════════════════════════════════════════════
regime = summary.get("regime", "?")
regime_badge = ("badge-pos" if regime == "POSITIVE" else
                "badge-neg" if regime == "NEGATIVE" else "badge-neu")
net_gex = summary.get("total_net_gex_m", 0)
net_gex_clr = "pos" if net_gex >= 0 else "neg"

zf = summary.get("zero_gamma")
dist_flip = summary.get("distance_to_flip")
dist_txt = f"{dist_flip:+.2f}%" if dist_flip is not None else "—"

st.markdown(f"""
<div class="ticker-strip">
  <div class="item"><span class="lbl">Ticker</span><span class="val" style="color:#58A6FF">{ticker}</span></div>
  <div class="item"><span class="lbl">Spot</span><span class="val">${spot:,.2f}</span></div>
  <div class="item"><span class="lbl">Net GEX</span>
    <span class="val {net_gex_clr}">${net_gex:+,.1f}M</span></div>
  <div class="item"><span class="lbl">Régimen</span>
    <span class="val"><span class="badge {regime_badge}">{regime} GAMMA</span></span></div>
  <div class="item"><span class="lbl">Zero-Γ</span>
    <span class="val">{f'${zf:,.2f}' if zf else '—'}</span></div>
  <div class="item"><span class="lbl">Dist. a Flip</span><span class="val amb">{dist_txt}</span></div>
  <div class="item"><span class="lbl">Updated</span>
    <span class="val" style="font-size:0.78rem;color:#8B949E">{ts_fetch}</span></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# KEY LEVELS — CARDS superiores (estilo gexbot)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-h">KEY LEVELS</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

def card(col, label, value, sub="", color_class=""):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <h4>{label}</h4>
          <div class="val {color_class}">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

card(c1, "Call Wall", f"${summary.get('call_wall', 0):,.0f}" if summary.get('call_wall') else "—",
     "Mayor resistencia GEX", "neg")
card(c2, "Put Wall", f"${summary.get('put_wall', 0):,.0f}" if summary.get('put_wall') else "—",
     "Mayor soporte GEX", "pos")
card(c3, "Zero-Gamma", f"${summary.get('zero_gamma', 0):,.2f}" if summary.get('zero_gamma') else "—",
     "Flip régimen +/−", "amb")
card(c4, "Max Pain", f"${summary.get('max_pain', 0):,.0f}" if summary.get('max_pain') else "—",
     "Strike de mayor dolor", "")
card(c5, "Expected Move", f"±${summary.get('expected_move', 0):,.2f}" if summary.get('expected_move') else "—",
     "1-SD straddle ATM", "")
card(c6, "Put/Call Ratio", f"{summary.get('put_call_ratio', 0):.3f}" if summary.get('put_call_ratio') else "—",
     "OI Ratio", "")

# ═══════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ═══════════════════════════════════════════════════════════════
tab_gex, tab_dex, tab_vanna, tab_oi, tab_raw, tab_playbook = st.tabs([
    "📊 GEX PROFILE", "⚡ DEX", "🔄 VANNA/CHARM",
    "📖 OPEN INTEREST", "🗂 RAW DATA", "🎯 PLAYBOOK"
])

with tab_gex:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(
            gex_profile_chart(gex_df, spot, summary, strike_rng, ticker),
            use_container_width=True, config={"displayModeBar": False}
        )
    with col_b:
        st.plotly_chart(
            cumulative_gex_chart(gex_df, spot, summary, ticker),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.plotly_chart(
            gex_by_expiry_chart(gex_by_exp, ticker),
            use_container_width=True, config={"displayModeBar": False}
        )

    # Top walls tabla
    st.markdown('<div class="section-h">TOP WALLS</div>', unsafe_allow_html=True)
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("**🔴 Call Walls (Resistencia)**")
        cw = pd.DataFrame(summary.get("call_walls", []))
        if not cw.empty:
            cw["dist_%"] = ((cw["strike"] / spot - 1) * 100).round(2)
            cw["calls_gex"] = cw["calls_gex"].round(2)
            st.dataframe(cw.rename(columns={"strike": "Strike",
                                            "calls_gex": "GEX ($M)",
                                            "dist_%": "Dist %"}),
                         use_container_width=True, hide_index=True)
    with wc2:
        st.markdown("**🟢 Put Walls (Soporte)**")
        pw = pd.DataFrame(summary.get("put_walls", []))
        if not pw.empty:
            pw["dist_%"] = ((pw["strike"] / spot - 1) * 100).round(2)
            pw["puts_gex"] = pw["puts_gex"].round(2)
            st.dataframe(pw.rename(columns={"strike": "Strike",
                                            "puts_gex": "GEX ($M)",
                                            "dist_%": "Dist %"}),
                         use_container_width=True, hide_index=True)

with tab_dex:
    st.plotly_chart(
        dex_profile_chart(dex_df, spot, ticker),
        use_container_width=True, config={"displayModeBar": False}
    )
    total_dex = float(dex_df["net_dex"].sum()) if not dex_df.empty else 0
    st.metric("Total Net DEX", f"${total_dex:+.1f}M",
              help="Dealer delta exposure. Positivo: dealers netamente largos delta.")

with tab_vanna:
    st.plotly_chart(
        vanna_charm_chart(vex_df, charm_df, spot, ticker),
        use_container_width=True, config={"displayModeBar": False}
    )
    c1, c2 = st.columns(2)
    with c1:
        total_vex = float(vex_df["net_vex"].sum()) if not vex_df.empty else 0
        st.metric("Total Net VEX", f"${total_vex:+.2f}M",
                  help="Vanna rally trigger: VEX > 0 + IV cayendo → presión alcista")
    with c2:
        total_charm = float(charm_df["net_charm"].sum()) if not charm_df.empty else 0
        st.metric("Total Net Charm", f"${total_charm:+.2f}M",
                  help="Charm flow: hedgeo por decay de delta, fuerte cerca de OPEX")

with tab_oi:
    st.plotly_chart(
        oi_chart(chains, spot, ticker),
        use_container_width=True, config={"displayModeBar": False}
    )

with tab_raw:
    st.markdown("#### Chains con Greeks calculados")
    exp_sel = st.selectbox("Vencimiento", list(chains.keys()))
    side_sel = st.radio("Lado", ["calls", "puts"], horizontal=True)
    df_show = chains[exp_sel][side_sel].copy()
    if not df_show.empty:
        cols = ["strike", "midPrice", "openInterest", "volume", "iv",
                "delta", "gamma", "vanna", "charm", "vega", "moneyness"]
        cols = [c for c in cols if c in df_show.columns]
        st.dataframe(
            df_show[cols].round(4),
            use_container_width=True, hide_index=True, height=500
        )
    st.caption(f"DTE: {chains[exp_sel]['dte']} | "
               f"Rows: {len(df_show)} (tras filtros de calidad)")

with tab_playbook:
    st.markdown(f"""
### 🎯 PLAYBOOK INTRADAY — {ticker}

**Régimen actual:** <span class="badge badge-{'pos' if regime == 'POSITIVE' else 'neg'}">{regime} GAMMA</span>

**Net GEX:** `${net_gex:+,.1f}M` por movimiento de 1%.
""", unsafe_allow_html=True)
    if regime == "POSITIVE":
        st.markdown(f"""
#### Entorno POSITIVE gamma → Dealers SUPRIMEN volatilidad

**Comportamiento esperado:**
- Rango estrecho; pullbacks al **Put Wall ${summary.get('put_wall', 0):,.0f}** se recompran.
- Rallies topan contra el **Call Wall ${summary.get('call_wall', 0):,.0f}** (dealer-fade).
- Cualquier movimiento hacia arriba → dealers venden futuros/acción para mantener delta-neutral.

**Tácticas mesa:**
1. **Mean-reversion intra-día:** fade rallies cerca del Call Wall, compra dips cerca del Put Wall.
2. **Vender primas:** short strangles/iron condors cerrando dentro del rango Call/Put Wall — theta decay favorece.
3. **Evitar breakouts direccionales** — el flujo de hedgeo los mata.
4. **Atento al Zero-Gamma ${summary.get('zero_gamma', 0):,.2f}**: si el spot se mueve hacia ese nivel, prepárate para switch de régimen — cierra cortos de vol.
""")
    else:
        st.markdown(f"""
#### Entorno NEGATIVE gamma → Dealers AMPLIFICAN volatilidad

**Comportamiento esperado:**
- Movimientos en la dirección del flujo se aceleran (dealers venden caídas, compran rallies).
- Gaps, barras amplias, ruptura de niveles técnicos con volumen.
- Alta sensibilidad a noticias y flujo institucional.

**Tácticas mesa:**
1. **Trend-following:** operar en la dirección del momentum; si rompe el Put Wall ${summary.get('put_wall', 0):,.0f} o Call Wall ${summary.get('call_wall', 0):,.0f} con volumen, perseguir.
2. **Comprar vol (long straddles/gamma)** — en régimen negativo la realizada suele superar la implícita.
3. **Evitar vender primas desnudas** — el riesgo de cola es real.
4. **Objetivo de recuperación:** el Zero-Gamma ${summary.get('zero_gamma', 0):,.2f} actúa como imán cuando llega el cierre de rebalanceo (típico fin de semana OPEX).
5. **Vigilar VIX** — en negative gamma, subidas de VIX > 3 pts en un día son comunes.
""")

    st.markdown(f"""
---
#### Cheatsheet de niveles
| Nivel | Valor | Uso |
|---|---|---|
| Spot | `${spot:,.2f}` | Referencia |
| Call Wall | `${summary.get('call_wall', 0):,.0f}` | Resistencia intra-día |
| Put Wall | `${summary.get('put_wall', 0):,.0f}` | Soporte intra-día |
| Zero-Gamma | `${summary.get('zero_gamma', 0):,.2f}` | Flip de régimen |
| Max Pain | `${summary.get('max_pain', 0):,.0f}` | Gravitación hacia expiración |
| Expected Move (±1σ) | `±${summary.get('expected_move', 0):,.2f}` | Rango implícito near-term |
| Put/Call Ratio | `{summary.get('put_call_ratio', 0):.3f}` | Sentimiento (>1 defensivo) |

#### Señales de transición
- Si **spot cruza Zero-Gamma al alza** → exit de ventas de vol, alpha en direccional.
- Si **Put Wall rompe con volumen** → gamma se negativiza rápido, esperar expansión.
- Si **Net GEX absoluto cae > 30%** en un día → "gamma unwind" post-OPEX, liquidity gap.
""")

st.markdown("---")
st.caption(f"Datos: Yahoo Finance · {len(chains)} vencimientos · "
           f"r={risk_free:.3f} q={div_yield:.3f} · "
           f"IV calculada con Brent's method (BS). "
           f"Para data en tiempo real profesional, usar Polygon/Tradier/CBOE — ver README.")

# Auto-refresh
if auto_refresh:
    time.sleep(300)
    st.rerun()
