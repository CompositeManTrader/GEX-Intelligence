# GEX Dashboard — Standalone

Clon funcional de [gexbot.com](https://gexbot.com) — análisis de Gamma Exposure para trading intraday en mesa de capitales y derivados.

## Estructura

```
gex_dashboard/
├── app.py          ← Streamlit UI (Bloomberg-dark theme)
├── gex_engine.py   ← Black-Scholes + Greeks + GEX/DEX/VEX/Charm/ZeroGamma/MaxPain
├── data_layer.py   ← Fetcher Yahoo con anti-rate-limit (curl_cffi Chrome124)
├── charts.py       ← Visualizaciones Plotly estilo gexbot
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Funcionalidades

- **GEX Profile** por strike (barras rojas calls / verdes puts)
- **Cumulative GEX** con Zero-Gamma visible
- **GEX by Expiry** (qué vencimiento concentra gamma)
- **DEX** (Delta Exposure) — dealer hedging direccional
- **Vanna & Charm** — flujos por IV y por decay
- **Open Interest** por strike
- **Key levels**: Call Wall, Put Wall, Zero-Gamma, Max Pain, Expected Move, PCR
- **Playbook intraday** generado automáticamente según régimen (positive/negative gamma)

## Datos

Por defecto usa Yahoo Finance (gratis, con rate-limits). Filtros de calidad estrictos:
- bid>0 AND ask>0 (nunca lastPrice)
- moneyness ±30%
- spread < 50% del mid
- OI ≥ 10
- mid ≥ $0.05

IV calculada desde cero con Brent's method (no se confía en la IV de Yahoo que es ruidosa).

## Producción

Para datos intraday en tiempo real, reemplazar `fetch_options_yahoo` por un fetcher de Polygon / Tradier / CBOE DataShop. Ver sección de costos en la conversación original.
