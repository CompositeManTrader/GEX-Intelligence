"""
data_layer.py
─────────────────────────────────────────────────────────────
Capa de datos: descarga de cadenas de opciones con estrategia
anti-rate-limit. Soporta múltiples fuentes con fallback:
  1) Yahoo Finance vía curl_cffi con TLS fingerprint Chrome124
  2) yfinance con backoff exponencial
  3) (Opcional) Polygon / Tradier / CBOE DataShop si se pasa API key

Filtros de calidad:
  - bid > 0 Y ask > 0 (mercado activo, NO lastPrice)
  - moneyness ∈ [0.70, 1.30]
  - spread < 50% del mid
  - OI ≥ 10
  - mid ≥ $0.05
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, date
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

log = logging.getLogger("gex_app.data")

_YF_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v7/finance/options/{ticker}",
    "https://query2.finance.yahoo.com/v7/finance/options/{ticker}",
)

def _get_cffi_session():
    try:
        from curl_cffi import requests as creq
        return creq.Session(impersonate="chrome124")
    except Exception as e:
        log.warning(f"curl_cffi no disponible: {e}")
        return None

def _clean_chain(df_raw: pd.DataFrame, spot: float,
                 min_oi: int = 10,
                 mny_lo: float = 0.70, mny_hi: float = 1.30,
                 max_spread_pct: float = 0.50) -> pd.DataFrame:
    df = df_raw.copy()
    for col in ["bid", "ask", "lastPrice", "openInterest", "volume", "strike", "impliedVolatility"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["strike"] > 0)]
    df["midPrice"] = 0.5 * (df["bid"] + df["ask"])
    df["moneyness"] = df["strike"] / spot
    df = df[df["moneyness"].between(mny_lo, mny_hi)]
    spread_pct = (df["ask"] - df["bid"]) / df["midPrice"].replace(0, np.nan)
    df = df[spread_pct < max_spread_pct]
    df = df[df["openInterest"] >= min_oi]
    df = df[df["midPrice"] >= 0.05]
    if "impliedVolatility" in df.columns:
        df = df.rename(columns={"impliedVolatility": "iv_yahoo"})
    return df.sort_values("strike").reset_index(drop=True)

def fetch_options_yahoo(ticker: str, n_exp: int = 6,
                        min_dte: int = 0) -> Tuple[Dict, Optional[float]]:
    """Retorna ({exp_str: {calls, puts, dte}}, spot)."""
    sess = _get_cffi_session()
    if sess is None:
        return fetch_options_yfinance(ticker, n_exp, min_dte)
    hdrs = {
        "Accept": "application/json,text/html,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
        "Origin":  "https://finance.yahoo.com",
    }
    delay = 0.8
    for ep_idx, ep in enumerate(_YF_ENDPOINTS):
        base = ep.format(ticker=ticker)
        try:
            r0 = sess.get(base, headers=hdrs, timeout=20)
            if r0.status_code == 429:
                time.sleep(3); continue
            r0.raise_for_status()
            root = r0.json()["optionChain"]["result"][0]
            spot = float(root["quote"].get("regularMarketPrice", 0))
            if not spot:
                raise ValueError("spot=0")
            timestamps = root.get("expirationDates", [])
            today = date.today()
            sel = sorted(
                [(ts, datetime.fromtimestamp(ts).date().strftime("%Y-%m-%d"),
                  (datetime.fromtimestamp(ts).date() - today).days)
                 for ts in timestamps
                 if (datetime.fromtimestamp(ts).date() - today).days >= min_dte],
                key=lambda x: x[2])[:n_exp]
            chains = {}
            for i, (ts, exp_str, dte) in enumerate(sel):
                time.sleep(delay)
                chain_ep = _YF_ENDPOINTS[i % len(_YF_ENDPOINTS)].format(ticker=ticker)
                try:
                    rx = sess.get(f"{chain_ep}?date={ts}", headers=hdrs, timeout=20)
                    if rx.status_code == 429:
                        delay = min(delay * 2, 5.0)
                        time.sleep(delay)
                        rx = sess.get(f"{chain_ep}?date={ts}", headers=hdrs, timeout=20)
                    rx.raise_for_status()
                    opts = rx.json()["optionChain"]["result"][0]["options"][0]
                    c_df = _clean_chain(pd.DataFrame(opts.get("calls", [])), spot)
                    p_df = _clean_chain(pd.DataFrame(opts.get("puts",  [])), spot)
                    if len(c_df) < 3 or len(p_df) < 3:
                        continue
                    chains[exp_str] = {"calls": c_df, "puts": p_df, "dte": max(dte, 1)}
                    delay = max(delay * 0.85, 0.8)
                except Exception as ex:
                    log.warning(f"chain {ticker} {exp_str}: {ex}")
            if chains:
                return chains, spot
        except Exception as e:
            log.warning(f"endpoint {ep_idx}: {e}")
            time.sleep(2)
    return fetch_options_yfinance(ticker, n_exp, min_dte)

def fetch_options_yfinance(ticker: str, n_exp: int = 6,
                           min_dte: int = 0) -> Tuple[Dict, Optional[float]]:
    try:
        import yfinance as yf
    except ImportError:
        return {}, None
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps: return {}, None
        hist = t.history(period="2d")
        spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if not spot: return {}, None
        today = date.today()
        valid = sorted(
            [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days)
             for e in exps
             if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= min_dte],
            key=lambda x: x[1])[:n_exp]
        chains = {}
        for exp_str, dte in valid:
            try:
                ch = t.option_chain(exp_str)
                chains[exp_str] = {
                    "calls": _clean_chain(ch.calls, spot),
                    "puts":  _clean_chain(ch.puts,  spot),
                    "dte":   max(dte, 1),
                }
                time.sleep(1.2)
            except Exception as ex:
                log.warning(f"yf chain {ticker} {exp_str}: {ex}")
        return chains, spot
    except Exception as e:
        log.error(f"fetch_options_yfinance: {e}")
        return {}, None

def fetch_options_polygon(ticker: str, api_key: str,
                          n_exp: int = 6) -> Tuple[Dict, Optional[float]]:
    """
    Placeholder: Polygon.io tiene endpoint /v3/snapshot/options/{underlying}
    que devuelve la cadena entera con IV, Greeks y OI calculados por ellos.
    Este método es preferido para producción (pro feeds).
    """
    raise NotImplementedError(
        "Polygon fetcher no incluido — ver sección 'Datos en tiempo real' "
        "del README. Requiere plan Options Advanced (~$199/mes)."
    )
