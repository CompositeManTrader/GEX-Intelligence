"""
data_layer.py
─────────────────────────────────────────────────────────────
Capa de datos con múltiples fuentes (fallback cascade):
  1) Cboe Delayed Quotes (15-min delay, GRATIS, sin auth)
     → Funciona desde Streamlit Cloud / AWS / GCP
     → Tickers: SPY, QQQ, AAPL... Índices: _SPX, _NDX, _RUT, _VIX
  2) curl_cffi + Yahoo (Chrome124 TLS fingerprint)
  3) yfinance con backoff exponencial

Filtros de calidad:
  - bid > 0 Y ask > 0
  - moneyness ∈ [0.70, 1.30]
  - spread < 50% del mid
  - OI ≥ 10
  - mid ≥ $0.05
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import json
import logging
import re
import time
from datetime import datetime, date
from typing import Dict, Tuple, Optional
import urllib.request
import pandas as pd
import numpy as np

log = logging.getLogger("gex_app.data")

# ═══════════════════════════════════════════════════════════════
# CBOE — FUENTE PRIMARIA (15-min delayed, gratis, sin auth)
# ═══════════════════════════════════════════════════════════════
_CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
# OCC21 option symbol: ROOT + YYMMDD + C/P + STRIKE*1000(8 digits)
_OCC21 = re.compile(r"^[\^_]?([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

def _parse_occ21(opt_sym: str) -> Optional[dict]:
    m = _OCC21.match(opt_sym)
    if not m:
        return None
    root, ymd, cp, strike_str = m.groups()
    try:
        exp = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "root": root,
        "expiry": exp,
        "side": "calls" if cp == "C" else "puts",
        "strike": int(strike_str) / 1000.0,
    }

def _cboe_symbol(ticker: str) -> str:
    """Índices Cboe necesitan prefijo _"""
    t = ticker.upper().lstrip("^_")
    if t in {"SPX", "NDX", "RUT", "VIX", "DJX", "OEX", "XSP"}:
        return f"_{t}"
    return t

def fetch_options_cboe(ticker: str, n_exp: int = 6,
                       min_dte: int = 0) -> Tuple[Dict, Optional[float]]:
    """Cboe public CDN, 15-min delayed, sin auth, sin rate-limit serio."""
    sym = _cboe_symbol(ticker)
    url = _CBOE_BASE.format(sym=sym)
    hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cboe.com/",
        "Origin":  "https://www.cboe.com",
    }
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning(f"Cboe fetch {ticker} failed: {e}")
        return {}, None

    data = raw.get("data", {}) or raw
    spot = data.get("current_price") or data.get("close") or data.get("last_trade_price")
    if not spot:
        spot = (data.get("quote") or {}).get("regularMarketPrice")
    try:
        spot = float(spot)
    except (TypeError, ValueError):
        log.warning(f"Cboe: spot no encontrado para {ticker}")
        return {}, None

    options = data.get("options") or []
    if not options:
        return {}, spot

    rows = []
    today = date.today()
    for opt in options:
        parsed = _parse_occ21(opt.get("option", ""))
        if not parsed:
            continue
        dte = (parsed["expiry"] - today).days
        if dte < min_dte:
            continue
        bid = float(opt.get("bid") or 0)
        ask = float(opt.get("ask") or 0)
        if bid <= 0 or ask <= 0:
            continue
        rows.append({
            "expiry":  parsed["expiry"].strftime("%Y-%m-%d"),
            "dte":     max(dte, 1),
            "side":    parsed["side"],
            "strike":  parsed["strike"],
            "bid":     bid,
            "ask":     ask,
            "midPrice": 0.5 * (bid + ask),
            "lastPrice": float(opt.get("last_trade_price") or 0),
            "openInterest": int(opt.get("open_interest") or 0),
            "volume":  int(opt.get("volume") or 0),
            "impliedVolatility": float(opt.get("iv") or 0),
            "delta":   float(opt.get("delta") or 0),
            "gamma":   float(opt.get("gamma") or 0),
            "theta":   float(opt.get("theta") or 0),
            "vega":    float(opt.get("vega") or 0),
        })
    if not rows:
        return {}, spot
    df_all = pd.DataFrame(rows)

    # Top n_exp vencimientos más cercanos
    expiries_sorted = sorted(df_all["expiry"].unique(),
                             key=lambda e: datetime.strptime(e, "%Y-%m-%d").date())
    expiries_sorted = expiries_sorted[:n_exp]
    df_all = df_all[df_all["expiry"].isin(expiries_sorted)]

    # Filtros de calidad
    df_all = df_all[df_all["strike"] > 0]
    df_all["moneyness"] = df_all["strike"] / spot
    df_all = df_all[df_all["moneyness"].between(0.70, 1.30)]
    spread_pct = (df_all["ask"] - df_all["bid"]) / df_all["midPrice"].replace(0, np.nan)
    df_all = df_all[spread_pct < 0.50]
    df_all = df_all[df_all["openInterest"] >= 10]
    df_all = df_all[df_all["midPrice"] >= 0.05]
    df_all = df_all.rename(columns={"impliedVolatility": "iv_cboe"})

    chains = {}
    for exp_str, df_exp in df_all.groupby("expiry"):
        dte = int(df_exp["dte"].iloc[0])
        calls = df_exp[df_exp["side"] == "calls"].drop(
            columns=["side", "expiry", "dte"]).sort_values("strike").reset_index(drop=True)
        puts = df_exp[df_exp["side"] == "puts"].drop(
            columns=["side", "expiry", "dte"]).sort_values("strike").reset_index(drop=True)
        if len(calls) < 3 or len(puts) < 3:
            continue
        chains[exp_str] = {"calls": calls, "puts": puts, "dte": dte}

    log.info(f"Cboe {ticker}: {len(chains)} chains · spot=${spot:.2f}")
    return chains, spot

# ═══════════════════════════════════════════════════════════════
# YAHOO — FALLBACK
# ═══════════════════════════════════════════════════════════════
_YF_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v7/finance/options/{ticker}",
    "https://query2.finance.yahoo.com/v7/finance/options/{ticker}",
)

def _get_cffi_session():
    try:
        from curl_cffi import requests as creq
        return creq.Session(impersonate="chrome124")
    except Exception:
        return None

def _clean_yahoo_chain(df_raw: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = df_raw.copy()
    for col in ["bid", "ask", "lastPrice", "openInterest", "volume", "strike", "impliedVolatility"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["strike"] > 0)]
    df["midPrice"] = 0.5 * (df["bid"] + df["ask"])
    df["moneyness"] = df["strike"] / spot
    df = df[df["moneyness"].between(0.70, 1.30)]
    spread_pct = (df["ask"] - df["bid"]) / df["midPrice"].replace(0, np.nan)
    df = df[spread_pct < 0.50]
    df = df[df["openInterest"] >= 10]
    df = df[df["midPrice"] >= 0.05]
    if "impliedVolatility" in df.columns:
        df = df.rename(columns={"impliedVolatility": "iv_yahoo"})
    return df.sort_values("strike").reset_index(drop=True)

def fetch_options_yahoo(ticker: str, n_exp: int = 6,
                        min_dte: int = 0) -> Tuple[Dict, Optional[float]]:
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
            if r0.status_code in (401, 403, 429):
                time.sleep(2); continue
            r0.raise_for_status()
            root = r0.json()["optionChain"]["result"][0]
            spot = float(root["quote"].get("regularMarketPrice", 0))
            if not spot: continue
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
                    if rx.status_code in (401, 403, 429):
                        delay = min(delay * 2, 5.0)
                        time.sleep(delay)
                        rx = sess.get(f"{chain_ep}?date={ts}", headers=hdrs, timeout=20)
                    rx.raise_for_status()
                    opts = rx.json()["optionChain"]["result"][0]["options"][0]
                    c_df = _clean_yahoo_chain(pd.DataFrame(opts.get("calls", [])), spot)
                    p_df = _clean_yahoo_chain(pd.DataFrame(opts.get("puts",  [])), spot)
                    if len(c_df) < 3 or len(p_df) < 3: continue
                    chains[exp_str] = {"calls": c_df, "puts": p_df, "dte": max(dte, 1)}
                    delay = max(delay * 0.85, 0.8)
                except Exception as ex:
                    log.warning(f"yahoo chain {ticker} {exp_str}: {ex}")
            if chains:
                return chains, spot
        except Exception as e:
            log.warning(f"yahoo endpoint {ep_idx}: {e}")
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
                    "calls": _clean_yahoo_chain(ch.calls, spot),
                    "puts":  _clean_yahoo_chain(ch.puts,  spot),
                    "dte":   max(dte, 1),
                }
                time.sleep(1.2)
            except Exception as ex:
                log.warning(f"yf chain {ticker} {exp_str}: {ex}")
        return chains, spot
    except Exception as e:
        log.error(f"fetch_options_yfinance: {e}")
        return {}, None

# ═══════════════════════════════════════════════════════════════
# FETCHER UNIFICADO — cascada Cboe → Yahoo
# ═══════════════════════════════════════════════════════════════
def fetch_options(ticker: str, n_exp: int = 6,
                  min_dte: int = 0,
                  source: str = "auto") -> Tuple[Dict, Optional[float], str]:
    """
    Retorna (chains, spot, source_used).
    source: "auto" | "cboe" | "yahoo"
    """
    if source in ("auto", "cboe"):
        chains, spot = fetch_options_cboe(ticker, n_exp, min_dte)
        if chains and spot:
            return chains, spot, "Cboe (15-min delayed)"
        if source == "cboe":
            return {}, None, "Cboe (sin datos)"

    if source in ("auto", "yahoo"):
        chains, spot = fetch_options_yahoo(ticker, n_exp, min_dte)
        if chains and spot:
            return chains, spot, "Yahoo Finance"

    return {}, None, "Sin datos disponibles"
