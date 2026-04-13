"""
gex_engine.py
─────────────────────────────────────────────────────────────
Motor cuantitativo para análisis de Gamma Exposure (GEX),
Vanna Exposure (VEX), Charm Exposure, DEX y métricas
derivadas al estilo de gexbot.com / SpotGamma / Menthor Q.

Convención dealer (market-maker) estándar de la industria:
- Dealers venden calls a retail     → short gamma en calls
- Dealers venden puts a retail      → short gamma en puts (antes)
  o long puts (cuando retail compra protección)

La convención más usada (y la que usa gexbot) es:
  dealer_gex_call = -OI × Gamma × S² × 100 × 0.01   (short calls)
  dealer_gex_put  = +OI × Gamma × S² × 100 × 0.01   (long puts
                    desde la óptica de que retail vende puts
                    cubiertos y compra puts de protección)

Net Dealer GEX = Σ puts_gex + Σ calls_gex (ya con signos)

Interpretación:
  Net GEX > 0  → régimen "positive gamma": dealers compran caídas,
                 venden rallies → volatilidad REALIZADA se suprime,
                 mercado ancla a ciertos strikes (pinning).
  Net GEX < 0  → régimen "negative gamma": dealers venden caídas,
                 compran rallies → volatilidad se amplifica, gaps.

Unidades: expresamos GEX en $M por cada movimiento de 1% del spot.
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, date
from scipy.stats import norm
from typing import Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# BLACK-SCHOLES GREEKS
# ═══════════════════════════════════════════════════════════════
SQRT_2PI = np.sqrt(2.0 * np.pi)

def _d1_d2(S, K, r, T, sigma, q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_gamma(S, K, r, T, sigma, q=0.0):
    """∂²V/∂S² — igual para call y put."""
    d1, _ = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return 0.0
    return np.exp(-q * T) * np.exp(-0.5 * d1 * d1) / (SQRT_2PI * S * sigma * np.sqrt(T))

def bs_delta(S, K, r, T, sigma, q=0.0, option_type="call"):
    d1, _ = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return 0.0
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)

def bs_vanna(S, K, r, T, sigma, q=0.0):
    """∂Δ/∂σ = ∂Vega/∂S — igual signo para call y put."""
    d1, d2 = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return 0.0
    return -np.exp(-q * T) * np.exp(-0.5 * d1 * d1) / SQRT_2PI * d2 / sigma

def bs_charm(S, K, r, T, sigma, q=0.0, option_type="call"):
    """∂Δ/∂T — decay del delta con el paso del tiempo."""
    d1, d2 = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return 0.0
    nd1 = np.exp(-0.5 * d1 * d1) / SQRT_2PI
    term = nd1 * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == "call":
        return -np.exp(-q * T) * (q * norm.cdf(d1) - term)
    return -np.exp(-q * T) * (-q * norm.cdf(-d1) - term)

def bs_vega(S, K, r, T, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return 0.0
    return S * np.exp(-q * T) * np.exp(-0.5 * d1 * d1) / SQRT_2PI * np.sqrt(T)

# ═══════════════════════════════════════════════════════════════
# IV IMPLÍCITA (Brent) — no confiar en IV de proveedor
# ═══════════════════════════════════════════════════════════════
def _bs_price(S, K, r, T, sigma, q, option_type):
    d1, d2 = _d1_d2(S, K, r, T, sigma, q)
    if np.isnan(d1):
        return np.nan
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def implied_vol(S, K, r, T, price, option_type, q=0.0, tol=1e-6, max_iter=100):
    """Brent's method para IV. Devuelve NaN si no converge."""
    from scipy.optimize import brentq
    if price <= 0 or T <= 0:
        return np.nan
    # límites razonables de vol
    try:
        f = lambda s: _bs_price(S, K, r, T, s, q, option_type) - price
        # chequear cambio de signo
        f_lo, f_hi = f(1e-4), f(5.0)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(f, 1e-4, 5.0, xtol=tol, maxiter=max_iter)
    except Exception:
        return np.nan

# ═══════════════════════════════════════════════════════════════
# CHAIN PROCESSING — añadir greeks a la cadena
# ═══════════════════════════════════════════════════════════════
def enrich_chain_with_greeks(chains: Dict, spot: float,
                             r: float = 0.045, q: float = 0.013) -> Dict:
    """
    Toma chains = {exp_str: {"calls": df, "puts": df, "dte": n}}
    y añade: iv (calculada con BS), gamma, delta, vanna, charm, vega
    """
    out = {}
    for exp_str, ch in chains.items():
        dte = ch["dte"]
        T = max(dte, 1) / 365.0
        new_ch = {"dte": dte}
        for side in ("calls", "puts"):
            df = ch[side].copy()
            if df.empty:
                new_ch[side] = df
                continue
            opt_type = "call" if side == "calls" else "put"
            # IV si no viene ya buena
            if "iv" not in df.columns or df["iv"].isna().all():
                df["iv"] = df.apply(
                    lambda row: implied_vol(
                        spot, row["strike"], r, T, row["midPrice"], opt_type, q
                    ), axis=1
                )
            df["iv"] = df["iv"].ffill().bfill().fillna(0.20)
            df["iv"] = df["iv"].clip(0.05, 3.0)  # sanity
            df["gamma"] = df.apply(
                lambda r_: bs_gamma(spot, r_["strike"], r, T, r_["iv"], q), axis=1
            )
            df["delta"] = df.apply(
                lambda r_: bs_delta(spot, r_["strike"], r, T, r_["iv"], q, opt_type), axis=1
            )
            df["vanna"] = df.apply(
                lambda r_: bs_vanna(spot, r_["strike"], r, T, r_["iv"], q), axis=1
            )
            df["charm"] = df.apply(
                lambda r_: bs_charm(spot, r_["strike"], r, T, r_["iv"], q, opt_type), axis=1
            )
            df["vega"] = df.apply(
                lambda r_: bs_vega(spot, r_["strike"], r, T, r_["iv"], q), axis=1
            )
            new_ch[side] = df
        out[exp_str] = new_ch
    return out

# ═══════════════════════════════════════════════════════════════
# GEX PROFILE — PERFIL POR STRIKE
# ═══════════════════════════════════════════════════════════════
CONTRACT_MULT = 100.0  # SPY, SPX, QQQ estándar
PCT_MOVE = 0.01        # 1% de movimiento de spot para normalizar

def compute_gex_profile(chains: Dict, spot: float,
                        r: float = 0.045, q: float = 0.013,
                        expiries: Optional[list] = None) -> pd.DataFrame:
    """
    Perfil GEX por strike agregando los vencimientos seleccionados.
    Devuelve DataFrame con: strike, calls_gex, puts_gex, net_gex
    (todo en $M por movimiento de 1% del spot).
    """
    rows = []
    sel = expiries if expiries else list(chains.keys())
    for exp_str in sel:
        if exp_str not in chains:
            continue
        ch = chains[exp_str]
        for side in ("calls", "puts"):
            df = ch[side]
            if df.empty or "gamma" not in df.columns:
                continue
            # Convención dealer
            sign = -1.0 if side == "calls" else +1.0
            gex_usd = (sign * df["openInterest"].astype(float)
                       * df["gamma"] * spot**2
                       * CONTRACT_MULT * PCT_MOVE)
            for k, g in zip(df["strike"], gex_usd):
                rows.append({"strike": float(k), "side": side, "gex_usd": float(g)})
    if not rows:
        return pd.DataFrame(columns=["strike", "calls_gex", "puts_gex", "net_gex"])
    df_all = pd.DataFrame(rows)
    piv = (df_all.groupby(["strike", "side"])["gex_usd"]
           .sum().unstack(fill_value=0.0).reset_index())
    for c in ("calls", "puts"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["calls_gex"] = piv["calls"] / 1e6
    piv["puts_gex"] = piv["puts"] / 1e6
    piv["net_gex"] = piv["calls_gex"] + piv["puts_gex"]
    return piv[["strike", "calls_gex", "puts_gex", "net_gex"]].sort_values("strike").reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# MÉTRICAS DERIVADAS ESTILO GEXBOT
# ═══════════════════════════════════════════════════════════════
def compute_zero_gamma(gex_df: pd.DataFrame) -> Optional[float]:
    """
    Zero-Gamma / Gamma Flip: strike donde el GEX acumulado cruza cero.
    Es el nivel debajo del cual dealers están short gamma (régimen negativo).
    Interpolación lineal entre los dos strikes que rodean el cruce.
    """
    if gex_df.empty:
        return None
    df = gex_df.sort_values("strike").reset_index(drop=True)
    cum = df["net_gex"].cumsum().values
    strikes = df["strike"].values
    for i in range(1, len(cum)):
        if cum[i-1] * cum[i] < 0:
            # interpolación lineal
            x0, x1 = strikes[i-1], strikes[i]
            y0, y1 = cum[i-1], cum[i]
            if y1 == y0:
                return float(x0)
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None

def compute_walls(gex_df: pd.DataFrame, n: int = 3) -> Dict:
    """
    Call Walls (resistencia, mayor GEX positivo de calls)
    Put Walls (soporte, mayor GEX de puts).
    Devuelve top N de cada lado.
    """
    if gex_df.empty:
        return {"call_walls": [], "put_walls": []}
    calls = gex_df.nlargest(n, "calls_gex")[["strike", "calls_gex"]]
    puts = gex_df.nlargest(n, "puts_gex")[["strike", "puts_gex"]]
    return {
        "call_walls": calls.to_dict("records"),
        "put_walls":  puts.to_dict("records"),
    }

def compute_max_pain(chains: Dict, expiries: Optional[list] = None) -> Optional[float]:
    """
    Max Pain: strike donde el valor intrínseco total de las opciones
    (que expiran ITM) es MÍNIMO — punto de mayor pérdida para holders
    long de opciones.
    """
    rows = []
    sel = expiries if expiries else list(chains.keys())
    for exp_str in sel:
        if exp_str not in chains:
            continue
        ch = chains[exp_str]
        for side in ("calls", "puts"):
            df = ch[side]
            if df.empty: continue
            for _, r_ in df.iterrows():
                rows.append({"strike": r_["strike"],
                             "oi": r_["openInterest"],
                             "side": side})
    if not rows:
        return None
    df_all = pd.DataFrame(rows)
    strikes = np.sort(df_all["strike"].unique())
    total_pain = []
    for K_test in strikes:
        pain_c = ((K_test - df_all.loc[df_all.side == "calls", "strike"])
                  .clip(upper=0).abs() *
                  df_all.loc[df_all.side == "calls", "oi"]).sum()
        pain_p = ((df_all.loc[df_all.side == "puts", "strike"] - K_test)
                  .clip(lower=0) *
                  df_all.loc[df_all.side == "puts", "oi"]).sum()
        total_pain.append(pain_c + pain_p)
    idx = int(np.argmin(total_pain))
    return float(strikes[idx])

def compute_dex_profile(chains: Dict, spot: float,
                        expiries: Optional[list] = None) -> pd.DataFrame:
    """
    Delta Exposure (DEX) por strike. Dealer convention:
    dealer_delta_call = -OI × Δ × 100       (short calls)
    dealer_delta_put  = -OI × Δ_put × 100   (short puts también, negativo)
    Realmente para DEX se suele reportar como dealer notional delta.
    Positivo → dealers largos delta (cubren vendiendo si sube).
    """
    rows = []
    sel = expiries if expiries else list(chains.keys())
    for exp_str in sel:
        if exp_str not in chains:
            continue
        ch = chains[exp_str]
        for side in ("calls", "puts"):
            df = ch[side]
            if df.empty or "delta" not in df.columns: continue
            sign = -1.0  # dealers short ambas (convención)
            dex = sign * df["openInterest"].astype(float) * df["delta"] * spot * CONTRACT_MULT
            for k, d in zip(df["strike"], dex):
                rows.append({"strike": float(k), "side": side, "dex_usd": float(d)})
    if not rows:
        return pd.DataFrame(columns=["strike", "calls_dex", "puts_dex", "net_dex"])
    df_all = pd.DataFrame(rows)
    piv = (df_all.groupby(["strike", "side"])["dex_usd"]
           .sum().unstack(fill_value=0.0).reset_index())
    for c in ("calls", "puts"):
        if c not in piv.columns: piv[c] = 0.0
    piv["calls_dex"] = piv["calls"] / 1e6
    piv["puts_dex"] = piv["puts"] / 1e6
    piv["net_dex"] = piv["calls_dex"] + piv["puts_dex"]
    return piv[["strike", "calls_dex", "puts_dex", "net_dex"]].sort_values("strike").reset_index(drop=True)

def compute_vex_profile(chains: Dict, spot: float,
                        expiries: Optional[list] = None) -> pd.DataFrame:
    """
    Vanna Exposure (VEX): ∂Δ/∂σ × OI × multiplier × spot.
    Mide cuánto hedgeo delta generan los dealers por cambio de IV.
    Cuando IV baja (típico en rallies), vanna positiva → dealers
    compran más underlying → refuerza el rally (vanna rally).
    """
    rows = []
    sel = expiries if expiries else list(chains.keys())
    for exp_str in sel:
        if exp_str not in chains:
            continue
        ch = chains[exp_str]
        for side in ("calls", "puts"):
            df = ch[side]
            if df.empty or "vanna" not in df.columns: continue
            sign = -1.0 if side == "calls" else +1.0
            vex = (sign * df["openInterest"].astype(float) *
                   df["vanna"] * spot * CONTRACT_MULT)
            for k, v in zip(df["strike"], vex):
                rows.append({"strike": float(k), "side": side, "vex": float(v)})
    if not rows:
        return pd.DataFrame(columns=["strike", "calls_vex", "puts_vex", "net_vex"])
    df_all = pd.DataFrame(rows)
    piv = (df_all.groupby(["strike", "side"])["vex"]
           .sum().unstack(fill_value=0.0).reset_index())
    for c in ("calls", "puts"):
        if c not in piv.columns: piv[c] = 0.0
    piv["calls_vex"] = piv["calls"] / 1e6
    piv["puts_vex"] = piv["puts"] / 1e6
    piv["net_vex"] = piv["calls_vex"] + piv["puts_vex"]
    return piv[["strike", "calls_vex", "puts_vex", "net_vex"]].sort_values("strike").reset_index(drop=True)

def compute_charm_profile(chains: Dict, spot: float,
                          expiries: Optional[list] = None) -> pd.DataFrame:
    """
    Charm Exposure: ∂Δ/∂T — decay de delta. Responsable del
    "charm flow" clásico post-OPEX / intra-día cerca de expiración.
    """
    rows = []
    sel = expiries if expiries else list(chains.keys())
    for exp_str in sel:
        if exp_str not in chains:
            continue
        ch = chains[exp_str]
        for side in ("calls", "puts"):
            df = ch[side]
            if df.empty or "charm" not in df.columns: continue
            sign = -1.0 if side == "calls" else +1.0
            ch_ = (sign * df["openInterest"].astype(float) *
                   df["charm"] * spot * CONTRACT_MULT)
            for k, v in zip(df["strike"], ch_):
                rows.append({"strike": float(k), "side": side, "charm_usd": float(v)})
    if not rows:
        return pd.DataFrame(columns=["strike", "calls_charm", "puts_charm", "net_charm"])
    df_all = pd.DataFrame(rows)
    piv = (df_all.groupby(["strike", "side"])["charm_usd"]
           .sum().unstack(fill_value=0.0).reset_index())
    for c in ("calls", "puts"):
        if c not in piv.columns: piv[c] = 0.0
    piv["calls_charm"] = piv["calls"] / 1e6
    piv["puts_charm"] = piv["puts"] / 1e6
    piv["net_charm"] = piv["calls_charm"] + piv["puts_charm"]
    return piv[["strike", "calls_charm", "puts_charm", "net_charm"]].sort_values("strike").reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# EXPECTED MOVE (straddle ATM)
# ═══════════════════════════════════════════════════════════════
def compute_expected_move(chains: Dict, spot: float,
                          expiry: Optional[str] = None) -> Optional[float]:
    """
    Expected move = precio del straddle ATM × 0.85 (convención trader).
    Si no se especifica expiry, usa la más cercana.
    """
    if not chains:
        return None
    exp_str = expiry or list(chains.keys())[0]
    if exp_str not in chains:
        return None
    ch = chains[exp_str]
    calls, puts = ch["calls"], ch["puts"]
    if calls.empty or puts.empty:
        return None
    # strike ATM más cercano
    atm_c = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    atm_p = puts.iloc[(puts["strike"] - spot).abs().argsort().iloc[0]]
    straddle = atm_c["midPrice"] + atm_p["midPrice"]
    return float(straddle * 0.85)

# ═══════════════════════════════════════════════════════════════
# SUMMARY — DASHBOARD KPIs
# ═══════════════════════════════════════════════════════════════
def compute_summary(gex_df: pd.DataFrame, chains: Dict, spot: float,
                    expiries: Optional[list] = None) -> Dict:
    if gex_df.empty or spot <= 0:
        return {}
    total_gex = float(gex_df["net_gex"].sum())
    abs_gex = float(gex_df[["calls_gex", "puts_gex"]].abs().sum().sum())
    zero_g = compute_zero_gamma(gex_df)
    walls = compute_walls(gex_df, n=3)
    max_pain = compute_max_pain(chains, expiries)
    em = compute_expected_move(chains, spot)
    # Put/Call Ratio por OI
    call_oi = 0; put_oi = 0
    for exp, ch in chains.items():
        if expiries and exp not in expiries: continue
        call_oi += ch["calls"]["openInterest"].sum() if not ch["calls"].empty else 0
        put_oi += ch["puts"]["openInterest"].sum() if not ch["puts"].empty else 0
    pcr = put_oi / call_oi if call_oi > 0 else None
    return {
        "spot": round(spot, 2),
        "total_net_gex_m": round(total_gex, 2),
        "abs_gex_m":       round(abs_gex, 2),
        "zero_gamma":      round(zero_g, 2) if zero_g else None,
        "call_wall":       round(walls["call_walls"][0]["strike"], 2) if walls["call_walls"] else None,
        "put_wall":        round(walls["put_walls"][0]["strike"], 2)  if walls["put_walls"] else None,
        "call_walls":      walls["call_walls"],
        "put_walls":       walls["put_walls"],
        "max_pain":        max_pain,
        "expected_move":   round(em, 2) if em else None,
        "put_call_ratio":  round(pcr, 3) if pcr else None,
        "regime":          "POSITIVE" if total_gex > 0 else "NEGATIVE",
        "distance_to_flip": round(((zero_g or spot) / spot - 1) * 100, 2) if zero_g else None,
    }
