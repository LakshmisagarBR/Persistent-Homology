"""
data.py — MODULE 1: Market Data Generation
Produces calibrated synthetic S&P 500 returns using Cholesky-correlated GBM
with GARCH-style volatility clustering and three embedded stress regimes.

Stress regimes create the topological crash signatures:
  • During stress: correlations spike toward 1 → β₀ collapses → β₁ erupts → collapses
  • Persistence landscape shows dramatic spike-then-collapse (the crash fingerprint)
"""

import numpy as np
import pandas as pd
from config import CONFIG, TICKERS, SECTORS


def log(msg: str) -> None:
    import time
    print(f"  [DATA] {msg}")


def _build_base_correlation(tickers: list, sectors: dict) -> np.ndarray:
    """
    Build historically-grounded intra/inter sector correlation matrix.
    Intra-sector correlations: 0.55–0.70 (realistic S&P 500 peers).
    Inter-sector correlations: 0.20–0.40 (diversification).
    """
    n = len(tickers)
    C = np.eye(n)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    # Map each ticker → sector name
    ticker_sector = {}
    for sector_name, members in sectors.items():
        for t in members:
            ticker_sector[t] = sector_name

    # Intra-sector base correlations (historical averages)
    intra_base = {
        "Technology":             0.68,
        "Financials":             0.72,
        "Healthcare":             0.58,
        "Energy":                 0.74,
        "Consumer Discretionary": 0.62,
        "Industrials":            0.65,
    }

    # Inter-sector correlation matrix
    inter_base = {
        ("Technology",   "Financials"):             0.38,
        ("Technology",   "Healthcare"):             0.32,
        ("Technology",   "Energy"):                 0.18,
        ("Technology",   "Consumer Discretionary"): 0.45,
        ("Technology",   "Industrials"):            0.36,
        ("Financials",   "Healthcare"):             0.30,
        ("Financials",   "Energy"):                 0.35,
        ("Financials",   "Consumer Discretionary"): 0.40,
        ("Financials",   "Industrials"):            0.42,
        ("Healthcare",   "Energy"):                 0.15,
        ("Healthcare",   "Consumer Discretionary"): 0.28,
        ("Healthcare",   "Industrials"):            0.25,
        ("Energy",       "Consumer Discretionary"): 0.22,
        ("Energy",       "Industrials"):            0.38,
        ("Consumer Discretionary", "Industrials"):  0.45,
    }

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i >= j:
                continue
            si, sj = ticker_sector[ti], ticker_sector[tj]
            if si == sj:
                rho = intra_base[si]
                # Idiosyncratic noise within sector
                np.random.seed(hash(ti + tj) % (2**31))
                rho += np.random.uniform(-0.08, 0.08)
            else:
                key = (si, sj) if (si, sj) in inter_base else (sj, si)
                rho = inter_base.get(key, 0.30)
                np.random.seed(hash(ti + tj) % (2**31))
                rho += np.random.uniform(-0.06, 0.06)
            rho = np.clip(rho, 0.05, 0.95)
            C[i, j] = rho
            C[j, i] = rho

    # Ensure positive-definiteness via nearest PD
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-6)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C


def generate_data(config: dict = CONFIG, tickers: list = TICKERS,
                  sectors: dict = SECTORS):
    """
    Generate calibrated synthetic S&P 500 returns.

    Returns
    -------
    returns   : pd.DataFrame (T × N), daily log-returns
    dates     : pd.DatetimeIndex
    vol_proxy : np.ndarray (T,) — cross-sectional realized volatility
    tickers   : list[str]
    stress    : list of (start_idx, end_idx, vol_multiplier) tuples
    """
    np.random.seed(config["SEED"])
    T = config["T_DAYS"]
    n = len(tickers)

    log(f"Generating {T}-day × {n}-asset synthetic return series")

    # ── 1. Base correlation structure ─────────────────────────────────────────
    C_base = _build_base_correlation(tickers, sectors)
    L_base = np.linalg.cholesky(C_base)   # Cholesky factor

    # ── 2. Per-asset calibrated parameters ────────────────────────────────────
    # Annual drift and vol (historical S&P 500 style)
    mu_annual = np.array([
        0.22, 0.18, 0.35, 0.20, 0.25,   # Tech (high growth)
        0.12, 0.10, 0.15, 0.14, 0.16,   # Financials
        0.10, 0.12, 0.08, 0.11, 0.09,   # Healthcare (defensive)
        0.08, 0.07, 0.10, 0.12, 0.11,   # Energy (cyclical)
        0.16, 0.28, 0.13, 0.11, 0.14,   # Con. Disc.
        0.10, 0.09, 0.08, 0.10, 0.06,   # Industrials
    ])
    sig_annual = np.array([
        0.30, 0.26, 0.55, 0.25, 0.32,   # Tech (volatile)
        0.22, 0.24, 0.26, 0.28, 0.20,   # Financials
        0.18, 0.20, 0.22, 0.21, 0.19,   # Healthcare
        0.28, 0.26, 0.30, 0.34, 0.32,   # Energy
        0.28, 0.60, 0.20, 0.22, 0.18,   # Con. Disc.
        0.22, 0.24, 0.20, 0.22, 0.28,   # Industrials
    ])
    dt = 1 / 252
    mu_daily  = (mu_annual - 0.5 * sig_annual**2) * dt
    sig_daily = sig_annual * np.sqrt(dt)

    # ── 3. GARCH-style volatility clusters ────────────────────────────────────
    # Simple exponential smoothing of squared shocks
    vol_scale = np.ones((T, n))
    alpha_garch = 0.08
    beta_garch  = 0.90
    h = np.ones(n)   # conditional variance
    for t in range(1, T):
        eps_sq = np.random.standard_normal(n)**2
        h = alpha_garch * eps_sq + beta_garch * h
        vol_scale[t] = np.sqrt(np.clip(h, 0.5, 4.0))

    # ── 4. Stress regime definitions ──────────────────────────────────────────
    # Three crash episodes embedded in the 2-year period
    stress_regimes = [
        (80,  115, 3.2),   # Regime 1 — sharp flash crash (35 days, 3.2× vol)
        (220, 265, 4.5),   # Regime 2 — prolonged crisis (45 days, 4.5× vol)
        (390, 420, 3.8),   # Regime 3 — sudden shock    (30 days, 3.8× vol)
    ]

    # ── 5. Build stress correlation ramp ──────────────────────────────────────
    # During stress: correlations ramp toward a "panic" matrix (all ≈ 0.90)
    C_panic = np.full((n, n), 0.88)
    np.fill_diagonal(C_panic, 1.0)
    # Nearest PD
    eigvals, eigvecs = np.linalg.eigh(C_panic)
    eigvals = np.maximum(eigvals, 1e-5)
    C_panic = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(C_panic))
    C_panic = C_panic / np.outer(d, d)
    L_panic = np.linalg.cholesky(C_panic)

    # Stress weight at each time step (0 = calm, 1 = full panic)
    stress_weight = np.zeros(T)
    for s_start, s_end, _ in stress_regimes:
        length = s_end - s_start
        for t in range(s_start, min(s_end, T)):
            progress = (t - s_start) / length
            # Ramp up sharply, decay more slowly
            w = np.where(progress < 0.3,
                         progress / 0.3,
                         1.0 - (progress - 0.3) / 0.7)
            stress_weight[t] = float(np.clip(w, 0, 1))

    # ── 6. Simulate daily returns ─────────────────────────────────────────────
    returns = np.zeros((T, n))
    for t in range(T):
        sw = stress_weight[t]
        # Interpolate Cholesky between calm and panic
        # (blend by scalar on correlation — approximate but fast)
        L_t = (1 - sw) * L_base + sw * L_panic

        # Extra vol multiplier during stress
        extra_vol = 1.0
        for s_start, s_end, vmul in stress_regimes:
            if s_start <= t < s_end:
                progress = (t - s_start) / (s_end - s_start)
                w = np.where(progress < 0.3,
                             progress / 0.3,
                             1.0 - (progress - 0.3) / 0.7)
                extra_vol = 1.0 + w * (vmul - 1.0)
                break

        z = np.random.standard_normal(n)
        eps = L_t @ z                                     # correlated shocks
        eps_scaled = eps * vol_scale[t] * extra_vol       # apply GARCH + stress
        returns[t] = mu_daily + sig_daily * eps_scaled

    # ── 7. Clip extremes (fat-tail realism) ───────────────────────────────────
    returns = np.clip(returns, -0.15, 0.15)

    # ── 8. Build outputs ──────────────────────────────────────────────────────
    dates = pd.bdate_range("2022-01-03", periods=T)
    df_returns = pd.DataFrame(returns, index=dates, columns=tickers)
    vol_proxy  = np.std(returns, axis=1) * np.sqrt(252)  # ann. cross-sec. vol

    log(f"Returns shape: {df_returns.shape}")
    log(f"Stress regimes: {[(s, e) for s, e, _ in stress_regimes]}")
    log(f"Mean daily return: {returns.mean():.5f} | Mean daily vol: {returns.std():.5f}")

    return {
        "returns":    df_returns,
        "dates":      dates,
        "vol_proxy":  vol_proxy,
        "tickers":    tickers,
        "stress":     stress_regimes,
    }