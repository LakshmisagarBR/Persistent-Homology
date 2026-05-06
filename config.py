"""
config.py — Persistent Homology: Topological Fingerprint of Market Crashes
All constants, design system, colormaps, and tickers in one place.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ═══════════════════════════════════════════════════════════════════════════════
# BLOOMBERG DARK DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

THEME = {
    "BG":           "#000000",
    "PANEL_BG":     "#0a0a0a",
    "GRID":         "#1a1a1a",
    "SPINE":        "#333333",
    "TEXT":         "#ffffff",
    "TEXT_DIM":     "#aaaaaa",
    "ORANGE":       "#ff9500",
    "ORANGE_HOT":   "#ff6b00",
    "CYAN":         "#00f2ff",
    "YELLOW":       "#ffd400",
    "GREEN":        "#00ff41",
    "RED":          "#ff3050",
    "MAGENTA":      "#ff1493",
    "PINK":         "#ff2a9e",
    "BLUE":         "#00bfff",
    "PURPLE":       "#9932cc",
    "FONT":         "DejaVu Sans",
    "WATERMARK":    "@Laksh",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM COLORMAPS
# ═══════════════════════════════════════════════════════════════════════════════

# Main persistence landscape surface — encodes the topological story:
#   Low Z  (calm)       →  void blacks / deep navy
#   Mid Z  (tension)    →  dark indigo → deep purple → magenta  (H₁ emerging)
#   High Z (pre-crash)  →  orange → yellow → white (maximum persistence)
CMAP_TOPO = LinearSegmentedColormap.from_list(
    "persistent_homology",
    [
        "#000000",   # 0.00 — void (no topological structure)
        "#000828",   # 0.10 — deep space blue
        "#001a4d",   # 0.20 — dark navy
        "#0a0040",   # 0.30 — dark indigo (H₀ dominant)
        "#4a0080",   # 0.42 — deep purple (transition)
        "#cc00cc",   # 0.55 — hot purple (H₁ emerging)
        "#ff1493",   # 0.65 — magenta (H₁ dominant, pre-crash)
        "#ff6b00",   # 0.78 — orange-hot (high persistence)
        "#ff9500",   # 0.87 — orange (near-crash)
        "#ffd400",   # 0.94 — yellow (crisis)
        "#ffffff",   # 1.00 — white (maximum persistence)
    ],
    N=512,
)

# Wasserstein ridge colormap — topological change indicator
CMAP_WASS = LinearSegmentedColormap.from_list(
    "wasserstein",
    ["#000000", "#001a4d", "#00bfff", "#ffffff"],
    N=256,
)

# ═══════════════════════════════════════════════════════════════════════════════
# S&P 500 UNIVERSE — 30 tickers across 6 sectors (5 per sector)
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    # Financials
    "JPM",  "BAC",  "GS",   "MS",    "BLK",
    # Healthcare
    "JNJ",  "UNH",  "PFE",  "ABBV",  "MRK",
    # Energy
    "XOM",  "CVX",  "COP",  "SLB",   "EOG",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD",   "NKE",   "MCD",
    # Industrials
    "CAT",  "DE",   "RTX",  "UPS",   "BA",
]

SECTORS = {
    "Technology":              TICKERS[0:5],
    "Financials":              TICKERS[5:10],
    "Healthcare":              TICKERS[10:15],
    "Energy":                  TICKERS[15:20],
    "Consumer Discretionary":  TICKERS[20:25],
    "Industrials":             TICKERS[25:30],
}

SECTOR_COLORS = {
    "Technology":             "#00f2ff",   # CYAN
    "Financials":             "#ff9500",   # ORANGE
    "Healthcare":             "#00ff41",   # GREEN
    "Energy":                 "#ffd400",   # YELLOW
    "Consumer Discretionary": "#ff1493",   # MAGENTA
    "Industrials":            "#00bfff",   # BLUE
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Data generation
    "T_DAYS":       504,     # 2 trading years
    "SEED":         42,

    # Engine (topology)
    "CORR_WINDOW":  60,      # rolling correlation window (days)
    "N_EPS":        80,      # filtration resolution (number of ε steps)
    "EPS_MAX":      1.80,    # max distance ≈ sqrt(2*(1-(-0.8)))
    "SUBSAMPLE":    75,      # number of time-points on 3D surface
    "N_LANDSCAPE":  80,      # ε evaluation points for landscape

    # Output
    "OUT_DIR":      "outputs",
    "STATIC_PNG":   "outputs/persistent_homology_markets.png",
    "ANIM_GIF":     "outputs/persistent_homology_animation.gif",
    "DPI":          100,
    "FIG_SIZE":     (19.2, 10.8),

    # Animation
    "GIF_DPI":      80,
    "GIF_FPS":      10,
    "GIF_FRAMES":   120,

    # 3D view defaults
    "ELEV_DEFAULT": 28,
    "AZIM_DEFAULT": -58,
    "BOX_ASPECT":   [1.8, 2.2, 0.85],
}