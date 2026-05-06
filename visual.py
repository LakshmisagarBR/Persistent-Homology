"""
visual.py — MODULE 3: Bloomberg Dark Static Visualization
1920×1080 PNG — Layout Type B: 3D persistence landscape + 4-panel right column.

Left (70%): 3D persistence landscape surface L(ε, t)
Right (4 stacked panels):
  R1 — Wasserstein distance time series W(t)
  R2 — Betti numbers β₀(t) and β₁(t) at fixed ε
  R3 — Birth-death persistence diagram (final time step)
  R4 — Euler characteristic χ(ε) at 3 key time points
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

from config import CONFIG, THEME, CMAP_TOPO, CMAP_WASS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"  [VISUAL] {msg}")


def _style_ax(ax, theme=THEME):
    """Apply Bloomberg Dark styling to a 2D axes."""
    ax.set_facecolor(theme["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(theme["SPINE"])
        sp.set_linewidth(0.5)
    ax.tick_params(colors=theme["TEXT_DIM"], labelsize=8,
                   direction="in", length=3)
    ax.yaxis.grid(True, color=theme["GRID"], linewidth=0.3, alpha=0.45)
    ax.xaxis.grid(False)


def _style_3d(ax, theme=THEME):
    """Apply Bloomberg Dark styling to a 3D axes."""
    pane_color = (0.02, 0.02, 0.02, 1.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.6)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.set_facecolor(theme["BG"])


def _double_glow_line_3d(ax, x, y, z, color, lw_outer=5.0, lw_inner=1.8,
                          alpha_outer=0.12, alpha_inner=0.90):
    """Draw a double-glow ridge line on a 3D axes."""
    ax.plot(x, y, z, color=color, lw=lw_outer, alpha=alpha_outer, zorder=20)
    ax.plot(x, y, z, color=color, lw=lw_inner, alpha=alpha_inner, zorder=21)


# ─────────────────────────────────────────────────────────────────────────────
# CRASH MARKER POSITIONS (derived from stress regimes in time_idx space)
# ─────────────────────────────────────────────────────────────────────────────

def _find_crash_time_indices(topo: dict, data: dict) -> list:
    """Map stress regime day-indices to subsampled time_idx positions."""
    stress    = data["stress"]       # list of (start, end, vmul)
    time_idx  = topo["time_idx"]    # subsampled day indices
    crash_t   = []
    for s_start, s_end, _ in stress:
        mid_day  = (s_start + s_end) // 2
        closest  = int(np.argmin(np.abs(time_idx - mid_day)))
        crash_t.append(closest)
    return crash_t


# ─────────────────────────────────────────────────────────────────────────────
# PANEL: 3D PERSISTENCE LANDSCAPE SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def _draw_surface(ax, topo: dict, data: dict):
    """Render the main 3D persistence landscape surface."""

    EPS  = topo["EPS_grid"]          # (N_land, N_t)
    T    = topo["T_grid"]            # (N_land, N_t)
    Z    = topo["landscape_norm"]    # (N_land, N_t) normalised to [0,1]
    land_eps = topo["land_eps"]
    N_t = topo["N_t"]

    z_floor = -0.05

    log("Rendering 3D persistence landscape surface...")

    # ── Dark panes ────────────────────────────────────────────────────────────
    _style_3d(ax)

    # ── Main surface ──────────────────────────────────────────────────────────
    ax.plot_surface(
        EPS, T, Z,
        cmap=CMAP_TOPO,
        alpha=0.92,
        rstride=1, cstride=1,
        edgecolor=(1.0, 0.30, 0.55, 0.15),
        linewidth=0.25,
        antialiased=True,
        zorder=1,
    )

    # ── Floor contour shadow ──────────────────────────────────────────────────
    ax.contourf(
        EPS, T, Z,
        zdir="z", offset=z_floor,
        cmap=CMAP_TOPO,
        alpha=0.30,
        levels=16,
    )

    # ── VIX proxy ridge at fixed ε ≈ 0.8 ─────────────────────────────────────
    eps_idx_08 = int(0.8 / CONFIG["EPS_MAX"] * topo["N_land"])
    eps_idx_08 = min(eps_idx_08, topo["N_land"] - 1)
    t_arr = np.arange(N_t)
    ridge_z = Z[eps_idx_08, :]
    ridge_eps = np.full(N_t, land_eps[eps_idx_08])
    _double_glow_line_3d(ax, ridge_eps, t_arr, ridge_z,
                         color=THEME["CYAN"], lw_outer=6.0, lw_inner=2.0)
    # End-point dot
    ax.scatter([ridge_eps[-1]], [t_arr[-1]], [ridge_z[-1]],
               s=38, color=THEME["YELLOW"],
               edgecolor="white", linewidth=0.5, zorder=30)

    # ── Wasserstein ridge at fixed ε ≈ 0.5 ────────────────────────────────────
    eps_idx_05 = int(0.5 / CONFIG["EPS_MAX"] * topo["N_land"])
    wass_norm = topo["wasserstein_smooth"]
    wass_norm = wass_norm / (wass_norm.max() + 1e-12)
    wass_ridge_eps = np.full(N_t, land_eps[eps_idx_05])
    _double_glow_line_3d(ax, wass_ridge_eps, t_arr, wass_norm,
                         color=THEME["YELLOW"], lw_outer=5.0, lw_inner=1.6)

    # ── Crash curtains (red vertical sheets at crash peaks) ───────────────────
    crash_t_idx = _find_crash_time_indices(topo, data)
    for j_crash in crash_t_idx:
        # Vertical red lines from surface to floor
        for eps_val in np.linspace(land_eps[0], land_eps[-1], 12):
            e_idx = int(eps_val / CONFIG["EPS_MAX"] * (topo["N_land"] - 1))
            ax.plot([eps_val, eps_val], [j_crash, j_crash],
                    [z_floor, Z[e_idx, j_crash]],
                    color=THEME["RED"], alpha=0.18, lw=0.9)

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlabel("FILTRATION  ε", fontsize=10, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14,
                  fontfamily=THEME["FONT"])
    ax.set_ylabel("TIME  (rolling windows)", fontsize=10, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14,
                  fontfamily=THEME["FONT"])
    ax.set_zlabel(r"$\lambda_1(\varepsilon, t)$", fontsize=11,
                  color=THEME["TEXT_DIM"], labelpad=12,
                  fontfamily=THEME["FONT"])
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=7)
    ax.set_box_aspect(CONFIG["BOX_ASPECT"])
    ax.view_init(elev=CONFIG["ELEV_DEFAULT"], azim=CONFIG["AZIM_DEFAULT"])
    ax.set_zlim(z_floor, 1.05)

    # ── Legend annotation ─────────────────────────────────────────────────────
    ax.text2D(0.02, 0.97, "— CYAN: Vol proxy ridge (ε=0.80)",
              transform=ax.transAxes,
              color=THEME["CYAN"], fontsize=7.5, fontfamily=THEME["FONT"])
    ax.text2D(0.02, 0.94, "— YELLOW: Wasserstein ridge (ε=0.50)",
              transform=ax.transAxes,
              color=THEME["YELLOW"], fontsize=7.5, fontfamily=THEME["FONT"])
    ax.text2D(0.02, 0.91, "— RED: Crash regime markers",
              transform=ax.transAxes,
              color=THEME["RED"], fontsize=7.5, fontfamily=THEME["FONT"])


# ─────────────────────────────────────────────────────────────────────────────
# PANEL R1: WASSERSTEIN DISTANCE TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def _draw_wasserstein(ax, topo: dict, data: dict):
    """Wasserstein distance W(t) — topological change over time."""
    wass = topo["wasserstein_smooth"]
    t    = np.arange(len(wass))
    wmax = wass.max() + 1e-12

    _style_ax(ax)

    # Fill under line
    ax.fill_between(t, 0, wass, color=THEME["BLUE"], alpha=0.18)
    ax.plot(t, wass, color=THEME["CYAN"], lw=1.5, alpha=0.90)

    # Crash spike highlights
    crash_t_idx = _find_crash_time_indices(topo, data)
    for j_crash in crash_t_idx:
        ax.axvline(j_crash, color=THEME["RED"], lw=1.2, alpha=0.60, ls="--")
        ax.scatter([j_crash], [wass[j_crash]],
                   s=22, color=THEME["RED"], edgecolor="white",
                   linewidth=0.4, zorder=10)

    ax.set_xlim(0, len(wass) - 1)
    ax.set_ylim(0, wmax * 1.15)
    ax.set_ylabel("W(t)", color=THEME["TEXT_DIM"], fontsize=8,
                  fontfamily=THEME["FONT"])
    ax.set_title("WASSERSTEIN DISTANCE  ─  Topological Regime Change",
                 color=THEME["ORANGE"], fontsize=8, fontweight="bold",
                 fontfamily=THEME["FONT"], loc="left", pad=4)
    ax.tick_params(labelbottom=False)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL R2: BETTI NUMBERS β₀ AND β₁
# ─────────────────────────────────────────────────────────────────────────────

def _draw_betti(ax, topo: dict, data: dict):
    """Betti number time series at fixed ε = 0.8."""
    eps_vals = topo["eps_vals"]
    eps_target = 0.8
    eps_idx  = int(np.argmin(np.abs(eps_vals - eps_target)))

    beta0_t  = topo["beta0_surface"][eps_idx, :]
    beta1_t  = topo["beta1_surface"][eps_idx, :]
    t        = np.arange(len(beta0_t))

    _style_ax(ax)

    ax2 = ax.twinx()
    ax2.set_facecolor(THEME["PANEL_BG"])
    ax2.tick_params(colors=THEME["TEXT_DIM"], labelsize=7, direction="in")

    ax.fill_between(t, 0, beta0_t, color=THEME["ORANGE"], alpha=0.15)
    ax.plot(t, beta0_t, color=THEME["ORANGE"], lw=1.5, label=r"$\beta_0$ (components)")

    ax2.fill_between(t, 0, beta1_t, color=THEME["MAGENTA"], alpha=0.15)
    ax2.plot(t, beta1_t, color=THEME["MAGENTA"], lw=1.5, label=r"$\beta_1$ (loops)", ls="--")

    crash_t_idx = _find_crash_time_indices(topo, data)
    for j_crash in crash_t_idx:
        ax.axvline(j_crash, color=THEME["RED"], lw=0.8, alpha=0.50, ls=":")

    ax.set_xlim(0, len(beta0_t) - 1)
    ax.set_ylabel(r"$\beta_0$", color=THEME["ORANGE"], fontsize=8,
                  fontfamily=THEME["FONT"])
    ax2.set_ylabel(r"$\beta_1$", color=THEME["MAGENTA"], fontsize=8,
                   fontfamily=THEME["FONT"])
    ax.set_title(r"BETTI NUMBERS  $\beta_0$ / $\beta_1$  at $\varepsilon$=0.80",
                 color=THEME["ORANGE"], fontsize=8, fontweight="bold",
                 fontfamily=THEME["FONT"], loc="left", pad=4)
    ax.tick_params(labelbottom=False)

    # Legends
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    leg = ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
                    fontsize=6.5, facecolor=THEME["BG"],
                    edgecolor=THEME["GRID"])
    for txt in leg.get_texts():
        txt.set_color(THEME["TEXT_DIM"])


# ─────────────────────────────────────────────────────────────────────────────
# PANEL R3: BIRTH-DEATH PERSISTENCE DIAGRAM (FINAL WINDOW)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_persistence_diagram(ax, topo: dict):
    """
    Scatter plot of (birth, death) pairs across all time windows.
    Points far from the diagonal = long-lived features = real market structure.
    Reconstructs approximate H₀ and H₁ birth-death pairs from the Betti
    surface data: transitions in β₀ mark H₀ deaths, transitions in β₁ mark
    H₁ births and deaths.
    """
    _style_ax(ax)

    eps_vals = topo["eps_vals"]
    eps_max  = eps_vals[-1]
    N_t      = topo["N_t"]

    births, deaths, pers_vals = [], [], []

    # ── Reconstruct H₀ pairs from β₀ surface ─────────────────────────────
    # Each drop in β₀(ε) at a given time means a component merged (died)
    beta0_sf = topo["beta0_surface"]   # (N_eps, N_t)
    sample_times = range(0, N_t, max(1, N_t // 25))
    for t_idx in sample_times:
        b0 = beta0_sf[:, t_idx]
        for k in range(1, len(b0)):
            if b0[k] < b0[k - 1]:
                n_merges = int(b0[k - 1] - b0[k])
                for _ in range(min(n_merges, 3)):
                    births.append(0.0)
                    deaths.append(float(eps_vals[k]))
                    pers_vals.append(float(eps_vals[k]))

    # ── Reconstruct H₁ pairs from β₁ surface ─────────────────────────────
    beta1_sf = topo["beta1_surface"]   # (N_eps, N_t)
    for t_idx in sample_times:
        b1 = beta1_sf[:, t_idx]
        # Detect rises (births) and falls (deaths) in β₁
        active_births = []
        for k in range(1, len(b1)):
            if b1[k] > b1[k - 1]:
                n_new = int(b1[k] - b1[k - 1])
                for _ in range(min(n_new, 3)):
                    active_births.append(float(eps_vals[k]))
            elif b1[k] < b1[k - 1] and active_births:
                n_die = int(b1[k - 1] - b1[k])
                for _ in range(min(n_die, len(active_births))):
                    b = active_births.pop(0)
                    d = float(eps_vals[k])
                    if d > b + 0.01:
                        births.append(b)
                        deaths.append(d)
                        pers_vals.append(d - b)
        # Remaining active loops die at eps_max
        for b in active_births:
            d = eps_max
            if d > b + 0.01:
                births.append(b)
                deaths.append(d)
                pers_vals.append(d - b)

    if not births:
        # Fallback: use top_persistence values
        top_pers = topo["top_persistence"]
        n_steps = len(top_pers)
        for idx in range(0, n_steps, max(1, n_steps // 40)):
            pers_arr = top_pers[idx]
            for k, p in enumerate(pers_arr):
                if p > 1e-4:
                    b = 0.1 + k * (eps_max * 0.15)
                    d = b + p * eps_max
                    births.append(b)
                    deaths.append(min(d, eps_max))
                    pers_vals.append(d - b)

    if not births:
        return

    births   = np.array(births)
    deaths   = np.array(deaths)
    pers_arr = np.array(pers_vals)
    # Normalise persistence for colormap
    p_max    = pers_arr.max() + 1e-12
    colors_z = np.clip(pers_arr / p_max, 0, 1)

    # Diagonal (y = x)
    diag_line = np.linspace(0, eps_max, 100)
    ax.plot(diag_line, diag_line, color=THEME["SPINE"],
            lw=1.0, ls="--", alpha=0.6)

    # H₀ points (near diagonal) vs H₁ points (far from diagonal)
    h0_mask = (births == 0.0)
    h1_mask = ~h0_mask

    if h0_mask.any():
        ax.scatter(births[h0_mask], deaths[h0_mask],
                   c=colors_z[h0_mask], cmap=CMAP_WASS,
                   s=14, alpha=0.55, edgecolors="none", zorder=4,
                   marker="s", label=r"$H_0$")
    if h1_mask.any():
        ax.scatter(births[h1_mask], deaths[h1_mask],
                   c=colors_z[h1_mask], cmap=CMAP_TOPO,
                   s=22, alpha=0.80, edgecolors="none", zorder=5,
                   marker="o", label=r"$H_1$")

    ax.set_xlim(-0.02, eps_max * 1.05)
    ax.set_ylim(-0.02, eps_max * 1.05)
    ax.set_xlabel("Birth  ε_b", color=THEME["TEXT_DIM"], fontsize=7,
                  fontfamily=THEME["FONT"])
    ax.set_ylabel("Death  ε_d", color=THEME["TEXT_DIM"], fontsize=7,
                  fontfamily=THEME["FONT"])
    ax.set_title("PERSISTENCE DIAGRAM  ─  All Windows Sampled",
                 color=THEME["ORANGE"], fontsize=8, fontweight="bold",
                 fontfamily=THEME["FONT"], loc="left", pad=4)

    # Legend
    leg = ax.legend(loc="lower right", fontsize=6.5,
                    facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for txt in leg.get_texts():
        txt.set_color(THEME["TEXT_DIM"])

    # Annotation
    ax.text(eps_max * 0.55, eps_max * 0.08,
            "← NOISE    STRUCTURE →",
            color=THEME["TEXT_DIM"], fontsize=6, alpha=0.7,
            fontfamily=THEME["FONT"])


# ─────────────────────────────────────────────────────────────────────────────
# PANEL R4: EULER CHARACTERISTIC χ(ε)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_euler(ax, topo: dict, data: dict):
    """
    Euler characteristic χ(ε) = β₀(ε) − β₁(ε) at 3 key times:
      calm period, stress peak, post-crash.
    """
    _style_ax(ax)

    eps_vals  = topo["eps_vals"]
    euler_sf  = topo["euler_surface"]
    N_t       = topo["N_t"]
    crash_t   = _find_crash_time_indices(topo, data)

    # Pick 3 representative time indices
    t_calm   = N_t // 6              # early calm period
    t_stress = crash_t[1] if len(crash_t) > 1 else N_t // 2  # peak stress
    t_post   = min(crash_t[-1] + 10, N_t - 1) if crash_t else N_t - 1

    label_style = dict(fontsize=6.5, fontfamily=THEME["FONT"])

    for t_idx, color, label in [
        (t_calm,   THEME["CYAN"],    "Calm"),
        (t_stress, THEME["MAGENTA"], "Stress peak"),
        (t_post,   THEME["YELLOW"],  "Post-crash"),
    ]:
        chi = euler_sf[:, t_idx]
        ax.plot(eps_vals, chi, color=color, lw=1.6, alpha=0.90, label=label)
        ax.fill_between(eps_vals, 0, chi, color=color, alpha=0.10)

    ax.axhline(0, color=THEME["SPINE"], lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Filtration  ε", color=THEME["TEXT_DIM"], fontsize=7,
                  fontfamily=THEME["FONT"])
    ax.set_ylabel("χ(ε)", color=THEME["TEXT_DIM"], fontsize=7,
                  fontfamily=THEME["FONT"])
    ax.set_title(r"EULER CHARACTERISTIC  $\chi(\varepsilon) = \beta_0 - \beta_1$",
                 color=THEME["ORANGE"], fontsize=8, fontweight="bold",
                 fontfamily=THEME["FONT"], loc="left", pad=4)
    ax.set_xlim(eps_vals[0], eps_vals[-1])

    leg = ax.legend(loc="upper right", fontsize=6.5,
                    facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for txt in leg.get_texts():
        txt.set_color(THEME["TEXT_DIM"])

# ─────────────────────────────────────────────────────────────────────────────
# TITLE BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def _draw_title_block(fig, topo: dict, data: dict):
    """Bloomberg Dark title block — main title, subtitle, HUD stats, watermark."""
    font = THEME["FONT"]

    # ── 1. Main title ─────────────────────────────────────────────────────────
    fig.text(0.50, 0.966,
             "PERSISTENT HOMOLOGY  │  TOPOLOGICAL FINGERPRINT OF MARKET CRASHES",
             ha="center", va="center",
             fontsize=24, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=font)

    # ── 2. Subtitle ───────────────────────────────────────────────────────────
    fig.text(0.50, 0.936,
             r"Persistence Landscape  $\lambda_1(\varepsilon, t)$  via Vietoris-Rips Filtration"
             f"     N={len(data['tickers'])} assets     Window={CONFIG['CORR_WINDOW']}d"
             f"     T={CONFIG['T_DAYS']}d",
             ha="center", va="center",
             fontsize=11, color=THEME["TEXT_DIM"], fontfamily=font)

    # ── 3. HUD stats ──────────────────────────────────────────────────────────
    ls_max   = topo["ls_max"]
    wass_max = topo["wass_max"]
    # Fraction of time in H₁-dominant phase (β₁ > 0 at ε=0.8)
    eps_vals = topo["eps_vals"]
    eps_idx  = int(np.argmin(np.abs(eps_vals - 0.8)))
    h1_frac  = float((topo["beta1_surface"][eps_idx, :] > 0).mean())

    fig.text(0.96, 0.900,
             f"Max Persistence  {ls_max:.4f}     "
             f"Peak Wasserstein  {wass_max:.4f}     "
             f"H₁-Dominant Phase  {h1_frac:.1%}",
             ha="right", va="center",
             fontsize=10.5, fontweight="bold",
             color=THEME["YELLOW"], fontfamily=font)

    # ── 4. Watermark ──────────────────────────────────────────────────────────
    fig.text(0.985, 0.012, THEME["WATERMARK"],
             ha="right", va="bottom",
             fontsize=10, color=THEME["TEXT_DIM"],
             fontfamily=font, alpha=0.60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_static(topo: dict, data: dict, config: dict = CONFIG) -> None:
    """
    Render the full 1920×1080 Bloomberg Dark static PNG.

    Layout Type B — Bloomberg Multi-Panel Dashboard:
      Left (3D landscape) : right column (4 stacked panels)
    """
    os.makedirs(config["OUT_DIR"], exist_ok=True)
    log("Building figure layout (1920×1080)...")

    fig = plt.figure(figsize=config["FIG_SIZE"], dpi=config["DPI"],
                     facecolor=THEME["BG"])

    # ── GridSpec — Layout Type B ───────────────────────────────────────────────
    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[2.4, 1],
        left=0.03, right=0.97,
        top=0.900, bottom=0.055,
        hspace=0.38, wspace=0.10,
    )

    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_r1 = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_r3 = fig.add_subplot(gs[2, 1])
    ax_r4 = fig.add_subplot(gs[3, 1])

    # ── Render all panels ─────────────────────────────────────────────────────
    log("Drawing 3D persistence landscape surface...")
    _draw_surface(ax_3d, topo, data)

    log("Drawing Wasserstein distance panel...")
    _draw_wasserstein(ax_r1, topo, data)

    log("Drawing Betti numbers panel...")
    _draw_betti(ax_r2, topo, data)

    log("Drawing persistence diagram panel...")
    _draw_persistence_diagram(ax_r3, topo)

    log("Drawing Euler characteristic panel...")
    _draw_euler(ax_r4, topo, data)

    # ── Title block ───────────────────────────────────────────────────────────
    log("Rendering title block...")
    _draw_title_block(fig, topo, data)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = config["STATIC_PNG"]
    fig.savefig(out_path, dpi=config["DPI"],
                facecolor=THEME["BG"], edgecolor="none",
                pad_inches=0.1)
    plt.close(fig)
    log(f"Static PNG saved → {out_path}")