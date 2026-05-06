"""
animate.py — MODULE 4: Smooth 120-Frame Bloomberg Dark Animation

Three cinematic phases:
  Phase 1 — REVEAL (40 frames): Surface sweeps in L->R, quintic easing, camera rises.
  Phase 2 — HOLD   (20 frames): Full surface, camera breathes on sine wave.
  Phase 3 — ORBIT  (60 frames): Full 360deg rotation, elevation undulates.

Black background is composited at the frame level so GIF encoding stays correct.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os
import io
from PIL import Image

from config import CONFIG, THEME, CMAP_TOPO


def log(msg):
    print(f"  [ANIMATE] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# EASING
# ─────────────────────────────────────────────────────────────────────────────

def _quintic(t):
    return 6*t**5 - 15*t**4 + 10*t**3

def _cubic(t):
    return 3*t**2 - 2*t**3


# ─────────────────────────────────────────────────────────────────────────────
# FRAME SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def _build_schedule(topo, data, n_frames=120):
    N_REV, N_HLD, N_ORB = 40, 20, 60
    elev0, elev_def = 6.0, 28.0
    azim_def = -58.0

    crash_fracs = []
    for s, e, _ in data["stress"]:
        mid = (s + e) // 2
        ci  = int(np.argmin(np.abs(topo["time_idx"] - mid)))
        crash_fracs.append(ci / topo["N_t"])

    sched = []

    # Phase 1 — REVEAL
    for fi in range(N_REV):
        rf  = fi / (N_REV - 1)
        ef  = _quintic(rf)
        warp = 1.0
        for cf in crash_fracs:
            dist = abs(ef - cf)
            if dist < 0.08:
                warp *= 1.0 + 0.5 * (0.08 - dist) / 0.08
        tc = min(ef * warp, 1.0)
        sched.append(dict(
            frame_idx=fi, phase="REVEAL", phase_frac=rf,
            t_cutoff_frac=tc,
            elev=elev0 + (elev_def - elev0) * _cubic(rf),
            azim=azim_def + (-50 - azim_def) * _cubic(rf),
        ))

    # Phase 2 — HOLD
    for fi in range(N_HLD):
        pf = fi / max(N_HLD - 1, 1)
        sched.append(dict(
            frame_idx=N_REV + fi, phase="HOLD", phase_frac=pf,
            t_cutoff_frac=1.0,
            elev=elev_def + 3.0 * np.sin(2*np.pi*0.75*pf),
            azim=-50.0    + 4.0 * np.sin(2*np.pi*0.75*pf),
        ))

    # Phase 3 — ORBIT
    for fi in range(N_ORB):
        pf = fi / max(N_ORB - 1, 1)
        sched.append(dict(
            frame_idx=N_REV + N_HLD + fi, phase="ORBIT", phase_frac=pf,
            t_cutoff_frac=1.0,
            elev=float(elev_def + 22.0 * np.sin(np.pi * pf * 1.8)),
            azim=float(-58.0 + 360.0 * pf),
        ))

    return sched


# ─────────────────────────────────────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _style3d(ax):
    pane = (0.02, 0.02, 0.02, 1.0)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.35

def _style2d(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"])
        sp.set_linewidth(0.4)
    ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=7, direction="in", length=2)
    ax.yaxis.grid(True, color=THEME["GRID"], lw=0.25, alpha=0.4)
    ax.set_axisbelow(True)

def _style2d_twin(ax):
    """Style a twinx axes — must explicitly set facecolor to transparent
    so it doesn't cover the primary axes."""
    ax.set_facecolor((0, 0, 0, 0))   # transparent — primary sets the bg
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"])
        sp.set_linewidth(0.4)
    ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=6, direction="in", length=2)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FRAME RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame(fp, topo, data, config, crash_t_idx):
    t_frac  = fp["t_cutoff_frac"]
    elev    = fp["elev"]
    azim    = fp["azim"]
    phase   = fp["phase"]
    fi      = fp["frame_idx"]
    n_total = config["GIF_FRAMES"]

    N_t_full = topo["N_t"]
    N_land   = topo["N_land"]
    t_show   = max(2, int(np.ceil(t_frac * N_t_full)))

    land_eps = topo["land_eps"]
    eps_vals = topo["eps_vals"]
    EPS_full = topo["EPS_grid"]
    T_full   = topo["T_grid"]
    Z_full   = topo["landscape_norm"]

    EPS = EPS_full[:, :t_show]
    T   = T_full[:,  :t_show]
    Z   = Z_full[:,  :t_show]

    fig = plt.figure(figsize=config["FIG_SIZE"], dpi=config["GIF_DPI"],
                     facecolor=THEME["BG"])

    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[2.4, 1],
        left=0.03, right=0.97,
        top=0.900, bottom=0.085,
        hspace=0.44, wspace=0.12,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_r1 = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_r3 = fig.add_subplot(gs[2, 1])
    ax_r4 = fig.add_subplot(gs[3, 1])

    z_floor = -0.05

    # ── 3D surface ────────────────────────────────────────────────────────────
    _style3d(ax3d)
    ax3d.set_facecolor(THEME["BG"])

    if t_show >= 2:
        ax3d.plot_surface(
            EPS, T, Z, cmap=CMAP_TOPO,
            alpha=0.91, rstride=1, cstride=1,
            edgecolor=(1.0, 0.30, 0.55, 0.14), linewidth=0.22,
            antialiased=True, zorder=1,
        )
        ax3d.contourf(
            EPS, T, Z, zdir="z", offset=z_floor,
            cmap=CMAP_TOPO, alpha=0.28, levels=14,
        )
        # Cyan VIX ridge at eps=0.8
        ei08 = min(int(0.8 / config["EPS_MAX"] * N_land), N_land - 1)
        ta   = np.arange(t_show)
        re   = np.full(t_show, land_eps[ei08])
        rz   = Z_full[ei08, :t_show]
        ax3d.plot(re, ta, rz, color=THEME["CYAN"], lw=4.5, alpha=0.12, zorder=20)
        ax3d.plot(re, ta, rz, color=THEME["CYAN"], lw=1.7, alpha=0.90, zorder=21)

        # Red crash curtains
        for jc in crash_t_idx:
            if jc >= t_show:
                continue
            for ev in np.linspace(land_eps[0], land_eps[-1], 10):
                eidx = int(ev / config["EPS_MAX"] * (N_land - 1))
                ax3d.plot([ev, ev], [jc, jc], [z_floor, Z_full[eidx, jc]],
                          color=THEME["RED"], alpha=0.20, lw=0.85)

    ax3d.set_xlabel("FILTRATION  ε", fontsize=8, color=THEME["TEXT_DIM"],
                    labelpad=10, fontfamily=THEME["FONT"])
    ax3d.set_ylabel("TIME", fontsize=8, color=THEME["TEXT_DIM"],
                    labelpad=10, fontfamily=THEME["FONT"])
    ax3d.set_zlabel(r"$\lambda_1$", fontsize=9, color=THEME["TEXT_DIM"],
                    labelpad=8, fontfamily=THEME["FONT"])
    ax3d.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=6)
    ax3d.set_box_aspect(config["BOX_ASPECT"])
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_zlim(z_floor, 1.05)

    phase_col = {"REVEAL": THEME["CYAN"], "HOLD": THEME["YELLOW"], "ORBIT": THEME["MAGENTA"]}
    ax3d.text2D(0.02, 0.97, f"● {phase}",
                transform=ax3d.transAxes,
                color=phase_col[phase], fontsize=9, fontweight="bold",
                fontfamily=THEME["FONT"])

    # ── R1: Wasserstein ───────────────────────────────────────────────────────
    _style2d(ax_r1)
    wass      = topo["wasserstein_smooth"]
    wass_show = wass[:t_show]
    tw        = np.arange(len(wass_show))
    ax_r1.fill_between(tw, 0, wass_show, color=THEME["BLUE"], alpha=0.18)
    ax_r1.plot(tw, wass_show, color=THEME["CYAN"], lw=1.4, alpha=0.90)
    for jc in crash_t_idx:
        if jc < t_show:
            ax_r1.axvline(jc, color=THEME["RED"], lw=1.0, alpha=0.55, ls="--")
    ax_r1.set_xlim(0, N_t_full)
    ax_r1.set_ylim(0, max(wass.max() * 1.15, 0.01))
    ax_r1.set_title("WASSERSTEIN  W(t)", color=THEME["ORANGE"],
                    fontsize=7.5, fontweight="bold",
                    fontfamily=THEME["FONT"], loc="left", pad=3)
    ax_r1.tick_params(labelbottom=False)
    ax_r1.set_ylabel("W(t)", color=THEME["TEXT_DIM"], fontsize=7)

    # ── R2: Betti numbers ─────────────────────────────────────────────────────
    _style2d(ax_r2)
    ei08b   = int(np.argmin(np.abs(eps_vals - 0.8)))
    beta0_t = topo["beta0_surface"][ei08b, :t_show]
    beta1_t = topo["beta1_surface"][ei08b, :t_show]
    tb      = np.arange(t_show)

    ax_r2.fill_between(tb, 0, beta0_t, color=THEME["ORANGE"],  alpha=0.16)
    ax_r2.plot(tb, beta0_t, color=THEME["ORANGE"],  lw=1.4, label=r"$\beta_0$")
    ax_r2.set_xlim(0, N_t_full)
    ax_r2.set_ylim(0, max(topo["beta0_surface"][ei08b].max() * 1.1, 1))
    ax_r2.set_ylabel(r"$\beta_0$", color=THEME["ORANGE"], fontsize=7)
    ax_r2.tick_params(labelbottom=False)
    ax_r2.set_title(r"BETTI  $\beta_0$ / $\beta_1$  @ ε=0.80",
                    color=THEME["ORANGE"], fontsize=7.5, fontweight="bold",
                    fontfamily=THEME["FONT"], loc="left", pad=3)

    ax_r2b = ax_r2.twinx()
    _style2d_twin(ax_r2b)
    ax_r2b.fill_between(tb, 0, beta1_t, color=THEME["MAGENTA"], alpha=0.16)
    ax_r2b.plot(tb, beta1_t, color=THEME["MAGENTA"], lw=1.4, ls="--")
    ax_r2b.set_xlim(0, N_t_full)
    ax_r2b.set_ylim(0, max(topo["beta1_surface"][ei08b].max() * 1.1, 1))
    ax_r2b.set_ylabel(r"$\beta_1$", color=THEME["MAGENTA"], fontsize=7)

    # Live HUD
    b0_now = int(beta0_t[-1]) if len(beta0_t) else 0
    b1_now = int(beta1_t[-1]) if len(beta1_t) else 0
    ax_r2.text(0.97, 0.88, f"β₀={b0_now}  β₁={b1_now}",
               transform=ax_r2.transAxes, ha="right",
               color=THEME["YELLOW"], fontsize=7.5, fontweight="bold",
               fontfamily=THEME["FONT"])

    # ── R3: Euler characteristic ──────────────────────────────────────────────
    _style2d(ax_r3)
    t_now   = min(t_show - 1, topo["N_t"] - 1)
    chi_now = topo["euler_surface"][:, t_now]
    ax_r3.plot(eps_vals, chi_now, color=THEME["CYAN"], lw=1.5, alpha=0.90)
    ax_r3.fill_between(eps_vals, 0, chi_now, color=THEME["CYAN"], alpha=0.12)
    ax_r3.axhline(0, color=THEME["SPINE"], lw=0.7, ls="--", alpha=0.5)
    ax_r3.set_xlim(eps_vals[0], eps_vals[-1])
    chi_abs = np.abs(chi_now).max()
    ax_r3.set_ylim(-chi_abs * 1.1 - 1, chi_abs * 0.3 + 1)
    ax_r3.set_title(r"EULER  $\chi(\varepsilon)$  current",
                    color=THEME["ORANGE"], fontsize=7.5, fontweight="bold",
                    fontfamily=THEME["FONT"], loc="left", pad=3)
    ax_r3.set_xlabel("ε", color=THEME["TEXT_DIM"], fontsize=7,
                     fontfamily=THEME["FONT"])
    ax_r3.tick_params(labelbottom=True)

    # ── R4: Top H1 persistence bars ───────────────────────────────────────────
    _style2d(ax_r4)
    top_idx = min(t_now, len(topo["top_persistence"]) - 1)
    top5    = topo["top_persistence"][top_idx] if topo["top_persistence"] else np.zeros(5)
    bar_colors = [THEME["MAGENTA"], THEME["ORANGE_HOT"],
                  THEME["ORANGE"], THEME["YELLOW"], THEME["CYAN"]]
    ax_r4.barh(range(5), top5[::-1], color=bar_colors, alpha=0.82, height=0.55)
    ax_r4.set_yticks(range(5))
    ax_r4.set_yticklabels([f"H₁ #{5-k}" for k in range(5)],
                           color=THEME["TEXT_DIM"], fontsize=6.5)
    ax_r4.set_xlabel("Persistence  (d − b)", color=THEME["TEXT_DIM"],
                     fontsize=7, fontfamily=THEME["FONT"])
    ax_r4.set_title("TOP H₁ PERSISTENCE  ─ current window",
                    color=THEME["ORANGE"], fontsize=7.5, fontweight="bold",
                    fontfamily=THEME["FONT"], loc="left", pad=3)
    ax_r4.set_xlim(0, max(top5.max() * 1.2, 0.01))

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(0.50, 0.966,
             "PERSISTENT HOMOLOGY  │  TOPOLOGICAL FINGERPRINT OF MARKET CRASHES",
             ha="center", va="center",
             fontsize=18, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.937,
             r"Persistence Landscape  $\lambda_1(\varepsilon,t)$  via Vietoris-Rips Filtration"
             f"   Frame {fi+1}/{n_total}",
             ha="center", va="center",
             fontsize=9, color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    ls_max   = topo["ls_max"]
    wass_max = topo["wass_max"]
    ei_hud   = int(np.argmin(np.abs(eps_vals - 0.8)))
    h1_frac  = float((topo["beta1_surface"][ei_hud, :t_show] > 0).mean()) \
               if t_show > 0 else 0.0
    fig.text(0.96, 0.905,
             f"Max λ₁  {ls_max:.4f}     Peak W  {wass_max:.4f}     H₁ Phase  {h1_frac:.1%}",
             ha="right", va="center",
             fontsize=9, fontweight="bold",
             color=THEME["YELLOW"], fontfamily=THEME["FONT"])
    fig.text(0.985, 0.012, THEME["WATERMARK"],
             ha="right", va="bottom",
             fontsize=9, color=THEME["TEXT_DIM"],
             fontfamily=THEME["FONT"], alpha=0.60)

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar_y, bar_h  = 0.034, 0.013
    bar_l, bar_r  = 0.03, 0.97
    bar_w         = bar_r - bar_l
    prog          = (fi + 1) / n_total
    fill_col      = {"REVEAL": THEME["CYAN"],
                     "HOLD":   THEME["YELLOW"],
                     "ORBIT":  THEME["MAGENTA"]}[phase]

    # Background track
    bg_rect = mpatches.Rectangle(
        (bar_l, bar_y), bar_w, bar_h,
        transform=fig.transFigure, figure=fig,
        facecolor="#111111", edgecolor=THEME["SPINE"],
        linewidth=0.5, zorder=10, clip_on=False,
    )
    fig.add_artist(bg_rect)

    # Filled portion
    fill_rect = mpatches.Rectangle(
        (bar_l, bar_y), bar_w * prog, bar_h,
        transform=fig.transFigure, figure=fig,
        facecolor=fill_col, alpha=0.85,
        linewidth=0, zorder=11, clip_on=False,
    )
    fig.add_artist(fill_rect)

    # Phase labels above bar
    for mfrac, mlabel in [(40/n_total, "REVEAL"), (60/n_total, "HOLD"), (1.0, "ORBIT")]:
        fig.text(bar_l + bar_w * mfrac, bar_y + bar_h + 0.007, mlabel,
                 ha="center", va="bottom",
                 fontsize=5.5, color=THEME["TEXT_DIM"],
                 fontfamily=THEME["FONT"])

    # ── Render to PIL Image with guaranteed black background ──────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=config["GIF_DPI"],
                facecolor=THEME["BG"], bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Composite onto pure black BEFORE converting to RGB.
    # PIL's convert("RGB") maps transparency -> WHITE if done directly.
    img_raw  = Image.open(buf)
    img_rgba = img_raw.convert("RGBA")
    black_bg = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
    black_bg.alpha_composite(img_rgba)
    return black_bg.convert("RGB")   # fully opaque RGB, no white bleed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANIMATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_animation(topo, data, config=CONFIG):
    """Render the 120-frame Bloomberg Dark animated GIF."""
    os.makedirs(config["OUT_DIR"], exist_ok=True)
    n_frames = config["GIF_FRAMES"]
    fps      = config["GIF_FPS"]

    log(f"Building frame schedule ({n_frames} frames @ {fps}fps)...")
    schedule = _build_schedule(topo, data, n_frames)

    crash_t_idx = []
    for s, e, _ in data["stress"]:
        mid = (s + e) // 2
        ci  = int(np.argmin(np.abs(topo["time_idx"] - mid)))
        crash_t_idx.append(ci)

    log(f"Rendering {n_frames} frames...")
    frames = []
    for i, fp in enumerate(schedule):
        if i % 10 == 0 or i == n_frames - 1:
            log(f"  Frame {i+1:3d}/{n_frames}  phase={fp['phase']}"
                f"  elev={fp['elev']:.1f}  azim={fp['azim']:.1f}")
        frames.append(_render_frame(fp, topo, data, config, crash_t_idx))

    out_path  = config["ANIM_GIF"]
    frame_dur = int(1000 / fps)
    log(f"Encoding GIF -> {out_path}  ({len(frames)} frames, {frame_dur}ms/frame)...")

    # All frames are already clean RGB (black composited).
    # Let Pillow handle palette quantization internally — no manual .quantize().
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_dur,
        loop=0,
    )
    log(f"Animation saved -> {out_path}")
