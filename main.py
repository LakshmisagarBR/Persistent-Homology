"""
main.py — Persistent Homology: Topological Fingerprint of Market Crashes
Orchestrator that chains all four modules in sequence.

Pipeline:
  MODULE 1 — data.py    : Generate calibrated synthetic S&P 500 returns
  MODULE 2 — engine.py  : Compute Vietoris-Rips topology across rolling windows
  MODULE 3 — visual.py  : Render 1920×1080 Bloomberg Dark static PNG
  MODULE 4 — animate.py : Render 120-frame Bloomberg Dark animated GIF
"""

import os
import sys
import time


BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║       PERSISTENT HOMOLOGY — TOPOLOGICAL FINGERPRINT OF MARKET CRASHES        ║
║       Vietoris-Rips Filtration  ·  Persistence Landscape  ·  Bloomberg Dark  ║ 
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def _section(title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def main(skip_animation: bool = False) -> None:
    print(BANNER)
    t0_total = time.time()

    # ── Ensure output directory exists ────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 1 — DATA
    # ══════════════════════════════════════════════════════════════════════════
    _section("MODULE 1 — DATA GENERATION")
    t0 = time.time()

    from config import CONFIG, TICKERS, SECTORS
    from data import generate_data

    data_bundle = generate_data(config=CONFIG, tickers=TICKERS, sectors=SECTORS)
    print(f"  ✓ Data generated in {time.time() - t0:.2f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 2 — ENGINE (Topology)
    # ══════════════════════════════════════════════════════════════════════════
    _section("MODULE 2 — TOPOLOGICAL ENGINE  (Vietoris-Rips · Persistence)")
    t0 = time.time()

    from engine import compute_topology

    topo_bundle = compute_topology(data_bundle, config=CONFIG)
    print(f"  ✓ Topology computed in {time.time() - t0:.2f}s")
    print(f"  ✓ Landscape surface shape:  {topo_bundle['landscape_surface'].shape}")
    print(f"  ✓ Max persistence value:    {topo_bundle['ls_max']:.6f}")
    print(f"  ✓ Peak Wasserstein dist:    {topo_bundle['wass_max']:.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 3 — STATIC VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    _section("MODULE 3 — STATIC VISUALIZATION  (1920×1080 Bloomberg Dark PNG)")
    t0 = time.time()

    from visual import render_static

    render_static(topo_bundle, data_bundle, config=CONFIG)
    print(f"  ✓ Static PNG rendered in {time.time() - t0:.2f}s")
    print(f"  ✓ Saved → {CONFIG['STATIC_PNG']}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODULE 4 — ANIMATION
    # ══════════════════════════════════════════════════════════════════════════
    if not skip_animation:
        _section("MODULE 4 — ANIMATION  (120-frame Bloomberg Dark GIF)")
        t0 = time.time()

        from animate import render_animation

        render_animation(topo_bundle, data_bundle, config=CONFIG)
        print(f"  ✓ Animation rendered in {time.time() - t0:.2f}s")
        print(f"  ✓ Saved → {CONFIG['ANIM_GIF']}")
    else:
        print("\n  [SKIPPED] Animation (pass skip_animation=False to enable)")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    t_total = time.time() - t0_total
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE                                                           ║
║  Total runtime : {t_total:>6.1f}s                                            ║  
║  Static PNG    : {CONFIG['STATIC_PNG']:<55}                                  ║
║  Animation GIF : {CONFIG['ANIM_GIF']:<55}                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    # Pass --no-anim flag to skip the GIF (much faster for iteration)
    skip = "--no-anim" in sys.argv
    main(skip_animation=skip)
