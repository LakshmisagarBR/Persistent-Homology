# Persistent Homology — Topological Fingerprint of Market Crashes

> **The shape of fear, rendered in three dimensions.**
> A from-scratch implementation of Topological Data Analysis (TDA) applied to S&P 500 correlation networks, visualised in the Bloomberg Dark aesthetic.

---

## What This Is

Most quantitative finance research looks at *statistics* of markets — returns, volatilities, correlations. This project looks at something fundamentally different: the **topological shape** of the market network — how many clusters exist, how many loops, and how they evolve as a continuous function of scale.

The central output is the **Persistence Landscape Surface** `λ₁(ε, t)` — a 3D surface where the X-axis is the filtration parameter ε (a "zoom level" on the market network), Y is calendar time, and Z is the dominant topological feature at that scale and time. During calm markets the surface is dark and nearly flat. Before a crash, dramatic ridges erupt as long-lived correlation loops form. During a crash, the surface collapses — all correlations merge into one undifferentiated blob of panic.

This specific visualisation — the full persistence landscape surface in 3D, animated through time — has not appeared in the academic literature on TDA in finance. The existing work (Gidea & Katz 2018, Gidea 2017, Yen & Cheong 2021) presents only static 2D plots, Betti number time series, or persistence diagram scatter plots.

---

## The Mathematics

### Step 1 — From Correlations to a Metric Space

Given the rolling 60-day correlation matrix ρᵢⱼ between 30 S&P 500 assets, we convert it to a proper metric using the **correlation distance**:

```
d(i, j) = √(2(1 − ρᵢⱼ))
```

This maps perfectly correlated assets (ρ = 1) to distance 0 and perfectly anti-correlated assets (ρ = −1) to distance 2. Every asset is now a point in a 30-dimensional metric space, and the market network has a well-defined geometry.

### Step 2 — Vietoris-Rips Filtration

Rather than picking one threshold and asking "which assets are connected?", we ask that question at *every* threshold simultaneously. For each ε ∈ [0, 1.8], we build the **Vietoris-Rips complex**: add an edge between assets i and j if d(i,j) ≤ ε, and add a triangle (i,j,k) if all three pairwise distances are ≤ ε. As ε grows from 0 to 2, the complex evolves from N isolated points to one fully-connected clique.

### Step 3 — Betti Numbers

At each threshold ε, we compute:

- **β₀(ε)** — the number of connected components (market clusters). Starts at N=30 (all isolated), ends at 1 (single market).
- **β₁(ε)** — the number of independent loops (correlation echo-chambers). Computed exactly via the Euler formula for graphs: β₁ = #edges − #nodes + β₀.

A spike in β₁ before a crash means the market has formed closed feedback circuits — assets are reinforcing each other's moves in closed loops of panic.

### Step 4 — Persistence Pairs

For each topological feature, we track exactly when it was **born** (appeared) and **died** (merged or was filled in):

**H₀ pairs** are found via Kruskal's Minimum Spanning Tree. Each MST edge of weight w corresponds to two clusters merging at filtration level w — that feature was born at ε=0 and died at ε=w. Persistence = w.

**H₁ pairs** are found via cycle detection. Each non-MST edge (i,j) of weight w_birth closes a 1-cycle — a loop is born at that threshold. The loop dies when the enclosing triangle is filled in, at:

```
death = max(w_birth,  min over all k of  max(d[i,k], d[j,k]))
```

This guarantees death ≥ birth (positive persistence) and is the correct formula for Vietoris-Rips complexes. Long-lived H₁ pairs (high persistence = death − birth) are real market structure. Short-lived ones are noise.

### Step 5 — Persistence Landscape

The persistence landscape converts a set of birth-death pairs into a *function* amenable to statistical analysis (Bubenik 2015). For each pair (b, d) we define a tent function:

```
Λ(b,d)(ε) = max(0, min(ε − b, d − ε))
```

The dominant landscape λ₁(ε) = the maximum of all tent functions at each ε. We compute this for every rolling 60-day window, producing the **landscape surface** `λ₁(ε, t)` — the main 3D visual output.

### Step 6 — Wasserstein Distance

Between consecutive time windows, we compute the L2 distance between the β₁(ε) curves:

```
W(t) ≈ ||β₁(ε, t) − β₁(ε, t−1)||₂
```

A spike in W(t) means the loop-structure of the market changed shape abruptly — a topological regime change that often leads price dislocations by several days.

---

## Project Structure

The pipeline follows the canonical three-module architecture:

```
persistent_homology/
├── config.py     — Design system (Bloomberg Dark), colormaps, tickers, CONFIG dict
├── data.py       — MODULE 1: Calibrated synthetic S&P 500 returns (GARCH + stress regimes)
├── engine.py     — MODULE 2: Vietoris-Rips engine (union-find, MST, landscape)
├── visual.py     — MODULE 3: 1920×1080 static PNG renderer
├── animate.py    — MODULE 4: 120-frame animated GIF renderer
├── main.py       — Orchestrator chaining all modules
└── outputs/
    ├── persistent_homology_markets.png     ← static output
    └── persistent_homology_animation.gif   ← animated output
```

Each module is self-contained and can be imported independently. The pipeline takes approximately 3–4 minutes to run end-to-end (dominated by the 120-frame GIF render).

---

## Visual Outputs

### Static PNG — 1920×1080

The static image uses **Layout Type B** (Bloomberg Multi-Panel Dashboard):

The left panel (70%) renders the 3D persistence landscape surface with full CMAP_TOPO colormap encoding — void blacks for calm periods, deep purple/magenta where H₁ loops dominate pre-crash, orange/yellow/white at peak persistence. A cyan double-glow ridge traces the volatility proxy at ε=0.8 across time. Red vertical curtains mark the three embedded crash regimes.

The right column shows four stacked 2D panels: the Wasserstein distance time series W(t) with crash spikes highlighted in red; Betti number curves β₀(t) and β₁(t) on dual axes at fixed ε=0.8; a birth-death persistence diagram sampled across all windows; and the Euler characteristic χ(ε) = β₀ − β₁ at three representative market phases (calm, stress peak, post-crash).

### Animated GIF — 120 frames at 10fps

The animation has three distinct phases:

**Phase 1 — REVEAL (frames 1–40):** The landscape surface sweeps in left to right as the time axis builds with quintic easing. The camera rises from a near-flat angle (elev=6°) to the default view (elev=28°), with a gentle azimuth drift. As the surface builds through each crash regime, the peaks erupt dramatically before collapsing — the most visually compelling moment in the animation.

**Phase 2 — HOLD (frames 41–60):** The full surface is visible. The camera breathes on a slow sine wave (±3° elevation, ±4° azimuth), giving the viewer time to absorb the surface structure. All HUD statistics are visible.

**Phase 3 — ORBIT (frames 61–120):** A full 360° azimuth rotation with a sinusoidal elevation profile (28° + 22°·sin(π·t·1.8)). At different viewing angles, distinct structures become visible: the time-evolution of crash peaks from the front, the ε-axis staircase structure of H₀ mergers from the side, and the Wasserstein ridge from behind.

A progress bar at the bottom fills with the phase colour (cyan/yellow/magenta) and shows phase markers.

---

## Data Generation

The project uses calibrated synthetic data rather than live market data, making it fully reproducible. The data generation pipeline:

**Correlation structure** is built from historically-grounded intra/inter sector correlations across 6 sectors (Technology, Financials, Healthcare, Energy, Consumer Discretionary, Industrials), with intra-sector correlations of 0.55–0.74 and inter-sector of 0.15–0.45 — matching empirical S&P 500 structure.

**GARCH volatility clustering** is implemented via exponential smoothing of squared shocks (α=0.08, β=0.90), producing realistic volatility persistence without requiring external packages.

**Three stress regimes** are embedded at days 80–115 (3.2× vol), 220–265 (4.5× vol), and 390–420 (3.8× vol). During each regime, correlations ramp toward a "panic matrix" (all ρ ≈ 0.88) via smooth interpolation — this is what creates the topological crash signatures in the landscape surface.

---

## Design System

The entire visual output conforms to the **Bloomberg Dark** aesthetic — a near-pitch-black terminal look with electric neon data at high contrast. Key elements:

The background is void black (`#000000`). The main surface uses `CMAP_TOPO`, a custom 11-stop colormap encoding the topological story from void → deep navy → dark indigo → deep purple → hot magenta (H₁ dominant) → orange-hot → orange → yellow → white at maximum persistence. Ridge lines use the signature double-glow technique: a thick outer trace at low alpha creates a diffuse halo, a thin inner trace at full alpha carries the information. Every panel background is `#0a0a0a`, grids are `#1a1a1a` at 0.3–0.5 linewidth, spines are `#333333`.

The title block follows the four-element mandatory structure: ALL-CAPS orange main title at 24pt bold; LaTeX subtitle at 11pt in `#aaaaaa`; HUD stats in yellow at 10.5pt bold in the top right; watermark `@Laksh` at alpha=0.60 in the bottom right.

---

## Installation & Usage

```bash
# Clone and enter
git clone https://github.com/your-username/persistent-homology-markets.git
cd persistent-homology-markets

# Install dependencies (Python 3.9+)
pip install numpy scipy matplotlib pillow

# Run the full pipeline
python main.py

# Run without animation (faster iteration)
python main.py --no-anim
```

The pipeline prints timestamped progress for each module and reports total runtime on completion. On a standard machine, the static PNG takes under 10 seconds. The 120-frame animation takes 3–4 minutes.

---

## Dependencies

This project is intentionally minimal — no full TDA libraries (gudhi, ripser, scikit-tda). Every topological primitive is implemented from scratch using only:

- **numpy** — array operations, Cholesky simulation, meshgrid
- **scipy** — `uniform_filter1d` for Wasserstein smoothing; `linalg.eigh` for nearest-PD projection
- **matplotlib** — 3D surface rendering (Agg backend), animation frame generation
- **Pillow (PIL)** — GIF encoding from rendered frame sequence

The from-scratch implementation serves two purposes: the project is fully self-contained and educational, and avoiding library overhead keeps the engine fast enough to run in under 3 seconds for 75 rolling windows.

---

## References

- Bubenik, P. (2015). *Statistical Topological Data Analysis using Persistence Landscapes.* Journal of Machine Learning Research.
- Gidea, M. & Katz, Y. (2018). *Topological Data Analysis of Financial Time Series.* Physica A.
- Gidea, M. (2017). *Topology Data Analysis of Critical Transitions in Financial Networks.* arXiv.
- Yen, P.T.W. & Cheong, S.A. (2021). *Using Topological Data Analysis (TDA) and Persistent Homology to Analyze the Stock Markets in Singapore and Taiwan.* Frontiers in Physics.

---

*@Laksh*
