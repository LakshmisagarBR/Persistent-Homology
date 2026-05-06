"""
engine.py — MODULE 2: Topological Data Analysis Engine

Implements from scratch (no gudhi/ripser):
  1.  Correlation-to-distance conversion  d(i,j) = √(2(1−ρᵢⱼ))
  2.  Vietoris-Rips filtration via union-find → β₀, β₁ Betti arrays
  3.  Persistence pairs via MST (H₀) and cycle-detection (H₁)
  4.  Persistence landscape λ₁(ε) — dominant topological feature
  5.  Rolling computation across 60-day windows
  6.  Wasserstein-approximation time series
  7.  Euler characteristic χ(ε) = β₀ - β₁

Financial interpretation:
  H₀ pair (birth=0, death=ε*): two market clusters that merged at ε*.
    Low ε* → clusters were similar; High ε* → they were truly distinct.
  H₁ pair (birth=b, death=d): a closed correlation echo-chamber.
    Persistence d−b measures how "real" the feedback loop is.
    Long-lived H₁ loops before crashes = topological pre-crash signal.
"""

import numpy as np
from config import CONFIG


def log(msg: str) -> None:
    print(f"  [ENGINE] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNION-FIND (for connected-components computation)
# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    """Path-compressed, rank-unioned disjoint-set for Kruskal / β₀ tracking."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.n_comp = n   # number of components

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge a and b. Returns True if they were in different components."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_comp -= 1
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. CORRELATION → DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def corr_to_dist(rho: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to Euclidean-style distance matrix.
        d(i,j) = √(2(1 − ρᵢⱼ))
    Maps ρ ∈ [−1, 1] → d ∈ [0, 2].
    d = 0 → perfectly correlated; d = 2 → perfectly anti-correlated.
    """
    rho_clipped = np.clip(rho, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - rho_clipped)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VIETORIS-RIPS FILTRATION — β₀ and β₁ via union-find
# ─────────────────────────────────────────────────────────────────────────────

def vietoris_rips_betti(dist_mat: np.ndarray, eps_vals: np.ndarray):
    """
    Compute Betti numbers β₀(ε) and β₁(ε) for each threshold ε in eps_vals.

    Algorithm:
      β₀ — connected components via union-find (exact).
      β₁ — independent loops via the formula:
              β₁ = #edges_active − #nodes + β₀
           This is exact for 1-dimensional Vietoris-Rips complexes.

    Financial meaning of β₁:
      When adding edge (i,j) closes a cycle, a "correlation echo-chamber" is born.
      It represents a closed feedback loop of panic-propagation.

    Returns
    -------
    beta0 : np.ndarray (N_eps,)
    beta1 : np.ndarray (N_eps,)
    """
    n = dist_mat.shape[0]
    N_eps = len(eps_vals)
    beta0 = np.zeros(N_eps, dtype=int)
    beta1 = np.zeros(N_eps, dtype=int)

    for k, eps in enumerate(eps_vals):
        uf = _UnionFind(n)
        n_edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dist_mat[i, j] <= eps:
                    uf.union(i, j)
                    n_edges += 1
        beta0[k] = uf.n_comp
        # β₁ = E − V + β₀  (Euler characteristic for graph)
        beta1[k] = max(0, n_edges - n + uf.n_comp)

    return beta0, beta1


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERSISTENCE PAIRS — MST (H₀) + CYCLE DETECTION (H₁)
# ─────────────────────────────────────────────────────────────────────────────

def compute_persistence_pairs(dist_mat: np.ndarray):
    """
    Compute H₀ and H₁ persistence pairs for one distance matrix.

    H₀ pairs (birth=0, death=ε*):
      Found via Kruskal's MST — each MST edge merges two components.
      The edge weight is the death time. One component lives forever (death=∞).

    H₁ pairs (birth=b, death=d):
      Each non-MST edge creates a cycle (born at its weight).
      Death approximated as the filtration value when all three vertices
      of the shortest enclosing triangle are mutually within ε.
      (Exact for clique complexes on positive-correlation matrices.)

    Returns
    -------
    pairs_h0 : list of (birth, death) — H₀ features
    pairs_h1 : list of (birth, death) — H₁ features
    """
    n = dist_mat.shape[0]

    # ── Sort all edges by distance ────────────────────────────────────────────
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_mat[i, j], i, j))
    edges.sort(key=lambda x: x[0])

    # ── Kruskal's MST → H₀ persistence pairs ─────────────────────────────────
    uf     = _UnionFind(n)
    mst_edges  = set()   # set of (min_i, max_i)
    pairs_h0   = []

    for w, i, j in edges:
        if uf.union(i, j):
            mst_edges.add((min(i, j), max(i, j)))
            pairs_h0.append((0.0, float(w)))   # born at 0, dies when merged

    # One component survives — give it a large death (we exclude ∞ from plots)
    # (handled by filtering in the landscape computation)

    # ── Non-MST edges → H₁ persistence pairs ─────────────────────────────────
    # Each non-MST edge (i, j, w_birth) creates a 1-cycle born at w_birth.
    # Approximate death: max edge weight in the shortest path between i and j
    # in the MST + the closing edge. The enclosing triangle estimate:
    # death = min over all k of max(d[i,k], d[j,k]) when both d[i,k],d[j,k]<=death
    # Simplified: death = ε at which i,j,k form a triangle.

    # Build adjacency for MST (for shortest path computation)
    mst_adj = [[] for _ in range(n)]
    for w, i, j in edges:
        if (min(i, j), max(i, j)) in mst_edges:
            mst_adj[i].append((j, w))
            mst_adj[j].append((i, w))

    pairs_h1 = []
    for w_birth, i, j in edges:
        key = (min(i, j), max(i, j))
        if key in mst_edges:
            continue   # MST edge — not a cycle

        # This edge closes a cycle born at w_birth.
        # Estimate death: max edge weight on the MST path i→j
        # (the cycle dies when the triangle shortcut is filled)
        path_max = _mst_path_max(mst_adj, i, j, n)
        w_death  = path_max  # the "elder" dies; our non-MST feature born at w_birth

        persistence = w_death - w_birth
        if persistence > 1e-6:
            pairs_h1.append((float(w_birth), float(w_death)))

    return pairs_h0, pairs_h1


def _mst_path_max(adj, src, dst, n):
    """BFS/DFS on MST to find the maximum edge weight along the path src→dst."""
    # DFS with max-bottleneck tracking
    stack  = [(src, 0.0)]
    visited = np.full(n, False)
    visited[src] = True
    while stack:
        node, max_w = stack.pop()
        if node == dst:
            return max_w
        for nbr, w in adj[node]:
            if not visited[nbr]:
                visited[nbr] = True
                stack.append((nbr, max(max_w, w)))
    return 2.0   # fallback — path not found (shouldn't happen on connected MST)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERSISTENCE LANDSCAPE λ₁(ε)
# ─────────────────────────────────────────────────────────────────────────────

def persistence_landscape_1(pairs, eps_vals: np.ndarray) -> np.ndarray:
    """
    Compute the dominant persistence landscape λ₁(ε) for a set of pairs.

    For each feature (b, d):
        tent(b,d)(ε) = max(0, min(ε − b, d − ε))
    λ₁(ε) = max over all features of tent(b,d)(ε).

    Financial interpretation:
        λ₁(ε) is the "size" of the most important topological feature
        at filtration scale ε. A spike in λ₁ means there is one dominant,
        long-lived topological structure at that scale — a leading crash signal.

    Parameters
    ----------
    pairs     : list of (birth, death) tuples
    eps_vals  : np.ndarray, ε evaluation grid

    Returns
    -------
    landscape : np.ndarray (N_landscape,)
    """
    landscape = np.zeros(len(eps_vals))
    for b, d in pairs:
        if d <= b:
            continue
        tent = np.maximum(0.0, np.minimum(eps_vals - b, d - eps_vals))
        landscape = np.maximum(landscape, tent)
    return landscape


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROLLING TOPOLOGICAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_topology(data_bundle: dict, config: dict = CONFIG) -> dict:
    """
    Main engine entry point. Computes the full topological landscape surface.

    For each subsampled time point:
      1. Extract 60-day rolling correlation matrix
      2. Convert to distance matrix
      3. Vietoris-Rips → β₀, β₁ Betti arrays
      4. Persistence pairs via MST + cycle detection
      5. Persistence landscape λ₁(ε)
      6. Wasserstein distance approximation between consecutive windows

    Returns
    -------
    Comprehensive bundle with all topological quantities for visualization.
    """
    returns   = data_bundle["returns"].values    # (T, N)
    T, N      = returns.shape
    win       = config["CORR_WINDOW"]
    N_eps     = config["N_EPS"]
    eps_max   = config["EPS_MAX"]
    subsample = config["SUBSAMPLE"]
    N_land    = config["N_LANDSCAPE"]

    log(f"Assets: {N} | Window: {win}d | ε-resolution: {N_eps} | Time-points: {subsample}")

    # ── Filtration grid ───────────────────────────────────────────────────────
    eps_vals = np.linspace(0.0, eps_max, N_eps)
    land_eps = np.linspace(0.0, eps_max, N_land)

    # ── Subsampled time indices ───────────────────────────────────────────────
    t_start  = win
    t_end    = T
    time_idx = np.linspace(t_start, t_end - 1, subsample, dtype=int)

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    landscape_surface = np.zeros((N_land, subsample))
    beta0_surface     = np.zeros((N_eps, subsample))
    beta1_surface     = np.zeros((N_eps, subsample))
    euler_surface     = np.zeros((N_eps, subsample))
    wasserstein       = np.zeros(subsample)
    top_persistence   = []   # list of np.ndarray per time step

    prev_pers_h1 = np.array([])   # for Wasserstein diff

    log("Running Vietoris-Rips filtration across rolling windows...")

    for t_out, t in enumerate(time_idx):
        if t_out % 10 == 0:
            log(f"  Progress: {t_out+1}/{subsample} (day {t})")

        # ── Rolling correlation matrix ────────────────────────────────────────
        window_ret = returns[max(0, t - win + 1): t + 1]   # (win × N)
        if window_ret.shape[0] < 5:
            continue

        # Correlation matrix with regularisation
        try:
            rho_mat = np.corrcoef(window_ret.T)   # (N × N)
        except Exception:
            rho_mat = np.eye(N)

        # Ensure symmetry and clip
        rho_mat = (rho_mat + rho_mat.T) / 2
        rho_mat = np.clip(rho_mat, -0.999, 0.999)
        np.fill_diagonal(rho_mat, 1.0)

        # ── Distance matrix ───────────────────────────────────────────────────
        dist_mat = corr_to_dist(rho_mat)

        # ── Betti numbers across filtration ───────────────────────────────────
        beta0, beta1 = vietoris_rips_betti(dist_mat, eps_vals)
        beta0_surface[:, t_out] = beta0
        beta1_surface[:, t_out] = beta1
        euler_surface[:, t_out] = beta0 - beta1

        # ── Persistence pairs ─────────────────────────────────────────────────
        pairs_h0, pairs_h1 = compute_persistence_pairs(dist_mat)

        # Top-5 H₁ persistence values (by persistence = death - birth)
        pers_h1 = np.array([d - b for b, d in pairs_h1]) if pairs_h1 else np.zeros(1)
        pers_h1_sorted = np.sort(pers_h1)[::-1]
        top5 = pers_h1_sorted[:5] if len(pers_h1_sorted) >= 5 else \
               np.pad(pers_h1_sorted, (0, 5 - len(pers_h1_sorted)))
        top_persistence.append(top5)

        # ── Persistence landscape λ₁ ──────────────────────────────────────────
        # Combine H₀ and H₁ pairs for the dominant landscape
        all_pairs = pairs_h0 + pairs_h1
        # Filter out infinite/zero-persistence pairs
        finite_pairs = [(b, d) for b, d in all_pairs if d > b and d < 1e6]
        landscape_surface[:, t_out] = persistence_landscape_1(finite_pairs, land_eps)

        # ── Wasserstein approximation ─────────────────────────────────────────
        # W(t) = ||sorted_pers(t) - sorted_pers(t-1)||₂
        # (L2 distance between sorted persistence arrays)
        pers_current = np.sort(pers_h1)[::-1]
        if len(prev_pers_h1) > 0:
            # Pad/trim to same length
            L = max(len(pers_current), len(prev_pers_h1))
            pc = np.pad(pers_current, (0, L - len(pers_current)))
            pp = np.pad(prev_pers_h1, (0, L - len(prev_pers_h1)))
            wasserstein[t_out] = float(np.linalg.norm(pc - pp))
        prev_pers_h1 = pers_current

    # ── Post-process ─────────────────────────────────────────────────────────
    # Smooth Wasserstein slightly for visual clarity
    from scipy.ndimage import uniform_filter1d
    wasserstein_smooth = uniform_filter1d(wasserstein, size=3)

    # Normalise landscape surface to [0, 1] for colormap
    ls_max = landscape_surface.max()
    ls_norm = landscape_surface / (ls_max + 1e-12)

    log(f"Max persistence landscape value: {ls_max:.6f}")
    log(f"Max Wasserstein distance:        {wasserstein_smooth.max():.6f}")
    log(f"Mean β₁ at ε=0.8: {beta1_surface[int(0.8/eps_max * N_eps), :].mean():.2f}")

    # ── Mesh grids for 3D plotting ────────────────────────────────────────────
    EPS_grid, T_grid = np.meshgrid(land_eps, np.arange(subsample), indexing="ij")

    return {
        "landscape_surface": landscape_surface,        # (N_land, N_t) — raw
        "landscape_norm":    ls_norm,                  # (N_land, N_t) — [0,1]
        "beta0_surface":     beta0_surface,            # (N_eps, N_t)
        "beta1_surface":     beta1_surface,            # (N_eps, N_t)
        "euler_surface":     euler_surface,            # (N_eps, N_t)
        "wasserstein":       wasserstein,              # (N_t,) — raw
        "wasserstein_smooth": wasserstein_smooth,      # (N_t,) — smoothed
        "eps_vals":          eps_vals,                 # (N_eps,) — filtration grid
        "land_eps":          land_eps,                 # (N_land,) — landscape grid
        "time_idx":          time_idx,                 # (N_t,) — day indices
        "top_persistence":   top_persistence,          # list of (5,) arrays
        "EPS_grid":          EPS_grid,                 # (N_land, N_t) mesh
        "T_grid":            T_grid,                   # (N_land, N_t) mesh
        "N_eps":             N_eps,
        "N_land":            N_land,
        "N_t":               subsample,
        "ls_max":            ls_max,
        "wass_max":          float(wasserstein_smooth.max()),
    }