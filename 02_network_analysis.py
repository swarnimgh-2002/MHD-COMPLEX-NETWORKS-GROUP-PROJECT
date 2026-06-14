"""
02_network_analysis.py
Builds velocity and magnetic interaction networks from the Orszag-Tang
snapshots and runs a five-threshold sensitivity sweep.

Memory strategy (important for laptops with limited RAM):
  * The dense correlation matrix is never formed. For each threshold we build
    the sparse adjacency directly, block by block, thresholding correlations
    on the fly. Peak memory for the correlation step is one block (~20 MB).
  * Thresholds are processed strictly one at a time. After each threshold we
    delete every large object and call gc.collect() before starting the next.
  * Sparse CSR matrices are used throughout; no dense adjacency is duplicated.
  * The expensive centralities (betweenness, closeness, eigenvector) are
    computed only at the headline threshold theta = 0.40, for both layers.
    The sweep itself only needs cheap quantities (degree, density, clustering,
    communities, Fiedler value), which keeps the loop light.
  * Louvain runs on the scipy CSR matrix via scikit-network (Cython), so a
    heavy Python graph object is never created.
  * Snapshots are loaded once, reduced to the two magnitude time series, and
    the raw archive handle is released immediately.
"""

import os
import gc
import csv
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh
from sknetwork.clustering import Louvain, get_modularity

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

THRESHOLDS = [0.25, 0.35, 0.40, 0.50, 0.65]
HEADLINE = 0.40
GRID = 96
BLOCK = 512
BET_SAMPLES = 64
SEED = 42


def standardise(F):
    """Return a (T, N) float32 matrix with zero mean and unit variance per
    column (per grid cell). Computed in place to avoid extra copies."""
    F = F.astype(np.float32, copy=True)
    F -= F.mean(axis=0, keepdims=True)
    std = F.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    F /= std
    return F


def build_sparse_upper(S, theta):
    """Build the upper-triangular thresholded correlation adjacency without
    ever forming the full dense N x N matrix. Returns a symmetric CSR matrix.

    For a block of columns we compute their correlations against all columns
    (block_size x N), threshold, and keep only entries with j > i. Memory is
    bounded by one block."""
    T, N = S.shape
    rows_all, cols_all, vals_all = [], [], []
    for a in range(0, N, BLOCK):
        b = min(a + BLOCK, N)
        # correlations of columns [a:b] against all columns
        Rblock = (S[:, a:b].T @ S) / T          # (b-a, N) float32
        # keep only strict upper triangle: global column index > global row
        for local_i in range(b - a):
            gi = a + local_i
            row = Rblock[local_i]
            mask = row > theta
            mask[:gi + 1] = False                # enforce j > i, drop self
            js = np.nonzero(mask)[0]
            if js.size:
                rows_all.append(np.full(js.size, gi, dtype=np.int32))
                cols_all.append(js.astype(np.int32))
                vals_all.append(row[js].astype(np.float32))
        del Rblock
    if rows_all:
        r = np.concatenate(rows_all)
        c = np.concatenate(cols_all)
        v = np.concatenate(vals_all)
    else:
        r = np.empty(0, np.int32); c = np.empty(0, np.int32)
        v = np.empty(0, np.float32)
    del rows_all, cols_all, vals_all
    upper = sparse.coo_matrix((v, (r, c)), shape=(N, N), dtype=np.float32)
    A = (upper + upper.T).tocsr()
    del upper, r, c, v
    gc.collect()
    return A


def giant_component(A):
    """Return the CSR adjacency restricted to the largest connected component
    plus the original node indices it contains."""
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return A, np.arange(A.shape[0])
    sizes = np.bincount(labels)
    keep = np.nonzero(labels == sizes.argmax())[0]
    return A[keep][:, keep].tocsr(), keep


def avg_clustering_sparse(A):
    """Weighted-graph-agnostic average local clustering coefficient computed
    block by block to bound memory. Uses the binary structure of A."""
    Ab = (A > 0).astype(np.float32).tocsr()
    n = Ab.shape[0]
    deg = np.asarray(Ab.sum(axis=1)).ravel()
    clust = np.zeros(n, dtype=np.float64)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        AA = (Ab[a:b] @ Ab)                       # (b-a, n) sparse
        tri = np.asarray(AA.multiply(Ab[a:b]).sum(axis=1)).ravel()
        d = deg[a:b]
        denom = d * (d - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(denom > 0, tri / denom, 0.0)
        clust[a:b] = c
        del AA
    del Ab
    gc.collect()
    return float(clust.mean())


def fiedler_value(A):
    """Algebraic connectivity (second-smallest eigenvalue) of the normalised
    Laplacian, computed matrix-free with Lanczos. 'SA' (smallest algebraic)
    is used rather than 'SM' to avoid the slow convergence that plagues the
    smallest-magnitude mode on bounded spectra."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    D = sparse.diags(dinv)
    L = sparse.identity(n, format="csr") - D @ A @ D
    L = (L + L.T) * 0.5
    try:
        vals = eigsh(L, k=2, which="SA", tol=1e-3, maxiter=3000,
                     return_eigenvectors=False)
        vals = np.sort(vals)
        lam2 = float(vals[1])
    except Exception:
        lam2 = float("nan")
    del L, D
    gc.collect()
    return lam2


def louvain_communities(A):
    louvain = Louvain(random_state=SEED)
    labels = louvain.fit_predict(A)
    Q = float(get_modularity(A, labels))
    n_comm = int(labels.max() + 1) if labels.size else 0
    return labels, n_comm, Q


def closeness_chunked(A):
    """Closeness centrality via unweighted BFS in source batches, so the full
    distance matrix is never held in memory."""
    n = A.shape[0]
    Ab = (A > 0).astype(np.int8)
    closeness = np.zeros(n)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        d = shortest_path(Ab, method="D", unweighted=True, indices=np.arange(a, b))
        d[~np.isfinite(d)] = 0.0
        s = d.sum(axis=1)
        closeness[a:b] = np.where(s > 0, (n - 1) / s, 0.0)
        del d
    del Ab
    gc.collect()
    return closeness


def eigenvector_centrality(A):
    try:
        vals, vecs = eigsh(A.astype(np.float64), k=1, which="LA", tol=1e-4,
                           maxiter=3000)
        v = np.abs(vecs[:, 0])
        return v / (v.max() if v.max() > 0 else 1.0)
    except Exception:
        return np.zeros(A.shape[0])


def betweenness_igraph(A, nodes):
    """Sampled betweenness via igraph (C backend). Built only at the headline
    threshold, for the two layers, so the transient graph object is created at
    most twice in the whole run."""
    import igraph as ig
    coo = sparse.triu(A, k=1).tocoo()
    # pass the numpy edge array directly; igraph accepts it via the buffer
    # protocol, which avoids materialising a multi-million-entry Python list.
    edges = np.column_stack([coo.row, coo.col]).astype(np.int64)
    g = ig.Graph(n=A.shape[0], edges=edges)
    k = min(BET_SAMPLES, A.shape[0])
    rng = np.random.default_rng(SEED)
    sources = rng.choice(A.shape[0], size=k, replace=False).tolist()
    bet = np.array(g.betweenness(vertices=None, sources=sources), dtype=float)
    norm = (A.shape[0] - 1) * (A.shape[0] - 2) / 2.0
    bet = bet / norm if norm > 0 else bet
    del g, edges, coo
    gc.collect()
    return bet


def analyse_threshold(S, theta, layer, full=False):
    print(f"\n[{layer}] theta = {theta:.2f}", flush=True)
    A = build_sparse_upper(S, theta)
    A_lcc, keep = giant_component(A)
    del A
    gc.collect()

    n = A_lcc.shape[0]
    m = A_lcc.nnz // 2
    deg = np.asarray((A_lcc > 0).sum(axis=1)).ravel()
    mean_deg = float(deg.mean()) if n else 0.0
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    print(f"    LCC: {n} nodes, {m} edges, mean deg {mean_deg:.1f}", flush=True)

    clustering = avg_clustering_sparse(A_lcc)
    labels, n_comm, Q = louvain_communities(A_lcc)
    lam2 = fiedler_value(A_lcc)
    print(f"    communities={n_comm}  Q={Q:.4f}  lambda2={lam2:.4f}  "
          f"C={clustering:.4f}", flush=True)

    summary = dict(layer=layer, theta=theta, nodes=n, edges=m,
                   mean_degree=mean_deg, density=density,
                   clustering=clustering, communities=n_comm,
                   modularity=Q, fiedler=lam2)

    if full:
        print("    full centralities (betweenness/closeness/eigenvector) ...",
              flush=True)
        bet = betweenness_igraph(A_lcc, keep)
        clo = closeness_chunked(A_lcc)
        eig = eigenvector_centrality(A_lcc)
        deg_c = deg / (n - 1) if n > 1 else deg.astype(float)
        np.savez_compressed(
            os.path.join(DATA_PROC, f"centralities_{layer}.npz"),
            nodes=keep.astype(np.int32),
            degree=deg_c.astype(np.float32),
            betweenness=bet.astype(np.float32),
            closeness=clo.astype(np.float32),
            eigenvector=eig.astype(np.float32),
            labels=labels.astype(np.int32),
        )
        del bet, clo, eig, deg_c
        gc.collect()

    del A_lcc, keep, deg, labels
    gc.collect()
    return summary


def main():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    Tn = len(arch["t"])
    vmag = np.asarray(arch["vmag"]).reshape(Tn, -1)
    Bmag = np.asarray(arch["Bmag"]).reshape(Tn, -1)
    arch.close()
    del arch
    gc.collect()

    Sv = standardise(vmag); del vmag; gc.collect()
    Sb = standardise(Bmag); del Bmag; gc.collect()

    rows = []
    for theta in THRESHOLDS:
        full = abs(theta - HEADLINE) < 1e-9
        rows.append(analyse_threshold(Sv, theta, "velocity", full=full))
        rows.append(analyse_threshold(Sb, theta, "magnetic", full=full))

    # write the sweep summary as a plain CSV (lightweight, human readable)
    out_csv = os.path.join(DATA_PROC, "threshold_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved sweep summary to {out_csv}", flush=True)

    del Sv, Sb
    gc.collect()


if __name__ == "__main__":
    main()
