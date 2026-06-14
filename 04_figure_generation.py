"""
04_figure_generation.py
Generates all figures for the report. Every figure is closed immediately
after saving (plt.close) so that no more than one figure is ever held in
memory. Snapshots and processed arrays are loaded lazily per figure group
and released afterwards.

Figure groups:
  field_evolution/      initial conditions, field evolution, energy, vorticity
  spectral_analysis/    energy spectra, spectral build-up
  network_analysis/     communities, centrality histograms, Laplacian
  physical_correlations/ degree vs variability, betweenness vs current
  threshold_sweep/      percolation curve, lambda2/Q/communities vs theta
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = os.path.dirname(__file__)
DATA_RAW = os.path.join(ROOT, "..", "data", "raw")
DATA_PROC = os.path.join(ROOT, "..", "data", "processed")
FIG = os.path.join(ROOT, "..", "figures")
GRID = 96
SEED = 42

DIRS = {k: os.path.join(FIG, k) for k in
        ("field_evolution", "spectral_analysis", "network_analysis",
         "physical_correlations", "threshold_sweep", "supplementary")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def save(fig, group, name):
    fig.savefig(os.path.join(DIRS[group], name + ".pdf"))
    fig.savefig(os.path.join(DIRS[group], name + ".png"), dpi=150)
    plt.close(fig)
    gc.collect()


def reshape_grid(values, nodes, N=GRID):
    g = np.full(N * N, np.nan)
    g[nodes] = values
    return g.reshape(N, N)


# ----------------------------------------------------------------------
# Field-evolution figures
# ----------------------------------------------------------------------
def field_figures():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    t = np.asarray(arch["t"])
    ext = [0, 2 * np.pi, 0, 2 * np.pi]

    vx0, vy0 = np.asarray(arch["vx"])[0], np.asarray(arch["vy"])[0]
    Bx0, By0 = np.asarray(arch["Bx"])[0], np.asarray(arch["By"])[0]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 8))
    for a, (f, lab, cm) in zip(ax.flat, [(vx0, r"$v_x$", "RdBu_r"),
                                         (vy0, r"$v_y$", "RdBu_r"),
                                         (Bx0, r"$B_x$", "PiYG"),
                                         (By0, r"$B_y$", "PiYG")]):
        im = a.imshow(f, origin="lower", extent=ext, cmap=cm, aspect="equal")
        a.set_title(lab); a.set_xlabel(r"$x$"); a.set_ylabel(r"$y$")
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("Orszag-Tang initial conditions")
    fig.tight_layout()
    save(fig, "field_evolution", "fig01_initial_conditions")

    idx = [0, len(t) // 4, len(t) // 2, len(t) - 1]
    vmag = np.asarray(arch["vmag"]); Bmag = np.asarray(arch["Bmag"])
    J = np.asarray(arch["J"])
    fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))
    rows = [(vmag, r"$|\mathbf{v}|$", "viridis"),
            (Bmag, r"$|\mathbf{B}|$", "plasma"),
            (J, r"$J$", "inferno")]
    for r, (f, lab, cm) in enumerate(rows):
        for c, i in enumerate(idx):
            a = ax[r, c]
            im = a.imshow(f[i], origin="lower", extent=ext, cmap=cm,
                          aspect="equal")
            if r == 0:
                a.set_title(rf"$t={t[i]:.2f}$")
            if c == 0:
                a.set_ylabel(lab)
            a.set_xticks([]); a.set_yticks([])
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("Evolution of field magnitudes and current density")
    fig.tight_layout()
    save(fig, "field_evolution", "fig02_field_evolution")

    KE = np.asarray(arch["KE"]); ME = np.asarray(arch["ME"])
    fig, a = plt.subplots(figsize=(7.2, 4.6))
    a.plot(t, KE, "-o", ms=3, lw=1.5, label="Kinetic")
    a.plot(t, ME, "-s", ms=3, lw=1.5, label="Magnetic")
    a.plot(t, KE + ME, "--^", ms=3, lw=1.5, color="k", label="Total")
    a.set_xlabel(r"$t$"); a.set_ylabel("Domain-averaged energy")
    a.set_title("Energy evolution"); a.legend()
    fig.tight_layout()
    save(fig, "field_evolution", "fig03_energy_evolution")

    om = np.asarray(arch["omega"])
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
    for c, i in enumerate(idx):
        im = ax[0, c].imshow(om[i], origin="lower", extent=ext, cmap="RdBu_r",
                             aspect="equal")
        ax[0, c].set_title(rf"$\omega$, $t={t[i]:.2f}$")
        ax[0, c].set_xticks([]); ax[0, c].set_yticks([])
        plt.colorbar(im, ax=ax[0, c], fraction=0.046, pad=0.04)
        im2 = ax[1, c].imshow(J[i], origin="lower", extent=ext, cmap="inferno",
                              aspect="equal")
        ax[1, c].set_title(rf"$J$, $t={t[i]:.2f}$")
        ax[1, c].set_xticks([]); ax[1, c].set_yticks([])
        plt.colorbar(im2, ax=ax[1, c], fraction=0.046, pad=0.04)
    fig.suptitle("Vorticity and current density")
    fig.tight_layout()
    save(fig, "field_evolution", "fig04_vorticity_current")

    arch.close(); del arch, vmag, Bmag, J, om, KE, ME
    gc.collect()


# ----------------------------------------------------------------------
# Spectral figures
# ----------------------------------------------------------------------
def spectral_figures():
    s = np.load(os.path.join(DATA_PROC, "spectra.npz"))
    k = np.asarray(s["k"]); t = np.asarray(s["t"])
    tot = np.asarray(s["total"]); kin = np.asarray(s["kinetic"])
    mag = np.asarray(s["magnetic"])
    slope = float(s["slope_late"]); inter = float(s["intercept_late"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for i in [0, len(t) // 2, len(t) - 1]:
        ax[0].loglog(k[1:], tot[i][1:], label=rf"$t={t[i]:.2f}$")
    kf = k[(k >= 2) & (k <= 8)]
    ax[0].loglog(kf, np.exp(inter) * kf ** slope, "k--",
                 label=rf"$E\propto k^{{{slope:.2f}}}$")
    ax[0].set_xlabel(r"$k$"); ax[0].set_ylabel(r"$E(k)$")
    ax[0].set_title("Total energy spectrum"); ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    ax[1].loglog(k[1:], kin[-1][1:], "b-o", ms=3, label="Kinetic")
    ax[1].loglog(k[1:], mag[-1][1:], "r-s", ms=3, label="Magnetic")
    ax[1].set_xlabel(r"$k$"); ax[1].set_ylabel(r"$E(k)$, final time")
    ax[1].set_title("Kinetic vs magnetic"); ax[1].legend()
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, "spectral_analysis", "fig05_energy_spectra")

    fig, a = plt.subplots(figsize=(8, 5))
    norm = mpl.colors.Normalize(t.min(), t.max())
    cmap = plt.get_cmap("viridis")
    for i in range(0, len(t), max(1, len(t) // 8)):
        a.loglog(k[1:], tot[i][1:], color=cmap(norm(t[i])), lw=1.3)
    a.set_xlabel(r"$k$"); a.set_ylabel(r"$E(k,t)$")
    a.set_title("Spectral build-up")
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a,
                 label=r"$t$")
    a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, "spectral_analysis", "fig06_spectral_buildup")
    del s, tot, kin, mag
    gc.collect()


# ----------------------------------------------------------------------
# Network figures (headline theta = 0.40 centralities)
# ----------------------------------------------------------------------
def network_figures():
    for layer, cmap_c, comm_cmap in [("velocity", "steelblue", "Set2"),
                                     ("magnetic", "indianred", "Set1")]:
        path = os.path.join(DATA_PROC, f"centralities_{layer}.npz")
        if not os.path.exists(path):
            continue
        c = np.load(path)
        nodes = np.asarray(c["nodes"])
        labels = np.asarray(c["labels"])
        xs, ys = nodes % GRID, nodes // GRID

        fig, a = plt.subplots(figsize=(6.8, 6))
        sc = a.scatter(xs, ys, c=labels,
                       cmap=plt.get_cmap(comm_cmap, int(labels.max() + 1)),
                       s=10, marker="s")
        a.set_xlabel("grid $x$"); a.set_ylabel("grid $y$")
        a.set_title(f"{layer.capitalize()} network communities "
                    f"({int(labels.max() + 1)})")
        a.set_aspect("equal")
        plt.colorbar(sc, ax=a, ticks=range(int(labels.max() + 1)),
                     label="community")
        fig.tight_layout()
        save(fig, "network_analysis", f"fig07_{layer}_communities")

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        for a, key, name in zip(ax.flat,
                                ["degree", "betweenness", "closeness",
                                 "eigenvector"],
                                ["Degree", "Betweenness", "Closeness",
                                 "Eigenvector"]):
            d = np.asarray(c[key])
            a.hist(d, bins=40, color=cmap_c, edgecolor="black", alpha=0.8)
            a.axvline(d.mean(), ls="--", color="black",
                      label=f"mean={d.mean():.4f}")
            a.set_xlabel("value"); a.set_ylabel("count")
            a.set_title(name); a.legend()
        fig.suptitle(f"{layer.capitalize()} network centrality distributions")
        fig.tight_layout()
        save(fig, "network_analysis", f"fig08_{layer}_centrality_hist")

        arr = np.vstack([c["degree"], c["betweenness"], c["closeness"],
                         c["eigenvector"]])
        C = np.corrcoef(arr)
        fig, a = plt.subplots(figsize=(5.8, 5))
        im = a.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1)
        labs = ["Deg", "Bet", "Clo", "Eig"]
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(labs); a.set_yticklabels(labs)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                       fontweight="bold")
        a.set_title(f"{layer.capitalize()} centrality correlations")
        plt.colorbar(im, ax=a, label="Pearson $r$")
        fig.tight_layout()
        save(fig, "network_analysis", f"fig09_{layer}_centrality_corr")
        del c, arr, C
        gc.collect()


# ----------------------------------------------------------------------
# Physical-correlation figures
# ----------------------------------------------------------------------
def physical_figures():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    Tn = len(arch["t"])
    vmag = np.asarray(arch["vmag"]).reshape(Tn, -1)
    Bmag = np.asarray(arch["Bmag"]).reshape(Tn, -1)
    J = np.asarray(arch["J"]).reshape(Tn, -1)
    arch.close()
    sig_v = vmag.std(axis=0); sig_B = Bmag.std(axis=0)
    meanJ = np.abs(J).mean(axis=0)
    del vmag, Bmag, J; gc.collect()

    cv = np.load(os.path.join(DATA_PROC, "centralities_velocity.npz"))
    cm = np.load(os.path.join(DATA_PROC, "centralities_magnetic.npz"))
    vn = np.asarray(cv["nodes"]); mn = np.asarray(cm["nodes"])

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    r1 = np.corrcoef(cv["degree"], sig_v[vn])[0, 1]
    ax[0].scatter(sig_v[vn], cv["degree"], s=5, alpha=0.4, color="steelblue")
    ax[0].set_xlabel(r"$\sigma(|\mathbf{v}|)$"); ax[0].set_ylabel("degree")
    ax[0].set_title(f"Velocity: degree vs variability ($r={r1:.2f}$)")
    r2 = np.corrcoef(cm["degree"], sig_B[mn])[0, 1]
    ax[1].scatter(sig_B[mn], cm["degree"], s=5, alpha=0.4, color="indianred")
    ax[1].set_xlabel(r"$\sigma(|\mathbf{B}|)$"); ax[1].set_ylabel("degree")
    ax[1].set_title(f"Magnetic: degree vs variability ($r={r2:.2f}$)")
    r3 = np.corrcoef(cm["betweenness"], meanJ[mn])[0, 1]
    ax[2].scatter(meanJ[mn], cm["betweenness"], s=5, alpha=0.4, color="darkred")
    ax[2].set_xlabel(r"$\langle|J|\rangle$"); ax[2].set_ylabel("betweenness")
    ax[2].set_title(f"Magnetic: betweenness vs current ($r={r3:.2f}$)")
    fig.tight_layout()
    save(fig, "physical_correlations", "fig10_physical_correlations")

    vdeg = reshape_grid(cv["degree"], vn)
    mbet = reshape_grid(cm["betweenness"], mn)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.5))
    im = ax[0, 0].imshow(sig_v.reshape(GRID, GRID), origin="lower",
                         cmap="viridis", aspect="equal")
    ax[0, 0].set_title(r"$\sigma(|\mathbf{v}|)$")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(meanJ.reshape(GRID, GRID), origin="lower",
                         cmap="inferno", aspect="equal")
    ax[0, 1].set_title(r"$\langle|J|\rangle$")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
    im = ax[1, 0].imshow(vdeg, origin="lower", cmap="Blues", aspect="equal")
    ax[1, 0].set_title("velocity degree centrality")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(mbet, origin="lower", cmap="magma", aspect="equal")
    ax[1, 1].set_title("magnetic betweenness centrality")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("Network metrics and physical fields in real space")
    fig.tight_layout()
    save(fig, "physical_correlations", "fig11_spatial_overlays")
    del cv, cm, sig_v, sig_B, meanJ
    gc.collect()


# ----------------------------------------------------------------------
# Threshold-sweep figures
# ----------------------------------------------------------------------
def sweep_figures():
    # prefer the unified 0.25-0.95 sweep; fall back to the 0.25-0.65 sweep
    full = os.path.join(DATA_PROC, "sweep_full.csv")
    df = pd.read_csv(full) if os.path.exists(full) else \
        pd.read_csv(os.path.join(DATA_PROC, "threshold_sweep.csv"))
    N = float(GRID * GRID)
    v = df[df.layer == "velocity"].sort_values("theta")
    m = df[df.layer == "magnetic"].sort_values("theta")

    # fig12: percolation across the full range (LCC fraction + edge count)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(v.theta, 100 * v.nodes / N, "-o", color="steelblue",
               label="velocity")
    ax[0].plot(m.theta, 100 * m.nodes / N, "-s", color="indianred",
               label="magnetic")
    ax[0].axvline(0.40, ls="--", color="grey", alpha=0.7, label=r"$\theta=0.40$")
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel("giant-component size (\\% of nodes)")
    ax[0].set_ylim(90, 101)
    ax[0].set_title("Percolation: giant component vs threshold")
    ax[0].legend()
    ax[1].semilogy(v.theta, v.edges, "-o", color="steelblue", label="velocity")
    ax[1].semilogy(m.theta, m.edges, "-s", color="indianred", label="magnetic")
    ax[1].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("edges (log scale)")
    ax[1].set_title("Edge count vs threshold"); ax[1].legend()
    fig.tight_layout()
    save(fig, "threshold_sweep", "fig12_percolation")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(v.theta, v.fiedler, "-o", color="steelblue", label="velocity")
    ax[0].plot(m.theta, m.fiedler, "-s", color="indianred", label="magnetic")
    ax[0].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel(r"Fiedler value $\lambda_2$")
    ax[0].set_title("Algebraic connectivity vs threshold"); ax[0].legend()
    ax[1].plot(v.theta, v.modularity, "-o", color="steelblue", label="velocity")
    ax[1].plot(m.theta, m.modularity, "-s", color="indianred", label="magnetic")
    ax[1].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel(r"modularity $Q$")
    ax[1].set_title("Modularity vs threshold"); ax[1].legend()
    fig.tight_layout()
    save(fig, "threshold_sweep", "fig13_lambda_modularity")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(v.theta, v.communities, "-o", color="steelblue", label="velocity")
    ax[0].plot(m.theta, m.communities, "-s", color="indianred", label="magnetic")
    ax[0].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[0].set_xlabel(r"$\theta$"); ax[0].set_ylabel("number of communities")
    ax[0].set_title("Community count vs threshold"); ax[0].legend()
    ax[1].plot(v.theta, v.density, "-o", color="steelblue", label="velocity")
    ax[1].plot(m.theta, m.density, "-s", color="indianred", label="magnetic")
    ax[1].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("network density")
    ax[1].set_yscale("log")
    ax[1].set_title("Density vs threshold"); ax[1].legend()
    fig.tight_layout()
    save(fig, "threshold_sweep", "fig14_communities_density")

    fig, a = plt.subplots(figsize=(7.2, 4.8))
    a.plot(v.theta, v.clustering, "-o", color="steelblue", label="velocity")
    a.plot(m.theta, m.clustering, "-s", color="indianred", label="magnetic")
    a.axvline(0.40, ls="--", color="grey", alpha=0.7)
    a.set_xlabel(r"$\theta$"); a.set_ylabel("average clustering coefficient")
    a.set_title("Clustering vs threshold"); a.legend()
    fig.tight_layout()
    save(fig, "threshold_sweep", "fig15_clustering")
    del df, v, m
    gc.collect()


def extra_field_figures():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    t = np.asarray(arch["t"])
    KE = np.asarray(arch["KE"]); ME = np.asarray(arch["ME"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(t, KE / (KE + ME), "-o", ms=3, color="darkgreen")
    ax[0].axhline(0.5, ls="--", color="grey", alpha=0.7, label="equipartition")
    ax[0].set_xlabel(r"$t$")
    ax[0].set_ylabel(r"$E_K/(E_K+E_M)$")
    ax[0].set_title("Kinetic energy fraction"); ax[0].legend()
    ax[1].plot(t, ME / KE, "-s", ms=3, color="darkred")
    ax[1].axhline(1.0, ls="--", color="grey", alpha=0.7, label="equipartition")
    ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$E_M/E_K$")
    ax[1].set_title("Magnetic-to-kinetic ratio"); ax[1].legend()
    fig.tight_layout()
    save(fig, "field_evolution", "fig16_energy_partition")

    J = np.asarray(arch["J"]); om = np.asarray(arch["omega"])
    idx = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i in idx:
        ax[0].hist(J[i].ravel(), bins=80, density=True, histtype="step",
                   label=rf"$t={t[i]:.2f}$")
        ax[1].hist(om[i].ravel(), bins=80, density=True, histtype="step",
                   label=rf"$t={t[i]:.2f}$")
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"$J$"); ax[0].set_ylabel("PDF")
    ax[0].set_title("Current-density distribution"); ax[0].legend()
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\omega$"); ax[1].set_ylabel("PDF")
    ax[1].set_title("Vorticity distribution"); ax[1].legend()
    fig.tight_layout()
    save(fig, "supplementary", "fig17_pdfs")
    arch.close(); del J, om, KE, ME
    gc.collect()


def ranking_figure():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for layer, color in [("velocity", "steelblue"), ("magnetic", "indianred")]:
        p = os.path.join(DATA_PROC, f"centralities_{layer}.npz")
        if not os.path.exists(p):
            continue
        c = np.load(p)
        d = np.sort(np.asarray(c["degree"]))[::-1]
        e = np.sort(np.asarray(c["eigenvector"]))[::-1]
        ax[0].plot(np.arange(1, d.size + 1), d, color=color, lw=1.4,
                   label=layer)
        ax[1].plot(np.arange(1, e.size + 1), e, color=color, lw=1.4,
                   label=layer)
        del c
    ax[0].set_xlabel("rank"); ax[0].set_ylabel("degree centrality")
    ax[0].set_title("Degree ranking"); ax[0].legend()
    ax[1].set_xlabel("rank"); ax[1].set_ylabel("eigenvector centrality")
    ax[1].set_title("Eigenvector ranking"); ax[1].legend()
    fig.tight_layout()
    save(fig, "network_analysis", "fig18_ranking")


def edgeweight_figure():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    Tn = len(arch["t"])
    vmag = np.asarray(arch["vmag"]).reshape(Tn, -1)
    Bmag = np.asarray(arch["Bmag"]).reshape(Tn, -1)
    arch.close()
    rng = np.random.default_rng(0)
    sel = rng.choice(vmag.shape[1], size=1500, replace=False)

    def corr_sample(F):
        X = F[:, sel].astype(np.float64)
        X -= X.mean(0, keepdims=True)
        s = X.std(0, keepdims=True); s[s == 0] = 1
        X /= s
        R = (X.T @ X) / X.shape[0]
        iu = np.triu_indices(R.shape[0], 1)
        return R[iu]

    rv = corr_sample(vmag); rm = corr_sample(Bmag)
    fig, a = plt.subplots(figsize=(7.2, 4.6))
    a.hist(rv, bins=80, density=True, alpha=0.55, color="steelblue",
           label="velocity")
    a.hist(rm, bins=80, density=True, alpha=0.55, color="indianred",
           label="magnetic")
    for th in (0.25, 0.40, 0.65):
        a.axvline(th, ls="--", color="grey", alpha=0.6)
    a.set_xlabel(r"pairwise correlation $r_{ij}$")
    a.set_ylabel("PDF")
    a.set_title("Correlation distribution (1500-cell sample)")
    a.legend()
    fig.tight_layout()
    save(fig, "network_analysis", "fig19_edgeweights")
    del vmag, Bmag
    gc.collect()


def percolation_full_figure():
    """Combine the main sweep (0.25-0.65) with the extended sweep (0.70-0.95)
    into a single percolation curve covering the full threshold range."""
    sweep = pd.read_csv(os.path.join(DATA_PROC, "threshold_sweep.csv"))
    ext_path = os.path.join(DATA_PROC, "percolation_extended.csv")
    if not os.path.exists(ext_path):
        return
    ext = pd.read_csv(ext_path)
    N = float(GRID * GRID)

    def series(layer):
        s = sweep[sweep.layer == layer][["theta", "nodes", "edges"]].copy()
        s["frac"] = s["nodes"] / N
        e = ext[ext.layer == layer][["theta", "lcc_fraction", "edges"]].copy()
        e = e.rename(columns={"lcc_fraction": "frac"})
        th = list(s["theta"]) + list(e["theta"])
        fr = list(s["frac"]) + list(e["frac"])
        ed = list(s["edges"]) + list(e["edges"])
        order = np.argsort(th)
        return (np.array(th)[order], np.array(fr)[order], np.array(ed)[order])

    tv, fv, ev = series("velocity")
    tm, fm, em = series("magnetic")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(tv, 100 * fv, "-o", color="steelblue", label="velocity")
    ax[0].plot(tm, 100 * fm, "-s", color="indianred", label="magnetic")
    ax[0].axvline(0.40, ls="--", color="grey", alpha=0.7, label=r"$\theta=0.40$")
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel("giant-component size (\\% of nodes)")
    ax[0].set_ylim(90, 101)
    ax[0].set_title("Giant component across the full threshold range")
    ax[0].legend()
    ax[1].semilogy(tv, ev, "-o", color="steelblue", label="velocity")
    ax[1].semilogy(tm, em, "-s", color="indianred", label="magnetic")
    ax[1].axvline(0.40, ls="--", color="grey", alpha=0.7)
    ax[1].set_xlabel(r"$\theta$"); ax[1].set_ylabel("edges (log scale)")
    ax[1].set_title("Edge count vs threshold"); ax[1].legend()
    fig.tight_layout()
    save(fig, "threshold_sweep", "fig20_percolation_full")


def spearman_figure():
    p = os.path.join(DATA_PROC, "spearman_check.npz")
    if not os.path.exists(p):
        return
    sp = np.load(p, allow_pickle=True)
    spv = sp["velocity"][0]; spm = sp["magnetic"][0]
    sweep = pd.read_csv(os.path.join(DATA_PROC, "threshold_sweep.csv"))
    pv = sweep[(sweep.layer == "velocity") & (np.isclose(sweep.theta, 0.40))].iloc[0]
    pm = sweep[(sweep.layer == "magnetic") & (np.isclose(sweep.theta, 0.40))].iloc[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(2); w = 0.35
    # modularity
    ax[0].bar(x - w / 2, [pv.modularity, pm.modularity], w, label="Pearson",
              color="steelblue")
    ax[0].bar(x + w / 2, [spv["modularity"], spm["modularity"]], w,
              label="Spearman", color="mediumseagreen")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["velocity", "magnetic"])
    ax[0].set_ylabel("modularity $Q$")
    ax[0].set_title("Modularity: Pearson vs Spearman"); ax[0].legend()
    # fiedler
    ax[1].bar(x - w / 2, [pv.fiedler, pm.fiedler], w, label="Pearson",
              color="steelblue")
    ax[1].bar(x + w / 2, [spv["fiedler"], spm["fiedler"]], w, label="Spearman",
              color="mediumseagreen")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["velocity", "magnetic"])
    ax[1].set_ylabel(r"Fiedler value $\lambda_2$")
    ax[1].set_title("Algebraic connectivity: Pearson vs Spearman")
    ax[1].legend()
    fig.tight_layout()
    save(fig, "network_analysis", "fig21_spearman_compare")


def multiplex_figure():
    p = os.path.join(DATA_PROC, "multiplex_overlap.npz")
    if not os.path.exists(p):
        return
    ov = np.load(p)
    cv = np.load(os.path.join(DATA_PROC, "centralities_velocity.npz"))
    cm = np.load(os.path.join(DATA_PROC, "centralities_magnetic.npz"))
    vn = np.asarray(cv["nodes"]); mn = np.asarray(cm["nodes"])
    common = np.intersect1d(vn, mn)
    vidx = {int(x): i for i, x in enumerate(vn)}
    midx = {int(x): i for i, x in enumerate(mn)}
    vi = np.array([vidx[int(x)] for x in common])
    mi = np.array([midx[int(x)] for x in common])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(cv["degree"][vi], cm["degree"][mi], s=5, alpha=0.3,
                  color="purple")
    ax[0].set_xlabel("velocity degree centrality")
    ax[0].set_ylabel("magnetic degree centrality")
    ax[0].set_title(f"Inter-layer degree (r = {float(ov['r_degree']):+.3f})")
    metrics = ["degree", "eigenvector", "betweenness", "comm. NMI"]
    vals = [float(ov["r_degree"]), float(ov["r_eigenvector"]),
            float(ov["r_betweenness"]), float(ov["community_nmi"])]
    colors = ["steelblue" if v >= 0 else "indianred" for v in vals]
    ax[1].bar(metrics, vals, color=colors)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_ylabel("correlation / NMI")
    ax[1].set_ylim(-0.3, 0.3)
    ax[1].set_title("Inter-layer overlap (near zero)")
    for i, v in enumerate(vals):
        ax[1].text(i, v + (0.02 if v >= 0 else -0.04), f"{v:+.3f}",
                   ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "network_analysis", "fig22_multiplex")
    del cv, cm


def network_viz_figures():
    """Community-aggregated graph and a sampled force-directed layout for each
    layer. The full graph is far too dense to draw, so these reduced views are
    the meaningful visualisations."""
    import networkx as nx
    for layer, cmap_name in (("velocity", "Set2"), ("magnetic", "Set1")):
        cg = os.path.join(DATA_PROC, f"commgraph_{layer}.npz")
        sg = os.path.join(DATA_PROC, f"subgraph_{layer}.npz")
        if not (os.path.exists(cg) and os.path.exists(sg)):
            continue
        # (a) community-aggregated graph
        d = np.load(cg)
        M = np.asarray(d["matrix"], dtype=float).copy()
        sizes = np.asarray(d["sizes"])
        C = M.shape[0]
        np.fill_diagonal(M, 0.0)
        G = nx.Graph()
        for i in range(C):
            G.add_node(i, size=int(sizes[i]))
        for i in range(C):
            for j in range(i + 1, C):
                if M[i, j] > 0:
                    G.add_edge(i, j, weight=float(M[i, j]))
        fig, a = plt.subplots(figsize=(6.6, 5.6))
        pos = nx.spring_layout(G, seed=SEED, weight="weight")
        node_sz = 300 + 4000 * sizes / sizes.max()
        ew = np.array([G[u][v]["weight"] for u, v in G.edges()])
        ew = 1.0 + 6.0 * ew / ew.max() if ew.size else ew
        nx.draw_networkx_edges(G, pos, ax=a, width=ew, edge_color="grey",
                               alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=a, node_size=node_sz,
                               node_color=range(C),
                               cmap=plt.get_cmap(cmap_name, C),
                               edgecolors="black")
        nx.draw_networkx_labels(G, pos, ax=a, font_size=10)
        a.set_title(f"{layer.capitalize()} network: community graph "
                    f"($\\theta=0.40$)")
        a.axis("off")
        fig.tight_layout()
        save(fig, "network_analysis", f"fig23_{layer}_commgraph")
        del d, M, G

        # (b) sampled force-directed layout, coloured by community
        s = np.load(sg)
        lay = np.asarray(s["layout"]); edges = np.asarray(s["edges"])
        comm = np.asarray(s["community"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
        # force-directed
        for (i, j) in edges[::max(1, len(edges) // 4000)]:
            ax[0].plot([lay[i, 0], lay[j, 0]], [lay[i, 1], lay[j, 1]],
                       color="grey", alpha=0.06, lw=0.4, zorder=1)
        ax[0].scatter(lay[:, 0], lay[:, 1], c=comm,
                      cmap=plt.get_cmap(cmap_name, int(comm.max() + 1)),
                      s=28, edgecolor="black", linewidth=0.3, zorder=2)
        ax[0].set_title("Force-directed layout (node sample)")
        ax[0].axis("off")
        # same sample placed at true grid coordinates
        ax[1].scatter(s["gridx"], s["gridy"], c=comm,
                      cmap=plt.get_cmap(cmap_name, int(comm.max() + 1)),
                      s=28, edgecolor="black", linewidth=0.3)
        ax[1].set_title("Same nodes in physical space")
        ax[1].set_xlabel("grid $x$"); ax[1].set_ylabel("grid $y$")
        ax[1].set_aspect("equal")
        fig.suptitle(f"{layer.capitalize()} network sample "
                     f"({lay.shape[0]} nodes), coloured by community")
        fig.tight_layout()
        save(fig, "network_analysis", f"fig24_{layer}_subgraph")
        del s, lay, edges, comm
        gc.collect()


def main():
    print("field figures ...", flush=True); field_figures()
    print("spectral figures ...", flush=True); spectral_figures()
    print("network figures ...", flush=True); network_figures()
    print("physical figures ...", flush=True); physical_figures()
    print("sweep figures ...", flush=True); sweep_figures()
    print("extra field figures ...", flush=True); extra_field_figures()
    print("ranking figure ...", flush=True); ranking_figure()
    print("edge-weight figure ...", flush=True); edgeweight_figure()
    print("spearman figure ...", flush=True); spearman_figure()
    print("multiplex figure ...", flush=True); multiplex_figure()
    print("network-viz figures ...", flush=True); network_viz_figures()
    print(f"All figures written under {FIG}", flush=True)


if __name__ == "__main__":
    main()
