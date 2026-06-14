"""
05_table_generation.py
Writes the LaTeX tables used in the report, driven entirely by the
processed data files. Tables are emitted as standalone .tex fragments.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(__file__)
DATA_RAW = os.path.join(ROOT, "..", "data", "raw")
DATA_PROC = os.path.join(ROOT, "..", "data", "processed")
TAB_MAIN = os.path.join(ROOT, "..", "tables", "main_text")
TAB_APP = os.path.join(ROOT, "..", "tables", "appendix")
os.makedirs(TAB_MAIN, exist_ok=True)
os.makedirs(TAB_APP, exist_ok=True)


def write(path, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def tab_simulation_parameters():
    body = r"""\begin{table}[htbp]
\centering
\caption{Simulation parameters for the Orszag--Tang run.}
\label{tab:sim_params}
\begin{tabular}{lcc}
\toprule
Parameter & Symbol & Value \\
\midrule
Grid resolution & $N\times N$ & $96\times 96 = 9216$ cells \\
Domain & $\Omega$ & $[0,2\pi]\times[0,2\pi]$ \\
Kinematic viscosity & $\nu$ & $0.02$ \\
Magnetic diffusivity & $\eta$ & $0.02$ \\
Time step & $\Delta t$ & $0.005$ \\
Integration steps & $N_t$ & $200$ \\
Snapshot interval & --- & every $2$ steps \\
Stored snapshots & $T$ & $101$ \\
Dealiasing & --- & $2/3$ rule \\
Integrator & --- & classical RK4 \\
\bottomrule
\end{tabular}
\end{table}
"""
    write(os.path.join(TAB_MAIN, "tab01_sim_params.tex"), body)


def tab_sweep(df):
    order = [0.25, 0.35, 0.40, 0.50, 0.65]
    def block(layer):
        d = df[df.layer == layer].set_index("theta")
        rows = ""
        for th in order:
            r = d.loc[th]
            rows += (f"{th:.2f} & {int(r.nodes)} & {int(r.edges):,} & "
                     f"{r.mean_degree:.0f} & {r.density:.3f} & "
                     f"{r.clustering:.3f} & {int(r.communities)} & "
                     f"{r.modularity:.3f} & {r.fiedler:.4f} \\\\\n")
        return rows
    body = (r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Threshold sensitivity sweep for both interaction "
            r"layers. The headline threshold $\theta=0.40$ is shown in the "
            r"middle row of each block.}" "\n"
            r"\label{tab:sweep}" "\n"
            r"\begin{tabular}{lcccccccc}" "\n" r"\toprule" "\n"
            r"$\theta$ & Nodes & Edges & $\langle k\rangle$ & Density & "
            r"$C$ & Comm. & $Q$ & $\lambda_2$ \\" "\n" r"\midrule" "\n"
            r"\multicolumn{9}{l}{\textit{Velocity network}}\\" "\n"
            + block("velocity") +
            r"\midrule" "\n"
            r"\multicolumn{9}{l}{\textit{Magnetic network}}\\" "\n"
            + block("magnetic") +
            r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(os.path.join(TAB_MAIN, "tab02_sweep.tex"), body)


def tab_headline(df):
    v = df[(df.layer == "velocity") & (np.isclose(df.theta, 0.40))].iloc[0]
    m = df[(df.layer == "magnetic") & (np.isclose(df.theta, 0.40))].iloc[0]
    body = rf"""\begin{{table}}[htbp]
\centering
\caption{{Network statistics at the headline threshold $\theta=0.40$.}}
\label{{tab:headline}}
\begin{{tabular}}{{lcc}}
\toprule
Metric & Velocity & Magnetic \\
\midrule
Nodes (giant component) & {int(v.nodes)} & {int(m.nodes)} \\
Edges & {int(v.edges):,} & {int(m.edges):,} \\
Mean degree & {v.mean_degree:.1f} & {m.mean_degree:.1f} \\
Density & {v.density:.3f} & {m.density:.3f} \\
Clustering coefficient & {v.clustering:.3f} & {m.clustering:.3f} \\
Communities (Louvain) & {int(v.communities)} & {int(m.communities)} \\
Modularity $Q$ & {v.modularity:.3f} & {m.modularity:.3f} \\
Fiedler value $\lambda_2$ & {v.fiedler:.4f} & {m.fiedler:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(TAB_MAIN, "tab03_headline.tex"), body)


def tab_robustness(df):
    sub = df[(df.layer == "velocity") &
             (df.theta.isin([0.35, 0.40, 0.50]))].sort_values("theta")
    subm = df[(df.layer == "magnetic") &
              (df.theta.isin([0.35, 0.40, 0.50]))].sort_values("theta")
    def row(r):
        return (f"{r.theta:.2f} & {int(r.communities)} & {r.modularity:.3f} & "
                f"{r.fiedler:.4f} \\\\\n")
    body = (r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Local robustness of the key diagnostics around the "
            r"operating threshold. Community count, modularity and Fiedler "
            r"value vary only mildly across $0.35\le\theta\le0.50$.}" "\n"
            r"\label{tab:robust}" "\n"
            r"\begin{tabular}{lccc}" "\n" r"\toprule" "\n"
            r"$\theta$ & Communities & $Q$ & $\lambda_2$ \\" "\n" r"\midrule"
            "\n" r"\multicolumn{4}{l}{\textit{Velocity}}\\" "\n")
    for _, r in sub.iterrows():
        body += row(r)
    body += r"\midrule" "\n" r"\multicolumn{4}{l}{\textit{Magnetic}}\\" "\n"
    for _, r in subm.iterrows():
        body += row(r)
    body += r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n"
    write(os.path.join(TAB_MAIN, "tab04_robustness.tex"), body)


def tab_centrality_corr():
    for layer in ("velocity", "magnetic"):
        p = os.path.join(DATA_PROC, f"centralities_{layer}.npz")
        if not os.path.exists(p):
            continue
        c = np.load(p)
        arr = np.vstack([c["degree"], c["betweenness"], c["closeness"],
                         c["eigenvector"]])
        C = np.corrcoef(arr)
        labs = ["Degree", "Betweenness", "Closeness", "Eigenvector"]
        body = (r"\begin{table}[htbp]" "\n" r"\centering" "\n"
                rf"\caption{{Centrality correlations, {layer} network "
                r"($\theta=0.40$).}" "\n"
                rf"\label{{tab:corr_{layer}}}" "\n"
                r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
                r"& Deg & Bet & Clo & Eig \\" "\n" r"\midrule" "\n")
        for i, lab in enumerate(labs):
            body += lab + "".join(f" & {C[i, j]:.2f}" for j in range(4)) + " \\\\\n"
        body += r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n"
        write(os.path.join(TAB_MAIN, f"tab_corr_{layer}.tex"), body)
        del c, arr, C


def tab_physical(df):
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    Tn = len(arch["t"])
    vmag = np.asarray(arch["vmag"]).reshape(Tn, -1)
    Bmag = np.asarray(arch["Bmag"]).reshape(Tn, -1)
    J = np.asarray(arch["J"]).reshape(Tn, -1)
    arch.close()
    sig_v = vmag.std(axis=0); sig_B = Bmag.std(axis=0)
    meanJ = np.abs(J).mean(axis=0)
    cv = np.load(os.path.join(DATA_PROC, "centralities_velocity.npz"))
    cm = np.load(os.path.join(DATA_PROC, "centralities_magnetic.npz"))
    vn = np.asarray(cv["nodes"]); mn = np.asarray(cm["nodes"])
    r1 = np.corrcoef(cv["degree"], sig_v[vn])[0, 1]
    r2 = np.corrcoef(cm["degree"], sig_B[mn])[0, 1]
    r3 = np.corrcoef(cm["betweenness"], meanJ[mn])[0, 1]
    body = rf"""\begin{{table}}[htbp]
\centering
\caption{{Correlations between network centralities and physical fields at
$\theta=0.40$. The current-density correlation is weak and is interpreted
cautiously in the text.}}
\label{{tab:phys}}
\begin{{tabular}}{{llc}}
\toprule
Network & Pairing & Pearson $r$ \\
\midrule
Velocity & Degree vs $\sigma(|\mathbf{{v}}|)$ & ${r1:+.3f}$ \\
Magnetic & Degree vs $\sigma(|\mathbf{{B}}|)$ & ${r2:+.3f}$ \\
Magnetic & Betweenness vs $\langle|J|\rangle$ & ${r3:+.3f}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(TAB_MAIN, "tab05_physical.tex"), body)
    del vmag, Bmag, J


def tab_dimensionless():
    body = r"""\begin{table}[htbp]
\centering
\caption{Characteristic dimensionless numbers for the simulation. With
$\nu=\eta=0.02$ and order-unity velocity and length scales, the run sits at
moderate Reynolds and magnetic Reynolds numbers with unit magnetic Prandtl
number.}
\label{tab:dimensionless}
\begin{tabular}{llc}
\toprule
Number & Definition & Approximate value \\
\midrule
Reynolds & $\mathrm{Re}=UL/\nu$ & $\sim 50$ \\
Magnetic Reynolds & $\mathrm{Rm}=UL/\eta$ & $\sim 50$ \\
Magnetic Prandtl & $\mathrm{Pm}=\nu/\eta$ & $1$ \\
Lundquist & $S=Lv_A/\eta$ & $\sim 50$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    write(os.path.join(TAB_APP, "tab06_dimensionless.tex"), body)


def tab_centrality_stats():
    rows = ""
    for layer in ("velocity", "magnetic"):
        p = os.path.join(DATA_PROC, f"centralities_{layer}.npz")
        if not os.path.exists(p):
            continue
        c = np.load(p)
        rows += rf"\multicolumn{{5}}{{l}}{{\textit{{{layer.capitalize()} network}}}}\\" + "\n"
        for key, name in [("degree", "Degree"), ("betweenness", "Betweenness"),
                          ("closeness", "Closeness"),
                          ("eigenvector", "Eigenvector")]:
            d = np.asarray(c[key])
            rows += (f"{name} & {d.mean():.4f} & {d.std():.4f} & "
                     f"{d.min():.4f} & {d.max():.4f} \\\\\n")
        del c
    body = (r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Summary statistics of the four centrality measures at the "
            r"headline threshold $\theta=0.40$.}" "\n"
            r"\label{tab:centstats}" "\n"
            r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
            r"Centrality & Mean & SD & Min & Max \\" "\n" r"\midrule" "\n"
            + rows + r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(os.path.join(TAB_APP, "tab07_centrality_stats.tex"), body)


def tab_percolation_extended():
    p = os.path.join(DATA_PROC, "percolation_extended.csv")
    if not os.path.exists(p):
        return
    e = pd.read_csv(p)
    def block(layer):
        d = e[e.layer == layer].sort_values("theta")
        rows = ""
        for _, r in d.iterrows():
            rows += (f"{r.theta:.2f} & {int(r.n_components)} & "
                     f"{int(r.lcc_nodes)} & {100*r.lcc_fraction:.1f}\\% & "
                     f"{int(r.edges):,} & {int(r.communities)} & "
                     f"{r.fiedler:.4f} \\\\\n")
        return rows
    body = (r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Extended high-threshold sweep. Even at $\theta=0.95$ the "
            r"giant component still contains almost all nodes, so no "
            r"fragmentation transition is reached in the tested range.}" "\n"
            r"\label{tab:percolation}" "\n"
            r"\begin{tabular}{lcccccc}" "\n" r"\toprule" "\n"
            r"$\theta$ & Comp. & LCC nodes & LCC frac. & Edges & Comm. & "
            r"$\lambda_2$ \\" "\n" r"\midrule" "\n"
            r"\multicolumn{7}{l}{\textit{Velocity network}}\\" "\n"
            + block("velocity") +
            r"\midrule" "\n"
            r"\multicolumn{7}{l}{\textit{Magnetic network}}\\" "\n"
            + block("magnetic") +
            r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(os.path.join(TAB_APP, "tab08_percolation.tex"), body)


def tab_spearman(df):
    p = os.path.join(DATA_PROC, "spearman_check.npz")
    if not os.path.exists(p):
        return
    sp = np.load(p, allow_pickle=True)
    spv = sp["velocity"][0]; spm = sp["magnetic"][0]
    pv = df[(df.layer == "velocity") & (np.isclose(df.theta, 0.40))].iloc[0]
    pm = df[(df.layer == "magnetic") & (np.isclose(df.theta, 0.40))].iloc[0]
    body = rf"""\begin{{table}}[htbp]
\centering
\caption{{Headline diagnostics at $\theta=0.40$ under the Pearson and Spearman
similarity measures. The two measures give very similar results, so the
conclusions do not depend on the linear nature of the Pearson coefficient.}}
\label{{tab:spearman}}
\begin{{tabular}}{{llcc}}
\toprule
Layer & Measure & Communities & $Q$ \quad / \quad $\lambda_2$ \\
\midrule
Velocity & Pearson  & {int(pv.communities)} & {pv.modularity:.3f} / {pv.fiedler:.4f} \\
Velocity & Spearman & {int(spv['communities'])} & {spv['modularity']:.3f} / {spv['fiedler']:.4f} \\
Magnetic & Pearson  & {int(pm.communities)} & {pm.modularity:.3f} / {pm.fiedler:.4f} \\
Magnetic & Spearman & {int(spm['communities'])} & {spm['modularity']:.3f} / {spm['fiedler']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(TAB_MAIN, "tab09_spearman.tex"), body)


def tab_multiplex():
    p = os.path.join(DATA_PROC, "multiplex_overlap.npz")
    if not os.path.exists(p):
        return
    ov = np.load(p)
    body = rf"""\begin{{table}}[htbp]
\centering
\caption{{Inter-layer overlap between the velocity and magnetic networks at
$\theta=0.40$, evaluated cell by cell. All measures are close to zero, so the two
layers organise largely independent structure.}}
\label{{tab:multiplex}}
\begin{{tabular}}{{lc}}
\toprule
Quantity (velocity vs magnetic) & Value \\
\midrule
Degree correlation $r$ & ${float(ov['r_degree']):+.3f}$ \\
Eigenvector correlation $r$ & ${float(ov['r_eigenvector']):+.3f}$ \\
Betweenness correlation $r$ & ${float(ov['r_betweenness']):+.3f}$ \\
Community NMI & ${float(ov['community_nmi']):.3f}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(os.path.join(TAB_MAIN, "tab10_multiplex.tex"), body)


def main():
    df = pd.read_csv(os.path.join(DATA_PROC, "threshold_sweep.csv"))
    tab_simulation_parameters()
    tab_sweep(df)
    tab_headline(df)
    tab_robustness(df)
    tab_centrality_corr()
    tab_physical(df)
    tab_dimensionless()
    tab_centrality_stats()
    tab_percolation_extended()
    tab_spearman(df)
    tab_multiplex()
    print("All tables written.", flush=True)


if __name__ == "__main__":
    main()
