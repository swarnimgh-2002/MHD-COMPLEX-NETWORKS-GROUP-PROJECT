"""
03_spectral_analysis.py
Isotropic kinetic, magnetic and total energy spectra E(k) with an
inertial-range power-law fit. Snapshots are loaded once and processed
in a single pass; nothing large is retained.
"""

import os
import gc
import numpy as np
from numpy.fft import fft2

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)


def isotropic_spectrum(vx, vy):
    N = vx.shape[0]
    Vx = fft2(vx) / (N * N)
    Vy = fft2(vy) / (N * N)
    e2d = 0.5 * (np.abs(Vx) ** 2 + np.abs(Vy) ** 2)
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    kint = np.round(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    kmax = N // 2
    E = np.zeros(kmax + 1)
    for kk in range(kmax + 1):
        sel = kint == kk
        if sel.any():
            E[kk] = e2d[sel].sum()
    return np.arange(kmax + 1), E


def inertial_fit(k, E, k_min=2, k_max=8):
    sel = (k >= k_min) & (k <= k_max) & (E > 0)
    if sel.sum() < 3:
        return float("nan"), float("nan")
    coef = np.polyfit(np.log(k[sel]), np.log(E[sel]), 1)
    return float(coef[0]), float(coef[1])


def main():
    arch = np.load(os.path.join(DATA_RAW, "ot_snapshots.npz"),
                   allow_pickle=True)
    t = np.asarray(arch["t"])
    vx = np.asarray(arch["vx"]); vy = np.asarray(arch["vy"])
    Bx = np.asarray(arch["Bx"]); By = np.asarray(arch["By"])
    arch.close()

    kin, mag, tot = [], [], []
    for i in range(len(t)):
        k, Ek = isotropic_spectrum(vx[i], vy[i])
        _, Em = isotropic_spectrum(Bx[i], By[i])
        kin.append(Ek); mag.append(Em); tot.append(Ek + Em)
    kin = np.asarray(kin); mag = np.asarray(mag); tot = np.asarray(tot)

    slope, intercept = inertial_fit(k, tot[-1])
    print(f"Late-time spectral slope (k in [2,8]): {slope:.3f}", flush=True)

    np.savez_compressed(os.path.join(DATA_PROC, "spectra.npz"),
                        k=k, t=t, kinetic=kin, magnetic=mag, total=tot,
                        slope_late=slope, intercept_late=intercept)
    print(f"Saved spectra to {DATA_PROC}/spectra.npz", flush=True)
    del vx, vy, Bx, By, kin, mag, tot
    gc.collect()


if __name__ == "__main__":
    main()
