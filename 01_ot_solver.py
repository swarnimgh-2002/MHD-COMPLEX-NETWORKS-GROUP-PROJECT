"""
01_ot_solver.py
Pseudo-spectral solver for the 2D incompressible MHD equations in
vorticity-flux form, applied to the Orszag-Tang vortex.

Final iteration: 96 x 96 grid, 200 RK4 steps, one snapshot every 2 steps
(T = 101 stored snapshots).

Memory notes:
  * Snapshots are stored as float32 to halve the in-memory and on-disk size.
  * Only the scalar magnitudes and current density needed downstream are kept;
    we do not retain the full spectral history.
"""

import os
import gc
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)


def make_grid(N, L=2.0 * np.pi):
    x = np.linspace(0.0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    k = fftfreq(N, d=1.0 / N) * (2.0 * np.pi / L)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX ** 2 + KY ** 2
    K2_inv = np.where(K2 == 0, 1.0, 1.0 / K2)
    K2_inv[0, 0] = 0.0
    return X, Y, KX, KY, K2, K2_inv


def dealias_mask(N):
    # Classical 2/3 rule: zero all modes with |k| > N/3 after each product.
    kmax = N // 3
    k = fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    return (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)


def initial_conditions(X, Y):
    omega0 = -np.cos(X) - np.cos(Y)
    A0 = np.cos(Y) + 0.5 * np.cos(2.0 * X)
    return omega0, A0


def velocity_from_psi(psi_hat, KX, KY):
    vx = np.real(ifft2(1j * KY * psi_hat))
    vy = np.real(ifft2(-1j * KX * psi_hat))
    return vx, vy


def magnetic_from_A(A_hat, KX, KY):
    Bx = np.real(ifft2(1j * KY * A_hat))
    By = np.real(ifft2(-1j * KX * A_hat))
    return Bx, By


def rhs(omega_hat, A_hat, KX, KY, K2, K2_inv, nu, eta, mask):
    psi_hat = -omega_hat * K2_inv
    vx, vy = velocity_from_psi(psi_hat, KX, KY)
    Bx, By = magnetic_from_A(A_hat, KX, KY)

    J_hat = -K2 * A_hat
    domega_dx = np.real(ifft2(1j * KX * omega_hat))
    domega_dy = np.real(ifft2(1j * KY * omega_hat))
    dJ_dx = np.real(ifft2(1j * KX * J_hat))
    dJ_dy = np.real(ifft2(1j * KY * J_hat))
    dA_dx = np.real(ifft2(1j * KX * A_hat))
    dA_dy = np.real(ifft2(1j * KY * A_hat))

    adv_omega = vx * domega_dx + vy * domega_dy
    lorentz = Bx * dJ_dx + By * dJ_dy
    adv_A = vx * dA_dx + vy * dA_dy

    domega_hat = (-fft2(adv_omega) + fft2(lorentz) - nu * K2 * omega_hat) * mask
    dA_hat = (-fft2(adv_A) - eta * K2 * A_hat) * mask
    return domega_hat, dA_hat


def rk4_step(omega_hat, A_hat, dt, KX, KY, K2, K2_inv, nu, eta, mask):
    k1o, k1a = rhs(omega_hat, A_hat, KX, KY, K2, K2_inv, nu, eta, mask)
    k2o, k2a = rhs(omega_hat + 0.5 * dt * k1o, A_hat + 0.5 * dt * k1a,
                   KX, KY, K2, K2_inv, nu, eta, mask)
    k3o, k3a = rhs(omega_hat + 0.5 * dt * k2o, A_hat + 0.5 * dt * k2a,
                   KX, KY, K2, K2_inv, nu, eta, mask)
    k4o, k4a = rhs(omega_hat + dt * k3o, A_hat + dt * k3a,
                   KX, KY, K2, K2_inv, nu, eta, mask)
    omega_hat = omega_hat + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
    A_hat = A_hat + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
    return omega_hat, A_hat


def run_simulation(N=96, n_steps=200, dt=0.005, nu=0.02, eta=0.02,
                   snapshot_every=2, outdir=DATA_RAW):
    X, Y, KX, KY, K2, K2_inv = make_grid(N)
    mask = dealias_mask(N)
    omega, A = initial_conditions(X, Y)
    omega_hat = fft2(omega) * mask
    A_hat = fft2(A) * mask

    snaps = {k: [] for k in ("t", "vx", "vy", "Bx", "By",
                             "vmag", "Bmag", "J", "omega", "KE", "ME")}

    for step in range(n_steps + 1):
        t = step * dt
        if step % snapshot_every == 0:
            psi_hat = -omega_hat * K2_inv
            vx, vy = velocity_from_psi(psi_hat, KX, KY)
            Bx, By = magnetic_from_A(A_hat, KX, KY)
            J = np.real(ifft2(-K2 * A_hat))
            om = np.real(ifft2(omega_hat))
            vmag = np.sqrt(vx ** 2 + vy ** 2)
            Bmag = np.sqrt(Bx ** 2 + By ** 2)
            snaps["t"].append(t)
            snaps["vx"].append(vx.astype(np.float32))
            snaps["vy"].append(vy.astype(np.float32))
            snaps["Bx"].append(Bx.astype(np.float32))
            snaps["By"].append(By.astype(np.float32))
            snaps["vmag"].append(vmag.astype(np.float32))
            snaps["Bmag"].append(Bmag.astype(np.float32))
            snaps["J"].append(J.astype(np.float32))
            snaps["omega"].append(om.astype(np.float32))
            snaps["KE"].append(0.5 * float(np.mean(vmag ** 2)))
            snaps["ME"].append(0.5 * float(np.mean(Bmag ** 2)))
            if step % 20 == 0:
                print(f"  step {step:4d}  t={t:.3f}  "
                      f"KE={snaps['KE'][-1]:.4f}  ME={snaps['ME'][-1]:.4f}",
                      flush=True)
        if step < n_steps:
            omega_hat, A_hat = rk4_step(omega_hat, A_hat, dt,
                                        KX, KY, K2, K2_inv, nu, eta, mask)

    out = {k: np.asarray(v) for k, v in snaps.items()}
    out["X"] = X.astype(np.float32)
    out["Y"] = Y.astype(np.float32)
    out["params"] = np.array([N, n_steps, dt, nu, eta, snapshot_every])
    path = os.path.join(outdir, "ot_snapshots.npz")
    np.savez_compressed(path, **out)
    print(f"Saved {len(out['t'])} snapshots ({N}x{N}) to {path}", flush=True)

    del snaps, out, omega_hat, A_hat
    gc.collect()


if __name__ == "__main__":
    run_simulation()
