#!/usr/bin/env python3
"""
Experiment II: 3-D shear-mode population response for D3Q19 and D3Q27.

Validates the CE2 population-space prediction
    g_Q^CE2 = C0 * Lhat_Q * q_F(k,U)
against the kinetic population vector extracted from a full nonlinear
periodic BGK LBM simulation initialized with an exact discrete shear mode.

Primary diagnostics:
    E_Q   = ||g_LBM - g_CE2||_w / ||g_CE2||_w
    R_Q   = ||g_LBM||_w / ||g_CE2||_w
    chi_Q = weighted population-space alignment

The exact Lhat_Q matrices are loaded from machine-readable CSV files generated
by the symbolic theory. Population ordering is therefore inherited directly
from those files and cannot silently drift from the theoretical convention.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


CS2 = 1.0 / 3.0
CS4 = CS2 * CS2


# -----------------------------------------------------------------------------
# Lattice/operator loading
# -----------------------------------------------------------------------------

@dataclass
class Lattice:
    name: str
    c: np.ndarray          # (Q,3), integer velocities
    w: np.ndarray          # (Q,)
    lhat: np.ndarray       # (Q,15)
    q_names: List[str]
    Winv: np.ndarray
    P_eq: np.ndarray
    P_le1: np.ndarray
    P_le2: np.ndarray
    P2: np.ndarray
    PK: np.ndarray
    N: np.ndarray


def _parse_rational(s: str) -> float:
    return float(Fraction(s))


def weighted_projector(columns: Sequence[np.ndarray], winv: np.ndarray) -> np.ndarray:
    phi = np.column_stack(columns).astype(complex)
    gram = phi.conj().T @ winv @ phi
    return phi @ np.linalg.solve(gram, phi.conj().T @ winv)


def load_lattice(operator_csv: Path) -> Lattice:
    rows = list(csv.reader(operator_csv.open()))
    header = rows[0]
    q_names = header[4:]
    body = rows[1:]

    c = np.array([[int(r[0]), int(r[1]), int(r[2])] for r in body], dtype=int)
    w = np.array([_parse_rational(r[3]) for r in body], dtype=float)
    lhat = np.array([[_parse_rational(x) for x in r[4:]] for r in body], dtype=float)

    Q = len(w)
    if lhat.shape != (Q, 15):
        raise ValueError(f"Expected Qx15 Lhat, got {lhat.shape}")

    if not np.isclose(np.sum(w), 1.0, atol=1e-14):
        raise ValueError("Lattice weights do not sum to one.")

    winv = np.diag(1.0 / w)

    # Discrete Hermite population columns through second order.
    rho = w.copy()
    jx = w * c[:, 0]
    jy = w * c[:, 1]
    jz = w * c[:, 2]

    hxx = w * (c[:, 0] ** 2 - CS2)
    hyy = w * (c[:, 1] ** 2 - CS2)
    hzz = w * (c[:, 2] ** 2 - CS2)
    hxy = w * (c[:, 0] * c[:, 1])
    hxz = w * (c[:, 0] * c[:, 2])
    hyz = w * (c[:, 1] * c[:, 2])

    p_le1 = weighted_projector([rho, jx, jy, jz], winv)
    p_le2 = weighted_projector(
        [rho, jx, jy, jz, hxx, hyy, hzz, hxy, hxz, hyz], winv
    )
    p2 = p_le2 - p_le1
    pk = np.eye(Q, dtype=complex) - p_le2

    p_eq = np.empty((Q, Q), dtype=complex)
    for i in range(Q):
        for j in range(Q):
            p_eq[i, j] = w[i] * (1.0 + np.dot(c[i], c[j]) / CS2)

    nproj = np.eye(Q, dtype=complex) - p_eq

    # Basic projector audits.
    if np.linalg.norm(pk @ pk - pk) > 5e-12:
        raise RuntimeError(f"{operator_csv.name}: PK is not idempotent.")
    if np.linalg.norm(pk @ p_le2) > 5e-12:
        raise RuntimeError(f"{operator_csv.name}: PK not orthogonal to <=2 space.")

    name = f"D3Q{Q}"
    return Lattice(
        name=name,
        c=c,
        w=w,
        lhat=lhat,
        q_names=q_names,
        Winv=winv,
        P_eq=p_eq,
        P_le1=p_le1,
        P_le2=p_le2,
        P2=p2,
        PK=pk,
        N=nproj,
    )


def weighted_norm(lat: Lattice, v: np.ndarray) -> float:
    z = np.vdot(v, lat.Winv @ v)
    return float(np.sqrt(max(z.real, 0.0)))


# -----------------------------------------------------------------------------
# Exact Fourier amplification and shear-mode selection
# -----------------------------------------------------------------------------

def collision_matrix(lat: Lattice, tau: float) -> np.ndarray:
    Q = len(lat.w)
    return (1.0 - 1.0 / tau) * np.eye(Q) + (1.0 / tau) * lat.P_eq


def amplification_matrix(lat: Lattice, kvec: np.ndarray, tau: float) -> np.ndarray:
    phase = np.exp(-1j * (lat.c @ kvec))
    return np.diag(phase) @ collision_matrix(lat, tau)


def hydrodynamic_moments(lat: Lattice, v: np.ndarray) -> Tuple[complex, np.ndarray]:
    drho = np.sum(v)
    dj = lat.c.T @ v
    return drho, dj


@dataclass
class ShearMode:
    eigenvalue: complex
    eigenvector: np.ndarray
    target_polarization: np.ndarray
    measured_polarization: np.ndarray
    density_fraction: float
    longitudinal_fraction: float
    polarization_alignment: float
    branch_gap: float


def select_shear_mode(
    lat: Lattice,
    kvec: np.ndarray,
    target_pol: np.ndarray,
    tau: float,
    degeneracy_tol: float = 2e-8,
) -> ShearMode:
    """
    Select a transverse shear eigenmode with the requested polarization.

    If two shear eigenvalues are numerically degenerate, form the linear
    combination within that eigenspace whose momentum best matches target_pol.
    """
    k = np.linalg.norm(kvec)
    n = kvec / k
    e = np.asarray(target_pol, dtype=float)
    e = e - np.dot(e, n) * n
    e /= np.linalg.norm(e)

    A = amplification_matrix(lat, kvec, tau)
    evals, evecs = np.linalg.eig(A)

    # Hydrodynamic score for each eigenvector.
    records = []
    for j in range(len(evals)):
        v = evecs[:, j]
        drho, dj = hydrodynamic_moments(lat, v)
        jmag = np.linalg.norm(dj)
        if jmag < 1e-14:
            continue
        long = abs(np.dot(n, dj)) / jmag
        dens = abs(drho) / jmag
        pol = abs(np.vdot(e, dj)) / jmag
        # Favor slow hydrodynamic modes and transverse momentum.
        score = pol * np.exp(-4.0 * long) * np.exp(-2.0 * dens) * abs(evals[j])
        records.append((score, j, long, dens, pol))

    if not records:
        raise RuntimeError("No hydrodynamic candidate found.")

    records.sort(reverse=True, key=lambda x: x[0])
    _, j0, _, _, _ = records[0]
    lam0 = evals[j0]

    # Collect near-degenerate transverse candidates around the best eigenvalue.
    cand = []
    for _, j, long, dens, pol in records:
        if abs(evals[j] - lam0) <= degeneracy_tol * max(1.0, abs(lam0)):
            if long < 5e-4 and dens < 5e-4:
                cand.append(j)

    if len(cand) >= 2:
        V = evecs[:, cand]
        # Momentum map of the candidate eigenspace: 3 x r.
        M = lat.c.T @ V
        # Find coefficients minimizing ||M a - e||_2.
        coeff, *_ = np.linalg.lstsq(M, e.astype(complex), rcond=None)
        v = V @ coeff
        # All candidates share lam0 within tolerance; use Rayleigh quotient.
        Av = A @ v
        lam = np.vdot(v, Av) / np.vdot(v, v)
        branch_gap = float(max(abs(evals[j] - lam0) for j in cand))
    else:
        v = evecs[:, j0]
        lam = lam0
        branch_gap = 0.0

    drho, dj = hydrodynamic_moments(lat, v)
    amp = np.dot(e, dj)
    if abs(amp) < 1e-14:
        raise RuntimeError("Selected shear mode has negligible target momentum.")
    v = v / amp

    drho, dj = hydrodynamic_moments(lat, v)
    jmag = np.linalg.norm(dj)
    measured_pol = np.real_if_close(dj / np.dot(e, dj))

    return ShearMode(
        eigenvalue=lam,
        eigenvector=v,
        target_polarization=e,
        measured_polarization=np.asarray(measured_pol),
        density_fraction=float(abs(drho) / (jmag + 1e-300)),
        longitudinal_fraction=float(abs(np.dot(n, dj)) / (jmag + 1e-300)),
        polarization_alignment=float(abs(np.vdot(e, dj)) / (jmag + 1e-300)),
        branch_gap=branch_gap,
    )


# -----------------------------------------------------------------------------
# Full nonlinear 3-D BGK LBM
# -----------------------------------------------------------------------------

def macroscopic(lat: Lattice, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # f: (Q,Nz,Ny,Nx)
    rho = np.sum(f, axis=0)
    j = np.tensordot(lat.c.T, f, axes=(1, 0))  # (3,Nz,Ny,Nx)
    u = j / rho[None, ...]
    return rho, u


def equilibrium(lat: Lattice, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    Q = len(lat.w)
    usq = np.sum(u * u, axis=0)
    feq = np.empty((Q,) + rho.shape, dtype=float)
    for i, ci in enumerate(lat.c):
        cu = ci[0] * u[0] + ci[1] * u[1] + ci[2] * u[2]
        feq[i] = lat.w[i] * rho * (
            1.0
            + cu / CS2
            + 0.5 * cu * cu / (CS2 * CS2)
            - 0.5 * usq / CS2
        )
    return feq


def collide_and_stream(lat: Lattice, f: np.ndarray, tau: float) -> np.ndarray:
    rho, u = macroscopic(lat, f)
    feq = equilibrium(lat, rho, u)
    fpost = f - (f - feq) / tau

    fout = np.empty_like(f)
    for i, (cx, cy, cz) in enumerate(lat.c):
        # axes are (z,y,x)
        fout[i] = np.roll(fpost[i], shift=(cz, cy, cx), axis=(0, 1, 2))
    return fout


def initialize_exact_mode(
    lat: Lattice,
    N: int,
    mode_indices: Tuple[int, int, int],
    polarization: np.ndarray,
    tau: float,
    epsilon: float = 1e-6,
    rho0: float = 1.0,
) -> Tuple[np.ndarray, ShearMode, np.ndarray]:
    m = np.array(mode_indices, dtype=int)
    kvec = 2.0 * np.pi * m / N
    mode = select_shear_mode(lat, kvec, polarization, tau)

    x = np.arange(N)[None, None, :]
    y = np.arange(N)[None, :, None]
    z = np.arange(N)[:, None, None]
    phase = np.exp(1j * (kvec[0] * x + kvec[1] * y + kvec[2] * z))

    Q = len(lat.w)
    f = np.empty((Q, N, N, N), dtype=float)
    for i in range(Q):
        f[i] = lat.w[i] * rho0 + epsilon * np.real(mode.eigenvector[i] * phase)

    return f, mode, kvec


# -----------------------------------------------------------------------------
# Fourier diagnostics and CE2 prediction
# -----------------------------------------------------------------------------

def extract_fourier_mode(
    lat: Lattice,
    f: np.ndarray,
    kvec: np.ndarray,
    rho0: float = 1.0,
) -> np.ndarray:
    _, Nz, Ny, Nx = f.shape
    x = np.arange(Nx)[None, None, :]
    y = np.arange(Ny)[None, :, None]
    z = np.arange(Nz)[:, None, None]
    phase = np.exp(-1j * (kvec[0] * x + kvec[1] * y + kvec[2] * z))

    base = lat.w[:, None, None, None] * rho0
    df = f - base
    return np.sum(df * phase[None, ...], axis=(1, 2, 3)) / (Nx * Ny * Nz)


def q_fourier(kvec: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Curvature vector in the exact manuscript ordering:
    ux_xy, ux_xz, ux_yy, ux_zz, ux_yz,
    uy_xy, uy_yz, uy_xx, uy_zz, uy_xz,
    uz_xz, uz_yz, uz_xx, uz_yy, uz_xy.
    """
    kx, ky, kz = kvec
    ux, uy, uz = U
    return -np.array(
        [
            kx * ky * ux,
            kx * kz * ux,
            ky * ky * ux,
            kz * kz * ux,
            ky * kz * ux,
            kx * ky * uy,
            ky * kz * uy,
            kx * kx * uy,
            kz * kz * uy,
            kx * kz * uy,
            kx * kz * uz,
            ky * kz * uz,
            kx * kx * uz,
            ky * ky * uz,
            kx * ky * uz,
        ],
        dtype=complex,
    )


def ce2_prefactor(tau: float, rho0: float = 1.0) -> float:
    return 2.0 * rho0 * tau * (tau - 0.5) * CS4


@dataclass
class Diagnostic:
    E: float
    R: float
    chi: float
    norm_lbm: float
    norm_ce2: float
    density_fraction: float
    longitudinal_fraction: float
    polarization_alignment: float
    decay_ratio: complex
    g_lbm: np.ndarray
    g_ce2: np.ndarray
    fhat: np.ndarray


def diagnose(
    lat: Lattice,
    f: np.ndarray,
    kvec: np.ndarray,
    target_pol: np.ndarray,
    tau: float,
    previous_fhat: np.ndarray | None = None,
    rho0: float = 1.0,
) -> Diagnostic:
    fhat = extract_fourier_mode(lat, f, kvec, rho0)

    fneq = lat.N @ fhat
    g_lbm = lat.PK @ fneq

    drho, dj = hydrodynamic_moments(lat, fhat)
    U = dj / rho0
    qf = q_fourier(kvec, U)
    g_ce2 = ce2_prefactor(tau, rho0) * (lat.lhat @ qf)

    n_lbm = weighted_norm(lat, g_lbm)
    n_ce2 = weighted_norm(lat, g_ce2)

    E = weighted_norm(lat, g_lbm - g_ce2) / (n_ce2 + 1e-300)
    R = n_lbm / (n_ce2 + 1e-300)
    overlap = np.vdot(g_ce2, lat.Winv @ g_lbm)
    chi = abs(overlap) / ((n_ce2 * n_lbm) + 1e-300)

    n = kvec / np.linalg.norm(kvec)
    e = np.asarray(target_pol, dtype=float)
    e = e - np.dot(e, n) * n
    e /= np.linalg.norm(e)

    jmag = np.linalg.norm(dj)
    dens = abs(drho) / (jmag + 1e-300)
    longi = abs(np.dot(n, dj)) / (jmag + 1e-300)
    palign = abs(np.vdot(e, dj)) / (jmag + 1e-300)

    decay = np.nan + 1j * np.nan
    if previous_fhat is not None:
        denom = np.vdot(previous_fhat, previous_fhat)
        if abs(denom) > 0:
            decay = np.vdot(previous_fhat, fhat) / denom

    return Diagnostic(
        E=float(E),
        R=float(R),
        chi=float(chi),
        norm_lbm=n_lbm,
        norm_ce2=n_ce2,
        density_fraction=float(dens),
        longitudinal_fraction=float(longi),
        polarization_alignment=float(palign),
        decay_ratio=decay,
        g_lbm=g_lbm,
        g_ce2=g_ce2,
        fhat=fhat,
    )



def fit_convergence(results: List["CaseResult"]) -> List[dict]:
    """Fit E ~ C k^p for each lattice/direction/polarization group."""
    groups: Dict[Tuple[str, str, str], List[CaseResult]] = {}
    for r in results:
        groups.setdefault((r.lattice, r.direction, r.polarization), []).append(r)

    rows = []
    for key, vals in sorted(groups.items()):
        vals = sorted(vals, key=lambda x: x.k)
        # Use the smallest up to five points, provided they are above numerical floor.
        use = [r for r in vals if r.E_mean > 1e-10][:5]
        if len(use) < 3:
            use = vals[: min(5, len(vals))]
        x = np.log(np.array([r.k for r in use]))
        y = np.log(np.array([r.E_mean for r in use]))
        pfit, logc = np.polyfit(x, y, 1)
        r0 = vals[0]
        rows.append({
            "lattice": key[0],
            "direction": key[1],
            "polarization": key[2],
            "n_fit": len(use),
            "k_min": min(r.k for r in use),
            "k_max": max(r.k for r in use),
            "observed_order": float(pfit),
            "smallest_k_E": r0.E_mean,
            "smallest_k_R": r0.R_mean,
            "smallest_k_chi": r0.chi_mean,
        })
    return rows


def write_fit_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def make_plots(results: List["CaseResult"], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Principal convergence figure.
    plt.figure(figsize=(7.2, 5.2))
    for key in sorted(set((r.lattice, r.direction, r.polarization) for r in results)):
        vals = sorted([r for r in results if (r.lattice, r.direction, r.polarization) == key],
                      key=lambda x: x.k)
        k = np.array([r.k for r in vals])
        E = np.array([r.E_mean for r in vals])
        label = f"{key[0]} [{key[1]}] {key[2]}"
        plt.loglog(k, E, "o-", label=label)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E_Q$")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "figure_population_vector_error.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Amplitude + alignment figure.
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for key in sorted(set((r.lattice, r.direction, r.polarization) for r in results)):
        vals = sorted([r for r in results if (r.lattice, r.direction, r.polarization) == key],
                      key=lambda x: x.k)
        k = np.array([r.k for r in vals])
        R = np.array([r.R_mean for r in vals])
        chi = np.array([r.chi_mean for r in vals])
        label = f"{key[0]} [{key[1]}] {key[2]}"
        ax[0].plot(k, R, "o-", label=label)
        ax[1].plot(k, chi, "o-", label=label)

    ax[0].axhline(1.0, linestyle="--", linewidth=1)
    ax[1].axhline(1.0, linestyle="--", linewidth=1)
    ax[0].set_xlabel(r"$k$")
    ax[1].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$R_Q$")
    ax[1].set_ylabel(r"$\chi_Q$")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "figure_amplitude_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sweep(
    operator_dir: Path,
    outdir: Path,
    N: int = 48,
    tau: float = 1.0,
    epsilon: float = 1e-6,
    nsteps: int = 5,
    mvalues: Iterable[int] = (1, 2, 3, 4),
    max_k: float = 0.55,
) -> List["CaseResult"]:
    outdir.mkdir(parents=True, exist_ok=True)
    results: List[CaseResult] = []

    for q in (19, 27):
        lat = load_lattice(operator_dir / f"Lhat_D3Q{q}.csv")
        for direction in ("100", "110", "111"):
            for m in mvalues:
                mode_indices, pols = direction_definition(direction, m)
                kvec = 2.0 * np.pi * np.array(mode_indices, dtype=float) / N
                if np.linalg.norm(kvec) > max_k:
                    continue
                for pname, pol in pols:
                    print(f"{lat.name} [{direction}] {pname:4s} m={m} "
                          f"k={np.linalg.norm(kvec):.6f}")
                    r = run_case(
                        lat=lat,
                        N=N,
                        direction=direction,
                        polarization_name=pname,
                        polarization=pol,
                        m=m,
                        tau=tau,
                        epsilon=epsilon,
                        nsteps=nsteps,
                    )
                    results.append(r)

    write_csv(results, outdir / "experiment_II_raw_results.csv")
    fitrows = fit_convergence(results)
    write_fit_csv(fitrows, outdir / "experiment_II_convergence_fits.csv")
    make_plots(results, outdir)
    return results


# -----------------------------------------------------------------------------
# Experiment definitions and driver
# -----------------------------------------------------------------------------

def direction_definition(label: str, m: int) -> Tuple[Tuple[int, int, int], List[Tuple[str, np.ndarray]]]:
    if label == "100":
        return (m, 0, 0), [
            ("p_y", np.array([0.0, 1.0, 0.0])),
            ("p_z", np.array([0.0, 0.0, 1.0])),
        ]
    if label == "110":
        return (m, m, 0), [
            ("p_in", np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)),
            ("p_z", np.array([0.0, 0.0, 1.0])),
        ]
    if label == "111":
        return (m, m, m), [
            ("p_1", np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)),
            ("p_2", np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)),
        ]
    raise ValueError(label)


@dataclass
class CaseResult:
    lattice: str
    direction: str
    polarization: str
    N: int
    m: int
    tau: float
    epsilon: float
    k: float
    E_mean: float
    E_std_rel: float
    R_mean: float
    chi_mean: float
    norm_lbm_mean: float
    norm_ce2_mean: float
    exact_eigenvalue: complex
    measured_decay: complex
    density_fraction: float
    longitudinal_fraction: float
    polarization_alignment: float
    branch_gap: float


def run_case(
    lat: Lattice,
    N: int,
    direction: str,
    polarization_name: str,
    polarization: np.ndarray,
    m: int,
    tau: float = 1.0,
    epsilon: float = 1e-6,
    nsteps: int = 6,
    rho0: float = 1.0,
) -> CaseResult:
    mode_indices, _ = direction_definition(direction, m)
    f, mode, kvec = initialize_exact_mode(
        lat, N, mode_indices, polarization, tau, epsilon, rho0
    )

    Es, Rs, chis, nl, nc, decays = [], [], [], [], [], []
    prev = None
    last = None

    for _ in range(nsteps + 1):
        d = diagnose(lat, f, kvec, polarization, tau, prev, rho0)
        Es.append(d.E)
        Rs.append(d.R)
        chis.append(d.chi)
        nl.append(d.norm_lbm)
        nc.append(d.norm_ce2)
        if prev is not None:
            decays.append(d.decay_ratio)
        prev = d.fhat
        last = d
        f = collide_and_stream(lat, f, tau)

    Earr = np.asarray(Es)
    Emean = float(np.mean(Earr))
    Estdrel = float(np.std(Earr) / (abs(Emean) + 1e-300))

    measured_decay = np.mean(decays) if decays else np.nan + 1j * np.nan

    return CaseResult(
        lattice=lat.name,
        direction=direction,
        polarization=polarization_name,
        N=N,
        m=m,
        tau=tau,
        epsilon=epsilon,
        k=float(np.linalg.norm(kvec)),
        E_mean=Emean,
        E_std_rel=Estdrel,
        R_mean=float(np.mean(Rs)),
        chi_mean=float(np.mean(chis)),
        norm_lbm_mean=float(np.mean(nl)),
        norm_ce2_mean=float(np.mean(nc)),
        exact_eigenvalue=mode.eigenvalue,
        measured_decay=measured_decay,
        density_fraction=last.density_fraction,
        longitudinal_fraction=last.longitudinal_fraction,
        polarization_alignment=last.polarization_alignment,
        branch_gap=mode.branch_gap,
    )


def write_csv(results: List[CaseResult], path: Path) -> None:
    fields = [
        "lattice", "direction", "polarization", "N", "m", "tau", "epsilon", "k",
        "E_mean", "E_std_rel", "R_mean", "chi_mean",
        "norm_lbm_mean", "norm_ce2_mean",
        "eigenvalue_real", "eigenvalue_imag",
        "measured_decay_real", "measured_decay_imag",
        "density_fraction", "longitudinal_fraction",
        "polarization_alignment", "branch_gap",
    ]
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in results:
            wr.writerow({
                "lattice": r.lattice,
                "direction": r.direction,
                "polarization": r.polarization,
                "N": r.N,
                "m": r.m,
                "tau": r.tau,
                "epsilon": r.epsilon,
                "k": r.k,
                "E_mean": r.E_mean,
                "E_std_rel": r.E_std_rel,
                "R_mean": r.R_mean,
                "chi_mean": r.chi_mean,
                "norm_lbm_mean": r.norm_lbm_mean,
                "norm_ce2_mean": r.norm_ce2_mean,
                "eigenvalue_real": r.exact_eigenvalue.real,
                "eigenvalue_imag": r.exact_eigenvalue.imag,
                "measured_decay_real": r.measured_decay.real,
                "measured_decay_imag": r.measured_decay.imag,
                "density_fraction": r.density_fraction,
                "longitudinal_fraction": r.longitudinal_fraction,
                "polarization_alignment": r.polarization_alignment,
                "branch_gap": r.branch_gap,
            })


def print_case(r: CaseResult) -> None:
    print(f"{r.lattice} {r.direction}/{r.polarization} m={r.m} N={r.N}")
    print(f"k                     = {r.k:.12g}")
    print(f"E                     = {r.E_mean:.6e}")
    print(f"R                     = {r.R_mean:.12g}")
    print(f"chi                   = {r.chi_mean:.12g}")
    print(f"||g_LBM||             = {r.norm_lbm_mean:.6e}")
    print(f"||g_CE2||             = {r.norm_ce2_mean:.6e}")
    print(f"temporal rel std(E)   = {r.E_std_rel:.3e}")
    print(f"density fraction      = {r.density_fraction:.3e}")
    print(f"longitudinal fraction = {r.longitudinal_fraction:.3e}")
    print(f"polarization align.   = {r.polarization_alignment:.12g}")
    print(f"lambda exact          = {r.exact_eigenvalue.real:+.12g}{r.exact_eigenvalue.imag:+.12g}j")
    print(f"lambda measured       = {r.measured_decay.real:+.12g}{r.measured_decay.imag:+.12g}j")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("demo", "sweep"), default="demo")
    p.add_argument("--operator-dir", type=Path, default=Path("."))
    p.add_argument("--outdir", type=Path, default=Path("experiment_II_output"))
    p.add_argument("--lattice", choices=("19", "27"), default="19")
    p.add_argument("--N", type=int, default=24)
    p.add_argument("--direction", choices=("100", "110", "111"), default="110")
    p.add_argument("--polarization", default=None,
                   help="polarization label; default is first listed for direction")
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--max-k", type=float, default=0.55)
    args = p.parse_args()

    if args.mode == "sweep":
        results = run_sweep(
            operator_dir=args.operator_dir,
            outdir=args.outdir,
            N=args.N,
            tau=args.tau,
            epsilon=args.epsilon,
            nsteps=args.steps,
            mvalues=range(1, 6),
            max_k=args.max_k,
        )
        print(f"Completed {len(results)} cases; outputs in {args.outdir.resolve()}")
        return

    op = args.operator_dir / f"Lhat_D3Q{args.lattice}.csv"
    lat = load_lattice(op)
    _, pols = direction_definition(args.direction, args.m)

    if args.polarization is None:
        pname, pol = pols[0]
    else:
        matches = [(n, v) for n, v in pols if n == args.polarization]
        if not matches:
            raise ValueError(f"Unknown polarization {args.polarization}; choices={[n for n,_ in pols]}")
        pname, pol = matches[0]

    r = run_case(
        lat=lat,
        N=args.N,
        direction=args.direction,
        polarization_name=pname,
        polarization=pol,
        m=args.m,
        tau=args.tau,
        epsilon=args.epsilon,
        nsteps=args.steps,
    )
    print_case(r)


if __name__ == "__main__":
    main()
