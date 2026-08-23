#!/usr/bin/env python3
"""
Experiment I: D2Q9 exact shear-mode LBM validation.

This script implements the numerical experiment described in
"experiment_I_D2Q9_exact_shear_mode.md".

Main capabilities
-----------------
1. Construct the exact linear D2Q9 amplification matrix A(k,tau).
2. Identify the transverse shear eigenmode from its hydrodynamic content.
3. Initialize a real periodic LBM field from that exact complex eigenmode.
4. Evolve the *full nonlinear* BGK D2Q9 scheme with periodic streaming.
5. Extract the initialized Fourier mode from the time-evolved populations.
6. Compute
       eta_K = || P_K f^neq ||_w / || P_2 f^neq ||_w
   in Fourier space.
7. Compare eta_K against
       A(theta,tau) k [1 + B(theta,tau) k^2].
8. Run a compact parameter sweep and generate CSV data + publication-ready plots.

The implementation is deliberately small and self-contained.  It uses only
NumPy and Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# D2Q9 constants
# -----------------------------------------------------------------------------

CS2 = 1.0 / 3.0

C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=float,
)

W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=float,
)

WINV = np.diag(1.0 / W)


# -----------------------------------------------------------------------------
# Linear projectors and amplification matrix
# -----------------------------------------------------------------------------

def weighted_projector(columns: Iterable[np.ndarray]) -> np.ndarray:
    """Weighted-Hessian orthogonal projector onto span(columns)."""
    phi = np.column_stack(list(columns)).astype(complex)
    gram = phi.conj().T @ WINV @ phi
    return phi @ np.linalg.solve(gram, phi.conj().T @ WINV)


def build_projectors() -> Dict[str, np.ndarray]:
    """Return P_eq, P_<=1, P_2, P_<=2, P_K, and N."""
    rho = W.copy()
    jx = W * C[:, 0]
    jy = W * C[:, 1]
    hxx = W * (C[:, 0] ** 2 - CS2)
    hyy = W * (C[:, 1] ** 2 - CS2)
    hxy = W * (C[:, 0] * C[:, 1])

    p_le1 = weighted_projector([rho, jx, jy])
    p_le2 = weighted_projector([rho, jx, jy, hxx, hyy, hxy])
    p2 = p_le2 - p_le1
    pk = np.eye(9, dtype=complex) - p_le2

    # Exact linear-equilibrium projector around rho=1, u=0.
    p_eq = np.empty((9, 9), dtype=complex)
    for i in range(9):
        for j in range(9):
            p_eq[i, j] = W[i] * (1.0 + np.dot(C[i], C[j]) / CS2)

    nproj = np.eye(9, dtype=complex) - p_eq

    return {
        "P_eq": p_eq,
        "P_le1": p_le1,
        "P_le2": p_le2,
        "P2": p2,
        "PK": pk,
        "N": nproj,
    }


PROJECTORS = build_projectors()


def collision_matrix(tau: float) -> np.ndarray:
    """Linearized BGK collision matrix around rho=1, u=0."""
    return (1.0 - 1.0 / tau) * np.eye(9) + (1.0 / tau) * PROJECTORS["P_eq"]


def amplification_matrix(kx: float, ky: float, tau: float) -> np.ndarray:
    """
    Exact D2Q9 Fourier amplification matrix.

    Convention:
        delta f(x,t) = fhat(t) exp(i k.x)
        S_ii = exp(-i k.c_i)
    """
    phase = np.exp(-1j * (C[:, 0] * kx + C[:, 1] * ky))
    stream = np.diag(phase)
    return stream @ collision_matrix(tau)


def weighted_norm(v: np.ndarray) -> float:
    """Reduced equilibrium-Hessian norm."""
    z = np.vdot(v, WINV @ v)
    return float(np.sqrt(max(z.real, 0.0)))


def hydrodynamic_moments(v: np.ndarray) -> Tuple[complex, np.ndarray]:
    """Return density and momentum perturbations of a population vector."""
    drho = np.sum(v)
    dj = np.array([np.dot(C[:, 0], v), np.dot(C[:, 1], v)], dtype=complex)
    return drho, dj


@dataclass
class ShearMode:
    eigenvalue: complex
    eigenvector: np.ndarray
    k: float
    theta: float
    e_parallel: np.ndarray
    e_perp: np.ndarray
    selection_score: float


def select_shear_mode(kx: float, ky: float, tau: float) -> ShearMode:
    """
    Select the transverse shear eigenmode by hydrodynamic-vector content,
    not by eigenvalue continuity alone.
    """
    kmag = float(np.hypot(kx, ky))
    if kmag <= 0.0:
        raise ValueError("Nonzero wavevector required.")

    epar = np.array([kx, ky], dtype=float) / kmag
    eperp = np.array([-epar[1], epar[0]], dtype=float)

    amp = amplification_matrix(kx, ky, tau)
    evals, evecs = np.linalg.eig(amp)

    # k->0 transverse equilibrium momentum vector.
    v0 = W * (C @ eperp) / CS2
    v0n = weighted_norm(v0)

    best = None
    for j in range(9):
        v = evecs[:, j]
        vn = weighted_norm(v)
        if vn == 0.0:
            continue

        drho, dj = hydrodynamic_moments(v)
        jpar = np.dot(epar, dj)
        jperp = np.dot(eperp, dj)
        hydro_mag = np.sqrt(abs(drho) ** 2 + abs(jpar) ** 2 + abs(jperp) ** 2)

        overlap = abs(np.vdot(v0, WINV @ v)) / (v0n * vn + 1e-300)
        transverse_fraction = abs(jperp) / (hydro_mag + 1e-300)
        density_penalty = abs(drho) / (hydro_mag + 1e-300)

        # Hydrodynamic branches remain close to the unit circle at small k.
        hydro_weight = min(1.0, abs(evals[j]) / 0.8)
        score = hydro_weight * (0.65 * overlap + 0.35 * transverse_fraction) * (1.0 - 0.25 * density_penalty)

        if best is None or score > best[0]:
            best = (score, evals[j], v)

    if best is None:
        raise RuntimeError("Could not identify a shear eigenmode.")

    score, lam, v = best

    # Normalize so that transverse momentum amplitude is unity.
    _, dj = hydrodynamic_moments(v)
    jperp = np.dot(eperp, dj)
    if abs(jperp) < 1e-14:
        raise RuntimeError("Selected mode has negligible transverse momentum.")
    v = v / jperp

    return ShearMode(
        eigenvalue=lam,
        eigenvector=v,
        k=kmag,
        theta=float(np.arctan2(ky, kx)),
        e_parallel=epar,
        e_perp=eperp,
        selection_score=float(score),
    )


# -----------------------------------------------------------------------------
# Full nonlinear D2Q9 LBM
# -----------------------------------------------------------------------------

def macroscopic(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    f shape: (9, Ny, Nx)
    returns rho shape (Ny,Nx), u shape (2,Ny,Nx)
    """
    rho = np.sum(f, axis=0)
    jx = np.tensordot(C[:, 0], f, axes=(0, 0))
    jy = np.tensordot(C[:, 1], f, axes=(0, 0))
    u = np.stack((jx / rho, jy / rho), axis=0)
    return rho, u


def equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quadratic D2Q9 equilibrium; returns shape (9,Ny,Nx)."""
    ux, uy = u[0], u[1]
    usq = ux * ux + uy * uy
    feq = np.empty((9,) + rho.shape, dtype=float)
    for i, (cx, cy) in enumerate(C):
        cu = cx * ux + cy * uy
        feq[i] = W[i] * rho * (
            1.0 + cu / CS2 + 0.5 * (cu * cu) / (CS2 * CS2) - 0.5 * usq / CS2
        )
    return feq


def collide_and_stream(f: np.ndarray, tau: float) -> np.ndarray:
    """One full BGK collision + periodic streaming step."""
    rho, u = macroscopic(f)
    feq = equilibrium(rho, u)
    fpost = f - (f - feq) / tau

    fout = np.empty_like(f)
    for i, (cx, cy) in enumerate(C.astype(int)):
        # Array axes are (y,x); population streams to x+c_i.
        fout[i] = np.roll(fpost[i], shift=(cy, cx), axis=(0, 1))
    return fout


def initialize_exact_mode(
    nx: int,
    ny: int,
    mx: int,
    my: int,
    tau: float,
    epsilon: float = 1e-6,
    rho0: float = 1.0,
) -> Tuple[np.ndarray, ShearMode, float, float]:
    """Initialize real populations from the exact complex linear shear eigenmode."""
    kx = 2.0 * np.pi * mx / nx
    ky = 2.0 * np.pi * my / ny
    mode = select_shear_mode(kx, ky, tau)

    x = np.arange(nx)[None, :]
    y = np.arange(ny)[:, None]
    phase = np.exp(1j * (kx * x + ky * y))

    f = np.empty((9, ny, nx), dtype=float)
    for i in range(9):
        f[i] = W[i] * rho0 + epsilon * np.real(mode.eigenvector[i] * phase)

    return f, mode, kx, ky


# -----------------------------------------------------------------------------
# Fourier diagnostics
# -----------------------------------------------------------------------------

def extract_population_fourier_mode(
    f: np.ndarray,
    kx: float,
    ky: float,
    rho0: float = 1.0,
) -> np.ndarray:
    """
    Extract +k Fourier coefficient of population perturbations.
    Normalization is spatial average, so Re[v exp(ikx)] -> v/2.
    """
    _, ny, nx = f.shape
    x = np.arange(nx)[None, :]
    y = np.arange(ny)[:, None]
    phase = np.exp(-1j * (kx * x + ky * y))
    base = W[:, None, None] * rho0
    df = f - base
    return np.sum(df * phase[None, :, :], axis=(1, 2)) / (nx * ny)


@dataclass
class ModeDiagnostic:
    eta: float
    hnorm: float
    knorm: float
    density_mode: complex
    momentum_mode: np.ndarray
    shear_decay_ratio: complex


def diagnose_mode(
    f: np.ndarray,
    kx: float,
    ky: float,
    previous_fhat: np.ndarray | None = None,
    rho0: float = 1.0,
) -> Tuple[ModeDiagnostic, np.ndarray]:
    fhat = extract_population_fourier_mode(f, kx, ky, rho0=rho0)

    fneq = PROJECTORS["N"] @ fhat
    h = PROJECTORS["P2"] @ fneq
    kk = PROJECTORS["PK"] @ fneq

    hnorm = weighted_norm(h)
    knorm = weighted_norm(kk)
    eta = knorm / hnorm

    drho, dj = hydrodynamic_moments(fhat)

    decay = np.nan + 1j * np.nan
    if previous_fhat is not None:
        # Least-squares complex modal ratio.
        denom = np.vdot(previous_fhat, previous_fhat)
        if abs(denom) > 0:
            decay = np.vdot(previous_fhat, fhat) / denom

    return (
        ModeDiagnostic(
            eta=eta,
            hnorm=hnorm,
            knorm=knorm,
            density_mode=drho,
            momentum_mode=dj,
            shear_decay_ratio=decay,
        ),
        fhat,
    )


# -----------------------------------------------------------------------------
# Theory
# -----------------------------------------------------------------------------

def theory_A_B(theta: float, tau: float) -> Tuple[float, float]:
    """Analytical D2Q9 coefficients A(theta,tau), B(theta,tau)."""
    q = (np.cos(theta) ** 2) * (np.sin(theta) ** 2)
    xi = 2.0 * tau - 1.0

    A2 = xi * xi * (1.0 - 3.0 * q) / 6.0
    A = np.sqrt(A2)

    numerator = (
        xi**4
        + q * (-21.0 * xi**4 + 10.0 * xi**2 + 1.0)
        + q**2 * (72.0 * xi**4 - 42.0 * xi**2 - 4.0)
    )
    denominator = 12.0 * xi**2 * (1.0 - 3.0 * q)
    B = numerator / denominator
    return float(A), float(B)


# -----------------------------------------------------------------------------
# Single-case and sweep drivers
# -----------------------------------------------------------------------------

@dataclass
class CaseResult:
    tau: float
    mx: int
    my: int
    nx: int
    ny: int
    epsilon: float
    k: float
    theta: float
    A_exact: float
    B_exact: float
    eta_mean: float
    eta_std_rel: float
    eigenvalue: complex
    measured_decay: complex
    branch_score: float
    density_to_momentum: float
    transverse_error: float

    @property
    def eta_over_k(self) -> float:
        return self.eta_mean / self.k

    @property
    def RB(self) -> float:
        return (self.eta_mean / (self.A_exact * self.k) - 1.0) / (self.k**2)

    @property
    def EAB(self) -> float:
        eta_ab = self.A_exact * self.k * (1.0 + self.B_exact * self.k**2)
        return abs(self.eta_mean - eta_ab) / abs(self.eta_mean)


def run_case(
    nx: int,
    ny: int,
    mx: int,
    my: int,
    tau: float,
    epsilon: float = 1e-6,
    nsteps: int = 8,
    rho0: float = 1.0,
) -> CaseResult:
    f, mode, kx, ky = initialize_exact_mode(nx, ny, mx, my, tau, epsilon, rho0)

    etas: List[float] = []
    decays: List[complex] = []
    prev = None
    last_diag = None

    for _ in range(nsteps + 1):
        diag, fhat = diagnose_mode(f, kx, ky, previous_fhat=prev, rho0=rho0)
        etas.append(diag.eta)
        if prev is not None:
            decays.append(diag.shear_decay_ratio)
        prev = fhat
        last_diag = diag
        f = collide_and_stream(f, tau)

    etas_arr = np.asarray(etas)
    eta_mean = float(np.mean(etas_arr))
    eta_std_rel = float(np.std(etas_arr) / abs(eta_mean))

    measured_decay = np.mean(decays) if decays else np.nan + 1j * np.nan

    A, B = theory_A_B(mode.theta, tau)

    dj = last_diag.momentum_mode
    jmag = np.linalg.norm(dj)
    density_to_momentum = abs(last_diag.density_mode) / (jmag + 1e-300)
    transverse_error = abs(np.dot(mode.e_parallel, dj)) / (jmag + 1e-300)

    return CaseResult(
        tau=tau,
        mx=mx,
        my=my,
        nx=nx,
        ny=ny,
        epsilon=epsilon,
        k=mode.k,
        theta=mode.theta,
        A_exact=A,
        B_exact=B,
        eta_mean=eta_mean,
        eta_std_rel=eta_std_rel,
        eigenvalue=mode.eigenvalue,
        measured_decay=measured_decay,
        branch_score=mode.selection_score,
        density_to_momentum=float(density_to_momentum),
        transverse_error=float(transverse_error),
    )


def direction_modes(label: str, m: int) -> Tuple[int, int]:
    if label == "axis":
        return m, 0
    if label == "oblique":
        return 2 * m, m
    if label == "diag":
        return m, m
    raise ValueError(label)


def run_sweep(
    outdir: Path,
    nx: int = 128,
    nsteps: int = 8,
    epsilon: float = 1e-6,
    taus: Iterable[float] = (0.9, 1.0, 1.1),
    directions: Iterable[str] = ("axis", "oblique", "diag"),
    mvalues: Iterable[int] = (1, 2, 3, 4, 5),
) -> List[CaseResult]:
    outdir.mkdir(parents=True, exist_ok=True)
    results: List[CaseResult] = []

    for tau in taus:
        for label in directions:
            for m in mvalues:
                mx, my = direction_modes(label, m)
                # Avoid unnecessarily large k in the oblique branch.
                k = 2.0 * np.pi * np.hypot(mx, my) / nx
                if k > 0.58:
                    continue

                print(f"tau={tau:.3f} direction={label:7s} mode=({mx},{my}) k={k:.5f}")
                result = run_case(
                    nx=nx,
                    ny=nx,
                    mx=mx,
                    my=my,
                    tau=tau,
                    epsilon=epsilon,
                    nsteps=nsteps,
                )
                results.append(result)

    write_results_csv(results, outdir / "d2q9_shear_results.csv")
    make_plots(results, outdir)
    write_fit_table(results, outdir / "d2q9_fitted_coefficients.csv")
    return results


def write_results_csv(results: List[CaseResult], path: Path) -> None:
    fields = [
        "tau", "mx", "my", "nx", "ny", "epsilon", "k", "theta",
        "A_exact", "B_exact", "eta_mean", "eta_std_rel",
        "eigenvalue_real", "eigenvalue_imag",
        "measured_decay_real", "measured_decay_imag",
        "branch_score", "density_to_momentum", "transverse_error",
        "eta_over_k", "RB", "EAB",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "tau": r.tau,
                "mx": r.mx,
                "my": r.my,
                "nx": r.nx,
                "ny": r.ny,
                "epsilon": r.epsilon,
                "k": r.k,
                "theta": r.theta,
                "A_exact": r.A_exact,
                "B_exact": r.B_exact,
                "eta_mean": r.eta_mean,
                "eta_std_rel": r.eta_std_rel,
                "eigenvalue_real": r.eigenvalue.real,
                "eigenvalue_imag": r.eigenvalue.imag,
                "measured_decay_real": r.measured_decay.real,
                "measured_decay_imag": r.measured_decay.imag,
                "branch_score": r.branch_score,
                "density_to_momentum": r.density_to_momentum,
                "transverse_error": r.transverse_error,
                "eta_over_k": r.eta_over_k,
                "RB": r.RB,
                "EAB": r.EAB,
            })


def _group(results: List[CaseResult]):
    grouped = {}
    for r in results:
        key = (r.tau, r.mx / max(abs(r.mx), abs(r.my), 1), r.my / max(abs(r.mx), abs(r.my), 1))
        grouped.setdefault(key, []).append(r)
    for vals in grouped.values():
        vals.sort(key=lambda x: x.k)
    return grouped


def make_plots(results: List[CaseResult], outdir: Path) -> None:
    # Figure 1: eta/(A k) vs k^2, use tau=1 if available.
    target_tau = min({r.tau for r in results}, key=lambda x: abs(x - 1.0))
    subset = [r for r in results if abs(r.tau - target_tau) < 1e-12]

    plt.figure(figsize=(7, 5))
    for label in ("axis", "oblique", "diag"):
        vals = []
        for r in subset:
            mx0, my0 = direction_modes(label, 1)
            if r.mx * my0 == r.my * mx0:
                vals.append(r)
        vals.sort(key=lambda x: x.k)
        if not vals:
            continue
        x = np.array([r.k**2 for r in vals])
        y = np.array([r.eta_mean / (r.A_exact * r.k) for r in vals])
        plt.plot(x, y, "o-", label=label)
        b = vals[0].B_exact
        plt.plot(x, 1.0 + b * x, "--", linewidth=1)

    plt.xlabel(r"$k^2$")
    plt.ylabel(r"$\eta_K^{LBM}/(A k)$")
    plt.title(fr"D2Q9 shear transfer, $\tau={target_tau:g}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure1_transfer_relation.png", dpi=180)
    plt.close()

    # Figure 2a: R_B vs k^2
    plt.figure(figsize=(7, 5))
    for label in ("axis", "oblique", "diag"):
        vals = []
        for r in subset:
            mx0, my0 = direction_modes(label, 1)
            if r.mx * my0 == r.my * mx0:
                vals.append(r)
        vals.sort(key=lambda x: x.k)
        if not vals:
            continue
        x = np.array([r.k**2 for r in vals])
        y = np.array([r.RB for r in vals])
        plt.plot(x, y, "o-", label=f"{label}; B={vals[0].B_exact:.6g}")
        plt.axhline(vals[0].B_exact, linestyle="--", linewidth=1)

    plt.xlabel(r"$k^2$")
    plt.ylabel(r"$[\eta/(Ak)-1]/k^2$")
    plt.title(fr"Recovery of $B(\theta,\tau)$, $\tau={target_tau:g}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure2a_B_recovery.png", dpi=180)
    plt.close()

    # Figure 2b: relative cubic error vs k
    plt.figure(figsize=(7, 5))
    all_k = []
    all_e = []
    for label in ("axis", "oblique", "diag"):
        vals = []
        for r in subset:
            mx0, my0 = direction_modes(label, 1)
            if r.mx * my0 == r.my * mx0:
                vals.append(r)
        vals.sort(key=lambda x: x.k)
        if not vals:
            continue
        k = np.array([r.k for r in vals])
        e = np.array([r.EAB for r in vals])
        plt.loglog(k, e, "o-", label=label)
        all_k.extend(k.tolist())
        all_e.extend(e.tolist())

    if all_k and all_e:
        kref = np.array(sorted(all_k))
        # Reference k^4 line anchored to the smallest positive error.
        idx = np.argmin(kref)
        k0 = kref[idx]
        e0 = min(e for e in all_e if e > 0)
        plt.loglog(kref, e0 * (kref / k0) ** 4, "--", label=r"$k^4$ reference")

    plt.xlabel(r"$k$")
    plt.ylabel(r"relative error of $Ak(1+Bk^2)$")
    plt.title(fr"Cubic truncation error, $\tau={target_tau:g}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "figure2b_cubic_error.png", dpi=180)
    plt.close()


def write_fit_table(results: List[CaseResult], path: Path) -> None:
    """
    Fit eta/k = A + A B k^2 + C k^4 separately for each tau/direction.
    """
    rows = []

    # Group by tau and primitive direction ratio.
    groups: Dict[Tuple[float, int, int], List[CaseResult]] = {}
    for r in results:
        g = np.gcd(abs(r.mx), abs(r.my))
        if g == 0:
            g = max(abs(r.mx), abs(r.my), 1)
        dx = r.mx // g
        dy = r.my // g
        groups.setdefault((r.tau, dx, dy), []).append(r)

    for (tau, dx, dy), vals in sorted(groups.items()):
        vals = sorted(vals, key=lambda x: x.k)
        if len(vals) < 3:
            continue

        # Use the smallest up to 5 modes.
        vals_fit = vals[: min(5, len(vals))]
        x = np.array([r.k**2 for r in vals_fit])
        y = np.array([r.eta_mean / r.k for r in vals_fit])

        # y = a0 + a1 x + a2 x^2
        X = np.column_stack([np.ones_like(x), x, x*x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        Afit = coef[0]
        Bfit = coef[1] / Afit

        Aexact = vals[0].A_exact
        Bexact = vals[0].B_exact
        rows.append({
            "tau": tau,
            "direction_mx": dx,
            "direction_my": dy,
            "theta": vals[0].theta,
            "A_exact": Aexact,
            "A_fit": Afit,
            "A_rel_error": abs(Afit - Aexact) / abs(Aexact),
            "B_exact": Bexact,
            "B_fit": Bfit,
            "B_rel_error": abs(Bfit - Bexact) / (abs(Bexact) + 1e-300),
            "n_fit": len(vals_fit),
        })

    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def print_case_summary(r: CaseResult) -> None:
    print("\nSingle-case summary")
    print("-------------------")
    print(f"tau                     : {r.tau}")
    print(f"mode (mx,my)            : ({r.mx},{r.my})")
    print(f"k                       : {r.k:.12g}")
    print(f"theta [deg]             : {np.degrees(r.theta):.8f}")
    print(f"A exact                 : {r.A_exact:.12g}")
    print(f"B exact                 : {r.B_exact:.12g}")
    print(f"eta mean                : {r.eta_mean:.12g}")
    print(f"eta temporal rel. std   : {r.eta_std_rel:.3e}")
    print(f"R_B                     : {r.RB:.12g}")
    print(f"cubic relative error    : {r.EAB:.3e}")
    print(f"shear branch score      : {r.branch_score:.6f}")
    print(f"density/momentum        : {r.density_to_momentum:.3e}")
    print(f"longitudinal/momentum   : {r.transverse_error:.3e}")
    print(f"exact eigenvalue        : {r.eigenvalue.real:+.12g}{r.eigenvalue.imag:+.12g}j")
    print(f"measured step ratio     : {r.measured_decay.real:+.12g}{r.measured_decay.imag:+.12g}j")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "sweep"), default="demo")
    parser.add_argument("--outdir", type=Path, default=Path("d2q9_shear_output"))
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--mx", type=int, default=2)
    parser.add_argument("--my", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "demo":
        r = run_case(
            nx=args.N,
            ny=args.N,
            mx=args.mx,
            my=args.my,
            tau=args.tau,
            epsilon=args.epsilon,
            nsteps=args.steps,
        )
        print_case_summary(r)
    else:
        results = run_sweep(
            outdir=args.outdir,
            nx=args.N,
            nsteps=args.steps,
            epsilon=args.epsilon,
        )
        print(f"\nCompleted {len(results)} cases.")
        print(f"Results written to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
