#!/usr/bin/env python3
"""
Experiment III: decaying Taylor-Green vortex reserve test.

Purpose
-------
Probe whether the linear CE2 kinetic population operator
    g_Q^CE2 = C0 * Lhat_Q * q(x,t)
continues to describe the leading kinetic population structure in a weakly
nonlinear, multimode 3-D flow.

The experiment is intentionally modest and is meant as a reviewer-response
reserve calculation, not as part of the core manuscript validation.

Dependencies
------------
numpy
matplotlib (only for optional production plots)

The exact Lhat_D3Q19.csv and Lhat_D3Q27.csv operators must be available in
--operator-dir and must use the same population ordering as the simulation.
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
TWOPI = 2.0 * np.pi


# -----------------------------------------------------------------------------
# Lattice/operator definitions
# -----------------------------------------------------------------------------

@dataclass
class Lattice:
    name: str
    c: np.ndarray
    w: np.ndarray
    lhat: np.ndarray
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
        raise ValueError("Lattice weights do not sum to unity.")

    winv = np.diag(1.0 / w)

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

    if np.linalg.norm(pk @ pk - pk) > 5e-12:
        raise RuntimeError(f"{operator_csv.name}: PK not idempotent.")
    if np.linalg.norm(pk @ p_le2) > 5e-12:
        raise RuntimeError(f"{operator_csv.name}: PK not orthogonal to <=2 sector.")

    return Lattice(
        name=f"D3Q{Q}",
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


def weighted_norm_sq_field(lat: Lattice, g: np.ndarray) -> np.ndarray:
    # g shape: (Q,Nz,Ny,Nx), real or complex
    return np.sum((np.abs(g) ** 2) / lat.w[:, None, None, None], axis=0)


# -----------------------------------------------------------------------------
# BGK solver
# -----------------------------------------------------------------------------

def macroscopic(lat: Lattice, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sum(f, axis=0)
    j = np.tensordot(lat.c.T, f, axes=(1, 0))
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
        fout[i] = np.roll(fpost[i], shift=(cz, cy, cx), axis=(0, 1, 2))
    return fout


# -----------------------------------------------------------------------------
# Taylor-Green initialization
# -----------------------------------------------------------------------------

def taylor_green_velocity(N: int, U0: float) -> np.ndarray:
    x = TWOPI * np.arange(N) / N
    y = TWOPI * np.arange(N) / N
    z = TWOPI * np.arange(N) / N
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    ux = U0 * np.sin(xx) * np.cos(yy) * np.cos(zz)
    uy = -U0 * np.cos(xx) * np.sin(yy) * np.cos(zz)
    uz = np.zeros_like(ux)
    return np.stack([ux, uy, uz], axis=0)


def initialize_taylor_green(
    lat: Lattice,
    N: int,
    U0: float,
    rho0: float = 1.0,
) -> np.ndarray:
    u = taylor_green_velocity(N, U0)
    rho = np.full((N, N, N), rho0, dtype=float)
    return equilibrium(lat, rho, u)


# -----------------------------------------------------------------------------
# Spectral derivatives and curvature construction
# -----------------------------------------------------------------------------

def spectral_wave_numbers(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fourier wavenumbers in LBM lattice coordinates.

    The analytical Taylor-Green field is written on X in [0,2pi), with
    X = (2pi/N) x_LB.  The CE2 operator Lhat_Q is expressed in lattice
    derivatives, so the mode n has k_LB = 2pi*n/N, not the physical-coordinate
    integer wavenumber n.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    return kx, ky, kz


def velocity_fft(u: np.ndarray) -> np.ndarray:
    return np.fft.fftn(u, axes=(1, 2, 3))


def spectral_first_derivatives(u: np.ndarray) -> np.ndarray:
    """
    Return du[a,b] = partial_b u_a, shape (3,3,Nz,Ny,Nx).
    """
    N = u.shape[1]
    kx, ky, kz = spectral_wave_numbers(N)
    ks = [kx, ky, kz]
    uhat = velocity_fft(u)
    du = np.empty((3, 3, N, N, N), dtype=float)
    for a in range(3):
        for b in range(3):
            arr = np.fft.ifftn(1j * ks[b] * uhat[a], axes=(0,1,2)).real
            du[a, b] = arr
    return du


def spectral_second_derivatives(u: np.ndarray) -> np.ndarray:
    """
    Return d2u[a,b,c] = partial_b partial_c u_a,
    shape (3,3,3,Nz,Ny,Nx).
    """
    N = u.shape[1]
    kx, ky, kz = spectral_wave_numbers(N)
    ks = [kx, ky, kz]
    uhat = velocity_fft(u)
    d2 = np.empty((3, 3, 3, N, N, N), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                arr = np.fft.ifftn(
                    -(ks[b] * ks[c]) * uhat[a], axes=(0,1,2)
                ).real
                d2[a, b, c] = arr
    return d2


def curvature_q_field(u: np.ndarray) -> np.ndarray:
    d2 = spectral_second_derivatives(u)
    q = np.stack(
        [
            d2[0,0,1],  # ux_xy
            d2[0,0,2],  # ux_xz
            d2[0,1,1],  # ux_yy
            d2[0,2,2],  # ux_zz
            d2[0,1,2],  # ux_yz
            d2[1,0,1],  # uy_xy
            d2[1,1,2],  # uy_yz
            d2[1,0,0],  # uy_xx
            d2[1,2,2],  # uy_zz
            d2[1,0,2],  # uy_xz
            d2[2,0,2],  # uz_xz
            d2[2,1,2],  # uz_yz
            d2[2,0,0],  # uz_xx
            d2[2,1,1],  # uz_yy
            d2[2,0,1],  # uz_xy
        ],
        axis=0,
    )
    return q


def ce2_prefactor(tau: float, rho0: float = 1.0) -> float:
    return 2.0 * rho0 * tau * (tau - 0.5) * CS4


def ce2_population_field(lat: Lattice, u: np.ndarray, tau: float, rho0: float = 1.0) -> np.ndarray:
    q = curvature_q_field(u)
    # lhat: (Q,15), q: (15,Nz,Ny,Nx)
    g = np.tensordot(lat.lhat, q, axes=(1, 0))
    return ce2_prefactor(tau, rho0) * g


def measured_kinetic_field(lat: Lattice, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, u = macroscopic(lat, f)
    feq = equilibrium(lat, rho, u)
    fneq = f - feq
    g = np.tensordot(lat.PK.real, fneq, axes=(1, 0))
    return g, rho, u


# -----------------------------------------------------------------------------
# Global diagnostics
# -----------------------------------------------------------------------------

def global_population_metrics(lat: Lattice, g_lbm: np.ndarray, g_ce2: np.ndarray) -> Tuple[float,float,float]:
    diff = g_lbm - g_ce2
    n_diff2 = np.mean(weighted_norm_sq_field(lat, diff))
    n_lbm2 = np.mean(weighted_norm_sq_field(lat, g_lbm))
    n_ce22 = np.mean(weighted_norm_sq_field(lat, g_ce2))

    E = np.sqrt(n_diff2 / (n_ce22 + 1e-300))
    R = np.sqrt(n_lbm2 / (n_ce22 + 1e-300))

    overlap_field = np.sum(
        g_ce2 * g_lbm / lat.w[:, None, None, None], axis=0
    )
    overlap = np.mean(overlap_field)
    X = abs(overlap) / (np.sqrt(n_ce22 * n_lbm2) + 1e-300)
    return float(E), float(R), float(X)


def divergence_rms(u: np.ndarray) -> float:
    du = spectral_first_derivatives(u)
    div = du[0,0] + du[1,1] + du[2,2]
    return float(np.sqrt(np.mean(div * div)))


def strain_field(u: np.ndarray) -> np.ndarray:
    du = spectral_first_derivatives(u)
    S = np.empty_like(du)
    for a in range(3):
        for b in range(3):
            S[a,b] = 0.5 * (du[a,b] + du[b,a])
    return S


def vorticity_field(u: np.ndarray) -> np.ndarray:
    du = spectral_first_derivatives(u)
    wx = du[2,1] - du[1,2]
    wy = du[0,2] - du[2,0]
    wz = du[1,0] - du[0,1]
    return np.stack([wx,wy,wz],axis=0)


def enstrophy_palinstrophy(u: np.ndarray) -> Tuple[float,float]:
    """
    Omega = 1/2 <|omega|^2>
    P     = 1/2 <|grad omega|^2>
    """
    omega = vorticity_field(u)
    Omega = 0.5 * float(np.mean(np.sum(omega * omega, axis=0)))

    domega = spectral_first_derivatives(omega)
    P = 0.5 * float(np.mean(np.sum(domega * domega, axis=(0,1))))
    return Omega, P


def physical_stress_norm_sq_mean(u: np.ndarray, nu: float, rho0: float = 1.0) -> float:
    S = strain_field(u)
    s2 = np.sum(S * S, axis=(0,1))
    return float(np.mean(4.0 * rho0 * rho0 * nu * nu * s2))


def gamma_q(lat: Lattice) -> float:
    if lat.name == "D3Q19":
        return 23.0 / 35.0
    if lat.name == "D3Q27":
        return 26.0 / 35.0
    raise ValueError("Gamma_Q only configured for D3Q19/D3Q27.")


def spectral_centroid_estimator(
    lat: Lattice,
    g_lbm: np.ndarray,
    u: np.ndarray,
    tau: float,
    rho0: float = 1.0,
) -> Tuple[float,float,float]:
    """
    Returns population estimator, P/Omega, and their ratio.

    This is exploratory only for Taylor-Green because HIT assumptions do not hold.
    """
    Kmean = rho0 * float(np.mean(weighted_norm_sq_field(lat, g_lbm)))
    nu = CS2 * (tau - 0.5)
    stress2 = physical_stress_norm_sq_mean(u, nu, rho0)
    Lam = gamma_q(lat) * tau * tau / CS2
    kpop2 = Kmean / (Lam * stress2 + 1e-300)

    Omega, P = enstrophy_palinstrophy(u)
    kspec2 = P / (Omega + 1e-300)
    return float(kpop2), float(kspec2), float(kpop2 / (kspec2 + 1e-300))


# -----------------------------------------------------------------------------
# Time diagnostics
# -----------------------------------------------------------------------------

@dataclass
class TGVRecord:
    step: int
    t_convective: float
    lattice: str
    N: int
    U0: float
    tau: float
    mass: float
    max_density_deviation: float
    max_speed: float
    max_mach: float
    divergence_rms: float
    E: float
    R: float
    X: float
    norm_lbm_rms: float
    norm_ce2_rms: float
    Omega: float
    P: float
    P_over_Omega: float
    kpop2: float
    kpop2_over_POmega: float


def diagnose_state(
    lat: Lattice,
    f: np.ndarray,
    tau: float,
    U0: float,
    step: int,
    rho0: float = 1.0,
) -> TGVRecord:
    g_lbm, rho, u = measured_kinetic_field(lat, f)
    g_ce2 = ce2_population_field(lat, u, tau, rho0)

    E,R,X = global_population_metrics(lat, g_lbm, g_ce2)
    n_lbm = np.sqrt(float(np.mean(weighted_norm_sq_field(lat, g_lbm))))
    n_ce2 = np.sqrt(float(np.mean(weighted_norm_sq_field(lat, g_ce2))))

    Omega,P = enstrophy_palinstrophy(u)
    kpop2,kspec2,kratio = spectral_centroid_estimator(lat, g_lbm, u, tau, rho0)

    mass = float(np.sum(rho))
    maxrho = float(np.max(np.abs(rho-rho0)))
    speed = np.sqrt(np.sum(u*u,axis=0))
    maxspeed = float(np.max(speed))
    divrms = divergence_rms(u)

    # Convective time in lattice units, using L=N/(2pi) wavelengths convention:
    # physical TGV wavelength is 2pi; one lattice step corresponds dx_phys=2pi/N.
    # u_lattice converts to physical-coordinate speed by dx_phys/dt, so
    # t* = U0 * t_lattice * 2pi/N / (2pi) = U0*t/N.
    tstar = U0 * step / f.shape[1]

    return TGVRecord(
        step=step,
        t_convective=tstar,
        lattice=lat.name,
        N=f.shape[1],
        U0=U0,
        tau=tau,
        mass=mass,
        max_density_deviation=maxrho,
        max_speed=maxspeed,
        max_mach=maxspeed/np.sqrt(CS2),
        divergence_rms=divrms,
        E=E,
        R=R,
        X=X,
        norm_lbm_rms=n_lbm,
        norm_ce2_rms=n_ce2,
        Omega=Omega,
        P=P,
        P_over_Omega=kspec2,
        kpop2=kpop2,
        kpop2_over_POmega=kratio,
    )


def write_records(records: List[TGVRecord], path: Path) -> None:
    if not records:
        return
    fields = list(records[0].__dict__.keys())
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in records:
            wr.writerow(r.__dict__)


# -----------------------------------------------------------------------------
# Runs and plotting
# -----------------------------------------------------------------------------

def suggested_output_steps(N: int, U0: float, max_tstar: float = 0.5) -> List[int]:
    # t* = U0 * step / N
    final_step = max(10, int(round(max_tstar * N / U0)))
    vals = [0, 1, 2, 5, 10]
    for frac in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0):
        vals.append(int(round(frac * final_step)))
    return sorted(set(v for v in vals if 0 <= v <= final_step))


def run_case(
    lat: Lattice,
    N: int = 32,
    U0: float = 0.02,
    tau: float = 0.8,
    rho0: float = 1.0,
    max_tstar: float = 0.15,
    output_steps: Iterable[int] | None = None,
    startup_discard: int = 5,
) -> List[TGVRecord]:
    f = initialize_taylor_green(lat, N, U0, rho0)
    if output_steps is None:
        output_steps = suggested_output_steps(N, U0, max_tstar)
    output_steps = sorted(set(int(x) for x in output_steps))
    final_step = max(output_steps)

    records = []
    for step in range(final_step + 1):
        if step in output_steps:
            rec = diagnose_state(lat, f, tau, U0, step, rho0)
            records.append(rec)
            tag = "(startup)" if step < startup_discard else ""
            print(
                f"{lat.name} N={N} U0={U0:g} tau={tau:g} step={step:5d} "
                f"t*={rec.t_convective:.4f} E={rec.E:.4e} R={rec.R:.5f} "
                f"X={rec.X:.6f} Ma={rec.max_mach:.4f} {tag}"
            )
        if step < final_step:
            f = collide_and_stream(lat, f, tau)
    return records


def make_case_plots(records: List[TGVRecord], outdir: Path, startup_discard: int = 5) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str,float],List[TGVRecord]] = {}
    for r in records:
        if r.step >= startup_discard:
            groups.setdefault((r.lattice,r.U0),[]).append(r)

    plt.figure(figsize=(7.2,5.2))
    for key, vals in sorted(groups.items()):
        vals=sorted(vals,key=lambda r:r.t_convective)
        plt.plot([r.t_convective for r in vals],[r.E for r in vals],"o-",
                 label=f"{key[0]}, U0={key[1]:g}")
    plt.xlabel(r"$t^*$")
    plt.ylabel(r"$\mathcal{E}_Q$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"figure_III_1_global_CE2_error.png",dpi=300,bbox_inches="tight")
    plt.close()

    fig,ax=plt.subplots(1,2,figsize=(10.0,4.2))
    for key,vals in sorted(groups.items()):
        vals=sorted(vals,key=lambda r:r.t_convective)
        t=[r.t_convective for r in vals]
        ax[0].plot(t,[r.R for r in vals],"o-",label=f"{key[0]}, U0={key[1]:g}")
        ax[1].plot(t,[r.X for r in vals],"o-",label=f"{key[0]}, U0={key[1]:g}")
    ax[0].axhline(1.0,linestyle="--",linewidth=1)
    ax[1].axhline(1.0,linestyle="--",linewidth=1)
    ax[0].set_xlabel(r"$t^*$"); ax[1].set_xlabel(r"$t^*$")
    ax[0].set_ylabel(r"$\mathcal{R}_Q$"); ax[1].set_ylabel(r"$\mathcal{X}_Q$")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir/"figure_III_2_amplitude_alignment.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(7.2,5.2))
    for key,vals in sorted(groups.items()):
        vals=sorted(vals,key=lambda r:r.t_convective)
        plt.plot([r.t_convective for r in vals],[r.kpop2 for r in vals],"o-",
                 label=f"{key[0]} pop, U0={key[1]:g}")
    # Plot spectral reference once per U0, using first lattice group.
    for U0 in sorted(set(r.U0 for r in records)):
        vals=[r for r in records if r.U0==U0 and r.step>=startup_discard]
        if vals:
            vals=sorted(vals,key=lambda r:r.t_convective)
            # Use D3Q19 if present to avoid duplicate line.
            chosen=[r for r in vals if r.lattice=="D3Q19"] or vals
            # deduplicate by step
            bystep={}
            for r in chosen:
                bystep[r.step]=r
            vv=sorted(bystep.values(),key=lambda r:r.t_convective)
            plt.plot([r.t_convective for r in vv],[r.P_over_Omega for r in vv],"k--",
                     alpha=0.5,label=f"P/Omega, U0={U0:g}")
    plt.xlabel(r"$t^*$")
    plt.ylabel(r"$k_*^2$")
    plt.legend(fontsize=8,ncol=2)
    plt.tight_layout()
    plt.savefig(outdir/"figure_III_3_spectral_centroid.png",dpi=300,bbox_inches="tight")
    plt.close()


def run_sweep(
    operator_dir: Path,
    outdir: Path,
    N: int = 64,
    tau: float = 0.8,
    U0_values: Iterable[float] = (0.01,0.02,0.04),
    max_tstar: float = 0.25,
    startup_discard: int = 5,
) -> List[TGVRecord]:
    outdir.mkdir(parents=True,exist_ok=True)
    all_records=[]
    for q in (19,27):
        lat=load_lattice(operator_dir/f"Lhat_D3Q{q}.csv")
        for U0 in U0_values:
            recs=run_case(
                lat=lat,N=N,U0=U0,tau=tau,max_tstar=max_tstar,
                startup_discard=startup_discard
            )
            all_records.extend(recs)
    write_records(all_records,outdir/"experiment_III_results.csv")
    make_case_plots(all_records,outdir,startup_discard)
    return all_records


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-dir",type=Path,default=Path("."))
    p.add_argument("--outdir",type=Path,default=Path("experiment_III_output"))
    p.add_argument("--lattice",choices=("19","27"),default="19")
    p.add_argument("--N",type=int,default=24)
    p.add_argument("--U0",type=float,default=0.02)
    p.add_argument("--tau",type=float,default=0.8)
    p.add_argument("--max-tstar",type=float,default=0.05)
    p.add_argument("--mode",choices=("demo","sweep"),default="demo")
    args=p.parse_args()

    if args.mode=="sweep":
        recs=run_sweep(
            operator_dir=args.operator_dir,
            outdir=args.outdir,
            N=args.N,
            tau=args.tau,
            U0_values=(0.01,0.02,0.04),
            max_tstar=args.max_tstar,
        )
        print(f"Completed {len(recs)} diagnostic records.")
        return

    lat=load_lattice(args.operator_dir/f"Lhat_D3Q{args.lattice}.csv")
    recs=run_case(
        lat=lat,
        N=args.N,
        U0=args.U0,
        tau=args.tau,
        max_tstar=args.max_tstar,
    )
    args.outdir.mkdir(parents=True,exist_ok=True)
    write_records(recs,args.outdir/"demo_results.csv")
    make_case_plots(recs,args.outdir)
    print(f"Outputs written to {args.outdir.resolve()}")


if __name__=="__main__":
    main()
